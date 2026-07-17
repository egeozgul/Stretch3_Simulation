import torch 
import numpy as np

from torch.distributions import Categorical
from .envs_runner import EnvsRunner
from .models import Actor, Critic
from .utils import Agent, get_joint_avail_actions, get_conditional_action, get_conditional_logits

class MAC(object):
    
    def __init__(self, env, obs_last_action=False, 
                 a_mlp_layer_size=64, a_rnn_layer_size=64, 
                 c_mlp_layer_size=64,c_rnn_layer_size=64,
                 device='cpu',
                 use_instructions=False,
                 instruction_fusion='concat',
                 freeze_bert=True):

        self.env = env
        self.n_agent = env.n_agent
        self.obs_last_action = obs_last_action

        self.a_mlp_layer_size = a_mlp_layer_size
        self.a_rnn_layer_size = a_rnn_layer_size
        self.c_mlp_layer_size = c_mlp_layer_size
        self.c_rnn_layer_size = c_rnn_layer_size

        self.device = device
        self.use_instructions = use_instructions
        self.instruction_fusion = instruction_fusion
        self.freeze_bert = freeze_bert

        # Optional cached instruction embedding (actor/critic share the same projection dim)
        self._instruction_emb = None

        # Build agent
        self._build_agent()

    def select_action(self, obses, last_actions, h_state, valids, avail_actions, eps=0.0, test_mode=False, using_tgt_net=False, instruction_emb=None):
        actions = [] # List[Int]
        with torch.no_grad():
            # Check if we have dynamic instruction timing (instruction provided even when not all agents need new actions)
            has_instruction = instruction_emb is not None and max(valids) != 1.0

            # Use agent with optional instruction embedding
            inst_emb = instruction_emb if instruction_emb is not None else self._instruction_emb

            if max(valids) == 1.0 or has_instruction:
                joint_avail_actions = get_joint_avail_actions(avail_actions)

                if not using_tgt_net:
                    action_logits, new_h_state = self.agent.actor_net(torch.cat(obses, dim=1).view(1,1,-1),
                                                                      h_state,
                                                                      eps=eps,
                                                                      test_mode=test_mode,
                                                                      instruction_emb=inst_emb)
                else:
                    action_logits, new_h_state = self.agent.actor_tgt_net(torch.cat(obses, dim=1).view(1,1,-1),
                                                                          h_state,
                                                                          eps=eps,
                                                                          test_mode=test_mode,
                                                                          instruction_emb=inst_emb)

                # Ensure inputs for conditional masking are Tensors
                last_actions_tensor = torch.cat(last_actions, dim=1) if isinstance(last_actions, list) else last_actions
                valids_tensor = torch.cat(valids, dim=1) if isinstance(valids, list) else valids
                # Convert valids to boolean for proper indexing in get_conditional_action
                valids_bool = valids_tensor.bool()
                action_logits = get_conditional_logits(action_logits,
                                                       get_conditional_action(last_actions_tensor,
                                                                              valids_bool),
                                                       joint_avail_actions,
                                                       self.env.n_action)

                # Sample new action with instruction guidance
                new_action_prob = Categorical(logits=action_logits[0])
                new_action = new_action_prob.sample().item()

                if has_instruction:
                    # Dynamic instruction timing: compare current vs new action
                    action = self._compare_actions(obses, last_actions, h_state, valids, new_action, eps, test_mode, using_tgt_net)
                else:
                    # Standard macro-action selection
                    action = new_action

                actions = np.unravel_index(action, self.env.n_action)
            else:
                actions = last_actions
                new_h_state = h_state
        return actions, new_h_state

    def _compare_actions(self, obses, last_actions, h_state, valids, new_action, eps, test_mode, using_tgt_net):
        """
        Compare current macro-action with new instruction-guided action and decide whether to switch.

        Note: This is a simplified implementation. Full macro-action resumption would require
        tracking which agents are in the middle of which macro-actions.

        For now: Always switch to instruction-guided action when instructions are provided mid-execution
        """
        # Simplified: Always use instruction-guided action when provided mid-execution
        # In a full implementation, this would compare current vs new action utility
        return new_action

    def _build_agent(self):
        """Build agent (trainable, optionally uses instructions)"""
        # Convert layer sizes to lists if they are integers
        a_mlp_size = [self.a_mlp_layer_size, self.a_mlp_layer_size] if isinstance(self.a_mlp_layer_size, int) else self.a_mlp_layer_size
        c_mlp_size = [self.c_mlp_layer_size, self.c_mlp_layer_size] if isinstance(self.c_mlp_layer_size, int) else self.c_mlp_layer_size
        
        self.agent = Agent()
        self.agent.actor_net = Actor(self._get_input_dim(), 
                                                 self._get_output_dim(), 
                                                 a_mlp_size, 
                                                 self.a_rnn_layer_size,
                                                 use_instructions=self.use_instructions,
                                                 instruction_fusion=self.instruction_fusion,
                                                 freeze_bert=self.freeze_bert).to(self.device)
        self.agent.actor_tgt_net = Actor(self._get_input_dim(), 
                                                     self._get_output_dim(), 
                                                     a_mlp_size, 
                                                     self.a_rnn_layer_size,
                                                     use_instructions=self.use_instructions,
                                                     instruction_fusion=self.instruction_fusion,
                                                     freeze_bert=self.freeze_bert).to(self.device)
        self.agent.actor_tgt_net.load_state_dict(self.agent.actor_net.state_dict())
        
        # Instruct Critic (V_{Ψ_δ}) - conditioned on instructions
        self.agent.critic_net = Critic(self._get_input_dim(), 
                                                   1, 
                                                   c_mlp_size, 
                                                   self.c_rnn_layer_size,
                                                   use_instructions=self.use_instructions,
                                                   instruction_fusion=self.instruction_fusion,
                                                   freeze_bert=self.freeze_bert).to(self.device)
        self.agent.critic_tgt_net = Critic(self._get_input_dim(), 
                                                       1, 
                                                       c_mlp_size, 
                                                       self.c_rnn_layer_size,
                                                       use_instructions=self.use_instructions,
                                                       instruction_fusion=self.instruction_fusion,
                                                       freeze_bert=self.freeze_bert).to(self.device)
        self.agent.critic_tgt_net.load_state_dict(self.agent.critic_net.state_dict())
        
        # Normal Critic (V_Ψ) - NOT conditioned on instructions (HRI Project)
        self.agent.normal_critic_net = Critic(self._get_input_dim(), 
                                                          1, 
                                                          c_mlp_size, 
                                                          self.c_rnn_layer_size,
                                                          use_instructions=False,  # Normal critic doesn't use instructions
                                                          instruction_fusion='concat',
                                                          freeze_bert=True).to(self.device)
        self.agent.normal_critic_tgt_net = Critic(self._get_input_dim(), 
                                                              1, 
                                                              c_mlp_size, 
                                                              self.c_rnn_layer_size,
                                                              use_instructions=False,
                                                              instruction_fusion='concat',
                                                              freeze_bert=True).to(self.device)
        self.agent.normal_critic_tgt_net.load_state_dict(self.agent.normal_critic_net.state_dict())
        
        print("Agent created (trainable) with dual critics for HRI")

    def set_instruction_text(self, instruction_text):
        """Encode and cache instruction embedding for use during action selection/training."""
        if not self.use_instructions or instruction_text is None:
            self._instruction_emb = None
            return
        # Use agent's actor encoder (same projection dim is used for critic)
        self.agent.actor_net.eval()
        with torch.no_grad():
            emb = self.agent.actor_net.encode_instruction(instruction_text)
        self._instruction_emb = emb.to(self.device)

    def clear_instruction(self):
        self._instruction_emb = None

    def get_instruction_emb(self):
        return self._instruction_emb

    def _get_input_dim(self):
        if not self.obs_last_action:
            return sum(self.env.obs_size)
        else:
            return sum([o_dim + a_dim for o_dim, a_dim in zip(*[self.env.obs_size, self.env.n_action])])

    def _get_output_dim(self):
        return np.prod(self.env.n_action)
