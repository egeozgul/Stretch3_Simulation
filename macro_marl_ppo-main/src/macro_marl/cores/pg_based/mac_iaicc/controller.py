import torch 

from torch.distributions import Categorical

from .models import Actor, Critic, get_shared_instruction_encoder
from .utils import Agent

class MAC(object):
    
    def __init__(self, env, obs_last_action=False, 
                 a_mlp_layer_size=32, a_rnn_layer_size=32, 
                 c_mlp_layer_size=32, c_rnn_layer_size=32,
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

        # Load shared BERT encoder once if instructions are enabled
        if use_instructions:
            self.shared_encoder, self.shared_tokenizer = get_shared_instruction_encoder(device)
        else:
            self.shared_encoder = None
            self.shared_tokenizer = None

        self._build_agent()
        self._init_critic()

    def select_action(self, obses, h_states, valids, avail_actions, eps=0.0, test_mode=False, using_tgt_net=False, instruction_emb=None):
        actions = [] # List[Int]
        new_h_states = []
        with torch.no_grad():
            for idx, agent in enumerate(self.agents):
                if valids[idx]:
                    # Pass instruction embedding if provided
                    # Support per-agent instructions: if instruction_emb is a list, use instruction_emb[idx]
                    if instruction_emb is not None:
                        if isinstance(instruction_emb, list):
                            agent_instruction = instruction_emb[idx] if idx < len(instruction_emb) else None
                        else:
                            agent_instruction = instruction_emb
                    else:
                        agent_instruction = None
                    
                    if not using_tgt_net:
                        action_logits, new_h_state = agent.actor_net(obses[idx].view(1,1,-1), 
                                                                     h_states[idx], 
                                                                     eps=eps, 
                                                                     test_mode=test_mode,
                                                                     instruction_emb=agent_instruction)
                    else:
                        action_logits, new_h_state = agent.actor_tgt_net(obses[idx].view(1,1,-1), 
                                                                         h_states[idx], 
                                                                         eps=eps, 
                                                                         test_mode=test_mode,
                                                                         instruction_emb=agent_instruction)
                    action_logits = self._get_masked_logits(action_logits[0], avail_actions[idx])
                    #TODO check action_logitis shape
                    action_prob = Categorical(logits=action_logits[0])
                    action = action_prob.sample().item()
                    actions.append(action)
                    new_h_states.append(new_h_state)
                else:
                    actions.append(-1)
                    new_h_states.append(h_states[idx])
        return actions, new_h_states

    def _get_masked_logits(self, masked_logits, avail_action):
        avail_action = avail_action.to(masked_logits.device)
        return masked_logits.masked_fill(avail_action == 0.0, -float('inf'))


    def _build_agent(self):
        self.agents = []
        print(f"[_build_agent] Creating {self.n_agent} agents (from env.n_agent={self.env.n_agent})")
        for idx in range(self.n_agent):
            agent = Agent()
            agent.idx = idx
            agent.actor_net = Actor(self._get_actor_input_shape(idx), self.env.n_action[idx], self.a_mlp_layer_size, self.a_rnn_layer_size,
                                    use_instructions=self.use_instructions,
                                    instruction_fusion=self.instruction_fusion,
                                    shared_encoder=self.shared_encoder,
                                    shared_tokenizer=self.shared_tokenizer).to(self.device)
            agent.actor_tgt_net = Actor(self._get_actor_input_shape(idx), self.env.n_action[idx], self.a_mlp_layer_size, self.a_rnn_layer_size,
                                        use_instructions=self.use_instructions,
                                        instruction_fusion=self.instruction_fusion,
                                        shared_encoder=self.shared_encoder,
                                        shared_tokenizer=self.shared_tokenizer).to(self.device)
            agent.actor_tgt_net.load_state_dict(agent.actor_net.state_dict())
            self.agents.append(agent)
            print(f"[_build_agent] Created agent {idx}")
        print(f"[_build_agent] Total agents created: {len(self.agents)}")

    def _init_critic(self):
        for agent in self.agents:
            # Centralized critic uses joint observations (without last_action).
            # Each agent has its OWN centralized critic pair (Instruct + Normal).
            joint_obs_size = self._get_joint_obs_size()

            # Instruct Critic (V_{\Psi_\delta}) — centralized (joint obs), instruction-conditioned.
            agent.critic_net = Critic(joint_obs_size, 1, self.c_mlp_layer_size, self.c_rnn_layer_size,
                                      use_instructions=self.use_instructions,
                                      instruction_fusion=self.instruction_fusion,
                                      shared_encoder=self.shared_encoder,
                                      shared_tokenizer=self.shared_tokenizer).to(self.device)
            agent.critic_tgt_net = Critic(joint_obs_size, 1, self.c_mlp_layer_size, self.c_rnn_layer_size,
                                          use_instructions=self.use_instructions,
                                          instruction_fusion=self.instruction_fusion,
                                          shared_encoder=self.shared_encoder,
                                          shared_tokenizer=self.shared_tokenizer).to(self.device)
            agent.critic_tgt_net.load_state_dict(agent.critic_net.state_dict())

            # Normal Critic (V_{\Psi}) — centralized (joint obs), no instruction conditioning.
            # Used with the Instruct Critic to implement the value-cancellation / chain-break
            # return decomposition (see learner_1.py).
            agent.normal_critic_net = Critic(joint_obs_size, 1, self.c_mlp_layer_size, self.c_rnn_layer_size,
                                             use_instructions=False,
                                             instruction_fusion=self.instruction_fusion).to(self.device)
            agent.normal_critic_tgt_net = Critic(joint_obs_size, 1, self.c_mlp_layer_size, self.c_rnn_layer_size,
                                                 use_instructions=False,
                                                 instruction_fusion=self.instruction_fusion).to(self.device)
            agent.normal_critic_tgt_net.load_state_dict(agent.normal_critic_net.state_dict())

    def _get_joint_obs_size(self):
        """Get the size of joint observations (all agents' obs concatenated, without last actions)"""
        return sum(self.env.obs_size)

    def _get_actor_input_shape(self, agent_idx):
        if not self.obs_last_action:
            return self.env.obs_size[agent_idx]
        else:
            return self.env.obs_size[agent_idx] + self.env.n_action[agent_idx]

    def _get_critic_input_shape(self):
        if not self.obs_last_action:
            return sum(self.env.obs_size)
        else:
            return sum([o_dim + a_dim for o_dim, a_dim in zip(*[self.env.obs_size, self.env.n_action])])
