import torch 

from torch.distributions import Categorical

from macro_marl.cores.pg_based.acac.models import (
    AgentCentricGRUActor, AgentCentricGRUCritic, GRUEncoder
)
from macro_marl.cores.pg_based.acac.utils import Agent

class MAC(object):
    
    def __init__(self, env, obs_last_action=False, 
                a_mlp_layer_size=32, a_rnn_layer_size=32, 
                c_mlp_layer_size=32, c_rnn_layer_size=32,
                device='cpu', time_emb=False, time_emb_actor=False,
                init_critic=True, share_encoder=True, use_attention=True, 
                cc_n_head=2, enc_n_head=4, n_layer=1, value_head="concat", time_emb_alg='sinu',
                time_emb_dim=4, max_timestep=200, use_actor_ln=False, duplicate=False,
                use_popart=False,
                use_instructions=False, n_instructions=0):

        self.env = env
        self.n_agent = env.n_agent
        self.obs_last_action = obs_last_action
        self.init_critic = init_critic
        self.share_encoder = share_encoder
        self.use_attention = use_attention
        self.value_head = value_head
        self.a_mlp_layer_size = a_mlp_layer_size
        self.a_rnn_layer_size = a_rnn_layer_size
        self.c_mlp_layer_size = c_mlp_layer_size
        self.c_rnn_layer_size = c_rnn_layer_size
        self.cc_n_head = cc_n_head
        self.enc_n_head = enc_n_head
        self.n_layer = n_layer
        self.time_emb = time_emb
        self.time_emb_actor = time_emb_actor
        self.time_emb_alg = time_emb_alg
        self.time_emb_dim = time_emb_dim
        self.max_timestep = max_timestep
        self.use_actor_ln = use_actor_ln
        self.use_popart = use_popart
        self.duplicate = duplicate
        self.freeze_encoder = False
        self.use_instructions = use_instructions
        self.n_instructions = n_instructions
        
        self.device = device

        self._build_agent()
        if self.init_critic:
            self._init_critic()
    
    def select_action(
        self,
        obses,
        h_states,
        valids,
        avail_actions,
        eps=0.0,
        test_mode=False,
        using_tgt_net=False,
        obs_history=None,
        attn_mask=None,
        time_seq=None,
        instruction_emb=None
    ):
        # obses: n_envs x n_agent x (tensor: 1 x obs_dim)
        # h_states: n_envs x n_agent x hidden_dim (only for lstm, otherwise can be None)
        # valids: n_envs x n_agent
        # avail_actions: n_envs x n_agent x n_action
        #
        # obs_history is either:
        #   - a single tensor of shape (n_envs, n_agent, trace_len, obs_dim)
        #     when all agents share an obs_dim (Overcooked, BoxPushing); or
        #   - a list of n_agent tensors of shape (n_envs, trace_len, obs_dim_i)
        #     when agents have heterogeneous obs sizes (warehouse / OSD).
        # See envs_runner._obs_history_to_tensor for where the form is chosen.
        obs_history_is_list = isinstance(obs_history, (list, tuple))
        if obs_history_is_list:
            n_envs = obs_history[0].shape[0]
        else:
            n_envs = obs_history.shape[0]
        actions = [[] for _ in range(n_envs)] # List[Int], n_envs x n_agents

        # NOTE: parallel forward along envs, but still serial along agents
        with torch.no_grad():
            for idx, agent in enumerate(self.agents):
                if using_tgt_net:
                    actor_net = agent.actor_tgt_net
                else:
                    actor_net = agent.actor_net

                
                    
                # NOTE: hidden states is not 'batch first': 1 x n_envs x hidden_dim
                #       And, hidden states device is already assigned
                time_emb = None 
                if self.time_emb_actor:
                    time_emb = time_seq[:, idx].long().to(self.device) 
                    time_emb = time_emb[:,-1].unsqueeze(-1)
                
                # Get per-agent instruction embedding
                agent_inst = None
                if self.use_instructions and instruction_emb is not None:
                    agent_insts = []
                    for env_idx in range(n_envs):
                        if (instruction_emb[env_idx] is not None and 
                            idx < len(instruction_emb[env_idx]) and 
                            instruction_emb[env_idx][idx] is not None):
                            inst = instruction_emb[env_idx][idx]
                            if inst.dim() == 1:
                                inst = inst.unsqueeze(0)  # 1 x n_instructions
                            agent_insts.append(inst)
                        else:
                            agent_insts.append(torch.zeros(1, self.n_instructions))
                    agent_inst = torch.cat(agent_insts, dim=0).to(self.device)  # n_envs x n_instructions
                
                # Slice this agent's last obs as a (n_envs, 1, obs_dim_i) tensor.
                # Both forms produce the same final shape for actor_net.
                if obs_history_is_list:
                    agent_last_obs = obs_history[idx][:, -1].unsqueeze(1).to(self.device)
                else:
                    agent_last_obs = obs_history[:, [idx], -1].to(self.device)

                action_logits, new_h_state = actor_net(agent_last_obs,
                                                        h_states[:, [idx]].transpose(1, 0), 
                                                        eps=eps, 
                                                        test_mode=test_mode,
                                                        time_emb=time_emb,
                                                        instruction_emb=agent_inst)
                        
                action_logits = action_logits.cpu()
                for env_idx in range(action_logits.shape[0]):
                    # NOTE: Is it okay the dims differ?
                    # action_logits[env_idx]: act_dim / avail_actions[env_idx][idx]: 1 x act_dim
                    action_logits[env_idx] = self._get_masked_logits(action_logits[env_idx], avail_actions[env_idx][idx])

                action_prob = Categorical(logits=action_logits)
                action = action_prob.sample()   # tensor(n_envs)
                for env_idx in range(len(action)):
                    if valids[env_idx][idx]:
                        actions[env_idx].append(action[env_idx].item())
                        h_states[env_idx, idx] = new_h_state[0][env_idx]
                    else:
                        actions[env_idx].append(-1)
                
        return actions, h_states
    
    def _get_masked_logits(self, action_logits, avail_action):
        masked_logits = action_logits.clone()
        return  masked_logits.masked_fill(avail_action==0.0, -float('inf'))

    def _build_agent(self):
        self.agents = []
        for idx in range(self.n_agent):
            agent = Agent()
            agent.idx = idx
            if self.share_encoder:
                agent.encoder = GRUEncoder(
                    self._get_actor_input_shape(idx), 
                    self.a_mlp_layer_size, 
                    self.a_rnn_layer_size,
                    use_time_emb=self.time_emb_actor, 
                    time_emb_alg=self.time_emb_alg, 
                    time_emb_dim=self.time_emb_dim, 
                    max_timestep=self.max_timestep,
                    use_instructions=self.use_instructions,
                    n_instructions=self.n_instructions,
                ).to(self.device)
                agent.encoder_tgt = GRUEncoder(
                    self._get_actor_input_shape(idx), 
                    self.a_mlp_layer_size, 
                    self.a_rnn_layer_size,
                    use_time_emb=self.time_emb_actor, 
                    time_emb_alg=self.time_emb_alg, 
                    time_emb_dim=self.time_emb_dim, 
                    max_timestep=self.max_timestep,
                    use_instructions=self.use_instructions,
                    n_instructions=self.n_instructions,
                ).to(self.device)
            agent.actor_net = AgentCentricGRUActor(self._get_actor_input_shape(idx), self.env.n_action[idx], 
                                                    self.a_mlp_layer_size, self.a_rnn_layer_size, 
                                                    encoder=agent.encoder,
                                                    use_time_emb=self.time_emb_actor, 
                                                    time_emb_alg=self.time_emb_alg, time_emb_dim=self.time_emb_dim, 
                                                    max_timestep=self.max_timestep, use_ln=self.use_actor_ln,
                                                    use_instructions=self.use_instructions,
                                                    n_instructions=self.n_instructions).to(self.device)
            agent.actor_tgt_net = AgentCentricGRUActor(self._get_actor_input_shape(idx), self.env.n_action[idx], 
                                                        self.a_mlp_layer_size, self.a_rnn_layer_size, 
                                                        encoder=agent.encoder_tgt,
                                                        use_time_emb=self.time_emb_actor, 
                                                        time_emb_alg = self.time_emb_alg,time_emb_dim=self.time_emb_dim, 
                                                        max_timestep=self.max_timestep, use_ln=self.use_actor_ln,
                                                        use_instructions=self.use_instructions,
                                                        n_instructions=self.n_instructions).to(self.device)
                
            agent.actor_tgt_net.load_state_dict(agent.actor_net.state_dict())
            self.agents.append(agent)

    def _init_critic(self):
        if self.share_encoder:
            encoders = [agent.encoder for agent in self.agents]
            encoders_tgt = [agent.encoder_tgt for agent in self.agents]
        else:
            encoders = encoders_tgt = None
        critic_use_instructions = self.use_instructions
        critic_n_instructions = self.n_instructions
        for agent in self.agents:
            
            # Instruct Critic (V_{Psi_delta}) — conditioned on instructions.
            # Used together with the Normal Critic below for the value-cancellation /
            # chain-break return decomposition (see learner_acac.py).
            agent.critic_net = AgentCentricGRUCritic(
                self._get_critic_input_shape(),
                1,
                self.c_mlp_layer_size,
                self.c_rnn_layer_size,
                n_agent=self.n_agent,
                encoders=encoders,
                use_attention=self.use_attention,
                cc_n_head=self.cc_n_head,
                value_head=self.value_head,
                use_time_emb=self.time_emb,
                time_emb_alg = self.time_emb_alg,
                time_emb_dim=self.time_emb_dim, 
                max_timestep=self.max_timestep,
                use_popart=self.use_popart,
                duplicate=self.duplicate,
                use_instructions=critic_use_instructions,
                n_instructions=critic_n_instructions,).to(self.device)
            agent.critic_tgt_net = AgentCentricGRUCritic(
                self._get_critic_input_shape(),
                1,
                self.c_mlp_layer_size,
                self.c_rnn_layer_size,
                n_agent=self.n_agent,
                encoders=encoders_tgt,
                use_attention=self.use_attention,
                cc_n_head=self.cc_n_head,
                value_head=self.value_head,
                use_time_emb=self.time_emb,
                time_emb_alg = self.time_emb_alg,
                time_emb_dim=self.time_emb_dim, 
                max_timestep=self.max_timestep,
                use_popart=self.use_popart,
                duplicate=self.duplicate,
                use_instructions=critic_use_instructions,
                n_instructions=critic_n_instructions,).to(self.device)
            
            agent.critic_tgt_net.load_state_dict(agent.critic_net.state_dict())

            # Normal Critic (V_{Psi}) — NOT conditioned on instructions.
            # Trained only on "benign" (no-instruction) segments; used to bootstrap
            # at the Normal->Instruct boundary in the chain-break return decomposition.
            # Uses its own (non-shared) encoders so gradients from the normal critic
            # do not contaminate the instruction-conditioned actor/encoder pipeline.
            agent.normal_critic_net = AgentCentricGRUCritic(
                self._get_critic_input_shape(),
                1,
                self.c_mlp_layer_size,
                self.c_rnn_layer_size,
                n_agent=self.n_agent,
                encoders=None,
                use_attention=self.use_attention,
                cc_n_head=self.cc_n_head,
                value_head=self.value_head,
                use_time_emb=self.time_emb,
                time_emb_alg=self.time_emb_alg,
                time_emb_dim=self.time_emb_dim,
                max_timestep=self.max_timestep,
                use_popart=self.use_popart,
                duplicate=self.duplicate,
                use_instructions=False,
                n_instructions=0,).to(self.device)
            agent.normal_critic_tgt_net = AgentCentricGRUCritic(
                self._get_critic_input_shape(),
                1,
                self.c_mlp_layer_size,
                self.c_rnn_layer_size,
                n_agent=self.n_agent,
                encoders=None,
                use_attention=self.use_attention,
                cc_n_head=self.cc_n_head,
                value_head=self.value_head,
                use_time_emb=self.time_emb,
                time_emb_alg=self.time_emb_alg,
                time_emb_dim=self.time_emb_dim,
                max_timestep=self.max_timestep,
                use_popart=self.use_popart,
                duplicate=self.duplicate,
                use_instructions=False,
                n_instructions=0,).to(self.device)
            agent.normal_critic_tgt_net.load_state_dict(agent.normal_critic_net.state_dict())

    def _get_actor_input_shape(self, agent_idx):
        if not self.obs_last_action:
            return self.env.obs_size[agent_idx]
        else:
            return self.env.obs_size[agent_idx] + self.env.n_action[agent_idx]

    def _get_critic_input_shape(self):
        if not self.obs_last_action:
            per_agent = list(self.env.obs_size)
        else:
            per_agent = [o_dim + a_dim for o_dim, a_dim in zip(*[self.env.obs_size, self.env.n_action])]
        # AgentCentricGRUCritic.forward reshapes the joint obs as
        # (n_batch, trace_len, n_agent, -1), which requires a uniform per-agent
        # obs dim. Overcooked / BoxPushing already satisfy this. Warehouse / OSD
        # has heterogeneous sizes (e.g. [T, T, F]); we pad to the max so the
        # reshape works, and the learner's _sep_joint_exps zero-pads each
        # agent's obs to match before building the joint tensor.
        if len(set(per_agent)) > 1:
            max_dim = max(per_agent)
            return [max_dim] * len(per_agent)
        return per_agent

