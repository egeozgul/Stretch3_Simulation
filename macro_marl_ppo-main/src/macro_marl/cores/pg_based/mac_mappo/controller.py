import torch 
import numpy as np

from torch.distributions import Categorical
from .envs_runner import EnvsRunner
from .models import Actor, Critic
from .utils import Agent, get_joint_avail_actions, get_conditional_action, get_conditional_logits

class MAC(object):
    
    def __init__(self, 
                 env, 
                 obs_last_action,
                 a_mlp_layer_size, 
                 a_rnn_layer_size, 
                 c_mlp_layer_size, 
                 c_rnn_layer_size, 
                 device,
                 enable_joint_critic=False):
        
        self.env = env
        self.n_agent = env.n_agent
        self.obs_last_action = obs_last_action
        self.device = device
        self.enable_joint_critic = enable_joint_critic
        
        self.agents = []
        for i in range(self.n_agent):
            agent = Agent()
            agent.idx = i
            input_dim = self._get_local_input_dim(i)
            output_dim = self.env.n_action[i]
            actor = Actor(input_dim, output_dim, mlp_layer_size=a_mlp_layer_size, rnn_layer_size=a_rnn_layer_size).to(self.device)
            agent.actor_net = actor
            self.agents.append(agent)

    def select_action(self, obses, last_actions, h_state, valids, avail_actions, eps=0.0, test_mode=False, using_tgt_net=False):
        actions = []  # List[Int]
        agent_log_probs = None
        with torch.no_grad():
            if max(valids) == 1.0:
                eps_safe = float(eps)
                if not test_mode:
                    eps_safe = max(0.0, min(1.0 - 1e-6, eps_safe))
                else:
                    eps_safe = 0.0
                new_h_state = None
                actions = []
                agent_log_probs = []
                for i, agent in enumerate(self.agents):
                    obs_i = obses[i].to(self.device).view(1, 1, -1)
                    logits_i, _ = agent.actor_net(obs_i, eps=eps_safe, test_mode=test_mode)
                    # mask with availability
                    avail_i = avail_actions[i].to(self.device).view(1, -1)
                    logits_i = logits_i.view(1, -1)
                    invalid_mask = (avail_i == 0.0)
                    logits_i = logits_i.masked_fill(invalid_mask, -1e10)
                    if torch.all(invalid_mask):
                        # Fallback: make all actions equally likely to avoid NaNs
                        logits_i = torch.zeros_like(logits_i)
                    dist_i = Categorical(logits=logits_i)
                    a_i = dist_i.sample()
                    actions.append(a_i.item())
                    agent_log_probs.append(dist_i.log_prob(a_i).cpu().detach().view(1, 1))
            else:
                actions = last_actions
                new_h_state = h_state
                agent_log_probs = [torch.tensor(0.0).view(1, 1) for _ in range(self.n_agent)]
        return actions, new_h_state, agent_log_probs

    def _get_action(self, obs, avail_a, eps, test_mode):
        
        actions = []
        action_logits = []
        for agent_i, agent in enumerate(self.agents):
            a, a_logit = agent(obs[agent_i].unsqueeze(0), 
                               avail_a[agent_i].unsqueeze(0),
                               eps,
                               test_mode)
            actions.append(a)
            action_logits.append(a_logit)

        return action_logits, None # new_h_state is not returned by the new _get_action

    def _get_local_input_dim(self, agent_index):
        if not self.obs_last_action:
            return self.env.obs_size[agent_index]
        else:
            return self.env.obs_size[agent_index] + self.env.n_action[agent_index]
