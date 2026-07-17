import torch
import copy
import numpy as np
import os
import wandb
from torch.optim import Adam
import torch.nn.functional as F
from torch.nn.utils import clip_grad_value_, clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence

from itertools import chain
from .models import Critic, AgentCentricGRUCritic

class Learner(object):
    
    def __init__(self, 
                 env, 
                 run_id,
                 save_dir, 
                 controller, 
                 memory, 
                 gamma, 
                 tracking,
                 ppo_clip_value=0.2, 
                 ppo_epochs=10, 
                 obs_last_action=False,
                 a_lr=1e-4, 
                 c_lr=1e-3,
                 c_mlp_layer_size=[256, 256], 
                 c_rnn_layer_size=128,
                 tau=0.01,
                 grad_clip_value=None, 
                 grad_clip_norm=0.5,
                 TD_lambda=0.95,
                 device='cpu',
                 vf_coef=0.5,
                 n_train_repeat=None,
                 optimappo=False):

        self.env = env
        self.n_agent = env.n_agent
        self.controller = controller
        self.memory = memory
        self.gamma = gamma
        self.GAE_lambda = TD_lambda
        self.run_id = run_id
        self.save_dir = save_dir
        self.tracking = tracking
      
        self.a_lr = a_lr
        self.c_lr = c_lr
        self.ppo_epochs = ppo_epochs
        self.ppo_clip_value = ppo_clip_value
        # ACAC-style aliases
        self.clip_ratio = ppo_clip_value
        self.vf_coef = vf_coef
        self.n_train_repeat = ppo_epochs if n_train_repeat is None else n_train_repeat
        self.optimappo = optimappo
        self.obs_last_action = obs_last_action
        self.tau = tau
        self.grad_clip_value = grad_clip_value
        self.grad_clip_norm = grad_clip_norm
        self.device = device        
        self.n_iter = 0

        # Create critic based on enable_joint_critic parameter
        if self.controller.enable_joint_critic:
            # ACAC-style agent-centric critic
            if not self.obs_last_action:
                critic_input_dim = self.env.obs_size
            else:
                critic_input_dim = [o_dim + a_dim for o_dim, a_dim in zip(self.env.obs_size, self.env.n_action)]
            self.critic = AgentCentricGRUCritic(
                critic_input_dim, 1, c_mlp_layer_size, c_rnn_layer_size, 
                n_agent=self.n_agent
            ).to(self.device)
        else:
            # Original MAPPO-style simple critic
            critic_input_dim = sum(self.env.obs_size)
            self.critic = Critic(critic_input_dim, 1, c_mlp_layer_size, c_rnn_layer_size).to(self.device)
        
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.c_lr)
        
        self._set_optimizer()

    def _calculate_advantages_and_returns(self, reward, V_value, bootstrap,
                                          discount, terminate, exp_valid,
                                          epi_len, epsilon_std=1e-5):
        mac_discount = discount / torch.cat((self.gamma**-1*torch.ones((discount.shape[0],1,1)).to(self.device),
                                             discount[:,0:-1,:]),
                                             axis=1) 
        
        mask = mac_discount.isnan()
        mac_discount[mask] = 0.0
        
        advantage = torch.zeros_like(reward).to(self.device)
        Gt = torch.zeros_like(reward).to(self.device)
        Gt_valid = torch.zeros_like(reward).to(torch.bool).to(self.device)

        for epi_idx, epi_r in enumerate(reward):
            end_step_idx = epi_len[epi_idx].item() - 1
            
            # Handle the last timestep
            if not terminate[epi_idx][end_step_idx]:
                advantage[epi_idx][end_step_idx] = epi_r[end_step_idx] + mac_discount[epi_idx][end_step_idx] * bootstrap[epi_idx][end_step_idx] - bootstrap[epi_idx][end_step_idx-1]
            else:
                advantage[epi_idx][end_step_idx] = epi_r[end_step_idx] - bootstrap[epi_idx][end_step_idx-1]
            
            # Backward pass for GAE calculation
            for idx in range(end_step_idx-1, -1, -1):
                if idx == 0:
                    # For the first timestep, use initial value (V_value at idx=0)
                    delta = epi_r[idx] + mac_discount[epi_idx][idx] * bootstrap[epi_idx][idx] - V_value[epi_idx][idx]
                else:
                    delta = epi_r[idx] + mac_discount[epi_idx][idx] * bootstrap[epi_idx][idx] - bootstrap[epi_idx][idx-1]
                advantage[epi_idx][idx] = delta + mac_discount[epi_idx][idx] * self.GAE_lambda * advantage[epi_idx][idx + 1]
            
            # Calculate returns (Gt)
            value = torch.zeros_like(reward[epi_idx])
            value[:end_step_idx+1] = torch.cat([V_value[epi_idx][:1], bootstrap[epi_idx][:end_step_idx]], dim=0)
            Gt[epi_idx] = advantage[epi_idx] + value
            Gt_valid[epi_idx][:epi_len[epi_idx].item()] = True

        # Apply normalization to advantages (keeping MAPPO-style normalization)
        valid_mask_bool = exp_valid.squeeze(-1).bool()
        if valid_mask_bool.any():
            masked_advantages = advantage[valid_mask_bool]
            adv_mean = masked_advantages.mean()
            adv_std = masked_advantages.std()
            advantage = (advantage - adv_mean) / (adv_std + epsilon_std)
        
        # OptiMAPPO: Clip negative advantages to zero (equation 6)
        if self.optimappo:
            advantage = torch.clamp(advantage, min=0)
        
        normalized_advantages = advantage * exp_valid
        gae_lambda_returns = Gt * exp_valid

        return normalized_advantages, gae_lambda_returns

    def train(self, eps, c_hys_value, adv_hys_value, etrpy_w, critic_hys=False, adv_hys=False):
        batch, trace_len, epi_len = self.memory.sample()
        batch_size = len(batch)
        
        ### Make cen_batches and dec_batches (prepare training)    
        cen_exps, dec_exps = self._sep_joint_exps(batch)
        
        cen_batch, cen_trace_len, cen_epi_len = self._squeeze_cen_exp(cen_exps, batch_size, trace_len)
        jobs, j_r, n_jobs, j_terminate, j_discount, j_exp_valid, bootstrap = cen_batch
        
        dec_batches, dec_trace_lens, dec_epi_lens = self._squeeze_dec_exp(dec_exps, batch_size, trace_len)
        
        ### Prepare training
        with torch.no_grad():
            # Compute Gt, advantage using ACAC-style GAE
            if self.controller.enable_joint_critic:
                # For agent-centric critic, create valid mask for each agent
                mac_v_expanded = torch.ones(jobs.shape[0], jobs.shape[1], self.n_agent, dtype=torch.bool).to(self.device)
                V_s = self.critic(jobs, mac_v_expanded)[0]
            else:
                # For simple critic, just use joint observations
                V_s = self.critic(jobs)[0]
            
            actor_advantages, critic_targets = self._calculate_advantages_and_returns(
                j_r, V_s, bootstrap, j_discount, j_terminate, j_exp_valid, cen_epi_len
            )

            ### Compute log_old_pi(a|s) for each agent
            old_log_pi_a_agent = []
            for agent, d_batch in zip(self.controller.agents, dec_batches):
                obs, action, exp_valid, _ = d_batch
                
                if obs.shape[1] == 0:
                    continue

                action_logits = agent.actor_net(obs, eps=eps)[0]
                ### Append detached log_pi_a for each agent
                # Ensure action tensor is of correct dtype for gather operation
                action_long = action.long()
                old_log_pi_a_agent.append(action_logits.gather(-1, action_long))

        ### Repeat training for n_train_repeat times with the same batch
        for _ in range(self.n_train_repeat):
            actor_critic_loss = 0

            ### Train joint critic
            if self.controller.enable_joint_critic:
                # For agent-centric critic, create valid mask for each agent
                mac_v_expanded = torch.ones(jobs.shape[0], jobs.shape[1], self.n_agent, dtype=torch.bool).to(self.device)
                V_pred = self.critic(jobs, mac_v_expanded)[0]
            else:
                # For simple critic, just use joint observations
                V_pred = self.critic(jobs)[0]
            
            TD = critic_targets - V_pred
            
            if critic_hys:
                TD = torch.max(TD * c_hys_value, TD)
            joint_critic_loss = torch.sum(j_exp_valid * TD * TD) / j_exp_valid.sum()
            actor_critic_loss += self.vf_coef * joint_critic_loss
            
            if self.tracking:
                if not hasattr(self, 'diagnostics'):
                    self.diagnostics = {}
                if 'Joint/CriticLoss' not in self.diagnostics:
                    self.diagnostics['Joint/CriticLoss'] = []
                    self.diagnostics['Joint/Value'] = []
                    self.diagnostics['Joint/Advantage'] = []
                    self.diagnostics['Joint/VfCoef'] = []
                
                self.diagnostics['Joint/CriticLoss'].append(joint_critic_loss.detach().cpu().numpy())
                self.diagnostics['Joint/Value'].append((torch.sum(j_exp_valid * V_pred) / j_exp_valid.sum()).detach().cpu().numpy())
                self.diagnostics['Joint/Advantage'].append((torch.sum(j_exp_valid * actor_advantages) / j_exp_valid.sum()).detach().cpu().numpy())
                self.diagnostics['Joint/VfCoef'].append(self.vf_coef)

            self.critic_optimizer.zero_grad()
            joint_critic_loss.backward()
            if self.grad_clip_value:
                clip_grad_value_(self.critic.parameters(), self.grad_clip_value)
            if self.grad_clip_norm:
                clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
            self.critic_optimizer.step()

            ### Train actors
            for agent, d_batch, old_log_pi_a in zip(self.controller.agents, 
                                                   dec_batches, 
                                                   old_log_pi_a_agent):
                
                obs, action, exp_valid, _ = d_batch
                
                ### Get log_pi_a for each agent
                action_logits = agent.actor_net(obs, eps=eps)[0]
                # Ensure action tensor is of correct dtype for gather operation
                action_long = action.long()
                log_pi_a = action_logits.gather(-1, action_long)

                ### Calculate actor loss
                is_ratio = torch.exp((log_pi_a - old_log_pi_a) * exp_valid)
                clipped_rate = torch.sum(torch.logical_or(is_ratio < 1-self.clip_ratio, is_ratio > 1 + self.clip_ratio).to(float)) / torch.sum(exp_valid)

                # Use the joint advantages but ensure they match the per-agent tensor dimensions
                # The advantages are computed from centralized experience, so we use them directly
                # but we need to make sure dimensions align with exp_valid
                adv_agent = actor_advantages
                if adv_agent.shape[1] != exp_valid.shape[1]:
                    # Trim or pad to match exp_valid dimensions
                    min_len = min(adv_agent.shape[1], exp_valid.shape[1])
                    adv_agent = adv_agent[:, :min_len]
                    exp_valid = exp_valid[:, :min_len]
                    is_ratio = is_ratio[:, :min_len]
                    log_pi_a = log_pi_a[:, :min_len]
                    old_log_pi_a = old_log_pi_a[:, :min_len]
                    action_logits = action_logits[:, :min_len]
                
                # Apply advantage hysteric if enabled
                if adv_hys:
                    adv_agent = torch.max(adv_agent * adv_hys_value, adv_agent)

                ### Calculate entropy (after dimension alignment)
                pi_entropy = torch.distributions.Categorical(logits=action_logits * exp_valid).entropy()
                pi_entropy = pi_entropy.view(obs.shape[0], -1, 1)

                pg_loss1 = exp_valid * adv_agent * is_ratio
                pg_loss2 = exp_valid * adv_agent * torch.clamp(is_ratio, min=1-self.clip_ratio, max=1+self.clip_ratio)
                agent.actor_loss = (-torch.sum(torch.min(pg_loss1, pg_loss2), dim=-1, keepdim=True) * exp_valid).sum() / exp_valid.sum()
                actor_critic_loss += agent.actor_loss

                if self.tracking:
                    if f'Agent{agent.idx}/ActorLoss' not in self.diagnostics:
                        self.diagnostics[f'Agent{agent.idx}/ActorLoss'] = []
                        self.diagnostics[f'Agent{agent.idx}/Entropy'] = []
                    
                    self.diagnostics[f'Agent{agent.idx}/ActorLoss'].append(agent.actor_loss.detach().cpu().numpy())
                    self.diagnostics[f'Agent{agent.idx}/Entropy'].append((torch.sum(exp_valid * pi_entropy) / exp_valid.sum()).detach().cpu().numpy())

                agent.actor_optimizer.zero_grad()
                agent.actor_loss.backward()
                if self.grad_clip_value:
                    clip_grad_value_(agent.actor_net.parameters(), self.grad_clip_value)
                if self.grad_clip_norm:
                    clip_grad_norm_(agent.actor_net.parameters(), self.grad_clip_norm)
                agent.actor_optimizer.step()

        # Log diagnostics to wandb after all training repeats are completed
        if self.tracking:
            self._log_diagnostics_to_wandb()
        
        # Increment iteration counter
        self.n_iter += 1

    def _log_diagnostics_to_wandb(self):
        """Log collected diagnostics to wandb"""
        if not hasattr(self, 'diagnostics') or not self.diagnostics:
            return
            
        # Prepare the metrics dictionary for wandb logging
        wandb_metrics = {}
        
        # Log joint critic metrics (take the mean of recent values)
        for metric_key in ['Joint/CriticLoss', 'Joint/Value', 'Joint/Advantage', 'Joint/VfCoef']:
            if metric_key in self.diagnostics and len(self.diagnostics[metric_key]) > 0:
                # Take the mean of values from the most recent training batch
                recent_values = self.diagnostics[metric_key][-self.n_train_repeat:]
                wandb_metrics[metric_key] = np.mean(recent_values)
        
        # Log per-agent metrics
        for agent_idx in range(self.n_agent):
            for metric_suffix in ['ActorLoss', 'Entropy']:
                metric_key = f'Agent{agent_idx}/{metric_suffix}'
                if metric_key in self.diagnostics and len(self.diagnostics[metric_key]) > 0:
                    # Take the mean of values from the most recent training batch
                    recent_values = self.diagnostics[metric_key][-self.n_train_repeat:]
                    wandb_metrics[metric_key] = np.mean(recent_values)
        
        # Add iteration counter
        wandb_metrics['Learner/Iteration'] = self.n_iter
        
        # Log to wandb
        wandb.log(wandb_metrics, step=self.n_iter)
        
        # Optionally clear old diagnostic values to prevent memory buildup
        # Keep only the most recent values for each metric
        max_history = self.n_train_repeat * 10  # Keep last 10 training batches worth of data
        for metric_key in self.diagnostics:
            if len(self.diagnostics[metric_key]) > max_history:
                self.diagnostics[metric_key] = self.diagnostics[metric_key][-max_history:]

    def _sep_joint_exps(self, joint_exps):
        cen_exps = []
        dec_exps = [[] for _ in range(self.n_agent)]
        for last_o, last_avail_a, a, r, j_r, o, avail_a, t, mac_v, j_mac_v, exp_v, log_probs in chain(*joint_exps):
            # Centralized trajectories use the joint state at action time (last_o) and next joint state (o)
            cen_exps.append([
                torch.cat(last_o, dim=1).view(1, -1),
                j_r,
                torch.cat(o, dim=1).view(1, -1),
                t,
                j_mac_v,
                exp_v[0]
            ])
            # Decentralized per-agent records
            # log_probs may come as a single tensor in older paths; normalize to per-agent list
            if not isinstance(log_probs, (list, tuple)):
                log_probs = [log_probs for _ in range(self.n_agent)]
            for i in range(self.n_agent):
                dec_exps[i].append([last_o[i], a[i], mac_v[i], exp_v[i], log_probs[i]])
        return cen_exps, dec_exps

    def _squeeze_cen_exp(self, cen_batch, batch_size, trace_len):
        jobs_b, r_b, n_jobs_b, t_b, mac_v_b, exp_v_b = zip(*cen_batch)
        
        jo_b = torch.cat(jobs_b).view(batch_size, trace_len, -1)
        r_b = torch.cat(r_b).view(batch_size, trace_len, -1)
        n_jo_b = torch.cat(n_jobs_b).view(batch_size, trace_len, -1)
        t_b = torch.cat(t_b).view(batch_size, trace_len, -1)
        mac_v_b = torch.cat(mac_v_b).view(batch_size, trace_len)
        exp_v_b = torch.cat(exp_v_b).view(batch_size, trace_len, -1)
        discount_b = torch.pow(torch.ones(jo_b.shape[0], 1) * self.gamma, torch.arange(jo_b.shape[1])).unsqueeze(-1)

        squ_epi_len = mac_v_b.sum(1)
        
        def _squeeze_and_pad(tensor):
            # Filter out zero-length episodes to avoid empty sequences in RNN
            valid_lengths = squ_epi_len[squ_epi_len > 0]
            if len(valid_lengths) == 0:
                # If no valid episodes, return a minimal tensor
                return torch.zeros(1, 1, tensor.shape[-1]).to(self.device)
            
            valid_tensor = tensor[mac_v_b]
            squ_tensor = torch.split_with_sizes(valid_tensor, list(valid_lengths))
            padded = pad_sequence(squ_tensor, padding_value=0.0, batch_first=True).to(self.device)
            
            # If we filtered out some episodes, we need to pad back to original batch size
            if len(valid_lengths) < len(squ_epi_len):
                batch_diff = len(squ_epi_len) - len(valid_lengths)
                padding_tensor = torch.zeros(batch_diff, padded.shape[1], padded.shape[2]).to(self.device)
                padded = torch.cat([padded, padding_tensor], dim=0)
            
            return padded

        squ_jo_b = _squeeze_and_pad(jo_b)
        squ_r_b = _squeeze_and_pad(r_b)
        squ_n_jo_b = _squeeze_and_pad(n_jo_b)
        squ_t_b = _squeeze_and_pad(t_b)
        squ_exp_v_b = _squeeze_and_pad(exp_v_b)
        squ_discount_b = _squeeze_and_pad(discount_b)

        if self.controller.enable_joint_critic:
            # For agent-centric critic, create valid mask for each agent
            # Use the episode valid mask to create proper agent-wise valid masks
            mac_v_expanded = torch.ones(squ_jo_b.shape[0], squ_jo_b.shape[1], self.n_agent, dtype=torch.bool).to(self.device)
            bootstrap_values = self.critic(squ_n_jo_b, mac_v_expanded)[0].detach()
        else:
            bootstrap_values = self.critic(squ_n_jo_b)[0].detach()

        squ_cen_batch = (squ_jo_b, squ_r_b, squ_n_jo_b, squ_t_b, squ_discount_b, squ_exp_v_b, bootstrap_values)
        return squ_cen_batch, squ_jo_b.shape[1], squ_epi_len

    def _squeeze_dec_exp(self, dec_batches, batch_size, trace_len):
        squ_dec_batches = []
        for batch in dec_batches:
            obs_b, a_b, mac_v_b, exp_v_b, old_log_pi_a_b = zip(*batch)
            
            o_b = torch.cat(obs_b).view(batch_size, trace_len, -1)
            a_b = torch.cat(a_b).view(batch_size, trace_len, -1)
            mac_v_b = torch.cat(mac_v_b).view(batch_size, trace_len)
            exp_v_b = torch.cat(exp_v_b).view(batch_size, trace_len, -1)
            old_log_pi_a_b = torch.cat(old_log_pi_a_b).view(batch_size, trace_len, -1)

            squ_epi_len = mac_v_b.sum(1)
            
            def _squeeze_and_pad(tensor):
                # Filter out zero-length episodes to avoid empty sequences in RNN
                valid_lengths = squ_epi_len[squ_epi_len > 0]
                if len(valid_lengths) == 0:
                    # If no valid episodes, return a minimal tensor
                    return torch.zeros(1, 1, tensor.shape[-1]).to(self.device)
                
                valid_tensor = tensor[mac_v_b]
                squ_tensor = torch.split_with_sizes(valid_tensor, list(valid_lengths))
                padded = pad_sequence(squ_tensor, padding_value=0.0, batch_first=True).to(self.device)
                
                # If we filtered out some episodes, we need to pad back to original batch size
                if len(valid_lengths) < len(squ_epi_len):
                    batch_diff = len(squ_epi_len) - len(valid_lengths)
                    padding_tensor = torch.zeros(batch_diff, padded.shape[1], padded.shape[2]).to(self.device)
                    padded = torch.cat([padded, padding_tensor], dim=0)
                
                return padded

            squ_o_b = _squeeze_and_pad(o_b)
            squ_a_b = _squeeze_and_pad(a_b)
            squ_exp_v_b = _squeeze_and_pad(exp_v_b)
            squ_old_log_pi_a_b = _squeeze_and_pad(old_log_pi_a_b)

            squ_dec_batches.append((squ_o_b, squ_a_b, squ_exp_v_b, squ_old_log_pi_a_b))

        return squ_dec_batches, squ_dec_batches[0][0].shape[1], [b[0].shape[0] for b in squ_dec_batches]

    def _squeeze_tensor_by_mac_valid(self, tensor, mac_v, padding_value=0.0):
        """
        tensor: (batch, trace_len, n_agent) or (batch, trace_len) 
        mac_v: (batch, trace_len) 
        """
        squ_epi_len = mac_v.sum(1)
        squ_tensor = torch.split_with_sizes(tensor[mac_v], list(squ_epi_len))   
        
        padded_tensor = pad_sequence(squ_tensor, padding_value=torch.tensor(padding_value), batch_first=True).to(self.device)
        return padded_tensor

    def update_critic_target_net(self, soft=False):
        """Update critic target network (placeholder for ACAC compatibility)"""
        # MAPPO doesn't typically use target networks, but keeping for ACAC compatibility
        pass

    def update_actor_target_net(self, soft=False):
        """Update actor target networks (placeholder for ACAC compatibility)"""
        # MAPPO doesn't typically use target networks, but keeping for ACAC compatibility
        pass

    def _set_optimizer(self):
        for agent in self.controller.agents:
            agent.actor_optimizer = Adam(agent.actor_net.parameters(), lr=self.a_lr)
