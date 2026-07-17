import copy
import os
from itertools import chain
import numpy as np
import torch
from torch.optim import Adam
from torch.nn.utils import clip_grad_value_, clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence

class Learner_ACAC(object):
    
    def __init__(self, 
                 env, 
                 controller, 
                 memory, 
                 gamma, 
                 obs_last_action=False,
                 a_lr=1e-2, 
                 c_lr=1e-2, 
                 c_mlp_layer_size=64, 
                 c_rnn_layer_size=64,
                 c_train_iteration=1, 
                 c_target_update_freq=50, 
                 n_train_repeat=1, 
                 n_minibatch=8,
                 tau=0.01,
                 grad_clip_value=None, 
                 grad_clip_norm=None,
                 n_step_TD=0, 
                 TD_lambda=0.0,
                 GAE_lambda=0.95,
                 device='cpu',
                 clip_ratio=0.1, 
                 vf_coef=0.5,
                 **kwargs):

        self.env = env
        self.n_agent = env.n_agent
        self.controller = controller
        self.memory = memory
        self.gamma = gamma

        self.a_lr = a_lr
        self.c_lr = c_lr
        self.c_mlp_layer_size = c_mlp_layer_size
        self.c_rnn_layer_size = c_rnn_layer_size
        self.c_train_iteration = c_train_iteration
        self.c_target_update_freq = c_target_update_freq
        self.n_train_repeat = n_train_repeat
        self.n_minibatch = n_minibatch

        self.obs_last_action = obs_last_action
        self.tau = tau
        self.grad_clip_value = grad_clip_value
        self.grad_clip_norm = grad_clip_norm
        self.n_step_TD = n_step_TD
        self.TD_lambda = TD_lambda
        self.GAE_lambda = GAE_lambda
        self.device = device
        self.clip_ratio = clip_ratio
        self.vf_coef = vf_coef

        # Dual-critic return handling toggles (mirrors mac_iac / mac_iaicc / mac_cac).
        # 1 (default): segmented chain-break returns — instruction rewards cannot
        #    leak into the benign (Normal) return chain.
        # 0: fall back to standard per-step bootstrap selection (no segmentation).
        self.use_chain_break = os.environ.get("USE_CHAIN_BREAK", "1") == "1"
        self.use_value_cancellation = os.environ.get("USE_VALUE_CANCELLATION", "1") == "1"
        print(f"[acac Learner_ACAC] USE_CHAIN_BREAK={int(self.use_chain_break)} "
              f"USE_VALUE_CANCELLATION={int(self.use_value_cancellation)} "
              f"(chain-break segmentation {'ACTIVE' if self.use_chain_break and self.use_value_cancellation else 'disabled'})")

        self.diagnostics = {}
        self.diagnostics[f'IPPO/Value'] = []
        self.diagnostics[f'IPPO/Advantage'] = []
        self.diagnostics[f'IPPO/CriticLoss'] = []
        self.diagnostics[f'IPPO/VfCoef'] = []
        for idx in range(self.n_agent):
            self.diagnostics[f'Agent{idx}/ActorLoss'] = []
            self.diagnostics[f'Agent{idx}/Entropy'] = []
            self.diagnostics[f'Agent{idx}/ISRatio'] = []
            self.diagnostics[f'Agent{idx}/ClipRate'] = []
        self._set_optimizer()

        print('Agent Centric Actor Critic ====================================================')
        if self.controller.use_popart:
            print('    with PopArt')
        

    def train(self, eps, c_hys_value, adv_hys_value, etrpy_w, critic_hys=False, adv_hys=False):

        batch, trace_len, epi_len = self.memory.sample()
        batch_size = len(batch)
        
        # Prepare centralized sequences and decentralized per-agent batches.
        cen_batch, dec_batches = self._sep_joint_exps(batch)
        cen_batch, _, cen_epi_len = self._squeeze_cen_exp(cen_batch, batch_size, trace_len)
        jobs, jr, n_jobs, j_terminate, mac_v_b, j_mac_v_b, j_discount, exp_valid, mac_st, _jobs_seq, _n_jobs_seq, j_inst = cen_batch
        dec_batches, dec_trace_lens, _ = self._squeeze_dec_exp(dec_batches, batch_size, trace_len, mac_v_b)

        # No positional encoding path in this learner.
        jobs_seq = None
        n_jobs_seq = None

        # Build is_instruct mask once from the joint-instruction tensor.
        # j_inst has shape (B, trace, joint_inst_dim); a step is an "instruct" step
        # when ANY agent's instruction slice is non-zero there.
        is_instruct_joint = None
        if j_inst is not None and j_inst.shape[-1] > 0:
            is_instruct_joint = (j_inst.abs().sum(dim=-1, keepdim=True) > 1e-6)

        with torch.no_grad():
            old_log_pi_a_agent = []
            agent_adv_values = []
            agent_returns = []
            agent_normal_returns = []
            agent_is_instruct = []

            for agent, d_batch, d_trace_len in zip(self.controller.agents, dec_batches, dec_trace_lens):
                obs_agent, action_agent, _discount_agent, exp_valid_agent, _obs_mask_agent, _obs_seq_agent, _mac_st_j, inst_agent = d_batch

                if obs_agent.shape[1] == 0:
                    old_log_pi_a_agent.append(None)
                    agent_adv_values.append(None)
                    agent_returns.append(None)
                    agent_normal_returns.append(None)
                    agent_is_instruct.append(None)
                    continue

                action_logits_agent = agent.actor_net(obs_agent, eps=eps, time_emb=None, instruction_emb=inst_agent)[0]
                old_log_pi_a_agent.append(action_logits_agent.gather(-1, action_agent).detach())

                # Dual-critic bootstrap (when the normal critic exists).
                if hasattr(agent, 'normal_critic_tgt_net'):
                    boot_result = self._get_bootstrap(
                        jobs, n_jobs, j_mac_v_b, mac_v_b, n_jobs_seq,
                        critic_tgt_net=agent.critic_tgt_net,
                        instruction_emb=j_inst,
                        normal_critic_tgt_net=agent.normal_critic_tgt_net,
                    )
                    old_init_value, old_bootstrap, old_normal_init, old_normal_bootstrap = boot_result
                    adv_value, Gt = self._get_gae(
                        jr, old_init_value, old_bootstrap, j_discount, j_terminate, cen_epi_len,
                        normal_initial_value=old_normal_init,
                        normal_bootstrap=old_normal_bootstrap,
                        is_instruct=is_instruct_joint,
                    )
                else:
                    old_init_value, old_bootstrap = self._get_bootstrap(
                        jobs, n_jobs, j_mac_v_b, mac_v_b, n_jobs_seq,
                        critic_tgt_net=agent.critic_tgt_net, instruction_emb=j_inst
                    )
                    adv_value, Gt = self._get_gae(jr, old_init_value, old_bootstrap, j_discount, j_terminate, cen_epi_len)

                agent_adv_values.append(torch.clamp(adv_value.detach(), min=0))
                agent_returns.append(Gt.detach())
                agent_is_instruct.append(is_instruct_joint)

        for _ in range(self.n_train_repeat):
            for agent, d_batch, d_trace_len, old_log_pi_a, adv_value, Gt in zip(
                self.controller.agents, dec_batches, dec_trace_lens, old_log_pi_a_agent, agent_adv_values, agent_returns
            ):
                obs_agent, action_agent, _discount_agent, exp_valid_agent, _obs_mask_agent, _obs_seq_agent, mac_st_j, inst_agent = d_batch
                if obs_agent.shape[1] == 0:
                    continue

                # Dual-critic update: when chain-break is active, each critic is
                # trained only on the segment it is responsible for — V_{Psi_delta}
                # on instruct steps, V_{Psi} on benign steps — so instruction
                # rewards cannot poison the benign value estimates.
                use_dual = (hasattr(agent, 'normal_critic_net')
                            and self.use_chain_break and self.use_value_cancellation
                            and is_instruct_joint is not None)

                for _ in range(self.c_train_iteration):
                    # Instruct critic forward pass (conditioned on j_inst).
                    values = agent.critic_net(jobs, mac_st, time_emb=jobs_seq, instruction_emb=j_inst)[0]
                    V_value = torch.split_with_sizes(values[torch.amax(mac_st, dim=-1).to(torch.bool)], list(cen_epi_len))
                    V_value = pad_sequence(V_value, padding_value=torch.tensor(0.0), batch_first=True).to(self.device)
                    TD = Gt - V_value
                    if critic_hys:
                        TD = torch.max(TD * c_hys_value, TD)

                    if use_dual:
                        # Only train V_{Psi_delta} on instruct steps.
                        is_instruct_mask_f = is_instruct_joint.to(V_value.dtype)
                        instruct_mask = exp_valid * is_instruct_mask_f
                        denom_inst = instruct_mask.sum().clamp(min=1.0)
                        instruct_critic_loss = torch.sum(instruct_mask * TD * TD) / denom_inst

                        # Normal critic forward pass (no instruction conditioning).
                        normal_values = agent.normal_critic_net(jobs, mac_st, time_emb=jobs_seq, instruction_emb=None)[0]
                        V_normal = torch.split_with_sizes(normal_values[torch.amax(mac_st, dim=-1).to(torch.bool)], list(cen_epi_len))
                        V_normal = pad_sequence(V_normal, padding_value=torch.tensor(0.0), batch_first=True).to(self.device)
                        TD_normal = Gt - V_normal
                        if critic_hys:
                            TD_normal = torch.max(TD_normal * c_hys_value, TD_normal)
                        benign_mask = exp_valid * (1.0 - is_instruct_mask_f)
                        denom_ben = benign_mask.sum().clamp(min=1.0)
                        normal_critic_loss = torch.sum(benign_mask * TD_normal * TD_normal) / denom_ben

                        critic_loss = instruct_critic_loss + normal_critic_loss
                    else:
                        critic_loss = torch.sum(exp_valid * TD * TD) / exp_valid.sum()

                    agent.critic_loss = critic_loss

                    agent.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    if self.grad_clip_value:
                        params = list(agent.critic_net.parameters())
                        if use_dual:
                            params += list(agent.normal_critic_net.parameters())
                        clip_grad_value_(params, self.grad_clip_value)
                    if self.grad_clip_norm:
                        params = list(agent.critic_net.parameters())
                        if use_dual:
                            params += list(agent.normal_critic_net.parameters())
                        clip_grad_norm_(params, self.grad_clip_norm)
                    agent.critic_optimizer.step()

                # Actor update (PPO clipped objective; per-agent instruction input).
                action_logits_agent = agent.actor_net(obs_agent, eps=eps, time_emb=None, instruction_emb=inst_agent)[0]
                log_pi_a = action_logits_agent.gather(-1, action_agent)
                pi_entropy = torch.distributions.Categorical(logits=action_logits_agent * exp_valid_agent).entropy()
                pi_entropy = pi_entropy.view(obs_agent.shape[0], d_trace_len, 1)

                is_ratio = torch.exp((log_pi_a - old_log_pi_a) * exp_valid_agent)
                clipped_rate = torch.sum(
                    torch.logical_or(is_ratio < 1 - self.clip_ratio, is_ratio > 1 + self.clip_ratio).to(float)
                ) / torch.sum(exp_valid_agent)

                adv_agent = self._squeeze_tensor_by_mac_valid(adv_value, mac_st_j)
                pg_loss1 = exp_valid_agent * adv_agent * is_ratio
                pg_loss2 = exp_valid_agent * adv_agent * torch.clamp(is_ratio, min=1 - self.clip_ratio, max=1 + self.clip_ratio)
                pg_loss = (-torch.sum(torch.min(pg_loss1, pg_loss2), dim=-1, keepdim=True) * exp_valid_agent).sum() / exp_valid_agent.sum()
                entropy_bonus = (torch.sum(exp_valid_agent * pi_entropy) / exp_valid_agent.sum())
                agent.actor_loss = pg_loss - etrpy_w * entropy_bonus

                agent.actor_optimizer.zero_grad()
                agent.actor_loss.backward()
                if self.grad_clip_value:
                    clip_grad_value_(agent.actor_net.parameters(), self.grad_clip_value)
                if self.grad_clip_norm:
                    clip_grad_norm_(agent.actor_net.parameters(), self.grad_clip_norm)
                agent.actor_optimizer.step()

                self.diagnostics[f'IPPO/CriticLoss'].append(agent.critic_loss.detach().cpu().numpy())
                self.diagnostics[f'IPPO/Value'].append((torch.sum(exp_valid * V_value) / exp_valid.sum()).detach().cpu().numpy())
                self.diagnostics[f'IPPO/Advantage'].append((torch.sum(exp_valid * TD) / exp_valid.sum()).detach().cpu().numpy())
                self.diagnostics[f'IPPO/VfCoef'].append(self.vf_coef)
                self.diagnostics[f'Agent{agent.idx}/ActorLoss'].append(agent.actor_loss.detach().cpu().numpy())
                self.diagnostics[f'Agent{agent.idx}/Entropy'].append((torch.sum(exp_valid_agent * pi_entropy) / exp_valid_agent.sum()).detach().cpu().numpy())
                self.diagnostics[f'Agent{agent.idx}/ISRatio'].append((torch.sum(is_ratio * exp_valid_agent) / torch.sum(exp_valid_agent)).detach().cpu().numpy())
                self.diagnostics[f'Agent{agent.idx}/ClipRate'].append(clipped_rate.detach().cpu().numpy())

    def update_critic_target_net(self, soft=False):
        if not soft:
            for agent in self.controller.agents:
                agent.critic_tgt_net.load_state_dict(agent.critic_net.state_dict())
                if hasattr(agent, 'normal_critic_net'):
                    agent.normal_critic_tgt_net.load_state_dict(agent.normal_critic_net.state_dict())
        else:
            for agent in self.controller.agents:
                with torch.no_grad():
                    for q, q_targ in zip(agent.critic_net.parameters(), agent.critic_tgt_net.parameters()):
                        q_targ.data.mul_(1 - self.tau)
                        q_targ.data.add_(self.tau * q.data)
                    if hasattr(agent, 'normal_critic_net'):
                        for q, q_targ in zip(agent.normal_critic_net.parameters(), agent.normal_critic_tgt_net.parameters()):
                            q_targ.data.mul_(1 - self.tau)
                            q_targ.data.add_(self.tau * q.data)

    def update_actor_target_net(self, soft=False):
        for agent in self.controller.agents:
            if not soft:
                agent.actor_tgt_net.load_state_dict(agent.actor_net.state_dict())
            else:
                with torch.no_grad():
                    for q, q_targ in zip(agent.actor_net.parameters(), agent.actor_tgt_net.parameters()):
                        q_targ.data.mul_(1 - self.tau)
                        q_targ.data.add_(self.tau * q.data)

    def get_diagnostics(self):
        diag = copy.deepcopy(self.diagnostics)
        for k in self.diagnostics.keys():
            self.diagnostics[k] = []
        return diag

    def _get_input_shape(self):
        if not self.obs_last_action:
            return self.env.obs_size # sum(self.env.obs_size)
        else:
            return [o_dim + a_dim for o_dim, a_dim in zip(*[self.env.obs_size, self.env.n_action])]

    def _set_optimizer(self):
        for agent in self.controller.agents:
            agent.actor_optimizer = Adam(agent.actor_net.parameters(), lr=self.a_lr)
            # Combine instruct + normal critic parameters under one optimizer so both
            # dual critics are updated together (mirrors mac_iac / mac_iaicc / mac_cac).
            if hasattr(agent, 'normal_critic_net'):
                critic_params = list(agent.critic_net.parameters()) + list(agent.normal_critic_net.parameters())
            else:
                critic_params = list(agent.critic_net.parameters())
            agent.critic_optimizer = Adam(critic_params, lr=self.c_lr)

    def _squeeze_tensor_by_mac_valid(self, tensor, mac_v, padding_value=0.0, popart=False):
        """
        tensor: (batch, trace_len, n_agent) or (batch, trace_len) 
        mac_v: (batch, trace_len) 
        """
        squ_epi_len = mac_v.sum(1)
        squ_tensor = torch.split_with_sizes(tensor[mac_v], list(squ_epi_len))   
        
        if popart:
            # PopArt path is disabled in IPPO learner.
            squ_tensor = [squ_t for squ_t in squ_tensor]

        padded_tensor = pad_sequence(squ_tensor, padding_value=torch.tensor(padding_value), batch_first=True).to(self.device)
        return padded_tensor

    def _get_bootstrap(self, squ_jo_b, squ_n_jo_b, squ_j_mac_v_b, squ_mac_v_b, squ_n_jo_seq, critic_tgt_net, instruction_emb=None, normal_critic_tgt_net=None):
        """
        Compute bootstrap values for GAE. Returns a tuple of
        (init_values, bootstrap_values[, normal_init_values, normal_bootstrap_values]).

        When ``normal_critic_tgt_net`` is provided, we additionally run the
        instruction-free critic (V_Psi) over the same sequence, which is used by
        the chain-break / value-cancellation return decomposition in ``_get_gae``.
        """
        jobs = torch.cat([squ_jo_b[:,0].unsqueeze(1),squ_n_jo_b],dim=1)
        mac_v = torch.cat([torch.ones([squ_mac_v_b.shape[0], 1, squ_mac_v_b.shape[2]]).to(self.device), squ_mac_v_b],dim=1).to(torch.bool)
        
        if squ_n_jo_seq is not None:
            n_jo_seq = torch.cat([torch.ones([squ_n_jo_seq.shape[0], 1]).to(self.device), squ_n_jo_seq], dim=1).to(torch.int64)
        else:
            n_jo_seq = None

        # Prepend first step's instruction to align with bootstrap obs (obs[0], n_obs[0..T])
        boot_inst = None
        if instruction_emb is not None:
            boot_inst = torch.cat([instruction_emb[:,0].unsqueeze(1), instruction_emb], dim=1)

        bootstrap_values = critic_tgt_net(
                jobs,
                mac_v,
                time_emb=n_jo_seq,
                instruction_emb=boot_inst
                )[0]
        
        init_values = bootstrap_values[:,0,:].unsqueeze(1)
        squ_bootstrap = self._squeeze_tensor_by_mac_valid(bootstrap_values[:,1:,:], squ_j_mac_v_b, popart=False)

        if normal_critic_tgt_net is None:
            return init_values, squ_bootstrap

        # Normal critic bootstrap — no instruction conditioning.
        normal_bootstrap_values = normal_critic_tgt_net(
                jobs,
                mac_v,
                time_emb=n_jo_seq,
                instruction_emb=None
                )[0]
        normal_init_values = normal_bootstrap_values[:,0,:].unsqueeze(1)
        squ_normal_bootstrap = self._squeeze_tensor_by_mac_valid(normal_bootstrap_values[:,1:,:], squ_j_mac_v_b, popart=False)
        return init_values, squ_bootstrap, normal_init_values, squ_normal_bootstrap


    def _sep_joint_exps(self, joint_exps):
        cen_exps = []
        dec_exps = [[] for _ in range(self.n_agent)]

        # Per-agent obs dims (accounting for obs_last_action, which the runner
        # has already concatenated into each agent's obs). When agents have
        # heterogeneous obs sizes (warehouse / OSD), we zero-pad each agent's
        # obs to max_obs_dim before concatenating into the joint obs, so the
        # centralized critic's reshape((B, T, n_agent, max_obs_dim)) works.
        # For homogeneous envs (Overcooked, BoxPushing) max_obs_dim equals
        # every per-agent dim and this is a no-op.
        if self.obs_last_action:
            per_agent_obs_dim = [o + a for o, a in zip(self.env.obs_size, self.env.n_action)]
        else:
            per_agent_obs_dim = list(self.env.obs_size)
        max_obs_dim = max(per_agent_obs_dim)
        heterogeneous_obs = len(set(per_agent_obs_dim)) > 1

        def _pad_to_max(o_tuple):
            if not heterogeneous_obs:
                return o_tuple
            return [
                torch.nn.functional.pad(o_tuple[i], (0, max_obs_dim - o_tuple[i].shape[-1]))
                for i in range(self.n_agent)
            ]

        for exp in chain(*joint_exps):
            # Handle both 12-element (no instruction) and 14-element (with instruction) tuples
            o, a_st, _avail_a, a, r, j_r, n_o, _n_avail_a, t, mac_v, j_mac_v, exp_v = exp[:12]
            inst_embs = exp[12] if len(exp) > 12 else None
            # inst_texts = exp[13] if len(exp) > 13 else None  # not needed for training

            # Build joint instruction: concatenate all agents' inst embeddings
            if inst_embs is not None:
                joint_inst = torch.cat(
                    [inst_embs[i] if inst_embs[i] is not None
                     else torch.zeros(1, self.controller.n_instructions)
                     for i in range(self.n_agent)],
                    dim=-1)  # 1 x (n_instructions * n_agent)
            else:
                joint_inst = torch.zeros(1, self.controller.n_instructions * self.n_agent)

            o_padded = _pad_to_max(o)
            n_o_padded = _pad_to_max(n_o)

            cen_exps.append([torch.cat(o_padded, dim=1).view(1,-1),
                         torch.cat(a_st, dim=1).view(1,-1),
                         max(a_st),
                         j_r,
                         torch.cat(n_o_padded, dim=1).view(1,-1),
                         t,
                         torch.cat(mac_v).view(1,-1),
                         j_mac_v,
                         exp_v[0],
                         joint_inst])
            
            for i in range(self.n_agent):
                agent_inst = inst_embs[i] if inst_embs is not None else None
                dec_exps[i].append([o[i], 
                                a_st[i],
                                max(a_st),
                                a[i], 
                                r[i], 
                                n_o[i], 
                                t, 
                                mac_v[i], 
                                j_mac_v,
                                exp_v[i],
                                agent_inst])
        return cen_exps, dec_exps

    def _squeeze_dec_exp(self, dec_batches, batch_size, trace_len, j_padded_mac_v_b):

        """
        squeeze experience for each agent and re-padding
        """

        squ_dec_batches = []
        squ_epi_lens = []
        squ_trace_lens = []

        j_padded_mac_v_b = j_padded_mac_v_b.to(self.device)

        for idx, batch in enumerate(dec_batches):
            # seperate elements in the batch
            obs_b, action_start_b, jaction_start_b, action_b, reward_b, next_obs_b, terminate_b, mac_valid_b, j_mac_valid_b, exp_valid_b, inst_b = zip(*batch)
            assert len(obs_b) == trace_len * batch_size, "number of states mismatch ..."
            assert len(next_obs_b) == trace_len * batch_size, "number of next states mismatch ..."
            o_b = torch.cat(obs_b).view(batch_size, trace_len, -1).to(self.device)
            a_b = torch.cat(action_b).view(batch_size, trace_len, -1).to(self.device)
            a_st_b = torch.cat(action_start_b).view(batch_size, trace_len).to(self.device)
            ja_st_b = torch.cat(jaction_start_b).view(batch_size, trace_len).to(self.device)
            mac_v_b = torch.cat(mac_valid_b).view(batch_size, trace_len).to(self.device)
            j_mac_v_b = torch.cat(j_mac_valid_b).view(batch_size, trace_len).to(self.device)
            exp_v_b = torch.cat(exp_valid_b).view(batch_size, trace_len, -1).to(self.device)
            discount_b = torch.pow(torch.ones(o_b.shape[0],1)*self.gamma, torch.arange(o_b.shape[1])).unsqueeze(-1).to(self.device) 

            # Process instruction embeddings
            has_instructions = self.controller.use_instructions and any(i is not None for i in inst_b)
            if has_instructions:
                inst_dim = self.controller.n_instructions
                inst_tensors = [i if i is not None else torch.zeros(1, inst_dim) for i in inst_b]
                inst_b_tensor = torch.cat(inst_tensors).view(batch_size, trace_len, -1).to(self.device)
            else:
                inst_b_tensor = None


            if not (ja_st_b.sum(1) == j_mac_v_b.sum(1)).all():
                self._mac_start_filter(ja_st_b, j_mac_v_b)
            assert all(ja_st_b.sum(1) == j_mac_v_b.sum(1)), "mask for joint mac start does not match with mask of joint mac done ..."

            if not (a_st_b.sum(1) == mac_v_b.sum(1)).all():
                self._mac_start_filter(a_st_b, mac_v_b)
            assert all(a_st_b.sum(1) == mac_v_b.sum(1)), "mask for mac start does not match with mask of mac done ..."

            # squeeze process
            squ_epi_len = mac_v_b.sum(1)
            assert all(squ_epi_len == j_padded_mac_v_b[:,:,idx].sum(1)), "Valid mask doesn't match ..."
            
            squ_o_b = self._squeeze_tensor_by_mac_valid(o_b, mac_v_b)
            squ_a_b = self._squeeze_tensor_by_mac_valid(a_b, mac_v_b)
            squ_discount_b = self._squeeze_tensor_by_mac_valid(discount_b, mac_v_b)
            squ_exp_v_b = self._squeeze_tensor_by_mac_valid(exp_v_b, mac_v_b)
            
            squ_o_b_attn_mask = self._generate_masking(squ_epi_len).to(self.device)
            
            if self.controller.time_emb:
                o_seq = torch.stack([torch.arange(1,o_b.shape[1]+1) for _ in range(0,o_b.shape[0])],dim=0).to(self.device)
                squ_o_seq = torch.split_with_sizes(o_seq[a_st_b], list(squ_epi_len)) 
                squ_o_seq = pad_sequence(squ_o_seq, padding_value=torch.tensor(0.0), batch_first=True).to(self.device)
            else:
                squ_o_seq = None

            squ_j_epi_len = j_mac_v_b.sum(1)
            squ_mac_st_jb = torch.split_with_sizes(a_st_b[ja_st_b], list(squ_j_epi_len))
            squ_mac_st_jb = pad_sequence(squ_mac_st_jb, padding_value=torch.tensor(0.0), batch_first=True).to(self.device)

            # Squeeze instruction embeddings
            squ_inst_b = None
            if inst_b_tensor is not None:
                squ_inst_b = self._squeeze_tensor_by_mac_valid(inst_b_tensor, mac_v_b)

            squ_dec_batches.append((squ_o_b,
                                    squ_a_b,
                                    squ_discount_b,
                                    squ_exp_v_b,
                                    squ_o_b_attn_mask,
                                    squ_o_seq,
                                    squ_mac_st_jb,
                                    squ_inst_b))

            squ_epi_lens.append(squ_epi_len)
            squ_trace_lens.append(squ_o_b.shape[1])

        return squ_dec_batches, squ_trace_lens, squ_epi_lens

    def _squeeze_cen_exp(self, cen_batch, batch_size, trace_len):

        """
        squeeze experience for each agent and re-padding
        """

        # seperate elements in the batch
        jobs_b, action_start_b, jaction_start_b, reward_b, next_jobs_b, terminate_b, mac_valid_b, j_mac_valid_b, exp_valid_b, joint_inst_b = zip(*cen_batch)
        assert len(jobs_b) == trace_len * batch_size, "number of states mismatch ..."
        assert len(next_jobs_b) == trace_len * batch_size, "number of next states mismatch ..."
        jo_b = torch.cat(jobs_b).view(batch_size, trace_len, -1).to(self.device)

        a_st_b = torch.cat(action_start_b).view(batch_size, trace_len, -1).to(self.device) 
        ja_st_b = torch.cat(jaction_start_b).view(batch_size, trace_len).to(self.device)

        r_b = torch.cat(reward_b).view(batch_size, trace_len, -1).to(self.device)
        n_jo_b = torch.cat(next_jobs_b).view(batch_size, trace_len, -1).to(self.device)
        t_b = torch.cat(terminate_b).view(batch_size, trace_len, -1).to(self.device)

        mac_v_b = torch.cat(mac_valid_b).view(batch_size, trace_len, -1).to(self.device)
        j_mac_v_b = torch.cat(j_mac_valid_b).view(batch_size, trace_len).to(self.device)
        exp_v_b = torch.cat(exp_valid_b).view(batch_size, trace_len, -1).to(self.device)
        discount_b = torch.pow(torch.ones(jo_b.shape[0],1)*self.gamma, torch.arange(jo_b.shape[1])).unsqueeze(-1).to(self.device)

        if not (ja_st_b.sum(1) == j_mac_v_b.sum(1)).all():
            self._mac_start_filter(ja_st_b, j_mac_v_b)
        assert all(ja_st_b.sum(1) == j_mac_v_b.sum(1)), "mask for joint mac start does not match with mask of joint mac done ..."

        squ_epi_len = j_mac_v_b.sum(1)
        squ_jo_b = self._squeeze_tensor_by_mac_valid(jo_b, j_mac_v_b) # joint observation
        squ_r_b = self._squeeze_tensor_by_mac_valid(r_b, j_mac_v_b) # reward
        squ_n_jo_b = self._squeeze_tensor_by_mac_valid(n_jo_b, j_mac_v_b) # next joint observation
        squ_t_b = self._squeeze_tensor_by_mac_valid(t_b, j_mac_v_b, padding_value=1.0) # terminated
        squ_mac_st_b = self._squeeze_tensor_by_mac_valid(a_st_b, ja_st_b) # macro action start
        squ_mac_v_b = self._squeeze_tensor_by_mac_valid(mac_v_b, j_mac_v_b) # macto action valid
        squ_j_mac_v_b = self._squeeze_tensor_by_mac_valid(j_mac_v_b, j_mac_v_b) # joint macro action valid
        squ_exp_v_b = self._squeeze_tensor_by_mac_valid(exp_v_b, j_mac_v_b) # experiment valid
        squ_discount_b = self._squeeze_tensor_by_mac_valid(discount_b, j_mac_v_b) # discounts

        if self.controller.time_emb:
            jo_seq = torch.stack([torch.arange(1,jo_b.shape[1]+1) for _ in range(0,jo_b.shape[0])],dim=0).to(self.device)
            n_jo_seq = torch.stack([torch.arange(2,jo_b.shape[1]+2) for _ in range(0,jo_b.shape[0])],dim=0).to(self.device)
            squ_jo_seq = self._squeeze_tensor_by_mac_valid(jo_seq, ja_st_b)
            squ_n_jo_seq = self._squeeze_tensor_by_mac_valid(n_jo_seq, j_mac_v_b)
        else:
            squ_jo_seq = squ_n_jo_seq = None

        # Squeeze joint instruction embeddings
        j_inst_b = torch.cat(joint_inst_b).view(batch_size, trace_len, -1).to(self.device)
        squ_j_inst_b = self._squeeze_tensor_by_mac_valid(j_inst_b, j_mac_v_b)

        squ_cen_batch = (squ_jo_b,
                         squ_r_b,
                         squ_n_jo_b,
                         squ_t_b,
                         squ_mac_v_b,
                         squ_j_mac_v_b,
                         squ_discount_b,
                         squ_exp_v_b,
                         squ_mac_st_b,
                         squ_jo_seq,
                         squ_n_jo_seq,
                         squ_j_inst_b,
                         )
        return squ_cen_batch, squ_jo_b.shape[1], squ_epi_len

    def _generate_masking(self, epi_length):
        max_seq_len = torch.max(epi_length).item()
        masking = []
        for length in epi_length:
            seq_len = length.item()
            pad_len = max_seq_len-seq_len
            masking.append(torch.cat([torch.ones([1,seq_len]),torch.zeros([1,pad_len])],dim=1)) #1,max_seq_len
        
        return torch.cat(masking,dim=0)

    def _mac_start_filter(self, mac_start, mac_end):

        mask = mac_start.sum(1) != mac_end.sum(1)
        selected_items = mac_start[mask]
        indices = torch.cat([i[-1].view(-1,2) for i in torch.split_with_sizes(selected_items.nonzero(as_tuple=False), 
                                                                              list(selected_items.sum(1)))], 
                                                                              dim=0)
        selected_items.scatter_(-1, indices[:,1].view(-1,1), 0.0)
        mac_start[mask] = selected_items
    
    def _get_gae(self, reward, initial_value, bootstrap, discount, terminate, epi_len,
                 normal_initial_value=None, normal_bootstrap=None, is_instruct=None):
        """
        Compute GAE advantage and return.

        With chain-break (``self.use_chain_break`` and ``self.use_value_cancellation``
        both true) and when ``is_instruct`` / ``normal_*`` are supplied, we:
          * Locate the first instruction step T per episode;
          * Run GAE independently on the benign segment [0, T-1] using V_{Psi}
            (normal critic) as value / bootstrap at every step, with boundary
            bootstrap V_{Psi}(s_T);
          * Run GAE independently on the instruct segment [T, end] using
            V_{Psi_delta} (instruct critic) as value / bootstrap at every step,
            with terminal bootstrap V_{Psi_delta}(s_{end+1});
          * The two segments share no GAE propagation — instruction returns
            cannot leak into the benign return chain (poisoning guarantee).

        Without chain-break, GAE is computed the original way using only the
        instruct critic's value/bootstrap (unchanged behavior for legacy runs).
        """
        # reward: n_batch x max_epi_length_agent
        mac_discount = discount / torch.cat((self.gamma**-1*torch.ones((discount.shape[0],1,1)).to(self.device),
                                             discount[:,0:-1,:]),
                                             axis=1) 
        
        mask = mac_discount.isnan()
        mac_discount[mask] = 0.0
        advantage = torch.zeros_like(reward).to(self.device)
        Gt = torch.zeros_like(reward).to(self.device)

        chain_break_active = (self.use_chain_break and self.use_value_cancellation
                              and is_instruct is not None
                              and normal_initial_value is not None
                              and normal_bootstrap is not None)

        # Standard (legacy) path — single critic.
        if not chain_break_active:
            for epi_idx, epi_r in enumerate(reward):
                end_step_idx = epi_len[epi_idx]-1
                if not terminate[epi_idx][end_step_idx]:
                    advantage[epi_idx][end_step_idx] = epi_r[end_step_idx] + mac_discount[epi_idx][end_step_idx] * bootstrap[epi_idx][end_step_idx] - bootstrap[epi_idx][end_step_idx-1]
                else:
                    advantage[epi_idx][end_step_idx] = epi_r[end_step_idx] - bootstrap[epi_idx][end_step_idx-1]

                for idx in range(end_step_idx-1, -1, -1):
                    if idx == 0:
                        delta = epi_r[idx] + mac_discount[epi_idx][idx] * bootstrap[epi_idx][idx] - initial_value[epi_idx]
                    else:
                        delta = epi_r[idx] + mac_discount[epi_idx][idx] * bootstrap[epi_idx][idx] - bootstrap[epi_idx][idx-1]
                    advantage[epi_idx][idx] = delta + mac_discount[epi_idx][idx] * self.GAE_lambda * advantage[epi_idx][idx + 1]
                value = torch.zeros_like(reward[epi_idx])
                value[:end_step_idx] = torch.cat([initial_value[epi_idx], bootstrap[epi_idx][:end_step_idx-1]], dim=0)
                Gt[epi_idx] = advantage[epi_idx] + value
            return advantage, Gt

        # ---- Chain-break path: segmented GAE ----
        def _segment_gae(seg_start, seg_end, init_val, boot, terminal_end, epi_idx, epi_r):
            """GAE over [seg_start, seg_end] using ``boot`` as V(s_{t+1}) values and
            ``init_val`` as V(s_0) for the very first step in the segment. The
            segment is treated as terminal at ``seg_end`` iff ``terminal_end`` is
            True; otherwise we bootstrap with ``boot[seg_end]``.
            """
            if seg_start > seg_end:
                return
            # Last step of segment.
            if not terminal_end:
                delta_end = epi_r[seg_end] + mac_discount[epi_idx][seg_end] * boot[seg_end] - (
                    boot[seg_end-1] if seg_end >= 1 else init_val.squeeze(0)
                )
            else:
                delta_end = epi_r[seg_end] - (
                    boot[seg_end-1] if seg_end >= 1 else init_val.squeeze(0)
                )
            advantage[epi_idx][seg_end] = delta_end
            # Backward through segment.
            for idx in range(seg_end - 1, seg_start - 1, -1):
                if idx == 0:
                    prev_v = init_val.squeeze(0)
                else:
                    prev_v = boot[idx - 1]
                delta = epi_r[idx] + mac_discount[epi_idx][idx] * boot[idx] - prev_v
                advantage[epi_idx][idx] = delta + mac_discount[epi_idx][idx] * self.GAE_lambda * advantage[epi_idx][idx + 1]

        for epi_idx, epi_r in enumerate(reward):
            end_step_idx = int(epi_len[epi_idx]) - 1

            # Locate the first instruction step T (chain-break boundary).
            T = end_step_idx + 1
            for t in range(end_step_idx + 1):
                # is_instruct shape: (B, trace, 1)
                if is_instruct[epi_idx][t].item():
                    T = t
                    break

            # Instruct segment [T, end_step_idx] — uses V_{Psi_delta}
            if T <= end_step_idx:
                _segment_gae(T, end_step_idx,
                             initial_value[epi_idx] if T == 0 else bootstrap[epi_idx][T-1:T],
                             bootstrap[epi_idx],
                             terminal_end=bool(terminate[epi_idx][end_step_idx]),
                             epi_idx=epi_idx, epi_r=epi_r)

            # Benign segment [0, min(T-1, end_step_idx)] — uses V_{Psi}.
            # Chain break: never propagates from the instruct segment; the benign
            # boundary bootstraps with V_{Psi}(s_T) regardless of whether the
            # episode actually continues into an instruct phase.
            if T > 0:
                benign_end = min(T - 1, end_step_idx)
                if T <= end_step_idx:
                    # Instruct follows: boundary is non-terminal, bootstrap with V_{Psi}.
                    _segment_gae(0, benign_end,
                                 normal_initial_value[epi_idx],
                                 normal_bootstrap[epi_idx],
                                 terminal_end=False,
                                 epi_idx=epi_idx, epi_r=epi_r)
                else:
                    # No instructions at all — standard end-of-episode bootstrap.
                    _segment_gae(0, benign_end,
                                 normal_initial_value[epi_idx],
                                 normal_bootstrap[epi_idx],
                                 terminal_end=bool(terminate[epi_idx][benign_end]),
                                 epi_idx=epi_idx, epi_r=epi_r)

            # Reconstruct Gt = advantage + V, with V coming from the correct
            # critic per segment so Gt is consistent with which critic trains
            # on this step.
            value = torch.zeros_like(reward[epi_idx])
            for t in range(end_step_idx + 1):
                if t < T:
                    # Benign step: use normal critic value.
                    if t == 0:
                        value[t] = normal_initial_value[epi_idx].squeeze(0)
                    else:
                        value[t] = normal_bootstrap[epi_idx][t-1]
                else:
                    # Instruct step: use instruct critic value.
                    if t == 0:
                        value[t] = initial_value[epi_idx].squeeze(0)
                    else:
                        value[t] = bootstrap[epi_idx][t-1]
            Gt[epi_idx] = advantage[epi_idx] + value

        return advantage, Gt