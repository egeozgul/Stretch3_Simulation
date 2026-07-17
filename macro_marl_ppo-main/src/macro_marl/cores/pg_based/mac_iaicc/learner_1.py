import torch
import copy
import numpy as np

from torch.optim import Adam
from torch.nn.utils import clip_grad_value_, clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence

from itertools import chain
from .learner import Learner

class Learner_1(Learner):

    """
    V(h), where h is from the joint obs at any moment when any agent got new obs;
    
    THIS IS THE MAIN LEARNER FOR MAC-IAICC not learner.py
    """

    def __init__(self, *args, **kwargs):

        super(Learner_1, self).__init__(*args, **kwargs)

    def train(self, eps, c_hys_value, adv_hys_value, etrpy_w, critic_hys=False, adv_hys=False):

        batch, trace_len, epi_len = self.memory.sample()
        batch_size = len(batch)

        ############################# train individual centralized critic ###################################

        dec_batches = self._sep_joint_exps(batch)
        dec_batches, trace_lens, epi_lens = self._squeeze_dec_exp(self.controller.agents,
                                                                  dec_batches, 
                                                                  batch_size, 
                                                                  trace_len)

        for agent, batch, trace_len, epi_len in zip(self.controller.agents,
                                                    dec_batches,
                                                    trace_lens,
                                                    epi_lens):

            # Unpack batch with dual bootstraps + is_instruct mask.
            if len(batch) >= 13:
                (obs, jobs, action, reward, terminate, discount, exp_valid,
                 normal_bootstrap, instruct_bootstrap, is_instruct_local,
                 mac_st, joint_inst, local_inst) = batch
            elif len(batch) == 11:  # Older 11-tuple (single-bootstrap) path
                obs, jobs, action, reward, terminate, discount, exp_valid, bootstrap, mac_st, joint_inst, local_inst = batch
                normal_bootstrap = bootstrap
                instruct_bootstrap = bootstrap
                is_instruct_local = None
            else:  # Backward compatibility (no instructions at all)
                obs, jobs, action, reward, terminate, discount, exp_valid, bootstrap, mac_st = batch
                normal_bootstrap = bootstrap
                instruct_bootstrap = bootstrap
                is_instruct_local = None
                joint_inst = None
                local_inst = None

            if obs.shape[1] == 0:
                continue

            ##############################  calculate critic loss and optimize the critic_net ####################################
            for _ in range(self.c_train_iteration):
                if not self.TD_lambda:
                    Gt = self._get_bootstrap_return(reward,
                                                    normal_bootstrap,
                                                    instruct_bootstrap,
                                                    is_instruct_local,
                                                    discount,
                                                    terminate,
                                                    epi_len)
                else:
                    Gt = self._get_td_lambda_return(obs.shape[0],
                                                    trace_len,
                                                    epi_len,
                                                    reward,
                                                    normal_bootstrap,
                                                    instruct_bootstrap,
                                                    is_instruct_local,
                                                    discount,
                                                    terminate)

                # Current Value at local (macro-start) steps — per-step selection
                # between V_{Psi_delta} (Instruct) and V_{Psi} (Normal).
                V_value = self._current_V_at_local_steps(
                    agent, jobs, joint_inst, mac_st, epi_len, is_instruct_local)

                TD = Gt - V_value
                if critic_hys:
                    TD = torch.max(TD*c_hys_value, TD)
                agent.critic_loss = torch.sum(exp_valid * TD * TD) / torch.sum(exp_valid)
                agent.critic_optimizer.zero_grad()
                agent.critic_loss.backward()
                if self.grad_clip_value:
                    clip_grad_params = list(agent.critic_net.parameters())
                    if hasattr(agent, 'normal_critic_net'):
                        clip_grad_params += list(agent.normal_critic_net.parameters())
                    clip_grad_value_(clip_grad_params, self.grad_clip_value)
                if self.grad_clip_norm:
                    clip_grad_params = list(agent.critic_net.parameters())
                    if hasattr(agent, 'normal_critic_net'):
                        clip_grad_params += list(agent.normal_critic_net.parameters())
                    clip_grad_norm_(clip_grad_params, self.grad_clip_norm)
                agent.critic_optimizer.step()

            ##############################  calculate actor loss using the updated critic ####################################
            V_value = self._current_V_at_local_steps(
                agent, jobs, joint_inst, mac_st, epi_len, is_instruct_local,
                detach=True)
            adv_value = Gt - V_value
            if adv_hys:
                adv_value = torch.max(adv_value*adv_hys_value, adv_value)

            # Actor forward with local instructions
            if local_inst is not None:
                action_logits = agent.actor_net(obs, eps=eps, instruction_emb=local_inst)[0]
            else:
                action_logits = agent.actor_net(obs, eps=eps)[0]
            
            log_pi_a = action_logits.gather(-1, action)
            pi_entropy = torch.distributions.Categorical(logits=action_logits).entropy().view(obs.shape[0], 
                                                                                              trace_len, 
                                                                                              1)
            actor_loss = torch.sum(exp_valid * discount * (log_pi_a * adv_value + etrpy_w * pi_entropy), dim=1)
            agent.actor_loss = -1 * torch.sum(actor_loss) / exp_valid.sum()

            ############################# contrastive loss on instruction projection ######################
            # Push apart projected BERT embeddings of different instructions
            # so semantically similar texts (left vs right board) get distinct representations
            if len(self.instruction_texts) >= 2:
                c_loss = agent.actor_net.contrastive_loss(self.instruction_texts, margin=4.0)
                agent.actor_loss = agent.actor_loss + self.contrastive_weight * c_loss

            agent.actor_optimizer.zero_grad()
            agent.actor_loss.backward()
            if self.grad_clip_value:
                clip_grad_value_(agent.actor_net.parameters(), self.grad_clip_value)
            if self.grad_clip_norm:
                clip_grad_norm_(agent.actor_net.parameters(), self.grad_clip_norm)
            agent.actor_optimizer.step()

    def _sep_joint_exps(self, joint_exps):

        """
        seperate the joint experience for individual agents
        """

        exps = [[] for _ in range(self.n_agent)]
        for exp in chain(*joint_exps):
            # Unpack with or without instructions
            if len(exp) >= 14:  # With instructions
                o, a_st, avail_a, a, r, j_r, n_o, n_avail_a, t, mac_v, j_mac_v, exp_v, inst_embs, inst_texts = exp
            else:  # Without instructions (backward compatibility)
                o, a_st, avail_a, a, r, j_r, n_o, n_avail_a, t, mac_v, j_mac_v, exp_v = exp
                inst_embs = [z.clone() for z in self.memory.ZERO_INSTRUCTION]  # Correctly sized zero instruction
                inst_texts = [None] * self.n_agent
            
            for i in range(self.n_agent):
                # Get per-agent instruction embedding
                agent_inst_emb = inst_embs[i] if isinstance(inst_embs, list) and i < len(inst_embs) else inst_embs
                if agent_inst_emb is None:
                    agent_inst_emb = self.memory.ZERO_INSTRUCTION[i].clone()
                if not isinstance(agent_inst_emb, torch.Tensor):
                    if isinstance(agent_inst_emb, list) and agent_inst_emb:
                        if isinstance(agent_inst_emb[0], torch.Tensor):
                            agent_inst_emb = agent_inst_emb[0]
                        else:
                            agent_inst_emb = torch.tensor(agent_inst_emb, dtype=self.memory.ZERO_INSTRUCTION[0].dtype)
                    else:
                        agent_inst_emb = torch.as_tensor(agent_inst_emb, dtype=self.memory.ZERO_INSTRUCTION[0].dtype)
                # Ensure consistent 1D shape [emb_dim] (some embeddings may be [1, emb_dim])
                if agent_inst_emb.dim() > 1:
                    agent_inst_emb = agent_inst_emb.squeeze(0)
                
                exps[i].append([o[i], 
                                a_st[i],
                                max(a_st),
                                torch.cat(o, dim=1).view(1,-1), 
                                a[i], 
                                r[i], 
                                torch.cat(n_o, dim=1).view(1,-1), 
                                t, 
                                mac_v[i], 
                                j_mac_v,
                                exp_v[i],
                                agent_inst_emb])  # Per-agent instruction
        return exps

    def _squeeze_dec_exp(self, agents, dec_batches, batch_size, trace_len):

        """
        squeeze experience for each agent and re-padding
        """

        squ_dec_batches = []
        squ_epi_lens = []
        squ_trace_lens = []

        for agent, batch in zip(agents, dec_batches):

            # Unpack with instruction support
            if len(batch[0]) >= 12:  # Has instruction
                (obs_b, 
                action_start_b, 
                jaction_start_b,
                jobs_b, 
                action_b, 
                reward_b, 
                next_jobs_b, 
                terminate_b, 
                mac_valid_b, 
                j_mac_valid_b, 
                exp_valid_b,
                inst_b) = zip(*batch)
            else:  # No instruction (backward compatibility)
                (obs_b, 
                action_start_b, 
                jaction_start_b,
                jobs_b, 
                action_b, 
                reward_b, 
                next_jobs_b, 
                terminate_b, 
                mac_valid_b, 
                j_mac_valid_b, 
                exp_valid_b) = zip(*batch)
                inst_b = [self.memory.ZERO_INSTRUCTION[0].clone() for _ in range(len(obs_b))]  # Correctly sized zero instructions

            assert len(obs_b) == trace_len * batch_size, "number of obses mismatch ..."
            assert len(next_jobs_b) == trace_len * batch_size, "number of next joint obses mismatch ..."

            # Move tensors to the correct device
            o_b = torch.cat(obs_b).view(batch_size, trace_len, -1).to(self.device)
            a_st_b = torch.cat(action_start_b).view(batch_size, trace_len).to(self.device)
            ja_st_b = torch.cat(jaction_start_b).view(batch_size, trace_len).to(self.device)
            jo_b = torch.cat(jobs_b).view(batch_size, trace_len, -1).to(self.device)
            a_b = torch.cat(action_b).view(batch_size, trace_len, -1).to(self.device)
            r_b = torch.cat(reward_b).view(batch_size, trace_len, -1).to(self.device)
            n_jo_b = torch.cat(next_jobs_b).view(batch_size, trace_len, -1).to(self.device)
            t_b = torch.cat(terminate_b).view(batch_size, trace_len, -1).to(self.device)
            mac_v_b = torch.cat(mac_valid_b).view(batch_size, trace_len).to(self.device)
            j_mac_v_b = torch.cat(j_mac_valid_b).view(batch_size, trace_len).to(self.device)
            exp_v_b = torch.cat(exp_valid_b).view(batch_size, trace_len, -1).to(self.device)
            discount_b = torch.pow(torch.ones(o_b.shape[0],1).to(self.device)*self.gamma, torch.arange(o_b.shape[1]).to(self.device)).unsqueeze(-1)
            
            # Handle instruction embeddings — squeeze any [1, emb_dim] to [emb_dim] for consistent stacking
            def _normalize_inst(inst):
                if not isinstance(inst, torch.Tensor):
                    return self.memory.ZERO_INSTRUCTION[0].clone()
                if inst.dim() > 1:
                    return inst.squeeze(0)
                return inst
            inst_stacked = torch.stack([_normalize_inst(inst) for inst in inst_b])
            inst_dim = inst_stacked.shape[-1]
            inst_b_tensor = inst_stacked.view(batch_size, trace_len, -1).to(self.device)

            # Filter process
            if not (a_st_b.sum(1) == mac_v_b.sum(1)).all():
                self._mac_start_filter(a_st_b, mac_v_b)
            if not (ja_st_b.sum(1) == j_mac_v_b.sum(1)).all():
                self._mac_start_filter(ja_st_b, j_mac_v_b)

            assert (a_st_b.sum(1) == mac_v_b.sum(1)).all(), "mask for mac strat does not match with mask of mac done ..."
            assert (ja_st_b.sum(1) == j_mac_v_b.sum(1)).all(), "mask for joint mac strat does not match with mask of joint mac done ..."

            # Centralized perspective with squeezed joint observations
            squ_j_epi_len = j_mac_v_b.sum(1).to(self.device)
            squ_j_epi_len_list = squ_j_epi_len.tolist()
            squ_jo_b = torch.split_with_sizes(jo_b[j_mac_v_b], squ_j_epi_len_list)
            squ_n_jo_b = torch.split_with_sizes(n_jo_b[j_mac_v_b], squ_j_epi_len_list)
            squ_mac_v_b = torch.split_with_sizes(mac_v_b[j_mac_v_b], squ_j_epi_len_list)
            squ_inst_b = torch.split_with_sizes(inst_b_tensor[j_mac_v_b], squ_j_epi_len_list)
            squ_jo_b = pad_sequence(list(squ_jo_b), padding_value=0.0, batch_first=True)
            squ_n_jo_b = pad_sequence(list(squ_n_jo_b), padding_value=0.0, batch_first=True)
            squ_mac_v_b = pad_sequence(list(squ_mac_v_b), padding_value=0.0, batch_first=True)
            squ_inst_b_padded = pad_sequence(list(squ_inst_b), padding_value=0.0, batch_first=True)

            # Ensure tensors are on the same device for the comparison
            assert (squ_mac_v_b.sum(1).to(self.device) == mac_v_b.sum(1).to(self.device)).all(), "number of bootstrap_values will mismatch with local termination exp ..."

            # Bootstrap values from each agent's own centralized target critics.
            # Compute BOTH: V_{Psi_delta} (instruct-conditioned) and V_{Psi} (normal).
            # The learner chooses between them per step based on whether that step
            # carries a non-zero instruction.
            n_state_seq = torch.cat([squ_jo_b[:,0].unsqueeze(1), squ_n_jo_b], dim=1)
            inst_for_boot = torch.cat([squ_inst_b_padded[:,0].unsqueeze(1), squ_inst_b_padded], dim=1)

            instruct_bootstrap_values = agent.critic_tgt_net(
                n_state_seq, instruction_emb=inst_for_boot)[0].detach()[:,1:,:]
            if hasattr(agent, 'normal_critic_tgt_net'):
                normal_bootstrap_values = agent.normal_critic_tgt_net(
                    n_state_seq)[0].detach()[:,1:,:]
            else:
                # Fallback when value-cancellation isn't set up on this agent.
                normal_bootstrap_values = instruct_bootstrap_values

            instruct_bootstrap_values = instruct_bootstrap_values[squ_mac_v_b]
            normal_bootstrap_values = normal_bootstrap_values[squ_mac_v_b]

            # Check joint actions starting moments
            assert (jo_b[ja_st_b] == jo_b[j_mac_v_b]).all(), "joint observations do not match ..."
            squ_mac_st_b = torch.split_with_sizes(a_st_b[ja_st_b], squ_j_epi_len_list)
            squ_mac_st_b = pad_sequence(list(squ_mac_st_b), padding_value=0.0, batch_first=True)

            # Local perspective squeeze process
            squ_epi_len = mac_v_b.sum(1).to(self.device)
            squ_epi_len_list = squ_epi_len.tolist()
            squ_o_b = torch.split_with_sizes(o_b[mac_v_b], squ_epi_len_list)
            squ_a_b = torch.split_with_sizes(a_b[mac_v_b], squ_epi_len_list)
            squ_r_b = torch.split_with_sizes(r_b[mac_v_b], squ_epi_len_list)
            squ_t_b = torch.split_with_sizes(t_b[mac_v_b], squ_epi_len_list)
            squ_exp_v_b = torch.split_with_sizes(exp_v_b[mac_v_b], squ_epi_len_list)
            squ_normal_bootstraps = torch.split_with_sizes(normal_bootstrap_values, squ_epi_len_list)
            squ_instruct_bootstraps = torch.split_with_sizes(instruct_bootstrap_values, squ_epi_len_list)
            squ_discount_b = torch.split_with_sizes(discount_b[mac_v_b], squ_epi_len_list)

            # Re-padding
            squ_o_b = pad_sequence(list(squ_o_b), padding_value=0.0, batch_first=True)
            squ_a_b = pad_sequence(list(squ_a_b), padding_value=0.0, batch_first=True)
            squ_r_b = pad_sequence(list(squ_r_b), padding_value=0.0, batch_first=True)
            squ_t_b = pad_sequence(list(squ_t_b), padding_value=1.0, batch_first=True)
            squ_exp_v_b = pad_sequence(list(squ_exp_v_b), padding_value=0.0, batch_first=True)
            squ_normal_bootstraps = pad_sequence(list(squ_normal_bootstraps), padding_value=0.0, batch_first=True)
            squ_instruct_bootstraps = pad_sequence(list(squ_instruct_bootstraps), padding_value=0.0, batch_first=True)
            squ_discount_b = pad_sequence(list(squ_discount_b), padding_value=0.0, batch_first=True)

            # Squeeze instructions for local perspective
            squ_local_inst_b = torch.split_with_sizes(inst_b_tensor[mac_v_b], squ_epi_len_list)
            squ_local_inst_b = pad_sequence(list(squ_local_inst_b), padding_value=0.0, batch_first=True)

            # Per-local-step instruction mask: True when that step carries a non-zero instruction.
            # Drives per-step bootstrap / V_value selection and chain-break segmentation.
            is_instruct_local = (squ_local_inst_b.abs().sum(dim=-1, keepdim=True) > 1e-6)

            squ_dec_batches.append((squ_o_b,
                                    squ_jo_b,
                                    squ_a_b,
                                    squ_r_b,
                                    squ_t_b,
                                    squ_discount_b,
                                    squ_exp_v_b,
                                    squ_normal_bootstraps,
                                    squ_instruct_bootstraps,
                                    is_instruct_local,
                                    squ_mac_st_b,
                                    squ_inst_b_padded,    # Joint-level instructions
                                    squ_local_inst_b))    # Local-level instructions

            squ_epi_lens.append(squ_epi_len)
            squ_trace_lens.append(squ_o_b.shape[1])

        return squ_dec_batches, squ_trace_lens, squ_epi_lens


    def _current_V_at_local_steps(self, agent, jobs, joint_inst, mac_st, epi_len,
                                   is_instruct_local, detach=False):
        """Compute V at local (macro-start) steps with per-step critic selection.

        At each local step: V = V_{Psi_delta}(jobs, inst) if is_instruct_local
        else V_{Psi}(jobs). This routes critic gradients to the correct branch
        and produces a value that's consistent with the bootstrap used in Gt.
        """
        if joint_inst is not None:
            V_instruct = agent.critic_net(jobs, instruction_emb=joint_inst)[0]
        else:
            V_instruct = agent.critic_net(jobs)[0]
        if detach:
            V_instruct = V_instruct.detach()

        if hasattr(agent, 'normal_critic_net') and is_instruct_local is not None:
            V_normal = agent.normal_critic_net(jobs)[0]
            if detach:
                V_normal = V_normal.detach()
        else:
            V_normal = None

        epi_len_list = epi_len.tolist()

        def _to_local(V):
            V_local = torch.split_with_sizes(V[mac_st], epi_len_list)
            return pad_sequence(list(V_local), padding_value=0.0, batch_first=True).to(self.device)

        V_instruct_local = _to_local(V_instruct)
        if V_normal is None:
            return V_instruct_local

        V_normal_local = _to_local(V_normal)
        # is_instruct_local has shape (batch, max_local_epi, 1), already aligned with V_*_local.
        return torch.where(is_instruct_local, V_instruct_local, V_normal_local)

    def _mac_start_filter(self, mac_start, mac_end):

        mask = mac_start.sum(1) != mac_end.sum(1)
        selected_items = mac_start[mask]
        indices = torch.cat([i[-1].view(-1,2) for i in torch.split_with_sizes(selected_items.nonzero(as_tuple=False), 
                                                                              list(selected_items.sum(1)))], 
                                                                              dim=0)
        selected_items.scatter_(-1, indices[:,1].view(-1,1), 0.0)
        mac_start[mask] = selected_items

    def _get_bootstrap_return(self, reward, normal_bootstrap, instruct_bootstrap,
                              is_instruct, discount, terminate, epi_len):
        """
        n-step TD return with optional chain-break at Normal->Instruct boundary.

        When chain-break is enabled:
          - Benign segment [0, T-1]:  uses r_env and bootstraps with V_{Psi}.
          - Instruct segment [T, end]: uses r_env (+penalty if wrapped) and bootstraps with V_{Psi_delta}.
          The two segments are computed independently so no instruction returns
          can leak into benign returns (poisoning guarantee).

        Otherwise: standard n-step TD with per-step bootstrap selection
        (V_{Psi_delta} where instruction is active, V_{Psi} otherwise).
        """
        mac_discount = discount / torch.cat(
            (
                self.gamma**-1 * torch.ones((discount.shape[0], 1, 1), device=discount.device),
                torch.ones((discount.shape[0], discount.shape[1]-1, 1), device=discount.device)
            ),
            dim=1
        )
        mask = mac_discount.isnan()
        mac_discount[mask] = 0.0

        # Per-step pre-selected bootstrap (used when chain-break is disabled or for 1-step TD).
        if is_instruct is not None:
            boot_selected = torch.where(is_instruct, instruct_bootstrap, normal_bootstrap)
        else:
            boot_selected = normal_bootstrap

        # ---- 1-step TD (no cross-step accumulation — per-step selection is sufficient) ----
        if not self.n_step_TD or self.n_step_TD == 1:
            Gt = reward + mac_discount * boot_selected * (-terminate + 1)
            return Gt

        n = self.n_step_TD
        Gt = copy.deepcopy(reward)

        # ---- Standard mode: n-step TD without chain-break segmentation ----
        if (not self.use_chain_break) or (not self.use_value_cancellation) or (is_instruct is None):
            for epi_idx in range(Gt.shape[0]):
                end_step_idx = int(epi_len[epi_idx]) - 1
                epi_r = Gt[epi_idx]
                if not terminate[epi_idx][end_step_idx]:
                    epi_r[end_step_idx] += mac_discount[epi_idx][end_step_idx] * boot_selected[epi_idx][end_step_idx]
                for idx in range(end_step_idx - 1, -1, -1):
                    if idx > end_step_idx - n:
                        epi_r[idx] = epi_r[idx] + mac_discount[epi_idx][idx] * epi_r[idx + 1]
                    else:
                        boot_val = boot_selected[epi_idx][idx + n - 1]
                        if idx == 0:
                            epi_r[idx] = self._get_n_step_discounted_bootstrap_return(
                                reward[epi_idx][idx:idx + n], boot_val,
                                discount[epi_idx][idx:idx + n] / self.gamma**-1)
                        else:
                            epi_r[idx] = self._get_n_step_discounted_bootstrap_return(
                                reward[epi_idx][idx:idx + n], boot_val,
                                discount[epi_idx][idx:idx + n] / discount[epi_idx][idx - 1])
            return Gt

        # ---- Chain-break mode: independent benign + instruct segments ----
        for epi_idx in range(Gt.shape[0]):
            end_step_idx = int(epi_len[epi_idx]) - 1
            epi_r = Gt[epi_idx]

            # Locate first instruction step T (chain-break boundary).
            T = end_step_idx + 1  # default: no instructions in this episode
            for t in range(end_step_idx + 1):
                if is_instruct[epi_idx][t].item():
                    T = t
                    break

            def _segment_returns(seg_start, seg_end, V_boot_array, terminal_end):
                if seg_start > seg_end:
                    return
                if not terminal_end:
                    epi_r[seg_end] += mac_discount[epi_idx][seg_end] * V_boot_array[seg_end]
                for idx in range(seg_end - 1, seg_start - 1, -1):
                    if idx > seg_end - n:
                        epi_r[idx] = epi_r[idx] + mac_discount[epi_idx][idx] * epi_r[idx + 1]
                    else:
                        boot_val = V_boot_array[idx + n - 1]
                        if idx == 0:
                            epi_r[idx] = self._get_n_step_discounted_bootstrap_return(
                                reward[epi_idx][idx:idx + n], boot_val,
                                discount[epi_idx][idx:idx + n] / self.gamma**-1)
                        else:
                            epi_r[idx] = self._get_n_step_discounted_bootstrap_return(
                                reward[epi_idx][idx:idx + n], boot_val,
                                discount[epi_idx][idx:idx + n] / discount[epi_idx][idx - 1])

            # Instruct segment [T, end_step_idx]: bootstrap with V_{Psi_delta}.
            if T <= end_step_idx:
                _segment_returns(T, end_step_idx,
                                 instruct_bootstrap[epi_idx],
                                 bool(terminate[epi_idx][end_step_idx]))

            # Benign segment [0, min(T-1, end_step_idx)]: bootstrap with V_{Psi}.
            if T > 0:
                benign_end = min(T - 1, end_step_idx)
                if T <= end_step_idx:
                    # Chain break: episode continues into instruct phase, so
                    # the benign segment treats the boundary as non-terminal
                    # and bootstraps with the normal critic.
                    _segment_returns(0, benign_end,
                                     normal_bootstrap[epi_idx], False)
                else:
                    _segment_returns(0, benign_end,
                                     normal_bootstrap[epi_idx],
                                     bool(terminate[epi_idx][benign_end]))

        return Gt

    def _get_n_step_discounted_bootstrap_return(self, reward, bootstrap, discount):
        reward_tensor = reward if isinstance(reward, torch.Tensor) else torch.as_tensor(reward)
        bootstrap_tensor = bootstrap if isinstance(bootstrap, torch.Tensor) else torch.as_tensor(bootstrap)
        if bootstrap_tensor.dim() == 0:
            bootstrap_tensor = bootstrap_tensor.view(1, 1)
        rewards = torch.cat((reward_tensor, bootstrap_tensor.reshape(-1, 1)), dim=0)

        discount_tensor = discount if isinstance(discount, torch.Tensor) else torch.as_tensor(discount)
        discounts = torch.cat((torch.ones((1, 1), device=discount_tensor.device), discount_tensor), dim=0)
        Gt = torch.sum(discounts * rewards) 
        return Gt

    def _get_td_lambda_return(self, batch_size, trace_len, epi_len, reward,
                              normal_bootstrap, instruct_bootstrap, is_instruct,
                              discount, terminate):
        # Monte-Carlo return (with chain-break segmentation if enabled).
        Gt = self._get_discounted_return(reward, normal_bootstrap, instruct_bootstrap,
                                         is_instruct, terminate, epi_len)
        # n-step bootstrap returns.
        self.n_step_TD = 0
        n_step_part = self._get_bootstrap_return(reward, normal_bootstrap, instruct_bootstrap,
                                                 is_instruct, discount, terminate, epi_len)
        for n in range(2, trace_len):
            self.n_step_TD = n
            next_n_step_part = self._get_bootstrap_return(reward, normal_bootstrap, instruct_bootstrap,
                                                          is_instruct, discount, terminate, epi_len)
            n_step_part = torch.cat([n_step_part, next_n_step_part], dim=-1)
        # lambda weights for n-step part.
        lmdas = torch.pow(torch.ones(1,1)*self.TD_lambda, torch.arange(trace_len-1)).repeat(trace_len, 1).unsqueeze(0).repeat(batch_size,1,1)
        mask = (torch.arange(trace_len).view(-1,1) + torch.arange(trace_len-1).view(1,-1)).squeeze(0).repeat(batch_size,1,1)
        mask = mask >= epi_len.view(batch_size, -1, 1)-1
        lmdas[mask] = 0.0
        # lambda weight for MC part.
        MC_lmdas = torch.zeros_like(Gt)
        for epi_id, length in enumerate(epi_len):
            last_step_lmda = torch.pow(torch.ones(1,1)*self.TD_lambda, torch.arange(length-1,-1,-1)).view(-1,1)
            MC_lmdas[epi_id][0:length] += last_step_lmda
        # TD(lambda) return.
        Gt = (1 - self.TD_lambda) * torch.sum(lmdas * n_step_part, dim=-1, keepdim=True) + MC_lmdas * Gt
        return Gt

    def _get_discounted_return(self, reward, normal_bootstrap, instruct_bootstrap,
                               is_instruct, terminate, epi_len):
        """
        Monte-Carlo return with optional chain-break at Normal->Instruct boundary.

        Chain-break mode (default):
          Benign segment [0, T-1]:  backward-accumulate, bootstrap at boundary with V_{Psi}.
          Instruct segment [T, end]: backward-accumulate, bootstrap at end with V_{Psi_delta}.
          The two segments are independent — no instruction returns leak into benign returns.

        Standard mode: per-step bootstrap selection (V_{Psi_delta} where active, V_{Psi} otherwise).
        """
        Gt = copy.deepcopy(reward)

        if is_instruct is not None:
            boot_selected = torch.where(is_instruct, instruct_bootstrap, normal_bootstrap)
        else:
            boot_selected = normal_bootstrap

        # ---- Standard mode ----
        if (not self.use_chain_break) or (not self.use_value_cancellation) or (is_instruct is None):
            for epi_idx in range(Gt.shape[0]):
                end_step_idx = int(epi_len[epi_idx]) - 1
                epi_r = Gt[epi_idx]
                if not terminate[epi_idx][end_step_idx]:
                    epi_r[end_step_idx] += self.gamma * boot_selected[epi_idx][end_step_idx]
                for idx in range(end_step_idx - 1, -1, -1):
                    epi_r[idx] = epi_r[idx] + self.gamma * epi_r[idx + 1]
            return Gt

        # ---- Chain-break mode ----
        for epi_idx in range(Gt.shape[0]):
            end_step_idx = int(epi_len[epi_idx]) - 1
            epi_r = Gt[epi_idx]

            T = end_step_idx + 1
            for t in range(end_step_idx + 1):
                if is_instruct[epi_idx][t].item():
                    T = t
                    break

            # Instruct segment [T, end_step_idx].
            if T <= end_step_idx:
                if not terminate[epi_idx][end_step_idx]:
                    epi_r[end_step_idx] += self.gamma * instruct_bootstrap[epi_idx][end_step_idx]
                for idx in range(end_step_idx - 1, T - 1, -1):
                    epi_r[idx] = epi_r[idx] + self.gamma * epi_r[idx + 1]

            # Benign segment [0, min(T-1, end_step_idx)].
            if T > 0:
                benign_end = min(T - 1, end_step_idx)
                if T <= end_step_idx:
                    # Chain break: bootstrap with V_{Psi}, episode continues.
                    epi_r[benign_end] += self.gamma * normal_bootstrap[epi_idx][benign_end]
                else:
                    if not terminate[epi_idx][benign_end]:
                        epi_r[benign_end] += self.gamma * normal_bootstrap[epi_idx][benign_end]
                for idx in range(benign_end - 1, -1, -1):
                    epi_r[idx] = epi_r[idx] + self.gamma * epi_r[idx + 1]

        return Gt
