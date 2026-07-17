import time
import numpy as np
import wandb
import os
import torch
import pdb
from datetime import datetime


from mujoco_maciac.cores.pg_based.mac_iac.memory import Memory_epi, Memory_rand
from mujoco_maciac.cores.pg_based.mac_iac.controller import MAC
from mujoco_maciac.cores.pg_based.mac_iac.envs_runner import EnvsRunner
from mujoco_maciac.cores.pg_based.mac_iac.learner import Learner
from mujoco_maciac.cores.pg_based.mac_iac.utils import Linear_Decay, save_train_data, save_test_data, save_checkpoint, load_checkpoint, save_policies
from mujoco_maciac.cores.pg_based.wandb_logging import init_wandb_run, build_eval_log_dict, clear_eval_buffers

class MacIAC(object):

    def __init__(self,
            env,
            env_terminate_step, 
            n_env, 
            alg,
            n_agent,                                                                                            
            total_epi, 
            gamma, 
            a_lr, 
            c_lr,                                                                                               
            c_train_iteration,                 
            eps_start, 
            eps_end, 
            eps_stable_at, 
            c_hys_start, 
            c_hys_end, 
            adv_hys_start, 
            adv_hys_end, 
            hys_stable_at, 
            critic_hys, 
            adv_hys, 
            etrpy_w_start, 
            etrpy_w_end, 
            etrpy_w_stable_at, 
            train_freq, 
            c_target_update_freq, 
            c_target_soft_update, 
            tau, 
            n_step_TD, 
            TD_lambda, 
            a_mlp_layer_size, 
            a_rnn_layer_size, 
            c_mlp_layer_size, 
            c_rnn_layer_size, 
            grad_clip_value, 
            grad_clip_norm, 
            obs_last_action, 
            eval_policy, 
            eval_freq, 
            eval_num_epi, 
            sample_epi, 
            trace_len, 
            seed, 
            run_id, 
            save_dir, 
            resume, 
            device, 
            *args, 
            **kwargs):

        self.total_epi = total_epi
        self.train_freq = train_freq
        self.eval_policy = eval_policy
        self.eval_freq = eval_freq
        self.eval_num_epi = eval_num_epi
        self.critic_hys = critic_hys                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
        self.adv_hys = adv_hys 
        self.sample_epi = sample_epi
        self.c_target_update_freq = c_target_update_freq
        self.c_target_soft_update = c_target_soft_update
        self.run_id = run_id
        self.save_dir = save_dir
        self.resume = resume
        self.alg = alg

        # collect params
        actor_params = {'a_mlp_layer_size': a_mlp_layer_size,
                        'a_rnn_layer_size': a_rnn_layer_size}

        critic_params = {'c_mlp_layer_size': c_mlp_layer_size,                                                              
                         'c_rnn_layer_size': c_rnn_layer_size}

        hyper_params = {'a_lr': a_lr,
                        'c_lr': c_lr,
                        'c_train_iteration': c_train_iteration,
                        'c_target_update_freq': c_target_update_freq,
                        'tau': tau,
                        'grad_clip_value': grad_clip_value,
                        'grad_clip_norm': grad_clip_norm,
                        'n_step_TD': n_step_TD,
                        'TD_lambda': TD_lambda,
                        'device': device}

        self.env = env
        # Toggle instruction flow via env var (INSTRUCTION_ENABLED=0 to disable)
        instr_enabled = os.environ.get("INSTRUCTION_ENABLED", "0") == "1"

        # Get instruction text from environment variable (defaults to environment-specific instructions)
        # Check for multiple instructions (space-separated) or single instruction
        instruction_env = os.environ.get("OVERCOOKED_INSTRUCTIONS", None) or os.environ.get("WAREHOUSE_INSTRUCTIONS", None)
        if instruction_env:
            # Prefer '||' delimiter (set by scripts); fallback to newlines; else treat as single instruction
            if '||' in instruction_env:
                instruction_text = [s for s in instruction_env.split('||') if s.strip()]
            elif '\n' in instruction_env:
                instruction_text = [s.strip() for s in instruction_env.splitlines() if s.strip()]
            else:
                instruction_text = [instruction_env.strip()] if instruction_env.strip() else None
        else:
            instruction_text = os.environ.get("OVERCOOKED_INSTRUCTION", None)

        # Determine number of instructions (still used for scheduling logic)
        if instr_enabled and instruction_text:
            if isinstance(instruction_text, list):
                n_instructions = len(instruction_text)
            else:
                n_instructions = 1
                instruction_text = [instruction_text]
        elif instr_enabled:
            # Default single instruction
            n_instructions = 1
        else:
            n_instructions = 0

        # create buffer - instruction_emb_dim = 8 for BERT projected embeddings
        # The BERT encoder projects 384-dim CLS token → 8 via instruction_projection
        # Kept small relative to obs_dim (37) so instruction doesn't dominate
        instruction_emb_dim = 8 if instr_enabled else 1
        if sample_epi:
            self.memory = Memory_epi(env.obs_size, env.n_action, obs_last_action, size=train_freq, instruction_emb_dim=instruction_emb_dim)
        else:
            self.memory = Memory_rand(trace_len, env.obs_size, env.n_action, obs_last_action, size=train_freq, instruction_emb_dim=instruction_emb_dim)

        # create controller - no n_instructions param needed, BERT is loaded inside
        self.controller = MAC(self.env, obs_last_action, **actor_params, **critic_params, device=device,
                              use_instructions=instr_enabled,
                              instruction_fusion='concat')
        # create parallel envs runner with instruction provider if enabled
        if instr_enabled:
            # Use environment-specific instruction(s) if provided, otherwise use default
            # Default varies by environment type
            if hasattr(env, 'n_objs'):  # Warehouse / OSD environment
                default_instruction = "fetch tool 0 first"
            elif hasattr(env, 'boxes'):  # Box Pushing environment
                default_instruction = "big_box_spot_0"
            else:  # Overcooked or other environments
                default_instruction = "get tomato"

            if instruction_text:
                # Pre-encode instruction texts via BERT using the actor's encode_instruction()
                # The controller has already loaded BERT, so we use agent 0's actor to encode
                instruction_embeddings = []
                instruction_texts = instruction_text
                for i, instr in enumerate(instruction_text):
                    with torch.no_grad():
                        emb = self.controller.agents[0].actor_net.encode_instruction(instr).detach()
                        # emb shape: (1, rnn_layer_size) — pre-encoded BERT embedding
                    instruction_embeddings.append(emb)
            else:
                # Use default instruction
                instruction_embeddings = []
                instruction_texts = [default_instruction]
                with torch.no_grad():
                    emb = self.controller.agents[0].actor_net.encode_instruction(default_instruction).detach()
                instruction_embeddings.append(emb)

            # Store instruction info for logging/debugging
            self.instruction_texts = instruction_texts
            self.instruction_embeddings = instruction_embeddings
            self.n_agent = n_agent
            self.n_env = n_env

            # Print instruction texts and embedding info
            print("\n" + "="*70)
            print("LOADED INSTRUCTIONS (BERT SENTENCE-TRANSFORMER):")
            for i, (text, emb) in enumerate(zip(self.instruction_texts, self.instruction_embeddings)):
                print(f"  Instruction {i}: '{text}' -> BERT embedding shape: {emb.shape}")
            print(f"Total instructions: {len(self.instruction_texts)}")
            print(f"Embedding dimension: {instruction_emb_dim} (projected from BERT 384-dim)")
            
            # All agents in an environment receive the SAME instruction,
            # randomly sampled per environment from the instruction pool.
            print(f"SHARED INSTRUCTION MODE:")
            print(f"  All {n_agent} agents in each environment receive the same instruction")
            print(f"  Instruction randomly sampled per environment from pool of {len(self.instruction_texts)}")
            self.instruction_mode = 'per_environment'
            self.env_instruction_indices = [np.random.randint(0, len(self.instruction_embeddings)) for _ in range(n_env)]
            for env_i in range(n_env):
                print(f"  Env {env_i}: '{self.instruction_texts[self.env_instruction_indices[env_i]]}'")
            
            print("="*70 + "\n")

            # Store instruction provider reference for later use (will be updated in learn method)
            self.instruction_provider = None
            self._current_episode = 0
            
            # Initialize stochastic instruction switching state
            self._instr_active = True  # Start with instructions enabled
            self._instr_idx = 0        # Start with first instruction
            self._last_schedule_update = 0
            self._last_instruction_idx = -1
        else:
            self.instruction_texts = []
            self.instruction_embeddings = []
            self.instruction_provider = None
            
        self.envs_runner = EnvsRunner(self.env, n_env, self.controller, self.memory, env_terminate_step, gamma, seed, obs_last_action, instruction_provider=None)
        # create learner - pass instruction texts for contrastive loss
        self.learner = Learner(self.env, self.controller, self.memory, gamma, **hyper_params,
                               instruction_texts=self.instruction_texts, contrastive_weight=0.1)
        # create epsilon calculator for implementing e-greedy exploration policy
        self.eps_call = Linear_Decay(eps_stable_at, eps_start, eps_end)
        # create hysteretic calculator for implementing hystgeritic value function updating
        self.c_hys_call = Linear_Decay(hys_stable_at, c_hys_start, c_hys_end)
        # create hysteretic calculator for implementing hystgeritic advantage esitimation
        self.adv_hys_call = Linear_Decay(hys_stable_at, adv_hys_start, adv_hys_end)
        # create entropy loss weight calculator
        self.etrpy_w_call = Linear_Decay(etrpy_w_stable_at, etrpy_w_start, etrpy_w_end)
        # record evaluation return
        self.eval_returns = []
        
        # Attach instruction metadata to config so it shows up in wandb's config
        # panel even though we group runs by save_dir (which doesn't contain
        # the instruction texts themselves).
        if instr_enabled and hasattr(self, 'instruction_texts'):
            hyper_params['instructions'] = self.instruction_texts
            hyper_params['instruction_encoding'] = 'bert_sentence_transformer'
            hyper_params['n_instructions'] = n_instructions

        # One wandb project per algorithm (self.alg == "MacIAC"), one run per
        # sweep config (name=self.save_dir). See wandb_logging.py for details.
        init_wandb_run(
            alg_name=self.alg,
            save_dir=self.save_dir,
            config=hyper_params,
            instr_enabled=instr_enabled,
            run_id=self.run_id,
        )

    def learn(self):
        epi_count = 0
        sum_avg_r = 0
        saved_3k_checkpoint = False  # Flag to ensure we only save once between 3000-4000
        if self.resume:
            epi_count, self.eval_returns = load_checkpoint(self.run_id, self.save_dir, self.controller, self.envs_runner)
            # Store current episode and update instruction provider after resuming from checkpoint
            self._current_episode = epi_count
            self.envs_runner.instruction_provider = self.create_instruction_provider(epi_count)

        while epi_count < self.total_epi:
            
            # Store current episode for instruction provider access
            self._current_episode = epi_count

            # Update instruction provider based on current episode count (instruction switching logic)
            old_provider = self.envs_runner.instruction_provider
            new_provider = self.create_instruction_provider(epi_count)
            self.envs_runner.instruction_provider = new_provider

            # Print when instructions are enabled/disabled 
            if self.run_id == 0 and epi_count % 500 == 0:
                if old_provider is None and new_provider is not None:
                    print(f"[{self.run_id}] Instructions enabled at episode {epi_count}")
                elif old_provider is not None and new_provider is None:
                    print(f"[{self.run_id}] Instructions disabled at episode {epi_count}")


            if self.eval_policy and epi_count % (self.eval_freq - (self.eval_freq % self.train_freq)) == 0:
                self.envs_runner.run(n_epis=self.eval_num_epi, test_mode=True)
                assert len(self.envs_runner.eval_returns) >= self.eval_num_epi, "Not evaluate enough episodes ..."
                self.eval_returns.append(np.mean(self.envs_runner.eval_returns[-self.eval_num_epi:]))
                self.envs_runner.eval_returns = []

                # Build the wandb log dict from whatever the envs_runner captured
                # this eval cycle. See wandb_logging.build_eval_log_dict for the
                # full set of metrics it computes when each buffer is present.
                instr_active = getattr(self, '_instr_active', False)
                log_dict = build_eval_log_dict(
                    epi_count=epi_count,
                    eval_return=self.eval_returns[-1],
                    envs_runner=self.envs_runner,
                    instruction_texts=getattr(self, 'instruction_texts', []),
                    encoder_agent=self.controller.agents[0] if getattr(self, 'instruction_texts', []) else None,
                    instr_active=instr_active,
                )

                # Console summary (compliance + completion split) using the
                # numbers already computed in log_dict so we don't recompute.
                compliance_msg = ""
                if 'Instruction_Compliance' in log_dict:
                    compliance_msg = f" | Compliance: {log_dict['Instruction_Compliance'] * 100:.1f}%"
                    per_inst_parts = [
                        f"'{k[len('Compliance/'):]}': {v*100:.1f}%"
                        for k, v in log_dict.items() if k.startswith('Compliance/')
                    ]
                    if per_inst_parts:
                        compliance_msg += f" ({', '.join(per_inst_parts)})"

                print(f"{[self.run_id]} Finished: {epi_count}/{self.total_epi} Evaluate learned policies with averaged returns {self.eval_returns[-1]:.2f}{compliance_msg} ...", flush=True)
                if self.run_id == 0 and (
                    'Completion_With_Instruction' in log_dict or 'Completion_Without_Instruction' in log_dict
                ):
                    msg_parts = []
                    if 'Completion_With_Instruction' in log_dict:
                        msg_parts.append(
                            f"with-inst done={log_dict['Completion_With_Instruction']*100:.1f}% "
                            f"trunc={log_dict.get('HorizonTrunc_With_Instruction', 0.0)*100:.1f}%"
                        )
                    if 'Completion_Without_Instruction' in log_dict:
                        msg_parts.append(
                            f"no-inst done={log_dict['Completion_Without_Instruction']*100:.1f}% "
                            f"trunc={log_dict.get('HorizonTrunc_Without_Instruction', 0.0)*100:.1f}%"
                        )
                    print(f"[{self.run_id}] Eval completion split: " + " | ".join(msg_parts), flush=True)

                sum_avg_r += self.eval_returns[-1]
                wandb.log(log_dict)
                clear_eval_buffers(self.envs_runner)
                
                # save the best policy
                if self.eval_returns[-1] == np.max(self.eval_returns):
                    save_policies(self.run_id, self.controller.agents, self.save_dir)
                
                # save policy once between episodes 100,000-120,000
                if 100000 <= epi_count < 120000 and not saved_3k_checkpoint:
                    save_policies(self.run_id, self.controller.agents, self.save_dir, suffix='_ep100k')
                    print(f"{[self.run_id]} Saved policy at episode {epi_count} (100k checkpoint)", flush=True)
                    saved_3k_checkpoint = True

            # No instruction epsilon to update
            
            # update eps
            eps = self.eps_call.get_value(epi_count)
            # update hys
            c_hys_value = self.c_hys_call.get_value(epi_count)
            adv_hys_value = self.adv_hys_call.get_value(epi_count)
            # update etrpy weight
            etrpy_w = self.etrpy_w_call.get_value(epi_count)
            # let envs run a certain number of episodes accourding to train_freq
            self.envs_runner.run(eps=eps, n_epis=self.train_freq)
            # perform hysteretic-ac update
            self.learner.train(eps, c_hys_value, adv_hys_value, etrpy_w, self.critic_hys, self.adv_hys)
            if not self.sample_epi:
                self.memory.buf.clear()

            epi_count += self.train_freq

            # update target net
            if self.c_target_soft_update:
                self.learner.update_critic_target_net(soft=True)
                self.learner.update_actor_target_net(soft=True)
            elif epi_count % self.c_target_update_freq == 0:
                self.learner.update_critic_target_net()
                self.learner.update_actor_target_net()
            
                
        

        ################################ saving in the end ###################################
        
        save_train_data(self.run_id, self.envs_runner.train_returns, self.save_dir)
        save_test_data(self.run_id, self.eval_returns, self.save_dir)
        save_checkpoint(self.run_id, 
                        epi_count, 
                        self.eval_returns, 
                        self.controller, 
                        self.envs_runner, 
                        self.save_dir)
        self.envs_runner.close()
        wandb.finish()
        

        print(f"{[self.run_id]} Finish entire training ... ", flush=True)

    def create_instruction_provider(self, current_episode):
        """Create instruction provider that alternates between instructions and no instructions"""
        # If instructions weren't enabled at init, always return None
        if not hasattr(self, 'instruction_embeddings') or len(self.instruction_embeddings) == 0:
            return None
            
        # Check switching mode from environment variable
        switch_mode = os.environ.get("INSTRUCTION_SWITCH_MODE", "stochastic")
        provided_prob = float(os.environ.get("INSTRUCTION_PROVIDED_PROB", "0.01"))

        if switch_mode == "fixed":
            # FIXED SCHEDULE: Switch every 5000 episodes
            if current_episode % 1000 == 0 or current_episode < 100:
                print(f"DEBUG: Using FIXED alternating instruction provider (5000 eps cycle)")
            
            def _fixed_instruction_provider(env_idx, step, agent_idx=None):
                current_episode = getattr(self, '_current_episode', 0)
                cycle_length = 10000  # 5000 with instruction + 5000 without
                position_in_cycle = current_episode % cycle_length
                
                if position_in_cycle < 5000:
                    # Randomly sample a fresh instruction each time the provider is called
                    inst_idx = np.random.randint(0, len(self.instruction_embeddings))
                    return (self.instruction_texts[inst_idx], self.instruction_embeddings[inst_idx])
                else:
                    # Instructions disabled
                    return None
            return _fixed_instruction_provider
                
        else:
            # STOCHASTIC SCHEDULE: per-step probability handled in envs_runner
            # Provider always returns instructions when called

            # Only print debug info every 1000 episodes or when creating provider
            if current_episode % 1000 == 0 or current_episode < 100:
                print(f"DEBUG create_instruction_provider: current_episode={current_episode}, mode=stochastic (per-step prob in envs_runner)")

            def _stochastic_instruction_provider(env_idx, step, agent_idx=None):
                # Randomly sample a fresh instruction each time the provider is called
                inst_idx = np.random.randint(0, len(self.instruction_embeddings))
                return (self.instruction_texts[inst_idx], self.instruction_embeddings[inst_idx])
            return _stochastic_instruction_provider