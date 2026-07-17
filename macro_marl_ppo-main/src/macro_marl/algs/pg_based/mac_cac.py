import time
import os
import numpy as np
import wandb
import torch

from macro_marl.cores.pg_based.mac_cac.memory import Memory_epi, Memory_rand
from macro_marl.cores.pg_based.mac_cac.controller import MAC
from macro_marl.cores.pg_based.mac_cac.envs_runner import EnvsRunner
from macro_marl.cores.pg_based.mac_cac.learner import Learner
from macro_marl.cores.pg_based.mac_cac.utils import Linear_Decay, save_train_data, save_test_data, save_checkpoint, load_checkpoint, save_policy
from macro_marl.cores.pg_based.wandb_logging import init_wandb_run, build_eval_log_dict, clear_eval_buffers

class MacCAC(object):

    def __init__(self,
            env,
            env_terminate_step,
            alg, 
            n_env, 
            n_agent, 
            seed, 
            run_id, 
            save_dir, 
            resume, 
            device,
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
            *args, 
            **kwargs):

        # Disable HF tokenizers parallelism to avoid fork warnings/deadlocks
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
        # create buffer
        instruction_emb_dim = a_rnn_layer_size
        if sample_epi:
            self.memory = Memory_epi(env.obs_size, env.n_action, obs_last_action, size=train_freq, instruction_emb_dim=instruction_emb_dim)
        else:
            self.memory = Memory_rand(trace_len, env.obs_size, env.n_action, obs_last_action, size=train_freq, instruction_emb_dim=instruction_emb_dim)
        instr_enabled = os.environ.get("INSTRUCTION_ENABLED", "1") == "1"

        # Accept either OVERCOOKED_INSTRUCTIONS (legacy / Overcooked scripts)
        # or WAREHOUSE_INSTRUCTIONS (OSD / warehouse scripts) — same '||' format.
        instruction_env = os.environ.get("OVERCOOKED_INSTRUCTIONS", None) or os.environ.get("WAREHOUSE_INSTRUCTIONS", None)
        if instruction_env:
            # Delimiter priority matches mac_iac / acac / mac_iaicc_V so a single
            # shared env-var string works across every pg_based algorithm:
            #   1) '||'  (preferred; set by all current shell scripts)
            #   2) ';'   (legacy mac_cac format)
            #   3) '\n'  (newline-separated)
            #   4) treat as a single instruction
            if '||' in instruction_env:
                instructions_list = [s.strip() for s in instruction_env.split('||') if s.strip()]
            elif ';' in instruction_env:
                instructions_list = [s.strip() for s in instruction_env.split(';') if s.strip()]
            elif '\n' in instruction_env:
                instructions_list = [s.strip() for s in instruction_env.splitlines() if s.strip()]
            else:
                instructions_list = [instruction_env.strip()] if instruction_env.strip() else []

            if len(instructions_list) > 1:
                instruction_text = instructions_list
            elif len(instructions_list) == 1:
                instruction_text = instructions_list[0]
            else:
                instruction_text = None
        else:
            instruction_text = os.environ.get("OVERCOOKED_INSTRUCTION", None)

        # create controller (instruction conditioned if enabled)
        self.controller = MAC(self.env, 
                              obs_last_action, 
                              **actor_params, 
                              **critic_params, 
                              device=device,
                              use_instructions=instr_enabled,
                              instruction_fusion='concat',
                              freeze_bert=True)

        if instr_enabled:
            # Use environment-specific instruction(s) if provided, otherwise use default
            if hasattr(env, 'n_objs'):  # Warehouse / OSD environment
                default_instruction = "fetch tool 0 first"
            elif hasattr(env, 'boxes'):  # Box Pushing environment
                default_instruction = "big_box_spot_0"
            else:  # Overcooked or other environments
                default_instruction = "push big box"

            if instruction_text:
                # Handle multiple instructions by encoding each separately
                if isinstance(instruction_text, list):
                    # Multiple instructions - encode each separately and cycle through them
                    instruction_embeddings = []
                    instruction_texts = []
                    for instr in instruction_text:
                        emb = self.controller.agent.actor_net.encode_instruction(instr).detach()
                        instruction_embeddings.append(emb)
                        instruction_texts.append(instr)

                    # Store all instruction embeddings and texts
                    self.instruction_embeddings = instruction_embeddings
                    self.instruction_texts = instruction_texts
                    # For backward compatibility, also set fixed_instruction_emb to the first one
                    self.fixed_instruction_emb = instruction_embeddings[0]
                    fixed_instruction_emb = instruction_embeddings[0]
                    self.instruction_text = instruction_texts[0]
                else:
                    # Single instruction
                    fixed_instruction_emb = self.controller.agent.actor_net.encode_instruction(instruction_text).detach()
                    self.instruction_text = instruction_text
                    self.fixed_instruction_emb = fixed_instruction_emb
                    self.instruction_embeddings = [fixed_instruction_emb]
                    self.instruction_texts = [instruction_text]
            else:
                # Use default instruction
                fixed_instruction_emb = self.controller.agent.actor_net.encode_instruction(default_instruction).detach()
                self.instruction_text = default_instruction
                self.fixed_instruction_emb = fixed_instruction_emb
                self.instruction_embeddings = [fixed_instruction_emb]
                self.instruction_texts = [default_instruction]

                # Store instruction text on controller for envs_runner access
                self.controller.instruction_text = self.instruction_text
        else:
            # Instructions disabled
            self.controller.instruction_text = None
            self.instruction_text = None
            self.fixed_instruction_emb = None
            self.instruction_embeddings = []
            self.instruction_texts = []

        # Store instruction provider reference for later use (will be updated in learn method)
        self.instruction_provider = None
        self._last_instruction_idx = -1
        
        # Initialize stochastic instruction switching state
        self._instr_active = True  # Start with instructions enabled
        self._instr_idx = 0        # Start with first instruction
        self._last_schedule_update = 0
        
        # Debug: print what instruction attributes were set (only for run_id 0)
        if self.run_id == 0:
            print(f"DEBUG init: instruction_text='{self.instruction_text}', fixed_instruction_emb={self.fixed_instruction_emb is not None}, instruction_embeddings={len(getattr(self, 'instruction_embeddings', []))}")

        # create parallel envs runner
        self.envs_runner = EnvsRunner(self.env, n_env, self.controller, self.memory, env_terminate_step, gamma, seed, obs_last_action, instruction_provider=None)
        # create learner
        self.learner = Learner(self.env, self.controller, self.memory, gamma, **hyper_params)

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
        # Attach instruction metadata to the config dict so it's discoverable
        # in the wandb UI even though we group/name runs by save_dir.
        if instr_enabled and self.instruction_texts:
            hyper_params['instructions'] = self.instruction_texts
            hyper_params['instruction_encoding'] = 'bert_sentence_transformer'
            hyper_params['n_instructions'] = len(self.instruction_texts)

        # project=self.alg ('MacCAC') groups all of MacCAC's runs under one
        # dashboard; name=self.save_dir keeps each sweep config distinguishable.
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

            # Update instruction provider based on current episode count
            old_provider = self.envs_runner.instruction_provider
            new_provider = self.create_instruction_provider(epi_count)
            self.envs_runner.instruction_provider = new_provider

            # Debug: print instruction provider status (every 100 episodes)
            if epi_count % 100 == 0:
                print(f"DEBUG: Episode {epi_count}: fixed_instruction_emb={self.fixed_instruction_emb is not None}, instruction_embeddings={len(getattr(self, 'instruction_embeddings', []))}")

            # Print when instructions are enabled/disabled
            if (old_provider is None and new_provider is not None):
                print(f"[{self.run_id}] Instructions enabled at episode {epi_count}")
            elif (old_provider is not None and new_provider is None):
                print(f"[{self.run_id}] Instructions disabled at episode {epi_count}")
            elif (new_provider is not None and
                  hasattr(self, 'instruction_texts') and self.instruction_texts and
                  len(self.instruction_texts) > 1):
                # Show current instruction being used (only for multiple instructions)
                instruction_idx = (epi_count // 5000) % len(self.instruction_texts)
                current_instruction = self.instruction_texts[instruction_idx]
                if self.instruction_text != current_instruction:
                    print(f"[{self.run_id}] Switched to instruction: '{current_instruction}' at episode {epi_count}")
                    self.instruction_text = current_instruction
                    self.controller.instruction_text = current_instruction

            if self.eval_policy and epi_count % (self.eval_freq - (self.eval_freq % self.train_freq)) == 0:
                self.envs_runner.run(n_epis=self.eval_num_epi, test_mode=True)
                assert len(self.envs_runner.eval_returns) >= self.eval_num_epi, "Not evaluate enough episodes ..."
                self.eval_returns.append(np.mean(self.envs_runner.eval_returns[-self.eval_num_epi:]))
                self.envs_runner.eval_returns = []

                # MacCAC is a single-agent-of-agents controller: controller.agent
                # (not .agents). Still expose its actor for BERT embedding
                # cosine/L2 metrics when instructions are enabled.
                instr_active = getattr(self, '_instr_active', False)
                encoder_agent = getattr(self.controller, 'agent', None) if self.instruction_texts else None
                log_dict = build_eval_log_dict(
                    epi_count=epi_count,
                    eval_return=self.eval_returns[-1],
                    envs_runner=self.envs_runner,
                    instruction_texts=self.instruction_texts,
                    encoder_agent=encoder_agent,
                    instr_active=instr_active,
                )

                # Console summary derived from the same numbers logged to wandb.
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
                sum_avg_r += self.eval_returns[-1]
                wandb.log(log_dict)
                clear_eval_buffers(self.envs_runner)

                # save the best policy
                if self.eval_returns[-1] == np.max(self.eval_returns):
                    save_policy(self.run_id, self.controller.agent, self.save_dir)
                
                # save policy once between episodes 3000-4000
                if 3000 <= epi_count < 4000 and not saved_3k_checkpoint:
                    save_policy(self.run_id, self.controller.agent, self.save_dir, suffix='_ep3000-4000')
                    print(f"{[self.run_id]} Saved policy at episode {epi_count} (3k checkpoint)", flush=True)
                    saved_3k_checkpoint = True

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
        save_checkpoint(self.run_id, epi_count, self.eval_returns, self.controller, self.envs_runner, self.save_dir)
        self.envs_runner.close()
        wandb.finish()

        print(f"{[self.run_id]} Finish entire training ... ", flush=True)

    def create_instruction_provider(self, current_episode):
        """Create instruction provider that alternates between instructions and no instructions"""
        # Check switching mode from environment variable
        switch_mode = os.environ.get("INSTRUCTION_SWITCH_MODE", "stochastic")
        if switch_mode == "fixed":
            # FIXED SCHEDULE: Switch every 5000 episodes
            if hasattr(self, 'instruction_embeddings') and self.instruction_embeddings and len(self.instruction_embeddings) > 0:
                if current_episode % 1000 == 0 or current_episode < 100:
                    print(f"DEBUG: Using FIXED alternating instruction provider (5000 eps cycle)")
                
                def instruction_provider(env_idx, step):
                    current_episode = getattr(self, '_current_episode', 0)
                    cycle_length = 10000  # 5000 with instruction + 5000 without
                    position_in_cycle = current_episode % cycle_length
                    
                    if position_in_cycle < 5000:
                        # Randomly select instruction with uniform probability
                        if len(self.instruction_texts) >= 2:
                            # Uniform random selection: 1/n probability for each instruction
                            instruction_idx = np.random.choice(len(self.instruction_texts))
                            selected_instruction = self.instruction_texts[instruction_idx]
                        else:
                            # Use original deterministic selection for edge cases
                            instruction_idx = (current_episode // cycle_length) % len(self.instruction_embeddings)
                            selected_instruction = self.instruction_texts[instruction_idx]
                        
                        if not hasattr(self, '_last_instruction_idx') or self._last_instruction_idx != instruction_idx:
                            print(f"INFO: Switched to instruction {instruction_idx}: '{selected_instruction}' at episode {current_episode}")
                            self._last_instruction_idx = instruction_idx
                            self._last_was_none = False
                        
                        return (self.instruction_embeddings[instruction_idx], selected_instruction)
                    else:
                        if not hasattr(self, '_last_was_none') or not self._last_was_none:
                            print(f"INFO: Switched to NO INSTRUCTION (baseline) at episode {current_episode}")
                            self._last_was_none = True
                            self._last_instruction_idx = -1
                        return None
                return instruction_provider
                
            elif self.fixed_instruction_emb is not None:
                print(f"DEBUG: Using FIXED alternating fixed instruction provider")
                def instruction_provider(env_idx, step):
                    current_episode = getattr(self, '_current_episode', 0)
                    cycle_length = 10000
                    position_in_cycle = current_episode % cycle_length
                    
                    if position_in_cycle < 5000:
                        return (self.fixed_instruction_emb, self.instruction_text)
                    else:
                        return None
                return instruction_provider
                
        else:
            # STOCHASTIC SCHEDULE: 1% chance to switch at every episode
            # Initialize state if not present
            if not hasattr(self, '_instr_active'):
                self._instr_active = True
                self._instr_idx = 0
                self._last_schedule_update = 0
                
            # Update stochastic schedule based on elapsed episodes
            episodes_passed = current_episode - getattr(self, '_last_schedule_update', 0)
            if episodes_passed > 0:
                # Simulate 1% switching chance for each passed episode
                for _ in range(int(episodes_passed)):
                    if np.random.random() < 0.1:
                        # Switch mode
                        self._instr_active = not self._instr_active
                        if self._instr_active:
                            # If switched TO active, increment instruction index
                            if hasattr(self, 'instruction_embeddings') and self.instruction_embeddings:
                                self._instr_idx = (self._instr_idx + 1) % len(self.instruction_embeddings)
                            else:
                                self._instr_idx = 0
                
                self._last_schedule_update = current_episode

            # Only print debug info every 1000 episodes or when creating provider
            if current_episode % 1000 == 0 or current_episode < 100:
                print(f"DEBUG create_instruction_provider: current_episode={current_episode}, fixed_instruction_emb={self.fixed_instruction_emb is not None}, instruction_embeddings_len={len(getattr(self, 'instruction_embeddings', []))}")

            if hasattr(self, 'instruction_embeddings') and self.instruction_embeddings and len(self.instruction_embeddings) > 0:
                if current_episode % 1000 == 0 or current_episode < 100:
                    print(f"DEBUG: Using STOCHASTIC instruction provider with {len(self.instruction_embeddings)} instructions")
                    
                def instruction_provider(env_idx, step):
                    if self._instr_active:
                        # Randomly select instruction with uniform probability
                        if len(self.instruction_texts) >= 2:
                            # Uniform random selection: 1/n probability for each instruction
                            idx = np.random.choice(len(self.instruction_texts))
                            selected_instruction = self.instruction_texts[idx]
                        else:
                            # Use original deterministic selection for edge cases
                            idx = self._instr_idx % len(self.instruction_embeddings)
                            selected_instruction = self.instruction_texts[idx]
                        
                        # Only print debug info when instruction changes
                        if not hasattr(self, '_last_instruction_idx') or self._last_instruction_idx != idx:
                            print(f"INFO: Switched to instruction {idx}: '{selected_instruction}' at episode {getattr(self, '_current_episode', 0)}")
                            self._last_instruction_idx = idx
                            self._last_was_none = False
                            
                        return (self.instruction_embeddings[idx], selected_instruction)
                    else:
                        if not hasattr(self, '_last_was_none') or not self._last_was_none:
                            print(f"INFO: Switched to NO INSTRUCTION at episode {getattr(self, '_current_episode', 0)}")
                            self._last_was_none = True
                            self._last_instruction_idx = -1
                        return None
                return instruction_provider
                
            elif self.fixed_instruction_emb is not None:
                # Fallback: stochastic switching between single instruction and no instruction
                print(f"DEBUG: Using STOCHASTIC fixed instruction provider")
                def instruction_provider(env_idx, step):
                    if self._instr_active:
                        return (self.fixed_instruction_emb, self.instruction_text)
                    else:
                        return None
                return instruction_provider
        
        # Default fallback if no instructions available
        print(f"DEBUG: No instructions available - returning None provider")
        def instruction_provider(env_idx, step):
            return None
        return instruction_provider
