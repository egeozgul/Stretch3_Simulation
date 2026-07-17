import os
import time
import numpy as np
import wandb
import torch
from datetime import datetime

from macro_marl.cores.pg_based.mac_iaicc.memory import Memory_epi, Memory_rand  # Use mac_iaicc memory with instruction support
from macro_marl.cores.pg_based.mac_iaicc.envs_runner import EnvsRunner
from macro_marl.cores.pg_based.mac_niacc.utils import Linear_Decay, save_train_data, save_test_data, save_policies
from macro_marl.cores.pg_based.mac_iaicc.controller import MAC
from macro_marl.cores.pg_based.mac_iaicc import Learner_1
from macro_marl.cores.pg_based.mac_iaicc.utils import save_checkpoint, load_checkpoint
from macro_marl.cores.pg_based.wandb_logging import init_wandb_run, build_eval_log_dict, clear_eval_buffers

Learners = [Learner_1]

class MacIAICC(object):

    def __init__(self,
            env,
            env_terminate_step, 
            n_env, 
            n_agent, 
            l_mode, 
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
            use_instructions=False,
            instruction_fusion='concat',
            freeze_bert=True,
            instruction_provider=None,
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
        # `alg` isn't in the explicit signature; pull it from **kwargs so the
        # wandb project name ('MacIAICC') is set consistently with the other
        # algs. Fallback to the class name if the arg parser didn't pass it.
        self.alg = kwargs.get('alg', 'MacIAICC')

        # collect params
        actor_params = {'a_mlp_layer_size': a_mlp_layer_size,
                        'a_rnn_layer_size': a_rnn_layer_size}

        critic_params = {'c_mlp_layer_size': c_mlp_layer_size,
                         'c_rnn_layer_size': c_rnn_layer_size}
        
        # Instruction params
        instruction_params = {'use_instructions': use_instructions,
                              'instruction_fusion': instruction_fusion,
                              'freeze_bert': freeze_bert}

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
        
        # create buffer
        # instruction_emb_dim = 8 for BERT projected embeddings
        # The BERT encoder projects 384-dim CLS token -> 8 via instruction_projection
        # Kept small relative to obs_dim so instruction doesn't dominate
        instruction_emb_dim = 8 if instr_enabled else 1
        if self.sample_epi:
            self.memory = Memory_epi(env.obs_size, env.n_action, obs_last_action, size=train_freq, instruction_emb_dim=instruction_emb_dim)
        else:
            self.memory = Memory_rand(trace_len, env.obs_size, env.n_action, obs_last_action, size=train_freq, instruction_emb_dim=instruction_emb_dim)
        
        # Get instruction text from environment variable (defaults to environment-specific instructions)
        # Accept either OVERCOOKED_INSTRUCTIONS (legacy / Overcooked scripts)
        # or WAREHOUSE_INSTRUCTIONS (OSD / warehouse scripts) — same '||' format.
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
        
        # cretate controller with instruction support
        self.controller = MAC(self.env, obs_last_action, **actor_params, **critic_params, device=device,
                              use_instructions=instr_enabled,
                              instruction_fusion='concat',
                              freeze_bert=True)
        
        # Set up instruction provider if instructions are enabled
        instruction_provider = None
        if instr_enabled:
            # Use environment-specific instruction(s) if provided, otherwise use default
            if hasattr(env, 'n_objs'):  # Warehouse / OSD environment
                default_instruction = "fetch tool 0 first"
            elif hasattr(env, 'boxes'):  # Box Pushing environment
                default_instruction = "big_box_spot_0"
            else:  # Overcooked or other environments
                default_instruction = "get tomato"

            if instruction_text:
                # Handle multiple instructions by keeping them separate
                if isinstance(instruction_text, list):
                    instruction_embeddings = []
                    instruction_texts = instruction_text
                    for instr in instruction_text:
                        emb = self.controller.agents[0].actor_net.encode_instruction(instr).detach()
                        instruction_embeddings.append(emb)
                else:
                    instruction_embeddings = [self.controller.agents[0].actor_net.encode_instruction(instruction_text).detach()]
                    instruction_texts = [instruction_text]
            else:
                instruction_embeddings = [self.controller.agents[0].actor_net.encode_instruction(default_instruction).detach()]
                instruction_texts = [default_instruction]

            # Store instruction info for logging/debugging
            self.instruction_texts = instruction_texts
            self.instruction_embeddings = instruction_embeddings
            self.n_agent = n_agent
            self.n_env = n_env

            # Print instruction texts and embedding shapes
            print("\n" + "="*70)
            print("MAC-IAICC LOADED INSTRUCTIONS:")
            for i, (text, emb) in enumerate(zip(self.instruction_texts, self.instruction_embeddings)):
                emb_shape = emb.shape if hasattr(emb, 'shape') else 'scalar'
                print(f"  Instruction {i}: '{text}' -> embedding shape: {emb_shape}")
            print(f"Total instructions: {len(self.instruction_texts)}")
            
            # Determine instruction assignment mode
            if len(self.instruction_texts) == n_agent:
                print(f"PER-AGENT FIXED ASSIGNMENT MODE:")
                print(f"  Each of {n_agent} agents will get their own fixed instruction")
                for i in range(n_agent):
                    print(f"  Agent {i}: '{self.instruction_texts[i]}'")
                self.instruction_mode = 'fixed_per_agent'
            elif len(self.instruction_texts) > n_agent:
                print(f"RANDOM PER-AGENT SAMPLING MODE:")
                print(f"  Each agent independently samples from {len(self.instruction_texts)} instructions")
                self.instruction_mode = 'random_per_agent'
                self.agent_instruction_indices = [[np.random.randint(0, len(self.instruction_embeddings)) 
                                                   for _ in range(n_agent)] for _ in range(n_env)]
            else:
                print(f"PER-ENVIRONMENT MODE:")
                print(f"  Instructions will be randomly assigned to {n_env} environments")
                self.instruction_mode = 'per_environment'
                self.env_instruction_indices = [np.random.randint(0, len(self.instruction_embeddings)) for _ in range(n_env)]
            
            print("="*70 + "\n")

            # Create instruction provider function for EnvsRunner
            # Signature must match how envs_runner.py calls it: (env_idx, step_count, agent_idx=agent_idx)
            def instruction_provider_func(env_idx, step_count, agent_idx=0):
                if self.instruction_mode == 'fixed_per_agent':
                    return (self.instruction_texts[agent_idx], self.instruction_embeddings[agent_idx])
                elif self.instruction_mode == 'random_per_agent':
                    if step_count > 0 and step_count % 100 == 0:
                        self.agent_instruction_indices[env_idx][agent_idx] = np.random.randint(0, len(self.instruction_embeddings))
                    idx = self.agent_instruction_indices[env_idx][agent_idx]
                    return (self.instruction_texts[idx], self.instruction_embeddings[idx])
                else:  # per_environment
                    idx = self.env_instruction_indices[env_idx]
                    return (self.instruction_texts[idx], self.instruction_embeddings[idx])
            
            # Store instruction provider reference
            self.instruction_provider_func = instruction_provider_func
            self._current_episode = 0
            instruction_provider = instruction_provider_func
        else:
            self.instruction_texts = []
            self.instruction_embeddings = []
            self.instruction_provider_func = None
        
        # create parallel envs runner with instruction support
        self.envs_runner = EnvsRunner(self.env, n_env, self.controller, self.memory, env_terminate_step, gamma, seed, obs_last_action, instruction_provider=None)
        # create learner
        self.learner = Learners[l_mode](self.env, self.controller, self.memory, gamma, obs_last_action,
                                         instruction_texts=getattr(self, 'instruction_texts', []),
                                         contrastive_weight=0.1,
                                         **hyper_params)
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
        
        # Attach instruction metadata to config so it's searchable in wandb's
        # UI next to the other hyperparameters.
        if instr_enabled and hasattr(self, 'instruction_texts'):
            hyper_params['instructions'] = self.instruction_texts
            hyper_params['instruction_encoding'] = 'bert_sentence_transformer'
            hyper_params['n_instructions'] = len(self.instruction_texts)

        # project=self.alg ('MacIAICC') groups all this algorithm's sweep runs
        # under one dashboard; name=self.save_dir keeps each hparam combo
        # separately labeled.
        init_wandb_run(
            alg_name=self.alg,
            save_dir=self.save_dir,
            config=hyper_params,
            instr_enabled=instr_enabled,
            run_id=self.run_id,
        )

    def learn(self):
        epi_count = 0
        if self.resume:
            epi_count, self.eval_returns = load_checkpoint(self.run_id, self.save_dir, self.controller, self.envs_runner)
            self._current_episode = epi_count
            self.envs_runner.instruction_provider = self.create_instruction_provider(epi_count)

        while epi_count < self.total_epi:

            # Store current episode for instruction provider access
            self._current_episode = epi_count

            # Update instruction provider based on current episode count
            old_provider = self.envs_runner.instruction_provider
            new_provider = self.create_instruction_provider(epi_count)
            self.envs_runner.instruction_provider = new_provider

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

                # MAC-IAICC uses .agents[0]'s BERT encoder for embedding
                # cosine/L2 metrics. _instr_active doesn't exist on this class
                # (switching is per-step in envs_runner), default to True.
                log_dict = build_eval_log_dict(
                    epi_count=epi_count,
                    eval_return=self.eval_returns[-1],
                    envs_runner=self.envs_runner,
                    instruction_texts=getattr(self, 'instruction_texts', []),
                    encoder_agent=self.controller.agents[0] if getattr(self, 'instruction_texts', []) else None,
                    instr_active=True,
                )

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
                wandb.log(log_dict)
                clear_eval_buffers(self.envs_runner)

                # save the best policy
                if self.eval_returns[-1] == np.max(self.eval_returns):
                    save_policies(self.run_id, self.controller.agents, self.save_dir)

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
        # save_train_data(self.run_id, self.envs_runner.train_returns, self.save_dir)
        # save_test_data(self.run_id, self.eval_returns, self.save_dir)
        # save_checkpoint(self.run_id, epi_count, self.eval_returns, self.controller, self.envs_runner, self.save_dir)
        self.envs_runner.close()
        wandb.finish()

        print(f"{[self.run_id]} Finish entire training ... ", flush=True)

    def create_instruction_provider(self, current_episode):
        """Create instruction provider that alternates between instructions and no instructions.
        Matches mac_iac.py behavior: supports 'fixed' (5000-episode cycles) and 'stochastic' 
        (per-step probability handled in envs_runner) modes via INSTRUCTION_SWITCH_MODE env var.
        """
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
