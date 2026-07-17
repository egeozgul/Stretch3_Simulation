import time
import numpy as np
import os
import torch

from macro_marl.cores.pg_based.acac.memory import Memory_epi
from macro_marl.cores.pg_based.acac.controller import MAC
from macro_marl.cores.pg_based.acac.envs_runner import EnvsRunner
from macro_marl.cores.pg_based.acac.learner_acac import Learner_ACAC
from macro_marl.cores.pg_based.acac.learner_acac_vanilla import Learner_ACAC_Vanilla
from macro_marl.cores.pg_based.acac.learner_acac_micro_gae import Learner_ACAC_Micro_GAE
from macro_marl.cores.pg_based.acac.utils import Linear_Decay, save_train_data, save_test_data, save_policies
from macro_marl.cores.pg_based.acac.ckpt_utils import save_checkpoint_cent, load_checkpoint_cent
from macro_marl.cores.pg_based.wandb_logging import init_wandb_run, build_eval_log_dict, clear_eval_buffers

import wandb

class ACAC(object):

    def __init__(self,
            env,
            env_terminate_step,
            n_env,
            n_agent,
            total_epi,
            gamma,
            ppo_clip_value,
            ppo_epochs,
            a_lr,
            c_lr,
            eps_start,
            eps_end,
            eps_stable_at,
            etrpy_w_start,
            etrpy_w_end,
            etrpy_w_stable_at,
            train_freq,
            tau,
            TD_lambda,
            tracking,
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
            alg,
            scenario,
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
        self.run_id = run_id
        self.save_dir = save_dir
        self.resume = resume
        self.alg = alg
        self.scenario = scenario
        self.tracking = tracking
        self.seed = seed

        c_train_iteration = kwargs.get('c_train_iteration', 1)
        c_target_update_freq = kwargs.get('c_target_update_freq', 50)
        c_target_soft_update = kwargs.get('c_target_soft_update', False)
        n_minibatch = kwargs.get('n_minibatch', 8)
        n_train_repeat = kwargs.get('n_train_repeat', ppo_epochs)
        GAE_lambda = kwargs.get('GAE_lambda', 0.95)
        vf_coef = kwargs.get('vf_coef', 0.5)
        clip_ratio = kwargs.get('clip_ratio', ppo_clip_value)

        critic_hys = kwargs.get('critic_hys', False)
        adv_hys = kwargs.get('adv_hys', False)
        c_hys_start = kwargs.get('c_hys_start', 1.0)
        c_hys_end = kwargs.get('c_hys_end', 1.0)
        adv_hys_start = kwargs.get('adv_hys_start', 1.0)
        adv_hys_end = kwargs.get('adv_hys_end', 1.0)
        hys_stable_at = kwargs.get('hys_stable_at', eps_stable_at)

        controller_opts = {
            'time_emb': False,
            'time_emb_actor': False,
            'init_critic': kwargs.get('init_critic', True),
            'share_encoder': kwargs.get('share_encoder', False),
            'use_attention': kwargs.get('use_attention', True),
            'cc_n_head': kwargs.get('cc_n_head', 2),
            'enc_n_head': kwargs.get('enc_n_head', 4),
            'n_layer': kwargs.get('n_layer', 1),
            'value_head': kwargs.get('value_head', 'concat'),
            'time_emb_alg': kwargs.get('time_emb_alg', 'sinu'),
            'time_emb_dim': kwargs.get('time_emb_dim', 4),
            'max_timestep': kwargs.get('max_timestep', env_terminate_step),
            'use_actor_ln': kwargs.get('use_actor_ln', False),
            'duplicate': kwargs.get('duplicate', False),
            'use_popart': kwargs.get('use_popart', False),
        }

        actor_params = {'a_mlp_layer_size': a_mlp_layer_size,
                        'a_rnn_layer_size': a_rnn_layer_size}

        critic_params = {'c_mlp_layer_size': c_mlp_layer_size,
                         'c_rnn_layer_size': c_rnn_layer_size}

        hyper_params = {
            'a_lr': a_lr,
            'c_lr': c_lr,
            'c_mlp_layer_size': c_mlp_layer_size,
            'c_rnn_layer_size': c_rnn_layer_size,
            'c_train_iteration': c_train_iteration,
            'c_target_update_freq': c_target_update_freq,
            'n_train_repeat': n_train_repeat,
            'n_minibatch': n_minibatch,
            'tau': tau,
            'grad_clip_value': grad_clip_value,
            'grad_clip_norm': grad_clip_norm,
            'n_step_TD': kwargs.get('n_step_TD', 0),
            'TD_lambda': TD_lambda,
            'GAE_lambda': GAE_lambda,
            'device': device,
            'clip_ratio': clip_ratio,
            'vf_coef': vf_coef,
        }

        self.env = env
        self.n_agent = n_agent
        self.n_env = n_env

        # ---- Instruction configuration (env-var driven, same as mac_iac) ----
        instr_enabled = os.environ.get("INSTRUCTION_ENABLED", "0") == "1"
        # Accept either OVERCOOKED_INSTRUCTIONS (legacy / Overcooked scripts)
        # or WAREHOUSE_INSTRUCTIONS (OSD / warehouse scripts) — same '||' format.
        instruction_env = os.environ.get("OVERCOOKED_INSTRUCTIONS", None) or os.environ.get("WAREHOUSE_INSTRUCTIONS", None)
        if instruction_env:
            if '||' in instruction_env:
                instruction_text = [s for s in instruction_env.split('||') if s.strip()]
            elif '\n' in instruction_env:
                instruction_text = [s.strip() for s in instruction_env.splitlines() if s.strip()]
            else:
                instruction_text = [instruction_env.strip()] if instruction_env.strip() else None
        else:
            instruction_text = os.environ.get("OVERCOOKED_INSTRUCTION", None)

        if instr_enabled and instruction_text:
            if isinstance(instruction_text, list):
                n_instructions = len(instruction_text)
            else:
                n_instructions = 1
                instruction_text = [instruction_text]
        elif instr_enabled:
            n_instructions = 1
        else:
            n_instructions = 0

        instruction_emb_dim = max(n_instructions, 1)

        self.memory = Memory_epi(env.obs_size, env.n_action, obs_last_action, size=train_freq, max_len=trace_len, instruction_emb_dim=instruction_emb_dim)

        controller_opts['use_instructions'] = instr_enabled
        controller_opts['n_instructions'] = n_instructions
        self.controller = MAC(self.env, obs_last_action, **actor_params, **critic_params, device=device, **controller_opts)

        # ---- Build instruction provider state ----
        if instr_enabled:
            if hasattr(env, 'n_objs'):  # Warehouse / OSD environment
                default_instruction = "fetch tool 0 first"
            elif hasattr(env, 'boxes'):  # Box Pushing environment
                default_instruction = "big_box_spot_0"
            else:  # Overcooked or other environments
                default_instruction = "get tomato"

            if instruction_text:
                instruction_embeddings = []
                instruction_texts = instruction_text
                for i, instr in enumerate(instruction_text):
                    one_hot = torch.zeros(1, n_instructions)
                    one_hot[0, i] = 1.0
                    instruction_embeddings.append(one_hot)
            else:
                instruction_embeddings = [torch.zeros(1, n_instructions)]
                instruction_embeddings[0][0, 0] = 1.0
                instruction_texts = [default_instruction]

            self.instruction_texts = instruction_texts
            self.instruction_embeddings = instruction_embeddings

            print("\n" + "="*70)
            print("ACAC - LOADED INSTRUCTIONS (ONE-HOT ENCODING):")
            for i, (text, emb) in enumerate(zip(self.instruction_texts, self.instruction_embeddings)):
                print(f"  Instruction {i}: '{text}' -> one-hot index: {i}, vector dim: {n_instructions}")
            print(f"Total instructions: {len(self.instruction_texts)}")
            print(f"One-hot dimension: {n_instructions}")
            print(f"Switch mode: {os.environ.get('INSTRUCTION_SWITCH_MODE', 'stochastic')}")
            print(f"Instruction provided prob: {os.environ.get('INSTRUCTION_PROVIDED_PROB', '0.01')}")

            if len(self.instruction_texts) == n_agent:
                self.instruction_mode = 'fixed_per_agent'
                print(f"PER-AGENT FIXED ASSIGNMENT MODE:")
                for i in range(n_agent):
                    print(f"  Agent {i}: '{self.instruction_texts[i]}'")
            elif len(self.instruction_texts) > n_agent:
                self.instruction_mode = 'random_per_agent'
                print(f"RANDOM PER-AGENT SAMPLING MODE: {len(self.instruction_texts)} instructions")
                self.agent_instruction_indices = [[np.random.randint(0, len(self.instruction_embeddings))
                                                   for _ in range(n_agent)] for _ in range(n_env)]
            else:
                self.instruction_mode = 'per_environment'
                print(f"PER-ENVIRONMENT MODE")
                self.env_instruction_indices = [np.random.randint(0, len(self.instruction_embeddings)) for _ in range(n_env)]
            print("="*70 + "\n")

            if self.instruction_mode == 'random_per_agent':
                self.global_step_count = [0] * n_env
                self.last_resample_step = [0] * n_env
                self.resample_interval = 100

            self.instruction_provider = None
            self._current_episode = 0
            self._instr_active = True
            self._instr_idx = 0
            self._last_schedule_update = 0
            self._last_instruction_idx = -1
        else:
            self.instruction_texts = []
            self.instruction_embeddings = []
            self.instruction_provider = None

        self.envs_runner = EnvsRunner(self.env, n_env, self.controller, self.memory, env_terminate_step, gamma, seed, obs_last_action, trace_len=trace_len, instruction_provider=None)
        self.learner = Learner_ACAC(self.env, self.controller, self.memory, gamma, **hyper_params)

        self.eps_call = Linear_Decay(eps_stable_at, eps_start, eps_end)
        self.etrpy_w_call = Linear_Decay(etrpy_w_stable_at, etrpy_w_start, etrpy_w_end)
        self.c_hys_call = Linear_Decay(hys_stable_at, c_hys_start, c_hys_end)
        self.adv_hys_call = Linear_Decay(hys_stable_at, adv_hys_start, adv_hys_end)

        self.c_target_update_freq = c_target_update_freq
        self.c_target_soft_update = c_target_soft_update
        self.critic_hys = critic_hys
        self.adv_hys = adv_hys

        self.eval_returns = []

        if self.tracking:
            # Attach instruction metadata to config so it's visible in the wandb
            # UI alongside the other hyperparameters.
            if instr_enabled and self.instruction_texts:
                hyper_params['instructions'] = self.instruction_texts
                hyper_params['instruction_encoding'] = 'one_hot'
                hyper_params['n_instructions'] = len(self.instruction_texts)
            # project=self.alg ('ACAC'), name=self.save_dir: all of ACAC's sweep
            # runs end up under one dashboard, with each hparam combo as a
            # separately-named run.
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
            epi_count, self.eval_returns = load_checkpoint_cent(self.run_id, self.save_dir, self.controller, self.learner, self.envs_runner)
            self._current_episode = epi_count
            if hasattr(self, 'instruction_embeddings') and len(self.instruction_embeddings) > 0:
                self.envs_runner.instruction_provider = self.create_instruction_provider(epi_count)

        while epi_count < self.total_epi:
            # Update instruction provider
            self._current_episode = epi_count
            if hasattr(self, 'instruction_embeddings') and len(self.instruction_embeddings) > 0:
                new_provider = self.create_instruction_provider(epi_count)
                self.envs_runner.instruction_provider = new_provider

            if self.eval_policy and epi_count % (self.eval_freq - (self.eval_freq % self.train_freq)) == 0:
                self.envs_runner.run(n_epis=self.eval_num_epi, test_mode=True)
                assert len(self.envs_runner.eval_returns) >= self.eval_num_epi, "Did Not Evaluate Sufficient Episodes ..."
                self.eval_returns.append(np.mean(self.envs_runner.eval_returns[-self.eval_num_epi:]))
                self.envs_runner.eval_returns = []

                # ACAC uses one-hot instruction encodings (no BERT), so we pass
                # encoder_agent=None to skip the embedding cosine/L2 metrics
                # (they'd be 0 / trivially orthogonal anyway for one-hot).
                instr_active = getattr(self, '_instr_active', True)
                log_dict = build_eval_log_dict(
                    epi_count=epi_count,
                    eval_return=self.eval_returns[-1],
                    envs_runner=self.envs_runner,
                    instruction_texts=self.instruction_texts,
                    encoder_agent=None,
                    instr_active=instr_active,
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

                if self.tracking:
                    wandb.log(log_dict)
                clear_eval_buffers(self.envs_runner)

                if self.eval_returns[-1] == np.max(self.eval_returns):
                    save_policies(self.run_id, self.controller.agents, self.save_dir)

            eps = self.eps_call.get_value(epi_count)
            etrpy_w = self.etrpy_w_call.get_value(epi_count)
            c_hys_value = self.c_hys_call.get_value(epi_count)
            adv_hys_value = self.adv_hys_call.get_value(epi_count)

            self.envs_runner.run(eps=eps, n_epis=self.train_freq)

            self.learner.train(eps, c_hys_value, adv_hys_value, etrpy_w, critic_hys=self.critic_hys, adv_hys=self.adv_hys)
            self.memory.buf.clear()

            epi_count += self.train_freq

            if self.c_target_soft_update:
                self.learner.update_critic_target_net(soft=True)
                self.learner.update_actor_target_net(soft=True)
            elif self.c_target_update_freq and epi_count % self.c_target_update_freq == 0:
                self.learner.update_critic_target_net()
                self.learner.update_actor_target_net()

        save_train_data(self.run_id, self.envs_runner.train_returns, self.save_dir)
        save_test_data(self.run_id, self.eval_returns, self.save_dir)
        save_checkpoint_cent(self.run_id,
                             epi_count,
                             self.eval_returns,
                             self.controller,
                             self.learner,
                             self.envs_runner,
                             self.save_dir)
        self.envs_runner.close()

        if self.tracking:
            wandb.finish()

        print(f"{[self.run_id]} Finish Training ... ", flush=True)

    def create_instruction_provider(self, current_episode):
        """Create instruction provider that alternates between instructions and no instructions."""
        if not hasattr(self, 'instruction_embeddings') or len(self.instruction_embeddings) == 0:
            return None

        switch_mode = os.environ.get("INSTRUCTION_SWITCH_MODE", "stochastic")
        provided_prob = float(os.environ.get("INSTRUCTION_PROVIDED_PROB", "0.01"))

        if switch_mode == "fixed":
            def _fixed_instruction_provider(env_idx, step, agent_idx=None):
                current_episode = getattr(self, '_current_episode', 0)
                cycle_length = 10000
                position_in_cycle = current_episode % cycle_length
                if position_in_cycle < 5000:
                    if self.instruction_mode == 'fixed_per_agent' and agent_idx is not None:
                        inst_idx = agent_idx % len(self.instruction_texts)
                        return (self.instruction_texts[inst_idx], self.instruction_embeddings[inst_idx])
                    elif self.instruction_mode == 'random_per_agent' and agent_idx is not None:
                        if agent_idx == 0:
                            self.global_step_count[env_idx] += 1
                        if agent_idx == 0 and self.global_step_count[env_idx] - self.last_resample_step[env_idx] >= self.resample_interval:
                            self.agent_instruction_indices[env_idx] = [np.random.randint(0, len(self.instruction_embeddings))
                                                                       for _ in range(self.n_agent)]
                            self.last_resample_step[env_idx] = self.global_step_count[env_idx]
                        inst_idx = self.agent_instruction_indices[env_idx][agent_idx]
                        return (self.instruction_texts[inst_idx], self.instruction_embeddings[inst_idx])
                    else:
                        # Per-environment mode should sample from the full pool
                        # whenever an instruction is assigned, not pin one fixed
                        # instruction forever per env.
                        inst_idx = np.random.randint(0, len(self.instruction_embeddings))
                        return (self.instruction_texts[inst_idx], self.instruction_embeddings[inst_idx])
                else:
                    return None
            return _fixed_instruction_provider
        else:
            # Stochastic schedule: per-step probability handled in envs_runner
            # Provider always returns instructions when called
            def _stochastic_instruction_provider(env_idx, step, agent_idx=None):
                if self.instruction_mode == 'fixed_per_agent' and agent_idx is not None:
                    inst_idx = agent_idx % len(self.instruction_texts)
                    return (self.instruction_texts[inst_idx], self.instruction_embeddings[inst_idx])
                elif self.instruction_mode == 'random_per_agent' and agent_idx is not None:
                    if agent_idx == 0:
                        self.global_step_count[env_idx] += 1
                    if agent_idx == 0 and self.global_step_count[env_idx] - self.last_resample_step[env_idx] >= self.resample_interval:
                        self.agent_instruction_indices[env_idx] = [np.random.randint(0, len(self.instruction_embeddings))
                                                                   for _ in range(self.n_agent)]
                        self.last_resample_step[env_idx] = self.global_step_count[env_idx]
                    inst_idx = self.agent_instruction_indices[env_idx][agent_idx]
                    return (self.instruction_texts[inst_idx], self.instruction_embeddings[inst_idx])
                else:
                    # Per-environment mode: sample a fresh instruction each time
                    # provider is queried.
                    inst_idx = np.random.randint(0, len(self.instruction_embeddings))
                    return (self.instruction_texts[inst_idx], self.instruction_embeddings[inst_idx])
            return _stochastic_instruction_provider


class ACAC_Vanilla(ACAC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Override learner with vanilla
        self.learner = Learner_ACAC_Vanilla(self.env, self.controller, self.memory, kwargs.get('gamma', args[5]))


class ACAC_Micro_GAE(ACAC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Override learner with micro GAE
        self.learner = Learner_ACAC_Micro_GAE(self.env, self.controller, self.memory, kwargs.get('gamma', args[5]))