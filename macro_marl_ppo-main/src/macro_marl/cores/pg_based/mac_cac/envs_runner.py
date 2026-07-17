import os
import numpy as np
import random
import torch
import torch.nn.functional as F

from multiprocessing import Process, Pipe

# Tunable instruction-shaping knobs (read once at import).
# - INSTRUCTION_PENALTY: per-step penalty for non-compliance (default -50.0).
#   Lower magnitudes are useful for "agents-learn-to-ignore" demos where the
#   prohibition is not strong enough to redirect the policy.
# - INSTRUCTION_DURATION_STEPS: 0 = persist until episode end (legacy
#   behavior). >0 clears the instruction text/embedding once that many
#   steps have elapsed since it was provided. Lets the demo show that a
#   short prohibition window is easy for the policy to wait out.
INSTRUCTION_PENALTY = float(os.environ.get("INSTRUCTION_PENALTY", "-50.0"))
INSTRUCTION_DURATION_STEPS = int(os.environ.get("INSTRUCTION_DURATION_STEPS", "0"))

def worker(child, env, gamma, seed):
    """
    Worker function which interacts with the environment over remote
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        while True:
            # wait cmd sent by parent
            cmd, data = child.recv()
            if cmd == 'step':
                obs, reward, terminate, info = env.step(data)
                action = info['cur_mac']
                valid = info['mac_done']

                # accumulate reward of individual macro-action
                for idx, v in enumerate(valid):
                    if last_valid[idx]:
                        accu_rewards[idx] = reward[idx]
                        mac_act_step[idx] = 1
                    else:
                        mac_act_step[idx] += 1
                        accu_rewards[idx] = accu_rewards[idx] + gamma**(mac_act_step[idx]-1)*reward[idx]

                # accumulate reward of joint-macro-action
                if last_joint_valid:
                    accu_joint_reward = sum(reward)/env.n_agent 
                    mac_joint_act_step = 1
                else:
                    mac_joint_act_step += 1
                    accu_joint_reward += gamma**(mac_joint_act_step-1)*sum(reward)/env.n_agent 

                last_valid = valid
                last_joint_valid = max(valid)
                avail_actions = env.get_avail_actions()

                warehouse_request_state = None
                if hasattr(env, 'humans') and hasattr(env, 'n_objs'):
                    warehouse_request_state = {
                        'currently_received_tools': [
                            int(human.next_request_obj_idx)
                            for human in env.humans
                            if human.next_requested_obj_obtained
                        ],
                    }

                # sent experience back
                child.send((last_obs,
                            action,
                            accu_rewards,
                            accu_joint_reward,
                            obs,
                            avail_actions,
                            terminate,
                            valid,
                            max(valid),
                            sum(reward)/env.n_agent,
                            warehouse_request_state))

                last_obs = obs
                R += gamma**step * sum(reward) / env.n_agent
                step += 1
            
            elif cmd == 'get_return':
                child.send(R)

            elif cmd == 'reset':
                last_obs =  env.reset() # List[array]
                h_state = None
                last_action = [-1] * env.n_agent
                last_valid = [1.0] * env.n_agent
                last_joint_valid = 1
                accu_rewards = [0.0] * env.n_agent
                accu_joint_reward = 0.0
                mac_act_step = [0] * env.n_agent
                mac_joint_act_step = 0
                avail_actions = env.get_avail_actions()
                step = 0
                R = 0.0

                child.send((last_obs, h_state, last_action, last_valid, avail_actions))
            elif cmd == 'close':
                child.close()
                break
            elif cmd == 'get_rand_states':
                rand_states = {'random_state': random.getstate(),
                               'np_random_state': np.random.get_state()}
                child.send(rand_states)
            elif cmd == 'load_rand_states':
                random.setstate(data['random_state'])
                np.random.set_state(data['np_random_state'])
            elif cmd == 'reset_macro_actions':
                # Force all agents to resample macro-actions (interrupt ongoing actions)
                for agent in env.env.macroAgent:
                    agent.cur_macro_action_done = True
                child.send('done')
            elif cmd == 'set_tool_order':
                # Change the tool delivery order for warehouse (OSD) environments.
                # data: list of tool indices, e.g. [1, 0, 2]
                # Must be called right after 'reset' (cur_step == 0 for all humans)
                # so the full order applies from the very first delivery.
                tool_order = data
                if tool_order is not None and hasattr(env, 'humans'):
                    for human in env.humans:
                        if len(tool_order) > 0:
                            human.request_objs_per_task_step = list(tool_order)
                            # cur_step is always 0 right after reset; set the first target tool
                            human.next_request_obj_idx = tool_order[0]
                            human.next_requested_obj_obtained = False
                child.send('done')
            else:
                raise NotImplementedError
 
    except KeyboardInterrupt:
        print('EnvRunner worker: caught keyboard interrupt')
    except Exception as e:
        print('EnvRunner worker: uncaught worker exception')
        raise

class EnvsRunner(object):
    """
    Environment runner which runs multiple environemnts in parallel in subprocesses
    and communicates with them via pipe
    """

    def __init__(self, env, n_envs, controller, memory, env_terminate_step, gamma, seed, obs_last_action=False, instruction_provider=None):
        
        self.env = env
        self.max_epi_step = env_terminate_step
        self.n_envs = n_envs
        self.n_agent = env.n_agent
        # controllers for getting next action via current actor nn
        self.controller = controller
        # create connections via Pipe
        self.parents, self.children = [list(i) for i in zip(*[Pipe() for _ in range(n_envs)])]
        # create multip processor with multiple envs
        self.envs = [Process(target=worker, args=(child, env, gamma, seed+idx)) for idx, child in enumerate(self.children)]
        # replay buffer
        self.memory = memory
        # observe last actions
        self.obs_last_action = obs_last_action
        # record parallel episodes
        self.episodes = [[] for i in range(n_envs)]
        # record train return
        self.train_returns = []
        # record eval return
        self.eval_returns = []
        # record instruction compliance during evaluation
        self.eval_compliance = []
        self.eval_compliance_per_instruction = []
        self.eval_compliance_per_agent_instruction = []
        # (return, instruction_text) per episode — instruction may be None.
        # Used by wandb_logging.build_eval_log_dict to split returns by
        # instruction-active vs no-instruction phases.
        self.train_episode_instructions = []
        self.eval_episode_instructions = []
        # Same shape, but holding the SHAPED return: raw env return with
        # the instruction-shaping penalty/bonus added in (i.e. what the
        # critic actually sees). Surfaces as Returns_With_Instruction_Shaped
        # in wandb so we can compare unshaped vs shaped at a glance.
        self.train_episode_instructions_shaped = []
        self.eval_episode_instructions_shaped = []
        # Per-env accumulator for the shaped (post-instruction-shaping)
        # joint reward this episode. Reset at episode boundaries.
        self._epi_shaped_R = [0.0] * n_envs
        # optional instruction provider: callable(env_idx:int, step:int) -> str | Tensor | None
        self.instruction_provider = instruction_provider
        self.instruction_embs = [None] * n_envs
        self.instruction_texts = [None] * n_envs
        # Initialize instruction start steps
        self.instruction_start_steps = [0] * n_envs
        # Per-env step at which the active instruction expires (0 = no expiry).
        # Set when a provider returns a non-None instruction; checked at the
        # top of each _step to clear stale instructions for the demo where
        # we want the prohibition to last only N steps.
        self.instruction_expire_steps = [0] * n_envs
        # Last-seen non-None instruction text within the current episode.
        # Used as the per-instruction compliance key so post-expiry stats
        # are filed correctly even after instruction_texts[idx] flips back
        # to None. Reset to None at episode boundaries.
        self._episode_active_instruction = [None] * n_envs
        # Per-env "fetch tool X first" priority tracking (warehouse / OSD).
        self.priority_tool: "list[int | None]" = [None] * n_envs
        self.priority_satisfied: "list[bool]" = [False] * n_envs

        # memory already contains ZERO_INSTRUCTION; no dynamic patching needed

        # trigger each processor
        for env in self.envs:
            env.daemon = True
            env.start()

        for child in self.children:
            child.close()

    def run(self, eps=0.0, n_epis=1, test_mode=False):

        self._reset()

        if test_mode:
            # Reset compliance + per-episode instruction tracking for this
            # eval round. build_eval_log_dict averages over these buffers,
            # so leaving stale entries from the previous eval cycle would
            # contaminate the current cycle's metrics.
            self.eval_compliance = []
            self.eval_compliance_per_instruction = []
            self.eval_compliance_per_agent_instruction = []
            self.eval_episode_instructions = []
            self.eval_episode_instructions_shaped = []
            while len(self.eval_returns) < n_epis:
                self._step(eps=eps, test_mode=test_mode)
        else:
            while self.n_epi_count < n_epis:
                self._step(eps=eps, test_mode=test_mode)

    def close(self):
        [parent.send(('close', None)) for parent in self.parents]
        [parent.close() for parent in self.parents]
        [env.terminate() for env in self.envs]
        [env.join() for env in self.envs]

    def _step(self, eps=0.0, test_mode=False):

        for idx, parent in enumerate(self.parents):

            # Expire the active instruction once the configured duration has
            # elapsed. INSTRUCTION_DURATION_STEPS=0 (default) means "never
            # expire mid-episode" — this branch is a no-op then.
            if (
                INSTRUCTION_DURATION_STEPS > 0
                and self.instruction_texts[idx] is not None
                and self.instruction_expire_steps[idx] > 0
                and self.step_count[idx] >= self.instruction_expire_steps[idx]
            ):
                self.instruction_embs[idx] = None
                self.instruction_texts[idx] = None
                self.instruction_expire_steps[idx] = 0

            # Provide instruction at a random step for non-warehouse environments.
            # Warehouse (OSD) envs are pre-assigned at episode start.
            if self.instruction_provider is not None and not hasattr(self.env, 'n_objs') and self.step_count[idx] == self.instruction_start_steps[idx]:
            # Provide instruction at the start of EVERY episode (step_count resets to 0 at episode start)
            # if self.instruction_provider is not None and self.step_count[idx] == self.max_epi_step // 2:
                # Get instruction from provider for this new episode
                current_episode = self._env_episode_count[idx]
                provider_result = self._call_instruction_provider(idx, self.step_count[idx])
                
                # Handle tuple return (embedding, text) or just embedding/text/None
                if provider_result is None:
                    inst = None
                    inst_text_from_provider = None
                elif isinstance(provider_result, tuple) and len(provider_result) == 2:
                    inst, inst_text_from_provider = provider_result
                else:
                    # Backward compatibility: if not a tuple, assume it's just the embedding
                    inst = provider_result
                    inst_text_from_provider = getattr(self.controller, 'instruction_text', None)
                
                # Use the instruction text from provider
                if inst_text_from_provider and idx == 0:
                    print(f"DEBUG instruction_provider: env {idx}, episode {current_episode}, instruction='{inst_text_from_provider}'")


                
                if inst is None:
                    self.instruction_embs[idx] = None
                    self.instruction_texts[idx] = None
                elif isinstance(inst, str) or (isinstance(inst, list) and len(inst) > 0 and isinstance(inst[0], str)):
                    self.instruction_embs[idx] = self.controller.agent.actor_net.encode_instruction(inst).detach()
                    self.instruction_texts[idx] = inst
                elif isinstance(inst, torch.Tensor):
                    self.instruction_embs[idx] = inst.detach()
                    # For tensor instructions, we need to find the corresponding text
                    # Store the text from provider
                    self.instruction_texts[idx] = inst_text_from_provider
                else:
                    self.instruction_embs[idx] = None
                    self.instruction_texts[idx] = None

                # If a duration is configured and this step actually set an
                # instruction, schedule its expiry. Otherwise leave the
                # expiry slot zero so the cleanup branch is a no-op.
                if INSTRUCTION_DURATION_STEPS > 0 and self.instruction_texts[idx] is not None:
                    self.instruction_expire_steps[idx] = self.step_count[idx] + INSTRUCTION_DURATION_STEPS
                else:
                    self.instruction_expire_steps[idx] = 0

                # Remember the last-seen non-None instruction text for this
                # episode so per-instruction compliance can be filed under
                # the right key even after the instruction expires.
                if self.instruction_texts[idx] is not None:
                    self._episode_active_instruction[idx] = self.instruction_texts[idx]

            actions, self.h_states[idx] = self.controller.select_action(self.last_obses[idx],
                                                                        self.last_actions[idx],
                                                                        self.h_states[idx],
                                                                        self.last_valids[idx],
                                                                        self.avail_actions[idx],
                                                                        eps=eps,
                                                                        test_mode=test_mode,
                                                                        instruction_emb=self.instruction_embs[idx])

            # Don't clear instruction - keep it for the entire episode
            # send cmd to trigger env step
            parent.send(("step", actions))
            self.step_count[idx] += 1

            # collect envs' returns
        for idx, parent in enumerate(self.parents):
            env_return = parent.recv()
            warehouse_request_state = env_return[10] if len(env_return) >= 11 else None
            env_return = env_return[:10]
            # attach current instruction embedding and text for this env to the transition
            env_return_with_inst = tuple(env_return) + (self.instruction_embs[idx], self.instruction_texts[idx])
            env_return = self._exp_to_tensor(idx, env_return_with_inst, eps)
            self.episodes[idx].append(env_return)

            # Unpack tuple for clarity
            last_obs, last_avail_actions, a, r, j_r, obs, avail_actions, t, mac_v, j_mac_v, exp_v, inst, inst_text = env_return

            # Modify rewards based on instruction compliance
            if test_mode and idx == 0 and self.step_count[idx] == 0:  # Debug only for first env, first step of evaluation
                print(f"DEBUG envs_runner: test_mode={test_mode}, instruction_provider={self.instruction_provider is not None}, controller.instruction_text='{getattr(self.controller, 'instruction_text', 'NONE')}', inst={inst is not None}, inst_text='{inst_text}'")
            r, j_r = self._check_instruction_compliance(idx, a, r, j_r, obs, self.step_count[idx], mac_v, inst, inst_text, test_mode, warehouse_request_state=warehouse_request_state)

            # Track the shaped joint reward for this step so the episode
            # total can be logged alongside the raw env return. j_r may be
            # a tensor or a Python scalar depending on the call path; the
            # `getattr(...).item() or float(...)` dance handles both.
            j_r_scalar = j_r.item() if isinstance(j_r, torch.Tensor) else float(j_r)
            self._epi_shaped_R[idx] += j_r_scalar

            # Update env_return with modified rewards
            env_return = (last_obs, last_avail_actions, a, r, j_r, obs, avail_actions, t, mac_v, j_mac_v, exp_v, inst, inst_text)

            self.last_obses[idx] = obs
            self.avail_actions[idx] = avail_actions
            self.last_valids[idx] = mac_v
            if max(self.last_valids[idx]) > 0:
                for nth in range(self.n_agent):
                    self.last_actions[idx][nth] = a[nth]

            # if episode is done, add it to memory buffer
            if t[0] or self.step_count[idx] == self.max_epi_step:
                self.n_epi_count += 1
                self._env_episode_count[idx] += 1  # Track per-environment episodes
                # collect the return
                parent.send(("get_return", None))
                R = parent.recv()
                
                # Print instruction statistics for this episode
                if self.instruction_provider is not None and hasattr(self, '_instruction_stats') and idx in self._instruction_stats:
                    stats = self._instruction_stats[idx]
                    total_actions = stats['compliant'] + stats['non_compliant']
                    compliance_rate = stats['compliant'] / max(total_actions, 1) * 100

                    # Only print every 10 episodes to avoid spam
                    if self.n_epi_count % 10 == 0:
                        print(f"\n{'='*60}")
                        mode_label = "EVAL Episode" if test_mode else "Episode"
                        print(f"Env {idx} | {mode_label} {self.n_epi_count} | Return: {R:.2f}")
                        
                        # For mac_cac, all agents share the same instruction
                        # Get the instruction text from the stored instruction for this environment
                        if hasattr(self, 'instruction_texts') and self.instruction_texts[idx] is not None:
                            instruction_text = self.instruction_texts[idx]
                        else:
                            instruction_text = None

                        # Per-agent expected behavior — covers per-agent-routed
                        # instructions like "go to small box" where the same
                        # broadcast text maps to different actions per agent.
                        expected = (
                            [self._get_expected_macro_action(instruction_text, agent_idx=i)
                             for i in range(self.n_agent)]
                            if instruction_text else None
                        )
                        print(f"Shared Instruction: '{instruction_text}' -> Expected (per agent): {expected}")
                        print(f"Compliance: {stats['compliant']}/{total_actions} ({compliance_rate:.1f}%)")
                        print(f"Action distribution: {dict(sorted(stats['action_counts'].items()))}")
                        print(f"{'='*60}\n")

                # Track instruction compliance for evaluation episodes.
                # Only contribute to the averages when the instruction was
                # actually active for at least one tracked step this
                # episode — episodes with no active instruction would
                # otherwise be filed as "100% compliant" and inflate the
                # metric.
                if test_mode and self.instruction_provider is not None:
                    episode_compliance = self._calculate_episode_compliance(idx)
                    if episode_compliance is not None:
                        self.eval_compliance.append(episode_compliance['overall'])
                        self.eval_compliance_per_instruction.append(episode_compliance['per_instruction'])
                        self.eval_compliance_per_agent_instruction.append(episode_compliance['per_agent_instruction'])
                
                # Active instruction (if any) for this episode — we use
                # _episode_active_instruction (last seen non-None) so a
                # mid-episode expiry doesn't reset us back to "no
                # instruction". Falls back to the current text if that
                # tracker isn't populated.
                epi_inst_text = (
                    self._episode_active_instruction[idx]
                    if hasattr(self, '_episode_active_instruction')
                       and self._episode_active_instruction[idx] is not None
                    else (
                        self.instruction_texts[idx]
                        if hasattr(self, 'instruction_texts')
                        else None
                    )
                )
                shaped_R = self._epi_shaped_R[idx]

                if not test_mode:
                    self.memory.scenario_cache += self.episodes[idx]
                    self.memory.flush_buf_cache()
                    self.train_returns.append(R)
                    self.train_episode_instructions.append((R, epi_inst_text))
                    self.train_episode_instructions_shaped.append((shaped_R, epi_inst_text))
                else:
                    self.eval_returns.append(R)
                    self.eval_episode_instructions.append((R, epi_inst_text))
                    self.eval_episode_instructions_shaped.append((shaped_R, epi_inst_text))

                # when episode is done, immediately start a new one
                parent.send(("reset", None))
                self.last_obses[idx], self.h_states[idx], self.last_actions[idx], self.last_valids[idx], self.avail_actions[idx] = parent.recv()
                self.last_obses[idx] = self.obs_to_tensor(self.last_obses[idx])
                self.last_actions[idx] = self.action_to_tensor(self.last_actions[idx])
                if self.obs_last_action:
                    self.last_obses[idx] = self.rebuild_obs(self.env, self.last_obses[idx], self.last_actions[idx])
                self.last_valids[idx] = self.mac_done_to_tensor(self.last_valids[idx])
                self.avail_actions[idx] = self.avail_action_to_tensor(self.avail_actions[idx])
                self.episodes[idx] = []

                # Reset instruction for new episode
                self.instruction_embs[idx] = None
                self.instruction_texts[idx] = None
                self.instruction_expire_steps[idx] = 0
                self._episode_active_instruction[idx] = None
                self._epi_shaped_R[idx] = 0.0

                # Pick new start step for next episode (non-warehouse only)
                if not hasattr(self.env, 'n_objs'):
                    low = 3
                    high = max(low + 1, self.max_epi_step)
                    self.instruction_start_steps[idx] = np.random.randint(low, high)

                self.step_count[idx] = 0

                # Warehouse: pre-assign instruction + set tool order at episode start
                if self.instruction_provider is not None:
                    self._pre_assign_warehouse_instruction(idx, parent)

                # Reset instruction stats for new episode
                if hasattr(self, '_instruction_stats') and idx in self._instruction_stats:
                    self._instruction_stats[idx] = {
                        'compliant': 0, 'non_compliant': 0, 'action_counts': {}
                    }

    def _check_instruction_compliance(self, env_idx, actions, rewards, joint_reward, obs, step_count, mac_v=None, inst_emb=None, inst_text=None, test_mode=False, warehouse_request_state=None):
        """
        Check if the agent's actions comply with the instruction that was active when the action was taken.

        Args:
            env_idx: Environment index
            actions: Actions taken by agents (macro-action indices)
            rewards: Individual agent rewards
            joint_reward: Joint reward
            obs: Current observations
            step_count: Current step count
            mac_v: Macro-action validity/completion status
            inst_emb: Instruction embedding that was active when the action was taken
            inst_text: Instruction text that was active when the action was taken
            test_mode: Whether this is evaluation mode

        Returns:
            modified_rewards, modified_joint_reward
        """
        # If no instruction embedding was provided for this transition, use current instruction
        if inst_emb is None or (isinstance(inst_emb, torch.Tensor) and inst_emb.numel() == 0):
            if self.instruction_provider is None:
                return rewards, joint_reward

            # Get the current instruction for this environment from stored instruction texts
            if hasattr(self, 'instruction_texts') and self.instruction_texts[env_idx] is not None:
                instruction_text = self.instruction_texts[env_idx]
            else:
                # Fallback if instruction info not available
                return rewards, joint_reward
        else:
            # Use the instruction text that was active when the action was taken
            # Check if inst_text is a string or tensor
            if inst_text is not None and not isinstance(inst_text, torch.Tensor):
                instruction_text = inst_text
            elif hasattr(self, 'instruction_texts') and self.instruction_texts[env_idx] is not None:
                instruction_text = self.instruction_texts[env_idx]
            else:
                return rewards, joint_reward

        # Debug: print instruction checking details
        if test_mode and env_idx == 0 and step_count % 100 == 0:
            print(f"DEBUG compliance: env_idx={env_idx}, step={step_count}, instruction_text='{instruction_text}', inst_emb={inst_emb is not None}, inst_text='{inst_text}'")

        # Fast path: if no known behavior at all, keep rewards unchanged.
        # Must be agent-aware — per-agent-routed instructions like "go to
        # small box" return None when called with agent_idx=None, so a
        # naive global check would skip shaping/counting for those.
        if all(
            self._get_expected_macro_action(instruction_text, agent_idx=i) is None
            for i in range(self.n_agent)
        ):
            return rewards, joint_reward

        # Initialize stats tracking if needed
        if not hasattr(self, '_instruction_stats'):
            self._instruction_stats = {}
        if env_idx not in self._instruction_stats:
            print(f"DEBUG compliance init: initializing stats for env {env_idx}")
            self._instruction_stats[env_idx] = {
                'compliant': 0, 'non_compliant': 0, 'action_counts': {}
            }
        else:
            # Debug: print current stats before updating
            if env_idx == 0 and step_count % 100 == 0:
                current_stats = self._instruction_stats[env_idx]
                print(f"DEBUG compliance update: env {env_idx}, step {step_count}, current_stats={current_stats}")

        # Apply reward shaping for each agent
        modified_rewards = []
        is_warehouse_inst = self._parse_warehouse_tool_order(instruction_text) is not None
        wh_penalty = -25
        wh_bonus = 10
        # After first delivery, a "fetch X first" instruction no longer applies.
        # Skip both shaping AND compliance counting in that case.
        skip_warehouse_shaping = is_warehouse_inst and self.priority_satisfied[env_idx]

        for agent_idx, action in enumerate(actions):
            action_value = action.item() if isinstance(action, torch.Tensor) else action
            expected_behavior = self._get_expected_macro_action(instruction_text, agent_idx=agent_idx)

            # Track action distribution
            if action_value not in self._instruction_stats[env_idx]['action_counts']:
                self._instruction_stats[env_idx]['action_counts'][action_value] = 0
            self._instruction_stats[env_idx]['action_counts'][action_value] += 1

            base_reward = rewards[agent_idx]

            # No matching behavior, or warehouse priority already satisfied:
            # keep the env reward unchanged so the centralized critic still
            # sees subtask / step-penalty / delivery signal.
            if expected_behavior is None or skip_warehouse_shaping:
                modified_rewards.append(base_reward)
                continue

            shaped_reward = base_reward

            # Non-warehouse magnitudes match mac_iac (which learns this task):
            # 0 bonus on compliance, -50 penalty on non-compliance. Warehouse
            # path keeps its tuned wh_penalty / wh_bonus + ±50 first-delivery
            # shot in _apply_warehouse_priority_terminal.
            if 'prohibited_actions' in expected_behavior:
                prohibited = expected_behavior['prohibited_actions']
                if action_value in prohibited:
                    shaped_reward = base_reward + (wh_penalty if is_warehouse_inst else INSTRUCTION_PENALTY)
                    self._instruction_stats[env_idx]['non_compliant'] += 1
                else:
                    shaped_reward = base_reward + (wh_bonus if is_warehouse_inst else 0)
                    self._instruction_stats[env_idx]['compliant'] += 1

            elif 'allowed_actions' in expected_behavior:
                allowed = expected_behavior['allowed_actions']
                if action_value in allowed:
                    shaped_reward = base_reward + (wh_bonus if is_warehouse_inst else 0)
                    self._instruction_stats[env_idx]['compliant'] += 1
                else:
                    shaped_reward = base_reward + (wh_penalty if is_warehouse_inst else INSTRUCTION_PENALTY)
                    self._instruction_stats[env_idx]['non_compliant'] += 1

            modified_rewards.append(shaped_reward)

        # One-shot ±50 terminal credit at the moment of first delivery.
        modified_rewards = self._apply_warehouse_priority_terminal(
            env_idx, modified_rewards, warehouse_request_state,
        )

        # Calculate joint reward as average of modified individual rewards (ensures consistency)
        modified_joint_reward = sum(modified_rewards) / self.n_agent

        return modified_rewards, modified_joint_reward

    def _apply_warehouse_priority_terminal(self, env_idx, modified_rewards, warehouse_request_state):
        """One-shot ±50 credit to Fetch on the first delivery of the episode.

        +50 if the first delivered tool matches priority_tool[env_idx], -50
        otherwise. Flips priority_satisfied=True so subsequent steps skip
        warehouse shaping entirely.
        """
        if warehouse_request_state is None:
            return modified_rewards
        if self.priority_tool[env_idx] is None or self.priority_satisfied[env_idx]:
            return modified_rewards

        received = warehouse_request_state.get('currently_received_tools', [])
        if not received:
            return modified_rewards

        first_tool = int(received[0])
        bonus = 50.0 if first_tool == self.priority_tool[env_idx] else -50.0
        fetch_idx = self.n_agent - 1
        if fetch_idx < len(modified_rewards):
            modified_rewards[fetch_idx] = modified_rewards[fetch_idx] + bonus
        self.priority_satisfied[env_idx] = True
        return modified_rewards

    def _get_expected_macro_action(self, instruction_text, agent_idx=None):
        """
        Map instruction text to expected behavior (allowed or prohibited actions).
        Supports both Overcooked and Box Pushing environments.
        
        Returns:
            dict with 'allowed_actions' or 'prohibited_actions', or None if no instruction match
        """
        if instruction_text is None:
            return None
            
        instruction_lower = instruction_text.lower().strip()
        
        # "No instruction" phrases - return None to disable reward shaping
        no_instruction_phrases = [
            "no instruction", "continue as normal", "no specific instruction", ""
        ]
        if instruction_lower in no_instruction_phrases:
            return None
        
        # === OVERCOOKED INSTRUCTIONS ===

        # Define macro-action indices (must match environment macroActionName)
        # All maps: ["stay", "get tomato", "get lettuce", "get onion", "get peas",
        #            "get plate 1", "get plate 2", "go to knife 1", "go to knife 2",
        #            "deliver", "chop", ...]
        STAY = 0
        GET_TOMATO = 1
        GET_LETTUCE = 2
        GET_ONION = 3
        GET_PEAS = 4
        GET_PLATE_1 = 5
        GET_PLATE_2 = 6
        GO_TO_KNIFE_1 = 7
        GO_TO_KNIFE_2 = 8
        DELIVER = 9
        CHOP = 10
        # Map D ovens
        USE_LEFT_OVEN = 14   # "go to oven 1"
        USE_RIGHT_OVEN = 15  # "go to oven 2"

        left_action = None
        try:
            if hasattr(self.env, "macroActionName") and "left" in self.env.macroActionName:
                left_action = self.env.macroActionName.index("left")
        except (ValueError, AttributeError, TypeError):
            left_action = None

        # Positive instructions (do X)
        overcooked_positive_map = {
            "stay": {'allowed_actions': [STAY]},
            "get tomato": {'allowed_actions': [GET_TOMATO]},
            "get lettuce": {'allowed_actions': [GET_LETTUCE]},
            "get onion": {'allowed_actions': [GET_ONION]},
            "get plate 1": {'allowed_actions': [GET_PLATE_1]},
            "get plate": {'allowed_actions': [GET_PLATE_1]},
            "go to knife 1": {'allowed_actions': [GO_TO_KNIFE_1]},
            "go to knife 2": {'allowed_actions': [GO_TO_KNIFE_2]},
            "deliver": {'allowed_actions': [DELIVER]},
            "chop": {'allowed_actions': [CHOP]},
        }

        if left_action is not None:
            overcooked_positive_map.update({
                "move left": {'allowed_actions': [left_action]},
                "move to the left": {'allowed_actions': [left_action]},
                "go left": {'allowed_actions': [left_action]},
            })

        # Negative instructions (don't do X)
        overcooked_negative_map = {
            "don't stay": {'prohibited_actions': [STAY]},
            "don't get tomato": {'prohibited_actions': [GET_TOMATO]},
            "don't touch the tomato": {'prohibited_actions': [GET_TOMATO]},
            "don't touch tomato": {'prohibited_actions': [GET_TOMATO]},
            "don't get lettuce": {'prohibited_actions': [GET_LETTUCE]},
            "don't touch the lettuce": {'prohibited_actions': [GET_LETTUCE]},
            "don't touch lettuce": {'prohibited_actions': [GET_LETTUCE]},
            "don't get onion": {'prohibited_actions': [GET_ONION]},
            "don't touch the onion": {'prohibited_actions': [GET_ONION]},
            "don't touch onion": {'prohibited_actions': [GET_ONION]},
            "don't get plate 1": {'prohibited_actions': [GET_PLATE_1]},
            "don't get plate": {'prohibited_actions': [GET_PLATE_1]},
            "don't go to knife 1": {'prohibited_actions': [GO_TO_KNIFE_1]},
            "don't go to knife 2": {'prohibited_actions': [GO_TO_KNIFE_2]},
            "don't use the left cutting board": {'prohibited_actions': [GO_TO_KNIFE_2]},
            "don't use left cutting board": {'prohibited_actions': [GO_TO_KNIFE_2]},
            "don't use the right cutting board": {'prohibited_actions': [GO_TO_KNIFE_1]},
            "don't use right cutting board": {'prohibited_actions': [GO_TO_KNIFE_1]},
            "don't deliver": {'prohibited_actions': [DELIVER]},
            "don't chop": {'prohibited_actions': [CHOP]},
            # Map D oven prohibitions
            "don't use the left oven": {'prohibited_actions': [USE_LEFT_OVEN]},
            "don't go to the left oven": {'prohibited_actions': [USE_LEFT_OVEN]},
            "avoid left oven": {'prohibited_actions': [USE_LEFT_OVEN]},
            "don't use the right oven": {'prohibited_actions': [USE_RIGHT_OVEN]},
            "don't go to the right oven": {'prohibited_actions': [USE_RIGHT_OVEN]},
            "avoid right oven": {'prohibited_actions': [USE_RIGHT_OVEN]},
        }
        
        # === BOX PUSHING INSTRUCTIONS ===

        # Define macro-action indices for Box Pushing environment
        GT_SMALL_BOX_0 = 0
        GT_SMALL_BOX_1 = 1
        GT_BIG_BOX_SPOT_0 = 2
        GT_BIG_BOX_SPOT_1 = 3
        PUSH = 4
        TURN_LEFT = 5
        TURN_RIGHT = 6
        STAY = 7

        # Per-agent: "go to small box" routes agent 0 -> small box 0,
        # agent 1 -> small box 1. Handled BEFORE the generic substring
        # matching loop below because that loop would otherwise hit both
        # "go to small box 0" and "go to small box 1" keys via substring
        # match on "go to small box", producing combined allowed_actions.
        if instruction_lower == "go to small box" or instruction_lower == "small box":
            if agent_idx == 0:
                return {'allowed_actions': [GT_SMALL_BOX_0]}
            elif agent_idx == 1:
                return {'allowed_actions': [GT_SMALL_BOX_1]}
            else:
                return None

        # Box Pushing positive instructions
        box_pushing_positive_map = {
            "go to small box 0": {'allowed_actions': [GT_SMALL_BOX_0]},
            "go to small box 1": {'allowed_actions': [GT_SMALL_BOX_1]},
            "go to big box spot 0": {'allowed_actions': [GT_BIG_BOX_SPOT_0]},
            "go to big box spot 1": {'allowed_actions': [GT_BIG_BOX_SPOT_1]},
            "push": {'allowed_actions': [PUSH]},
            "big_box_spot_0": {'allowed_actions': [GT_BIG_BOX_SPOT_0]},
            "big_box_spot_1": {'allowed_actions': [GT_BIG_BOX_SPOT_1]},
            "small_box_0": {'allowed_actions': [GT_SMALL_BOX_0]},
            "small_box_1": {'allowed_actions': [GT_SMALL_BOX_1]},
        }

        # Box Pushing negative instructions
        box_pushing_negative_map = {
            "don't go to small box 0": {'prohibited_actions': [GT_SMALL_BOX_0]},
            "don't go to small box 1": {'prohibited_actions': [GT_SMALL_BOX_1]},
            "don't go to any small box": {'prohibited_actions': [GT_SMALL_BOX_0, GT_SMALL_BOX_1]},
            "don't go to small boxes": {'prohibited_actions': [GT_SMALL_BOX_0, GT_SMALL_BOX_1]},
            "don't go to big box spot 0": {'prohibited_actions': [GT_BIG_BOX_SPOT_0]},
            "don't go to big box spot 1": {'prohibited_actions': [GT_BIG_BOX_SPOT_1]},
            "don't push": {'prohibited_actions': [PUSH]},
            "stop pushing the box": {'prohibited_actions': [PUSH]},
            "stop pushing": {'prohibited_actions': [PUSH]},
            "stop push": {'prohibited_actions': [PUSH]},
            "avoid small box 0": {'prohibited_actions': [GT_SMALL_BOX_0]},
            "avoid small box 1": {'prohibited_actions': [GT_SMALL_BOX_1]},
            "avoid small boxes": {'prohibited_actions': [GT_SMALL_BOX_0, GT_SMALL_BOX_1]},
            "avoid all small boxes": {'prohibited_actions': [GT_SMALL_BOX_0, GT_SMALL_BOX_1]},
        }
        
        # Handle multiple instructions by finding all matches and combining them
        all_prohibited = set()
        all_allowed = set()

        # Try all mappings - check if any key is contained in the instruction text
        for mapping in [overcooked_positive_map, overcooked_negative_map,
                       box_pushing_positive_map, box_pushing_negative_map]:
            for key in mapping:
                if key in instruction_lower:
                    expected = mapping[key]
                    if 'prohibited_actions' in expected:
                        all_prohibited.update(expected['prohibited_actions'])
                    if 'allowed_actions' in expected:
                        all_allowed.update(expected['allowed_actions'])

        if all_prohibited:
            return {'prohibited_actions': list(all_prohibited)}
        elif all_allowed:
            return {'allowed_actions': list(all_allowed)}

        # === WAREHOUSE (OSD) INSTRUCTIONS ===
        # Macro-action layout for Fetch robot (last agent):
        #   0 : Wait_Request
        #   1 : Pass_Obj_T0
        #   2 : Pass_Obj_T1
        #   3+i : Look_For_obj_i  (i = 0 ... n_objs-1)
        # Compliance is only enforced for Fetch robot; Turtlebots are ignored.
        import re as _re

        _wh_pos = _re.search(r'(?:fetch|deliver)\s+tool\s+(\d+)\s+first', instruction_lower)
        if _wh_pos:
            tool_idx = int(_wh_pos.group(1))
            if agent_idx is not None and agent_idx != self.n_agent - 1:
                return None
            try:
                n_objs = self.env.n_objs
            except AttributeError:
                n_objs = 3
            prohibited = [3 + i for i in range(n_objs) if i != tool_idx]
            if prohibited:
                return {'prohibited_actions': prohibited}
            return None

        _wh_neg = _re.search(r"don't\s+(?:fetch|search\s+for)\s+tool\s+(\d+)", instruction_lower)
        if _wh_neg:
            tool_idx = int(_wh_neg.group(1))
            if agent_idx is not None and agent_idx != self.n_agent - 1:
                return None
            return {'prohibited_actions': [3 + tool_idx]}

        return None


    def _reset(self):
        # send cmd to reset envs
        for parent in self.parents:
            parent.send(("reset", None))

        self.last_obses, self.h_states, self.last_actions, self.last_valids, self.avail_actions = [list(i) for i in zip(*[parent.recv() for parent in self.parents])]
        self.last_obses = [self.obs_to_tensor(obs) for obs in self.last_obses] #List[List[tensor]]
        self.last_actions = [self.action_to_tensor(a) for a in self.last_actions]
        if self.obs_last_action:
            # reconstruct obs to observe actions
            self.last_obses = [self.rebuild_obs(self.env, obs, a) for obs, a in zip(*[self.last_obses, self.last_actions])]
        self.last_valids = [self.mac_done_to_tensor(mac_done) for mac_done in self.last_valids]
        self.avail_actions = [self.avail_action_to_tensor(avail_action) for avail_action in self.avail_actions]

        self.n_epi_count = 0
        self.step_count = [0] * self.n_envs
        self.episodes = [[] for i in range(self.n_envs)]
        # Clear any previous instructions at reset to ensure fresh instructions for each run
        self.instruction_embs = [None] * self.n_envs
        self.instruction_texts = [None] * self.n_envs
        # Track per-environment episode count for instruction provision
        self._env_episode_count = [0] * self.n_envs

        # Initialize random start steps for instructions
        for i in range(self.n_envs):
            if not hasattr(self.env, 'n_objs'):
                low = 3
                high = max(low + 1, self.max_epi_step)
                self.instruction_start_steps[i] = np.random.randint(low, high)

        # Pre-assign instructions for warehouse environments
        for env_idx, parent in enumerate(self.parents):
            self._pre_assign_warehouse_instruction(env_idx, parent)

    def _call_instruction_provider(self, env_idx, step):
        """
        Call instruction provider while supporting both signatures:
        - provider(env_idx, step)
        - provider(env_idx, step, agent_idx=0)
        """
        if self.instruction_provider is None:
            return None
        try:
            return self.instruction_provider(env_idx, step, agent_idx=0)
        except TypeError:
            return self.instruction_provider(env_idx, step)

    def _pre_assign_warehouse_instruction(self, env_idx, parent):
        """
        For warehouse (OSD) environments only: decide at episode START whether
        this episode gets an instruction, and if so apply it immediately via
        'set_tool_order' before any steps are taken.

        The per-episode probability is derived from INSTRUCTION_PROVIDED_PROB
        (the per-step value used by other envs) scaled to one episode:
            P_episode = 1 - (1 - P_step) ** max_epi_step
        """
        # Reset per-episode priority state regardless of whether an instruction
        # is sampled.
        self.priority_tool[env_idx] = None
        self.priority_satisfied[env_idx] = False

        if self.instruction_provider is None or not hasattr(self.env, 'n_objs'):
            return

        provided_prob = float(os.environ.get("INSTRUCTION_PROVIDED_PROB", "0.01"))
        per_episode_prob = 1.0 - (1.0 - provided_prob) ** self.max_epi_step

        if np.random.random() >= per_episode_prob:
            return

        provider_result = self._call_instruction_provider(env_idx, 0)
        if provider_result is None:
            return

        inst = None
        inst_text = None

        if isinstance(provider_result, tuple) and len(provider_result) == 2:
            first, second = provider_result
            # Support both (emb, text) and (text, emb)
            if isinstance(first, torch.Tensor):
                inst = first
                inst_text = second if isinstance(second, str) else None
            elif isinstance(second, torch.Tensor):
                inst = second
                inst_text = first if isinstance(first, str) else None
        elif isinstance(provider_result, torch.Tensor):
            inst = provider_result
        elif isinstance(provider_result, str) or (isinstance(provider_result, list) and len(provider_result) > 0 and isinstance(provider_result[0], str)):
            inst = self.controller.agent.actor_net.encode_instruction(provider_result).detach()
            inst_text = provider_result if isinstance(provider_result, str) else None

        if inst is None:
            return

        self.instruction_embs[env_idx] = inst.detach()
        self.instruction_texts[env_idx] = inst_text

        tool_order = self._parse_warehouse_tool_order(inst_text)
        if tool_order is not None:
            self.priority_tool[env_idx] = int(tool_order[0])
            parent.send(('set_tool_order', tool_order))
            parent.recv()

    def _parse_warehouse_tool_order(self, instruction_text):
        """
        Parse warehouse instruction into explicit tool delivery order.
        Supported pattern: "fetch tool X first" / "deliver tool X first".
        """
        if instruction_text is None:
            return None

        import re as _re

        instruction_lower = instruction_text.lower().strip()
        match = _re.search(r'(?:fetch|deliver)\s+tool\s+(\d+)\s+first', instruction_lower)
        if not match:
            return None

        priority_tool = int(match.group(1))
        try:
            n_objs = self.env.n_objs
        except AttributeError:
            return None

        if priority_tool < 0 or priority_tool >= n_objs:
            return None

        rest = [i for i in range(n_objs) if i != priority_tool]
        return [priority_tool] + rest

    def _exp_to_tensor(self, env_idx, exp, eps):
        # exp (last_obs, a, r, j_r, obs, avail_actions, t, mac_v, j_mac_v, [exp_v], [instr_emb])
        last_obs = [torch.from_numpy(o).float().view(1,-1) for o in exp[0]]
        last_avail_actions = [torch.FloatTensor(avail_action).view(1,-1) for avail_action in self.avail_actions[env_idx]]
        a = [torch.tensor(a).view(1,-1) for a in exp[1]]
        r = [torch.tensor(r).float().view(1,-1) for r in exp[2]]
        j_r = torch.tensor(exp[3]).float().view(1,-1) 
        obs = [torch.from_numpy(o).float().view(1,-1) for o in exp[4]]
        avail_actions = [torch.FloatTensor(avail_action).view(1,-1) for avail_action in exp[5]]
        # re-construct obs if obs last action
        if self.obs_last_action:
            last_obs = self.rebuild_obs(self.env, last_obs, self.last_actions[env_idx])
            obs = self.rebuild_obs(self.env, obs, a)
        t = torch.tensor(exp[6]).float().view(1,-1)
        mac_v = [torch.tensor(v, dtype=torch.bool).view(1,-1) for v in exp[7]]
        j_mac_v = torch.tensor(exp[8], dtype=torch.bool).view(1,-1)
        exp_v = [torch.tensor([1.0]).view(1,-1)] * self.n_agent
        # instruction embedding may be None, string-encoded earlier, or a tensor
        inst = exp[10] if len(exp) > 10 else None
        inst_text = exp[11] if len(exp) > 11 else None
        if inst is None:
            inst_tensor = self.memory.ZERO_INSTRUCTION.clone()
        else:
            if isinstance(inst, torch.Tensor):
                inst_tensor = inst.view(1,-1)
            else:
                # should not happen (encoding done earlier), fallback to zeros
                inst_tensor = self.memory.ZERO_INSTRUCTION.clone()

        return (last_obs, last_avail_actions, a, r, j_r, obs, avail_actions, t, mac_v, j_mac_v, exp_v, inst_tensor, inst_text)

    def _calculate_episode_compliance(self, env_idx):
        """
        Calculate instruction compliance for an episode (mac_cac version).
        Compliance is only meaningful for steps where an instruction was
        active and had expected behavior — episodes with zero such steps
        return None so the caller can exclude them from the average.

        Returns:
            dict | None:
                None if the instruction was never active (or imposed no
                constraints) this episode — caller should skip recording.
                Otherwise:
                    'overall':   float compliance rate (0.0 to 1.0)
                    'per_instruction': {instruction_text: rate}
                    'per_agent_instruction': {}
        """
        if not hasattr(self, '_instruction_stats') or env_idx not in self._instruction_stats:
            return None

        stats = self._instruction_stats[env_idx]
        total = stats['compliant'] + stats['non_compliant']

        if total == 0:
            return None

        compliance_rate = stats['compliant'] / total

        # Print action distribution for debugging
        action_dist_str = ', '.join([f"action_{k}: {v}" for k, v in sorted(stats['action_counts'].items())])
        print(f"DEBUG compliance: env {env_idx}, compliant={stats['compliant']}, non_compliant={stats['non_compliant']}, total={total}, rate={compliance_rate:.3f}")
        print(f"  Action distribution: {action_dist_str}")

        # File this episode's compliance under the instruction that was
        # actually active when the stats were recorded. With
        # INSTRUCTION_DURATION_STEPS > 0 the active text gets cleared
        # mid-episode, so we prefer the last-seen non-None instruction
        # tracked separately in `_episode_active_instruction[env_idx]`.
        inst_text = None
        if hasattr(self, '_episode_active_instruction') and self._episode_active_instruction[env_idx] is not None:
            inst_text = self._episode_active_instruction[env_idx]
        elif hasattr(self, 'instruction_texts') and self.instruction_texts[env_idx] is not None:
            inst_text = self.instruction_texts[env_idx]
        inst_key = inst_text if inst_text is not None else "__no_instruction__"
        per_instruction = {inst_key: compliance_rate}

        return {
            'overall': compliance_rate,
            'per_instruction': per_instruction,
            'per_agent_instruction': {}
        }

    @staticmethod
    def obs_to_tensor(obs):
        return [torch.from_numpy(o).float().view(1,-1) for o in obs]

    @staticmethod
    def action_to_tensor(action):
        return [torch.tensor(a).view(1,-1) for a in action]

    @staticmethod
    def mac_done_to_tensor(mac_done):
        return [torch.tensor(done, dtype=torch.bool).view(1,-1) for done in mac_done]

    @staticmethod
    def rebuild_obs(env, obs, action):
        new_obs = []
        for o, a, a_dim in zip(*[obs, action, env.n_action]):
            if a == -1:
                one_hot_a = torch.zeros(a_dim).view(1,-1)
            else:
                one_hot_a = F.one_hot(a.view(-1), a_dim).float()
            new_obs.append(torch.cat([o, one_hot_a], dim=1))
        return new_obs

    @staticmethod
    def avail_action_to_tensor(avail_action):
        return [torch.FloatTensor(a).view(1,-1) for a in avail_action]
