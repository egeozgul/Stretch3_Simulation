import os
import numpy as np
import random
import torch
import torch.nn.functional as F
import copy
from multiprocessing import Process, Pipe
import time

# Tunable instruction-shaping knobs (mirror mac_cac / mac_iac /
# mac_iaicc so the same env-vars work across all four pg_based runners).
INSTRUCTION_PENALTY = float(os.environ.get("INSTRUCTION_PENALTY", "-50.0"))
INSTRUCTION_DURATION_STEPS = int(os.environ.get("INSTRUCTION_DURATION_STEPS", "0"))

def worker(child, env, gamma, seed, worker_idx):
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
                n_delivery += info.get("shelf_deliveries", 0)

                warehouse_request_state = None
                if hasattr(env, 'humans') and hasattr(env, 'n_objs') and hasattr(env, 'agents'):
                    pending_tools = [
                        int(human.next_request_obj_idx)
                        for human in env.humans
                        if (not human.whole_task_finished) and (not human.next_requested_obj_obtained)
                    ]
                    currently_received_tools = [
                        int(human.next_request_obj_idx)
                        for human in env.humans
                        if human.next_requested_obj_obtained
                    ]
                    fetch_agent = env.agents[env.n_agent - 1]
                    fetch_action_idx = None
                    if getattr(fetch_agent, 'cur_action', None) is not None:
                        fetch_action_idx = int(fetch_agent.cur_action.idx)
                    fetch_found_objs = [int(obj_idx) for obj_idx in getattr(fetch_agent, 'found_objs', [])]
                    warehouse_request_state = {
                        'pending_tools': pending_tools,
                        'currently_received_tools': currently_received_tools,
                        'fetch_action_idx': fetch_action_idx,
                        'fetch_found_objs': fetch_found_objs,
                    }

                # accumulate reward of individual macro-action
                for idx, v in enumerate(valid):
                    if last_valid[idx]:
                        accu_rewards[idx] = reward[idx]
                        mac_act_step[idx] = 1
                        last_mac_start[idx] = 1
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

                disc_r += gamma**step * sum(reward) / env.n_agent
                sum_r += sum(reward) / env.n_agent
                step += 1

                # sent experience back
                child.send((last_obs, 
                            last_mac_start,
                            action, 
                            accu_rewards, 
                            accu_joint_reward,
                            obs, 
                            avail_actions, 
                            terminate, 
                            valid, 
                            max(valid),
                            warehouse_request_state))

                last_mac_start = [0] * env.n_agent
                last_obs = obs
            elif cmd == "render":
                child.send(env.render(mode="rgb_array"))
            
            elif cmd == 'get_info':
                info = {'r': sum_r, 'l': step, 'R': disc_r, 'n_delivery': n_delivery}
                child.send(info)
            
            elif cmd == 'reset':
                last_obs =  env.reset() # List[array]
                h_state = [None] * env.n_agent
                last_action = [-1] * env.n_agent
                last_valid = [1.0] * env.n_agent
                # record the moment when a new macro-action start
                last_mac_start = [0] * env.n_agent
                last_joint_valid = 1
                accu_rewards = [0.0] * env.n_agent
                accu_joint_reward = 0.0
                mac_act_step = [0] * env.n_agent
                mac_joint_act_step = 0
                avail_actions = env.get_avail_actions()
                step = 0
                disc_r = 0.0
                sum_r = 0.0
                n_delivery = 0

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
                # Force all agents to resample macro-actions on the next step
                # (used by the instruction provider to interrupt an in-flight
                # macro-action so an arriving instruction takes effect now,
                # not several steps later).
                #
                # The flag name + accessor differs between env families:
                #   * Overcooked (wrapped by MacEnvWrapper):
                #       env.env.macroAgent[i].cur_macro_action_done
                #   * Warehouse / OSD and BoxPushing-MA (no wrapper):
                #       env.agents[i].cur_action_done
                # We dispatch by attribute presence and silently no-op on envs
                # that don't expose either (no harm — the next sample happens
                # at the natural macro-action boundary).
                if hasattr(env, 'env') and hasattr(env.env, 'macroAgent'):
                    for agent in env.env.macroAgent:
                        agent.cur_macro_action_done = True
                elif hasattr(env, 'agents'):
                    for agent in env.agents:
                        if hasattr(agent, 'cur_action_done'):
                            agent.cur_action_done = True
                child.send('done')
            elif cmd == 'set_tool_order':
                # Change the tool delivery order for warehouse (OSD) envs.
                # data: list of tool indices, e.g. [1, 0, 2]. Must be called
                # right after 'reset' (cur_step == 0 for all humans) so the
                # full order applies from the very first delivery.
                tool_order = data
                if tool_order is not None and hasattr(env, 'humans'):
                    for human in env.humans:
                        if len(tool_order) > 0:
                            human.request_objs_per_task_step = list(tool_order)
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
    Environment runner which runs mulitpl environemnts in parallel in subprocesses
    and communicates with them via pipe
    """

    def __init__(self, env, n_envs, controller, memory, env_terminate_step, gamma, seed, obs_last_action=False, trace_len=200, parallel=True, instruction_provider=None):
        self.env = env
        self.max_epi_step = env_terminate_step
        self.n_envs = n_envs
        self.n_agent = env.n_agent
        self.trace_len = trace_len
        self.parallel = parallel
        self.batch_idx = 0
        # controllers for getting next action via current actor nn
        self.controller = controller
        # create connections via Pipe
        self.parents, self.children = [list(i) for i in zip(*[Pipe() for _ in range(n_envs)])]
        # create multip processor with multiple envs
        self.envs = [Process(target=worker, args=(child, env, gamma, seed+idx, idx)) for idx, child in enumerate(self.children)]
        # replay buffer
        self.memory = memory
        # observe last actions
        self.obs_last_action = obs_last_action
        # record parallel episodes
        self.episodes = [[] for i in range(n_envs)]
        self.frames = [[] for _ in range(n_envs)]
        # record train return
        self.train_returns = []
        self.train_disc_returns = []
        self.train_epi_len = []
        self.train_returns_new = []
        self.train_disc_returns_new = []
        self.train_epi_len_new = []
        self.train_n_delivery_new = []
        self.train_episode_instructions = []
        self.train_episode_diagnostics = []

        self.train_len_macro_actions_new = [[] for _ in range(self.n_agent)]

        # record eval return
        self.eval_returns = []
        self.eval_disc_returns = []
        self.eval_epi_len = []
        self.eval_returns_new = []
        self.eval_disc_returns_new = []
        self.eval_epi_len_new = []
        self.eval_n_delivery_new = []
        self.eval_episode_instructions = []
        self.eval_episode_diagnostics = []
        # SHAPED return per episode (raw + INSTRUCTION_PENALTY) — wandb
        # surfaces this as Returns_With_Instruction_Shaped.
        self.train_episode_instructions_shaped = []
        self.eval_episode_instructions_shaped = []
        # Per-env shaped-reward accumulator + expiry timer (shared shape
        # with the other pg_based runners).
        self._epi_shaped_R = [0.0] * n_envs
        self.instruction_expire_steps = [0] * n_envs

        self.eval_len_macro_actions_new = [[] for _ in range(self.n_agent)]

        # record macro action stats
        self.macro_actions = [[] for _ in range(self.n_agent)]

        self.obs_history = [[[] for _ in range(self.n_agent)] for _ in range(n_envs)]
        self.time_seq = [[[] for _ in range(self.n_agent)] for _ in range(n_envs)]

        # Instruction support
        self.instruction_provider = instruction_provider
        self.instruction_embs = [[None] * self.n_agent for _ in range(n_envs)]
        self.instruction_texts_for_env = [[None] * self.n_agent for _ in range(n_envs)]
        self.instruction_refresh_interval = 100
        self.last_instruction_step = [0] * n_envs
        self.eval_compliance = []
        self.eval_compliance_per_instruction = []
        self.eval_compliance_per_agent_instruction = []
        self.instruction_counter = {}
        self.warehouse_missed_request_penalty = float(
            os.environ.get("WAREHOUSE_MISSED_REQUEST_PENALTY", "0.0")
        )
        # Per-env "fetch tool X first" priority tracking. priority_tool[i] is
        # the tool index that must be delivered first this episode (None if no
        # warehouse instruction is active). priority_satisfied[i] flips True
        # once the first delivery of any tool occurs; from that point onward
        # warehouse compliance shaping is disabled for env i.
        self.priority_tool: "list[int | None]" = [None] * n_envs
        self.priority_satisfied: "list[bool]" = [False] * n_envs

        # trigger each processor
        for env in self.envs:
            env.daemon = True
            env.start()

        for child in self.children:
            child.close()

    def run(self, eps=0.0, n_epis=1, test_mode=False, render=False):

        self._reset()

        if test_mode:
            while len(self.eval_returns) < n_epis:
                self._step(eps=eps, test_mode=test_mode, render=render)
        else:
            while self.n_epi_count < n_epis:
                self._step(eps=eps, test_mode=test_mode, render=render)

        if not test_mode:
            self.batch_idx += 1

    def close(self):
        [parent.send(('close', None)) for parent in self.parents]
        [parent.close() for parent in self.parents]
        [env.terminate() for env in self.envs]
        [env.join() for env in self.envs]

    def get_diagnostics(self):
        diagnostics = {
            'Train/Return': np.array(self.train_returns_new),
            'Train/DiscReturn': np.array(self.train_disc_returns_new),
            'Train/EpiLen': np.array(self.train_epi_len_new),
            'Train/NumDelivery': np.array(self.train_n_delivery_new),
            'Train/PickRate': np.array(self.train_n_delivery_new) * 3600 / np.array(self.train_epi_len_new),
            'Eval/Return': np.array(self.eval_returns_new),
            'Eval/DiscReturn': np.array(self.eval_disc_returns_new),
            'Eval/EpiLen': np.array(self.eval_epi_len_new),
            'Eval/NumDelivery': np.array(self.eval_n_delivery_new),
            'Eval/PickRate': np.array(self.eval_n_delivery_new) * 3600 / np.array(self.eval_epi_len_new),
        }
        for agent_idx in range(self.n_agent):
            diagnostics[f'Agent{agent_idx}/action_stats'] = np.array(self.macro_actions[agent_idx])
            diagnostics[f'Agent{agent_idx}/NMacroActions'] = np.array(self.train_len_macro_actions_new[agent_idx])

        self.train_returns_new = []
        self.train_disc_returns_new = []
        self.train_epi_len_new = []
        self.train_n_delivery_new = []

        self.eval_returns_new = []
        self.eval_disc_returns_new = []
        self.eval_epi_len_new = []
        self.eval_n_delivery_new = []

        self.macro_actions = [[] for _ in range(self.n_agent)]

        self.macro_step_count = [[0] * self.n_agent for _ in range(self.n_envs)]
        self.train_len_macro_actions_new = [[] for _ in range(self.n_agent)]
        self.eval_len_macro_actions_new = [[] for _ in range(self.n_agent)]

        return diagnostics
    
    def _step(self, eps=0.0, test_mode=False, render=False):
        # Expire active instructions when their duration window closes
        # (no-op when INSTRUCTION_DURATION_STEPS == 0).
        if INSTRUCTION_DURATION_STEPS > 0:
            for idx, parent in enumerate(self.parents):
                if (
                    self.instruction_expire_steps[idx] > 0
                    and self.step_count[idx] >= self.instruction_expire_steps[idx]
                    and any(emb is not None for emb in self.instruction_embs[idx])
                ):
                    for agent_idx in range(self.n_agent):
                        self.instruction_embs[idx][agent_idx] = None
                        self.instruction_texts_for_env[idx][agent_idx] = None
                    self.instruction_expire_steps[idx] = 0
                    self.last_valids[idx] = self.mac_done_to_tensor([1.0] * self.n_agent)
                    parent.send(("reset_macro_actions", None))
                    parent.recv()

        # Per-step stochastic instruction provision.
        # Each step, if no instruction yet, roll INSTRUCTION_PROVIDED_PROB.
        # Once provided, persists until episode end.
        # Warehouse (OSD) envs skip this block entirely: their instructions
        # are pre-assigned at episode start by _pre_assign_warehouse_instruction
        # so that the tool order is fixed before the first delivery happens.
        if self.instruction_provider is not None and not hasattr(self.env, 'n_objs'):
            provided_prob = float(os.environ.get("INSTRUCTION_PROVIDED_PROB", "0.01"))
            for idx, parent in enumerate(self.parents):
                # Only try to provide if no instruction assigned yet for this episode
                if all(emb is None for emb in self.instruction_embs[idx]):
                    # Roll the dice each step (default 1% chance per step)
                    if np.random.random() < provided_prob:
                        # Single shared instruction broadcast to all agents
                        # (matches mac_iac, the version known to learn this task).
                        inst = self.instruction_provider(idx, self.step_count[idx], agent_idx=0)
                        if inst is None:
                            inst_emb_shared = None
                            inst_text_shared = None
                        elif isinstance(inst, tuple) and len(inst) == 2:
                            inst_text_shared, emb = inst
                            inst_emb_shared = emb.detach() if isinstance(emb, torch.Tensor) else emb
                        elif isinstance(inst, torch.Tensor):
                            inst_emb_shared = inst.detach()
                            inst_text_shared = None
                        else:
                            inst_emb_shared = None
                            inst_text_shared = None

                        for agent_idx in range(self.n_agent):
                            self.instruction_embs[idx][agent_idx] = inst_emb_shared
                            self.instruction_texts_for_env[idx][agent_idx] = inst_text_shared

                        # Interrupt ongoing macro-actions when instructions arrive
                        self.last_valids[idx] = self.mac_done_to_tensor([1.0] * self.n_agent)
                        parent.send(("reset_macro_actions", None))
                        parent.recv()
                        self.last_instruction_step[idx] = self.step_count[idx]
                        if INSTRUCTION_DURATION_STEPS > 0 and inst_emb_shared is not None:
                            self.instruction_expire_steps[idx] = (
                                self.step_count[idx] + INSTRUCTION_DURATION_STEPS
                            )
                        else:
                            self.instruction_expire_steps[idx] = 0

        history, masking, time_seq = self._obs_history_to_tensor()
        actions, self.h_states = self.controller.select_action(
            self.last_obses, 
            self.h_states, 
            self.last_valids,
            self.avail_actions,
            eps=eps,
            test_mode=test_mode,
            obs_history=history,
            attn_mask=masking,
            time_seq=time_seq,
            instruction_emb=self.instruction_embs if self.instruction_provider is not None else None)

        # Save per-agent instruction embeddings before clearing
        saved_inst_embs = []
        saved_inst_texts = []
        for env_idx in range(len(self.parents)):
            inst_embs_per_agent = self.instruction_embs[env_idx]
            inst_texts_per_agent = self.instruction_texts_for_env[env_idx]
            inst_embs_per_agent = [
                emb if emb is not None else self.memory.ZERO_INSTRUCTION[0].clone()
                for emb in inst_embs_per_agent
            ]
            saved_inst_embs.append(inst_embs_per_agent)
            saved_inst_texts.append(inst_texts_per_agent)
        
        for env_idx, parent in enumerate(self.parents):
            for agent_idx, action in enumerate(actions[env_idx]):
                if action >= 0:
                    self.macro_actions[agent_idx].append(action)
                    self.macro_step_count[env_idx][agent_idx] += 1
            # send cmd to trigger env step
            parent.send(("step", actions[env_idx]))
            self.step_count[env_idx] += 1

        # collect envs' returns
        exp_to_tensor_time = 0
        construct_obs_history_time = 0
        for idx, parent in enumerate(self.parents):
            env_return_raw = parent.recv()
            warehouse_request_state = None
            if len(env_return_raw) > 10:
                warehouse_request_state = env_return_raw[10]
            t = time.time()
            env_return = self._exp_to_tensor(idx, env_return_raw, eps)
            exp_to_tensor_time += time.time() - t

            # Append instruction embeddings and texts to experience
            current_inst_embs = saved_inst_embs[idx]
            current_inst_texts = saved_inst_texts[idx]
            env_return = env_return + (current_inst_embs, current_inst_texts)

            # Apply instruction-based reward shaping
            if self.instruction_provider is not None or self.warehouse_missed_request_penalty != 0.0:
                env_return = self._inst_reward(
                    idx,
                    env_return,
                    warehouse_request_state=warehouse_request_state,
                )

            # Accumulate the shaped joint reward (mean of per-agent shaped
            # rewards) for the episode-level Returns_With_Instruction_Shaped
            # metric. env_return[4] holds the shaped per-agent rewards.
            shaped_per_agent = env_return[4]
            try:
                step_shaped_joint = float(
                    sum(rew.item() for rew in shaped_per_agent) / len(shaped_per_agent)
                )
            except Exception:
                step_shaped_joint = 0.0
            self._epi_shaped_R[idx] += step_shaped_joint

            self.episodes[idx].append(env_return)

            self.last_obses[idx] = env_return[6]
            self.avail_actions[idx] = env_return[7]
            self.last_valids[idx] = env_return[9]  # mac_v

            t = time.time()
            self._construct_obs_history(self.last_obses[idx], self.last_valids[idx], idx)
            construct_obs_history_time += time.time() - t
            if self.obs_last_action and sum(self.last_valids[idx]) > 0:
                for nth in range(self.n_agent):
                    if self.last_valids[idx][nth]:
                        self.last_actions[idx][nth] = env_return[3][nth]

            # if episode is done, add it to memory buffer
            if env_return[8][0] or self.step_count[idx] == self.max_epi_step:  # t (terminate)
                self.n_epi_count += 1

                # collect the return
                parent.send(("get_info", None))
                info = parent.recv()
                if not test_mode:
                    self.memory.scenario_cache += self.episodes[idx]
                    self.memory.flush_buf_cache()
                    self.train_returns.append(info['r'])
                    self.train_disc_returns.append(info['R'])
                    self.train_epi_len.append(info['l'])
                    self.train_returns_new.append(info['r'])
                    self.train_disc_returns_new.append(info['R'])
                    self.train_epi_len_new.append(info['l'])
                    self.train_n_delivery_new.append(info['n_delivery'])
                    for agent_idx in range(self.n_agent):
                        self.train_len_macro_actions_new[agent_idx].append(self.macro_step_count[idx][agent_idx])
                else:
                    self.eval_returns.append(info['r'])
                    self.eval_disc_returns.append(info['R'])
                    self.eval_epi_len.append(info['l'])
                    self.eval_returns_new.append(info['r'])
                    self.eval_disc_returns_new.append(info['R'])
                    self.eval_epi_len_new.append(info['l'])
                    self.eval_n_delivery_new.append(info['n_delivery'])
                    for agent_idx in range(self.n_agent):
                        self.eval_len_macro_actions_new[agent_idx].append(self.macro_step_count[idx][agent_idx])
                    # _calculate_episode_compliance returns None when no
                    # instruction-active step was tracked this episode;
                    # skip those so they don't inflate the average to 1.0.
                    if self.instruction_provider is not None and hasattr(self, '_instruction_stats'):
                        episode_compliance = self._calculate_episode_compliance(idx)
                        if episode_compliance is not None:
                            self.eval_compliance.append(episode_compliance['overall'])
                            self.eval_compliance_per_instruction.append(episode_compliance['per_instruction'])
                            self.eval_compliance_per_agent_instruction.append(
                                episode_compliance.get('per_agent_instruction', {})
                            )

                # Track per-episode instruction assignment and completion diagnostics
                epi_instruction_text = self.instruction_texts_for_env[idx][0] if self.instruction_provider is not None else None
                inst_key = epi_instruction_text if epi_instruction_text is not None else "__no_instruction__"
                self.instruction_counter[inst_key] = self.instruction_counter.get(inst_key, 0) + 1

                env_completed = bool(env_return[8][0].item()) if hasattr(env_return[8][0], "item") else bool(env_return[8][0])
                hit_horizon = (self.step_count[idx] == self.max_epi_step)
                epi_diag = {
                    'return': float(info['r']),
                    'instruction': epi_instruction_text,
                    'completed': env_completed,
                    'horizon_truncated': bool((not env_completed) and hit_horizon),
                    'episode_len': int(self.step_count[idx])
                }
                shaped_R = self._epi_shaped_R[idx]
                if not test_mode:
                    self.train_episode_instructions.append((info['r'], epi_instruction_text))
                    self.train_episode_instructions_shaped.append((shaped_R, epi_instruction_text))
                    self.train_episode_diagnostics.append(epi_diag)
                else:
                    self.eval_episode_instructions.append((info['r'], epi_instruction_text))
                    self.eval_episode_instructions_shaped.append((shaped_R, epi_instruction_text))
                    self.eval_episode_diagnostics.append(epi_diag)

                # when episode is done, immediately start a new one
                parent.send(("reset", None))
                self.last_obses[idx], h_states, self.last_actions[idx], self.last_valids[idx], self.avail_actions[idx] = parent.recv()
                assert all([h is None for h in h_states])
                
                self.h_states[idx] = torch.zeros(self.n_agent, self.controller.a_rnn_layer_size).to(self.controller.device)
                
                self.last_obses[idx] = self.obs_to_tensor(self.last_obses[idx])
                self.last_actions[idx] = self.action_to_tensor(self.last_actions[idx])
                if self.obs_last_action:
                    self.last_obses[idx] = self.rebuild_obs(self.env, self.last_obses[idx], self.last_actions[idx])
                self.last_valids[idx] = self.mac_done_to_tensor(self.last_valids[idx])
                self.avail_actions[idx] = self.avail_action_to_tensor(self.avail_actions[idx])
                self.episodes[idx] = []
                self.step_count[idx] = 0
                self._epi_shaped_R[idx] = 0.0
                self.instruction_expire_steps[idx] = 0
                self.macro_step_count[idx] = [0] * self.n_agent

                self.obs_history[idx] = [[] for _ in range(self.n_agent)]
                self.time_seq[idx] = [[] for _ in range(self.n_agent)]
                self._construct_obs_history(self.last_obses[idx], self.last_valids[idx], idx)

                # Clear per-agent instructions for new episode
                if self.instruction_provider is not None:
                    self.instruction_embs[idx] = [None] * self.n_agent
                    # Reset instruction stats for this env
                    if hasattr(self, '_instruction_stats') and idx in self._instruction_stats:
                        self._instruction_stats[idx] = {
                            'compliant': 0, 'non_compliant': 0, 'action_counts': {},
                            'instructions_by_agent': {}, 'expected_by_agent': {},
                            'per_instruction': {},
                            'per_agent_instruction': {}
                        }
                    self.instruction_texts_for_env[idx] = [None] * self.n_agent
                    self.last_instruction_step[idx] = 0
                    # Warehouse: pre-assign instruction + tool order at episode start
                    self._pre_assign_warehouse_instruction(idx, parent)

    def _reset(self):
        # send cmd to reset envs
        for parent in self.parents:
            parent.send(("reset", None))

        self.last_obses, self.h_states, self.last_actions, self.last_valids, self.avail_actions = [list(i) for i in zip(*[parent.recv() for parent in self.parents])]
        self.last_obses = [self.obs_to_tensor(obs) for obs in self.last_obses] #List[List[tensor]]
        assert all([_h is None for h in self.h_states for _h in h])
        
        self.h_states = torch.zeros(self.n_envs, self.n_agent, self.controller.a_rnn_layer_size).to(self.controller.device)
            
        #obs_history : env(idx) x agent x obs_dim
        self.obs_history = [[[] for _ in range(self.n_agent)] for _ in range(len(self.parents))]
        self.time_seq = [[[] for _ in range(self.n_agent)] for _ in range(len(self.parents))]
        self.step_count = [0] * self.n_envs
        
        for i in range(len(self.parents)):
            self._construct_obs_history(self.last_obses[i], self.last_valids[i], i)
        # self.obs_history
        self.last_actions = [self.action_to_tensor(a) for a in self.last_actions]
        if self.obs_last_action:
            # reconstruct obs to observe actions
            self.last_obses = [self.rebuild_obs(self.env, obs, a) for obs, a in zip(*[self.last_obses, self.last_actions])]
        self.last_valids = [self.mac_done_to_tensor(mac_done) for mac_done in self.last_valids]
        self.avail_actions = [self.avail_action_to_tensor(avail_action) for avail_action in self.avail_actions]

        self.n_epi_count = 0
        self.macro_step_count = [[0] * self.n_agent for _ in range(self.n_envs)]
        self.episodes = [[] for i in range(self.n_envs)]

        # Clear instruction state
        self.instruction_embs = [[None] * self.n_agent for _ in range(self.n_envs)]
        self.instruction_texts_for_env = [[None] * self.n_agent for _ in range(self.n_envs)]
        self.last_instruction_step = [0] * self.n_envs
        # Pre-assign instructions for warehouse (OSD) environments so the
        # tool order is set before the very first delivery. No-op on
        # non-warehouse envs.
        for env_idx, parent in enumerate(self.parents):
            self._pre_assign_warehouse_instruction(env_idx, parent)

    def _pre_assign_warehouse_instruction(self, env_idx, parent):
        """For warehouse (OSD) environments only: at episode start, decide
        whether this episode gets an instruction; if so, sample one, broadcast
        the embedding to every agent, and push the parsed tool order into the
        env via 'set_tool_order' before any steps are taken.

        Per-episode probability is derived from the per-step
        INSTRUCTION_PROVIDED_PROB:
            P_episode = 1 - (1 - P_step) ** max_epi_step
        so the long-run instruction frequency matches the Overcooked-style
        per-step gating used by other env families.
        """
        # Reset per-episode priority state regardless of whether an instruction
        # is sampled; absent a warehouse instruction, priority_tool stays None
        # and the shaping/terminal-bonus logic in _inst_reward becomes a no-op.
        self.priority_tool[env_idx] = None
        self.priority_satisfied[env_idx] = False

        if self.instruction_provider is None or not hasattr(self.env, 'n_objs'):
            return

        provided_prob = float(os.environ.get("INSTRUCTION_PROVIDED_PROB", "0.01"))
        per_episode_prob = 1.0 - (1.0 - provided_prob) ** self.max_epi_step
        if np.random.random() >= per_episode_prob:
            return  # no instruction this episode

        # For warehouse, the instruction is environment-level (shared by all
        # agents in that env). Pass agent_idx=None so ACAC does not force the
        # fixed_per_agent path (which would always pick agent 0's instruction).
        inst = self.instruction_provider(env_idx, 0, agent_idx=None)
        if inst is None:
            return
        if isinstance(inst, tuple) and len(inst) == 2:
            inst_text, emb = inst
            if not isinstance(emb, torch.Tensor):
                return
            inst_emb = emb.detach()
        else:
            return

        for agent_idx in range(self.n_agent):
            self.instruction_embs[env_idx][agent_idx] = inst_emb
            self.instruction_texts_for_env[env_idx][agent_idx] = inst_text

        tool_order = self._parse_warehouse_tool_order(inst_text)
        if tool_order is not None:
            self.priority_tool[env_idx] = int(tool_order[0])
            parent.send(('set_tool_order', tool_order))
            parent.recv()  # 'done'

    def _parse_warehouse_tool_order(self, instruction_text):
        """Parse an OSD instruction into an explicit tool delivery order.

        Recognised pattern (case-insensitive):
            "fetch tool X first" / "deliver tool X first"
              -> [X, ...remaining tools in natural order]

        Returns
        -------
        list[int] or None
            None if the text doesn't match the supported pattern or the env
            isn't a warehouse env.
        """
        if instruction_text is None:
            return None
        import re as _re
        match = _re.search(r'(?:fetch|deliver)\s+tool\s+(\d+)\s+first',
                           instruction_text.lower().strip())
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
        # exp (last_obs, a, r, obs, t, discnt)
        last_obs = [torch.from_numpy(o).float().view(1,-1) for o in exp[0]]
        last_mac_start = [torch.tensor(start, dtype=torch.bool).view(1,-1) for start in exp[1]]
        last_avail_actions = [torch.FloatTensor(avail_action).view(1,-1) for avail_action in self.avail_actions[env_idx]]
        a = [torch.tensor(a).view(1,-1) for a in exp[2]]
        r = [torch.tensor(r).float().view(1,-1) for r in exp[3]]
        j_r = torch.tensor(exp[4]).float().view(1,-1) 
        obs = [torch.from_numpy(o).float().view(1,-1) for o in exp[5]]
        avail_actions = [torch.FloatTensor(avail_action).view(1,-1) for avail_action in exp[6]]
        # re-construct obs if obs last action
        if self.obs_last_action:
            last_obs = self.rebuild_obs(self.env, last_obs, self.last_actions[env_idx])
            obs = self.rebuild_obs(self.env, obs, a)
        t = torch.tensor(exp[7]).float().view(1,-1)
        mac_v = [torch.tensor(v, dtype=torch.bool).view(1,-1) for v in exp[8]]
        j_mac_v = torch.tensor(exp[9], dtype=torch.bool).view(1,-1)
        exp_v = [torch.tensor([1.0]).view(1,-1)] * self.n_agent
        return (last_obs, last_mac_start, last_avail_actions, a, r, j_r, obs, avail_actions, t, mac_v, j_mac_v, exp_v)
    
    def _construct_obs_history(self, last_obs, last_valid, env_idx) :
        for agent, (obs, valid) in enumerate(zip(last_obs, last_valid)):
            if valid:
                self.time_seq[env_idx][agent].append(self.step_count[env_idx]+1)
                if type(torch.zeros([0])) != type(obs) :
                    obs = torch.from_numpy(obs).float()
                self.obs_history[env_idx][agent].append(obs)
                if len(self.obs_history[env_idx][agent]) > self.trace_len :
                    _ = self.obs_history[env_idx][agent].pop(0)
                    _ = self.time_seq[env_idx][agent].pop(0)

    def _obs_history_to_tensor(self):
        """Convert per-(env, agent) obs history lists into tensor(s) the
        controller / actor can consume.

        Overcooked + BoxPushing have a uniform obs_dim across agents, so we
        return a single (n_envs, n_agent, trace_len, obs_dim) tensor (the
        original behavior, kept verbatim for back-compat).

        Warehouse / OSD has heterogeneous obs sizes per agent (Turtlebots vs
        Fetch — see osd_ma_*.obs_size), so we instead return a *list* of
        per-agent tensors of shape (n_envs, trace_len, obs_dim_i).
        controller.select_action handles both forms.
        """
        per_agent_dims = [self.obs_history[0][i][0].shape[-1] for i in range(self.n_agent)]
        homogeneous = all(d == per_agent_dims[0] for d in per_agent_dims)

        masking = torch.zeros(self.n_envs, self.n_agent, self.trace_len)
        time_seq = torch.zeros(self.n_envs, self.n_agent, self.trace_len)

        if homogeneous:
            obs_dim = per_agent_dims[0]
            history = torch.zeros(self.n_envs, self.n_agent, self.trace_len, obs_dim)
            for env_idx in range(self.n_envs):
                for i in range(self.n_agent):
                    seq_len = len(self.obs_history[env_idx][i])
                    pad_len = self.trace_len - seq_len
                    history[env_idx][i][pad_len:] = torch.cat(self.obs_history[env_idx][i], dim=0)
                    masking[env_idx][i][pad_len:] = 1
                    time_seq[env_idx][i][pad_len:] = torch.LongTensor(self.time_seq[env_idx][i])
        else:
            history = []
            for i in range(self.n_agent):
                agent_obs_dim = per_agent_dims[i]
                agent_hist = torch.zeros(self.n_envs, self.trace_len, agent_obs_dim)
                for env_idx in range(self.n_envs):
                    seq_len = len(self.obs_history[env_idx][i])
                    pad_len = self.trace_len - seq_len
                    agent_hist[env_idx, pad_len:] = torch.cat(self.obs_history[env_idx][i], dim=0)
                    masking[env_idx][i][pad_len:] = 1
                    time_seq[env_idx][i][pad_len:] = torch.LongTensor(self.time_seq[env_idx][i])
                history.append(agent_hist)

        return history, masking, time_seq

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

    def _inst_reward(self, env_idx, env_return, warehouse_request_state=None):
        """Apply reward shaping based on instruction compliance for individual agents."""
        # Experience tuple: (..., inst_embs, inst_texts)
        # Unpack: the last two elements are instructions
        base_return = env_return[:-2]
        inst_embs = env_return[-2]
        inst_texts = env_return[-1]

        # base_return: (last_obs, last_mac_start, last_avail_actions, a, r, j_r, obs, avail_actions, t, mac_v, j_mac_v, exp_v)
        a = base_return[3]   # actions per agent
        r = [reward.clone() if isinstance(reward, torch.Tensor) else torch.tensor(reward).float().view(1, -1)
             for reward in base_return[4]]   # rewards per agent (mutable copy)
        j_r = base_return[5]  # joint reward

        shaped_rewards = [reward.clone() for reward in r]

        if inst_texts is not None and not all(text is None for text in inst_texts):
            if not hasattr(self, '_instruction_stats'):
                self._instruction_stats = {}
            if env_idx not in self._instruction_stats:
                self._instruction_stats[env_idx] = {
                    'compliant': 0, 'non_compliant': 0, 'action_counts': {},
                    'instructions_by_agent': {}, 'expected_by_agent': {},
                    'per_instruction': {},  # {instruction_text: {'compliant': int, 'non_compliant': int}}
                    'per_agent_instruction': {}  # {(agent_idx, instruction_text): {'compliant': int, 'non_compliant': int}}
                }

            for agent_idx, agent_action in enumerate(a):
                action_value = agent_action.item()
                agent_instruction_text = inst_texts[agent_idx] if agent_idx < len(inst_texts) else None

                if agent_instruction_text is None:
                    continue

                expected_behavior = self._get_expected_macro_action(agent_instruction_text, agent_idx=agent_idx)
                self._instruction_stats[env_idx]['instructions_by_agent'][agent_idx] = agent_instruction_text
                if expected_behavior:
                    self._instruction_stats[env_idx]['expected_by_agent'][agent_idx] = expected_behavior

                if action_value not in self._instruction_stats[env_idx]['action_counts']:
                    self._instruction_stats[env_idx]['action_counts'][action_value] = 0
                self._instruction_stats[env_idx]['action_counts'][action_value] += 1

                if expected_behavior is None:
                    continue

                # Ensure per-instruction tracking exists
                if agent_instruction_text not in self._instruction_stats[env_idx]['per_instruction']:
                    self._instruction_stats[env_idx]['per_instruction'][agent_instruction_text] = {
                        'compliant': 0, 'non_compliant': 0
                    }
                per_inst = self._instruction_stats[env_idx]['per_instruction'][agent_instruction_text]
                per_agent_key = (agent_idx, agent_instruction_text)
                if per_agent_key not in self._instruction_stats[env_idx]['per_agent_instruction']:
                    self._instruction_stats[env_idx]['per_agent_instruction'][per_agent_key] = {
                        'compliant': 0, 'non_compliant': 0
                    }
                per_agent_inst = self._instruction_stats[env_idx]['per_agent_instruction'][per_agent_key]

                is_warehouse_inst = self._parse_warehouse_tool_order(agent_instruction_text) is not None
                wh_penalty = -50.0
                wh_bonus = 20.0

                # Once the priority tool has been delivered (or some other
                # delivery has occurred), the "fetch tool X first" instruction
                # is no longer applicable: the agent must fetch other tools to
                # complete remaining requests. Skip dense shaping AND
                # compliance counting for warehouse instructions in that case.
                if is_warehouse_inst and self.priority_satisfied[env_idx]:
                    continue

                # Magnitudes match mac_iac: 0 bonus on compliance, -50 penalty
                # on non-compliance (non-warehouse). Warehouse path keeps its
                # tuned wh_penalty / wh_bonus + ±100 first-delivery shot.
                if 'prohibited_actions' in expected_behavior:
                    prohibited = expected_behavior['prohibited_actions']
                    if action_value in prohibited:
                        penalty = wh_penalty if is_warehouse_inst else INSTRUCTION_PENALTY
                        shaped_rewards[agent_idx] = shaped_rewards[agent_idx] + torch.tensor(penalty).float().view(1, -1)
                        self._instruction_stats[env_idx]['non_compliant'] += 1
                        per_inst['non_compliant'] += 1
                        per_agent_inst['non_compliant'] += 1
                    else:
                        if is_warehouse_inst:
                            shaped_rewards[agent_idx] = shaped_rewards[agent_idx] + torch.tensor(wh_bonus).float().view(1, -1)
                        self._instruction_stats[env_idx]['compliant'] += 1
                        per_inst['compliant'] += 1
                        per_agent_inst['compliant'] += 1
                elif 'allowed_actions' in expected_behavior:
                    allowed = expected_behavior['allowed_actions']
                    if action_value in allowed:
                        if is_warehouse_inst:
                            shaped_rewards[agent_idx] = shaped_rewards[agent_idx] + torch.tensor(wh_bonus).float().view(1, -1)
                        self._instruction_stats[env_idx]['compliant'] += 1
                        per_inst['compliant'] += 1
                        per_agent_inst['compliant'] += 1
                    else:
                        penalty = wh_penalty if is_warehouse_inst else INSTRUCTION_PENALTY
                        shaped_rewards[agent_idx] = shaped_rewards[agent_idx] + torch.tensor(penalty).float().view(1, -1)
                        self._instruction_stats[env_idx]['non_compliant'] += 1
                        per_inst['non_compliant'] += 1
                        per_agent_inst['non_compliant'] += 1

        shaped_rewards = self._apply_warehouse_priority_terminal(
            env_idx, shaped_rewards, warehouse_request_state,
        )

        shaped_rewards = self._apply_warehouse_request_penalty(
            shaped_rewards,
            warehouse_request_state,
        )

        # Reconstruct tuple with shaped rewards
        new_return = (base_return[0], base_return[1], base_return[2], base_return[3],
                      shaped_rewards, base_return[5], base_return[6], base_return[7],
                      base_return[8], base_return[9], base_return[10], base_return[11],
                      inst_embs, inst_texts)
        return new_return

    def _apply_warehouse_priority_terminal(self, env_idx, shaped_rewards, warehouse_request_state):
        """Apply a one-shot terminal credit at the moment of the first delivery.

        While priority_satisfied[env_idx] is False, watch the human-side
        'currently_received_tools' signal: the first step it becomes non-empty
        marks the first delivery this episode. Compare the delivered tool
        against priority_tool[env_idx] and credit the Fetch agent accordingly:
            +100 if the delivered tool matches the prioritized one,
            -100 otherwise.
        After this fires, priority_satisfied flips True, disabling further
        warehouse compliance shaping for the rest of the episode.
        """
        if warehouse_request_state is None:
            return shaped_rewards
        if self.priority_tool[env_idx] is None or self.priority_satisfied[env_idx]:
            return shaped_rewards

        received = warehouse_request_state.get('currently_received_tools', [])
        if not received:
            return shaped_rewards

        first_tool = int(received[0])
        bonus = 100.0 if first_tool == self.priority_tool[env_idx] else -100.0
        fetch_idx = self.n_agent - 1
        shaped_rewards[fetch_idx] = shaped_rewards[fetch_idx] + torch.tensor(bonus).float().view(1, -1)
        self.priority_satisfied[env_idx] = True
        return shaped_rewards

    def _apply_warehouse_request_penalty(self, shaped_rewards, warehouse_request_state):
        """Apply warehouse-specific shaping when humans are waiting for tools
        but Fetch is not grabbing a requested one."""
        if self.warehouse_missed_request_penalty == 0.0:
            return shaped_rewards
        if warehouse_request_state is None or not hasattr(self.env, 'n_objs'):
            return shaped_rewards

        pending_tools = {int(t) for t in warehouse_request_state.get('pending_tools', [])}
        if not pending_tools:
            return shaped_rewards

        fetch_found_objs = {int(t) for t in warehouse_request_state.get('fetch_found_objs', [])}
        if pending_tools & fetch_found_objs:
            return shaped_rewards

        fetch_action_idx = warehouse_request_state.get('fetch_action_idx', None)
        n_turtlebots = self.n_agent - 1
        look_action_base = 1 + n_turtlebots  # Wait_Request + all Pass_Obj_T*

        searching_requested_tool = False
        if fetch_action_idx is not None:
            fetch_action_idx = int(fetch_action_idx)
            if look_action_base <= fetch_action_idx < look_action_base + self.env.n_objs:
                search_tool_idx = fetch_action_idx - look_action_base
                searching_requested_tool = search_tool_idx in pending_tools

        if searching_requested_tool:
            return shaped_rewards

        fetch_idx = self.n_agent - 1
        shaped_rewards[fetch_idx] = shaped_rewards[fetch_idx] + torch.tensor(
            self.warehouse_missed_request_penalty
        ).float().view(1, -1)
        return shaped_rewards

    def _get_expected_macro_action(self, instruction_text, agent_idx=None):
        """Map instruction text to expected behavior (allowed or prohibited actions)."""
        instruction_lower = instruction_text.lower().strip()

        # Box Pushing macro-action indices
        GT_SMALL_BOX_0 = 0
        GT_SMALL_BOX_1 = 1
        GT_BIG_BOX_SPOT_0 = 2
        GT_BIG_BOX_SPOT_1 = 3
        PUSH = 4
        TURN_LEFT = 5
        TURN_RIGHT = 6
        STAY_BP = 7

        # Per-agent: "go to small box" routes agent 0 -> small box 0,
        # agent 1 -> small box 1.
        if instruction_lower in ["go to small box", "small box"]:
            if agent_idx == 0:
                return {'allowed_actions': [GT_SMALL_BOX_0]}
            elif agent_idx == 1:
                return {'allowed_actions': [GT_SMALL_BOX_1]}
            else:
                return None

        # Box Pushing instructions
        if instruction_lower in ["big_box_spot_0", "go to big box spot 0", "big box spot 0"]:
            return {'allowed_actions': [GT_BIG_BOX_SPOT_0]}
        elif instruction_lower in ["big_box_spot_1", "go to big box spot 1", "big box spot 1"]:
            return {'allowed_actions': [GT_BIG_BOX_SPOT_1]}
        elif instruction_lower in ["small_box_0", "go to small box 0", "small box 0"]:
            return {'allowed_actions': [GT_SMALL_BOX_0]}
        elif instruction_lower in ["small_box_1", "go to small box 1", "small box 1"]:
            return {'allowed_actions': [GT_SMALL_BOX_1]}
        elif instruction_lower == "push":
            return {'allowed_actions': [PUSH]}
        elif instruction_lower in ["don't go to small box 0", "avoid small box 0"]:
            return {'prohibited_actions': [GT_SMALL_BOX_0]}
        elif instruction_lower in ["don't go to small box 1", "avoid small box 1"]:
            return {'prohibited_actions': [GT_SMALL_BOX_1]}
        elif instruction_lower in ["don't go to any small box", "avoid small boxes", "avoid all small boxes"]:
            return {'prohibited_actions': [GT_SMALL_BOX_0, GT_SMALL_BOX_1]}
        elif instruction_lower in ["don't push", "stop pushing the box", "stop pushing", "stop push"]:
            return {'prohibited_actions': [PUSH]}

        # Overcooked macro-action indices (must match environment macroActionName)
        # All maps: ["stay", "get tomato", "get lettuce", "get onion", "get peas",
        #            "get plate 1", "get plate 2", "go to knife 1", "go to knife 2",
        #            "deliver", "chop", ...]
        STAY_OC = 0
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

        if instruction_lower == "stay":
            return {'allowed_actions': [STAY_OC]}
        elif instruction_lower in ["get tomato", "i will get the tomato"]:
            return {'allowed_actions': [GET_TOMATO]}
        elif instruction_lower == "get lettuce":
            return {'allowed_actions': [GET_LETTUCE]}
        elif instruction_lower == "get onion":
            return {'allowed_actions': [GET_ONION]}
        elif instruction_lower in ["get plate 1", "get plate"]:
            return {'allowed_actions': [GET_PLATE_1]}
        elif instruction_lower == "go to knife 1":
            return {'allowed_actions': [GO_TO_KNIFE_1]}
        elif instruction_lower == "go to knife 2":
            return {'allowed_actions': [GO_TO_KNIFE_2]}
        elif instruction_lower == "deliver":
            return {'allowed_actions': [DELIVER]}
        elif instruction_lower == "chop":
            return {'allowed_actions': [CHOP]}
        elif instruction_lower == "let me do all the chopping":
            return {'prohibited_actions': [GO_TO_KNIFE_1, GO_TO_KNIFE_2, CHOP]}
        # Map D oven prohibitions
        elif instruction_lower in ["don't use the left oven", "don't go to the left oven", "avoid left oven"]:
            return {'prohibited_actions': [USE_LEFT_OVEN]}
        elif instruction_lower in ["don't use the right oven", "don't go to the right oven", "avoid right oven"]:
            return {'prohibited_actions': [USE_RIGHT_OVEN]}
        elif instruction_lower in ["don't touch the tomato", "don't touch tomato", "don't get tomato"]:
            return {'prohibited_actions': [GET_TOMATO]}
        elif instruction_lower in ["don't touch the lettuce", "don't touch lettuce", "don't get lettuce"]:
            return {'prohibited_actions': [GET_LETTUCE]}
        elif instruction_lower in ["don't touch the onion", "don't get onion"]:
            return {'prohibited_actions': [GET_ONION]}
        elif instruction_lower in ["don't deliver", "i will deliver it myself", "i will deliver"]:
            return {'prohibited_actions': [DELIVER]}
        elif instruction_lower == "don't chop":
            return {'prohibited_actions': [CHOP]}
        elif instruction_lower in ["don't go to knife 1", "avoid knife 1", "don't use the right cutting board", "don't go to the right"]:
            return {'prohibited_actions': [GO_TO_KNIFE_1]}
        elif instruction_lower in ["don't go to knife 2", "avoid knife 2", "don't use the left cutting board"]:
            return {'prohibited_actions': [GO_TO_KNIFE_2]}

        # ------------------------------------------------------------------
        # Warehouse (OSD) instructions – change the tool delivery priority.
        #
        # Macro-action layout for the Fetch robot (agent index = n_agent-1):
        #   0 : Wait_Request
        #   1 : Pass_Obj_T0
        #   2 : Pass_Obj_T1
        #   3+i : Look_For_obj_i  (i = 0 ... n_objs-1)
        #
        # Compliance is only enforced for the Fetch robot. Turtlebot agents
        # (indices 0 ... n_agent-2) return None so they are not penalized.
        # ------------------------------------------------------------------
        import re as _re

        # "fetch tool X first" / "deliver tool X first"
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

        # "don't fetch tool X" / "don't search for tool X"
        _wh_neg = _re.search(r"don't\s+(?:fetch|search\s+for)\s+tool\s+(\d+)", instruction_lower)
        if _wh_neg:
            tool_idx = int(_wh_neg.group(1))
            if agent_idx is not None and agent_idx != self.n_agent - 1:
                return None
            return {'prohibited_actions': [3 + tool_idx]}

        return None

    def _calculate_episode_compliance(self, env_idx):
        """Calculate instruction compliance rate for a completed episode.

        Compliance is only meaningful when the instruction was active for
        at least one tracked step. Episodes with no instruction-active
        tracking return None so the caller can skip them — the previous
        behavior padded with 1.0 which inflated the rolling average.

        Per-instruction sub-buckets that ended up with zero tracked steps
        are also dropped from the dict instead of being filled with 1.0.
        """
        if env_idx not in self._instruction_stats:
            return None

        stats = self._instruction_stats[env_idx]
        total = stats['compliant'] + stats['non_compliant']
        if total == 0:
            return None

        result = {
            'overall': stats['compliant'] / total,
            'per_instruction': {},
            'per_agent_instruction': {},
        }

        for inst_text, inst_stats in stats.get('per_instruction', {}).items():
            inst_total = inst_stats['compliant'] + inst_stats['non_compliant']
            if inst_total > 0:
                result['per_instruction'][inst_text] = inst_stats['compliant'] / inst_total

        for key, inst_stats in stats.get('per_agent_instruction', {}).items():
            inst_total = inst_stats['compliant'] + inst_stats['non_compliant']
            if inst_total > 0:
                result['per_agent_instruction'][key] = inst_stats['compliant'] / inst_total

        return result
