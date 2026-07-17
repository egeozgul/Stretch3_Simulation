import os
import numpy as np
import random
import torch
import torch.nn.functional as F

from multiprocessing import Process, Pipe

# Tunable instruction-shaping knobs (mirror mac_cac / mac_iac so the
# same env-vars work across all four pg_based runners).
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
                R += gamma**step * sum(reward) / env.n_agent
                step += 1
            
            elif cmd == 'get_return':
                child.send(R)

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
                # Force all agents to resample macro-actions (interrupt ongoing actions).
                # Env-agnostic: handles Overcooked (env.env.macroAgent) and
                # warehouse/BoxPushing (env.agents.cur_action_done). No-op on
                # envs that expose neither.
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
    Environment runner for MAC-IAICC with instruction support.
    Runs multiple environments in parallel using subprocesses.
    """

    def __init__(self, env, n_envs, controller, memory, env_terminate_step, gamma, seed, obs_last_action=False,
                 instruction_provider=None, writer=None, log=False):
        
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
        # Per-episode instruction tracking: list of (return, instruction_text_or_None)
        self.train_episode_instructions = []
        self.eval_episode_instructions = []
        # Same shape, but holding the SHAPED return (raw + INSTRUCTION_PENALTY).
        # Surfaces as Returns_With_Instruction_Shaped in wandb.
        self.train_episode_instructions_shaped = []
        self.eval_episode_instructions_shaped = []
        # Per-env shaped-reward accumulator + expiry timer.
        self._epi_shaped_R = [0.0] * n_envs
        self.instruction_expire_steps = [0] * n_envs
        # log for tensorboard
        self.writer = writer
        self.log = log
        # optional instruction provider: callable(env_idx:int, step:int, agent_idx:int) -> str | Tensor | None
        self.instruction_provider = instruction_provider
        # Store per-agent instructions: instruction_embs[env_idx][agent_idx]
        self.instruction_embs = [[None] * self.n_agent for _ in range(n_envs)]
        self.instruction_texts_for_env = [[None] * self.n_agent for _ in range(n_envs)]
        # Track when instructions were last updated (for periodic refresh)
        self.instruction_refresh_interval = 100  # Refresh every 100 timesteps
        self.last_instruction_step = [0] * n_envs  # Track last refresh timestep per env
        # Per-env "fetch tool X first" priority tracking (warehouse / OSD).
        self.priority_tool: "list[int | None]" = [None] * n_envs
        self.priority_satisfied: "list[bool]" = [False] * n_envs

        # trigger each processor
        for env in self.envs:
            env.daemon = True
            env.start()

        for child in self.children:
            child.close()

    def run(self, eps=0.0, n_epis=1, test_mode=False):

        self._reset()

        if test_mode:
            # Reset compliance tracking for evaluation
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

            # Expire the active instruction when its duration window
            # closes. INSTRUCTION_DURATION_STEPS=0 (default) → no-op.
            if (
                INSTRUCTION_DURATION_STEPS > 0
                and self.instruction_expire_steps[idx] > 0
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
            # Once provided, the instruction persists until episode end.
            # All agents in the same env receive THE SAME instruction (mirrors
            # mac_iac, which is the version known to learn this task).
            # Warehouse (OSD) envs skip this block entirely: their instructions
            # are pre-assigned at episode start by _pre_assign_warehouse_instruction.
            if self.instruction_provider is not None and not hasattr(self.env, 'n_objs'):
                provided_prob = float(os.environ.get("INSTRUCTION_PROVIDED_PROB", "0.01"))
                if all(emb is None for emb in self.instruction_embs[idx]):
                    if np.random.random() < provided_prob:
                        inst = self.instruction_provider(idx, self.step_count[idx], agent_idx=0)
                        if inst is None:
                            inst_emb_shared = None
                            inst_text_shared = None
                        elif isinstance(inst, tuple) and len(inst) == 2:
                            inst_text_shared, emb = inst
                            inst_emb_shared = emb.detach() if isinstance(emb, torch.Tensor) else None
                        elif isinstance(inst, torch.Tensor):
                            inst_emb_shared = inst.detach()
                            inst_text_shared = None
                        else:
                            inst_emb_shared = None
                            inst_text_shared = None

                        for agent_idx in range(self.n_agent):
                            self.instruction_embs[idx][agent_idx] = inst_emb_shared
                            self.instruction_texts_for_env[idx][agent_idx] = inst_text_shared

                        # When instructions arrive, interrupt ongoing macro-actions so
                        # the agents replan against the new conditional immediately.
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
            
            # Pass list of per-agent instructions to select_action
            actions, self.h_states[idx] = self.controller.select_action(self.last_obses[idx], 
                                                                        self.h_states[idx], 
                                                                        self.last_valids[idx],
                                                                        self.avail_actions[idx],
                                                                        eps=eps,
                                                                        test_mode=test_mode,
                                                                        instruction_emb=self.instruction_embs[idx])
            # send cmd to trigger env step
            parent.send(("step", actions))
            self.step_count[idx] += 1

        # Save per-agent instructions before collecting returns
        saved_inst_embs = []
        saved_inst_texts = []
        for idx in range(len(self.parents)):
            inst_embs_per_agent = self.instruction_embs[idx] if idx < len(self.instruction_embs) else [None] * self.n_agent
            inst_texts_per_agent = self.instruction_texts_for_env[idx] if idx < len(self.instruction_texts_for_env) else [None] * self.n_agent
            
            # Replace None embeddings with zero tensors using memory's ZERO_INSTRUCTION (correctly sized)
            inst_embs_per_agent = [
                emb if emb is not None else self.memory.ZERO_INSTRUCTION[agent_i].clone()
                for agent_i, emb in enumerate(inst_embs_per_agent)
            ]
            
            saved_inst_embs.append(inst_embs_per_agent)
            saved_inst_texts.append(inst_texts_per_agent)

        # collect envs' returns
        for idx, parent in enumerate(self.parents):
            env_return = parent.recv()
            warehouse_request_state = env_return[10] if len(env_return) >= 11 else None
            env_return = env_return[:10]
            current_inst_embs = saved_inst_embs[idx]
            current_inst_texts = saved_inst_texts[idx]

            env_return = self._exp_to_tensor(idx, env_return, eps)
            # Append instruction embeddings and texts at the end
            env_return = env_return + (current_inst_embs, current_inst_texts)

            # Apply instruction-based reward shaping BEFORE storing in replay buffer
            if self.instruction_provider is not None:
                env_return = self._inst_reward(idx, env_return, warehouse_request_state=warehouse_request_state)

            # Accumulate the shaped joint reward (mean of per-agent shaped
            # rewards) for the episode-level Returns_With_Instruction_Shaped
            # metric. env_return[4] is the per-agent shaped reward list
            # AFTER _inst_reward applied INSTRUCTION_PENALTY / bonus.
            shaped_per_agent = env_return[4]
            try:
                step_shaped_joint = float(
                    sum(r.item() for r in shaped_per_agent) / len(shaped_per_agent)
                )
            except Exception:
                step_shaped_joint = 0.0
            self._epi_shaped_R[idx] += step_shaped_joint

            # Store the experience with shaped rewards in replay buffer
            self.episodes[idx].append(env_return)

            self.last_obses[idx] = env_return[6]
            self.avail_actions[idx] = env_return[7]
            self.last_valids[idx] = env_return[-5]  # Adjusted for instruction tuple
            if self.obs_last_action and sum(self.last_valids[idx]) > 0:
                for nth in range(self.n_agent):
                    if self.last_valids[idx][nth]:
                        self.last_actions[idx][nth] = env_return[3][nth]

            # if episode is done, add it to memory buffer
            if env_return[-6][0] or self.step_count[idx] == self.max_epi_step:
                self.n_epi_count += 1
                # collect the return
                parent.send(("get_return", None))
                R = parent.recv()

                # Print instruction statistics for this episode
                if self.instruction_provider is not None and hasattr(self, '_instruction_stats') and idx in self._instruction_stats:
                    stats = self._instruction_stats[idx]
                    total_actions = stats['compliant'] + stats['non_compliant']
                    compliance_rate = stats['compliant'] / max(total_actions, 1) * 100
                    
                    # Only print periodically to avoid spam (every 10 episodes)
                    if self.n_epi_count % 10 == 0:
                        print(f"\n{'='*60}")
                        print(f"Env {idx} | Episode {self.n_epi_count} | Return: {R:.2f}")
                        if 'instructions_by_agent' in stats:
                            for agent_idx, inst_text in stats['instructions_by_agent'].items():
                                expected = stats['expected_by_agent'].get(agent_idx, {})
                                print(f"Agent {agent_idx} Instruction: '{inst_text}' -> Expected: {expected}")
                        print(f"Compliance: {stats['compliant']}/{total_actions} ({compliance_rate:.1f}%)")
                        print(f"Action distribution: {dict(sorted(stats['action_counts'].items()))}")
                        print(f"{'='*60}\n")
                    
                    # Reset stats for next episode
                    self._instruction_stats[idx] = {
                        'compliant': 0, 'non_compliant': 0, 'action_counts': {},
                        'instructions_by_agent': {}, 'expected_by_agent': {},
                        'per_instruction': {}
                    }

                # Track instruction compliance for evaluation episodes
                # _calculate_episode_compliance returns None for episodes
                # with no instruction-active steps; skip those so they
                # don't artificially inflate the average to 1.0.
                if test_mode and self.instruction_provider is not None:
                    episode_compliance = self._calculate_episode_compliance(idx)
                    if episode_compliance is not None:
                        self.eval_compliance.append(episode_compliance['overall'])
                        self.eval_compliance_per_instruction.append(episode_compliance['per_instruction'])
                        self.eval_compliance_per_agent_instruction.append(episode_compliance['per_agent_instruction'])

                # Determine which instruction was active this episode (agent 0's instruction)
                epi_instruction_text = self.instruction_texts_for_env[idx][0] if self.instruction_provider is not None else None

                shaped_R = self._epi_shaped_R[idx]

                if not test_mode:
                    self.memory.scenario_cache += self.episodes[idx]
                    self.memory.flush_buf_cache()
                    self.train_returns.append(R)
                    self.train_episode_instructions.append((R, epi_instruction_text))
                    self.train_episode_instructions_shaped.append((shaped_R, epi_instruction_text))
                else:
                    self.eval_returns.append(R)
                    self.eval_episode_instructions.append((R, epi_instruction_text))
                    self.eval_episode_instructions_shaped.append((shaped_R, epi_instruction_text))

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
                self.step_count[idx] = 0
                self._epi_shaped_R[idx] = 0.0
                self.instruction_expire_steps[idx] = 0

                # Clear per-agent instructions for the new episode
                if self.instruction_provider is not None:
                    self.instruction_embs[idx] = [None] * self.n_agent
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
        self.last_actions = [self.action_to_tensor(a) for a in self.last_actions]
        if self.obs_last_action:
            # reconstruct obs to observe actions
            self.last_obses = [self.rebuild_obs(self.env, obs, a) for obs, a in zip(*[self.last_obses, self.last_actions])]
        self.last_valids = [self.mac_done_to_tensor(mac_done) for mac_done in self.last_valids]
        self.avail_actions = [self.avail_action_to_tensor(avail_action) for avail_action in self.avail_actions]

        self.n_epi_count = 0
        self.step_count = [0] * self.n_envs
        self.episodes = [[] for i in range(self.n_envs)]
        # Clear instructions at reset
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
        # is sampled.
        self.priority_tool[env_idx] = None
        self.priority_satisfied[env_idx] = False

        if self.instruction_provider is None or not hasattr(self.env, 'n_objs'):
            return

        provided_prob = float(os.environ.get("INSTRUCTION_PROVIDED_PROB", "0.01"))
        per_episode_prob = 1.0 - (1.0 - provided_prob) ** self.max_epi_step
        if np.random.random() >= per_episode_prob:
            return  # no instruction this episode

        inst = self.instruction_provider(env_idx, 0, agent_idx=0)
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
        # exp (last_obs, last_mac_start, action, accu_rewards, accu_joint_reward, obs, avail_actions, terminate, valid, max_valid)
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
        """
        Apply reward shaping based on instruction compliance for individual agents.

        mac_iaicc experience tuple (14 items):
        (last_obs, last_mac_start, last_avail_actions, a, r, j_r, obs, avail_actions, t, mac_v, j_mac_v, exp_v, inst_embs, inst_texts)
        """
        last_obs, last_mac_start, last_avail_actions, a, r, j_r, obs, avail_actions, t, mac_v, j_mac_v, exp_v, inst_embs, inst_texts = env_return

        if inst_texts is None or all(text is None for text in inst_texts):
            return env_return

        # Track compliance for statistics (per environment)
        if not hasattr(self, '_instruction_stats'):
            self._instruction_stats = {}
        if env_idx not in self._instruction_stats:
            self._instruction_stats[env_idx] = {
                'compliant': 0, 'non_compliant': 0, 'action_counts': {},
                'instructions_by_agent': {}, 'expected_by_agent': {},
                'per_instruction': {}
            }

        # Apply reward shaping for each agent individually based on their instruction
        shaped_rewards = []
        for agent_idx, agent_action in enumerate(a):
            action_value = agent_action.item()
            
            agent_instruction_text = inst_texts[agent_idx] if agent_idx < len(inst_texts) else None
            
            if agent_instruction_text is None:
                shaped_rewards.append(r[agent_idx])
                continue
            
            expected_behavior = self._get_expected_macro_action(agent_instruction_text, agent_idx=agent_idx)
            
            # Track instruction for this agent
            self._instruction_stats[env_idx]['instructions_by_agent'][agent_idx] = agent_instruction_text
            if expected_behavior:
                self._instruction_stats[env_idx]['expected_by_agent'][agent_idx] = expected_behavior
            
            if action_value not in self._instruction_stats[env_idx]['action_counts']:
                self._instruction_stats[env_idx]['action_counts'][action_value] = 0
            self._instruction_stats[env_idx]['action_counts'][action_value] += 1

            if agent_instruction_text not in self._instruction_stats[env_idx]['per_instruction']:
                self._instruction_stats[env_idx]['per_instruction'][agent_instruction_text] = {
                    'compliant': 0, 'non_compliant': 0
                }
            per_inst_stats = self._instruction_stats[env_idx]['per_instruction'][agent_instruction_text]
            
            if expected_behavior is None:
                shaped_rewards.append(r[agent_idx])
                continue
            
            is_warehouse_inst = self._parse_warehouse_tool_order(agent_instruction_text) is not None
            wh_penalty = -25
            wh_bonus = 10

            # After first delivery, the "fetch X first" instruction no longer
            # applies; skip dense shaping + compliance counting so the agent
            # is free to fetch other tools to complete remaining requests.
            if is_warehouse_inst and self.priority_satisfied[env_idx]:
                shaped_rewards.append(r[agent_idx])
                continue

            # Magnitudes match mac_iac (which learns this task). The strong
            # -50 penalty is safe under the chain-break / value-cancellation
            # framework in mac_iaicc/learner.py::_get_discounted_return:
            # instruction-segment returns are isolated from the task critic so
            # this spike does not pollute task learning.
            if 'prohibited_actions' in expected_behavior:
                prohibited = expected_behavior['prohibited_actions']
                if action_value in prohibited:
                    penalty = wh_penalty if is_warehouse_inst else INSTRUCTION_PENALTY
                    shaped_reward = r[agent_idx].item() + penalty
                    self._instruction_stats[env_idx]['non_compliant'] += 1
                    per_inst_stats['non_compliant'] += 1
                else:
                    bonus = wh_bonus if is_warehouse_inst else 0
                    shaped_reward = r[agent_idx].item() + bonus
                    self._instruction_stats[env_idx]['compliant'] += 1
                    per_inst_stats['compliant'] += 1

            elif 'allowed_actions' in expected_behavior:
                allowed = expected_behavior['allowed_actions']
                if action_value in allowed:
                    bonus = wh_bonus if is_warehouse_inst else 0
                    shaped_reward = r[agent_idx].item() + bonus
                    self._instruction_stats[env_idx]['compliant'] += 1
                    per_inst_stats['compliant'] += 1
                else:
                    penalty = wh_penalty if is_warehouse_inst else INSTRUCTION_PENALTY
                    shaped_reward = r[agent_idx].item() + penalty
                    self._instruction_stats[env_idx]['non_compliant'] += 1
                    per_inst_stats['non_compliant'] += 1
            else:
                shaped_reward = r[agent_idx].item()

            shaped_rewards.append(torch.tensor(shaped_reward).float().view(1,-1))

        shaped_rewards = self._apply_warehouse_priority_terminal(
            env_idx, shaped_rewards, warehouse_request_state,
        )

        return (last_obs, last_mac_start, last_avail_actions, a, shaped_rewards, j_r, obs, avail_actions, t, mac_v, j_mac_v, exp_v, inst_embs, inst_texts)

    def _apply_warehouse_priority_terminal(self, env_idx, shaped_rewards, warehouse_request_state):
        """One-shot ±50 credit to Fetch on the first delivery of the episode.

        +50 if the first delivered tool matches priority_tool[env_idx], -50
        otherwise. After firing, priority_satisfied[env_idx] flips True and
        dense shaping is disabled for the rest of the episode.
        """
        if warehouse_request_state is None:
            return shaped_rewards
        if self.priority_tool[env_idx] is None or self.priority_satisfied[env_idx]:
            return shaped_rewards

        received = warehouse_request_state.get('currently_received_tools', [])
        if not received:
            return shaped_rewards

        first_tool = int(received[0])
        bonus = 50.0 if first_tool == self.priority_tool[env_idx] else -50.0
        fetch_idx = self.n_agent - 1
        shaped_rewards[fetch_idx] = shaped_rewards[fetch_idx] + torch.tensor(bonus).float().view(1, -1)
        self.priority_satisfied[env_idx] = True
        return shaped_rewards

    def _get_expected_macro_action(self, instruction_text, agent_idx=None):
        """
        Map instruction text to expected behavior (either allowed or prohibited actions).

        agent_idx is used to route per-agent instructions like
        "go to small box" -> agent 0 gets GT_SMALL_BOX_0, agent 1 gets
        GT_SMALL_BOX_1.
        """
        instruction_lower = instruction_text.lower().strip()

        # ============== Box Pushing macro-action indices ==============
        GT_SMALL_BOX_0 = 0
        GT_SMALL_BOX_1 = 1
        GT_BIG_BOX_SPOT_0 = 2
        GT_BIG_BOX_SPOT_1 = 3
        PUSH = 4

        # Per-agent: "go to small box" routes agent 0 -> small box 0,
        # agent 1 -> small box 1. Higher-indexed agents get no shaping
        # (return None).
        if instruction_lower in ["go to small box", "small box"]:
            if agent_idx == 0:
                return {'allowed_actions': [GT_SMALL_BOX_0]}
            elif agent_idx == 1:
                return {'allowed_actions': [GT_SMALL_BOX_1]}
            else:
                return None

        # Standard Box Pushing positives
        if instruction_lower in ["go to small box 0", "small box 0", "small_box_0"]:
            return {'allowed_actions': [GT_SMALL_BOX_0]}
        if instruction_lower in ["go to small box 1", "small box 1", "small_box_1"]:
            return {'allowed_actions': [GT_SMALL_BOX_1]}
        if instruction_lower in ["go to big box spot 0", "big box spot 0", "big_box_spot_0"]:
            return {'allowed_actions': [GT_BIG_BOX_SPOT_0]}
        if instruction_lower in ["go to big box spot 1", "big box spot 1", "big_box_spot_1"]:
            return {'allowed_actions': [GT_BIG_BOX_SPOT_1]}
        if instruction_lower == "push":
            return {'allowed_actions': [PUSH]}

        # Standard Box Pushing negatives
        if instruction_lower in ["don't go to small box 0", "avoid small box 0"]:
            return {'prohibited_actions': [GT_SMALL_BOX_0]}
        if instruction_lower in ["don't go to small box 1", "avoid small box 1"]:
            return {'prohibited_actions': [GT_SMALL_BOX_1]}
        if instruction_lower in ["don't go to any small box", "don't go to small boxes",
                                  "avoid small boxes", "avoid all small boxes"]:
            return {'prohibited_actions': [GT_SMALL_BOX_0, GT_SMALL_BOX_1]}
        if instruction_lower in ["don't go to big box spot 0", "avoid big box spot 0"]:
            return {'prohibited_actions': [GT_BIG_BOX_SPOT_0]}
        if instruction_lower in ["don't go to big box spot 1", "avoid big box spot 1"]:
            return {'prohibited_actions': [GT_BIG_BOX_SPOT_1]}
        if instruction_lower in ["don't push", "stop pushing the box", "stop pushing", "stop push"]:
            return {'prohibited_actions': [PUSH]}

        # ============== Overcooked macro-action indices ==============
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

        # Positive instructions
        if instruction_lower == "stay":
            return {'allowed_actions': [STAY]}
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

        # Negative instructions
        elif instruction_lower == "don't stay":
            return {'prohibited_actions': [STAY]}
        elif instruction_lower in ["don't touch the tomato", "don't touch tomato", "don't get tomato"]:
            return {'prohibited_actions': [GET_TOMATO]}
        elif instruction_lower in ["don't touch the lettuce", "don't touch lettuce", "don't get lettuce"]:
            return {'prohibited_actions': [GET_LETTUCE]}
        elif instruction_lower in ["don't touch the onion", "don't get onion"]:
            return {'prohibited_actions': [GET_ONION]}
        elif instruction_lower in ["don't get plate", "don't get plate 1"]:
            return {'prohibited_actions': [GET_PLATE_1]}
        elif instruction_lower in ["don't go to knife 1", "avoid knife 1", "don't use the right cutting board", "don't go to the right"]:
            return {'prohibited_actions': [GO_TO_KNIFE_1]}
        elif instruction_lower in ["don't go to knife 2", "avoid knife 2", "don't use the left cutting board"]:
            return {'prohibited_actions': [GO_TO_KNIFE_2]}
        elif instruction_lower in ["don't deliver", "i will deliver it myself", "i will deliver"]:
            return {'prohibited_actions': [DELIVER]}
        elif instruction_lower == "don't chop":
            return {'prohibited_actions': [CHOP]}

        return None

    def _calculate_episode_compliance(self, env_idx):
        """
        Calculate instruction compliance for an episode.

        Compliance is only meaningful for (agent, action) pairs taken
        WHILE an instruction was active AND that instruction defines an
        expected behavior for the agent. Steps without those conditions
        are excluded from both `overall` and `per_instruction` so the
        metric reflects "compliance during active prohibition" and not
        the trivial baseline of no-instruction phases.

        Returns None when no instruction-active pair was tracked this
        episode (caller should skip recording).
        """
        if not self.episodes[env_idx]:
            return None

        total_compliant_actions = 0
        total_actions = 0
        per_instruction_counts = {}
        per_agent_instruction_counts = {}

        for experience in self.episodes[env_idx]:
            if len(experience) >= 14:  # With instructions
                actions = experience[3]
                inst_texts_data = experience[13]
                if isinstance(inst_texts_data, list):
                    instruction_texts = inst_texts_data
                else:
                    instruction_texts = [inst_texts_data] * len(actions) if inst_texts_data is not None else [None] * len(actions)
            else:
                actions = experience[3]
                instruction_texts = [None] * len(actions)

            for agent_idx, (action, instruction_text) in enumerate(zip(actions, instruction_texts)):
                if action.item() == -1:
                    continue
                if instruction_text is None:
                    continue
                expected = self._get_expected_macro_action(instruction_text, agent_idx=agent_idx)
                if expected is None:
                    continue

                total_actions += 1
                inst_key = instruction_text
                if inst_key not in per_instruction_counts:
                    per_instruction_counts[inst_key] = {'compliant': 0, 'total': 0}
                per_instruction_counts[inst_key]['total'] += 1

                agent_inst_key = (agent_idx, inst_key)
                if agent_inst_key not in per_agent_instruction_counts:
                    per_agent_instruction_counts[agent_inst_key] = {'compliant': 0, 'total': 0}
                per_agent_instruction_counts[agent_inst_key]['total'] += 1

                is_compliant = self._check_action_compliance_with_text(action.item(), instruction_text, agent_idx, env_idx)
                if is_compliant:
                    total_compliant_actions += 1
                    per_instruction_counts[inst_key]['compliant'] += 1
                    per_agent_instruction_counts[agent_inst_key]['compliant'] += 1

        if total_actions == 0:
            return None

        per_instruction = {
            inst_key: counts['compliant'] / counts['total']
            for inst_key, counts in per_instruction_counts.items()
        }
        per_agent_instruction = {
            (agent_idx, inst_key): counts['compliant'] / counts['total']
            for (agent_idx, inst_key), counts in per_agent_instruction_counts.items()
        }

        return {
            'overall': total_compliant_actions / total_actions,
            'per_instruction': per_instruction,
            'per_agent_instruction': per_agent_instruction
        }

    def _calculate_episode_compliance_counts(self, env_idx):
        """Count how many actions complied with the instruction over the episode."""
        if not self.episodes[env_idx]:
            return 0, 0

        total_compliant_actions = 0
        total_actions = 0

        for experience in self.episodes[env_idx]:
            if len(experience) >= 14:
                actions = experience[3]
                inst_texts_data = experience[13]
                if isinstance(inst_texts_data, list):
                    instruction_texts = inst_texts_data
                else:
                    instruction_texts = [inst_texts_data] * len(actions) if inst_texts_data is not None else [None] * len(actions)
            else:
                actions = experience[3]
                instruction_texts = [None] * len(actions)

            for agent_idx, (action, instruction_text) in enumerate(zip(actions, instruction_texts)):
                if action.item() == -1:
                    continue
                total_actions += 1
                if self._check_action_compliance_with_text(action.item(), instruction_text, agent_idx, env_idx):
                    total_compliant_actions += 1

        return total_compliant_actions, total_actions

    def _check_action_compliance_with_text(self, action, instruction_text, agent_idx, env_idx):
        """Check if an action complies with the given instruction text."""
        if instruction_text is None:
            return True

        expected_behavior = self._get_expected_macro_action(instruction_text, agent_idx=agent_idx)
        if expected_behavior is None:
            return True

        if 'allowed_actions' in expected_behavior:
            return action in expected_behavior['allowed_actions']
        elif 'prohibited_actions' in expected_behavior:
            return action not in expected_behavior['prohibited_actions']

        return True

    def get_rand_states(self):
        rand_states = []
        for parent in self.parents:
            parent.send(('get_rand_states', None))
        for parent in self.parents:
            rand_states.append(parent.recv())
        return rand_states

    def load_rand_states(self, rand_states):
        for parent, rand_state in zip(self.parents, rand_states):
            parent.send(('load_rand_states', rand_state))
