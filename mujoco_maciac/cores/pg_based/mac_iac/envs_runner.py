import os
import numpy as np
import random
import torch
import torch.nn.functional as F

from multiprocessing import Process, Pipe

# Tunable instruction-shaping knobs (read once at import). Mirror mac_cac
# so all four pg_based algorithms accept the same env-vars.
# - INSTRUCTION_PENALTY: per-step non-compliance penalty (default -50.0).
#   Smaller magnitude => "agents learn to ignore" failure-mode demos.
# - INSTRUCTION_DURATION_STEPS: 0 = persist until episode end. >0 clears
#   the per-agent instruction text/embedding once that many steps have
#   elapsed since it was provided.
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

                last_valid = valid
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

                # sent experience back (append mean reward and warehouse state)
                child.send((last_obs, action, accu_rewards, obs, avail_actions, terminate, valid, sum(reward)/env.n_agent, warehouse_request_state))

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
                accu_rewards = [0.0] * env.n_agent
                mac_act_step = [0] * env.n_agent
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
    Environment runner which runs mulitpl environemnts in parallel in subprocesses
    and communicates with them via pipe
    """

    def __init__(self, env, n_envs, controller, memory, env_terminate_step, gamma, seed, obs_last_action=False, writer=None, log=False, instruction_provider=None):
        
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
        # Same shape, but holding the SHAPED return (raw env return + the
        # INSTRUCTION_PENALTY incurred for every non-compliant step). Lets
        # wandb plot Returns_With_Instruction_Shaped alongside the raw
        # version so the gap reveals how much penalty the agent ate.
        self.train_episode_instructions_shaped = []
        self.eval_episode_instructions_shaped = []
        # Per-env accumulator for the post-shaping per-step joint reward
        # this episode (mean across agents). Reset at episode boundaries.
        self._epi_shaped_R = [0.0] * n_envs
        # Per-env step at which the active instruction expires (0 = no
        # expiry). Used when INSTRUCTION_DURATION_STEPS > 0 to clear the
        # broadcast per-agent instruction after a fixed window.
        self.instruction_expire_steps = [0] * n_envs
        # Per-episode diagnostics for debugging return gaps
        # each element: {
        #   'return': float,
        #   'instruction': str | None,
        #   'completed': bool,
        #   'horizon_truncated': bool,
        #   'episode_len': int,
        # }
        self.train_episode_diagnostics = []
        self.eval_episode_diagnostics = []
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
            self.eval_episode_diagnostics = []
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

            # Expire the active instruction once the configured duration
            # has elapsed. INSTRUCTION_DURATION_STEPS=0 (default) means
            # "never expire mid-episode" so this branch is a no-op then.
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
                # Force resample under the now-empty instruction state.
                self.last_valids[idx] = self.mac_done_to_tensor([1.0] * self.n_agent)
                parent.send(("reset_macro_actions", None))
                parent.recv()

            # Per-step stochastic instruction provision
            # Each step, if no instruction yet, roll INSTRUCTION_PROVIDED_PROB. Once provided, persists until episode end.
            # Warehouse (OSD) envs skip this block entirely: their instructions are
            # pre-assigned at episode start by _pre_assign_warehouse_instruction so
            # that the tool order is fixed before the first delivery happens.
            if self.instruction_provider is not None and not hasattr(self.env, 'n_objs'):
                provided_prob = float(os.environ.get("INSTRUCTION_PROVIDED_PROB", "0.01"))
                # Only try to provide if no instruction assigned yet for this episode
                if all(emb is None for emb in self.instruction_embs[idx]):
                    # Roll the dice each step (default 1% chance per step)
                    if np.random.random() < provided_prob:
                        # Get ONE instruction and broadcast to ALL agents
                        # (all agents in the same env must receive the same instruction)
                        inst = self.instruction_provider(idx, self.step_count[idx], agent_idx=0)
                        if inst is None:
                            inst_emb_shared = None
                            inst_text_shared = None
                        elif isinstance(inst, tuple) and len(inst) == 2:
                            inst_text_shared, emb = inst
                            if isinstance(emb, torch.Tensor):
                                # BERT embeddings are already float tensors (pre-encoded)
                                inst_emb_shared = emb.detach()
                            else:
                                inst_emb_shared = None
                        elif isinstance(inst, torch.Tensor):
                            inst_emb_shared = inst.detach()
                            inst_text_shared = None
                        else:
                            inst_emb_shared = None
                            inst_text_shared = None

                        for agent_idx in range(self.n_agent):
                            self.instruction_embs[idx][agent_idx] = inst_emb_shared
                            self.instruction_texts_for_env[idx][agent_idx] = inst_text_shared

                        # CRITICAL: When instructions arrive, interrupt ongoing macro-actions
                        self.last_valids[idx] = self.mac_done_to_tensor([1.0] * self.n_agent)
                        parent.send(("reset_macro_actions", None))
                        parent.recv()
                        self.last_instruction_step[idx] = self.step_count[idx]
                        # Schedule expiry if a duration is configured AND we
                        # actually set a non-None instruction this step.
                        if INSTRUCTION_DURATION_STEPS > 0 and inst_emb_shared is not None:
                            self.instruction_expire_steps[idx] = (
                                self.step_count[idx] + INSTRUCTION_DURATION_STEPS
                            )
                        else:
                            self.instruction_expire_steps[idx] = 0
                # else: Keep using the current instructions (no refresh needed yet)

            # Pass list of per-agent instructions to select_action
            actions, self.h_states[idx] = self.controller.select_action(self.last_obses[idx], 
                                                                        self.h_states[idx], 
                                                                        self.last_valids[idx],
                                                                        self.avail_actions[idx],
                                                                        eps=eps,
                                                                        test_mode=test_mode,
                                                                        instruction_emb=self.instruction_embs[idx])
            
            # Apply hard-coded policy for agent 2 when instruction is "let me do all the chopping"
            actions = self._apply_hardcoded_chop_policy(actions, idx)
            
            # send cmd to trigger env step
            parent.send(("step", actions))
            self.step_count[idx] += 1

        # Save per-agent instructions before clearing (for storage in replay buffer)
        saved_inst_embs = []
        saved_inst_texts = []
        for idx in range(len(self.parents)):
            # instruction_embs[idx] is a list of embeddings (one per agent)
            inst_embs_per_agent = self.instruction_embs[idx] if idx < len(self.instruction_embs) else [self.memory.ZERO_INSTRUCTION.clone()] * self.n_agent
            inst_texts_per_agent = self.instruction_texts_for_env[idx] if idx < len(self.instruction_texts_for_env) else [None] * self.n_agent
            
            # Replace None embeddings with zero tensors (when instructions are disabled)
            # Clone to avoid sharing the same tensor object
            inst_embs_per_agent = [
                emb if emb is not None else self.memory.ZERO_INSTRUCTION.clone()
                for emb in inst_embs_per_agent
            ]
            
            saved_inst_embs.append(inst_embs_per_agent)
            saved_inst_texts.append(inst_texts_per_agent)
        
        # collect envs' returns
        for idx, parent in enumerate(self.parents):
            env_return = parent.recv()
            # Strip warehouse_request_state from worker tuple before tensor
            # conversion; pass it through separately to _inst_reward.
            warehouse_request_state = env_return[8] if len(env_return) >= 9 else None
            env_return = env_return[:8]
            # Store the per-agent instructions that were actually used for this decision
            current_inst_embs = saved_inst_embs[idx]  # List of embeddings (one per agent)
            current_inst_texts = saved_inst_texts[idx]  # List of texts (one per agent)

            env_return = self._exp_to_tensor(idx, env_return, eps)
            # Append both instruction embeddings and texts for storage (at the end)
            # These are lists with one entry per agent
            env_return = env_return + (current_inst_embs, current_inst_texts)

            # Apply instruction-based reward shaping BEFORE storing in replay buffer
            # This ensures the agent learns from the shaped rewards
            if self.instruction_provider is not None:
                env_return = self._inst_reward(idx, env_return, warehouse_request_state=warehouse_request_state)

            # Accumulate the shaped joint reward (mean across agents) for
            # this step so the episode-level Returns_With_Instruction_Shaped
            # metric matches what the critic actually sees. env_return[3]
            # is the per-agent reward list AFTER _inst_reward applied the
            # INSTRUCTION_PENALTY / bonus on each agent.
            shaped_per_agent = env_return[3]
            try:
                step_shaped_joint = float(
                    sum(r.item() for r in shaped_per_agent) / len(shaped_per_agent)
                )
            except Exception:
                step_shaped_joint = 0.0
            self._epi_shaped_R[idx] += step_shaped_joint

            # Store the experience with shaped rewards in replay buffer
            self.episodes[idx].append(env_return)

            # Unpack experience tuple (with instruction at the end)
            # env_return: (last_obs, last_avail_actions, a, r, obs, avail_actions, t, mac_v, exp_v, inst_emb, inst_text)
            self.last_obses[idx] = env_return[4]  # obs
            self.avail_actions[idx] = env_return[5]  # avail_actions
            self.last_valids[idx] = env_return[7]  # mac_v
            # Instructions are now refreshed every 20 timesteps (handled at the start of _step)
            t = env_return[6]  # terminate flag

            if self.obs_last_action and sum(self.last_valids[idx]) > 0:
                for nth in range(self.n_agent):
                    if self.last_valids[idx][nth]:
                        self.last_actions[idx][nth] = env_return[2][nth]

            # if episode is done, add it to memory buffer
            if t[0] or self.step_count[idx] == self.max_epi_step:  # t (terminate)
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
                        # Print per-agent instructions
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

                # Track instruction compliance for evaluation episodes.
                # _calculate_episode_compliance now returns None when the
                # episode had no instruction-active steps, so those
                # episodes are excluded from the rolling averages instead
                # of inflating them with a 1.0 baseline.
                if test_mode and self.instruction_provider is not None:
                    episode_compliance = self._calculate_episode_compliance(idx)
                    if episode_compliance is not None:
                        self.eval_compliance.append(episode_compliance['overall'])
                        self.eval_compliance_per_instruction.append(episode_compliance['per_instruction'])
                        self.eval_compliance_per_agent_instruction.append(episode_compliance['per_agent_instruction'])

                # Determine which instruction was active this episode (agent 0's instruction)
                epi_instruction_text = self.instruction_texts_for_env[idx][0] if self.instruction_provider is not None else None
                # completion/truncation diagnostics for this episode
                env_completed = bool(t[0].item())
                hit_horizon = (self.step_count[idx] == self.max_epi_step)
                epi_diag = {
                    'return': float(R),
                    'instruction': epi_instruction_text,
                    'completed': env_completed,
                    'horizon_truncated': bool((not env_completed) and hit_horizon),
                    'episode_len': int(self.step_count[idx])
                }

                shaped_R = self._epi_shaped_R[idx]

                if not test_mode:
                    self.memory.scenario_cache += self.episodes[idx]
                    self.memory.flush_buf_cache()
                    self.train_returns.append(R)
                    self.train_episode_instructions.append((R, epi_instruction_text))
                    self.train_episode_instructions_shaped.append((shaped_R, epi_instruction_text))
                    self.train_episode_diagnostics.append(epi_diag)
                    if self.log and self.writer is not None:
                        self.writer.add_scalar('Return/train/', R, len(self.train_returns))
                        # Also log instruction compliance counts for this episode
                        if self.instruction_provider is not None:
                            compliant, total = self._calculate_episode_compliance_counts(idx)
                            self.writer.add_scalar('Instruction/train/compliant_actions', compliant, len(self.train_returns))
                            self.writer.add_scalar('Instruction/train/total_actions', total, len(self.train_returns))
                            self.writer.add_scalar('Instruction/train/compliance_rate', compliant / max(total, 1), len(self.train_returns))
                else:
                    self.eval_returns.append(R)
                    self.eval_episode_instructions.append((R, epi_instruction_text))
                    self.eval_episode_instructions_shaped.append((shaped_R, epi_instruction_text))
                    self.eval_episode_diagnostics.append(epi_diag)
                    # For eval episodes, keep the detailed compliance list and also log counts
                    if self.instruction_provider is not None and self.log and self.writer is not None:
                        compliant, total = self._calculate_episode_compliance_counts(idx)
                        self.writer.add_scalar('Instruction/eval/compliant_actions', compliant, len(self.eval_returns))
                        self.writer.add_scalar('Instruction/eval/total_actions', total, len(self.eval_returns))
                        self.writer.add_scalar('Instruction/eval/compliance_rate', compliant / max(total, 1), len(self.eval_returns))

                # when episode is done, immediately start a new one
                parent.send(("reset", None))
                self.last_obses[idx], self.h_states[idx], self.last_actions[idx], self.last_valids[idx], self.avail_actions[idx] = parent.recv()
                self.last_obses[idx] = self.obs_to_tensor(self.last_obses[idx])
                if self.obs_last_action:
                    self.last_actions[idx] = self.action_to_tensor(self.last_actions[idx])
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
                    self.last_instruction_step[idx] = 0  # Reset instruction timer
                    # Warehouse: pre-assign instruction + set tool order at episode start
                    self._pre_assign_warehouse_instruction(idx, parent)

    def _pre_assign_warehouse_instruction(self, env_idx, parent):
        """
        For warehouse (OSD) environments only: decide at episode START whether
        this episode gets an instruction, and if so apply it immediately via
        'set_tool_order' before any steps are taken.

        This avoids the mid-episode timing problem: the human's
        request_objs_per_task_step must be fixed at step 0 (right after reset)
        so every delivery is evaluated against the correct target tool.

        The per-episode probability is derived from INSTRUCTION_PROVIDED_PROB
        (the per-step value used by Overcooked) scaled to one episode:
            P_episode = 1 - (1 - P_step) ** max_epi_step
        so long-run instruction frequency is the same as the Overcooked approach.
        """
        # Reset per-episode priority state regardless of whether an instruction
        # is sampled.
        self.priority_tool[env_idx] = None
        self.priority_satisfied[env_idx] = False

        if self.instruction_provider is None or not hasattr(self.env, 'n_objs'):
            return

        provided_prob = float(os.environ.get("INSTRUCTION_PROVIDED_PROB", "0.01"))
        # Convert per-step probability to per-episode probability
        per_episode_prob = 1.0 - (1.0 - provided_prob) ** self.max_epi_step

        if np.random.random() >= per_episode_prob:
            # No instruction this episode — leave embs as None (actor receives zero instruction embedding)
            return

        # Sample an instruction from the pool
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

        # Broadcast to all agents in this environment
        for agent_idx in range(self.n_agent):
            self.instruction_embs[env_idx][agent_idx] = inst_emb
            self.instruction_texts_for_env[env_idx][agent_idx] = inst_text

        # Parse the tool order and send to the worker right away
        tool_order = self._parse_warehouse_tool_order(inst_text)
        if tool_order is not None:
            self.priority_tool[env_idx] = int(tool_order[0])
            parent.send(('set_tool_order', tool_order))
            parent.recv()  # wait for 'done'

    def _reset(self):
        # send cmd to reset envs
        for parent in self.parents:
            parent.send(("reset", None))

        self.last_obses, self.h_states, self.last_actions, self.last_valids, self.avail_actions = [list(i) for i in zip(*[parent.recv() for parent in self.parents])]
        self.last_obses = [self.obs_to_tensor(obs) for obs in self.last_obses] #List[List[tensor]]
        if self.obs_last_action:
            self.last_actions = [self.action_to_tensor(a) for a in self.last_actions]
            # reconstruct obs to observe actions
            self.last_obses = [self.rebuild_obs(self.env, obs, a) for obs, a in zip(*[self.last_obses, self.last_actions])]
        self.last_valids = [self.mac_done_to_tensor(mac_done) for mac_done in self.last_valids]
        self.avail_actions = [self.avail_action_to_tensor(avail_action) for avail_action in self.avail_actions]

        self.n_epi_count = 0
        self.step_count = [0] * self.n_envs
        self.episodes = [[] for i in range(self.n_envs)]
        # Clear any previous per-agent instructions at episode start
        self.instruction_embs = [[None] * self.n_agent for _ in range(self.n_envs)]
        self.instruction_texts_for_env = [[None] * self.n_agent for _ in range(self.n_envs)]
        # Pre-assign instructions for warehouse environments
        for env_idx, parent in enumerate(self.parents):
            self._pre_assign_warehouse_instruction(env_idx, parent)
        if hasattr(self, 'last_instruction_step'):
            self.last_instruction_step = [0] * self.n_envs  # Reset instruction timers

    def _apply_hardcoded_chop_policy(self, actions, env_idx):
        """
        Apply hard-coded policy for agent 2 (third agent) when instruction is "let me do all the chopping".
        
        If agent 2 has the instruction "let me do all the chopping" and there's food on the knife,
        force agent 2 to perform the chop action (action 10 for Overcooked).
        
        Args:
            actions: List of actions for all agents
            env_idx: Environment index
            
        Returns:
            Modified actions list with agent 2's action potentially overridden
        """
        # Only apply for 3+ agents (agent 2 is the third agent, index 2)
        if self.n_agent < 3:
            return actions
        
        # Check if agent 2 has the "let me do all the chopping" instruction
        agent_2_instruction = self.instruction_texts_for_env[env_idx][2] if env_idx < len(self.instruction_texts_for_env) and 2 < len(self.instruction_texts_for_env[env_idx]) else None
        
        if agent_2_instruction is None or agent_2_instruction.lower().strip() != "let me do all the chopping":
            return actions
        
        # Check if there's food on the knife (cutting board) by examining the observation
        obs = self.last_obses[env_idx]  # List of observation tensors for each agent
        
        # Check if there's food on the knife - look at the observation structure
        # The observation contains item positions and states (including chopped status for food)
        # Food items have 3 values: x, y, chopped_status
        # Knife items have 2 values: x, y (and possibly holding food)
        
        if self._is_food_on_knife(obs):
            # Force agent 2 to perform chop action (action 10 in Overcooked)
            # The chop action index is 10 for Map A/B/C (0-indexed)
            CHOP_ACTION = 10
            actions_list = list(actions) if not isinstance(actions, list) else actions
            actions_list[2] = CHOP_ACTION
            return actions_list
        
        return actions
    
    def _is_food_on_knife(self, obs_list):
        """
        Check if there's food on any knife by examining the observation.
        
        Agent 2 has FULL OBSERVATION of the environment (not partial), which provides
        complete visibility of all items and their states across the entire map.
        
        The observation structure includes items with their positions and states.
        We need to detect if there's any food currently on a knife (cutting board).
        
        Args:
            obs_list: List of observation tensors (one per agent) - agent 2 has full env observation
        Returns:
            bool: True if there's food on a knife, False otherwise
        """
        # Since agent 2 has full observation, we can reliably check all food items
        # Look for food items that might be on a knife. In Overcooked, food on a knife 
        # is indicated by the food's chopped_progress being between 0 and 1.
        
        # Get one observation (agent 2 has full state observation of the environment)
        obs = obs_list[0]  # Shape: [1, obs_size]
        obs_array = obs.squeeze(0).numpy() if isinstance(obs, torch.Tensor) else obs.numpy() if hasattr(obs, 'numpy') else obs
        
        # In the Overcooked observation with full visibility:
        # - Each food item takes 3 positions: x_norm, y_norm, chopped_progress
        # - The chopped_progress is the number of times chopped / required_chops
        # - If food is on the knife and being chopped, it will have chopped_progress in (0, 1)
        # - Items are arranged as: [item0_x, item0_y, item0_state (if food), item1_x, ...]
        
        # Count through observation to find food items with partial chopping
        i = 0
        while i < len(obs_array):
            # Each item starts with x, y position (2 values)
            if i + 2 >= len(obs_array):
                break
            
            # Check if there's a third value (indicates this is food with chopped state)
            if i + 2 < len(obs_array):
                chopped_progress = obs_array[i + 2]
                # If chopped_progress is between 0 and 1 (exclusive), food is on knife being chopped
                if 0 < chopped_progress < 1:
                    return True
                i += 3  # Food item: x, y, chopped_progress
            else:
                i += 2  # Non-food item: x, y
        
        return False

    def _exp_to_tensor(self, env_idx, exp, eps):
        # exp (last_obs, a, r, obs, avail_actions, t, mac_v, mean_reward)
        last_obs = [torch.from_numpy(o).float().view(1,-1) for o in exp[0]]
        last_avail_actions = [torch.FloatTensor(avail_action).view(1,-1) for avail_action in self.avail_actions[env_idx]]
        a = [torch.tensor(a).view(1,-1) for a in exp[1]]
        r = [torch.tensor(r).float().view(1,-1) for r in exp[2]]
        obs = [torch.from_numpy(o).float().view(1,-1) for o in exp[3]]
        avail_actions = [torch.FloatTensor(avail_action).view(1,-1) for avail_action in exp[4]]
        # re-construct obs if obs last action
        if self.obs_last_action:
            last_obs = self.rebuild_obs(self.env, last_obs, self.last_actions[env_idx])
            obs = self.rebuild_obs(self.env, obs, a)
        t = torch.tensor(exp[5]).float().view(1,-1)
        mac_v = [torch.tensor(v, dtype=torch.bool).view(1,-1) for v in exp[6]]
        exp_v = [torch.tensor([1.0]).view(1,-1)] * self.n_agent
        # mean reward at exp[7] (ignored here but preserved in tuple if needed elsewhere)
        return (last_obs, last_avail_actions, a, r, obs, avail_actions, t, mac_v, exp_v)

    def _inst_reward(self, env_idx, env_return, warehouse_request_state=None):
        """
        Apply reward shaping based on instruction compliance for individual agents.

        Args:
            env_idx: Environment index
            env_return: Experience tuple with instruction at the end
            warehouse_request_state: Per-step warehouse signal from the worker
                (carries 'currently_received_tools' for first-delivery detection).

        Returns:
            Modified env_return with shaped rewards
        """
        # Unpack the tuple
        last_obs, last_avail_actions, a, r, obs, avail_actions, t, mac_v, exp_v, inst_embs, inst_texts = env_return

        # inst_texts is now a list (one instruction per agent). When no
        # instruction is active this episode, priority_tool[env_idx] is None,
        # so warehouse priority shaping is a no-op — safe to early-return.
        if inst_texts is None or all(text is None for text in inst_texts):
            return env_return

        # Track compliance for statistics (per environment) - aggregate across agents
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
            
            # Get this agent's instruction
            agent_instruction_text = inst_texts[agent_idx] if agent_idx < len(inst_texts) else None
            
            if agent_instruction_text is None:
                # No instruction for this agent, no shaping
                shaped_rewards.append(r[agent_idx])
                continue

            is_warehouse_inst = self._parse_warehouse_tool_order(agent_instruction_text) is not None

            # Get expected behavior for this agent's instruction
            expected_behavior = self._get_expected_macro_action(agent_instruction_text, agent_idx=agent_idx)
            
            # Track instruction for this agent
            self._instruction_stats[env_idx]['instructions_by_agent'][agent_idx] = agent_instruction_text
            if expected_behavior:
                self._instruction_stats[env_idx]['expected_by_agent'][agent_idx] = expected_behavior
            
            # Track action distribution (count each primitive step once)
            if action_value not in self._instruction_stats[env_idx]['action_counts']:
                self._instruction_stats[env_idx]['action_counts'][action_value] = 0
            self._instruction_stats[env_idx]['action_counts'][action_value] += 1

            # Ensure per-instruction counters exist
            if agent_instruction_text not in self._instruction_stats[env_idx]['per_instruction']:
                self._instruction_stats[env_idx]['per_instruction'][agent_instruction_text] = {
                    'compliant': 0, 'non_compliant': 0
                }
            per_inst_stats = self._instruction_stats[env_idx]['per_instruction'][agent_instruction_text]
            
            if expected_behavior is None:
                # No specific behavior defined for this instruction
                shaped_rewards.append(r[agent_idx])
                continue

            # Once the priority tool has been delivered, skip dense shaping +
            # compliance counting for warehouse instructions; the agent must
            # be free to fetch other tools to complete remaining requests.
            if is_warehouse_inst and self.priority_satisfied[env_idx]:
                shaped_rewards.append(r[agent_idx])
                continue

            wh_penalty = -25
            wh_bonus = 10

            if 'prohibited_actions' in expected_behavior:
                # This is a negative instruction (don't do X)
                prohibited = expected_behavior['prohibited_actions']

                if action_value in prohibited:
                    # Strong penalty for violating prohibition. The magnitude is
                    # safe because the chain-break + dual-critic framework
                    # (see mac_iac/learner.py::_get_discounted_return) confines
                    # this spike to the instruct segment's return; V_Psi (task
                    # critic) never sees it, so task learning is not polluted.
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
                # This is a positive instruction (do X)
                allowed = expected_behavior['allowed_actions']
                if action_value in allowed:
                    bonus = wh_bonus if is_warehouse_inst else 0
                    shaped_reward = r[agent_idx].item() + bonus
                    self._instruction_stats[env_idx]['compliant'] += 1
                    per_inst_stats['compliant'] += 1
                else:
                    # Penalize non-compliant actions; see comment above for why
                    # the -50 magnitude is isolated by value cancellation.
                    penalty = wh_penalty if is_warehouse_inst else INSTRUCTION_PENALTY
                    shaped_reward = r[agent_idx].item() + penalty
                    self._instruction_stats[env_idx]['non_compliant'] += 1
                    per_inst_stats['non_compliant'] += 1
            
            else:
                # No specific behavior defined
                shaped_reward = r[agent_idx].item()

            shaped_rewards.append(torch.tensor(shaped_reward).float().view(1,-1))

        shaped_rewards = self._apply_warehouse_priority_terminal(
            env_idx, shaped_rewards, warehouse_request_state,
        )

        # Reconstruct the tuple with shaped rewards (with per-agent instructions)
        return (last_obs, last_avail_actions, a, shaped_rewards, obs, avail_actions, t, mac_v, exp_v, inst_embs, inst_texts)

    def _apply_warehouse_priority_terminal(self, env_idx, shaped_rewards, warehouse_request_state):
        """One-shot ±50 credit to Fetch on the first delivery of the episode.

        Compares the first delivered tool against priority_tool[env_idx]:
        +50 if it matches the prioritized tool, -50 otherwise. After firing,
        priority_satisfied[env_idx] flips True and dense shaping is disabled
        for the rest of the episode.
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

    def _parse_warehouse_tool_order(self, instruction_text):
        """
        Parse a warehouse instruction into an explicit tool delivery order.

        Supported patterns
        ------------------
        "fetch tool X first"  / "deliver tool X first"
            → put tool X at position 0; remaining tools keep natural order.
            e.g. "fetch tool 1 first" with n_objs=3  → [1, 0, 2]

        Returns
        -------
        list[int] or None
            Ordered list of tool indices, or None if not a warehouse instruction.
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
            return None  # Not a warehouse env

        # Reject out-of-range tool indices (would crash human.next_request_obj_idx)
        if priority_tool < 0 or priority_tool >= n_objs:
            return None

        # Build order: priority tool first, then the rest in natural order
        rest = [i for i in range(n_objs) if i != priority_tool]
        return [priority_tool] + rest

    def _get_expected_macro_action(self, instruction_text, agent_idx=None):
        """
        Map instruction text to expected behavior (either allowed or prohibited actions).

        Args:
            instruction_text: The natural-language instruction string.
            agent_idx: The index of the agent being checked.  When provided,
                warehouse instructions only apply to the Fetch robot (last agent).

        Returns:
            dict with either 'allowed_actions' or 'prohibited_actions' key, or None if not found
        """
        instruction_lower = instruction_text.lower().strip()

        # ============== Box Pushing macro-action indices ==============
        GT_SMALL_BOX_0 = 0
        GT_SMALL_BOX_1 = 1
        GT_BIG_BOX_SPOT_0 = 2
        GT_BIG_BOX_SPOT_1 = 3
        PUSH = 4

        # Per-agent: "go to small box" routes agent 0 -> small box 0,
        # agent 1 -> small box 1.
        if instruction_lower in ["go to small box", "small box"]:
            if agent_idx == 0:
                return {'allowed_actions': [GT_SMALL_BOX_0]}
            elif agent_idx == 1:
                return {'allowed_actions': [GT_SMALL_BOX_1]}
            else:
                return None

        # Box Pushing positives
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

        # Box Pushing negatives
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
        # Ovens (Map D):
        USE_LEFT_OVEN = 14  # "go to oven 1"
        USE_RIGHT_OVEN = 15  # "go to oven 2"

        # Positive instructions (do X) - exact phrase matching
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

        # New negative instructions for ovens
        elif instruction_lower in ["don't use the left oven", "don't go to the left oven", "avoid left oven"]:
            return {'prohibited_actions': [USE_LEFT_OVEN]}
        elif instruction_lower in ["don't use the right oven", "don't go to the right oven", "avoid right oven"]:
            return {'prohibited_actions': [USE_RIGHT_OVEN]}

        # Negative instructions (don't do X) - reward any action EXCEPT the prohibited ones
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

        # ------------------------------------------------------------------
        # Warehouse (OSD) instructions – change the tool delivery priority.
        #
        # Macro-action layout for the Fetch robot (agent index = n_agent-1):
        #   0 : Wait_Request
        #   1 : Pass_Obj_T0
        #   2 : Pass_Obj_T1
        #   3+i : Look_For_obj_i  (i = 0 … n_objs-1)
        #
        # Compliance is only enforced for the Fetch robot.  Turtlebot agents
        # (indices 0 … n_agent-2) always return None so they are not penalised.
        # ------------------------------------------------------------------
        import re as _re

        # "fetch tool X first" / "deliver tool X first"
        _wh_pos = _re.search(r'(?:fetch|deliver)\s+tool\s+(\d+)\s+first', instruction_lower)
        if _wh_pos:
            tool_idx = int(_wh_pos.group(1))
            # Turtlebot agents: no compliance shaping for this instruction type
            if agent_idx is not None and agent_idx != self.n_agent - 1:
                return None
            # Fetch robot: prohibited from looking for any tool OTHER than tool_idx
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

    def _calculate_episode_compliance(self, env_idx):
        """
        Calculate instruction compliance for an episode.

        Compliance is only meaningful for (agent, action) pairs taken
        WHILE an instruction was actually constraining behavior — i.e.
        instruction_text is not None AND that instruction defines an
        expected behavior for the agent in question. Steps without those
        conditions are excluded from both `overall` and `per_instruction`
        so the metric reflects "compliance during active prohibition" and
        not the trivial 100% baseline of no-instruction phases.

        Returns:
            dict | None: None when no instruction-active pair was tracked
                this episode (caller should skip recording). Otherwise:
                'overall':              float compliance rate (0.0 to 1.0)
                'per_instruction':      {instruction_text: rate}
                'per_agent_instruction': {(agent_idx, instruction_text): rate}
        """
        if not self.episodes[env_idx]:
            return None

        total_compliant_actions = 0
        total_actions = 0
        per_instruction_counts = {}
        per_agent_instruction_counts = {}  # keyed by (agent_idx, instruction_text)

        for experience in self.episodes[env_idx]:
            # Handle different experience tuple lengths (with/without instruction text)
            if len(experience) >= 11:  # Has instruction text
                # Unpack experience: (last_obs, last_avail_actions, a, r, obs, avail_actions, t, mac_v, exp_v, inst_embs, inst_texts)
                actions = experience[2]  # agent actions
                inst_texts_data = experience[10]  # instruction texts

                # instruction texts could be a list (per-agent) or single value (legacy)
                if isinstance(inst_texts_data, list):
                    # Per-agent instructions (new format)
                    instruction_texts = inst_texts_data
                else:
                    # Single instruction shared by all agents (legacy or per-environment mode)
                    instruction_texts = [inst_texts_data] * len(experience[2]) if inst_texts_data is not None else [None] * len(experience[2])
            else:  # Older format without instruction text
                # Unpack experience: (last_obs, last_avail_actions, a, r, obs, avail_actions, t, mac_v, exp_v, inst_emb)
                actions = experience[2]  # agent actions
                instruction_texts = [None] * len(actions)  # Create None placeholders

            for agent_idx, (action, instruction_text) in enumerate(zip(actions, instruction_texts)):
                if action.item() == -1:  # Invalid action (mac in progress)
                    continue
                # Skip no-instruction steps and steps where the active
                # instruction has no expected behavior for this agent.
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
        """
        Count how many actions complied with the instruction over the episode.

        Returns:
            (int, int): (num_compliant_actions, num_total_actions)
        """
        if not self.episodes[env_idx]:
            return 0, 0

        total_compliant_actions = 0
        total_actions = 0

        for experience in self.episodes[env_idx]:
            # Handle different experience tuple lengths (with/without instruction text)
            if len(experience) >= 11:  # Has instruction text
                actions = experience[2]
                inst_texts_data = experience[10]  # instruction texts
                
                # instruction texts could be a list (per-agent) or single value (legacy)
                if isinstance(inst_texts_data, list):
                    # Per-agent instructions (new format)
                    instruction_texts = inst_texts_data
                else:
                    # Single instruction shared by all agents (legacy or per-environment mode)
                    instruction_texts = [inst_texts_data] * len(experience[2]) if inst_texts_data is not None else [None] * len(experience[2])
            else:  # Older format without instruction text
                actions = experience[2]
                instruction_texts = [None] * len(actions)

            for agent_idx, (action, instruction_text) in enumerate(zip(actions, instruction_texts)):
                if action.item() == -1:
                    continue
                total_actions += 1
                if self._check_action_compliance_with_text(action.item(), instruction_text, agent_idx, env_idx):
                    total_compliant_actions += 1

        return total_compliant_actions, total_actions

    def _check_action_compliance(self, action, instruction_emb, agent_idx, env_idx):
        """
        Check if an action complies with the given instruction.

        Args:
            action: The action taken by the agent
            instruction_emb: The instruction embedding
            agent_idx: Index of the agent
            env_idx: Index of the environment

        Returns:
            bool: True if action complies with instruction
        """
        # Get the instruction text for this environment
        if hasattr(self.controller, 'instruction_texts') and hasattr(self.controller, 'env_instruction_indices'):
            current_instruction_idx = self.controller.env_instruction_indices[env_idx]
            instruction_text = self.controller.instruction_texts[current_instruction_idx]
        else:
            return True  # No instruction info available, assume compliant

        # Get expected behavior for this instruction (per-agent-aware so
        # routed instructions like "go to small box" resolve correctly).
        expected_behavior = self._get_expected_macro_action(instruction_text, agent_idx=agent_idx)
        if expected_behavior is None:
            return True  # No specific expected behavior

        # Check if action matches expected behavior
        if 'allowed_actions' in expected_behavior:
            return action in expected_behavior['allowed_actions']
        elif 'prohibited_actions' in expected_behavior:
            return action not in expected_behavior['prohibited_actions']

        return True  # Default to compliant if no specific rules

    def _check_action_compliance_with_text(self, action, instruction_text, agent_idx, env_idx):
        """
        Check if an action complies with the given instruction text.

        Args:
            action: The action taken by the agent
            instruction_text: The instruction text (already decoded)
            agent_idx: Index of the agent
            env_idx: Index of the environment

        Returns:
            bool: True if action complies with instruction
        """
        if instruction_text is None:
            return True  # No instruction, assume compliant

        # Get expected behavior for this instruction (pass agent_idx for warehouse routing)
        expected_behavior = self._get_expected_macro_action(instruction_text, agent_idx=agent_idx)
        if expected_behavior is None:
            return True  # No specific expected behavior

        # Check if action matches expected behavior
        if 'allowed_actions' in expected_behavior:
            return action in expected_behavior['allowed_actions']
        elif 'prohibited_actions' in expected_behavior:
            return action not in expected_behavior['prohibited_actions']

        return True  # Default to compliant if no specific rules

    def _get_instruction_text_for_embedding(self, env_idx, instruction_emb):
        """
        Get the instruction text that corresponds to a given instruction embedding.

        Args:
            env_idx: Environment index
            instruction_emb: Instruction embedding tensor

        Returns:
            str: The instruction text, or None if not found
        """
        if instruction_emb is None or not hasattr(self.controller, 'instruction_texts'):
            return None

        # For now, use the current instruction index for this environment
        # This works if instructions don't change during an episode
        if hasattr(self.controller, 'env_instruction_indices'):
            current_instruction_idx = self.controller.env_instruction_indices[env_idx]
            if current_instruction_idx < len(self.controller.instruction_texts):
                return self.controller.instruction_texts[current_instruction_idx]

        return None
