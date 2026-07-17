import argparse
import numpy as np
import torch
import os
import sys
import time
import gym

# Add paths for custom modules BEFORE importing them
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from torch.distributions import Categorical

# Importing macro_marl.my_env triggers gym.envs.register() for BP-MA-v0.
import macro_marl.my_env  # noqa: F401
from macro_marl.cores.pg_based.mac_iac.utils import Agent
from macro_marl.cores.pg_based.mac_iac.models import Actor

# Shared overcooked-style input UX (pyglet handlers + in-window overlay).
sys.path.append(os.path.dirname(__file__))
from _bp_instruction_input import BPInstructionInput  # noqa: E402

# ---------------------------------------------------------------------------
# Hardcoded policy loading config.
# Update POLICY_RUN_DIR to the folder you want to load from under
# POLICY_BASE_DIR. Mirrors the layout of test_overcooked_iac.py so the two
# tools feel familiar.
# ---------------------------------------------------------------------------
POLICY_BASE_DIR = "/home/willy/Documents/macro_marl_ppo/experiments/BoxPushing/policy_nns"
POLICY_RUN_DIR = "ma_cac_bp6_instructions_stochastic_chain_break_0"

# Box-Pushing macro-action layout (must match BoxPushing_harder.MAs).
MACRO_ACTION_NAMES = ["GT_SB0", "GT_SB1", "GT_BB0", "GT_BB1", "PUSH", "T_L", "T_R", "STAY"]
GT_SMALL_BOX_0, GT_SMALL_BOX_1 = 0, 1
GT_BIG_BOX_SPOT_0, GT_BIG_BOX_SPOT_1 = 2, 3
PUSH, T_L, T_R, STAY = 4, 5, 6, 7

# Instructions the BP runners understand (must match training script set
# and the entries in mac_iac/mac_cac/mac_iaicc/acac envs_runner.py).
INSTRUCTION_LIST = [
    "go to small box",                 # per-agent suboptimal (agent 0 -> SB0, agent 1 -> SB1)
    "don't push",                      # global prohibition
    "stop pushing the box",            # alias of don't push
    "don't go to small box 0",
    "don't go to small box 1",
    "don't go to any small box",
    "go to big box spot 0",
    "go to big box spot 1",
]

INSTRUCTION_ALIASES = {
    "dont push": "don't push",
    "stop push": "stop pushing the box",
    "stop pushing": "stop pushing the box",
    "dont go to small box 0": "don't go to small box 0",
    "dont go to small box 1": "don't go to small box 1",
    "dont go to any small box": "don't go to any small box",
    "avoid small boxes": "don't go to any small box",
    "avoid all small boxes": "don't go to any small box",
}


def _normalize_instruction(text):
    if text is None:
        return None
    normalized = text.lower().strip()
    normalized = normalized.replace("’", "'").replace("‘", "'")
    normalized = " ".join(normalized.split())
    return normalized


def resolve_policy_dir():
    """Resolve the on-disk folder for the trained policies."""
    if POLICY_RUN_DIR and os.path.isabs(POLICY_RUN_DIR):
        return POLICY_RUN_DIR
    if POLICY_RUN_DIR:
        return os.path.join(POLICY_BASE_DIR, POLICY_RUN_DIR)
    return POLICY_BASE_DIR


def instruction_to_embedding(text, model, device=None):
    """Convert instruction text to BERT embedding via the actor's encoder.

    Empty/None text returns None — the IAC actor's forward handles a
    missing instruction_emb internally, and using a model attribute that
    may not exist (e.g. on alt-build checkpoints) would crash.
    """
    if not getattr(model, "use_instructions", False):
        return None
    if not text:
        return None
    with torch.no_grad():
        emb = model.encode_instruction(text)
        if device is not None:
            emb = emb.to(device)
    return emb


def get_expected_actions(instruction_text, agent_idx):
    """Return per-agent expected/prohibited action indices for a BP instruction.

    Mirrors the routing in src/macro_marl/cores/pg_based/*/envs_runner.py so
    the test harness masks logits the same way the runner shapes rewards.

    Returns
    -------
    dict with one of:
        {'allowed_actions':    [int, ...]}
        {'prohibited_actions': [int, ...]}
        None  -- no shaping for this (instruction, agent) pair
    """
    canonical = _normalize_instruction(instruction_text)
    if canonical is None:
        return None
    canonical = INSTRUCTION_ALIASES.get(canonical, canonical)

    # Per-agent positive routing: agent 0 -> SB0, agent 1 -> SB1.
    if canonical in ("go to small box", "small box"):
        if agent_idx == 0:
            return {"allowed_actions": [GT_SMALL_BOX_0]}
        if agent_idx == 1:
            return {"allowed_actions": [GT_SMALL_BOX_1]}
        return None

    if canonical in ("go to big box spot 0", "big box spot 0", "big_box_spot_0"):
        return {"allowed_actions": [GT_BIG_BOX_SPOT_0]}
    if canonical in ("go to big box spot 1", "big box spot 1", "big_box_spot_1"):
        return {"allowed_actions": [GT_BIG_BOX_SPOT_1]}
    if canonical in ("go to small box 0", "small box 0", "small_box_0"):
        return {"allowed_actions": [GT_SMALL_BOX_0]}
    if canonical in ("go to small box 1", "small box 1", "small_box_1"):
        return {"allowed_actions": [GT_SMALL_BOX_1]}
    if canonical == "push":
        return {"allowed_actions": [PUSH]}
    if canonical in ("don't push", "stop pushing the box", "stop pushing", "stop push"):
        return {"prohibited_actions": [PUSH]}
    if canonical in ("don't go to small box 0", "avoid small box 0"):
        return {"prohibited_actions": [GT_SMALL_BOX_0]}
    if canonical in ("don't go to small box 1", "avoid small box 1"):
        return {"prohibited_actions": [GT_SMALL_BOX_1]}
    if canonical in ("don't go to any small box", "don't go to small boxes",
                     "avoid small boxes", "avoid all small boxes"):
        return {"prohibited_actions": [GT_SMALL_BOX_0, GT_SMALL_BOX_1]}

    return None


def reset_macro_actions(env):
    """Force every macro-agent to resample on the next step.

    The BP-MA-v0 env doesn't expose an interrupt, so we mark each agent's
    cur_action as None to provoke re-selection. This is best-effort — if the
    underlying agents class changes, the mid-episode-instruction-change path
    will silently fall back to "wait until current macro completes".
    """
    inner = getattr(env, "env", env)
    agents = getattr(inner, "agents", None)
    if not agents:
        return
    for ag in agents:
        cur = getattr(ag, "cur_action", None)
        if cur is not None and hasattr(cur, "t"):
            try:
                cur.t = 0  # mark macro as terminated so next step resamples
            except Exception:
                pass


def get_actions_and_h_states(env, agents, last_valid, obs_list, h_states_list,
                             current_instruction, force_resample=False,
                             apply_mask=True):
    """Sample actions for every learner-controlled agent.

    apply_mask=True (default) hard-masks instruction-violating actions in
    the policy logits before sampling, so the policy is forced into
    compliance. apply_mask=False shows what the unmasked policy actually
    picks at decision time — same condition as training/eval, and the
    right baseline for reading wandb compliance numbers."""
    actions, new_h_states = [], []

    with torch.no_grad():
        for agent in agents:
            obs = obs_list[agent.idx]

            # Pad/truncate observation to the actor's expected width.
            if obs.shape[0] < agent.expected_input_dim:
                padding = torch.zeros(agent.expected_input_dim - obs.shape[0])
                obs = torch.cat([obs, padding])
            elif obs.shape[0] > agent.expected_input_dim:
                obs = obs[:agent.expected_input_dim]

            instruction_emb = instruction_to_embedding(
                current_instruction,
                agent.policy_net,
                device=obs.device,
            )

            action_logits, new_h = agent.policy_net(
                obs.view(1, 1, agent.expected_input_dim),
                h_states_list[agent.idx],
                instruction_emb=instruction_emb,
            )
            n_actions = env.n_action[agent.idx]
            logits = action_logits[:, :, :n_actions].clone()

            # Apply per-agent instruction mask (skipped when --no_mask).
            expected = get_expected_actions(current_instruction, agent.idx) if apply_mask else None
            if expected is not None:
                if "allowed_actions" in expected:
                    allowed = set(expected["allowed_actions"])
                    for a_idx in range(n_actions):
                        if a_idx not in allowed:
                            logits[:, :, a_idx] = -float("inf")
                elif "prohibited_actions" in expected:
                    for p_idx in expected["prohibited_actions"]:
                        if p_idx < n_actions:
                            logits[:, :, p_idx] = -float("inf")

            action = Categorical(logits=logits[0]).sample().item()
            actions.append(action)
            new_h_states.append(new_h)

            if force_resample:
                print(f"  Agent {agent.idx} resampled -> {MACRO_ACTION_NAMES[action]}")

    return actions, new_h_states


def get_init_inputs(env, n_agent):
    reset_result = env.reset()
    if isinstance(reset_result, list):
        return [torch.from_numpy(o).float() for o in reset_result], [None] * n_agent
    return [torch.from_numpy(reset_result).float() for _ in range(n_agent)], [None] * n_agent


def load_agents(env, n_agent, policy_dir, p_id):
    """Locate and load mac_iac actor policies for each agent."""
    available_policies = []
    selected_run_id = None

    run_id_candidates = []
    for rid in [p_id, 0, 1, 2, 3, 4, 5]:
        if rid not in run_id_candidates:
            run_id_candidates.append(rid)

    for rid in run_id_candidates:
        candidate = []
        for i in range(n_agent):
            path = os.path.join(policy_dir, f"{rid}_agent_{i}.pt")
            if not os.path.exists(path):
                candidate = []
                break
            candidate.append((i, path))
        if candidate:
            available_policies = candidate
            selected_run_id = rid
            break

    if not available_policies:
        # Fallback patterns
        for i in range(n_agent):
            for fname in (f"stochastic_policy_agent_{i}.pt",
                          f"fixed_policy_agent_{i}.pt",
                          f"agent_state_dict_{i}.pt"):
                path = os.path.join(policy_dir, fname)
                if os.path.exists(path):
                    available_policies.append((i, path))
                    break

    if not available_policies:
        print(f"\nERROR: No policy files found in {policy_dir}")
        if os.path.isdir(policy_dir):
            print("Available .pt files:")
            for f in sorted(os.listdir(policy_dir)):
                if f.endswith(".pt"):
                    print(f"  - {f}")
        sys.exit(1)

    if selected_run_id is not None and selected_run_id != p_id:
        print(f"Requested p_id={p_id} not found; using run_id={selected_run_id} instead.")

    print(f"\nLoaded {len(available_policies)} policies:")
    for agent_idx, path in available_policies:
        print(f"  Agent {agent_idx}: {os.path.basename(path)}")

    agents = []
    for i in range(n_agent):
        agent = Agent()
        agent.idx = i

        policy_path = next((p for j, p in available_policies if j == i), None)
        if policy_path is None:
            print(f"ERROR: Policy file not found for agent {i}")
            sys.exit(1)

        loaded = torch.load(policy_path, map_location="cpu", weights_only=False)
        if isinstance(loaded, Actor):
            actor_net = loaded
            model_input_dim = actor_net.fc1.in_features
            fusion = getattr(actor_net, "instruction_fusion", "unknown")
            uses_inst = getattr(actor_net, "use_instructions", "unknown")
            print(f"  Agent {i}: full Actor object (fc1.in={model_input_dim}, "
                  f"use_instructions={uses_inst}, fusion={fusion})")
        else:
            model_input_dim = env.obs_size[i]
            actor_net = Actor(
                input_dim=model_input_dim,
                output_dim=env.n_action[i],
                use_instructions=True,
                instruction_fusion="attention",
            )
            state_dict = loaded["actor_net_state_dict"] if (
                isinstance(loaded, dict) and "actor_net_state_dict" in loaded
            ) else loaded
            actor_net.load_state_dict(state_dict)
            print(f"  Agent {i}: state dict (input_dim={model_input_dim})")

        actor_net.eval()
        agent.policy_net = actor_net

        # For 'concat' fusion the actor's first FC includes the instruction
        # dim — pad obs to (in_features - instruction_dim).
        if getattr(actor_net, "instruction_fusion", None) == "concat":
            instr_dim = (
                getattr(actor_net, "instruction_dim", None)
                or getattr(actor_net, "n_instructions", None)
                or (actor_net.instruction_projection.out_features
                    if hasattr(actor_net, "instruction_projection") else 32)
            )
            agent.expected_input_dim = model_input_dim - instr_dim
        else:
            agent.expected_input_dim = model_input_dim

        agents.append(agent)
    return agents


def prompt_instruction():
    """Ask the user for an instruction at the terminal between episodes.

    Empty input means "no instruction this episode". Type 'q' to quit.
    """
    print("\n" + "=" * 60)
    print("Available instructions (case-insensitive, paraphrases via aliases):")
    for s in INSTRUCTION_LIST:
        print(f"  - {s}")
    print("Press ENTER for NO instruction, type 'q' to quit.")
    print("=" * 60)
    raw = input("Instruction: ").strip()
    if raw.lower() in ("q", "quit", "exit"):
        return "__QUIT__"
    return raw or None


def test(env_id, env_terminate_step, grid_dim, n_agent, n_episode, p_id,
         instruction, interactive, render, render_delay,
         big_box_reward, small_box_reward, penalty, apply_mask=True):
    env_params = {
        "grid_dim": grid_dim,
        "n_agent": n_agent,
        "terminate_step": env_terminate_step,
        "penalty": penalty,
        "big_box_reward": big_box_reward,
        "small_box_reward": small_box_reward,
        "random_init": False,
        "render": render,
    }
    env = gym.make(env_id, **env_params)

    print(f"\n{'='*60}")
    print(f"Box Pushing IAC instruction-test — grid {grid_dim}, n_agent={n_agent}")
    print(f"{'='*60}")
    print(f"Env obs_size: {env.obs_size}, n_action: {env.n_action}")

    policy_dir = resolve_policy_dir()
    print(f"Loading policies from: {policy_dir}")
    agents = load_agents(env, n_agent, policy_dir, p_id)

    # If user passed --instruction "..." treat as a fixed instruction for
    # every episode unless --interactive overrides it.
    fixed_instruction = instruction or None

    discount = 0.95
    inst_input = BPInstructionInput(instruction_list=INSTRUCTION_LIST)

    for e in range(n_episode):
        if interactive:
            picked = prompt_instruction()
            if picked == "__QUIT__":
                break
            current_instruction = picked
        else:
            current_instruction = fixed_instruction
        inst_input.current_instruction = current_instruction

        print(f"\n--- Episode {e + 1}/{n_episode} ---")
        print(f"Instruction: {current_instruction!r}")

        last_obs, h_states = get_init_inputs(env, n_agent)
        last_valid = [1.0] * n_agent

        if render:
            try:
                env.render()
            except Exception:
                pass
            if inst_input.attach(env):
                print("\n[Press 't' on the game window to type an instruction. "
                      "ENTER confirms; empty + ENTER clears; ESC cancels.]\n")

        compliant_steps = 0
        scored_steps = 0
        action_counts = {a: 0 for a in range(len(MACRO_ACTION_NAMES))}

        R, step, t = 0.0, 0.0, 0
        while not t:
            if inst_input.is_active():
                if render:
                    try:
                        env.render()
                    except Exception:
                        pass
                    if render_delay > 0:
                        time.sleep(render_delay)
                continue

            if inst_input.consume_change():
                current_instruction = inst_input.current_instruction
                reset_macro_actions(env)
                last_valid = [1.0] * n_agent

            actions, h_states = get_actions_and_h_states(
                env, agents, last_valid, last_obs, h_states,
                current_instruction, force_resample=False,
                apply_mask=apply_mask,
            )

            # Per-step compliance bookkeeping (matches the runner's logic).
            for ag_idx, a in enumerate(actions):
                action_counts[a] = action_counts.get(a, 0) + 1
                exp = get_expected_actions(current_instruction, ag_idx)
                if exp is None:
                    continue
                scored_steps += 1
                if "allowed_actions" in exp and a in exp["allowed_actions"]:
                    compliant_steps += 1
                elif "prohibited_actions" in exp and a not in exp["prohibited_actions"]:
                    compliant_steps += 1

            obs, r, t, info = env.step(actions)
            last_obs = [torch.from_numpy(o).float() for o in obs]
            last_valid = info.get("mac_done", [1.0] * n_agent)
            R += (discount ** step) * (sum(r) / n_agent)
            step += 1.0

            if render and render_delay > 0:
                time.sleep(render_delay)

        compliance_rate = (compliant_steps / scored_steps) if scored_steps else float("nan")
        print(f"\nEpisode finished — return={R:.2f}, steps={int(step)}")
        if scored_steps:
            print(f"Compliance: {compliant_steps}/{scored_steps} ({compliance_rate * 100:.1f}%)")
        else:
            print("Compliance: n/a (no shaping for this instruction)")
        named_dist = {MACRO_ACTION_NAMES[a]: c for a, c in action_counts.items() if c > 0}
        print(f"Action distribution: {named_dist}")

    if render:
        time.sleep(0.5)
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Box-Pushing MacIAC instruction visualizer")
    parser.add_argument("--env_id", default="BP-MA-v0")
    parser.add_argument("--env_terminate_step", type=int, default=100)
    parser.add_argument("--grid_dim", type=int, nargs=2, default=[6, 6])
    parser.add_argument("--n_agent", type=int, default=2)
    parser.add_argument("--n_episode", type=int, default=1)
    parser.add_argument("--p_id", type=int, default=0,
                        help="Run id; tries this first then falls back to seeds 0..5")
    parser.add_argument("--instruction", type=str, default=None,
                        help='Fixed instruction text for every episode (e.g. "go to small box"). '
                             'Empty/None means "no instruction".')
    parser.add_argument("--interactive", action="store_true",
                        help="Prompt at the terminal for an instruction before each episode.")
    parser.add_argument("--no_render", dest="render", action="store_false",
                        help="Skip the pyglet viewer; useful for headless smoke tests.")
    parser.add_argument("--no_mask", dest="apply_mask", action="store_false",
                        help="Don't mask instruction-violating actions in the "
                             "logits — surfaces what the unmasked policy "
                             "actually picks.")
    parser.set_defaults(apply_mask=True)
    parser.add_argument("--render_delay", type=float, default=0.2,
                        help="Sleep between steps so the viewer is watchable.")
    parser.add_argument("--big_box_reward", type=int, default=300)
    parser.add_argument("--small_box_reward", type=int, default=20)
    parser.add_argument("--penalty", type=int, default=-10)
    parser.set_defaults(render=True)
    args = parser.parse_args()

    test(**vars(args))


if __name__ == "__main__":
    main()
