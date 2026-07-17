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
from macro_marl.cores.pg_based.acac.utils import Agent
from macro_marl.cores.pg_based.acac.models import AgentCentricGRUActor

# Shared overcooked-style input UX (pyglet handlers + in-window overlay).
sys.path.append(os.path.dirname(__file__))
from _bp_instruction_input import BPInstructionInput  # noqa: E402


def _interrupt_macros(env):
    inner = getattr(env, "unwrapped", env)
    for ag in getattr(inner, "agents", []) or []:
        cur = getattr(ag, "cur_action", None)
        if cur is not None and hasattr(cur, "t"):
            try:
                cur.t = 0
            except Exception:
                pass

# ---------------------------------------------------------------------------
# Hardcoded policy loading config.
# ACAC saves per-agent actors as {p_id}_agent_{i}.pt under POLICY_RUN_DIR.
# ---------------------------------------------------------------------------
POLICY_BASE_DIR = "/home/willy/Documents/macro_marl_ppo/experiments/BoxPushing/policy_nns"
POLICY_RUN_DIR = "acac_bp10_instructions_stochastic_chain_break_0"

# Box-Pushing macro-action layout (must match BoxPushing_harder.MAs).
MACRO_ACTION_NAMES = ["GT_SB0", "GT_SB1", "GT_BB0", "GT_BB1", "PUSH", "T_L", "T_R", "STAY"]
GT_SMALL_BOX_0, GT_SMALL_BOX_1 = 0, 1
GT_BIG_BOX_SPOT_0, GT_BIG_BOX_SPOT_1 = 2, 3
PUSH, T_L, T_R, STAY = 4, 5, 6, 7

INSTRUCTION_LIST = [
    "go to small box",
    "don't push",
    "stop pushing the box",
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
    if POLICY_RUN_DIR and os.path.isabs(POLICY_RUN_DIR):
        return POLICY_RUN_DIR
    if POLICY_RUN_DIR:
        return os.path.join(POLICY_BASE_DIR, POLICY_RUN_DIR)
    return POLICY_BASE_DIR


def instruction_to_embedding(text, model, device=None):
    """Return an instruction embedding tensor or None.

    Empty/None text means "no instruction this step". ACAC's actor builds
    an internal `torch.zeros(..., n_instructions)` when `instruction_emb`
    is None, so returning None here is the safe path.
    """
    if not getattr(model, "use_instructions", False):
        return None
    if not text:
        return None
    if hasattr(model, "encode_instruction"):
        with torch.no_grad():
            emb = model.encode_instruction(text)
            if device is not None:
                emb = emb.to(device)
        return emb
    # One-hot ACAC build with no encoder — can't synthesize from raw text.
    return None


def get_expected_actions(instruction_text, agent_idx):
    """Mirror src/macro_marl/cores/pg_based/acac/envs_runner.py routing."""
    canonical = _normalize_instruction(instruction_text)
    if canonical is None:
        return None
    canonical = INSTRUCTION_ALIASES.get(canonical, canonical)

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


def get_actions_and_h_states(env, agents, last_valid, obs_list, h_states_list,
                             current_instruction, force_resample=False,
                             apply_mask=True):
    """apply_mask=False shows what the unmasked policy actually picks."""
    actions, new_h_states = [], []
    with torch.no_grad():
        for agent in agents:
            obs = obs_list[agent.idx]
            if obs.shape[0] < agent.expected_input_dim:
                obs = torch.cat([obs, torch.zeros(agent.expected_input_dim - obs.shape[0])])
            elif obs.shape[0] > agent.expected_input_dim:
                obs = obs[:agent.expected_input_dim]

            instruction_emb = instruction_to_embedding(
                current_instruction, agent.actor_net, device=obs.device,
            )
            # AgentCentricGRUActor signature:
            #   forward(x, h=None, eps=0.0, test_mode=False, time_emb=None, instruction_emb=None)
            action_logits, new_h = agent.actor_net(
                obs.view(1, 1, agent.expected_input_dim),
                h_states_list[agent.idx],
                instruction_emb=instruction_emb,
            )
            n_actions = env.n_action[agent.idx]
            logits = action_logits[:, :, :n_actions].clone()

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
    """ACAC stores per-agent actors. Use {p_id}_agent_{i}.pt; if missing,
    fall back to seeds 0..5 then to the agent_state_dict_*.pt convention."""
    available = []
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
            available = candidate
            selected_run_id = rid
            break

    if not available:
        for i in range(n_agent):
            for fname in (f"agent_state_dict_{i}.pt",):
                path = os.path.join(policy_dir, fname)
                if os.path.exists(path):
                    available.append((i, path))
                    break

    if not available:
        print(f"\nERROR: No policy files found in {policy_dir}")
        if os.path.isdir(policy_dir):
            print("Available .pt files:")
            for f in sorted(os.listdir(policy_dir)):
                if f.endswith(".pt"):
                    print(f"  - {f}")
        sys.exit(1)

    if selected_run_id is not None and selected_run_id != p_id:
        print(f"Requested p_id={p_id} not found; using run_id={selected_run_id} instead.")

    print(f"\nLoaded {len(available)} policies:")
    for agent_idx, path in available:
        print(f"  Agent {agent_idx}: {os.path.basename(path)}")

    agents = []
    for i in range(n_agent):
        agent = Agent()
        agent.idx = i
        path = next((p for j, p in available if j == i), None)
        if path is None:
            print(f"ERROR: Policy file not found for agent {i}")
            sys.exit(1)

        loaded = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(loaded, AgentCentricGRUActor):
            agent.actor_net = loaded
            # Detect input width from the actor's encoder.
            enc = getattr(loaded, "encoder", None)
            fc1 = getattr(enc, "fc1", None) if enc is not None else None
            if fc1 is not None and hasattr(fc1, "in_features"):
                model_input_dim = fc1.in_features
            elif hasattr(loaded, "fc1") and hasattr(loaded.fc1, "in_features"):
                model_input_dim = loaded.fc1.in_features
            else:
                model_input_dim = env.obs_size[i]
        else:
            # State-dict path: build a default actor with env-shaped dims.
            model_input_dim = env.obs_size[i]
            agent.actor_net = AgentCentricGRUActor(
                input_dim=model_input_dim,
                output_dim=env.n_action[i],
                use_instructions=True,
            )
            state_dict = loaded["actor_net_state_dict"] if (
                isinstance(loaded, dict) and "actor_net_state_dict" in loaded
            ) else loaded
            agent.actor_net.load_state_dict(state_dict)

        agent.actor_net.eval()
        agent.expected_input_dim = model_input_dim
        agents.append(agent)
        print(f"  Agent {i}: expected_input_dim={agent.expected_input_dim}")
    return agents


def prompt_instruction():
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
    print(f"Box Pushing ACAC instruction-test — grid {grid_dim}, n_agent={n_agent}")
    print(f"{'='*60}")
    print(f"Env obs_size: {env.obs_size}, n_action: {env.n_action}")

    policy_dir = resolve_policy_dir()
    print(f"Loading policies from: {policy_dir}")
    agents = load_agents(env, n_agent, policy_dir, p_id)

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

        compliant_steps, scored_steps = 0, 0
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
                _interrupt_macros(env)
                last_valid = [1.0] * n_agent

            actions, h_states = get_actions_and_h_states(
                env, agents, last_valid, last_obs, h_states, current_instruction,
            )
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
    parser = argparse.ArgumentParser(description="Box-Pushing ACAC instruction visualizer")
    parser.add_argument("--env_id", default="BP-MA-v0")
    parser.add_argument("--env_terminate_step", type=int, default=100)
    parser.add_argument("--grid_dim", type=int, nargs=2, default=[10, 10])
    parser.add_argument("--n_agent", type=int, default=2)
    parser.add_argument("--n_episode", type=int, default=1)
    parser.add_argument("--p_id", type=int, default=0)
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--no_render", dest="render", action="store_false")
    parser.add_argument("--render_delay", type=float, default=0.2)
    parser.add_argument("--big_box_reward", type=int, default=300)
    parser.add_argument("--small_box_reward", type=int, default=20)
    parser.add_argument("--penalty", type=int, default=-10)
    parser.set_defaults(render=True)
    args = parser.parse_args()

    test(**vars(args))


if __name__ == "__main__":
    main()
