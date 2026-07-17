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
from macro_marl.cores.pg_based.mac_cac.utils import Agent
from macro_marl.cores.pg_based.mac_cac.models import Actor

# Shared overcooked-style input UX (pyglet handlers + in-window overlay).
sys.path.append(os.path.dirname(__file__))
from _bp_instruction_input import BPInstructionInput  # noqa: E402

# ---------------------------------------------------------------------------
# Hardcoded policy loading config.
# MacCAC uses ONE centralized actor shared across all agents — set
# POLICY_RUN_DIR to the run folder and the script will look for one of:
#   {POLICY_RUN_DIR}/{p_id}_agent_cen_MacCAC_run_{p_id}_*.pt   (CAC convention)
#   {POLICY_RUN_DIR}/{p_id}_agent_0.pt                         (legacy)
#   {POLICY_RUN_DIR}/joshua_cac.pt                             (legacy)
# ---------------------------------------------------------------------------
POLICY_BASE_DIR = "/home/willy/Documents/macro_marl_ppo/experiments/BoxPushing/policy_nns"
POLICY_RUN_DIR = "ma_cac_bp6_stop_push_ignore_pen-100_dur10_1"

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

    Empty/None text means "no instruction this step". The CAC actor
    handles `instruction_emb=None` internally (it builds the right zero
    tensor for the configured fusion), so we don't synthesize one here —
    the CAC `Actor` doesn't expose `instruction_dim` the way the IAC
    actor does.
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
    """Mirror src/macro_marl/cores/pg_based/mac_cac/envs_runner.py routing."""
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
    """Sample joint action from the centralized CAC actor.

    Mirrors src/macro_marl/cores/pg_based/mac_cac/controller.py: feed the
    joint observation into the central actor, then optionally mask
    instruction-violating joint indices before sampling.

    apply_mask=False shows what the unmasked policy actually picks at
    decision time — same condition the runner sees during training and
    eval. Use this when comparing the visualization to the wandb
    compliance numbers; with apply_mask=True the policy is forced into
    compliance regardless of what it learned.
    """
    central_actor = agents[0].policy_net  # all agents share the central actor

    # Build the joint obs (concat all per-agent observations, then pad/
    # truncate to the actor's expected input width).
    joint_obs = torch.cat([o.view(-1) for o in obs_list], dim=0)
    expected_input = agents[0].expected_input_dim
    if joint_obs.shape[0] < expected_input:
        joint_obs = torch.cat(
            [joint_obs, torch.zeros(expected_input - joint_obs.shape[0])]
        )
    elif joint_obs.shape[0] > expected_input:
        joint_obs = joint_obs[:expected_input]

    instruction_emb = instruction_to_embedding(
        current_instruction, central_actor, device=joint_obs.device,
    )

    with torch.no_grad():
        joint_logits, new_h = central_actor(
            joint_obs.view(1, 1, expected_input),
            h_states_list[0],
            instruction_emb=instruction_emb,
        )

    # Flatten over the joint action axis. CAC's actor outputs
    # n_action[0]*n_action[1]*... logits; np.unravel_index decodes joint
    # index -> per-agent action tuple in row-major order.
    n_actions_per_agent = list(env.n_action)
    flat_logits = joint_logits.view(-1).clone()
    n_joint = int(np.prod(n_actions_per_agent))

    # Per-agent instruction masking applied across the joint grid.
    # For each joint index, decode (a0, a1, ...) and zero it out if any
    # agent's action violates that agent's expected behavior. Skipped
    # entirely when apply_mask=False so the visualizer surfaces the
    # unmasked policy decision (matches what the runner sees at
    # training/eval time).
    expected_per_agent = [
        get_expected_actions(current_instruction, ag_idx)
        for ag_idx in range(env.n_agent)
    ]
    if apply_mask and any(exp is not None for exp in expected_per_agent):
        for j in range(n_joint):
            per_agent = np.unravel_index(j, n_actions_per_agent)
            mask_this = False
            for ag_idx, a in enumerate(per_agent):
                exp = expected_per_agent[ag_idx]
                if exp is None:
                    continue
                if "allowed_actions" in exp and a not in exp["allowed_actions"]:
                    mask_this = True
                    break
                if "prohibited_actions" in exp and a in exp["prohibited_actions"]:
                    mask_this = True
                    break
            if mask_this:
                flat_logits[j] = -float("inf")

    joint_action = Categorical(logits=flat_logits).sample().item()
    actions = list(np.unravel_index(joint_action, n_actions_per_agent))
    actions = [int(a) for a in actions]

    # CAC keeps a single shared h_state for the centralized actor; mirror
    # it across the agent slots so the caller's bookkeeping still works.
    new_h_states = [new_h] * env.n_agent

    if force_resample:
        for ag_idx, a in enumerate(actions):
            print(f"  Agent {ag_idx} resampled -> {MACRO_ACTION_NAMES[a]}")

    return actions, new_h_states


def get_init_inputs(env, n_agent):
    reset_result = env.reset()
    if isinstance(reset_result, list):
        return [torch.from_numpy(o).float() for o in reset_result], [None] * n_agent
    return [torch.from_numpy(reset_result).float() for _ in range(n_agent)], [None] * n_agent


def _interrupt_macros(env, n_agent):
    """Mid-episode instruction changes: terminate the current macro for
    every BP agent so the next step resamples under the new instruction."""
    inner = getattr(env, "unwrapped", env)
    for ag in getattr(inner, "agents", []) or []:
        cur = getattr(ag, "cur_action", None)
        if cur is not None and hasattr(cur, "t"):
            try:
                cur.t = 0
            except Exception:
                pass


def _find_central_policy(policy_dir, p_id, ckpt_filename=None):
    """Locate the single centralized policy file MacCAC saves.

    The CAC checkpoint filename pattern is
        {p_id}_agent_cen_{save_dir}__seed{p_id}__{YYYYMMDD-HHMMSS}.pt
    plus optional `_ep{lo}-{hi}` mid-training snapshots. We pick the most
    recent FINAL (no `_ep…` suffix) file for the requested p_id, or fall
    back to legacy single-file conventions.

    If `ckpt_filename` is provided (basename or absolute path), it overrides
    discovery — useful when the user wants to pin a specific timestamp.
    """
    if ckpt_filename:
        if os.path.isabs(ckpt_filename) and os.path.exists(ckpt_filename):
            return ckpt_filename
        candidate = os.path.join(policy_dir, ckpt_filename)
        if os.path.exists(candidate):
            return candidate

    if not os.path.isdir(policy_dir):
        return None

    prefix = f"{p_id}_agent_cen_"
    final_matches = []  # files with no `_ep…-…` mid-train suffix
    other_matches = []
    for f in os.listdir(policy_dir):
        if not (f.startswith(prefix) and f.endswith(".pt")):
            continue
        stem = f[:-3]  # drop .pt
        # Heuristic: skip mid-training "_ep<lo>-<hi>" snapshots in favor of
        # the final ones. They share the same prefix otherwise.
        if "_ep" in stem and stem.rsplit("_ep", 1)[-1].split("-", 1)[0].isdigit():
            other_matches.append(os.path.join(policy_dir, f))
        else:
            final_matches.append(os.path.join(policy_dir, f))

    if final_matches:
        return sorted(final_matches)[-1]  # newest timestamp lexicographically
    if other_matches:
        return sorted(other_matches)[-1]

    for fname in (f"{p_id}_agent_0.pt", "joshua_cac.pt", "agent_state_dict_0.pt"):
        path = os.path.join(policy_dir, fname)
        if os.path.exists(path):
            return path

    # Last-ditch: any *.pt
    pts = sorted(f for f in os.listdir(policy_dir) if f.endswith(".pt"))
    return os.path.join(policy_dir, pts[0]) if pts else None


def load_central_actor(env, policy_path):
    """Load one shared central actor and detect its expected obs width.

    Handles both full-Actor pickles and state-dict-only checkpoints; if a
    state dict is loaded, we infer use_instructions, instruction_dim, and
    fc1.in_features from the weight shapes (mirrors test_overcooked_cac.py).
    """
    loaded = torch.load(policy_path, map_location="cpu", weights_only=False)
    if isinstance(loaded, Actor):
        actor_net = loaded
        model_input_dim = actor_net.fc1.in_features
        fusion = getattr(actor_net, "instruction_fusion", "unknown")
        uses_inst = getattr(actor_net, "use_instructions", "unknown")
        print(f"Loaded full Actor object: fc1.in_features={model_input_dim}, "
              f"use_instructions={uses_inst}, fusion={fusion}")
        actor_net.eval()
        return actor_net, model_input_dim

    state_dict = loaded["actor_net_state_dict"] if (
        isinstance(loaded, dict) and "actor_net_state_dict" in loaded
    ) else loaded

    fc1_in = state_dict["fc1.weight"].shape[1]
    fc4_out = state_dict["fc4.weight"].shape[0]
    has_inst = any(("instruction_encoder" in k) or ("distilbert" in k)
                   or ("instruction_projection" in k) for k in state_dict.keys())
    instruction_dim = 0
    if has_inst:
        if "instruction_projection.weight" in state_dict:
            instruction_dim = state_dict["instruction_projection.weight"].shape[0]
        else:
            instruction_dim = 768
    model_input_dim = fc1_in - instruction_dim if has_inst else fc1_in

    actor_net = Actor(
        input_dim=model_input_dim,
        output_dim=fc4_out,
        use_instructions=has_inst,
        instruction_fusion="concat",
    )
    actor_net.load_state_dict(state_dict)
    actor_net.eval()
    print(f"Loaded state dict: input_dim={model_input_dim}, output_dim={fc4_out}, "
          f"use_instructions={has_inst}, instruction_dim={instruction_dim}")
    return actor_net, model_input_dim


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
         big_box_reward, small_box_reward, penalty, ckpt=None,
         apply_mask=True):
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
    print(f"Box Pushing CAC instruction-test — grid {grid_dim}, n_agent={n_agent}")
    print(f"{'='*60}")
    print(f"Env obs_size: {env.obs_size}, n_action: {env.n_action}")

    policy_dir = resolve_policy_dir()
    print(f"Loading centralized policy from: {policy_dir}")
    policy_path = _find_central_policy(policy_dir, p_id, ckpt_filename=ckpt)
    if policy_path is None:
        print(f"\nERROR: No policy file found in {policy_dir}")
        sys.exit(1)
    print(f"  using: {os.path.basename(policy_path)}")

    central_actor, model_input_dim = load_central_actor(env, policy_path)

    # All agents share the same central actor.
    agents = []
    for i in range(n_agent):
        agent = Agent()
        agent.idx = i
        agent.policy_net = central_actor
        if getattr(central_actor, "instruction_fusion", None) == "concat":
            instr_dim = (
                getattr(central_actor, "instruction_dim", None)
                or getattr(central_actor, "n_instructions", None)
                or 32
            )
            agent.expected_input_dim = model_input_dim
        else:
            agent.expected_input_dim = model_input_dim
        agents.append(agent)
        print(f"  Agent {i}: shared central actor, expected_input_dim={agent.expected_input_dim}")

    fixed_instruction = instruction or None
    discount = 0.95

    # Shared overcooked-style 't'-to-edit instruction controller. Attached
    # below once the pyglet window exists.
    inst_input = BPInstructionInput(instruction_list=INSTRUCTION_LIST)

    for e in range(n_episode):
        if interactive:
            picked = prompt_instruction()
            if picked == "__QUIT__":
                break
            current_instruction = picked
        else:
            current_instruction = fixed_instruction
        # Seed the controller with whatever the CLI/interactive prompt set
        # so the first overlay open reflects the active instruction.
        inst_input.current_instruction = current_instruction

        print(f"\n--- Episode {e + 1}/{n_episode} ---")
        print(f"Instruction: {current_instruction!r}")

        last_obs, h_states = get_init_inputs(env, n_agent)
        last_valid = [1.0] * n_agent

        # The viewer's pyglet window is created lazily inside the first
        # render() call. Trigger one render so the window exists, then hook
        # our keyboard handlers + overlay onto it.
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
            # If the user pressed 't', the controller's pyglet handlers
            # have already paused via is_active(); we keep redrawing the
            # frozen frame so the overlay shows up live.
            if inst_input.is_active():
                if render:
                    try:
                        env.render()
                    except Exception:
                        pass
                    if render_delay > 0:
                        time.sleep(render_delay)
                continue

            # Pick up any instruction change committed via the overlay.
            if inst_input.consume_change():
                current_instruction = inst_input.current_instruction
                _interrupt_macros(env, n_agent)
                last_valid = [1.0] * n_agent

            actions, h_states = get_actions_and_h_states(
                env, agents, last_valid, last_obs, h_states, current_instruction,
                apply_mask=apply_mask,
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
    parser = argparse.ArgumentParser(description="Box-Pushing MacCAC instruction visualizer")
    parser.add_argument("--env_id", default="BP-MA-v0")
    parser.add_argument("--env_terminate_step", type=int, default=100)
    parser.add_argument("--grid_dim", type=int, nargs=2, default=[6, 6])
    parser.add_argument("--n_agent", type=int, default=2)
    parser.add_argument("--n_episode", type=int, default=1)
    parser.add_argument("--p_id", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Pin a specific checkpoint by basename (relative to "
                             "POLICY_RUN_DIR) or absolute path. Overrides automatic "
                             "discovery. e.g. 0_agent_cen_..._seed0__YYYYMMDD-HHMMSS.pt")
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--no_render", dest="render", action="store_false")
    parser.add_argument("--no_mask", dest="apply_mask", action="store_false",
                        help="Skip the visualizer's instruction-action mask. "
                             "When set, the policy is free to pick any action "
                             "(including instruction-violating ones) — useful "
                             "for comparing what the policy actually learned "
                             "against the wandb compliance numbers.")
    parser.set_defaults(apply_mask=True)
    parser.add_argument("--render_delay", type=float, default=0.8)
    parser.add_argument("--big_box_reward", type=int, default=300)
    parser.add_argument("--small_box_reward", type=int, default=20)
    parser.add_argument("--penalty", type=int, default=-10)
    parser.set_defaults(render=True)
    args = parser.parse_args()

    test(**vars(args))


if __name__ == "__main__":
    main()
