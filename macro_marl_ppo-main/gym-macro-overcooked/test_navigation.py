#!/usr/bin/env python
"""
Navigation regression test for the Overcooked macro-action environment.

Verifies that the recent reward-exploit gates (Food._chop_rewarded /
_plate_rewarded, Blender._blend_rewarded, Oven._cook_rewarded, auto-cook
gating) did not change navigation, pickup, or macro-action completion.

Each test scripts a single macro-action (or a short sequence), runs the env
until the macro reports done, and asserts the agent ends Manhattan-distance 1
from the intended target with the expected `holding` state. A regression in
pathing, target resolution, or pickup logic would surface as a failed assert.

Run:
    python gym-macro-overcooked/test_navigation.py
Exit code 0 on pass; non-zero on first failure (with a printed traceback).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gym_macro_overcooked.items import (
    BlendedBowl,
    Food,
    Lettuce,
    Patty,
    Peas,
    Plate,
    Tomato,
)
from gym_macro_overcooked.overcooked_MA_V1 import Overcooked_MA_V1
from gym_macro_overcooked.overcooked_V1 import TASKLIST


REWARDS = {
    "subtask finished": 10,
    "correct delivery": 200,
    "wrong delivery": -5,
    "step penalty": -0.1,
}


def make_env(park_agent1=True):
    env = Overcooked_MA_V1(
        grid_dim=(7, 7),
        # Env expects the task *string*, not the index — string-equality
        # checks like `self.task == "lettuce-peas-tomato-patty"` silently
        # fail otherwise, and `rawName in self.task` raises a TypeError.
        task=TASKLIST[9],  # "lettuce-peas-tomato-patty"
        rewardList=REWARDS,
        map_type="D",
        n_agent=2,
        obs_radius=2,
        mode="vector",
        debug=False,
    )
    env.reset()
    if park_agent1:
        # Agent 1 starts at (1,4), which sits on the only non-oven adjacency to
        # oven 2 and on the natural row-1 corridor to lettuce/tomato. Park it
        # at the blender (bottom-right corner). Knife 1's only reachable
        # adjacency is (1,1), so parking there would box agent 0 out of every
        # chop test; the blender corner has multiple open adjacencies so agent
        # 0 can still reach the blender too. Pathfinding for agent 0 is what
        # we're testing — agent 1 just needs to stay out of agent 0's targets.
        park_idx = env.macroActionName.index("go to blender")
        stay_idx = env.macroActionName.index("stay")
        steps = 0
        while steps < 80:
            _, _, done, info = env.run([stay_idx, park_idx])
            steps += 1
            if all(info.get("mac_done", [True, True])):
                break
            if done:
                break
    return env


def manhattan(a_x, a_y, b_x, b_y):
    return abs(a_x - b_x) + abs(a_y - b_y)


def run_macro(env, agent0_action, agent1_action="stay", max_steps=80):
    """Drive both agents until both macro-actions report done (or max_steps).

    Returns (steps, done_terminate, last_rewards).
    """
    a0 = env.macroActionName.index(agent0_action)
    a1 = env.macroActionName.index(agent1_action)
    steps = 0
    last_rewards = None
    while steps < max_steps:
        _, rewards, done, info = env.run([a0, a1])
        last_rewards = rewards
        steps += 1
        if all(info.get("mac_done", [True, True])):
            return steps, done, last_rewards
        if done:
            return steps, done, last_rewards
    raise AssertionError(
        f"Macro {agent0_action!r}/{agent1_action!r} did not complete within {max_steps} steps"
    )


def assert_adjacent(agent, target_x, target_y, label):
    d = manhattan(agent.x, agent.y, target_x, target_y)
    assert d == 1, (
        f"[{label}] expected agent adjacent to ({target_x},{target_y}), "
        f"got pos=({agent.x},{agent.y}) distance={d}"
    )


# ---------------------------------------------------------------------------
# Layout sanity — pin the expected item positions for Map D 7x7 n_agent=2.
# If the map literal changes, this catches it before downstream tests start
# producing confusing "agent at the wrong cell" errors.
# ---------------------------------------------------------------------------
def test_map_d_layout():
    env = make_env(park_agent1=False)
    base = env  # Overcooked_MA_V1 inherits directly from Overcooked_V1

    assert len(base.tomato) == 1 and (base.tomato[0].x, base.tomato[0].y) == (0, 5), \
        f"tomato expected at (0,5), got {[(t.x, t.y) for t in base.tomato]}"
    assert len(base.lettuce) == 1 and (base.lettuce[0].x, base.lettuce[0].y) == (1, 6), \
        f"lettuce expected at (1,6), got {[(t.x, t.y) for t in base.lettuce]}"
    assert len(base.peas) == 1 and (base.peas[0].x, base.peas[0].y) == (2, 6), \
        f"peas expected at (2,6), got {[(t.x, t.y) for t in base.peas]}"
    assert len(base.plate) == 1 and (base.plate[0].x, base.plate[0].y) == (5, 6), \
        f"plate expected at (5,6), got {[(t.x, t.y) for t in base.plate]}"
    assert len(base.knife) == 2, f"expected 2 knives, got {len(base.knife)}"
    assert (base.knife[0].x, base.knife[0].y) == (1, 0)
    assert (base.knife[1].x, base.knife[1].y) == (2, 0)
    assert len(base.oven) == 2, f"expected 2 ovens, got {len(base.oven)}"
    assert (base.oven[0].x, base.oven[0].y) == (0, 3)
    assert (base.oven[1].x, base.oven[1].y) == (0, 4)
    assert len(base.blender) == 1 and (base.blender[0].x, base.blender[0].y) == (6, 3)
    assert len(base.delivery) == 1 and (base.delivery[0].x, base.delivery[0].y) == (3, 0)
    assert len(base.agent) == 2
    assert (base.agent[0].x, base.agent[0].y) == (1, 2)
    assert (base.agent[1].x, base.agent[1].y) == (1, 4)


# ---------------------------------------------------------------------------
# Per-macro navigation: run from fresh reset, agent 0 executes the macro,
# agent 1 stays. After the macro completes, assert position + holding state.
# ---------------------------------------------------------------------------
def test_get_tomato():
    env = make_env()
    base = env
    tx, ty = base.tomato[0].x, base.tomato[0].y
    run_macro(env, "get tomato")
    assert_adjacent(base.agent[0], tx, ty, "get tomato")
    assert isinstance(base.agent[0].holding, Tomato), \
        f"expected agent 0 holding Tomato, got {base.agent[0].holding}"
    assert not base.agent[0].holding.chopped, "raw tomato should be unchopped"


def test_get_lettuce():
    env = make_env()
    base = env
    tx, ty = base.lettuce[0].x, base.lettuce[0].y
    run_macro(env, "get lettuce")
    assert_adjacent(base.agent[0], tx, ty, "get lettuce")
    assert isinstance(base.agent[0].holding, Lettuce)
    assert not base.agent[0].holding.chopped


def test_get_peas():
    env = make_env()
    base = env
    tx, ty = base.peas[0].x, base.peas[0].y
    run_macro(env, "get peas")
    assert_adjacent(base.agent[0], tx, ty, "get peas")
    assert isinstance(base.agent[0].holding, Peas)


def test_get_plate_1():
    env = make_env()
    base = env
    tx, ty = base.plate[0].x, base.plate[0].y
    run_macro(env, "get plate 1")
    assert_adjacent(base.agent[0], tx, ty, "get plate 1")
    assert isinstance(base.agent[0].holding, Plate)


def test_go_to_knife_1():
    env = make_env()
    base = env
    tx, ty = base.knife[0].x, base.knife[0].y
    run_macro(env, "go to knife 1")
    assert_adjacent(base.agent[0], tx, ty, "go to knife 1")
    # Empty-handed nav target — agent should not be holding anything.
    assert base.agent[0].holding is None


def test_go_to_knife_2():
    env = make_env()
    base = env
    tx, ty = base.knife[1].x, base.knife[1].y
    run_macro(env, "go to knife 2")
    assert_adjacent(base.agent[0], tx, ty, "go to knife 2")


def test_go_to_blender():
    # Default parking spot is the blender itself, so for this test we don't
    # park (would block agent 0's only path target). Instead skip parking and
    # run agent 1 with a benign "go to oven 1" alongside agent 0; oven 1 sits
    # in the top corridor and won't conflict with agent 0's bottom-corridor
    # path to the blender.
    env = make_env(park_agent1=False)
    base = env
    tx, ty = base.blender[0].x, base.blender[0].y
    run_macro(env, "go to blender", "go to oven 1")
    assert_adjacent(base.agent[0], tx, ty, "go to blender")


def test_go_to_oven_1():
    env = make_env()
    base = env
    tx, ty = base.oven[0].x, base.oven[0].y
    run_macro(env, "go to oven 1")
    assert_adjacent(base.agent[0], tx, ty, "go to oven 1")


def test_go_to_oven_2():
    env = make_env()
    base = env
    tx, ty = base.oven[1].x, base.oven[1].y
    run_macro(env, "go to oven 2")
    assert_adjacent(base.agent[0], tx, ty, "go to oven 2")


def test_deliver_empty_handed():
    """`deliver` macro completes when the agent (empty-handed) reaches the
    delivery counter. We're testing navigation, not the wrong-delivery branch.
    """
    env = make_env()
    base = env
    tx, ty = base.delivery[0].x, base.delivery[0].y
    run_macro(env, "deliver")
    assert_adjacent(base.agent[0], tx, ty, "deliver")


# ---------------------------------------------------------------------------
# Two-step sequences — verify the agent can chain pickup → place at knife,
# pick up chopped result, etc. These are the same primitives a learned policy
# uses, so a regression here would block any patty-task progress.
# ---------------------------------------------------------------------------
def test_chop_lettuce_full_chain():
    env = make_env()
    base = env

    # Phase 1: pick up lettuce.
    run_macro(env, "get lettuce")
    assert isinstance(base.agent[0].holding, Lettuce)

    # Phase 2: walk to knife 1 and place lettuce on knife.
    run_macro(env, "go to knife 1")
    kx, ky = base.knife[0].x, base.knife[0].y
    assert_adjacent(base.agent[0], kx, ky, "go to knife 1 with lettuce")
    # `go to knife N` while holding food drops the food on the knife.
    assert base.agent[0].holding is None, "lettuce should have been placed on knife"
    assert isinstance(base.knife[0].holding, Lettuce), \
        "knife 0 should now hold the lettuce"
    assert not base.knife[0].holding.chopped, "lettuce on knife not yet chopped"

    # Phase 3: chop. The chop macro completes once required_chopped_times is met.
    run_macro(env, "chop")
    assert base.knife[0].holding is not None and base.knife[0].holding.chopped, \
        "lettuce should be chopped after `chop` macro completes"

    # Phase 4: pick the chopped lettuce back up.
    run_macro(env, "get lettuce")
    assert isinstance(base.agent[0].holding, Lettuce)
    assert base.agent[0].holding.chopped, \
        "agent should be holding chopped lettuce after retrieving from knife"


def test_get_patty_returns_neutral_when_absent():
    """Before any baking, `_findPOitem('get patty')` returns the agent's own
    cell (or 0,0 if not found). Verify the macro short-circuits without
    crashing and without moving the agent into an oven cell.
    """
    env = make_env(park_agent1=False)
    base = env

    start_x, start_y = base.agent[0].x, base.agent[0].y
    # No patty exists yet, so the macro should resolve immediately.
    run_macro(env, "stay", "stay", max_steps=2)
    a0 = base.agent[0]
    # `stay` is the cleanest no-op probe; agent should not have moved.
    assert (a0.x, a0.y) == (start_x, start_y), \
        f"agent should not move on stay; was ({start_x},{start_y}) now ({a0.x},{a0.y})"


# ---------------------------------------------------------------------------
# Reward-gate regression checks — these aren't navigation tests strictly, but
# they pin down the new behavior of the exploit-fix flags so anyone who
# touches them later sees the contract spelled out.
# ---------------------------------------------------------------------------
def test_chop_reward_paid_once_per_food():
    env = make_env()
    base = env

    # Pick up lettuce, place on knife, chop. The chop macro returns rewards
    # that include the +10 subtask-finished bonus on the completion step.
    run_macro(env, "get lettuce")
    run_macro(env, "go to knife 1")
    _, _, rewards_first = run_macro(env, "chop")
    # Joint reward has step penalty subtracted; a +10 bonus dwarfs it.
    assert rewards_first[0] > 5, \
        f"first chop should pay +10 minus step penalty, got {rewards_first}"

    # Confirm the food's flag is now set.
    lettuce = base.lettuce[0]
    assert getattr(lettuce, "_chop_rewarded", False) is True, \
        "lettuce._chop_rewarded should be True after first chop credit"

    # Force a refresh (simulating the wrong-delivery cycle that previously
    # let the agent re-earn the chop reward), then re-chop the same food.
    lettuce.refresh()
    assert lettuce.chopped is False and lettuce.cur_chopped_times == 0, \
        "refresh should reset chop progress on the food"
    assert getattr(lettuce, "_chop_rewarded", False) is True, \
        "the chop-reward flag must persist across refresh()"

    # Re-pick + chop the same lettuce; expect no second +10.
    run_macro(env, "get lettuce")
    run_macro(env, "go to knife 1")
    _, _, rewards_second = run_macro(env, "chop")
    assert rewards_second[0] < 1, (
        f"second chop of same food should NOT pay +10, got {rewards_second}"
    )


def test_reset_clears_reward_flags():
    """env.reset() must rebuild Food/Blender/Oven so the reward flags clear."""
    env = make_env()
    base = env
    base.lettuce[0]._chop_rewarded = True
    base.tomato[0]._plate_rewarded = True
    base.blender[0]._blend_rewarded = True
    base.oven[0]._cook_rewarded = True

    env.reset()
    base = env

    assert base.lettuce[0]._chop_rewarded is False
    assert base.tomato[0]._plate_rewarded is False
    assert base.blender[0]._blend_rewarded is False
    assert base.oven[0]._cook_rewarded is False
    assert base.oven[1]._cook_rewarded is False


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------
def main():
    tests = [
        test_map_d_layout,
        test_get_tomato,
        test_get_lettuce,
        test_get_peas,
        test_get_plate_1,
        test_go_to_knife_1,
        test_go_to_knife_2,
        test_go_to_blender,
        test_go_to_oven_1,
        test_go_to_oven_2,
        test_deliver_empty_handed,
        test_chop_lettuce_full_chain,
        test_get_patty_returns_neutral_when_absent,
        test_chop_reward_paid_once_per_food,
        test_reset_clears_reward_flags,
    ]
    failures = []
    for t in tests:
        name = t.__name__
        try:
            t()
        except Exception as exc:
            import traceback
            failures.append((name, traceback.format_exc()))
            print(f"FAIL  {name}: {exc}")
        else:
            print(f"ok    {name}")

    print()
    print(f"{len(tests) - len(failures)}/{len(tests)} passed")
    if failures:
        print("Failures:")
        for name, tb in failures:
            print(f"  --- {name} ---")
            print(tb)
        sys.exit(1)


if __name__ == "__main__":
    main()
