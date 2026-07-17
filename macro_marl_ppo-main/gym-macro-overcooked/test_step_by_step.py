#!/usr/bin/env python
"""Step-by-step test to understand macro-action behavior"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gym_macro_overcooked.overcooked_MA_V1 import Overcooked_MA_V1

def get_action_idx(env, action_name):
    return env.macroActionName.index(action_name)

def print_full_state(env, step_num, description=""):
    """Print detailed state"""
    print(f"\n{'='*70}")
    print(f"Step {step_num}: {description}")
    print(f"{'='*70}")
    
    for i, agent in enumerate(env.agent):
        holding = agent.holding.name if agent.holding else "Nothing"
        print(f"Agent {i}: pos=({agent.x},{agent.y}), holding={holding}")
    
    print(f"\nKnives:")
    for k in env.knife:
        holding = k.holding.name if k.holding else "Empty"
        print(f"  Knife at ({k.x},{k.y}): holding={holding}")
    
    if env.blender:
        b = env.blender[0]
        items = [f.name for f in b.containing]
        print(f"\nBlender at ({b.x},{b.y}): items={items}, blended={b.blended}")
    
    if env.oven:
        o = env.oven[0]
        items = [f.name for f in o.containing]
        print(f"Oven at ({o.x},{o.y}): items={items}, cooking={o.cooking}, cooked={o.cooked}")

def run_until_done(env, agent0_action, agent1_action, max_steps=30):
    """Run until both agents complete their macro-actions"""
    act0 = get_action_idx(env, agent0_action)
    act1 = get_action_idx(env, agent1_action)
    
    steps = 0
    while steps < max_steps:
        obs, rewards, done, info = env.run([act0, act1])
        steps += 1
        
        mac_done = info.get('mac_done', [True, True])
        if all(mac_done):
            break
        if done:
            break
    
    return steps, done

def test_cooperative_workflow():
    """Test with both agents working cooperatively to avoid blocking"""
    
    print("="*70)
    print("COOPERATIVE WORKFLOW TEST")
    print("="*70)
    
    env = Overcooked_MA_V1(
        grid_dim=(7, 7),
        task=9,
        rewardList={"subtask finished": 10, "correct delivery": 200, "wrong delivery": -5, "step penalty": -0.1},
        map_type="D",
        n_agent=2,
        obs_radius=2,
        mode="vector",
        debug=False
    )
    
    env.reset()
    print_full_state(env, 0, "Initial state")
    
    step_count = 0
    
    # Both agents get their items simultaneously
    print("\n--- Phase 1: Both agents get items ---")
    
    # Agent 0 gets lettuce, Agent 1 gets tomato - they go in different directions
    steps, done = run_until_done(env, "get lettuce", "get tomato")
    step_count += steps
    print(f"Steps: {steps}")
    print_full_state(env, step_count, "After getting items")
    
    # Both go to their respective knives
    print("\n--- Phase 2: Both go to knives ---")
    steps, done = run_until_done(env, "go to knife 1", "go to knife 2")
    step_count += steps
    print(f"Steps: {steps}")
    print_full_state(env, step_count, "At knives")
    
    # Both chop
    print("\n--- Phase 3: Both chop ---")
    steps, done = run_until_done(env, "chop", "chop")
    step_count += steps
    print(f"Steps: {steps}")
    print_full_state(env, step_count, "After chopping")
    
    # Get chopped items from knives
    print("\n--- Phase 4: Get chopped items ---")
    steps, done = run_until_done(env, "get lettuce", "get tomato")
    step_count += steps
    print(f"Steps: {steps}")
    print_full_state(env, step_count, "After getting chopped items")
    
    # Go to blender
    print("\n--- Phase 5: Go to blender ---")
    steps, done = run_until_done(env, "go to blender", "go to blender")
    step_count += steps
    print(f"Steps: {steps}")
    print_full_state(env, step_count, "At blender")
    
    # Agent 0 gets peas
    print("\n--- Phase 6: Get peas ---")
    steps, done = run_until_done(env, "get peas", "stay")
    step_count += steps
    print(f"Steps: {steps}")
    print_full_state(env, step_count, "After getting peas")
    
    # Agent 0 goes to blender with peas
    print("\n--- Phase 7: Go to blender with peas ---")
    steps, done = run_until_done(env, "go to blender", "stay")
    step_count += steps
    print(f"Steps: {steps}")
    print_full_state(env, step_count, "After adding peas to blender")
    
    # Check if blender has all items
    if env.blender:
        b = env.blender[0]
        print(f"\nBlender can blend: {b.can_blend()}")
    
    # Blend
    print("\n--- Phase 8: Blend ---")
    steps, done = run_until_done(env, "blend", "stay")
    step_count += steps
    print(f"Steps: {steps}")
    print_full_state(env, step_count, "After blending")
    
    # Pick up blended bowl
    print("\n--- Phase 9: Pick up blended bowl ---")
    steps, done = run_until_done(env, "go to blender", "stay")
    step_count += steps
    print(f"Steps: {steps}")
    print_full_state(env, step_count, "After picking up bowl")
    
    # Go to oven
    print("\n--- Phase 10: Go to oven ---")
    steps, done = run_until_done(env, "go to oven", "stay")
    step_count += steps
    print(f"Steps: {steps}")
    print_full_state(env, step_count, "After putting bowl in oven")
    
    # Wait for cooking (Agent 1 gets plate)
    print("\n--- Phase 11: Wait for cooking & get plate ---")
    steps, done = run_until_done(env, "stay", "get plate 1")
    step_count += steps
    print(f"Steps: {steps}")
    print_full_state(env, step_count, "Plate retrieved")
    
    # Wait more for cooking
    print("\n--- Phase 12: Wait for cooking to complete ---")
    for i in range(20):
        steps, done = run_until_done(env, "stay", "stay", max_steps=1)
        step_count += steps
        if env.oven and env.oven[0].cooked:
            print(f"Cooking complete after {i+1} wait steps")
            break
    print_full_state(env, step_count, "After cooking")
    
    # Agent 1 goes to oven with plate
    print("\n--- Phase 13: Get patty with plate ---")
    steps, done = run_until_done(env, "stay", "go to oven")
    step_count += steps
    print(f"Steps: {steps}")
    print_full_state(env, step_count, "After getting patty")
    
    # Deliver
    print("\n--- Phase 14: Deliver ---")
    steps, done = run_until_done(env, "stay", "deliver")
    step_count += steps
    print(f"Steps: {steps}")
    print_full_state(env, step_count, "After delivery")
    
    print(f"\n{'='*70}")
    print(f"TOTAL STEPS: {step_count}")
    print(f"DONE: {done}")
    print(f"{'='*70}")

if __name__ == "__main__":
    test_cooperative_workflow()
