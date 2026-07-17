#!/usr/bin/env python
"""
Test script to verify the full patty-making workflow with predetermined macro-actions.
This tests Map D with the lettuce-peas-tomato-patty task.

Workflow:
1. Chop lettuce and put in blender
2. Chop tomato and put in blender  
3. Get peas and put in blender
4. Blend (5 timesteps)
5. Pick up blended bowl and put in oven
6. Oven auto-cooks (10 timesteps)
7. Get plate and pick up patty from oven
8. Deliver

Controls (during visualization):
- SPACE: Pause/Resume
- UP/DOWN: Adjust speed
- Q/ESC: Quit
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pygame
import time

from gym_macro_overcooked.overcooked_MA_V1 import Overcooked_MA_V1

# Visualization settings
RENDER_DELAY = 0.01  # Delay between frames in seconds (adjustable with arrow keys)
MIN_DELAY = 0.02
MAX_DELAY = 1.0
DELAY_STEP = 0.02

def get_action_idx(env, action_name):
    """Get the action index for a named macro-action"""
    return env.macroActionName.index(action_name)

def print_state(env, step_num, description=""):
    """Print current state for debugging"""
    print(f"\n{'='*60}")
    print(f"Step {step_num}: {description}")
    print(f"{'='*60}")
    
    for i, agent in enumerate(env.agent):
        holding = agent.holding.name if agent.holding else "Nothing"
        print(f"Agent {i}: pos=({agent.x},{agent.y}), holding={holding}")
    
    if env.blender:
        b = env.blender[0]
        print(f"Blender: containing={len(b.containing)} items, blended={b.blended}")
        for f in b.containing:
            print(f"  - {f.name} (chopped={getattr(f, 'chopped', 'N/A')})")
    
    if env.oven:
        o = env.oven[0]
        print(f"Oven: cooking={o.cooking}, cooked={o.cooked}, cook_times={o.cur_cook_times}, name={o.name}")
        if o.containing:
            print(f"  Containing: {[f.name for f in o.containing]}")

def handle_pygame_events(render_delay):
    """Handle pygame events and return updated delay, paused state, and quit flag"""
    paused = False
    quit_requested = False
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            quit_requested = True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                quit_requested = True
            elif event.key == pygame.K_SPACE:
                paused = not paused
            elif event.key == pygame.K_UP:
                render_delay = max(MIN_DELAY, render_delay - DELAY_STEP)
                print(f"Speed up: delay = {render_delay:.2f}s")
            elif event.key == pygame.K_DOWN:
                render_delay = min(MAX_DELAY, render_delay + DELAY_STEP)
                print(f"Slow down: delay = {render_delay:.2f}s")
    
    return render_delay, paused, quit_requested

def wait_with_events(delay, env):
    """Wait for specified delay while handling pygame events"""
    start_time = time.time()
    current_delay = delay
    
    while time.time() - start_time < current_delay:
        current_delay, paused, quit_requested = handle_pygame_events(current_delay)
        
        if quit_requested:
            return current_delay, True
        
        # Handle pause
        while paused:
            env.render()
            current_delay, paused, quit_requested = handle_pygame_events(current_delay)
            if quit_requested:
                return current_delay, True
            time.sleep(0.05)
        
        time.sleep(0.01)  # Small sleep to prevent CPU hogging
    
    return current_delay, False

def run_macro_action(env, agent0_action, agent1_action, description, max_steps=50, verbose=True, render_delay=RENDER_DELAY):
    """Run a macro-action until both agents complete their actions"""
    act0 = get_action_idx(env, agent0_action)
    act1 = get_action_idx(env, agent1_action)
    
    steps = 0
    total_reward = 0
    done = False
    current_delay = render_delay
    
    while steps < max_steps:
        actions = [act0, act1]
        obs, rewards, done, info = env.run(actions)
        steps += 1
        total_reward += sum(rewards)
        
        # Render the current state
        env.render()
        
        # Wait and handle events
        current_delay, quit_requested = wait_with_events(current_delay, env)
        if quit_requested:
            return steps, total_reward, True, current_delay
        
        # Check if both agents have completed their macro-actions
        mac_done = info.get('mac_done', [True, True])
        if all(mac_done):
            break
        
        if done:
            break
    
    if verbose:
        print(f"\n[{description}] Took {steps} steps, reward={total_reward:.1f}")
        for i, agent in enumerate(env.agent):
            holding = agent.holding.name if agent.holding else "Nothing"
            print(f"  Agent {i}: pos=({agent.x},{agent.y}), holding={holding}")
    
    return steps, total_reward, done, current_delay

def test_patty_workflow():
    """Test the full patty making workflow with predetermined actions"""
    
    print("="*70)
    print("PATTY WORKFLOW TEST - Map D (with Pygame Visualization)")
    print("="*70)
    print("\nControls:")
    print("  SPACE: Pause/Resume")
    print("  UP: Speed up")
    print("  DOWN: Slow down")
    print("  Q/ESC: Quit")
    print("="*70)
    
    # Create environment - Map D, 2 agents, 7x7
    # Set debug=True to enable pygame rendering
    env = Overcooked_MA_V1(
        grid_dim=(7, 7),
        task=9,  # lettuce-peas-tomato-patty
        rewardList={"subtask finished": 10, "correct delivery": 200, "wrong delivery": -5, "step penalty": -0.1},
        map_type="D",
        n_agent=2,
        obs_radius=2,
        mode="vector",
        debug=True  # Enable pygame rendering
    )
    
    print("\nAvailable macro-actions:")
    for i, name in enumerate(env.macroActionName):
        print(f"  {i}: {name}")
    
    env.reset()
    
    # Initial render
    env.render()
    pygame.display.set_caption("Overcooked - Patty Workflow Test")
    
    print_state(env, 0, "Initial state")
    
    # Wait a moment at the start
    time.sleep(0.5)
    
    total_steps = 0
    total_reward = 0
    render_delay = RENDER_DELAY
    
    # ========================================
    # PHASE 1: Get and chop lettuce and tomato
    # ========================================
    print("\n" + "="*70)
    print("PHASE 1: Get and chop lettuce/tomato")
    print("="*70)
    
    # Agent 0 gets lettuce, Agent 1 gets tomato
    s, r, done, render_delay = run_macro_action(env, "get lettuce", "get tomato", "Get lettuce and tomato", render_delay=render_delay)
    total_steps += s; total_reward += r
    if done: return check_result(env, total_steps, total_reward)
    
    # Both go to their knives
    s, r, done, render_delay = run_macro_action(env, "go to knife 1", "go to knife 2", "Go to knives", render_delay=render_delay)
    total_steps += s; total_reward += r
    if done: return check_result(env, total_steps, total_reward)
    
    # Both chop (3 times each for chopping)
    s, r, done, render_delay = run_macro_action(env, "chop", "chop", "Chop lettuce and tomato", render_delay=render_delay)
    total_steps += s; total_reward += r
    if done: return check_result(env, total_steps, total_reward)
    
    print_state(env, total_steps, "After chopping")
    
    # ========================================
    # PHASE 2: Put chopped items in blender
    # ========================================
    print("\n" + "="*70)
    print("PHASE 2: Get chopped items and put in blender")
    print("="*70)
    
    # Get the chopped items from the knife
    s, r, done, render_delay = run_macro_action(env, "get lettuce", "get tomato", "Get chopped lettuce and tomato", render_delay=render_delay)
    total_steps += s; total_reward += r
    if done: return check_result(env, total_steps, total_reward)
    
    # Go to blender
    s, r, done, render_delay = run_macro_action(env, "go to blender", "go to blender", "Go to blender", render_delay=render_delay)
    total_steps += s; total_reward += r
    if done: return check_result(env, total_steps, total_reward)
    
    print_state(env, total_steps, "At blender with chopped items")
    
    # ========================================
    # PHASE 3: Get peas and add to blender
    # ========================================
    print("\n" + "="*70)
    print("PHASE 3: Get peas and add to blender")
    print("="*70)
    
    s, r, done, render_delay = run_macro_action(env, "get peas", "stay", "Get peas", render_delay=render_delay)
    total_steps += s; total_reward += r
    if done: return check_result(env, total_steps, total_reward)
    
    s, r, done, render_delay = run_macro_action(env, "go to blender", "stay", "Go to blender with peas", render_delay=render_delay)
    total_steps += s; total_reward += r
    if done: return check_result(env, total_steps, total_reward)
    
    print_state(env, total_steps, "After adding all ingredients")
    
    # ========================================
    # PHASE 4: Blend
    # ========================================
    print("\n" + "="*70)
    print("PHASE 4: Blend (5 steps)")
    print("="*70)
    
    s, r, done, render_delay = run_macro_action(env, "blend", "stay", "Blending", render_delay=render_delay)
    total_steps += s; total_reward += r
    if done: return check_result(env, total_steps, total_reward)
    
    print_state(env, total_steps, "After blending")
    
    # ========================================
    # PHASE 5: Pick up blended bowl and put in oven
    # ========================================
    print("\n" + "="*70)
    print("PHASE 5: Pick up blended bowl and put in oven")
    print("="*70)
    
    # Go to blender to pick up bowl
    s, r, done, render_delay = run_macro_action(env, "go to blender", "stay", "Go to blender to pick up bowl", render_delay=render_delay)
    total_steps += s; total_reward += r
    if done: return check_result(env, total_steps, total_reward)
    
    # Go to oven with bowl
    s, r, done, render_delay = run_macro_action(env, "go to oven 1", "stay", "Go to oven with bowl", render_delay=render_delay)
    total_steps += s; total_reward += r
    if done: return check_result(env, total_steps, total_reward)
    
    print_state(env, total_steps, "Bowl in oven, cooking starts")
    
    # ========================================
    # PHASE 6: Wait for cooking + get plate
    # ========================================
    print("\n" + "="*70)
    print("PHASE 6: Wait for cooking and get plate")
    print("="*70)
    
    # Agent 1 gets plate while oven cooks
    s, r, done, render_delay = run_macro_action(env, "stay", "get plate 1", "Get plate while cooking", render_delay=render_delay)
    total_steps += s; total_reward += r
    if done: return check_result(env, total_steps, total_reward)
    
    # Wait for cooking to complete (oven auto-cooks)
    for i in range(15):  # Extra steps to ensure cooking completes
        s, r, done, render_delay = run_macro_action(env, "stay", "stay", f"Waiting for cooking ({i+1})", verbose=False, render_delay=render_delay)
        total_steps += s; total_reward += r
        if env.oven[0].cooked:
            print(f"  Cooking complete after {i+1} wait cycles")
            break
        if done: return check_result(env, total_steps, total_reward)
    
    print_state(env, total_steps, "After cooking")
    
    # ========================================
    # PHASE 7: Pick up patty with plate
    # ========================================
    print("\n" + "="*70)
    print("PHASE 7: Pick up patty with plate")
    print("="*70)
    
    # Agent with plate goes to oven
    s, r, done, render_delay = run_macro_action(env, "stay", "go to oven 1", "Go to oven with plate", render_delay=render_delay)
    total_steps += s; total_reward += r
    if done: return check_result(env, total_steps, total_reward)
    
    print_state(env, total_steps, "Picked up patty")
    
    # ========================================
    # PHASE 8: Deliver
    # ========================================
    print("\n" + "="*70)
    print("PHASE 8: Deliver")
    print("="*70)
    
    s, r, done, render_delay = run_macro_action(env, "stay", "deliver", "Deliver patty", render_delay=render_delay)
    total_steps += s; total_reward += r
    
    result = check_result(env, total_steps, total_reward)
    
    # Keep window open for a moment to see final state
    print("\n(Window will close in 1.5 seconds, or press Q to close immediately)")
    start = time.time()
    while time.time() - start < 1.5:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key in [pygame.K_q, pygame.K_ESCAPE]):
                pygame.quit()
                return result
        env.render()
        time.sleep(0.1)
    
    pygame.quit()
    return result

def check_result(env, total_steps, total_reward):
    """Check and report final result"""
    print(f"\n{'*'*60}")
    print(f"FINAL RESULT")
    print(f"{'*'*60}")
    print(f"Total steps: {total_steps}")
    print(f"Total reward: {total_reward:.1f}")
    print_state(env, total_steps, "Final state")
    
    # Check if task was completed
    if total_reward > 50:  # delivery reward is 200, so > 50 means success
        print("\n✅ TASK COMPLETED SUCCESSFULLY!")
        return True
    else:
        print("\n❌ Task not completed")
        return False

def test_simple_interactions():
    """Test basic blender and oven interactions step by step"""
    
    print("\n" + "="*70)
    print("SIMPLE INTERACTION TEST")
    print("="*70)
    
    env = Overcooked_MA_V1(
        grid_dim=(7, 7),
        task=9,
        rewardList={"subtask finished": 10, "correct delivery": 200, "wrong delivery": -5, "step penalty": -0.1},
        map_type="D",
        n_agent=2,
        obs_radius=2,
        mode="vector",
        debug=True  # Enable pygame for this test too
    )
    
    env.reset()
    
    # Render the initial state
    env.render()
    pygame.display.set_caption("Overcooked - Simple Interaction Test")
    
    # Check initial items
    print("\nInitial items:")
    print(f"  Tomato: {[(t.x, t.y, t.name) for t in env.tomato]}")
    print(f"  Lettuce: {[(l.x, l.y, l.name) for l in env.lettuce]}")
    print(f"  Peas: {[(p.x, p.y, p.name) for p in env.peas]}")
    print(f"  Plate: {[(p.x, p.y) for p in env.plate]}")
    print(f"  Knife: {[(k.x, k.y) for k in env.knife]}")
    print(f"  Blender: {[(b.x, b.y, b.name) for b in env.blender]}")
    print(f"  Oven: {[(o.x, o.y, o.name) for o in env.oven]}")
    print(f"  Agents: {[(a.x, a.y, a.color) for a in env.agent]}")
    
    # Test using primitive actions to understand the map
    print("\nMap layout:")
    from gym_macro_overcooked.overcooked_V1 import ITEMNAME as V1_ITEMNAME
    for y in range(env.ylen):
        row = ""
        for x in range(env.xlen):
            row += f"{V1_ITEMNAME[env.map[x][y]]:8}"
        print(row)
    
    # Show the initial state for 1 second
    print("\n(Showing initial state for 1 second...)")
    start = time.time()
    while time.time() - start < 1.0:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return True
        env.render()
        time.sleep(0.1)
    
    pygame.quit()
    
    print("\n✓ Simple interaction test passed!")
    return True

if __name__ == "__main__":
    print("Starting Patty Workflow Tests with Pygame Visualization...\n")
    
    # First test basic setup
    test1_passed = test_simple_interactions()
    
    # Then test full workflow
    test2_passed = test_patty_workflow()
    
    print("\n" + "="*70)
    print("TEST RESULTS")
    print("="*70)
    print(f"Simple interactions: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"Patty workflow: {'PASSED' if test2_passed else 'FAILED'}")
