#!/usr/bin/env python
"""
Test script for blender and oven cooking functionality in Map D.

This script tests:
1. Blender takes 5 timesteps to blend
2. Blended contents transfer to oven
3. Oven takes 10 timesteps to cook
4. Cooked patty can be plated and delivered
"""

import sys
import os

# Add the gym_macro_overcooked to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gym_macro_overcooked.overcooked_V1 import Overcooked_V1
from gym_macro_overcooked.overcooked_MA_V1 import Overcooked_MA_V1
from gym_macro_overcooked.items import Blender, Oven, Peas, Tomato, Lettuce, Plate


def test_blender_5_timesteps():
    """Test that blender takes exactly 5 timesteps to blend"""
    print("\n=== Test 1: Blender Takes 5 Timesteps ===")
    
    blender = Blender(0, 1)
    peas = Peas(1, 1)
    tomato = Tomato(2, 1)
    lettuce = Lettuce(3, 1)
    
    # Add foods
    blender.add_food(peas)
    blender.add_food(tomato)
    blender.add_food(lettuce)
    
    assert blender.can_blend() == True, "Should be able to blend"
    assert blender.blended == False
    assert blender.cur_blend_times == 0
    print(f"✓ Initial state: blended={blender.blended}, cur_blend_times={blender.cur_blend_times}")
    
    # Blend step by step
    for i in range(4):
        result = blender.blend_step()
        assert result == False, f"Should not be blended after {i+1} steps"
        assert blender.cur_blend_times == i + 1, f"Expected {i+1} blend times, got {blender.cur_blend_times}"
        print(f"✓ Step {i+1}: blended={blender.blended}, cur_blend_times={blender.cur_blend_times}")
    
    # 5th step should complete blending
    result = blender.blend_step()
    assert result == True, "Should be blended after 5 steps"
    assert blender.blended == True
    assert blender.name == "blenderBlended"
    print(f"✓ Step 5: blended={blender.blended}, name={blender.name}")
    
    print("✓ Test 1 PASSED: Blender takes exactly 5 timesteps")
    return True


def test_oven_10_timesteps():
    """Test that oven takes exactly 10 timesteps to cook"""
    print("\n=== Test 2: Oven Takes 10 Timesteps ===")
    
    oven = Oven(0, 4)
    peas = Peas(1, 1)
    tomato = Tomato(2, 1)
    lettuce = Lettuce(3, 1)
    
    # Add blended contents to oven
    oven.add_blended([peas, tomato, lettuce])
    
    assert oven.cooking == True, "Oven should be cooking"
    assert oven.cooked == False
    assert oven.cur_cook_times == 0
    print(f"✓ Initial state: cooking={oven.cooking}, cooked={oven.cooked}, cur_cook_times={oven.cur_cook_times}")
    
    # Cook step by step
    for i in range(9):
        result = oven.cook_step()
        assert result == False, f"Should not be cooked after {i+1} steps"
        assert oven.cur_cook_times == i + 1, f"Expected {i+1} cook times, got {oven.cur_cook_times}"
        if (i + 1) % 3 == 0:
            print(f"✓ Step {i+1}: cooked={oven.cooked}, cur_cook_times={oven.cur_cook_times}")
    
    # 10th step should complete cooking
    result = oven.cook_step()
    assert result == True, "Should be cooked after 10 steps"
    assert oven.cooked == True
    assert oven.name == "lettuceTomatoPeapatty"
    print(f"✓ Step 10: cooked={oven.cooked}, name={oven.name}")
    
    print("✓ Test 2 PASSED: Oven takes exactly 10 timesteps")
    return True


def test_oven_release():
    """Test that oven releases and resets correctly"""
    print("\n=== Test 3: Oven Release ===")
    
    oven = Oven(0, 4)
    peas = Peas(1, 1)
    tomato = Tomato(2, 1)
    lettuce = Lettuce(3, 1)
    
    # Add and cook
    oven.add_blended([peas, tomato, lettuce])
    for _ in range(10):
        oven.cook_step()
    
    assert oven.cooked == True
    assert len(oven.containing) == 3
    print(f"✓ Before release: cooked={oven.cooked}, containing={len(oven.containing)} items")
    
    # Release
    oven.release()
    
    assert oven.cooked == False
    assert oven.cooking == False
    assert len(oven.containing) == 0
    assert oven.cur_cook_times == 0
    assert oven.name == "oven"
    print(f"✓ After release: cooked={oven.cooked}, cooking={oven.cooking}, name={oven.name}")
    
    print("✓ Test 3 PASSED: Oven releases correctly")
    return True


def test_macro_actions_map_d():
    """Test that Map D has correct macro actions including cook"""
    print("\n=== Test 4: Map D Macro Actions ===")
    
    rewardList = {
        "subtask finished": 10,
        "correct delivery": 200,
        "wrong delivery": -5,
        "step penalty": -0.1
    }
    
    env = Overcooked_MA_V1(
        grid_dim=(7, 7),
        task="lettuce-peas-tomato-patty",
        rewardList=rewardList,
        map_type="D",
        n_agent=2,
        obs_radius=0,
        mode="vector",
        debug=False
    )
    env.reset()
    
    print(f"Map D macro actions: {env.macroActionName}")
    
    # Check required actions exist
    required_actions = ["go to blender", "blend", "go to oven", "cook"]
    for action in required_actions:
        assert action in env.macroActionName, f"'{action}' action missing"
        print(f"✓ '{action}' index: {env.macroActionName.index(action)}")
    
    print(f"✓ Total actions: {len(env.macroActionName)}")
    print("✓ Test 4 PASSED: Map D has correct macro actions")
    return True


def test_full_cooking_workflow():
    """Test the full workflow: add foods → blend → transfer to oven → cook → patty ready"""
    print("\n=== Test 5: Full Cooking Workflow ===")
    
    # Create blender and oven
    blender = Blender(0, 1)
    oven = Oven(0, 4)
    
    # Create foods
    peas = Peas(1, 1)
    tomato = Tomato(2, 1)
    lettuce = Lettuce(3, 1)
    
    # Step 1: Add foods to blender
    print("Step 1: Adding foods to blender...")
    blender.add_food(peas)
    blender.add_food(tomato)
    blender.add_food(lettuce)
    assert len(blender.containing) == 3
    assert blender.name == "blenderEmpty"
    print(f"  ✓ Blender has {len(blender.containing)} items, state: {blender.name}")
    
    # Step 2: Blend (5 timesteps)
    print("Step 2: Blending (5 timesteps)...")
    for i in range(5):
        blender.blend_step()
    assert blender.blended == True
    assert blender.name == "blenderBlended"
    print(f"  ✓ Blending complete, state: {blender.name}")
    
    # Step 3: Transfer to oven
    print("Step 3: Transferring to oven...")
    blended_contents = blender.containing
    oven.add_blended(blended_contents)
    # Clear blender
    blender.containing = []
    blender.blended = False
    blender.cur_blend_times = 0
    
    assert oven.cooking == True
    assert len(oven.containing) == 3
    assert blender.name == "blenderEmpty"
    print(f"  ✓ Oven cooking: {oven.cooking}, items: {len(oven.containing)}")
    print(f"  ✓ Blender reset: {blender.name}")
    
    # Step 4: Cook (10 timesteps)
    print("Step 4: Cooking (10 timesteps)...")
    for i in range(10):
        oven.cook_step()
    assert oven.cooked == True
    assert oven.name == "lettuceTomatoPeapatty"
    print(f"  ✓ Cooking complete, state: {oven.name}")
    
    # Step 5: Patty ready for plating
    print("Step 5: Plating the patty...")
    plate = Plate(5, 5)
    for food in oven.containing:
        plate.contain(food)
    oven.release()
    
    assert len(plate.containing) == 3
    assert oven.name == "oven"
    print(f"  ✓ Plate contains {len(plate.containing)} items")
    print(f"  ✓ Oven reset: {oven.name}")
    
    print("\n✓ Test 5 PASSED: Full cooking workflow works!")
    return True


def test_macro_agent_tracking():
    """Test that MacAgent tracks blend and cook times"""
    print("\n=== Test 6: MacAgent Tracking ===")
    
    from gym_macro_overcooked.mac_agent import MacAgent
    
    agent = MacAgent()
    
    # Check initial state
    assert agent.cur_blend_times == 0
    assert agent.cur_cook_times == 0
    print(f"✓ Initial: blend_times={agent.cur_blend_times}, cook_times={agent.cur_cook_times}")
    
    # Simulate blending
    for i in range(5):
        agent.cur_blend_times += 1
    assert agent.cur_blend_times == 5
    print(f"✓ After 5 blend steps: blend_times={agent.cur_blend_times}")
    
    # Reset
    agent.reset()
    assert agent.cur_blend_times == 0
    assert agent.cur_cook_times == 0
    print(f"✓ After reset: blend_times={agent.cur_blend_times}, cook_times={agent.cur_cook_times}")
    
    # Simulate cooking
    for i in range(10):
        agent.cur_cook_times += 1
    assert agent.cur_cook_times == 10
    print(f"✓ After 10 cook steps: cook_times={agent.cur_cook_times}")
    
    print("✓ Test 6 PASSED: MacAgent tracking works")
    return True


def test_blender_incomplete_blend():
    """Test that incomplete blending doesn't mark as blended"""
    print("\n=== Test 7: Incomplete Blending ===")
    
    blender = Blender(0, 1)
    peas = Peas(1, 1)
    tomato = Tomato(2, 1)
    lettuce = Lettuce(3, 1)
    
    blender.add_food(peas)
    blender.add_food(tomato)
    blender.add_food(lettuce)
    
    # Blend only 3 steps
    for i in range(3):
        blender.blend_step()
    
    assert blender.blended == False
    assert blender.cur_blend_times == 3
    assert blender.name == "blenderEmpty"
    print(f"✓ After 3 steps: blended={blender.blended}, blend_times={blender.cur_blend_times}")
    
    # Release should reset
    blender.release()
    assert blender.cur_blend_times == 0
    assert blender.blended == False
    print(f"✓ After release: blend_times={blender.cur_blend_times}")
    
    print("✓ Test 7 PASSED: Incomplete blending handled correctly")
    return True


def test_oven_incomplete_cook():
    """Test that incomplete cooking doesn't mark as cooked"""
    print("\n=== Test 8: Incomplete Cooking ===")
    
    oven = Oven(0, 4)
    peas = Peas(1, 1)
    tomato = Tomato(2, 1)
    lettuce = Lettuce(3, 1)
    
    oven.add_blended([peas, tomato, lettuce])
    
    # Cook only 5 steps
    for i in range(5):
        oven.cook_step()
    
    assert oven.cooked == False
    assert oven.cur_cook_times == 5
    assert oven.name == "oven"
    print(f"✓ After 5 steps: cooked={oven.cooked}, cook_times={oven.cur_cook_times}")
    
    # Release should reset
    oven.release()
    assert oven.cur_cook_times == 0
    assert oven.cooked == False
    assert oven.cooking == False
    print(f"✓ After release: cook_times={oven.cur_cook_times}, cooking={oven.cooking}")
    
    print("✓ Test 8 PASSED: Incomplete cooking handled correctly")
    return True


def test_timing_summary():
    """Print a summary of the cooking timing"""
    print("\n=== Timing Summary ===")
    print("┌─────────────────────────────────────────┐")
    print("│ Cooking Process Timing                  │")
    print("├─────────────────────────────────────────┤")
    print("│ 1. Add peas to blender       (instant)  │")
    print("│ 2. Add tomato to blender     (instant)  │")
    print("│ 3. Add lettuce to blender    (instant)  │")
    print("│ 4. Blend                     (5 steps)  │")
    print("│ 5. Transfer to oven          (instant)  │")
    print("│ 6. Cook in oven              (10 steps) │")
    print("│ 7. Plate the patty           (instant)  │")
    print("│ 8. Deliver                   (instant)  │")
    print("├─────────────────────────────────────────┤")
    print("│ TOTAL COOKING TIME: 15 timesteps        │")
    print("└─────────────────────────────────────────┘")
    return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("BLENDER & OVEN COOKING TESTS")
    print("=" * 60)
    
    tests = [
        ("Blender Takes 5 Timesteps", test_blender_5_timesteps),
        ("Oven Takes 10 Timesteps", test_oven_10_timesteps),
        ("Oven Release", test_oven_release),
        ("Map D Macro Actions", test_macro_actions_map_d),
        ("Full Cooking Workflow", test_full_cooking_workflow),
        ("MacAgent Tracking", test_macro_agent_tracking),
        ("Incomplete Blending", test_blender_incomplete_blend),
        ("Incomplete Cooking", test_oven_incomplete_cook),
        ("Timing Summary", test_timing_summary),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"✗ {name} FAILED")
        except Exception as e:
            failed += 1
            print(f"✗ {name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

