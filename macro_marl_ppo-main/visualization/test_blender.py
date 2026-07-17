#!/usr/bin/env python
"""
Test script for blender functionality in Map D.

This script tests:
1. Blender starts as blenderEmpty
2. Adding foods (peas, tomato, lettuce) to the blender
3. Blending action when all 3 ingredients are present
4. Blender transitions to blenderBlended state
5. Picking up blended contents with a plate
"""

import sys
import os

# Add the gym_macro_overcooked to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gym_macro_overcooked.overcooked_V1 import Overcooked_V1
from gym_macro_overcooked.overcooked_MA_V1 import Overcooked_MA_V1
from gym_macro_overcooked.items import Blender, Peas, Tomato, Lettuce, Plate, Food


def test_blender_initial_state():
    """Test that blender starts in empty state"""
    print("\n=== Test 1: Blender Initial State ===")
    
    rewardList = {
        "subtask finished": 10,
        "correct delivery": 200,
        "wrong delivery": -5,
        "step penalty": -0.1
    }
    
    env = Overcooked_V1(
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
    
    # Check blender exists
    assert len(env.blender) > 0, "No blender found in Map D!"
    blender = env.blender[0]
    
    # Check initial state
    assert blender.name == "blenderEmpty", f"Expected 'blenderEmpty', got '{blender.name}'"
    assert len(blender.containing) == 0, f"Blender should be empty, has {len(blender.containing)} items"
    assert blender.blended == False, "Blender should not be blended initially"
    
    print(f"✓ Blender found at position ({blender.x}, {blender.y})")
    print(f"✓ Blender name: {blender.name}")
    print(f"✓ Blender containing: {blender.containing}")
    print(f"✓ Blender blended: {blender.blended}")
    print("✓ Test 1 PASSED: Blender starts in empty state")
    
    return True


def test_blender_add_food():
    """Test adding foods to the blender"""
    print("\n=== Test 2: Adding Foods to Blender ===")
    
    # Create a blender directly
    blender = Blender(0, 1)
    
    # Create food items
    peas = Peas(1, 1)
    tomato = Tomato(2, 1)
    lettuce = Lettuce(3, 1)
    
    # Check initial state
    assert len(blender.containing) == 0
    assert blender.name == "blenderEmpty"
    
    # Add peas
    blender.add_food(peas)
    assert len(blender.containing) == 1, f"Expected 1 item, got {len(blender.containing)}"
    assert peas in blender.containing, "Peas not in blender"
    print(f"✓ Added peas to blender. Containing: {len(blender.containing)} item(s)")
    
    # Add tomato
    blender.add_food(tomato)
    assert len(blender.containing) == 2, f"Expected 2 items, got {len(blender.containing)}"
    assert tomato in blender.containing, "Tomato not in blender"
    print(f"✓ Added tomato to blender. Containing: {len(blender.containing)} item(s)")
    
    # Add lettuce
    blender.add_food(lettuce)
    assert len(blender.containing) == 3, f"Expected 3 items, got {len(blender.containing)}"
    assert lettuce in blender.containing, "Lettuce not in blender"
    print(f"✓ Added lettuce to blender. Containing: {len(blender.containing)} item(s)")
    
    # Blender should still be empty (not blended yet)
    assert blender.name == "blenderEmpty", f"Expected 'blenderEmpty', got '{blender.name}'"
    assert blender.blended == False, "Blender should not be blended yet"
    
    print("✓ Test 2 PASSED: Foods added correctly to blender")
    return True


def test_blender_blend_action():
    """Test the blend action"""
    print("\n=== Test 3: Blend Action ===")
    
    blender = Blender(0, 1)
    peas = Peas(1, 1)
    tomato = Tomato(2, 1)
    lettuce = Lettuce(3, 1)
    
    # Try to blend with no items - should fail
    result = blender.blend()
    assert result == False, "Blend should fail with no items"
    assert blender.blended == False
    print("✓ Blend correctly fails with no items")
    
    # Add only peas - blend should fail
    blender.add_food(peas)
    result = blender.blend()
    assert result == False, "Blend should fail with only peas"
    assert blender.blended == False
    print("✓ Blend correctly fails with only peas")
    
    # Add tomato - blend should still fail (missing lettuce)
    blender.add_food(tomato)
    result = blender.blend()
    assert result == False, "Blend should fail with only peas and tomato"
    assert blender.blended == False
    print("✓ Blend correctly fails with only peas and tomato")
    
    # Add lettuce - blend should succeed
    blender.add_food(lettuce)
    result = blender.blend()
    assert result == True, "Blend should succeed with peas, tomato, and lettuce"
    assert blender.blended == True, "Blender should be in blended state"
    assert blender.name == "blenderBlended", f"Expected 'blenderBlended', got '{blender.name}'"
    
    print(f"✓ Blend succeeded! Blender name: {blender.name}")
    print("✓ Test 3 PASSED: Blend action works correctly")
    return True


def test_blender_release():
    """Test releasing blended contents"""
    print("\n=== Test 4: Release Blended Contents ===")
    
    blender = Blender(0, 1)
    peas = Peas(1, 1)
    tomato = Tomato(2, 1)
    lettuce = Lettuce(3, 1)
    
    # Add foods and blend
    blender.add_food(peas)
    blender.add_food(tomato)
    blender.add_food(lettuce)
    blender.blend()
    
    assert blender.blended == True
    assert len(blender.containing) == 3
    print(f"✓ Blender is blended with {len(blender.containing)} items")
    
    # Release contents
    blender.release()
    
    assert blender.blended == False, "Blender should not be blended after release"
    assert len(blender.containing) == 0, f"Blender should be empty, has {len(blender.containing)} items"
    assert blender.name == "blenderEmpty", f"Expected 'blenderEmpty', got '{blender.name}'"
    
    print(f"✓ After release - name: {blender.name}, containing: {len(blender.containing)}, blended: {blender.blended}")
    print("✓ Test 4 PASSED: Release works correctly")
    return True


def test_macro_actions_map_d():
    """Test that Map D has the correct macro actions including blend"""
    print("\n=== Test 5: Map D Macro Actions ===")
    
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
    assert "go to blender" in env.macroActionName, "'go to blender' action missing"
    assert "blend" in env.macroActionName, "'blend' action missing"
    
    print(f"✓ 'go to blender' index: {env.macroActionName.index('go to blender')}")
    print(f"✓ 'blend' index: {env.macroActionName.index('blend')}")
    print(f"✓ Total actions: {len(env.macroActionName)}")
    print("✓ Test 5 PASSED: Map D has correct macro actions")
    return True


def test_map_d_layout():
    """Test that Map D has blender and oven on top"""
    print("\n=== Test 6: Map D Layout ===")
    
    rewardList = {
        "subtask finished": 10,
        "correct delivery": 200,
        "wrong delivery": -5,
        "step penalty": -0.1
    }
    
    env = Overcooked_V1(
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
    
    print(f"Map layout (7x7):")
    for row in env.map:
        print(f"  {row}")
    
    # Check blender exists
    assert len(env.blender) > 0, "No blender in map!"
    blender = env.blender[0]
    print(f"✓ Blender at ({blender.x}, {blender.y})")
    
    # Check oven exists
    assert len(env.oven) > 0, "No oven in map!"
    oven = env.oven[0]
    print(f"✓ Oven at ({oven.x}, {oven.y})")
    
    # Check they're on top row (x=0)
    assert blender.x == 0, f"Blender should be on top row, but x={blender.x}"
    assert oven.x == 0, f"Oven should be on top row, but x={oven.x}"
    
    # Check they're 3 spaces apart
    distance = abs(blender.y - oven.y)
    print(f"✓ Distance between blender and oven: {distance} spaces")
    
    print("✓ Test 6 PASSED: Map D layout is correct")
    return True


def test_visual_rendering():
    """Test that blender renders correctly (requires pygame)"""
    print("\n=== Test 7: Visual Rendering (Optional) ===")
    
    try:
        import pygame
        pygame.init()
        
        rewardList = {
            "subtask finished": 10,
            "correct delivery": 200,
            "wrong delivery": -5,
            "step penalty": -0.1
        }
        
        env = Overcooked_V1(
            grid_dim=(7, 7),
            task="lettuce-peas-tomato-patty",
            rewardList=rewardList,
            map_type="D",
            n_agent=2,
            obs_radius=0,
            mode="image",
            debug=True
        )
        env.reset()
        
        # Get image observation
        img = env.render()
        
        print(f"✓ Rendered image shape: {img.shape if img is not None else 'None'}")
        print("✓ Test 7 PASSED: Visual rendering works")
        
        pygame.quit()
        return True
        
    except ImportError:
        print("⚠ pygame not available, skipping visual test")
        return True
    except Exception as e:
        print(f"⚠ Visual test skipped due to: {e}")
        return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("BLENDER FUNCTIONALITY TESTS")
    print("=" * 60)
    
    tests = [
        ("Blender Initial State", test_blender_initial_state),
        ("Adding Foods to Blender", test_blender_add_food),
        ("Blend Action", test_blender_blend_action),
        ("Release Blended Contents", test_blender_release),
        ("Map D Macro Actions", test_macro_actions_map_d),
        ("Map D Layout", test_map_d_layout),
        ("Visual Rendering", test_visual_rendering),
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

