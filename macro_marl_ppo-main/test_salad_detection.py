#!/usr/bin/env python3
"""
Test script to verify salad detection in Overcooked environment.
This demonstrates the check_salad_prepared() function.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "gym-macro-overcooked"))

import gym
from gym_macro_overcooked.macActEnvWrapper import MacEnvWrapper
from src.macro_marl.cores.pg_based.mac_cac.envs_runner import check_salad_prepared

def test_salad_detection():
    """Test salad detection in various states"""
    
    # Create environment
    TASKLIST = ["tomato salad", 
                "lettuce salad", 
                "onion salad", 
                "lettuce-tomato salad", 
                "onion-tomato salad", 
                "lettuce-onion salad", 
                "lettuce-onion-tomato salad"]
    
    rewardList = {"subtask finished": 10, 
                  "correct delivery": 200, 
                  "wrong delivery": -5, 
                  "step penalty": -0.1}
    
    env_params = {
        'grid_dim': [7, 7],
        'task': TASKLIST[6],  # lettuce-onion-tomato salad
        'rewardList': rewardList,
        'map_type': 'A',
        'n_agent': 2,
        'debug': False
    }
    
    env = gym.make('Overcooked-MA-v1', **env_params)
    env = MacEnvWrapper(env)
    
    print("="*60)
    print("Testing Salad Detection")
    print("="*60)
    print(f"Task: {env.env.task}")
    print(f"Required ingredients: {env.env.task.replace(' salad', '').split('-')}")
    print()
    
    # Test 1: Initial state (no salads)
    env.reset()
    status = check_salad_prepared(env.env, env.env.task)
    print("Test 1: Initial state")
    print(f"  Prepared: {status['prepared']}")
    print(f"  Count: {status['count']}")
    print(f"  Locations: {status['locations']}")
    print()
    
    # Test 2: Manually create a prepared salad
    print("Test 2: Manually placing prepared salad on plate")
    plate = env.env.plate[0]
    
    # Get food items and mark them as chopped
    tomato = env.env.tomato[0]
    lettuce = env.env.lettuce[0]
    onion = env.env.onion[0]
    
    # Chop all ingredients
    for _ in range(3):
        tomato.chop()
        lettuce.chop()
        onion.chop()
    
    # Put them on the plate
    plate.contain(tomato)
    plate.contain(lettuce)
    plate.contain(onion)
    
    status = check_salad_prepared(env.env, env.env.task)
    print(f"  Prepared: {status['prepared']}")
    print(f"  Count: {status['count']}")
    print(f"  Locations: {status['locations']}")
    print()
    
    # Test 3: Agent holding the prepared plate
    print("Test 3: Agent holding prepared salad")
    agent = env.env.agent[0]
    agent.pickup(plate)
    
    status = check_salad_prepared(env.env, env.env.task)
    print(f"  Prepared: {status['prepared']}")
    print(f"  Count: {status['count']}")
    print(f"  Locations: {status['locations']}")
    print()
    
    # Test 4: Incomplete salad (missing ingredient)
    print("Test 4: Incomplete salad (missing onion)")
    env.reset()
    plate2 = env.env.plate[0]
    tomato2 = env.env.tomato[0]
    lettuce2 = env.env.lettuce[0]
    
    for _ in range(3):
        tomato2.chop()
        lettuce2.chop()
    
    plate2.contain(tomato2)
    plate2.contain(lettuce2)
    # Missing onion
    
    status = check_salad_prepared(env.env, env.env.task)
    print(f"  Prepared: {status['prepared']}")
    print(f"  Count: {status['count']}")
    print(f"  Expected: False (missing onion)")
    print()
    
    # Test 5: Salad with unchopped ingredient
    print("Test 5: Salad with unchopped ingredient")
    env.reset()
    plate3 = env.env.plate[0]
    tomato3 = env.env.tomato[0]
    lettuce3 = env.env.lettuce[0]
    onion3 = env.env.onion[0]
    
    for _ in range(3):
        tomato3.chop()
        lettuce3.chop()
    # Don't chop onion
    
    plate3.contain(tomato3)
    plate3.contain(lettuce3)
    plate3.contain(onion3)
    
    status = check_salad_prepared(env.env, env.env.task)
    print(f"  Prepared: {status['prepared']}")
    print(f"  Count: {status['count']}")
    print(f"  Expected: False (onion not chopped)")
    print()
    
    print("="*60)
    print("All tests completed!")
    print("="*60)

if __name__ == '__main__':
    test_salad_detection()
