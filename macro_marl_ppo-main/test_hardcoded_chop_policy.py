#!/usr/bin/env python3
"""
Test script for hard-coded chop policy for agent 2.

This script runs the Overcooked environment with agent 2 receiving the instruction
"let me do all the chopping" and monitors whether the hard-coded policy correctly
forces agent 2 to perform the chop action when there's food on the knife.

Usage:
    python test_hardcoded_chop_policy.py \
        --map_type D \
        --policy_dir experiments/Overcooked/policy_nns/mac_iac_overcooked_D_desktop2_stochastic
"""

import argparse
import numpy as np
import torch
import os
import sys
import gym

# Add paths for custom modules
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "gym-macro-overcooked"))

from macro_marl.cores.pg_based.mac_iac.utils import Agent
from macro_marl.cores.pg_based.mac_iac.models import Actor
from macro_marl.cores.pg_based.mac_iac.envs_runner import EnvsRunner
from gym_macro_overcooked.macActEnvWrapper import MacEnvWrapper


class HardcodedPolicyTester:
    """Test harness for the hard-coded chop policy."""
    
    def __init__(self, map_type='D', policy_dir=None, n_episodes=5):
        self.map_type = map_type
        self.policy_dir = policy_dir
        self.n_episodes = n_episodes
        self.n_agent = 3
        
        # Statistics
        self.agent_2_actions = []
        self.food_on_knife_steps = 0
        self.chop_when_food_steps = 0
        self.chop_when_no_food_steps = 0
        self.total_steps = 0
        
    def create_environment(self):
        """Create the Overcooked environment."""
        TASKLIST = ["tomato salad", "lettuce salad", "onion salad", 
                    "lettuce-tomato salad", "onion-tomato salad", 
                    "lettuce-onion salad", "lettuce-onion-tomato salad"]
        
        rewardList = {
            "subtask finished": 10,
            "correct delivery": 200,
            "wrong delivery": -5,
            "step penalty": -0.1
        }
        
        task_idx = 9 if self.map_type == 'D' else 9
        task = TASKLIST[task_idx] if task_idx < len(TASKLIST) else "tomato salad"
        
        env_params = {
            'grid_dim': [7, 7],
            'task': task,
            'rewardList': rewardList,
            'map_type': self.map_type,
            'n_agent': self.n_agent,
            'debug': False
        }
        
        env = gym.make('Overcooked-MA-v1', **env_params)
        env = MacEnvWrapper(env)
        return env
    
    def load_policies(self, env, policy_dir):
        """Load pre-trained policies for agents 0 and 1."""
        agents = []
        p_id = 0  # Use run_id 0
        
        for i in range(self.n_agent):
            agent = Agent()
            agent.idx = i
            
            # Agent 2 doesn't need a loaded policy (we'll monitor its hard-coded behavior)
            if i == 2:
                agents.append(agent)
                continue
            
            # Try to load policy for agents 0 and 1
            if not policy_dir or not os.path.exists(policy_dir):
                print(f"Policy directory not found, using random policy for agent {i}")
                agent.policy_net = None
                agents.append(agent)
                continue
            
            policy_path = os.path.join(policy_dir, f"{p_id}_agent_{i}.pt")
            
            if not os.path.exists(policy_path):
                # Use random policy if actual policy not found
                print(f"Policy not found for agent {i}, using random policy")
                agent.policy_net = None
                agents.append(agent)
                continue
            
            try:
                loaded_data = torch.load(policy_path, map_location='cpu', weights_only=False)
                
                if isinstance(loaded_data, Actor):
                    actor_net = loaded_data
                else:
                    model_input_dim = env.obs_size[i]
                    actor_net = Actor(
                        input_dim=model_input_dim,
                        output_dim=env.n_action[i],
                        use_instructions=True,
                        instruction_fusion='attention'
                    )
                    
                    if isinstance(loaded_data, dict) and 'actor_net_state_dict' in loaded_data:
                        state_dict = loaded_data['actor_net_state_dict']
                    else:
                        state_dict = loaded_data
                    
                    try:
                        actor_net.load_state_dict(state_dict)
                    except RuntimeError as e:
                        print(f"Warning: Could not load state dict for agent {i} ({e}), using random policy")
                        agent.policy_net = None
                        agents.append(agent)
                        continue
                
                actor_net.eval()
                agent.policy_net = actor_net
                print(f"Loaded policy for agent {i}")
                
            except Exception as e:
                print(f"Error loading policy for agent {i}: {e}, using random policy")
                agent.policy_net = None
            
            agents.append(agent)
        
        return agents
    
    def get_action(self, agent, obs, h_state):
        """Get action from agent policy or return random action."""
        if agent.policy_net is None:
            # Random action
            return np.random.randint(0, 20), h_state
        
        obs_tensor = torch.from_numpy(obs).float().view(1, -1) if isinstance(obs, np.ndarray) else obs
        
        try:
            with torch.no_grad():
                # Try calling with instruction embedding (None for random policy)
                action_logits, new_h_state = agent.policy_net(
                    obs_tensor, h_state, instruction_emb=None
                )
                action = action_logits[0].argmax(dim=-1).item()
            return action, new_h_state
        except TypeError:
            # Model doesn't accept instruction_emb parameter
            try:
                with torch.no_grad():
                    action_logits, new_h_state = agent.policy_net(obs_tensor, h_state)
                    action = action_logits[0].argmax(dim=-1).item()
                return action, new_h_state
            except RuntimeError as e:
                # Shape mismatch - fall back to random
                print(f"Warning: Policy shape mismatch ({e}), using random action")
                return np.random.randint(0, 20), h_state
        except Exception as e:
            # Any other error - fall back to random
            print(f"Warning: Error calling policy ({e}), using random action")
            return np.random.randint(0, 20), h_state
    
    def check_food_on_knife(self, obs):
        """Check if there's food on the knife by examining observation."""
        obs_array = obs[0] if isinstance(obs, list) else obs
        if isinstance(obs_array, torch.Tensor):
            obs_array = obs_array.cpu().numpy()
        
        # Parse observation to find food items with partial chopping
        i = 0
        while i < len(obs_array):
            if i + 2 >= len(obs_array):
                break
            
            if i + 2 < len(obs_array):
                chopped_progress = float(obs_array[i + 2])
                # Food on knife: chopped_progress between 0 and 1 (exclusive)
                if 0 < chopped_progress < 1:
                    return True
                i += 3  # Food: x, y, chopped_progress
            else:
                i += 2  # Non-food: x, y
        
        return False
    
    def run_episode(self, env, agents, episode_num):
        """Run a single episode and monitor agent 2's behavior."""
        obs = env.reset()
        h_states = [None] * self.n_agent
        done = False
        step = 0
        episode_chops = 0
        episode_food_on_knife = 0
        
        print(f"\n{'='*70}")
        print(f"Episode {episode_num + 1}")
        print(f"{'='*70}")
        
        while not done and step < 200:
            actions = []
            
            # Get actions for all agents
            for agent_idx, agent in enumerate(agents):
                if agent_idx == 2:
                    # For agent 2, just get a default action
                    # The hard-coded policy will override it in the environment
                    action = 0  # Stay action
                else:
                    # Get action from policy
                    action, h_states[agent_idx] = self.get_action(agent, obs[agent_idx], h_states[agent_idx])
                
                actions.append(action)
            
            # Check if food is on knife BEFORE environment step
            food_on_knife = self.check_food_on_knife(obs)
            
            # Step environment (hard-coded policy will modify agent 2's action inside)
            obs, reward, done, info = env.step(actions)
            
            # Track metrics
            self.total_steps += 1
            
            if food_on_knife:
                episode_food_on_knife += 1
                self.food_on_knife_steps += 1
                
                # Check if agent 2 performed chop action (10)
                # In the environment, agent 2's action should be 10 if hard-coded policy worked
                if 'cur_mac' in info:
                    agent_2_action = info['cur_mac'][2] if len(info['cur_mac']) > 2 else -1
                    if agent_2_action == 10:  # Chop action
                        episode_chops += 1
                        self.chop_when_food_steps += 1
            
            step += 1
        
        # Print episode results
        if episode_food_on_knife > 0:
            chop_rate = (episode_chops / episode_food_on_knife) * 100
            print(f"Food on knife: {episode_food_on_knife} steps")
            print(f"Agent 2 chopped: {episode_chops}/{episode_food_on_knife} ({chop_rate:.1f}%)")
        else:
            print(f"No food on knife during this episode")
        
        return step
    
    def run_test(self):
        """Run the full test."""
        print(f"\n{'='*70}")
        print(f"HARD-CODED CHOP POLICY TEST")
        print(f"{'='*70}")
        print(f"Map Type: {self.map_type}")
        print(f"Number of Episodes: {self.n_episodes}")
        print(f"Number of Agents: {self.n_agent}")
        print(f"\nAgent 2 will receive instruction: 'let me do all the chopping'")
        print(f"Expected behavior: Agent 2 should perform chop action (10)")
        print(f"                  whenever food is on the knife")
        print(f"{'='*70}\n")
        
        # Create environment
        env = self.create_environment()
        print(f"Environment created: {env}")
        print(f"Observation sizes: {env.obs_size}")
        print(f"Action sizes: {env.n_action}")
        
        # Load policies
        if self.policy_dir and os.path.exists(self.policy_dir):
            agents = self.load_policies(env, self.policy_dir)
        else:
            print(f"Policy directory not found or not provided: {self.policy_dir}")
            print(f"Using random policies for agents 0 and 1")
            agents = []
            for i in range(self.n_agent):
                agent = Agent()
                agent.idx = i
                agent.policy_net = None
                agents.append(agent)
        
        # Run episodes
        total_episode_steps = 0
        for episode in range(self.n_episodes):
            steps = self.run_episode(env, agents, episode)
            total_episode_steps += steps
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"TEST SUMMARY")
        print(f"{'='*70}")
        print(f"Total episodes: {self.n_episodes}")
        print(f"Total steps: {self.total_steps}")
        print(f"Average steps per episode: {total_episode_steps / self.n_episodes:.1f}")
        print()
        print(f"Food on knife steps: {self.food_on_knife_steps}")
        
        if self.food_on_knife_steps > 0:
            chop_success_rate = (self.chop_when_food_steps / self.food_on_knife_steps) * 100
            print(f"Agent 2 chopped when food on knife: {self.chop_when_food_steps}/{self.food_on_knife_steps}")
            print(f"Success rate: {chop_success_rate:.1f}%")
            
            if chop_success_rate > 90:
                print(f"\n✓ PASS: Hard-coded policy is working correctly!")
            elif chop_success_rate > 70:
                print(f"\n⚠ PARTIAL: Hard-coded policy is mostly working")
            else:
                print(f"\n✗ FAIL: Hard-coded policy is not working properly")
        else:
            print(f"No food appeared on knife during test")
            print(f"Cannot evaluate hard-coded policy")
        
        print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test hard-coded chop policy for agent 2")
    parser.add_argument("--map_type", type=str, default="D", 
                        help="Map type (A, B, C, or D)")
    parser.add_argument("--policy_dir", type=str, default=None,
                        help="Directory containing trained policies")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of test episodes")
    
    args = parser.parse_args()
    
    # If policy_dir not specified, try to find it
    if args.policy_dir is None:
        policy_dir = os.path.join(
            os.path.dirname(__file__),
            "experiments", "Overcooked", "policy_nns",
            f"mac_iac_overcooked_{args.map_type}_desktop2_stochastic"
        )
    else:
        policy_dir = args.policy_dir
    
    tester = HardcodedPolicyTester(
        map_type=args.map_type,
        policy_dir=policy_dir,
        n_episodes=args.episodes
    )
    
    tester.run_test()
