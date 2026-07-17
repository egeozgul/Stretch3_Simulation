import argparse
import numpy as np
import torch
import os
import sys
sys.path.append("..")
import time
import gym
import pygame
from torch.distributions import Categorical

from macro_marl.cores.pg_based.acac.utils import Agent
from macro_marl.cores.pg_based.acac.models import AgentCentricGRUActor

import argparse
from gym_macro_overcooked.macActEnvWrapper import MacEnvWrapper

def get_human_action(env, agent_idx, mapType):
    """
    Get human action from keyboard input.
    Arrow keys for movement, number keys for macro-actions.
    
    Map A actions:
    0: stay, 1: get tomato, 2: get lettuce, 3: get onion, 4: get plate 1, 5: get plate 2,
    6: go to knife 1, 7: go to knife 2, 8: deliver, 9: chop, 10: right, 11: down, 12: left, 13: up
    
    Map B/C actions:
    0: stay, 1: get tomato, 2: get lettuce, 3: get onion, 4: get plate 1, 5: get plate 2,
    6: go to knife 1, 7: go to knife 2, 8: deliver, 9: chop, 10: go to counter, 11: right, 12: down, 13: left, 14: up
    """
    # Common macro-actions (same for all maps)
    common_actions = {
        pygame.K_0: 0,   # stay
        pygame.K_SPACE: 0,  # stay (spacebar)
        pygame.K_1: 1,   # get tomato
        pygame.K_2: 2,   # get lettuce
        pygame.K_3: 3,   # get onion
        pygame.K_4: 4,   # get plate 1
        pygame.K_5: 5,   # get plate 2
        pygame.K_6: 6,   # go to knife 1
        pygame.K_7: 7,   # go to knife 2
        pygame.K_d: 8,   # deliver
        pygame.K_8: 8,   # deliver (alternate)
        pygame.K_c: 9,   # chop
        pygame.K_9: 9,   # chop (alternate)
    }
    
    # Primitive movement actions
    if mapType == "A":
        movement_actions = {
            pygame.K_RIGHT: 10,  # right
            pygame.K_DOWN: 11,   # down
            pygame.K_LEFT: 12,   # left
            pygame.K_UP: 13,     # up
        }
    else:  # Map B, C, etc. have "go to counter" action
        movement_actions = {
            pygame.K_x: 10,      # go to counter (changed from C to X)
            pygame.K_RIGHT: 11,  # right
            pygame.K_DOWN: 12,   # down
            pygame.K_LEFT: 13,   # left
            pygame.K_UP: 14,     # up
        }
    
    # Combine action maps
    action_map = {**common_actions, **movement_actions}
    
    # Use pygame.event.pump() to update keyboard state without consuming events
    # This prevents conflicts with the environment's rendering system
    pygame.event.pump()
    
    # Handle quit events separately
    for event in pygame.event.get([pygame.QUIT]):
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    
    # Check currently pressed keys
    keys = pygame.key.get_pressed()
    
    # Check keys in priority order (letter keys and number keys before arrows)
    # This ensures macro-actions take precedence over movement
    for key in [pygame.K_c, pygame.K_d, pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, 
                pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9, pygame.K_0, pygame.K_SPACE]:
        if keys[key] and key in action_map:
            return action_map[key]
    
    # Check X key for counter (Map B/C)
    if keys[pygame.K_x] and pygame.K_x in action_map:
        return action_map[pygame.K_x]
    
    # Check arrow keys
    for key in [pygame.K_RIGHT, pygame.K_DOWN, pygame.K_LEFT, pygame.K_UP]:
        if keys[key] and key in action_map:
            return action_map[key]
    
    # Default to stay if no key is pressed
    return 0

def get_actions_and_h_states(env, agents, last_valid, joint_obs, joint_h_states, human_agent_idx=None, mapType="A"):
    """
    Get actions for all agents. If human_agent_idx is specified, that agent uses keyboard control.
    """
    with torch.no_grad():
        actions = []
        new_h_states = []
        for idx, agent in enumerate(agents):
            # Check if this agent is human-controlled
            if human_agent_idx is not None and idx == human_agent_idx:
                action = get_human_action(env, human_agent_idx, mapType)
                actions.append(action)
                new_h_states.append(joint_h_states[agent.idx])  # Keep h_state unchanged for human
            else:
                # AI agent uses trained policy
                obs = joint_obs[agent.idx]
                # Pad observation if it's smaller than expected
                if obs.shape[0] < agent.expected_input_dim:
                    padding = torch.zeros(agent.expected_input_dim - obs.shape[0])
                    obs = torch.cat([obs, padding])
                
                # ACAC uses different forward signature
                action_logits, new_h_state = agent.actor_net(
                    obs.view(1, 1, agent.expected_input_dim), 
                    joint_h_states[agent.idx],
                    eps=0.0,
                    test_mode=True  # Use greedy mode for testing
                )
                action_prob = Categorical(logits=action_logits[0])
                action = action_prob.sample().item()
                actions.append(action)
                new_h_states.append(new_h_state)
    return actions, new_h_states


def get_init_inputs(env, n_agent):
    return [torch.from_numpy(i).float() for i in env.reset()], [None]*n_agent

def setup_extended_display(env, human_agent_idx=None):
    """
    Resize the existing pygame display to add space for info panel below the game.
    """
    if not hasattr(setup_extended_display, 'initialized'):
        # Get original game dimensions
        game_width = env.env.game.width
        game_height = env.env.game.height
        
        # Create extended window with info panel space below
        info_panel_height = 100 if human_agent_idx is None else 120
        extended_height = game_height + info_panel_height
        
        # Store dimensions
        setup_extended_display.game_width = game_width
        setup_extended_display.game_height = game_height
        setup_extended_display.info_height = info_panel_height
        setup_extended_display.total_height = extended_height
        
        # Wait a moment for pygame to initialize from env
        time.sleep(0.1)
        
        # Resize the existing window to extended size
        try:
            setup_extended_display.screen = pygame.display.set_mode((game_width, extended_height))
            pygame.display.set_caption("Overcooked Multi-Agent (ACAC)")
        except:
            # If that fails, just use the existing display
            setup_extended_display.screen = pygame.display.get_surface()
            if setup_extended_display.screen is None:
                # Create new if none exists
                setup_extended_display.screen = pygame.display.set_mode((game_width, extended_height))
                pygame.display.set_caption("Overcooked Multi-Agent (ACAC)")
        
        # Initialize fonts
        pygame.font.init()
        setup_extended_display.font = pygame.font.SysFont('Arial', 24, bold=True)
        setup_extended_display.small_font = pygame.font.SysFont('Arial', 18)
        
        setup_extended_display.initialized = True
    
    return setup_extended_display.screen

def draw_reward_panel(env, step, total_reward, last_reward, human_agent_idx=None):
    """
    Draw reward information in a separate panel below the game area.
    """
    if not hasattr(setup_extended_display, 'initialized'):
        return
    
    try:
        screen = pygame.display.get_surface()
        if screen is None:
            return
        
        game_height = setup_extended_display.game_height
        panel_height = setup_extended_display.info_height
        game_width = setup_extended_display.game_width
        
        # Create info panel (separate from game area)
        panel = pygame.Surface((game_width, panel_height))
        panel.fill((40, 40, 40))  # Dark gray background
        
        # Draw separator line
        pygame.draw.line(panel, (100, 100, 100), (0, 0), (game_width, 0), 2)
        
        # Render text on panel
        y_offset = 15
        
        # Step counter
        step_text = setup_extended_display.font.render(f'Step: {int(step)}', True, (255, 255, 255))
        panel.blit(step_text, (20, y_offset))
        
        # Total reward
        reward_color = (100, 255, 100) if total_reward >= 0 else (255, 100, 100)
        total_reward_text = setup_extended_display.font.render(f'Total: {total_reward:.2f}', True, reward_color)
        panel.blit(total_reward_text, (200, y_offset))
        
        # Last step reward
        last_reward_color = (150, 255, 150) if last_reward >= 0 else (255, 150, 150)
        last_reward_text = setup_extended_display.small_font.render(f'Last: {last_reward:.2f}', True, last_reward_color)
        panel.blit(last_reward_text, (430, y_offset + 5))
        
        # Show control mode
        if human_agent_idx is not None:
            mode_text = setup_extended_display.small_font.render(
                f'Human: Agent {human_agent_idx}', 
                True, 
                (255, 255, 100)
            )
            panel.blit(mode_text, (20, y_offset + 45))
            
            # Show controls hint
            controls_text = setup_extended_display.small_font.render(
                f'1-7=Get Items, C=Chop, D=Deliver, Arrows=Move', 
                True, 
                (180, 180, 180)
            )
            panel.blit(controls_text, (170, y_offset + 45))
        
        # Blit panel below the game area
        screen.blit(panel, (0, game_height))
        
        # Don't update display here - let the main loop handle it
        # pygame.display.update((0, game_height, game_width, panel_height))
    except pygame.error:
        # Silently handle if display is not available
        pass

def test(env_id, grid_dim, mapType, task, n_agent, p_id, save_dir, human_agent_idx=None, render_delay=0.03):

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
    env_params = {'grid_dim': grid_dim,
                  'task': TASKLIST[task],
                  'rewardList': rewardList,
                  'map_type': mapType,
                  'n_agent': n_agent,
                  'debug': True
                  }
    env = gym.make(env_id, **env_params)
    env = MacEnvWrapper(env)

    # Debug: Print environment observation size
    print(f"Environment observation sizes: {env.obs_size}")
    print(f"Environment action sizes: {env.n_action}")
    
    if human_agent_idx is not None:
        print(f"\n{'='*60}")
        print(f"    HUMAN CONTROL MODE - Map {mapType} (ACAC)")
        print(f"{'='*60}")
        print(f"Agent {human_agent_idx} is human-controlled (you!)")
        print(f"Agent {1-human_agent_idx} uses trained policy")
        print(f"\n{'='*60}")
        print(f"MACRO-ACTIONS:")
        print(f"{'='*60}")
        print(f"  0/Space : Stay")
        print(f"  1       : Get Tomato")
        print(f"  2       : Get Lettuce") 
        print(f"  3       : Get Onion")
        print(f"  4       : Get Plate 1")
        print(f"  5       : Get Plate 2")
        print(f"  6       : Go to Knife 1")
        print(f"  7       : Go to Knife 2")
        print(f"  D (or 8): Deliver")
        print(f"  C (or 9): Chop")
        if mapType != "A":
            print(f"  X       : Go to Counter")
        print(f"\n{'='*60}")
        print(f"PRIMITIVE ACTIONS (Arrow Keys):")
        print(f"{'='*60}")
        print(f"  →/↓/←/↑ : Move Right/Down/Left/Up")
        print(f"\n{'='*60}")
        print(f"Close Window to Quit")
        print(f"{'='*60}\n")

    agents = []
    
    for i in range(n_agent):
        agent = Agent()
        agent.idx = i
        
        # Skip loading policy for human-controlled agent
        if human_agent_idx is not None and i == human_agent_idx:
            agent.expected_input_dim = None  # Not needed for human agent
            agents.append(agent)
            continue
        
        # Try loading ACAC policy - support both state dict and full model formats
        state_dict_path = f"./policy_nns/{save_dir}/agent_state_dict_{i}.pt"
        full_model_path = f"./policy_nns/{save_dir}/{p_id}_agent_{i}.pt"
        
        # Try state dict first (newer format)
        if os.path.exists(state_dict_path):
            print(f"Loading state dict from {state_dict_path}")
            checkpoint = torch.load(state_dict_path)
            is_state_dict = True
        # Fall back to full model (older format)
        elif os.path.exists(full_model_path):
            print(f"Loading full model from {full_model_path}")
            checkpoint = torch.load(full_model_path)
            is_state_dict = False
        else:
            print(f"Policy not found at either:")
            print(f"  {os.path.abspath(state_dict_path)}")
            print(f"  {os.path.abspath(full_model_path)}")
            print(f"Current working directory: {os.getcwd()}")
            raise FileNotFoundError(f"Policy not found. Make sure you've trained ACAC and saved policies.")
        
        # Get the expected input dimension from the checkpoint
        if is_state_dict:
            # State dict format: check encoder.fc1.weight
            if 'encoder.fc1.weight' in checkpoint:
                fc1_weight_shape = checkpoint['encoder.fc1.weight'].shape
            else:
                raise KeyError(f"Expected 'encoder.fc1.weight' in checkpoint but found keys: {list(checkpoint.keys())[:10]}...")
            expected_input_dim = fc1_weight_shape[1]  # Second dimension is input_dim
            
            print(f"Agent {i}: Current env obs_size={env.obs_size[i]}, Model expects={expected_input_dim}")
            
            # Create ACAC actor model with the expected input dimension
            agent.actor_net = AgentCentricGRUActor(
                expected_input_dim, 
                env.n_action[i],
                mlp_layer_size=[32, 32],
                rnn_layer_size=32,
                use_time_emb=False  # Set to True if your training used time embedding
            )
            agent.actor_net.load_state_dict(checkpoint)
        else:
            # Full model format: use the loaded model directly
            agent.actor_net = checkpoint
            # Get input dim from the loaded model
            if hasattr(agent.actor_net, 'encoder') and hasattr(agent.actor_net.encoder, 'fc1'):
                expected_input_dim = agent.actor_net.encoder.fc1.in_features
            else:
                expected_input_dim = env.obs_size[i]
                print(f"Warning: Could not detect input dim from model, using env obs_size: {expected_input_dim}")
            
            print(f"Agent {i}: Current env obs_size={env.obs_size[i]}, Model expects={expected_input_dim}")
        
        agent.actor_net.eval()
        
        # Store the expected input dimension for padding observations later
        agent.expected_input_dim = expected_input_dim
        
        agents.append(agent)

    R = 0
    discount = 0.99
    step = 0.0
    n_episode = 1
    last_reward = 0.0

    # Setup extended display with info panel
    screen = setup_extended_display(env, human_agent_idx)
    
    for e in range(n_episode):
        t = 0
        last_obs, h_states = get_init_inputs(env, n_agent)
        
        # Initial render
        env.render()
        draw_reward_panel(env, step, R, last_reward, human_agent_idx)
        pygame.display.flip()
        
        last_valid = [1.0] * n_agent
        while not t:
            a, h_states = get_actions_and_h_states(env, agents, last_valid, last_obs, h_states, human_agent_idx, mapType)
            last_obs, r, t, info = env.step(a)
            
            # Update rewards
            last_reward = r[0]
            R += discount**step*last_reward
            step += 1.0
            
            # Render environment (this will update the game area)
            env.render()
            
            # Draw reward panel below (but don't update display yet)
            draw_reward_panel(env, step, R, last_reward, human_agent_idx)
            
            # Single display update for the entire screen
            pygame.display.flip()
            
            # Add a small delay for human control mode to make it playable
            if human_agent_idx is not None:
                time.sleep(render_delay)  # Configurable delay for control responsiveness
            
            last_obs = [torch.from_numpy(o).float() for o in last_obs]
            last_valid = info['mac_done']
        
        print(f"\n{'='*60}")
        print(f"Episode Complete!")
        print(f"{'='*60}")
        print(f"Final Total Reward: {R:.2f}")
        print(f"Total Steps: {int(step)}")
        print(f"{'='*60}\n")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', action='store', type=str, default='Overcooked-MA-v1')
    parser.add_argument('--grid_dim', action='store', type=int, nargs=2, default=[7,7], choices=[[7, 7], [9, 9]])
    parser.add_argument('--mapType', action='store', type=str, default="A", choices=["A", "B", "C", "D", "E", "F"])
    parser.add_argument('--task', action='store', type=int, default=6, choices=[3, 6])
    parser.add_argument('--n_agent', action='store', type=int, default=2)
    parser.add_argument('--p_id', action='store', type=int, default=0, help="The specific policy_id")
    parser.add_argument('--save_dir', action='store', type=str, default='Overcooked/mapA/acac', 
                        help="Directory path where policies are saved (e.g., 'Overcooked/mapA/acac' or 'acac_overcooked_A')")
    parser.add_argument('--human_agent_idx', action='store', type=int, default=None, 
                        help="Index of human-controlled agent (0 or 1). If None, all agents use trained policies.")
    parser.add_argument('--render_delay', action='store', type=float, default=0.08,
                        help="Delay between frames in seconds (default: 0.08). Lower = faster, 0.001 = max speed.")

    test(**vars(parser.parse_args()))

if __name__ == '__main__':
    main()

