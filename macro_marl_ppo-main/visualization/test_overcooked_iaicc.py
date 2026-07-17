import argparse
import inspect
import numpy as np
import torch
import os
import sys
import time
import gym
 
# Add paths for custom modules BEFORE importing them
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "gym-macro-overcooked"))
 
# Import pygame only if available
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    pygame = None
from torch.distributions import Categorical
 
from macro_marl.cores.pg_based.mac_iaicc.utils import Agent
from macro_marl.cores.pg_based.mac_niacc.models import Actor  # IAICC uses mac_niacc models
from gym_macro_overcooked.macActEnvWrapper import MacEnvWrapper
 
# Global variable to store current instruction
current_instruction = None
instruction_input_active = False
instruction_text = ''
instruction_changed = False  # Flag to trigger resampling when instruction changes
instruction_persistence_counter = 0
 
# Global display state
_display_state = {
    'screen': None,
    'initialized': False,
    'game_width': None,
    'game_height': None,
    'info_height': None,
    'total_height': None,
    'font': None,
    'small_font': None
}
 
def get_human_action(env, agent_idx, mapType):
    """
    Get human action from keyboard input.
    Arrow keys for movement, number keys for macro-actions.
    't' key to enter instruction mode.

    Map A actions:
    0: stay, 1: get tomato, 2: get lettuce, 3: get onion, 4: get peas (no-op), 5: get plate 1, 6: get plate 2 (no-op),
    7: go to knife 1, 8: go to knife 2, 9: deliver, 10: chop, 11: right, 12: down, 13: left, 14: up

    Map B/C actions:
    0: stay, 1: get tomato, 2: get lettuce, 3: get onion, 4: get peas (no-op), 5: get plate 1, 6: get plate 2,
    7: go to knife 1, 8: go to knife 2, 9: deliver, 10: chop, 11: go to counter, 12: right, 13: down, 14: left, 15: up
    
    Map D actions:
    0: stay, 1: get tomato, 2: get lettuce, 3: get onion, 4: get peas, 5: get plate 1, 6: get plate 2 (no-op),
    7: go to knife 1, 8: go to knife 2, 9: deliver, 10: chop, 11: go to blender, 12: blend, 
    13: go to oven 1, 14: go to oven 2, 15: cook, 16: right, 17: down, 18: left, 19: up
    """
    global current_instruction, instruction_input_active, instruction_text
    
    # Skip pygame operations if pygame is not available or not initialized
    if not PYGAME_AVAILABLE or not pygame.display.get_init():
        return 0  # Default to stay action
    
    # If instruction input is active, just return stay (typing mode)
    if instruction_input_active:
        return 0
 
    # Common macro-actions (same for all maps)
    # Macro-action indices: 0=stay, 1=tomato, 2=lettuce, 3=onion, 4=peas, 5=plate1, 6=plate2,
    #                       7=knife1, 8=knife2, 9=deliver, 10=chop, then movement actions
    common_actions = {
        pygame.K_0: 0,   # stay
        pygame.K_SPACE: 0,  # stay (spacebar)
        pygame.K_1: 1,   # get tomato
        pygame.K_2: 2,   # get lettuce
        pygame.K_3: 3,   # get onion
        pygame.K_4: 4,   # get peas (Map D only, no-op on other maps)
        pygame.K_5: 5,   # get plate 1
        pygame.K_6: 6,   # get plate 2 (Maps B/C only, no-op on Map A which has 1 plate)
        pygame.K_7: 7,   # go to knife 1
        pygame.K_8: 8,   # go to knife 2
        pygame.K_d: 9,   # deliver
        pygame.K_9: 9,   # deliver (alternate)
        pygame.K_c: 10,  # chop
        pygame.K_t: -1,  # Special key for instruction mode
    }
    
    # Primitive movement actions
    if mapType == "A":
        movement_actions = {
            pygame.K_UP: env.macroActionName.index("up"),
            pygame.K_DOWN: env.macroActionName.index("down"),
            pygame.K_LEFT: env.macroActionName.index("left"),
            pygame.K_RIGHT: env.macroActionName.index("right"),
        }
    elif mapType == "D":
        # Map D has blender and oven actions instead of "go to counter"
        movement_actions = {
            pygame.K_b: env.macroActionName.index("go to blender"),
            pygame.K_n: env.macroActionName.index("blend"),
            pygame.K_o: env.macroActionName.index("go to oven 1"),
            pygame.K_i: env.macroActionName.index("go to oven 2"),
            pygame.K_k: env.macroActionName.index("cook"),
            pygame.K_UP: env.macroActionName.index("up"),
            pygame.K_DOWN: env.macroActionName.index("down"),
            pygame.K_LEFT: env.macroActionName.index("left"),
            pygame.K_RIGHT: env.macroActionName.index("right"),
        }
    else:  # Map B, C have "go to counter" action
        movement_actions = {
            pygame.K_x: env.macroActionName.index("go to counter"),
            pygame.K_UP: env.macroActionName.index("up"),
            pygame.K_DOWN: env.macroActionName.index("down"),
            pygame.K_LEFT: env.macroActionName.index("left"),
            pygame.K_RIGHT: env.macroActionName.index("right"),
        }
    
    # Combine action maps
    action_map = {**common_actions, **movement_actions}
    
    # Check currently pressed keys
    keys = pygame.key.get_pressed()

    # Check for movement keys first
    for key_code, action in movement_actions.items():
        if keys[key_code]:
            return action

    # Check for other macro-action keys
    for key_code, action in common_actions.items():
        if keys[key_code]:
            return action
 
    # Default to stay if no key is pressed
    return 0
 
def handle_instruction_input_events():
    """
    Handle keyboard events for instruction text input.
    Returns True if events were consumed (instruction mode active).
    """
    global instruction_input_active, instruction_text, current_instruction, instruction_changed, instruction_persistence_counter
    
    if not PYGAME_AVAILABLE:
        return False
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if not instruction_input_active:
                # Check if 't' is pressed to start instruction input
                if event.key == pygame.K_t:
                    instruction_input_active = True
                    instruction_text = ''
                    print("\n⏸  GAME PAUSED - All agents stopped for instruction input")
                    return True
            else:
                # In instruction input mode
                if event.key == pygame.K_RETURN:
                    # Submit instruction
                    old_instruction = current_instruction
                    
                    # Check if instruction is empty
                    if not instruction_text.strip():
                        current_instruction = None
                        print(f"\n▶  RESUMING - Instruction cleared (None)")
                    else:
                        current_instruction = instruction_text
                        print(f"\n▶  RESUMING - Instruction set: '{instruction_text}'")
                    
                    instruction_input_active = False
                    instruction_text = ''
                    # Set flag to trigger resampling if instruction actually changed
                    if old_instruction != current_instruction:
                        instruction_changed = True
                        instruction_persistence_counter = -1  # Set persistence counter (infinite)
                        print("→ Agents will resample actions based on new instruction!")
                elif event.key == pygame.K_ESCAPE:
                    # Cancel instruction input
                    print("\n▶  RESUMING - Instruction cancelled")
                    instruction_input_active = False
                    instruction_text = ''
                elif event.key == pygame.K_BACKSPACE:
                    # Delete last character
                    instruction_text = instruction_text[:-1]
                else:
                    # Add character to instruction text
                    instruction_text += event.unicode
                return True
    
    return instruction_input_active
 
def get_actions_and_h_states(env, agents, last_valid, joint_obs, joint_h_states, human_agent_idx=None, mapType="A", force_resample=False):
    """
    Get actions for all agents. If human_agent_idx is specified, that agent uses keyboard control.
    For MAC_IAICC, each agent has its own individual actor policy.
    
    Args:
        force_resample: If True, forces agents to resample their macro-actions immediately
                       (used when instruction changes)
    """
    global current_instruction, instruction_input_active
 
    with torch.no_grad():
        actions = []
        new_h_states = []
        for idx, agent in enumerate(agents):
            # Check if this agent is human-controlled
            if human_agent_idx is not None and idx == human_agent_idx:
                # Human always returns stay when instruction input is active
                if instruction_input_active:
                    actions.append(0)  # Stay action
                else:
                    action = get_human_action(env, human_agent_idx, mapType)
                    actions.append(action)
                new_h_states.append(joint_h_states[agent.idx])  # Keep h_state unchanged for human
            else:
                # AI agent - if instruction input is active, all agents stay
                if instruction_input_active:
                    actions.append(0)  # Stay action
                    new_h_states.append(joint_h_states[agent.idx])  # Keep h_state unchanged
                else:
                    # AI agent uses trained policy with instruction
                    obs = joint_obs[agent.idx]
                    
                    # Pad or truncate observation to match expected dimensions
                    # (expected_input_dim is the actual model input dimension from checkpoint)
                    if obs.shape[0] < agent.expected_input_dim:
                        padding = torch.zeros(agent.expected_input_dim - obs.shape[0])
                        obs = torch.cat([obs, padding])
                    elif obs.shape[0] > agent.expected_input_dim:
                        obs = obs[:agent.expected_input_dim]
 
                    # If force_resample is True or last_valid indicates the agent should resample,
                    # we force the agent to sample a new macro-action
                    agent_valid = last_valid[agent.idx] if not force_resample else 0.0
                    
                    # Pass instruction to the model - try different parameter names
                    # mac_niacc uses 'sentence', mac_iac uses 'instruction'
                    obs_input = obs.view(1, 1, agent.expected_input_dim)
                    h_input = joint_h_states[agent.idx]
                    
                    # Check which instruction parameter the model expects
                    forward_params = inspect.signature(agent.actor_net.forward).parameters
                    if 'sentence' in forward_params:
                        action_logits, new_h_state = agent.actor_net(
                            obs_input, h_input, sentence=current_instruction
                        )
                    elif 'instruction' in forward_params:
                        action_logits, new_h_state = agent.actor_net(
                            obs_input, h_input, instruction=current_instruction
                        )
                    else:
                        # No instruction support
                        action_logits, new_h_state = agent.actor_net(obs_input, h_input)
                    # The model might output more logits than there are valid actions.
                    # Truncate to the actual number of actions available in the environment.
                    action_logits_truncated = action_logits[:, :, :env.n_action[agent.idx]]
                    action_prob = Categorical(logits=action_logits_truncated[0])
                    action = action_prob.sample().item()
                    actions.append(action)
                    new_h_states.append(new_h_state)
                    
                    if force_resample:
                        # Get the macro-action name from the environment
                        action_name = env.env.macroActionName[action] if hasattr(env.env, 'macroActionName') else str(action)
                        print(f"  Agent {idx} resampled action: {action_name} (instruction: '{current_instruction}')")
                    
    return actions, new_h_states
 
 
def get_init_inputs(env, n_agent):
    reset_result = env.reset()
    # Handle case where reset returns a single array vs list of arrays
    if isinstance(reset_result, list):
        return [torch.from_numpy(i).float() for i in reset_result], [None]*n_agent
    else:
        # If reset returns a single array, split it for multiple agents
        obs_size = env.obs_size[0] if hasattr(env.obs_size, '__getitem__') else env.obs_size
        obs_per_agent = len(reset_result) // n_agent
        obs_list = []
        for i in range(n_agent):
            start_idx = i * obs_per_agent
            end_idx = (i + 1) * obs_per_agent
            obs_list.append(reset_result[start_idx:end_idx])
        return [torch.from_numpy(i).float() for i in obs_list], [None]*n_agent
 
def setup_extended_display(env, human_agent_idx=None):
    """
    Resize the existing pygame display to add space for info panel below the game.
    """
    if not PYGAME_AVAILABLE:
        return None
 
    global _display_state
 
    if not _display_state['initialized']:
        # Get original game dimensions
        game_width = env.env.game.width
        game_height = env.env.game.height
 
        # Create extended window with info panel space below
        info_panel_height = 100 if human_agent_idx is None else 140
        extended_height = game_height + info_panel_height
 
        # Store dimensions in global state
        _display_state['game_width'] = game_width
        _display_state['game_height'] = game_height
        _display_state['info_height'] = info_panel_height
        _display_state['total_height'] = extended_height
 
        # Wait a moment for pygame to initialize from env
        time.sleep(0.1)
 
        # Only initialize pygame display if pygame is available and not already initialized
        if pygame.display.get_init():
            # Resize the existing window to extended size
            try:
                _display_state['screen'] = pygame.display.set_mode((game_width, extended_height))
                pygame.display.set_caption("Overcooked MAC_IAICC Test (Pause Mode)")
            except:
                # If that fails, just use the existing display
                _display_state['screen'] = pygame.display.get_surface()
                if _display_state['screen'] is None:
                    # Create new if none exists
                    _display_state['screen'] = pygame.display.set_mode((game_width, extended_height))
                    pygame.display.set_caption("Overcooked MAC_IAICC Test (Pause Mode)")
 
            # Initialize fonts
            pygame.font.init()
            _display_state['font'] = pygame.font.SysFont('Arial', 24, bold=True)
            _display_state['small_font'] = pygame.font.SysFont('Arial', 18)
 
        _display_state['initialized'] = True
 
    return _display_state['screen']
 
def draw_instruction_input_overlay(screen):
    """
    Draw instruction input overlay when typing (with pause indicator).
    """
    global instruction_text, _display_state
    
    if not PYGAME_AVAILABLE or not _display_state['initialized']:
        return
    
    try:
        screen_width = _display_state['game_width']
        screen_height = _display_state['total_height']
        font = _display_state['font']
        
        # Draw semi-transparent overlay
        overlay = pygame.Surface((screen_width, screen_height))
        overlay.set_alpha(200)  # More opaque to indicate pause
        overlay.fill((0, 0, 0))
        screen.blit(overlay, (0, 0))
        
        # Draw pause indicator at top
        pause_text = font.render("⏸  GAME PAUSED  ⏸", True, (255, 200, 50))
        pause_rect = pause_text.get_rect(center=(screen_width // 2, 30))
        screen.blit(pause_text, pause_rect)
        
        # Draw instruction input box
        box_width = screen_width - 100
        box_height = 120
        box_x = 50
        box_y = (screen_height - box_height) // 2
        
        # Draw box background
        pygame.draw.rect(screen, (60, 60, 60), (box_x, box_y, box_width, box_height))
        pygame.draw.rect(screen, (100, 150, 255), (box_x, box_y, box_width, box_height), 3)
        
        # Draw prompt text
        prompt_text = font.render("Enter Instruction:", True, (255, 255, 255))
        screen.blit(prompt_text, (box_x + 10, box_y + 10))
        
        # Draw input text
        input_text_surface = font.render(instruction_text + "|", True, (100, 255, 100))
        screen.blit(input_text_surface, (box_x + 10, box_y + 50))
        
        # Draw help text
        small_font = _display_state['small_font']
        help_text = small_font.render("Press ENTER to submit & resume, ESC to cancel & resume", True, (200, 200, 200))
        screen.blit(help_text, (box_x + 10, box_y + box_height - 25))
        
    except Exception as e:
        pass
 
def draw_reward_panel(env, step, total_reward, last_reward, human_agent_idx=None):
    """
    Draw reward information in a separate panel below the game area.
    """
    global _display_state
 
    if not PYGAME_AVAILABLE or not _display_state['initialized'] or not pygame.display.get_init():
        return
 
    try:
        screen = pygame.display.get_surface()
        if screen is None:
            return
 
        game_height = _display_state['game_height']
        panel_height = _display_state['info_height']
        game_width = _display_state['game_width']
        font = _display_state['font']
        small_font = _display_state['small_font']
 
        # Create info panel (separate from game area)
        panel = pygame.Surface((game_width, panel_height))
        
        # Change panel color when paused
        if instruction_input_active:
            panel.fill((60, 40, 40))  # Darker reddish when paused
        else:
            panel.fill((40, 40, 40))  # Dark gray background
 
        # Draw separator line
        pygame.draw.line(panel, (100, 100, 100), (0, 0), (game_width, 0), 2)
 
        # Render text on panel
        y_offset = 15
 
        # Step counter
        step_text = font.render(f'Step: {int(step)}', True, (255, 255, 255))
        panel.blit(step_text, (20, y_offset))
 
        # Total reward
        reward_color = (100, 255, 100) if total_reward >= 0 else (255, 100, 100)
        total_reward_text = font.render(f'Total: {total_reward:.2f}', True, reward_color)
        panel.blit(total_reward_text, (200, y_offset))
 
        # Last step reward
        last_reward_color = (150, 255, 150) if last_reward >= 0 else (255, 150, 150)
        last_reward_text = small_font.render(f'Last: {last_reward:.2f}', True, last_reward_color)
        panel.blit(last_reward_text, (430, y_offset + 5))
 
        # Show control mode
        if human_agent_idx is not None:
            mode_text = small_font.render(
                f'Human: Agent {human_agent_idx}',
                True,
                (255, 255, 100)
            )
            panel.blit(mode_text, (20, y_offset + 45))
 
            # Show controls hint
            controls_text = small_font.render(
                f'1-7=Get Items, C=Chop, D=Deliver, T=Instruction (PAUSES), Arrows=Move',
                True,
                (180, 180, 180)
            )
            panel.blit(controls_text, (170, y_offset + 45))
            
        # Show current instruction if set
        if current_instruction:
            instruction_display = small_font.render(
                f'Instruction: "{current_instruction}"',
                True,
                (100, 255, 100)
            )
            panel.blit(instruction_display, (20, y_offset + 75))
        
        # Show pause status
        if instruction_input_active:
            pause_status = small_font.render(
                "⏸  PAUSED - Waiting for instruction input",
                True,
                (255, 200, 50)
            )
            panel.blit(pause_status, (20, y_offset + 100))
 
        # Blit panel below the game area
        screen.blit(panel, (0, game_height))
 
    except pygame.error:
        # Silently handle if display is not available
        pass
 
def test(env_id, grid_dim, mapType, task, n_agent, p_id, human_agent_idx=None, render_delay=0.2):
 
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
        print(f"      HUMAN CONTROL MODE (PAUSE) - Map {mapType}")
        print(f"{'='*60}")
        print(f"Total agents: {n_agent}")
        print(f"Agent {human_agent_idx} is human-controlled (you!)")
        other_agents = [i for i in range(n_agent) if i != human_agent_idx]
        if len(other_agents) == 1:
            print(f"Agent {other_agents[0]} uses trained AI policy")
        elif len(other_agents) > 1:
            print(f"Agents {', '.join(map(str, other_agents))} use trained AI policies")
        print(f"\n{'='*60}")
        print(f"MACRO-ACTIONS:")
        print(f"{'='*60}")
        print(f"  0/Space : Stay")
        print(f"  1       : Get Tomato")
        print(f"  2       : Get Lettuce")
        print(f"  3       : Get Onion")
        print(f"  4       : Get Peas (Map D only)")
        print(f"  5       : Get Plate 1")
        print(f"  6       : Get Plate 2 (Maps B/C only)")
        print(f"  7       : Go to Knife 1")
        print(f"  8       : Go to Knife 2")
        print(f"  D (or 9): Deliver")
        print(f"  C       : Chop")
        print(f"  T       : Set Instruction (PAUSES GAME)")
        if mapType != "A":
            print(f"  X       : Go to Counter")
        print(f"\n{'='*60}")
        print(f"PRIMITIVE ACTIONS (Arrow Keys):")
        print(f"{'='*60}")
        print(f"  →/↓/←/↑ : Move Right/Down/Left/Up")
        print(f"\n{'='*60}")
        print(f"NOTE: Pressing 'T' will PAUSE all agents until you")
        print(f"      finish typing and press ENTER or ESC.")
        print(f"{'='*60}")
        print(f"Close Window to Quit")
        print(f"{'='*60}\n")
 
    # For mac_iaicc, we load SEPARATE actor policies for each agent
    # (Individual Actor, Individual Centralized Critic)
    # Manually define policy paths for each agent
    policy_paths = {
        0: "/home/willy/macro_marl_ppo/visualization/policy_nns/Overcooked/mapD/iaicc/1_agent_0_absurd-frost-7.pt",
        1: "/home/willy/macro_marl_ppo/visualization/policy_nns/Overcooked/mapD/iaicc/1_agent_1_absurd-frost-7.pt",
        # 2: "/path/to/agent_2_policy.pt",  # Uncomment and set path for 3rd agent if needed
    }
    
    print(f"Loading {n_agent} separate IAICC actor policies:")
    available_policies = []
    for i in range(n_agent):
        if i in policy_paths:
            path = policy_paths[i]
            if os.path.exists(path):
                available_policies.append((i, path))
                print(f"  Agent {i}: {path}")
            else:
                print(f"  Agent {i}: WARNING - Policy file not found: {path}")
        else:
            print(f"  Agent {i}: No policy path defined")

    agents = []
    for i in range(n_agent):
        agent = Agent()
        agent.idx = i
        
        # Skip loading policy for human-controlled agent
        if human_agent_idx is not None and i == human_agent_idx:
            agent.expected_input_dim = None  # Not needed for human agent
            agents.append(agent)
            print(f"Agent {i}: Human-controlled (no policy needed)")
            continue
        
        # Find the policy path for this agent
        policy_path = None
        for agent_idx, path in available_policies:
            if agent_idx == i:
                policy_path = path
                break
        
        if policy_path is None:
            print(f"ERROR: Policy file not found for AI agent {i}")
            sys.exit(1)
        
        # Load the policy - handle both full model and state_dict formats
        loaded_data = torch.load(policy_path, map_location=torch.device('cpu'), weights_only=False)
        
        # Check if it's already an Actor model object or a state dict
        # Use hasattr check since isinstance might fail if Actor class differs between save/load
        is_model_object = hasattr(loaded_data, 'fc1') and hasattr(loaded_data, 'forward')
        
        if is_model_object:
            # Loaded the entire model object - use it directly
            actor_net = loaded_data
            # Detect input dimension from the loaded model
            # Check if model uses instructions (has instruction_encoder or distilbert)
            if hasattr(actor_net, 'distilbert'):
                distilbert_dim = actor_net.distilbert.config.dim
                model_input_dim = actor_net.fc1.in_features - distilbert_dim
            elif hasattr(actor_net, 'use_instructions') and actor_net.use_instructions:
                # mac_iac style Actor with instruction_encoder
                model_input_dim = actor_net.fc1.in_features - 32  # rnn_layer_size default
            else:
                # Simple Actor without instruction encoding
                model_input_dim = actor_net.fc1.in_features
            print(f"Agent {i}: Loaded full model object (obs_dim={model_input_dim})")
        else:
            # Loaded a state dict - need to create model and load weights
            # Handle nested state_dict (e.g., from checkpoint)
            if isinstance(loaded_data, dict) and 'actor_net_state_dict' in loaded_data:
                state_dict = loaded_data['actor_net_state_dict']
            elif isinstance(loaded_data, dict):
                state_dict = loaded_data
            else:
                print(f"ERROR: Unexpected policy format for agent {i}: {type(loaded_data)}")
                sys.exit(1)
            
            # Detect dimensions from the state dict to handle training/testing dimension mismatches
            # fc1.weight has shape [mlp_layer_size[0], input_dim + distilbert_dim]
            fc1_weight_shape = state_dict['fc1.weight'].shape
            fc4_weight_shape = state_dict['fc4.weight'].shape
            distilbert_dim = 768  # DistilBERT embedding dimension (fixed)
            
            # Calculate the actual input and output dimensions from checkpoint
            model_input_dim = fc1_weight_shape[1] - distilbert_dim  # Subtract DistilBERT embedding size
            model_output_dim = fc4_weight_shape[0]  # Output dimension from fc4 weight shape
            
            print(f"Agent {i}: Detected from checkpoint: input_dim={model_input_dim}, output_dim={model_output_dim}")
            print(f"Agent {i}: Environment provides: obs_size={env.obs_size[i]}, action_size={env.n_action[i]}")
            
            # Create model with checkpoint dimensions (not environment dimensions)
            actor_net = Actor(
                input_dim=model_input_dim,
                output_dim=model_output_dim,
            )
            
            actor_net.load_state_dict(state_dict)
            print(f"Agent {i}: Loaded state dict successfully")
        
        actor_net.eval()
        
        # Assign this agent's actor policy
        agent.actor_net = actor_net
        
        # Set the expected input dimension based on what the model actually expects
        agent.expected_input_dim = model_input_dim
        print(f"Agent {i}: Loaded individual actor policy. Env obs_size={env.obs_size[i]}, Model expects={agent.expected_input_dim}")
        
        agents.append(agent)
 
    R = 0
    discount=0.99
    step = 0.0
    macro_step = 0.0
    n_episode = 1
    last_reward = 0.0
 
    for e in range(n_episode):
        global instruction_changed, current_instruction, instruction_persistence_counter
        
        t = 0
        last_obs, h_states = get_init_inputs(env, n_agent)
 
        # Render once to initialize pygame display
        env.render()
        
        # Now setup extended display with info panel (after pygame is initialized)
        screen = setup_extended_display(env, human_agent_idx)
        
        if PYGAME_AVAILABLE and pygame.display.get_init():
            # Initial render with info panel
            env.render()
            draw_reward_panel(env, macro_step, R, last_reward, human_agent_idx)
            pygame.display.flip()
 
        last_valid = [1.0] * n_agent
        while not t:
            # Handle instruction input events first
            handle_instruction_input_events()

            # If an instruction is active, decrement its persistence counter
            if current_instruction and instruction_persistence_counter > 0:
                instruction_persistence_counter -= 1
                if instruction_persistence_counter == 0:
                    print(f"\nInstruction '{current_instruction}' has expired after 300 steps.")
                    current_instruction = None
                    instruction_changed = True  # Trigger resample to no-instruction behavior
            
            # Check if instruction changed - if so, force agents to resample
            force_resample = instruction_changed
            if instruction_changed:
                instruction_changed = False  # Reset the flag
            
            a, h_states = get_actions_and_h_states(env, agents, last_valid, last_obs, h_states, human_agent_idx, mapType, force_resample=force_resample)
            
            # Only step the environment if not in instruction input mode
            # When paused, all agents return stay action but we still step the environment
            step_result = env.step(a)
            last_obs, r, t, info = step_result
 
            # Update rewards (even when paused, stay actions might have small penalty)
            last_reward = r[0]
            R += discount**step*last_reward
            step += 1.0
            
            if any(info['mac_done']):
                macro_step += 1.0
 
            if PYGAME_AVAILABLE and pygame.display.get_init():
                # Render environment (this will update the game area)
                env.render()
 
                # Draw reward panel below (but don't update display yet)
                draw_reward_panel(env, macro_step, R, last_reward, human_agent_idx)
                
                # Draw instruction input overlay if active
                if instruction_input_active:
                    screen = pygame.display.get_surface()
                    if screen:
                        draw_instruction_input_overlay(screen)
 
                # Single display update for the entire screen
                pygame.display.flip()
 
                # Add a small delay each step to slow down execution so human control is playable
                # and so the test runs at a more manageable speed even without a human.
                time.sleep(render_delay)
 
            # Convert observations to tensors if they're not already
            if last_obs and isinstance(last_obs[0], (int, float)):
                # If observations are scalars, convert them
                last_obs = [torch.tensor(float(o)) for o in last_obs]
            elif last_obs and hasattr(last_obs[0], 'shape'):
                # If observations are numpy arrays, convert them
                last_obs = [torch.from_numpy(o).float() for o in last_obs]
            # If already tensors, keep them as is
 
            last_valid = info['mac_done']
        
        print(f"\n{'='*60}")
        print(f"Episode Complete!")
        print(f"{'='*60}")
        print(f"Final Total Reward: {R:.2f}")
        print(f"Total Steps: {int(macro_step)}")
        print(f"{'='*60}")
        print(f"\nAnalysis:")
        print(f"  Expected value of 1 delivery: +200")
        print(f"  Step penalty accumulated: {-0.1 * step:.1f} over {int(step)} primitive steps")
        print(f"  Macro steps executed: {int(macro_step)}")
        print(f"  Discounted return above already includes step penalties and deliveries")
        print(f"{'='*60}\n")
    
    # Close environment and pygame after all episodes complete
    env.close()
    if PYGAME_AVAILABLE and pygame.display.get_init():
        pygame.quit()
        time.sleep(0.5)  # Give pygame time to close the window
    print("Environment closed.")
    sys.exit(0)
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', action='store', type=str, default='Overcooked-MA-v1')
    parser.add_argument('--grid_dim', action='store', type=int, nargs=2, default=[7,7], choices=[[7, 7], [9, 9]])
    parser.add_argument('--mapType', action='store', type=str, default="A", choices=["A", "B", "C", "D", "E", "F"])
    parser.add_argument('--task', action='store', type=int, default=6, choices=[3, 6])
    parser.add_argument('--n_agent', action='store', type=int, default=2,
                        help="Number of agents (default: 3)")
    parser.add_argument('--p_id', action='store', type=int, default=0, help="The specific policy_id")
    parser.add_argument('--human_agent_idx', action='store', type=int, nargs='?', const=2, default=None,
                        help="Index of human-controlled agent. Use without value for agent 2, or specify 0/1/2. Omit flag for all AI.")
    parser.add_argument('--render_delay', action='store', type=float, default=0.1,
                        help="Delay between frames in seconds (default: 0.01). Lower = faster, 0.001 = max speed.")
 
    test(**vars(parser.parse_args()))
 
if __name__ == '__main__':
    main()
