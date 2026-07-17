import argparse
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

from macro_marl.cores.pg_based.mac_iac.utils import Agent
from macro_marl.cores.pg_based.mac_iac.models import Actor
from gym_macro_overcooked.macActEnvWrapper import MacEnvWrapper

# Global variable to store current instruction
current_instruction = None
instruction_input_active = False
instruction_text = ''
instruction_changed = False  # Flag to trigger resampling when instruction changes

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
    0: stay, 1: get tomato, 2: get lettuce, 3: get onion, 4: get plate 1, 5: get plate 2,
    6: go to knife 1, 7: go to knife 2, 8: deliver, 9: chop, 10: right, 11: down, 12: left, 13: up

    Map B/C actions:
    0: stay, 1: get tomato, 2: get lettuce, 3: get onion, 4: get plate 1, 5: get plate 2,
    6: go to knife 1, 7: go to knife 2, 8: deliver, 9: chop, 10: go to counter, 11: right, 12: down, 13: left, 14: up
    """
    global current_instruction, instruction_input_active, instruction_text
    
    # Skip pygame operations if pygame is not available or not initialized
    if not PYGAME_AVAILABLE or not pygame.display.get_init():
        return 0  # Default to stay action
    
    # If instruction input is active, just return stay (typing mode)
    if instruction_input_active:
        return 0

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
        pygame.K_t: -1,  # Special key for instruction mode
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

def handle_instruction_input_events():
    """
    Handle keyboard events for instruction text input.
    Returns True if events were consumed (instruction mode active).
    """
    global instruction_input_active, instruction_text, current_instruction, instruction_changed
    
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
                    return True
            else:
                # In instruction input mode
                if event.key == pygame.K_RETURN:
                    # Submit instruction
                    old_instruction = current_instruction
                    current_instruction = instruction_text
                    print(f"\nInstruction set: '{instruction_text}'")
                    instruction_input_active = False
                    instruction_text = ''
                    # Set flag to trigger resampling if instruction actually changed
                    if old_instruction != current_instruction:
                        instruction_changed = True
                        print("→ Agents will resample actions based on new instruction!")
                elif event.key == pygame.K_ESCAPE:
                    # Cancel instruction input
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
    
    Args:
        force_resample: If True, forces agents to resample their macro-actions immediately
                       (used when instruction changes)
    """
    global current_instruction

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
                # AI agent uses trained policy with instruction
                obs = joint_obs[agent.idx]
                
                # NOTE: Current loaded models expect 30 dims but env provides 32 dims
                # This indicates a mismatch between model training and current environment
                # WARNING: Truncating observations loses the last 2 dimensions of information!
                # TODO: Either:
                #   1. Use models trained with correct observation size (32 dims), OR
                #   2. Use models trained with instruction support (use_instructions=True)
                #      which would properly handle observations + instructions
                
                # Pad or truncate observation to match expected dimensions
                if obs.shape[0] < agent.expected_input_dim:
                    padding = torch.zeros(agent.expected_input_dim - obs.shape[0])
                    obs = torch.cat([obs, padding])
                elif obs.shape[0] > agent.expected_input_dim:
                    obs = obs[:agent.expected_input_dim]  # LOSING INFORMATION HERE!

                # If force_resample is True or last_valid indicates the agent should resample,
                # we force the agent to sample a new macro-action
                agent_valid = last_valid[agent.idx] if not force_resample else 0.0
                
                # Pass instruction to the model
                # NOTE: If model was trained WITHOUT instruction support (use_instructions=False),
                # the instruction parameter will be ignored by the model's forward pass
                action_logits, new_h_state = agent.policy_net(
                    obs.view(1, 1, agent.expected_input_dim), 
                    joint_h_states[agent.idx],
                    instruction=current_instruction  # May be ignored if model doesn't support instructions
                )
                action_prob = Categorical(logits=action_logits[0])
                action = action_prob.sample().item()
                actions.append(action)
                new_h_states.append(new_h_state)
                
                if force_resample:
                    # Get the macro-action name from the environment
                    action_name = env.env.macroActionName[action] if hasattr(env.env, 'macroActionName') else str(action)
                    print(f"  Agent {idx} resampled action: {action_name} (instruction: '{current_instruction}')")
                    
    return actions, new_h_states


def get_init_inputs(env,n_agent):
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
        info_panel_height = 100 if human_agent_idx is None else 120
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
                pygame.display.set_caption("Overcooked Multi-Agent")
            except:
                # If that fails, just use the existing display
                _display_state['screen'] = pygame.display.get_surface()
                if _display_state['screen'] is None:
                    # Create new if none exists
                    _display_state['screen'] = pygame.display.set_mode((game_width, extended_height))
                    pygame.display.set_caption("Overcooked Multi-Agent")

            # Initialize fonts
            pygame.font.init()
            _display_state['font'] = pygame.font.SysFont('Arial', 24, bold=True)
            _display_state['small_font'] = pygame.font.SysFont('Arial', 18)

        _display_state['initialized'] = True

    return _display_state['screen']

def draw_instruction_input_overlay(screen):
    """
    Draw instruction input overlay when typing.
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
        overlay.set_alpha(180)
        overlay.fill((0, 0, 0))
        screen.blit(overlay, (0, 0))
        
        # Draw instruction input box
        box_width = screen_width - 100
        box_height = 100
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
        screen.blit(input_text_surface, (box_x + 10, box_y + 45))
        
        # Draw help text
        small_font = _display_state['small_font']
        help_text = small_font.render("Press ENTER to submit, ESC to cancel", True, (200, 200, 200))
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
                f'1-7=Get Items, C=Chop, D=Deliver, T=Instruction, Arrows=Move',
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
            panel.blit(instruction_display, (20, y_offset + 70))

        # Blit panel below the game area
        screen.blit(panel, (0, game_height))

        # Don't update display here - let the main loop handle it
        # pygame.display.update((0, game_height, game_width, panel_height))
    except pygame.error:
        # Silently handle if display is not available
        pass

def test(env_id, grid_dim, mapType, task, n_agent, p_id, human_agent_idx=None, render_delay=0.03):

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
        print(f"         HUMAN CONTROL MODE - Map {mapType}")
        print(f"{'='*60}")
        print(f"Agent {human_agent_idx} is human-controlled (you!)")
        other_agents = [i for i in range(n_agent) if i != human_agent_idx]
        if len(other_agents) == 1:
            print(f"Agent {other_agents[0]} uses trained policy")
        else:
            print(f"Agents {', '.join(map(str, other_agents))} use trained policies")
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
        print(f"  T       : Set Instruction")
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
        
        # Load full instruction-aware model from 4_agent_*.pt files
        policy_path = os.path.join(os.path.dirname(__file__), "policy_nns", "Overcooked", "map" + mapType, "4_agent_" + str(i) + ".pt")
        
        print(f"Loading policy from: {policy_path}")
        agent.policy_net = torch.load(policy_path, weights_only=False)
        agent.policy_net.eval()
        
        # Get expected input dimension from the loaded model
        # Check the fc1 layer to determine the input dimension
        if hasattr(agent.policy_net, 'fc1'):
            expected_input_dim = agent.policy_net.fc1.in_features
        else:
            # Fallback: get from state dict
            state_dict = agent.policy_net.state_dict()
            expected_input_dim = state_dict['fc1.weight'].shape[1]
        
        agent.expected_input_dim = expected_input_dim
        print(f"Agent {i}: Current env obs_size={env.obs_size[i]}, Model expects={expected_input_dim}")
        
        agents.append(agent)

    R = 0
    discount=0.99
    step = 0.0
    n_episode = 1
    last_reward = 0.0

    for e in range(n_episode):
        global instruction_changed
        
        t = 0
        last_obs, h_states = get_init_inputs(env, n_agent)

        # Render once to initialize pygame display
        env.render()
        
        # Now setup extended display with info panel (after pygame is initialized)
        screen = setup_extended_display(env, human_agent_idx)
        
        if PYGAME_AVAILABLE and pygame.display.get_init():
            # Initial render with info panel
            env.render()
            draw_reward_panel(env, step, R, last_reward, human_agent_idx)
            pygame.display.flip()

        last_valid = [1.0] * n_agent
        while not t:
            # Handle instruction input events first
            handle_instruction_input_events()
            
            # Check if instruction changed - if so, force agents to resample
            force_resample = instruction_changed
            if instruction_changed:
                instruction_changed = False  # Reset the flag
            
            a, h_states = get_actions_and_h_states(env, agents, last_valid, last_obs, h_states, human_agent_idx, mapType, force_resample=force_resample)
            step_result = env.step(a)
            last_obs, r, t, info = step_result

            # Update rewards
            last_reward = r[0]
            R += discount**step*last_reward
            step += 1.0

            if PYGAME_AVAILABLE and pygame.display.get_init():
                # Render environment (this will update the game area)
                env.render()

                # Draw reward panel below (but don't update display yet)
                draw_reward_panel(env, step, R, last_reward, human_agent_idx)
                
                # Draw instruction input overlay if active
                if instruction_input_active:
                    screen = pygame.display.get_surface()
                    if screen:
                        draw_instruction_input_overlay(screen)

                # Single display update for the entire screen
                pygame.display.flip()

                # Add a small delay for human control mode to make it playable
                if human_agent_idx is not None:
                    time.sleep(render_delay)  # Configurable delay for control responsiveness

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
        print(f"Total Steps: {int(step)}")
        print(f"{'='*60}\n")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', action='store', type=str, default='Overcooked-MA-v1')
    parser.add_argument('--grid_dim', action='store', type=int, nargs=2, default=[7,7], choices=[[7, 7], [9, 9]])
    parser.add_argument('--mapType', action='store', type=str, default="A", choices=["A", "B", "C", "D", "E", "F"])
    parser.add_argument('--task', action='store', type=int, default=6, choices=[3, 6])
    parser.add_argument('--n_agent', action='store', type=int, default=3)
    parser.add_argument('--p_id', action='store', type=int, default=0, help="The specific policy_id")
    parser.add_argument('--human_agent_idx', action='store', type=int, default=None, 
                        help="Index of human-controlled agent (0 or 1). If None, all agents use trained policies.")
    parser.add_argument('--render_delay', action='store', type=float, default=0.08,
                        help="Delay between frames in seconds (default: 0.03). Lower = faster, 0.001 = max speed.")

    test(**vars(parser.parse_args()))

if __name__ == '__main__':
    main()

