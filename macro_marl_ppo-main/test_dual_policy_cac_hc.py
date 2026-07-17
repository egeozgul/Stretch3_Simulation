import argparse
import numpy as np
import torch
import os
import sys
import time
import gym

# Add paths for custom modules BEFORE importing them
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "gym-macro-overcooked"))

# Import pygame only if available
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    pygame = None
from torch.distributions import Categorical

from macro_marl.cores.pg_based.mac_cac.models import Actor
from macro_marl.cores.pg_based.mac_cac.utils import get_joint_avail_actions, get_conditional_action, get_conditional_logits
from gym_macro_overcooked.macActEnvWrapper import MacEnvWrapper

# Global variables
current_instruction = None
instruction_input_active = False
instruction_text = ''
instruction_changed = False
using_instruction_policy = False  # Track which policy is active

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
                    print("\n⏸  GAME PAUSED - Enter instruction (will be used once, then auto-switch back)")
                    return True
            else:
                # In instruction input mode
                if event.key == pygame.K_RETURN:
                    # Submit instruction
                    old_instruction = current_instruction
                    current_instruction = instruction_text if instruction_text else None
                    if current_instruction:
                        print(f"\n▶  RESUMING - Instruction: '{instruction_text}' (will be used once)")
                    else:
                        print(f"\n▶  RESUMING - No instruction")
                    instruction_input_active = False
                    instruction_text = ''
                    # Set flag to trigger resampling if instruction actually changed
                    if old_instruction != current_instruction:
                        instruction_changed = True
                        if current_instruction:
                            print(f"→ Will use INSTRUCTION policy for next resample (one-time)")
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

def get_init_inputs(env, n_agent):
    reset_result = env.reset()
    # Handle case where reset returns a single array vs list of arrays
    # Controller expects list of tensors shaped [1, obs_dim] (not [1, 1, obs_dim])
    if isinstance(reset_result, list):
        return [torch.from_numpy(i).float().unsqueeze(0) for i in reset_result], None
    else:
        # If reset returns a single array, split it for multiple agents
        obs_size = env.obs_size[0] if hasattr(env.obs_size, '__getitem__') else env.obs_size
        obs_per_agent = len(reset_result) // n_agent
        obs_list = []
        for i in range(n_agent):
            start_idx = i * obs_per_agent
            end_idx = (i + 1) * obs_per_agent
            obs_list.append(reset_result[start_idx:end_idx])
        return [torch.from_numpy(i).float().unsqueeze(0) for i in obs_list], None

def setup_extended_display(env):
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
        info_panel_height = 140
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
                pygame.display.set_caption("Overcooked - Dual Policy Testing")
            except:
                # If that fails, just use the existing display
                _display_state['screen'] = pygame.display.get_surface()
                if _display_state['screen'] is None:
                    # Create new if none exists
                    _display_state['screen'] = pygame.display.set_mode((game_width, extended_height))
                    pygame.display.set_caption("Overcooked - Dual Policy Testing")

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
        prompt_text = font.render("Enter Instruction (used once, then auto-switches back):", True, (255, 255, 255))
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

def draw_reward_panel(env, step, total_reward, last_reward):
    """
    Draw reward information and policy status in a separate panel below the game area.
    """
    global _display_state, using_instruction_policy

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

        # Show current policy being used
        y_offset += 40
        policy_text = "DUAL POLICY MODE"
        policy_color = (255, 255, 100)
        policy_label = font.render(policy_text, True, policy_color)
        panel.blit(policy_label, (20, y_offset))

        # Show which sub-policy is active
        y_offset += 35
        if using_instruction_policy:
            active_text = "→ Using: INSTRUCTION Policy (hard-coded)"
            active_color = (100, 255, 100)
        else:
            active_text = "→ Using: PRETRAINED Policy (frozen)"
            active_color = (100, 200, 255)
        active_label = small_font.render(active_text, True, active_color)
        panel.blit(active_label, (30, y_offset))

        # Show current instruction if set
        y_offset += 30
        if current_instruction:
            instruction_display = small_font.render(
                f'Instruction: "{current_instruction}"',
                True,
                (100, 255, 100)
            )
            panel.blit(instruction_display, (30, y_offset))
        else:
            no_instr_text = small_font.render(
                'No instruction - using pretrained policy',
                True,
                (180, 180, 180)
            )
            panel.blit(no_instr_text, (30, y_offset))
        
        # Show controls hint
        y_offset += 30
        controls_text = small_font.render(
            'Press T to enter instruction (used once, then auto-switches back)',
            True,
            (180, 180, 180)
        )
        panel.blit(controls_text, (30, y_offset))
        
        # Show pause status
        if instruction_input_active:
            pause_status = small_font.render(
                "⏸  PAUSED - Waiting for instruction input",
                True,
                (255, 200, 50)
            )
            panel.blit(pause_status, (20, y_offset + 30))

        # Blit panel below the game area
        screen.blit(panel, (0, game_height))

    except pygame.error:
        # Silently handle if display is not available
        pass

def load_actor_policy(policy_path, use_instructions, **actor_kwargs):
    """
    Initializes and loads a pre-trained Actor model.
    
    Args:
        policy_path: Path to the checkpoint file
        use_instructions: Whether this actor uses instructions
        **actor_kwargs: Additional arguments for Actor initialization
        
    Returns:
        Loaded and eval-mode Actor model
    """
    print(f"Loading policy from {policy_path}...")
    
    # Initialize the actor with its specific configuration
    actor = Actor(
        use_instructions=use_instructions,
        **actor_kwargs
    )
    
    # Load the checkpoint
    checkpoint = torch.load(policy_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'actor_net' in checkpoint:
        actor.load_state_dict(checkpoint['actor_net'])
    else:
        actor.load_state_dict(checkpoint)
        
    actor.eval()
    print("✓ Policy loaded and set to eval() mode.")
    return actor

def get_hardcoded_instruction_action(instruction, env, n_agent):
    """
    Hard-coded instruction policy that maps instructions to actions.
    
    Args:
        instruction: The instruction string (e.g., "deliver", "get tomato")
        env: The environment
        n_agent: Number of agents
        
    Returns:
        Tuple of action indices for each agent, or None if instruction doesn't match
    """
    if not instruction:
        return None
    
    # Normalize instruction to lowercase for matching
    instruction_lower = instruction.lower().strip()
    
    # Search for matching action in the macro action names
    target_action_idx = None
    for idx, action_name in enumerate(env.macroActionName):
        action_name_lower = action_name.lower()
        # Check if instruction matches the action name
        if instruction_lower in action_name_lower or action_name_lower in instruction_lower:
            target_action_idx = idx
            print(f"Hard-coded policy: Matched instruction '{instruction}' to action '{action_name}' (idx={idx})")
            break
    
    if target_action_idx is None:
        print(f"Warning: No action found matching instruction '{instruction}'")
        print(f"Available actions: {env.macroActionName}")
        return None
    
    # All agents perform the same action
    actions = tuple([target_action_idx] * n_agent)
    return actions

def test(env_id, grid_dim, mapType, task, n_agent, 
         no_instr_policy_path, instr_policy_path,
         a_mlp_layer_size, a_rnn_layer_size,
         instruction_fusion, render_delay=0.05):

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

    print(f"\n{'='*80}")
    print(f"      DUAL POLICY TEST - Overcooked Map {mapType}")
    print(f"{'='*80}")
    print(f"Environment: {n_agent} agents, actions: {env.n_action}")
    print(f"Observation sizes: {env.obs_size}")
    print(f"\nTwo Policies:")
    print(f"  1. NO-INSTRUCTION Policy (pretrained): {no_instr_policy_path}")
    print(f"  2. INSTRUCTION Policy (hard-coded): Maps instructions to actions")
    print(f"\nPolicy Selection:")
    print(f"  - No instruction → Use NO-INSTRUCTION policy")
    print(f"  - With instruction → Use HARD-CODED instruction policy")
    print(f"\nHard-coded Instruction Examples:")
    print(f"  - 'deliver' → All agents perform deliver action (once)")
    print(f"  - 'get tomato' → All agents perform get tomato action (once)")
    print(f"\nBehavior:")
    print(f"  - Instruction is used for ONE action sample only")
    print(f"  - After sampling, automatically switches back to no-instruction policy")
    print(f"\nControls:")
    print(f"  - Press 'T' to enter instruction")
    print(f"{'='*80}\n")

    # Calculate input/output dimensions
    input_dim = sum(env.obs_size)
    output_dim = np.prod(env.n_action)

    # Define common actor arguments
    actor_args = {
        'input_dim': input_dim,
        'output_dim': output_dim,
        'mlp_layer_size': [a_mlp_layer_size, a_mlp_layer_size],
        'rnn_layer_size': a_rnn_layer_size
    }

    # Load NO-INSTRUCTION policy
    no_instr_actor = load_actor_policy(
        policy_path=no_instr_policy_path,
        use_instructions=False,
        **actor_args
    )

    # Note: No longer loading instruction policy - using hard-coded policy instead
    print("✓ Using hard-coded instruction policy (no loading required)")

    R = 0
    discount = 0.99
    step = 0.0
    n_episode = 1
    last_reward = 0.0

    for e in range(n_episode):
        global instruction_changed, current_instruction, using_instruction_policy
        
        t = 0
        last_obs, h_state = get_init_inputs(env, n_agent)

        # Render once to initialize pygame display
        env.render()
        
        # Now setup extended display with info panel (after pygame is initialized)
        screen = setup_extended_display(env)
        
        if PYGAME_AVAILABLE and pygame.display.get_init():
            # Initial render with info panel
            env.render()
            draw_reward_panel(env, step, R, last_reward)
            pygame.display.flip()

        last_valid = [1] * n_agent
        last_actions = [torch.LongTensor([[0]]) for _ in range(n_agent)]
        valids_list = [torch.LongTensor([[1]]) for _ in range(n_agent)]
        avail_actions = [torch.ones(1, env.n_action[i]) for i in range(n_agent)]
        
        while not t:
            # Handle instruction input events first
            handle_instruction_input_events()

            # If in instruction input mode, skip to rendering
            if instruction_input_active:
                if PYGAME_AVAILABLE and pygame.display.get_init():
                    env.render()
                    draw_reward_panel(env, step, R, last_reward)
                    screen = pygame.display.get_surface()
                    if screen:
                        draw_instruction_input_overlay(screen)
                    pygame.display.flip()
                    time.sleep(render_delay)
                continue

            # Check if instruction changed
            if instruction_changed:
                instruction_changed = False
                # Force resampling by setting last_valid to all 1s
                last_valid = [1] * n_agent
                valids_list = [torch.LongTensor([[1]]) for _ in range(n_agent)]
            
            # Select actions
            if max(last_valid) == 1:
                # Check if we should use hard-coded instruction policy
                if current_instruction:
                    using_instruction_policy = True
                    # Use hard-coded policy (only once)
                    actions = get_hardcoded_instruction_action(current_instruction, env, n_agent)
                    # Clear instruction after using it once - switch back to no-instruction policy
                    current_instruction = None
                    print("→ Instruction used once, switching back to NO-INSTRUCTION policy for next resample")
                    if actions is None:
                        # Fallback to no-instruction policy if instruction doesn't match
                        print("Falling back to no-instruction policy")
                        using_instruction_policy = False
                        with torch.no_grad():
                            # Concatenate observations
                            joint_obs = torch.cat(last_obs, dim=1).view(1, 1, -1)
                            
                            # Get action logits from no-instruction actor
                            action_logits, h_state = no_instr_actor(
                                joint_obs,
                                h_state,
                                eps=0.0,
                                test_mode=True
                            )
                            
                            # Get joint available actions
                            joint_avail_actions = get_joint_avail_actions(avail_actions)
                            
                            # Apply conditional masking
                            last_actions_tensor = torch.cat(last_actions, dim=1)
                            valids_tensor = torch.cat(valids_list, dim=1).bool()
                            
                            action_logits = get_conditional_logits(
                                action_logits,
                                get_conditional_action(last_actions_tensor, valids_tensor),
                                joint_avail_actions,
                                env.n_action
                            )
                            
                            # Sample action
                            action_prob = Categorical(logits=action_logits[0])
                            action = action_prob.sample().item()
                            actions = np.unravel_index(action, env.n_action)
                else:
                    # Use no-instruction policy
                    using_instruction_policy = False
                    with torch.no_grad():
                        # Concatenate observations
                        joint_obs = torch.cat(last_obs, dim=1).view(1, 1, -1)
                        
                        # Get action logits from no-instruction actor
                        action_logits, h_state = no_instr_actor(
                            joint_obs,
                            h_state,
                            eps=0.0,
                            test_mode=True
                        )
                        
                        # Get joint available actions
                        joint_avail_actions = get_joint_avail_actions(avail_actions)
                        
                        # Apply conditional masking
                        last_actions_tensor = torch.cat(last_actions, dim=1)
                        valids_tensor = torch.cat(valids_list, dim=1).bool()
                        
                        action_logits = get_conditional_logits(
                            action_logits,
                            get_conditional_action(last_actions_tensor, valids_tensor),
                            joint_avail_actions,
                            env.n_action
                        )
                        
                        # Sample action
                        action_prob = Categorical(logits=action_logits[0])
                        action = action_prob.sample().item()
                        actions = np.unravel_index(action, env.n_action)
                
                # Update last_actions
                last_actions = [torch.LongTensor([[a]]) for a in actions]
            else:
                # Use previous actions
                actions = [la.item() for la in last_actions]
            
            # Print when new actions are sampled (resampled)
            if max(last_valid) == 1:
                resampled_agents = [i for i, v in enumerate(last_valid) if v == 1]
                action_names = [env.macroActionName[actions[i]] for i in resampled_agents]
                print(f"RESAMPLED: {', '.join([f'Agent {i}: {name}' for i, name in zip(resampled_agents, action_names)])}")
            
            # Step the environment
            step_result = env.step(actions)
            next_obs, r, t, info = step_result

            # Update rewards
            last_reward = r[0]
            R += discount**step * last_reward
            step += 1.0

            # Convert observations to tensors
            if isinstance(next_obs, list):
                last_obs = [torch.from_numpy(o).float().unsqueeze(0) for o in next_obs]
            else:
                obs_per_agent = len(next_obs) // n_agent
                obs_list = []
                for i in range(n_agent):
                    start_idx = i * obs_per_agent
                    end_idx = (i + 1) * obs_per_agent
                    agent_obs = torch.FloatTensor(next_obs[start_idx:end_idx]).unsqueeze(0)  # [1, obs_dim]
                    obs_list.append(agent_obs)
                last_obs = obs_list

            last_valid = info['mac_done']
            valids_list = [torch.LongTensor([[v]]) for v in last_valid]

            if PYGAME_AVAILABLE and pygame.display.get_init():
                # Render environment
                env.render()

                # Draw reward panel
                draw_reward_panel(env, step, R, last_reward)

                # Single display update
                pygame.display.flip()

                # Add delay
                time.sleep(render_delay)
        
        print(f"\n{'='*80}")
        print(f"Episode Complete!")
        print(f"{'='*80}")
        print(f"Final Total Reward: {R:.2f}")
        print(f"Total Steps: {int(step)}")
        print(f"{'='*80}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', action='store', type=str, default='Overcooked-MA-v1')
    parser.add_argument('--grid_dim', action='store', type=int, nargs=2, default=[7,7], choices=[[7, 7], [9, 9]])
    parser.add_argument('--mapType', action='store', type=str, default="A", choices=["A", "B", "C", "D", "E", "F"])
    parser.add_argument('--task', action='store', type=int, default=6, choices=[3, 6])
    parser.add_argument('--n_agent', action='store', type=int, default=2)
    parser.add_argument('--no_instr_policy_path', action='store', type=str, required=True ,default= '/home/willy/macro_marl_ppo/visualization/policy_nns/Overcooked/mapA/instr_disabled.pt',
                        help="Path to no-instruction policy checkpoint")
    parser.add_argument('--instr_policy_path', action='store', type=str, required=False , default=None,
                        help="(DEPRECATED - not used) Path to instruction policy checkpoint")
    parser.add_argument('--a_mlp_layer_size', action='store', type=int, default=32)
    parser.add_argument('--a_rnn_layer_size', action='store', type=int, default=32)
    parser.add_argument('--instruction_fusion', action='store', type=str, default='concat',
                        choices=['concat', 'film', 'attention'])
    parser.add_argument('--render_delay', action='store', type=float, default=0.05,
                        help="Delay between frames in seconds (default: 0.05)")

    test(**vars(parser.parse_args()))

if __name__ == '__main__':
    main()
