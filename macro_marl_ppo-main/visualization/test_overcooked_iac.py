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
from macro_marl.cores.pg_based.mac_iac.models import Actor  # Using iac model with individual agents
from gym_macro_overcooked.macActEnvWrapper import MacEnvWrapper
 
# Global variable to store current instruction
current_instruction = None
instruction_input_active = False
instruction_text = ''
instruction_changed = False  # Flag to trigger resampling when instruction changes
instruction_persistence_counter = 0
paused = False  # Game pause state

# Hardcoded policy loading config.
# Update POLICY_RUN_DIR to the folder you want to load from under POLICY_BASE_DIR.
POLICY_BASE_DIR = "/home/willy/macro_marl_ppo/experiments/Overcooked/policy_nns"
POLICY_RUN_DIR = "/home/willy/macro_marl_ppo/experiments/Overcooked/policy_nns/mac_iac_new_overcooked_D_desktop_glacier_0"

# Instruction list must match training order for embedding index lookup
# This must exactly match OVERCOOKED_A_INSTRUCTIONS in experiments/Overcooked/mac_iac.sh
INSTRUCTION_LIST = [
    "don't use the right cutting board",   # index 0
    "don't use the left cutting board",    # index 1
    # "let me do all the chopping",        # commented out in training script
]

INSTRUCTION_ALIASES = {
    "dont use the right cutting board": "don't use the right cutting board",
    "dont use the left cutting board": "don't use the left cutting board",
    "don't touch the right cutting board": "don't use the right cutting board",
    "don't touch the left cutting board": "don't use the left cutting board",
    "don't use right cutting board": "don't use the right cutting board",
    "don't use left cutting board": "don't use the left cutting board",
    "dont use right cutting board": "don't use the right cutting board",
    "dont use left cutting board": "don't use the left cutting board",
}


def resolve_policy_dir(map_type):
    """Resolve policy directory from hardcoded base path and optional run folder override."""
    if POLICY_RUN_DIR:
        return os.path.join(POLICY_BASE_DIR, POLICY_RUN_DIR)

    preferred_dir = os.path.join(POLICY_BASE_DIR, f"mac_iac_overcooked_{map_type}")
    if os.path.isdir(preferred_dir):
        return preferred_dir

    # Fallback: choose the first matching run directory for this map.
    prefix = f"mac_iac_overcooked_{map_type}_"
    if os.path.isdir(POLICY_BASE_DIR):
        matching_dirs = sorted(
            d for d in os.listdir(POLICY_BASE_DIR)
            if d.startswith(prefix) and os.path.isdir(os.path.join(POLICY_BASE_DIR, d))
        )
        if matching_dirs:
            return os.path.join(POLICY_BASE_DIR, matching_dirs[0])

    return preferred_dir


def _normalize_instruction(text):
    normalized = text.lower().strip()
    normalized = normalized.replace("’", "'").replace("‘", "'")
    normalized = " ".join(normalized.split())
    return normalized


def instruction_to_embedding(text, model, device=None):
    """Convert instruction text to BERT embedding using the model's encode_instruction().
    
    With BERT sentence-transformers, any natural language instruction is understood
    semantically — paraphrases like "don't touch the left cutting board" and 
    "don't go to the left cutting board" will produce similar embeddings automatically.
    
    Args:
        text: Instruction text string (any natural language).
        model: Actor model with encode_instruction() method.
        device: Target device.
    Returns:
        Embedding tensor of shape (1, instruction_dim) or None if no instructions.
    """
    if not hasattr(model, 'use_instructions') or not model.use_instructions:
        return None
    if not text:
        # No instruction: return zero vector
        return torch.zeros(1, model.instruction_dim, device=device)
    
    with torch.no_grad():
        emb = model.encode_instruction(text)  # (1, instruction_dim)
        if device is not None:
            emb = emb.to(device)
    print(f"[INFO] Instruction '{text}' -> BERT embedding shape: {emb.shape}")
    return emb


def get_prohibited_actions(instruction_text, macro_action_names):
    """
    Map an instruction to prohibited action indices using the environment's
    macroActionName list. Returns a list of action indices to mask, or empty list.
    """
    if not instruction_text:
        return []
    normalized = _normalize_instruction(instruction_text)
    canonical = INSTRUCTION_ALIASES.get(normalized, normalized)

    # Build index lookup from the actual environment action names
    name_to_idx = {name: i for i, name in enumerate(macro_action_names)}
    KNIFE_1 = name_to_idx.get("go to knife 1")
    KNIFE_2 = name_to_idx.get("go to knife 2")
    CHOP = name_to_idx.get("chop")

    if canonical == "don't use the right cutting board":
        return [i for i in [KNIFE_1] if i is not None]
    elif canonical == "don't use the left cutting board":
        return [i for i in [KNIFE_2] if i is not None]
    elif canonical == "let me do all the chopping":
        return [i for i in [KNIFE_1, KNIFE_2, CHOP] if i is not None]
    return []

 
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
    't' key to pause/unpause the game.
 
    Map A actions:
    0: stay, 1: get tomato, 2: get lettuce, 3: get onion, 4: get plate 1, 5: get plate 2,
    6: go to knife 1, 7: go to knife 2, 8: deliver, 9: chop, 10: right, 11: down, 12: left, 13: up
 
    Map B/C actions:
    0: stay, 1: get tomato, 2: get lettuce, 3: get onion, 4: get plate 1, 5: get plate 2,
    6: go to knife 1, 7: go to knife 2, 8: deliver, 9: chop, 10: go to counter, 11: right, 12: down, 13: left, 14: up
    """
    global current_instruction, instruction_input_active, instruction_text, paused
    
    # Skip pygame operations if pygame is not available or not initialized
    if not PYGAME_AVAILABLE or pygame is None or not pygame.display.get_init():
        return 0  # Default to stay action
    
    # If instruction input is active, just return stay (typing mode)
    if instruction_input_active:
        return 0
 
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
            pygame.K_m: env.macroActionName.index("get blended bowl"),
            pygame.K_p: env.macroActionName.index("get patty"),
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
    
    # Combine all actions
    all_actions = {**common_actions, **movement_actions}
    
    # Get pressed keys
    keys = pygame.key.get_pressed()
    
    for key, action in all_actions.items():
        if keys[key]:
            if action == -1:  # Instruction mode triggered by 't' key
                return -1
            return action
    
    return 0  # Default to stay action if no key pressed


def reset_macro_actions(env):
    """Force all agents to resample macro-actions on the next step."""
    try:
        macro_agents = env.env.macroAgent
    except AttributeError:
        return
    for macro_agent in macro_agents:
        macro_agent.cur_macro_action_done = True


def handle_instruction_input_events():
    """
    Handle keyboard events for instruction input mode.
    When 't' is pressed, activates instruction input mode.
    Type the instruction and press ENTER to confirm or ESC to cancel.
    """
    global current_instruction, instruction_input_active, instruction_text, instruction_changed, instruction_persistence_counter
    
    if not PYGAME_AVAILABLE or pygame is None or not pygame.display.get_init():
        return
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        
        if event.type == pygame.KEYDOWN:
            if instruction_input_active:
                # We're in instruction input mode
                if event.key == pygame.K_RETURN:
                    # Confirm instruction (or clear if empty)
                    if instruction_text.strip():  # Non-empty instruction
                        current_instruction = instruction_text
                        print(f"\n{'='*60}")
                        print(f"INSTRUCTION SET FOR ALL AGENTS: '{current_instruction}'")
                        print(f"{'='*60}")
                    else:  # Empty instruction - clear current instruction
                        current_instruction = None
                        print(f"\n{'='*60}")
                        print(f"INSTRUCTION CLEARED FOR ALL AGENTS (no instruction active)")
                        print(f"{'='*60}")
                    instruction_input_active = False
                    instruction_text = ''
                    instruction_changed = True  # Flag that instruction changed
                elif event.key == pygame.K_ESCAPE:
                    # Cancel instruction input
                    print("\nInstruction input cancelled")
                    instruction_input_active = False
                    instruction_text = ''
                elif event.key == pygame.K_BACKSPACE:
                    # Delete last character
                    instruction_text = instruction_text[:-1]
                else:
                    # Add character to instruction
                    instruction_text += event.unicode
            else:
                # Not in instruction input mode, check for 't' key to activate it
                if event.key == pygame.K_t:
                    instruction_input_active = True
                    instruction_text = ''
                    print("\n========== INSTRUCTION INPUT MODE ==========")
                    print("Type your instruction and press ENTER to confirm")
                    print("or press ENTER with empty text to clear instruction")
                    print("or press ESC to cancel.")
                    print("==========================================")


def draw_instruction_input_overlay(screen):
    """Draw the instruction input overlay on the screen"""
    if not PYGAME_AVAILABLE or pygame is None or not pygame.display.get_init():
        return
        
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)
    
    # Semi-transparent overlay
    overlay = pygame.Surface(screen.get_size())
    overlay.set_alpha(200)
    overlay.fill((50, 50, 50))
    screen.blit(overlay, (0, 0))
    
    # Title
    title_text = font.render("INSTRUCTION INPUT MODE", True, (255, 255, 100))
    title_rect = title_text.get_rect(center=(screen.get_width() // 2, 100))
    screen.blit(title_text, title_rect)
    
    # Instructions
    instruction_lines = [
        "Type your instruction and press ENTER to confirm",
        "Press ESC to cancel",
        "",
        "Current input:",
    ]
    
    y_offset = 150
    for line in instruction_lines:
        text = small_font.render(line, True, (255, 255, 255))
        text_rect = text.get_rect(center=(screen.get_width() // 2, y_offset))
        screen.blit(text, text_rect)
        y_offset += 30
    
    # Current instruction text (with cursor)
    input_display = instruction_text + "|"
    input_text = font.render(input_display, True, (100, 255, 100))
    input_rect = input_text.get_rect(center=(screen.get_width() // 2, y_offset + 20))
    screen.blit(input_text, input_rect)


def setup_extended_display(env, human_agent_idx=None):
    """Setup pygame display with extended height for info panel"""
    global _display_state
    
    if not PYGAME_AVAILABLE or pygame is None or not pygame.display.get_init():
        return None
    
    # Get the current game surface dimensions
    game_surface = pygame.display.get_surface()
    if game_surface is None:
        return None
    
    game_width, game_height = game_surface.get_size()
    info_height = 150  # Height for info panel
    total_height = game_height + info_height
    
    # Create new display with extended height
    screen = pygame.display.set_mode((game_width, total_height))
    pygame.display.set_caption("Overcooked MAC_IAC Test - 3 Agent Policies")
    
    # Initialize fonts
    pygame.font.init()
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)
    
    # Store display state
    _display_state = {
        'screen': screen,
        'initialized': True,
        'game_width': game_width,
        'game_height': game_height,
        'info_height': info_height,
        'total_height': total_height,
        'font': font,
        'small_font': small_font
    }
    
    return screen


def draw_reward_panel(env, step, total_reward, last_reward, human_agent_idx=None):
    """Draw the reward information panel below the game area"""
    global _display_state, current_instruction, instruction_input_active, paused
    
    if not PYGAME_AVAILABLE or pygame is None or not pygame.display.get_init() or not _display_state['initialized']:
        return
    
    try:
        screen = _display_state['screen']
        game_width = _display_state['game_width']
        game_height = _display_state['game_height']
        info_height = _display_state['info_height']
        font = _display_state['font']
        small_font = _display_state['small_font']
        
        # Create info panel
        panel = pygame.Surface((game_width, info_height))
        panel.fill((40, 40, 40))
        
        # Add border
        pygame.draw.rect(panel, (100, 100, 100), panel.get_rect(), 2)
        
        y_offset = 10
        
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
                f'1-7=Get Items, C=Chop, D=Deliver, T=Pause/Resume, Arrows=Move',
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
        if paused:
            pause_status = small_font.render(
                "⏸  PAUSED - Press T to resume",
                True,
                (255, 200, 50)
            )
            panel.blit(pause_status, (20, y_offset + 100))
 
        # Blit panel below the game area
        screen.blit(panel, (0, game_height))
 
        # Don't update display here - let the main loop handle it
    except pygame.error:
        # Silently handle if display is not available
        pass


def get_actions_and_h_states(env, agents, last_valid, obs_list, h_states_list, human_agent_idx=None, mapType="A", force_resample=False):
    """
    Get actions for all agents. If human_agent_idx is specified, that agent uses keyboard control.
    For MAC_IAC, each agent has its own policy network.
    
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
                new_h_states.append(h_states_list[agent.idx])  # Keep h_state unchanged for human
            else:
                # AI agent - if instruction input is active, all agents stay
                if instruction_input_active:
                    actions.append(0)  # Stay action
                    new_h_states.append(h_states_list[agent.idx])  # Keep h_state unchanged
                else:
                    # AI agent uses trained policy with instruction
                    obs = obs_list[agent.idx]
                    
                    # Pad or truncate observation to match expected dimensions
                    if obs.shape[0] < agent.expected_input_dim:
                        padding = torch.zeros(agent.expected_input_dim - obs.shape[0])
                        obs = torch.cat([obs, padding])
                    elif obs.shape[0] > agent.expected_input_dim:
                        obs = obs[:agent.expected_input_dim]
 
                    # If force_resample is True or last_valid indicates the agent should resample,
                    # we force the agent to sample a new macro-action
                    agent_valid = last_valid[agent.idx] if not force_resample else 0.0
                    
                    # Use model's learned embedding instead of one-hot
                    instruction_emb = instruction_to_embedding(
                        current_instruction,
                        agent.policy_net,
                        device=obs.device
                    )

                    action_logits, new_h_state = agent.policy_net(
                        obs.view(1, 1, agent.expected_input_dim),
                        h_states_list[agent.idx],
                        instruction_emb=instruction_emb
                    )
                    # The model outputs more logits than there are actions. Truncate them.
                    action_logits_truncated = action_logits[:, :, :env.n_action[agent.idx]]

                    # Mask prohibited actions based on active instruction
                    macro_names = env.env.macroActionName if hasattr(env.env, 'macroActionName') else []
                    prohibited = get_prohibited_actions(current_instruction, macro_names)
                    if prohibited:
                        for p_idx in prohibited:
                            if p_idx < action_logits_truncated.shape[-1]:
                                action_logits_truncated[:, :, p_idx] = -float('inf')

                    action_prob = Categorical(logits=action_logits_truncated[0])
                    action = action_prob.sample().item()
                    actions.append(action)
                    new_h_states.append(new_h_state)
                    
                    if force_resample:
                        # Get the macro-action name from the environment
                        action_name = env.env.macroActionName[action] if hasattr(env.env, 'macroActionName') else str(action)
                        print(f"  Agent {idx} received instruction and resampled -> {action_name}")
                    
    return actions, new_h_states
 
 
def get_init_inputs(env, n_agent):
    reset_result = env.reset()
    # Handle case where reset returns a single array vs list of arrays
    if isinstance(reset_result, list):
        return [torch.from_numpy(i).float() for i in reset_result], [None]*n_agent
    else:
        # If reset returns a single array, split it for multiple agents
        return [torch.tensor(reset_result).float() for _ in range(n_agent)], [None]*n_agent


def test(env_id, grid_dim, mapType, task, n_agent, p_id, human_agent_idx=None, render_delay=0.2):

    TASKLIST = ["tomato salad",
                "lettuce salad",
                "onion salad",
                "lettuce-tomato salad",
                "onion-tomato salad",
                "lettuce-onion salad",
                "lettuce-onion-tomato salad",
                "peas salad",
                "lettuce-peas salad",
                "lettuce-peas-tomato-patty"]
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
        print(f"  8/D     : Deliver")
        print(f"  9/C     : Chop")
        if mapType != "A":
            print(f"  X       : Go to Counter")
        print(f"  Arrows  : Move (up/down/left/right)")
        print(f"\n{'='*60}")
        print(f"SPECIAL:")
        print(f"{'='*60}")
        print(f"  T : Enter instruction mode (game pauses)")
        print(f"{'='*60}")
        print(f"Close Window to Quit")
        print(f"{'='*60}\n")

    # For mac_iac, we load SEPARATE actor policies for each agent.
    # Policy path is resolved from the hardcoded base path above.
    policy_dir = resolve_policy_dir(mapType)
    print(f"Using hardcoded policy base: {POLICY_BASE_DIR}")
    if POLICY_RUN_DIR:
        print(f"Using configured run folder: {POLICY_RUN_DIR}")
    
    # Determine if instructions should be used
    use_instructions_flag = True
    print(f"Loading {n_agent} separate IAC policies from: {policy_dir}")

    # Check what policy files actually exist in the directory
    available_policies = []
    selected_run_id = None

    # Prefer the requested run_id, then fall back to seeds 1-5 if needed.
    run_id_candidates = []
    for run_id in [p_id, 1, 2, 3, 4, 5]:
        if run_id not in run_id_candidates:
            run_id_candidates.append(run_id)

    for run_id in run_id_candidates:
        candidate_paths = []
        for i in range(n_agent):
            policy_path = os.path.join(policy_dir, f"{run_id}_agent_{i}.pt")
            if not os.path.exists(policy_path):
                candidate_paths = []
                break
            candidate_paths.append((i, policy_path))
        if candidate_paths:
            available_policies = candidate_paths
            selected_run_id = run_id
            break

    if selected_run_id is not None and selected_run_id != p_id:
        print(
            f"Requested p_id={p_id} not found for all agents; "
            f"using run_id={selected_run_id} instead."
        )

    if selected_run_id is None:
        for i in range(n_agent):
            # Try both naming patterns: stochastic_policy_agent_{idx}.pt and fixed_policy_agent_{idx}.pt
            policy_path2 = os.path.join(policy_dir, f"stochastic_policy_agent_{i}.pt")
            policy_path3 = os.path.join(policy_dir, f"fixed_policy_agent_{i}.pt")

            if os.path.exists(policy_path2):
                available_policies.append((i, policy_path2))
            elif os.path.exists(policy_path3):
                available_policies.append((i, policy_path3))
    
    if len(available_policies) == 0:
        print(f"\nERROR: No policy files found in: {policy_dir}")
        print(f"\nLooked for files matching:")
        print("  - {run_id}_agent_{0,1,2}.pt (run_id in [p_id,1,2,3,4,5])")
        print(f"  - stochastic_policy_agent_{{0,1,2}}.pt")
        print(f"  - fixed_policy_agent_{{0,1,2}}.pt")
        print(f"\nAvailable files in directory:")
        if os.path.exists(policy_dir):
            for f in os.listdir(policy_dir):
                if f.endswith('.pt'):
                    print(f"  - {f}")
        else:
            print(f"  Directory does not exist!")
        sys.exit(1)
    
    print(f"\nFound {len(available_policies)} agent policies:")
    for agent_idx, path in available_policies:
        print(f"  Agent {agent_idx}: {os.path.basename(path)}")
    
    # Adjust n_agent if we found fewer policies than expected
    if len(available_policies) < n_agent:
        print(f"\nWARNING: Found only {len(available_policies)} policies, but n_agent={n_agent}")
        print(f"Adjusting n_agent to {len(available_policies)}")
        n_agent = len(available_policies)

    agents = []
    for i in range(n_agent):
        agent = Agent()
        agent.idx = i
        
        # Skip loading policy for human-controlled agent
        if human_agent_idx is not None and i == human_agent_idx:
            agent.expected_input_dim = None  # Not needed for human agent
            agents.append(agent)
            continue
        
        # Find the policy path for this agent
        policy_path = None
        for agent_idx, path in available_policies:
            if agent_idx == i:
                policy_path = path
                break
        
        if policy_path is None:
            print(f"ERROR: Policy file not found for agent {i}")
            sys.exit(1)
        
        # Load the policy - handle both full model and state_dict formats
        # weights_only=False is needed because we may load full model objects
        loaded_data = torch.load(policy_path, map_location=torch.device('cpu'), weights_only=False)
        
        # Check if it's already an Actor model object or a state dict
        if isinstance(loaded_data, Actor):
            # Loaded the entire model object - use it directly
            actor_net = loaded_data
            # Detect input dimension from the loaded model
            # The fc1 layer's input features tells us what dimension the model expects
            model_input_dim = actor_net.fc1.in_features
            fusion_type = actor_net.instruction_fusion if hasattr(actor_net, 'instruction_fusion') else 'unknown'
            use_instr = actor_net.use_instructions if hasattr(actor_net, 'use_instructions') else 'unknown'
            print(f"Agent {i}: Loaded full model object (fc1.in_features={model_input_dim}, use_instructions={use_instr}, fusion={fusion_type})")
        else:
            # Loaded a state dict - need to create model and load weights
            # Use environment observation size as input dimension
            model_input_dim = env.obs_size[i]
            
            actor_net = Actor(
                input_dim=model_input_dim,
                output_dim=169,  # Adjust based on your environment
                use_instructions=use_instructions_flag,
                instruction_fusion='attention'  # MUST match training config in mac_iac.py
            )
            
            # Handle nested state_dict (e.g., from checkpoint)
            if isinstance(loaded_data, dict) and 'actor_net_state_dict' in loaded_data:
                state_dict = loaded_data['actor_net_state_dict']
            else:
                state_dict = loaded_data
            
            actor_net.load_state_dict(state_dict)
            print(f"Agent {i}: Loaded state dict (input_dim={model_input_dim})")
        
        actor_net.eval()
        
        # Assign this agent's policy
        agent.policy_net = actor_net
        
        # Set the expected input dimension based on what the model actually expects
        # IMPORTANT: For concat fusion with instructions enabled, fc1.in_features includes
        # the instruction embedding dim, so we subtract it. Vanilla models
        # (use_instructions=False) skip the concat in forward(), so fc1.in_features == obs_dim.
        uses_instr = getattr(actor_net, 'use_instructions', False)
        if uses_instr and hasattr(actor_net, 'instruction_fusion') and actor_net.instruction_fusion == 'concat':
            # For concat: fc1.in_features = obs_dim + instruction_dim
            if hasattr(actor_net, 'instruction_dim'):
                instr_dim = actor_net.instruction_dim
            elif hasattr(actor_net, 'n_instructions') and actor_net.n_instructions:
                instr_dim = actor_net.n_instructions
            elif hasattr(actor_net, 'instruction_projection'):
                instr_dim = actor_net.instruction_projection.out_features
            else:
                instr_dim = 32
            agent.expected_input_dim = model_input_dim - instr_dim
            print(
                f"Agent {i}: Loaded individual policy (CONCAT fusion). Env obs_size={env.obs_size[i]}, "
                f"Model fc1.in_features={model_input_dim}, instruction_dim={instr_dim}, "
                f"Obs padding target={agent.expected_input_dim}"
            )
        else:
            agent.expected_input_dim = model_input_dim
            print(f"Agent {i}: Loaded individual policy. Env obs_size={env.obs_size[i]}, Model expects={agent.expected_input_dim}")
        
        agents.append(agent)

    R = 0
    discount = 0.99
    step = 0.0
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
        
        if PYGAME_AVAILABLE and pygame is not None and pygame.display.get_init():
            # Initial render with info panel
            env.render()
            draw_reward_panel(env, step, R, last_reward, human_agent_idx)
            pygame.display.flip()

        last_valid = [1.0] * n_agent
        while not t:
            # Handle pause/instruction input events first
            handle_instruction_input_events()

            # If paused, skip game step but continue rendering
            if not paused:
                # Instructions now persist forever until explicitly cleared
                # (no automatic expiration)
                
                # Check if instruction changed - if so, force agents to resample
                force_resample = instruction_changed
                if instruction_changed:
                    reset_macro_actions(env)
                    instruction_changed = False  # Reset the flag
                
                a, h_states = get_actions_and_h_states(env, agents, last_valid, last_obs, h_states, human_agent_idx, mapType, force_resample=force_resample)
                
                # Step the environment
                step_result = env.step(a)
                last_obs, r, t, info = step_result

                # Update rewards
                last_reward = r[0]
                R += discount**step*last_reward
                step += 1.0
                
                last_valid = info['mac_done']

            # Only render if not paused, or render at reduced rate when paused
            if PYGAME_AVAILABLE and pygame is not None and pygame.display.get_init():
                if not paused:
                    # Get game surface from env without auto-flipping
                    game_surface = env.render()
                    
                    # Get the main screen
                    screen = pygame.display.get_surface()
                    if screen and game_surface:
                        # Blit game surface to top portion of screen
                        screen.blit(game_surface, (0, 0))
                        
                        # Draw reward panel below
                        draw_reward_panel(env, step, R, last_reward, human_agent_idx)
                        
                        # Draw instruction input overlay if active (currently unused)
                        if instruction_input_active:
                            draw_instruction_input_overlay(screen)

                        # Single display update for the entire screen
                        pygame.display.flip()

                    # Add a small delay each step to slow down execution so human control is playable
                    time.sleep(render_delay)
                else:
                    # When paused, just update the pause status without re-rendering the game
                    screen = pygame.display.get_surface()
                    if screen:
                        # Just update the reward panel to show pause status
                        draw_reward_panel(env, step, R, last_reward, human_agent_idx)
                        pygame.display.flip()
                    # Longer delay when paused to reduce CPU usage
                    time.sleep(0.1)

            # Convert observations to tensors if they're not already (only when not paused)
            if not paused:
                if last_obs and isinstance(last_obs[0], (int, float)):
                    # If observations are scalars, convert them
                    last_obs = [torch.tensor(float(o)) for o in last_obs]
                elif last_obs and hasattr(last_obs[0], 'shape'):
                    # If observations are numpy arrays, convert them
                    last_obs = [torch.from_numpy(o).float() for o in last_obs]
                # If already tensors, keep them as is
        
        print(f"\n{'='*60}")
        print(f"Episode Complete!")
        print(f"{'='*60}")
        print(f"Final Total Reward: {R:.2f}")
        print(f"Total Steps: {int(step)}")
        print(f"{'='*60}")
        print(f"\nAnalysis:")
        print(f"  Expected value of 1 delivery: +200")
        print(f"  Step penalty accumulated: {-0.1 * step:.1f} over {int(step)} steps")
        print(f"  Discounted return above already includes step penalties and deliveries")
        print(f"{'='*60}\n")
    
    # Close environment and pygame after all episodes complete
    env.close()
    if PYGAME_AVAILABLE and pygame is not None and pygame.display.get_init():
        pygame.quit()
        time.sleep(0.5)  # Give pygame time to close the window
    print("Environment closed.")
    sys.exit(0)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', action='store', type=str, default='Overcooked-MA-v1')
    parser.add_argument('--grid_dim', action='store', type=int, nargs=2, default=[7,7], choices=[[7, 7], [9, 9]])
    parser.add_argument('--mapType', action='store', type=str, default="D", choices=["A", "B", "C", "D", "E", "F"])
    parser.add_argument('--task', action='store', type=int, default=9,
                        help="Task index: 6=lettuce-onion-tomato salad, 9=lettuce-peas-tomato-patty (Map D)")
    parser.add_argument('--n_agent', action='store', type=int, default=2)
    parser.add_argument('--p_id', action='store', type=int, default=0, help="The specific policy_id")
    parser.add_argument('--human_agent_idx', action='store', type=int, default=None,
                        help="Index of human-controlled agent (0 or 1 for 2-agent, 0-2 for 3-agent). If None, all agents use trained policies.")
    parser.add_argument('--render_delay', action='store', type=float, default=0.1,
                        help="Delay between frames in seconds (default: 0.15). Lower = faster, 0.001 = max speed.")

    test(**vars(parser.parse_args()))

if __name__ == '__main__':
    main()

