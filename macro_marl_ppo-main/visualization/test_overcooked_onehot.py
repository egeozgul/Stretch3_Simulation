"""
Test script for Overcooked MAC-IAC policies trained with one-hot instruction encoding.

Instead of BERT embeddings, instructions are represented as one-hot vectors.
The instruction list is defined at the top and must match what the model was trained with.
Press 'T' to cycle through instructions or clear the active instruction.
"""
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

# ============================================================================
# INSTRUCTION DEFINITIONS
# Must match the instructions used during training (same order!)
# ============================================================================
INSTRUCTION_LIST = [
    "don't use the right cutting board",
    "don't use the left cutting board",
    "let me do all the chopping",
]

def make_one_hot(instruction_idx, n_instructions):
    """Create a one-hot tensor for the given instruction index."""
    one_hot = torch.zeros(1, n_instructions)
    if instruction_idx is not None and 0 <= instruction_idx < n_instructions:
        one_hot[0, instruction_idx] = 1.0
    return one_hot

# ============================================================================
# Global state
# ============================================================================
current_instruction_idx = None  # None = no instruction active
instruction_changed = False
paused = False

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
    """Get human action from keyboard input."""
    if not PYGAME_AVAILABLE or not pygame.display.get_init():
        return 0

    common_actions = {
        pygame.K_0: 0,   # stay
        pygame.K_SPACE: 0,
        pygame.K_1: 1,   # get tomato
        pygame.K_2: 2,   # get lettuce
        pygame.K_3: 3,   # get onion
        pygame.K_4: 4,   # get peas (Map D)
        pygame.K_5: 5,   # get plate 1
        pygame.K_6: 6,   # get plate 2
        pygame.K_7: 7,   # go to knife 1
        pygame.K_8: 8,   # go to knife 2
        pygame.K_d: 9,   # deliver
        pygame.K_9: 9,   # deliver
        pygame.K_c: 10,  # chop
    }

    if mapType == "A":
        movement_actions = {
            pygame.K_UP: env.macroActionName.index("up"),
            pygame.K_DOWN: env.macroActionName.index("down"),
            pygame.K_LEFT: env.macroActionName.index("left"),
            pygame.K_RIGHT: env.macroActionName.index("right"),
        }
    elif mapType == "D":
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
    else:
        movement_actions = {
            pygame.K_x: env.macroActionName.index("go to counter"),
            pygame.K_UP: env.macroActionName.index("up"),
            pygame.K_DOWN: env.macroActionName.index("down"),
            pygame.K_LEFT: env.macroActionName.index("left"),
            pygame.K_RIGHT: env.macroActionName.index("right"),
        }

    all_actions = {**common_actions, **movement_actions}
    keys = pygame.key.get_pressed()
    for key, action in all_actions.items():
        if keys[key]:
            return action
    return 0


def handle_events():
    """Handle pygame events. Press T to cycle instructions, ESC to quit."""
    global current_instruction_idx, instruction_changed, paused

    if not PYGAME_AVAILABLE or not pygame.display.get_init():
        return

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_t:
                # Cycle: None -> 0 -> 1 -> ... -> N-1 -> None
                n = len(INSTRUCTION_LIST)
                if current_instruction_idx is None:
                    current_instruction_idx = 0
                elif current_instruction_idx < n - 1:
                    current_instruction_idx += 1
                else:
                    current_instruction_idx = None

                instruction_changed = True
                if current_instruction_idx is not None:
                    print(f"\nInstruction {current_instruction_idx}: "
                          f"'{INSTRUCTION_LIST[current_instruction_idx]}'")
                else:
                    print(f"\nInstruction CLEARED (no instruction active)")


def setup_extended_display(env, human_agent_idx=None):
    """Setup pygame display with extended height for info panel."""
    global _display_state

    if not PYGAME_AVAILABLE or not pygame.display.get_init():
        return None

    game_surface = pygame.display.get_surface()
    if game_surface is None:
        return None

    game_width, game_height = game_surface.get_size()
    info_height = 180
    total_height = game_height + info_height

    screen = pygame.display.set_mode((game_width, total_height))
    pygame.display.set_caption("Overcooked MAC-IAC One-Hot Test")

    pygame.font.init()
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)

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
    """Draw the reward information panel below the game area."""
    global _display_state, current_instruction_idx, paused

    if not PYGAME_AVAILABLE or not pygame.display.get_init() or not _display_state['initialized']:
        return

    try:
        screen = _display_state['screen']
        game_width = _display_state['game_width']
        game_height = _display_state['game_height']
        info_height = _display_state['info_height']
        font = _display_state['font']
        small_font = _display_state['small_font']

        panel = pygame.Surface((game_width, info_height))
        panel.fill((40, 40, 40))
        pygame.draw.rect(panel, (100, 100, 100), panel.get_rect(), 2)

        y = 10

        # Step & reward
        step_text = font.render(f'Step: {int(step)}', True, (255, 255, 255))
        panel.blit(step_text, (20, y))

        reward_color = (100, 255, 100) if total_reward >= 0 else (255, 100, 100)
        total_text = font.render(f'Total: {total_reward:.2f}', True, reward_color)
        panel.blit(total_text, (200, y))

        last_color = (150, 255, 150) if last_reward >= 0 else (255, 150, 150)
        last_text = small_font.render(f'Last: {last_reward:.2f}', True, last_color)
        panel.blit(last_text, (430, y + 5))

        y += 40

        if human_agent_idx is not None:
            mode_text = small_font.render(f'Human: Agent {human_agent_idx}', True, (255, 255, 100))
            panel.blit(mode_text, (20, y))

        controls_text = small_font.render('T = cycle instruction', True, (180, 180, 180))
        panel.blit(controls_text, (250, y))
        y += 28

        # Current instruction
        if current_instruction_idx is not None:
            n = len(INSTRUCTION_LIST)
            one_hot_str = "[" + ", ".join(
                "1" if i == current_instruction_idx else "0" for i in range(n)
            ) + "]"
            inst_text = small_font.render(
                f'Instruction {current_instruction_idx}: "{INSTRUCTION_LIST[current_instruction_idx]}"',
                True, (100, 255, 100))
            panel.blit(inst_text, (20, y))
            y += 22
            oh_text = small_font.render(f'One-hot: {one_hot_str}', True, (180, 255, 180))
            panel.blit(oh_text, (20, y))
        else:
            no_inst = small_font.render('No instruction active (one-hot = all zeros)', True, (150, 150, 150))
            panel.blit(no_inst, (20, y))
            y += 22

        y += 22
        if paused:
            pause_text = small_font.render("PAUSED - Press T to resume", True, (255, 200, 50))
            panel.blit(pause_text, (20, y))

        screen.blit(panel, (0, game_height))
    except pygame.error:
        pass


def get_actions_and_h_states(env, agents, last_valid, obs_list, h_states_list,
                              n_instructions, human_agent_idx=None, mapType="A",
                              force_resample=False):
    """Get actions for all agents, passing one-hot instruction to AI agents."""
    global current_instruction_idx

    with torch.no_grad():
        actions = []
        new_h_states = []

        # Build one-hot instruction embedding
        inst_emb = make_one_hot(current_instruction_idx, n_instructions)

        for idx, agent in enumerate(agents):
            if human_agent_idx is not None and idx == human_agent_idx:
                actions.append(get_human_action(env, human_agent_idx, mapType))
                new_h_states.append(h_states_list[agent.idx])
            else:
                obs = obs_list[agent.idx]

                # Pad/truncate observation to match expected dim
                if obs.shape[0] < agent.expected_input_dim:
                    padding = torch.zeros(agent.expected_input_dim - obs.shape[0])
                    obs = torch.cat([obs, padding])
                elif obs.shape[0] > agent.expected_input_dim:
                    obs = obs[:agent.expected_input_dim]

                action_logits, new_h_state = agent.policy_net(
                    obs.view(1, 1, agent.expected_input_dim),
                    h_states_list[agent.idx],
                    instruction_emb=inst_emb
                )

                action_logits_truncated = action_logits[:, :, :env.n_action[agent.idx]]
                action_prob = Categorical(logits=action_logits_truncated[0])
                action = action_prob.sample().item()
                actions.append(action)
                new_h_states.append(new_h_state)

                if force_resample:
                    action_name = (env.env.macroActionName[action]
                                   if hasattr(env.env, 'macroActionName') else str(action))
                    idx_str = (f"'{INSTRUCTION_LIST[current_instruction_idx]}'"
                               if current_instruction_idx is not None else "None")
                    print(f"  Agent {idx} instruction={idx_str} -> {action_name}")

    return actions, new_h_states


def get_init_inputs(env, n_agent):
    reset_result = env.reset()
    if isinstance(reset_result, list):
        return [torch.from_numpy(i).float() for i in reset_result], [None] * n_agent
    else:
        return [torch.tensor(reset_result).float() for _ in range(n_agent)], [None] * n_agent


def test(env_id, grid_dim, mapType, task, n_agent, p_id, human_agent_idx=None,
         render_delay=0.2, instructions=None):

    global INSTRUCTION_LIST

    # Override instruction list if provided via CLI
    if instructions:
        INSTRUCTION_LIST = instructions
    n_instructions = len(INSTRUCTION_LIST)

    TASKLIST = ["tomato salad", "lettuce salad", "onion salad",
                "lettuce-tomato salad", "onion-tomato salad",
                "lettuce-onion salad", "lettuce-onion-tomato salad"]
    rewardList = {"subtask finished": 10, "correct delivery": 200,
                  "wrong delivery": -5, "step penalty": -0.1}
    env_params = {'grid_dim': grid_dim, 'task': TASKLIST[task],
                  'rewardList': rewardList, 'map_type': mapType,
                  'n_agent': n_agent, 'debug': True}

    env = gym.make(env_id, **env_params)
    env = MacEnvWrapper(env)

    print(f"Environment observation sizes: {env.obs_size}")
    print(f"Environment action sizes: {env.n_action}")
    print(f"\n{'='*60}")
    print(f"ONE-HOT INSTRUCTION TEST - Map {mapType}")
    print(f"{'='*60}")
    print(f"Instructions ({n_instructions} total):")
    for i, inst in enumerate(INSTRUCTION_LIST):
        print(f"  [{i}] '{inst}'")
    print(f"\nPress T to cycle through instructions")
    print(f"{'='*60}\n")

    if human_agent_idx is not None:
        print(f"Agent {human_agent_idx} is human-controlled")
        print(f"  Keys: 1-7=Items, C=Chop, D=Deliver, Arrows=Move\n")

    # ----------------------------------------------------------------
    # Load policies
    # ----------------------------------------------------------------
    policy_dir_env = os.environ.get("MAC_IAC_POLICY_DIR", None)
    if policy_dir_env:
        policy_dir = policy_dir_env
    else:
        policy_dir = os.path.join(os.path.dirname(__file__), "..", "experiments",
                                  "Overcooked", "policy_nns",
                                  "mac_iac_overcooked_" + mapType + "_desktop_home_1")
        if not os.path.exists(policy_dir):
            policy_dir = os.path.join(os.path.dirname(__file__), "policy_nns",
                                      "Overcooked", "map" + mapType)

    print(f"Loading {n_agent} IAC policies from: {policy_dir}")

    available_policies = []
    for i in range(n_agent):
        for pattern in [f"{p_id}_agent_{i}.pt", f"stochastic_policy_agent_{i}.pt",
                        f"fixed_policy_agent_{i}.pt"]:
            path = os.path.join(policy_dir, pattern)
            if os.path.exists(path):
                available_policies.append((i, path))
                break

    if not available_policies:
        print(f"\nERROR: No policy files found in: {policy_dir}")
        if os.path.exists(policy_dir):
            for f in sorted(os.listdir(policy_dir)):
                if f.endswith('.pt'):
                    print(f"  - {f}")
        sys.exit(1)

    print(f"Found {len(available_policies)} agent policies:")
    for agent_idx, path in available_policies:
        print(f"  Agent {agent_idx}: {os.path.basename(path)}")

    if len(available_policies) < n_agent:
        print(f"WARNING: Found only {len(available_policies)} policies, adjusting n_agent")
        n_agent = len(available_policies)

    agents = []
    for i in range(n_agent):
        agent = Agent()
        agent.idx = i

        if human_agent_idx is not None and i == human_agent_idx:
            agent.expected_input_dim = None
            agents.append(agent)
            continue

        policy_path = None
        for agent_idx, path in available_policies:
            if agent_idx == i:
                policy_path = path
                break
        if policy_path is None:
            print(f"ERROR: Policy file not found for agent {i}")
            sys.exit(1)

        loaded_data = torch.load(policy_path, map_location='cpu', weights_only=False)

        if isinstance(loaded_data, Actor):
            actor_net = loaded_data
            model_input_dim = actor_net.fc1.in_features
            use_instr = getattr(actor_net, 'use_instructions', False)
            n_instr_model = getattr(actor_net, 'n_instructions', 0)
            fusion = getattr(actor_net, 'instruction_fusion', 'unknown')
            print(f"Agent {i}: Full model (fc1.in={model_input_dim}, "
                  f"use_instructions={use_instr}, n_instructions={n_instr_model}, "
                  f"fusion={fusion})")
        else:
            model_input_dim = env.obs_size[i]
            actor_net = Actor(
                input_dim=model_input_dim,
                output_dim=env.n_action[i],
                use_instructions=True,
                instruction_fusion='concat',
                n_instructions=n_instructions
            )
            if isinstance(loaded_data, dict) and 'actor_net_state_dict' in loaded_data:
                state_dict = loaded_data['actor_net_state_dict']
            else:
                state_dict = loaded_data
            actor_net.load_state_dict(state_dict)
            print(f"Agent {i}: Loaded state dict (input_dim={model_input_dim})")

        actor_net.eval()
        agent.policy_net = actor_net

        # For concat fusion: fc1.in_features = obs_dim + n_instructions
        if (getattr(actor_net, 'instruction_fusion', '') == 'concat'
                and getattr(actor_net, 'use_instructions', False)):
            instr_dim = getattr(actor_net, 'instruction_dim', n_instructions)
            agent.expected_input_dim = model_input_dim - instr_dim
            print(f"  Obs padding target = {agent.expected_input_dim} "
                  f"(fc1.in={model_input_dim} - instruction_dim={instr_dim})")
        else:
            agent.expected_input_dim = model_input_dim
            print(f"  Obs padding target = {agent.expected_input_dim}")

        agents.append(agent)

    # ----------------------------------------------------------------
    # Run episodes
    # ----------------------------------------------------------------
    R = 0
    discount = 0.99
    step = 0.0
    n_episode = 1
    last_reward = 0.0

    for e in range(n_episode):
        global instruction_changed, current_instruction_idx
        t = 0
        last_obs, h_states = get_init_inputs(env, n_agent)

        env.render()
        screen = setup_extended_display(env, human_agent_idx)
        if PYGAME_AVAILABLE and pygame.display.get_init():
            env.render()
            draw_reward_panel(env, step, R, last_reward, human_agent_idx)
            pygame.display.flip()

        last_valid = [1.0] * n_agent
        while not t:
            handle_events()

            if not paused:
                force_resample = instruction_changed
                if instruction_changed:
                    instruction_changed = False

                a, h_states = get_actions_and_h_states(
                    env, agents, last_valid, last_obs, h_states,
                    n_instructions, human_agent_idx, mapType,
                    force_resample=force_resample)

                last_obs, r, t, info = env.step(a)
                last_reward = r[0]
                R += discount ** step * last_reward
                step += 1.0
                last_valid = info['mac_done']

            if PYGAME_AVAILABLE and pygame.display.get_init():
                if not paused:
                    game_surface = env.render()
                    screen = pygame.display.get_surface()
                    if screen and game_surface:
                        screen.blit(game_surface, (0, 0))
                        draw_reward_panel(env, step, R, last_reward, human_agent_idx)
                        pygame.display.flip()
                    time.sleep(render_delay)
                else:
                    screen = pygame.display.get_surface()
                    if screen:
                        draw_reward_panel(env, step, R, last_reward, human_agent_idx)
                        pygame.display.flip()
                    time.sleep(0.1)

            if not paused:
                if last_obs and isinstance(last_obs[0], (int, float)):
                    last_obs = [torch.tensor(float(o)) for o in last_obs]
                elif last_obs and hasattr(last_obs[0], 'shape'):
                    last_obs = [torch.from_numpy(o).float() for o in last_obs]

        print(f"\n{'='*60}")
        print(f"Episode Complete!  Return: {R:.2f}  Steps: {int(step)}")
        print(f"{'='*60}\n")

    env.close()
    if PYGAME_AVAILABLE and pygame.display.get_init():
        pygame.quit()
        time.sleep(0.5)
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="Test Overcooked IAC policies with one-hot instruction encoding")
    parser.add_argument('--env_id', type=str, default='Overcooked-MA-v1')
    parser.add_argument('--grid_dim', type=int, nargs=2, default=[7, 7])
    parser.add_argument('--mapType', type=str, default="A",
                        choices=["A", "B", "C", "D", "E", "F"])
    parser.add_argument('--task', type=int, default=6, choices=[3, 6])
    parser.add_argument('--n_agent', type=int, default=2)
    parser.add_argument('--p_id', type=int, default=0)
    parser.add_argument('--human_agent_idx', type=int, default=None,
                        help="Agent index for human control (None = all AI)")
    parser.add_argument('--render_delay', type=float, default=0.1)
    parser.add_argument('--instructions', type=str, nargs='+', default=None,
                        help="Override instruction list (space-separated, quote each)")

    test(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
