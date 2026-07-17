import argparse
import gym
import numpy as np
import torch
import os
import sys
sys.path.append("..")
import time
import imageio

from macro_marl.my_env.box_pushing_MA import BoxPushing_harder as BP_MA
from macro_marl.cores.pg_based.mac_cac.models import Actor

ACTIONS = ["GT_SB0", "GT_SB1", "GT_BB0", "GT_BB1", "PUSH", "T_L", "T_R", "STAY"]

def get_actions(env, actor_net, last_valid, joint_obs, h_state, instruction_emb=None, checkpoint_input_dim=None, checkpoint_output_dim=None):
    """
    Get actions for all agents using centralized actor network.

    Args:
        env: Environment
        actor_net: Centralized actor network
        last_valid: List of boolean flags indicating if each agent's action is done
        joint_obs: List of observations for each agent
        h_state: Hidden state for the centralized network
        instruction_emb: Optional instruction embedding
        checkpoint_input_dim: Expected base observation input dimension from checkpoint
        checkpoint_output_dim: Expected output dimension from checkpoint

    Returns:
        actions: List of actions for each agent
        h_state: Updated hidden state
    """
    with torch.no_grad():
        # Check if all agents need new actions (centralized controller)
        if max(last_valid) == 1:
            # Concatenate all agent observations - controller expects list of tensors
            obs_list = [o.view(1, -1) for o in joint_obs]  # List of [1, obs_size] tensors
            
            # Use instruction embedding if provided
            if instruction_emb is not None:
                inst_emb = instruction_emb
            else:
                inst_emb = None
            
            # Get action from centralized actor - it will concatenate observations internally
            Q, h_state = actor_net(torch.cat(obs_list, dim=1).view(1,1,-1), h_state, eps=0.0, test_mode=True, instruction_emb=inst_emb)

            # Q has shape [1, 1, n_joint_actions] where n_joint_actions = product of all agent action spaces
            Q_values = Q.squeeze(1)  # Shape: [1, n_joint_actions]
            
            # Get best joint action
            joint_action_idx = Q_values.max(1)[1].item()
            
            # Convert joint action index to individual agent actions
            actions = list(np.unravel_index(joint_action_idx, env.n_action))
            
            for idx, a in enumerate(actions):
                print(f"Agent {idx}: Action {a} ({ACTIONS[a]})")
        else:
            # Some agents still executing their macro-actions, keep previous actions
            actions = [env.agents[idx].cur_action.idx for idx in range(env.n_agent)]
    
    return actions, h_state

def get_init_inputs(env, n_agent):
    """Initialize observations and hidden states."""
    obs = [torch.from_numpy(o).float() for o in env.reset()]
    h_state = None
    return obs, h_state

def test(policy_path, env_id='BP-MA-v0', env_terminate_step=100, grid_dim=[6, 6], 
         n_agent=2, n_episode=5, use_instruction=False, instruction_text="don't go to any small box",
         save_video=True, video_dir='./videos'):
    """
    Test a trained mac_cac policy on Box Pushing environment.
    
    Args:
        policy_path: Path to the saved policy (.pt file)
        env_id: Environment ID
        env_terminate_step: Maximum steps per episode
        grid_dim: Grid dimensions [height, width]
        n_agent: Number of agents
        n_episode: Number of episodes to run
        use_instruction: Whether to use instruction embedding
        instruction_text: Instruction text (if use_instruction=True)
        save_video: Whether to save episodes as videos
        video_dir: Directory to save videos
    """
    # Create video directory if saving videos
    if save_video:
        os.makedirs(video_dir, exist_ok=True)
        print(f"Videos will be saved to: {video_dir}")
    
    env_params = {
        'grid_dim': tuple(grid_dim),
        'n_agent': n_agent,
        'penalty': -5,
        'big_box_reward': 300,
        'small_box_reward': 10,
        'terminate_step': env_terminate_step,
        'random_init': False,
        'render': False,  # Enable rendering for rgb_array capture
    }

    env = gym.make(env_id, **env_params)
    
    print("Capturing episodes as video (no window will open)")
    
    # Load centralized actor network
    obs_size = env.obs_size[0]  # Assuming all agents have same obs size
    n_action = env.n_action[0]   # Assuming all agents have same action space

    print(f"Environment dimensions:")
    print(f"  Obs size: {obs_size}")
    print(f"  N actions: {n_action}")

    # Load checkpoint first to get the actual dims
    # If policy_path is relative, make it absolute from current working directory
    if not os.path.isabs(policy_path):
        abs_policy_path = os.path.join(os.getcwd(), policy_path)
    else:
        abs_policy_path = policy_path

    checkpoint = torch.load(abs_policy_path, map_location='cpu')

    # Infer dimensions from checkpoint weights
    fc1_weight_shape = checkpoint['fc1.weight'].shape  # [out_features, in_features]
    fc4_weight_shape = checkpoint['fc4.weight'].shape  # [out_features, in_features]
    mlp_dim = fc1_weight_shape[0]
    fc1_input_dim = fc1_weight_shape[1]  # This is the ACTUAL input to fc1
    output_dim = fc4_weight_shape[0]
    rnn_dim = checkpoint['gru.weight_ih_l0'].shape[0] // 3  # GRU has 3 gates

    # Check if checkpoint has instruction parameters
    has_instruction_params = any('instruction' in key for key in checkpoint.keys())
    
    # Detect which attention implementation the checkpoint uses
    has_cross_attention = 'cross_attention.in_proj_weight' in checkpoint
    has_separate_attention = 'attention_query.weight' in checkpoint
    
    # Determine instruction fusion type from checkpoint
    if has_instruction_params:
        if has_cross_attention or has_separate_attention:
            instruction_fusion = 'attention'
        elif 'film_gamma.weight' in checkpoint:
            instruction_fusion = 'film'
        else:
            instruction_fusion = 'concat'
    else:
        instruction_fusion = None
    
    # Calculate the base observation input dimension
    # If instructions with concat fusion, fc1_input = obs_input + rnn_dim
    # Otherwise, fc1_input = obs_input
    if has_instruction_params and instruction_fusion == 'concat':
        obs_input_dim = fc1_input_dim - rnn_dim
    else:
        obs_input_dim = fc1_input_dim

    print(f"Detected model dimensions:")
    print(f"  FC1 input dim: {fc1_input_dim}")
    print(f"  Obs input dim: {obs_input_dim}")
    print(f"  Output dim: {output_dim}")
    print(f"  MLP layer size: {mlp_dim}")
    print(f"  RNN layer size: {rnn_dim}")
    print(f"  Has instructions: {has_instruction_params}")
    print(f"  Instruction fusion: {instruction_fusion}")

    # Create model with checkpoint dimensions - use obs_input_dim as input_dim
    actor_net = Actor(
        input_dim=obs_input_dim,  # Use base observation input dimension
        output_dim=output_dim,
        mlp_layer_size=[mlp_dim, mlp_dim],
        rnn_layer_size=rnn_dim,
        use_instructions=has_instruction_params,
        instruction_fusion=instruction_fusion if has_instruction_params else 'concat',
        freeze_bert=True
    )
    actor_net.rnn_layer_size = rnn_dim  # Store for later access

    # Convert old cross_attention checkpoint to new separate attention layers if needed
    if has_cross_attention and not has_separate_attention:
        print("Converting old cross_attention checkpoint format to new format...")
        # Extract the combined in_proj weights and biases
        in_proj_weight = checkpoint['cross_attention.in_proj_weight']  # Shape: [3*embed_dim, embed_dim]
        in_proj_bias = checkpoint['cross_attention.in_proj_bias']      # Shape: [3*embed_dim]
        
        embed_dim = mlp_dim
        # Split into Q, K, V
        query_weight, key_weight, value_weight = in_proj_weight.chunk(3, dim=0)
        query_bias, key_bias, value_bias = in_proj_bias.chunk(3, dim=0)
        
        # Map to new layer names
        checkpoint['attention_query.weight'] = query_weight
        checkpoint['attention_query.bias'] = query_bias
        checkpoint['attention_key.weight'] = key_weight
        checkpoint['attention_key.bias'] = key_bias
        checkpoint['attention_value.weight'] = value_weight
        checkpoint['attention_value.bias'] = value_bias
        
        # Remove old keys
        del checkpoint['cross_attention.in_proj_weight']
        del checkpoint['cross_attention.in_proj_bias']
        del checkpoint['cross_attention.out_proj.weight']
        del checkpoint['cross_attention.out_proj.bias']

    # Load trained weights
    print(f"Loading policy from: {abs_policy_path}")
    actor_net.load_state_dict(checkpoint, strict=False)  # Use strict=False to handle missing keys gracefully
    actor_net.eval()
    
    # Store checkpoint dimensions for later use
    checkpoint_input_dim = obs_input_dim
    checkpoint_output_dim = output_dim
    
    # Encode instruction if provided and model supports it
    instruction_emb = None
    if use_instruction and instruction_text and has_instruction_params:
        print(f"Using instruction: '{instruction_text}'")
        instruction_emb = actor_net.encode_instruction(instruction_text).detach()
    elif use_instruction and instruction_text:
        print(f"Instruction provided but model doesn't support instructions: '{instruction_text}'")
    elif has_instruction_params:
        print("Model supports instructions but none provided")
    
    total_reward = 0
    successful_episodes = 0
    
    print(f"\n{'='*60}")
    print(f"Testing policy for {n_episode} episodes")
    print(f"Environment: {env_id}, Grid: {grid_dim}, Agents: {n_agent}")
    print(f"{'='*60}\n")
    
    for e in range(n_episode):
        episode_reward = 0
        step_count = 0
        last_obs, h_state = get_init_inputs(env, n_agent)
        last_valid = [1] * n_agent
        terminated = False
        
        # Initialize video writer for this episode
        frames = []
        if save_video:
            # Capture initial frame
            frame = env.render(mode='rgb_array')
            frames.append(frame)
        
        print(f"\n--- Episode {e+1}/{n_episode} ---")
        
        while not terminated and step_count < env_terminate_step:
            # Get actions from centralized actor
            actions, h_state = get_actions(env, actor_net, last_valid, last_obs, h_state, instruction_emb, checkpoint_input_dim, checkpoint_output_dim)
            
            # Execute actions in environment
            last_obs, rewards, terminated, info = env.step(actions)
            
            # Capture frame for video
            if save_video:
                frame = env.render(mode='rgb_array')
                frames.append(frame)
            
            last_obs = [torch.from_numpy(o).float() for o in last_obs]
            last_valid = info['mac_done']
            
            episode_reward += rewards[0]
            step_count += 1
            
            if terminated:
                print(f"\nEpisode {e+1} finished in {step_count} steps!")
                print(f"Episode reward: {episode_reward:.2f}")
                if episode_reward > 200:  # Consider successful if got big box reward
                    successful_episodes += 1
                    print("✓ SUCCESS - Big box pushed!")
        
        if not terminated:
            print(f"\nEpisode {e+1} timeout after {step_count} steps")
            print(f"Episode reward: {episode_reward:.2f}")
        
        # Save video for this episode
        if save_video and len(frames) > 0:
            video_path = os.path.join(video_dir, f'episode_{e+1}_reward_{episode_reward:.1f}.mp4')
            print(f"Saving video to: {video_path}")
            imageio.mimsave(video_path, frames, fps=5)  # 5 fps for easy viewing
            print(f"Video saved ({len(frames)} frames)")
        
        total_reward += episode_reward
    
    avg_reward = total_reward / n_episode
    success_rate = successful_episodes / n_episode * 100
    
    print(f"\n{'='*60}")
    print(f"Test Results:")
    print(f"  Average Reward: {avg_reward:.2f}")
    print(f"  Success Rate: {success_rate:.1f}% ({successful_episodes}/{n_episode})")
    print(f"{'='*60}\n")

def main():
    parser = argparse.ArgumentParser(description='Test trained mac_cac policy on Box Pushing')
    parser.add_argument('--policy_path', type=str, required=True,
                        help='Path to saved policy file (.pt)')
    parser.add_argument('--env_id', type=str, default='BP-MA-v0',
                        help='Environment ID')
    parser.add_argument('--env_terminate_step', type=int, default=100,
                        help='Maximum steps per episode')
    parser.add_argument('--grid_dim', type=int, nargs=2, default=[6, 6],
                        help='Grid dimensions [height width]')
    parser.add_argument('--n_agent', type=int, default=2,
                        help='Number of agents')
    parser.add_argument('--n_episode', type=int, default=5,
                        help='Number of test episodes')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of test episodes (alias for n_episode)')
    parser.add_argument('--use_instruction', action='store_true',
                        help='Use instruction embedding')
    parser.add_argument('--no_instruction', action='store_true',
                        help='Disable instruction embedding (overrides use_instruction)')
    parser.add_argument('--instruction', type=str, default="don't go to any small box",
                        help='Instruction text (if use_instruction=True)')
    parser.add_argument('--save_video', action='store_true', default=True,
                        help='Save episodes as MP4 videos (default: True)')
    parser.add_argument('--no_video', action='store_true',
                        help='Disable video saving')
    parser.add_argument('--video_dir', type=str, default='/home/willy/macro_marl_ppo/visualization/policy_nns/BP_MA/6x6/videos',
                        help='Directory to save videos (default: ./policy_nns/BP_MA/6x6/videos)')
    
    args = parser.parse_args()

    # Use either n_episode or episodes argument
    n_episode = getattr(args, 'episodes', args.n_episode)

    test(
        policy_path=args.policy_path,
        env_id=args.env_id,
        env_terminate_step=args.env_terminate_step,
        grid_dim=args.grid_dim,
        n_agent=args.n_agent,
        n_episode=n_episode,
        use_instruction=args.use_instruction and not args.no_instruction,
        instruction_text=args.instruction,
        save_video=args.save_video and not args.no_video,
        video_dir=args.video_dir
    )

if __name__ == '__main__':
    main()

