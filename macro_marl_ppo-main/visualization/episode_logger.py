import pickle
import numpy as np
import torch
from datetime import datetime
import os

class EpisodeLogger:
    """
    Simple logger for storing episodes during gameplay.
    Stores: observations, actions, rewards, next_obs, done, info
    """
    
    def __init__(self, save_dir="./logged_episodes"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Current episode buffer
        self.current_episode = []
        
        # All completed episodes
        self.episodes = []
        
        # Episode metadata
        self.episode_returns = []
        self.episode_lengths = []
        
    def log_step(self, obs, actions, rewards, next_obs, done, info, h_states=None):
        """
        Log a single step/transition.
        
        Parameters
        ----------
        obs : List[np.ndarray] or List[torch.Tensor]
            Observations for each agent before action
        actions : List[int]
            Actions taken by each agent
        rewards : List[float]
            Rewards received by each agent
        next_obs : List[np.ndarray] or List[torch.Tensor]
            Observations after action
        done : bool
            Whether episode terminated
        info : dict
            Additional info (mac_done, human_messages, etc.)
        h_states : List (optional)
            Hidden states for RNN policies
        """
        
        # Convert to numpy if torch tensors
        def to_numpy(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            elif isinstance(x, list):
                return [to_numpy(item) for item in x]
            return x
        
        transition = {
            'obs': to_numpy(obs),
            'actions': actions,
            'rewards': rewards,
            'next_obs': to_numpy(next_obs),
            'done': done,
            'info': info.copy() if isinstance(info, dict) else info,
            'h_states': h_states,
            'timestamp': datetime.now().isoformat()
        }
        
        self.current_episode.append(transition)
    
    def end_episode(self):
        """
        Mark current episode as complete and compute statistics.
        """
        if len(self.current_episode) == 0:
            return
        
        # Compute episode return (sum of all agent rewards across timesteps)
        episode_return = sum([sum(t['rewards']) for t in self.current_episode])
        episode_length = len(self.current_episode)
        
        self.episode_returns.append(episode_return)
        self.episode_lengths.append(episode_length)
        
        # Store episode with metadata
        episode_data = {
            'transitions': self.current_episode,
            'return': episode_return,
            'length': episode_length,
            'timestamp': datetime.now().isoformat()
        }
        
        self.episodes.append(episode_data)
        
        # Reset current episode
        self.current_episode = []
        
        print(f"Episode completed: Return={episode_return:.2f}, Length={episode_length}")
        
    def save(self, filename=None):
        """
        Save all logged episodes to disk.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"episodes_{timestamp}.pkl"
        
        filepath = os.path.join(self.save_dir, filename)
        
        data = {
            'episodes': self.episodes,
            'episode_returns': self.episode_returns,
            'episode_lengths': self.episode_lengths,
            'num_episodes': len(self.episodes),
            'mean_return': np.mean(self.episode_returns) if self.episode_returns else 0,
            'mean_length': np.mean(self.episode_lengths) if self.episode_lengths else 0
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Saved {len(self.episodes)} episodes to {filepath}")
        print(f"  Mean return: {data['mean_return']:.2f}")
        print(f"  Mean length: {data['mean_length']:.2f}")
        
        return filepath
    
    def load(self, filepath):
        """
        Load previously saved episodes.
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.episodes = data['episodes']
        self.episode_returns = data['episode_returns']
        self.episode_lengths = data['episode_lengths']
        
        print(f"Loaded {len(self.episodes)} episodes from {filepath}")
        return data
    
    def get_statistics(self):
        """
        Get summary statistics of logged episodes.
        """
        if not self.episode_returns:
            return {}
        
        return {
            'num_episodes': len(self.episodes),
            'mean_return': np.mean(self.episode_returns),
            'std_return': np.std(self.episode_returns),
            'min_return': np.min(self.episode_returns),
            'max_return': np.max(self.episode_returns),
            'mean_length': np.mean(self.episode_lengths),
            'total_steps': sum(self.episode_lengths)
        }
    
    def clear(self):
        """
        Clear all logged data.
        """
        self.current_episode = []
        self.episodes = []
        self.episode_returns = []
        self.episode_lengths = []

