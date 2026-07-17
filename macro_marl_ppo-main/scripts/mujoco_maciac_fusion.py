#!/usr/bin/env python3
"""Standalone script to fuse a Mujoco macro-action simulator with MacIAC.

This script is designed to be a minimal training entrypoint for MacIAC
with instructions disabled. It assumes your Mujoco simulator exposes a
macro-action API similar to the repo's existing macro-action environments:

- env.reset() -> observation
- env.run(macro_actions) -> observation, reward, done, info
- env.get_avail_actions() -> list of per-agent avail-action masks
- env.n_agent, env.obs_size, env.n_action

If your Mujoco simulator uses a different interface, adapt `make_mujoco_env`
or the wrapper below accordingly.
"""

import argparse
import os
import random

import gym
import numpy as np
import torch

# Force WandB into offline mode so training can run without remote logging.
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_DISABLE_SERVICE", "true")

from macro_marl.algs import MacIAC


class MujocoMacEnvWrapper(gym.Wrapper):
    """Wrap a Mujoco macro-action simulator for MacIAC."""

    def __init__(self, env):
        super().__init__(env)

    @property
    def n_agent(self):
        return self.env.n_agent

    @property
    def obs_size(self):
        return self.env.obs_size

    @property
    def n_action(self):
        return self.env.n_action

    @property
    def action_spaces(self):
        return getattr(self.env, 'action_spaces', None)

    def reset(self):
        obs = self.env.reset()
        return obs

    def step(self, macro_actions):
        # MacIAC expects a macro-action list of length n_agent.
        return self.env.run(macro_actions)

    def get_avail_actions(self):
        return self.env.get_avail_actions()

    def action_space_sample(self):
        return self.env.macro_action_sample()


def make_mujoco_env(env_id: str, n_agent: int):
    """Create and wrap your Mujoco simulator environment.

    Replace the body of this function to import and instantiate your own
    Mujoco simulator. The returned object must support the macro-action
    interface described in the module docstring.
    """

    # Example stub for a custom Mujoco environment factory:
    # from my_mujoco_sim import MyMujocoMacroEnv
    # env = MyMujocoMacroEnv(env_id, n_agent=n_agent, ...)
    # return MujocoMacEnvWrapper(env)

    # ------------------------------------------------------------------
    # If your simulator is already gym-registered and provides the same
    # macro-action API, simply use gym.make(env_id):
    env = gym.make(env_id)
    return MujocoMacEnvWrapper(env)


def parse_args():
    parser = argparse.ArgumentParser(description="Train MacIAC on a Mujoco macro-action environment")
    parser.add_argument('--env_id', type=str, required=True, help='Gym env id or custom registered Mujoco env')
    parser.add_argument('--env_terminate_step', type=int, default=300, help='Max steps per episode')
    parser.add_argument('--n_env', type=int, default=1, help='Number of parallel environments')
    parser.add_argument('--n_agent', type=int, default=1, help='Number of agents in the Mujoco environment')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--run_id', type=int, default=0, help='Run id')
    parser.add_argument('--save_dir', type=str, default='mujoco_maciac_vanilla', help='Directory for training outputs')
    parser.add_argument('--resume', action='store_true', help='Resume from existing checkpoint')
    parser.add_argument('--device', type=str, default='cpu', help='Device for training: cpu or cuda')

    parser.add_argument('--total_epi', type=int, default=20000, help='Number of training episodes')
    parser.add_argument('--gamma', type=float, default=0.95, help='Discount factor')
    parser.add_argument('--a_lr', type=float, default=0.0003, help='Actor learning rate')
    parser.add_argument('--c_lr', type=float, default=0.001, help='Critic learning rate')
    parser.add_argument('--c_train_iteration', type=int, default=1, help='Critic update iterations per training step')
    parser.add_argument('--eps_start', type=float, default=1.0, help='Initial epsilon for exploration')
    parser.add_argument('--eps_end', type=float, default=0.01, help='Final epsilon for exploration')
    parser.add_argument('--eps_stable_at', type=int, default=4000, help='Episode at which epsilon reaches final value')
    parser.add_argument('--c_hys_start', type=float, default=1.0, help='Critic hysteresis start')
    parser.add_argument('--c_hys_end', type=float, default=1.0, help='Critic hysteresis end')
    parser.add_argument('--adv_hys_start', type=float, default=1.0, help='Advantage hysteresis start')
    parser.add_argument('--adv_hys_end', type=float, default=1.0, help='Advantage hysteresis end')
    parser.add_argument('--hys_stable_at', type=int, default=4000, help='Hysteresis schedule length')
    parser.add_argument('--critic_hys', action='store_true', help='Use hysteresis on critic updates')
    parser.add_argument('--adv_hys', action='store_true', help='Use hysteresis on advantage estimation')
    parser.add_argument('--etrpy_w_start', type=float, default=0.0, help='Entropy weight start')
    parser.add_argument('--etrpy_w_end', type=float, default=0.0, help='Entropy weight end')
    parser.add_argument('--etrpy_w_stable_at', type=int, default=4000, help='Entropy weight decay period')
    parser.add_argument('--train_freq', type=int, default=2, help='Training frequency (episodes)')
    parser.add_argument('--c_target_update_freq', type=int, default=16, help='Critic target update frequency')
    parser.add_argument('--c_target_soft_update', action='store_true', help='Use soft target updates for critic')
    parser.add_argument('--tau', type=float, default=0.01, help='Soft update rate for target critic')
    parser.add_argument('--n_step_TD', type=int, default=0, help='N-step TD returns')
    parser.add_argument('--TD_lambda', type=float, default=0.0, help='TD(lambda) parameter')
    parser.add_argument('--a_mlp_layer_size', type=int, nargs='+', default=[64, 64], help='Actor MLP layer sizes')
    parser.add_argument('--a_rnn_layer_size', type=int, default=64, help='Actor RNN hidden size')
    parser.add_argument('--c_mlp_layer_size', type=int, nargs='+', default=[64, 64], help='Critic MLP layer sizes')
    parser.add_argument('--c_rnn_layer_size', type=int, default=64, help='Critic RNN hidden size')
    parser.add_argument('--grad_clip_value', type=float, default=0.0, help='Gradient clipping value')
    parser.add_argument('--grad_clip_norm', type=float, default=0.0, help='Gradient clipping norm')
    parser.add_argument('--obs_last_action', action='store_true', help='Include last action in observation')
    parser.add_argument('--eval_policy', action='store_true', help='Run evaluation during training')
    parser.add_argument('--eval_freq', type=int, default=100, help='Evaluation frequency in episodes')
    parser.add_argument('--eval_num_epi', type=int, default=10, help='Number of evaluation episodes')
    parser.add_argument('--sample_epi', action='store_true', help='Use full-episode replay buffer instead of traces')
    parser.add_argument('--trace_len', type=int, default=10, help='Trace length for replay buffer')

    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = make_mujoco_env(args.env_id, args.n_agent)

    os.makedirs(args.save_dir, exist_ok=True)

    model = MacIAC(env, env_terminate_step=args.env_terminate_step, n_env=args.n_env,
                   alg='MacIAC', n_agent=args.n_agent,
                   total_epi=args.total_epi, gamma=args.gamma,
                   a_lr=args.a_lr, c_lr=args.c_lr,
                   c_train_iteration=args.c_train_iteration,
                   eps_start=args.eps_start, eps_end=args.eps_end,
                   eps_stable_at=args.eps_stable_at,
                   c_hys_start=args.c_hys_start, c_hys_end=args.c_hys_end,
                   adv_hys_start=args.adv_hys_start, adv_hys_end=args.adv_hys_end,
                   hys_stable_at=args.hys_stable_at,
                   critic_hys=args.critic_hys, adv_hys=args.adv_hys,
                   etrpy_w_start=args.etrpy_w_start, etrpy_w_end=args.etrpy_w_end,
                   etrpy_w_stable_at=args.etrpy_w_stable_at,
                   train_freq=args.train_freq,
                   c_target_update_freq=args.c_target_update_freq,
                   c_target_soft_update=args.c_target_soft_update,
                   tau=args.tau, n_step_TD=args.n_step_TD,
                   TD_lambda=args.TD_lambda,
                   a_mlp_layer_size=args.a_mlp_layer_size,
                   a_rnn_layer_size=args.a_rnn_layer_size,
                   c_mlp_layer_size=args.c_mlp_layer_size,
                   c_rnn_layer_size=args.c_rnn_layer_size,
                   grad_clip_value=args.grad_clip_value,
                   grad_clip_norm=args.grad_clip_norm,
                   obs_last_action=args.obs_last_action,
                   eval_policy=args.eval_policy,
                   eval_freq=args.eval_freq,
                   eval_num_epi=args.eval_num_epi,
                   sample_epi=args.sample_epi,
                   trace_len=args.trace_len,
                   seed=args.seed,
                   run_id=args.run_id,
                   save_dir=args.save_dir,
                   resume=args.resume,
                   device=args.device)

    model.learn()


if __name__ == '__main__':
    main()
