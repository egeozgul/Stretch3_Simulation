#!/usr/bin/env python3
"""Run standalone Mujoco MacIAC training with an external environment.

This script is intentionally minimal and configurable so you can connect a
separate Mujoco project without importing the rest of this repo.

Usage example:
  python mujoco_maciac/run.py \
    --env_path /path/to/your/mujoco/project \
    --env_module your_env_module \
    --env_class YourMujocoMacroEnv \
    --env_kwargs '{"n_agent": 2, "env_id": "YourEnvID"}' \
    --total_epi 20000

If your environment is already importable from Python, set --env_path to the
root of your project and pass --env_module / --env_class accordingly.
"""

import argparse
import importlib
import json
import os
import random
import sys

import numpy as np
import torch

from mujoco_maciac.algs import MacIAC

# Force offline WandB so training can run without remote logging.
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_DISABLE_SERVICE", "true")


class MujocoMacEnvWrapper:
    """Wrap an external Mujoco macro-action environment for MacIAC."""

    def __init__(self, env):
        self.env = env

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
        return self.env.reset()

    def step(self, macro_actions):
        return self.env.run(macro_actions)

    def get_avail_actions(self):
        return self.env.get_avail_actions()

    def action_space_sample(self):
        return self.env.macro_action_sample()


def load_env_class(env_path, env_module, env_class):
    if env_path:
        sys.path.insert(0, os.path.abspath(env_path))
    module = importlib.import_module(env_module)
    return getattr(module, env_class)


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone Mujoco MacIAC trainer")
    parser.add_argument('--env_path', type=str, default=None, help='Path to the external Mujoco project directory')
    parser.add_argument('--env_module', type=str, required=True, help='Python module name for the external Mujoco env')
    parser.add_argument('--env_class', type=str, required=True, help='Class name of the external Mujoco env')
    parser.add_argument('--env_kwargs', type=str, default='{}', help='JSON dict of kwargs for the env constructor')
    parser.add_argument('--n_agent', type=int, default=1, help='Number of agents in the Mujoco env')
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
    parser.add_argument('--env_terminate_step', type=int, default=300, help='Max steps per episode')
    parser.add_argument('--n_env', type=int, default=1, help='Number of parallel environments')
    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(args.run_id)
    np.random.seed(args.run_id)
    torch.manual_seed(args.run_id)

    env_kwargs = json.loads(args.env_kwargs)
    if args.n_agent is not None:
        env_kwargs.setdefault('n_agent', args.n_agent)

    env_cls = load_env_class(args.env_path, args.env_module, args.env_class)
    env = env_cls(**env_kwargs)
    env = MujocoMacEnvWrapper(env)

    os.makedirs(args.save_dir, exist_ok=True)

    model = MacIAC(env,
                   env_terminate_step=args.env_terminate_step,
                   n_env=args.n_env,
                   alg='MacIAC',
                   n_agent=args.n_agent,
                   total_epi=args.total_epi,
                   gamma=args.gamma,
                   a_lr=args.a_lr,
                   c_lr=args.c_lr,
                   c_train_iteration=args.c_train_iteration,
                   eps_start=args.eps_start,
                   eps_end=args.eps_end,
                   eps_stable_at=args.eps_stable_at,
                   c_hys_start=args.c_hys_start,
                   c_hys_end=args.c_hys_end,
                   adv_hys_start=args.adv_hys_start,
                   adv_hys_end=args.adv_hys_end,
                   hys_stable_at=args.hys_stable_at,
                   critic_hys=args.critic_hys,
                   adv_hys=args.adv_hys,
                   etrpy_w_start=args.etrpy_w_start,
                   etrpy_w_end=args.etrpy_w_end,
                   etrpy_w_stable_at=args.etrpy_w_stable_at,
                   train_freq=args.train_freq,
                   c_target_update_freq=args.c_target_update_freq,
                   c_target_soft_update=args.c_target_soft_update,
                   tau=args.tau,
                   n_step_TD=args.n_step_TD,
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
                   seed=args.run_id,
                   run_id=args.run_id,
                   save_dir=args.save_dir,
                   resume=args.resume,
                   device=args.device)

    model.learn()


if __name__ == '__main__':
    main()
