# Mujoco MacIAC Fusion

This folder contains a self-contained MacIAC training package that can be used with an external Mujoco macro-action environment.

## Contents

- `mujoco_maciac/`: package modules for MacIAC training
- `mujoco_maciac/run.py`: standalone runner script
- `mujoco_maciac/requirements.txt`: Python package dependencies

## How to use

1. Install dependencies:

```bash
pip install -r mujoco_maciac/requirements.txt
```

2. Run the trainer with your Mujoco env:

```bash
python mujoco_maciac/run.py \
  --env_path /path/to/your/mujoco/project \
  --env_module your_env_module \
  --env_class YourMujocoMacroEnv \
  --env_kwargs '{"n_agent": 2}' \
  --total_epi 20000 \
  --save_dir mujoco_maciac_training
```

3. If your environment constructor needs additional kwargs, pass them via `--env_kwargs` as JSON.

## Required external env interface

Your Mujoco environment must expose the following interface:

- `env.reset()` -> observation
- `env.run(macro_actions)` -> observation, reward, done, info
- `env.get_avail_actions()` -> per-agent available action masks
- properties: `n_agent`, `obs_size`, `n_action`

## Notes

- This folder is designed to be independent from the repo's `src/macro_marl` package.
- The trainer uses MacIAC in vanilla mode with instructions disabled by default.
- `wandb` is configured to run in offline mode.
