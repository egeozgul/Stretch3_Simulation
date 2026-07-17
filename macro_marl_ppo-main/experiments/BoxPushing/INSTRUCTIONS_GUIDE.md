# Box Pushing with Instructions - User Guide

## Overview
This guide explains how to test the Box Pushing environment with language-based instructions for agents.

### Instruction Refresh Policy
**Instructions are refreshed every 20 timesteps.** This means agents receive a new random instruction every 20 steps, giving them a consistent instruction to follow for a period of time. This helps with learning as agents have time to execute multi-step behaviors under the same instruction.

## Supported Instructions

### Positive Instructions (Agent should perform action)
The following instructions are supported for Box Pushing:

1. **Big Box Spot 0**
   - Instruction text: `"big_box_spot_0"`, `"go to big box spot 0"`, or `"big box spot 0"`
   - Expected macro-action: GT_BB0 (action index 2)
   - Description: Agent should go to big box waypoint 0

2. **Big Box Spot 1**
   - Instruction text: `"big_box_spot_1"`, `"go to big box spot 1"`, or `"big box spot 1"`
   - Expected macro-action: GT_BB1 (action index 3)
   - Description: Agent should go to big box waypoint 1

3. **Small Box 0**
   - Instruction text: `"small_box_0"`, `"go to small box 0"`, or `"small box 0"`
   - Expected macro-action: GT_SB0 (action index 0)
   - Description: Agent should go to small box 0

4. **Small Box 1**
   - Instruction text: `"small_box_1"`, `"go to small box 1"`, or `"small box 1"`
   - Expected macro-action: GT_SB1 (action index 1)
   - Description: Agent should go to small box 1

5. **Push**
   - Instruction text: `"push"`
   - Expected macro-action: Push (action index 4)
   - Description: Agent should push a box

### Negative Instructions (Agent should avoid action)

1. **Don't go to Big Box Spot 0**
   - Instruction text: `"don't go to big box spot 0"` or `"avoid big box spot 0"`
   - Prohibited action: GT_BB0 (action index 2)
   - Description: Agent should avoid going to big box waypoint 0

2. **Don't go to Big Box Spot 1**
   - Instruction text: `"don't go to big box spot 1"` or `"avoid big box spot 1"`
   - Prohibited action: GT_BB1 (action index 3)
   - Description: Agent should avoid going to big box waypoint 1

## Box Pushing Macro-Actions Reference

The Box Pushing environment has the following macro-actions:

| Index | Name     | Description                    |
|-------|----------|--------------------------------|
| 0     | GT_SB0   | Go to small box 0              |
| 1     | GT_SB1   | Go to small box 1              |
| 2     | GT_BB0   | Go to big box spot 0           |
| 3     | GT_BB1   | Go to big box spot 1           |
| 4     | Push     | Push box                       |
| 5     | T_L      | Turn left                      |
| 6     | T_R      | Turn right                     |
| 7     | Stay     | Stay in place                  |

## How to Run

### Using the Pre-configured Script

The easiest way to test is to use the pre-configured script:

```bash
cd /home/willy/macro_marl_ppo
bash experiments/BoxPushing/mac_iac_instructions.sh
```

This script is configured to test with two instructions:
- `"big_box_spot_0"`
- `"big_box_spot_1"`

Each environment will be randomly assigned one of these instructions at the start.

### Custom Configuration

To run with custom instructions, you can set the environment variables before running:

```bash
# Enable instructions
export INSTRUCTION_ENABLED=1

# Set multiple instructions (separated by ||)
export OVERCOOKED_INSTRUCTIONS="big_box_spot_0||big_box_spot_1"

# Run the training
pg_based_main.py --save_dir='ma_iac_bp4_instructions' \
                --alg='MacIAC' \
                --env_id='BP-MA-v0' \
                --n_agent=2 \
                --env_terminate_step=100 \
                --big_box_reward=300 \
                --a_lr=0.0005 \
                --c_lr=0.003 \
                --train_freq=32 \
                --n_env=32 \
                --c_target_update_freq=32 \
                --n_step_TD=0 \
                --grad_clip_norm=0 \
                --eps_start=1.0 \
                --eps_end=0.01 \
                --eps_stable_at=4_000 \
                --total_epi=40_000 \
                --grid_dim 4 4 \
                --gamma=0.98 \
                --eval_policy \
                --sample_epi \
                --run_id=0
```

### Testing Single Instruction

To test with a single instruction:

```bash
export INSTRUCTION_ENABLED=1
export OVERCOOKED_INSTRUCTIONS="big_box_spot_0"

# Run training...
```

### Testing Multiple Instructions

To test with more than two instructions:

```bash
export INSTRUCTION_ENABLED=1
export OVERCOOKED_INSTRUCTIONS="big_box_spot_0||big_box_spot_1||small_box_0||small_box_1"

# Run training...
```

## Reward Shaping

When instructions are enabled, the system applies reward shaping to encourage compliance:

### Positive Instructions
- **Compliance bonus**: +20 reward when agent performs the instructed action
- **Non-compliance penalty**: -0.5 reward when agent performs a different action

### Negative Instructions
- **Compliance bonus**: +20 reward when agent avoids the prohibited action
- **Violation penalty**: -0.5 reward when agent performs the prohibited action

## Monitoring

### WandB Logging

The system automatically logs to WandB:
- **Returns**: Average episode returns
- **Instruction_Compliance**: Percentage of actions that comply with instructions (during evaluation)
- **Tags**: Runs are tagged with `instructions_enabled` or `instructions_disabled`

### Console Output

During execution, you'll see console messages indicating:
- Instruction assignment per environment
- Compliance/violation events with reward adjustments
- Episode statistics

## Grid Sizes

The script supports different grid sizes. Uncomment the relevant section in `mac_iac_instructions.sh`:

- **4x4**: Quick testing (default, active)
- **6x6**: Medium complexity (commented out)
- **8x8**: Standard training (commented out)

## Environment Variables Summary

| Variable                   | Purpose                                  | Example                              |
|----------------------------|------------------------------------------|--------------------------------------|
| `INSTRUCTION_ENABLED`      | Toggle instructions on/off               | `0` (off) or `1` (on)                |
| `OVERCOOKED_INSTRUCTIONS`  | Set instruction texts (multiple with ||) | `"big_box_spot_0||big_box_spot_1"`   |

## Troubleshooting

### Instructions not working
1. Check that `INSTRUCTION_ENABLED=1` is set
2. Verify instruction text matches one of the supported phrases (case-insensitive)
3. Check console output for "Instruction fetched" messages

### Reward shaping too strong/weak
- Adjust the bonus/penalty values in `envs_runner.py` (search for `_inst_reward` method)
- Current values: +20 for compliance, -0.5 for non-compliance

### Environment-specific issues
- Ensure the environment has the expected macro-actions (check `box_pushing_MA.py`)
- Waypoint positions vary by grid size (4x4, 6x6, 8x8 have different coordinates)

## Next Steps

After running the test:
1. Check WandB dashboard for compliance metrics
2. Compare performance between instruction-enabled and disabled runs
3. Experiment with different instruction combinations
4. Try negative instructions to test avoidance behavior

