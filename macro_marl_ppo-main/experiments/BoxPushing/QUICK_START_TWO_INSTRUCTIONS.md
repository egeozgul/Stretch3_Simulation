# Quick Start: Testing Box Pushing with Two Instructions

## Goal
Test the Box Pushing environment with two specific instructions:
1. **big_box_spot_0** - Agent goes to big box waypoint 0 (macro-action index 2)
2. **big_box_spot_1** - Agent goes to big box waypoint 1 (macro-action index 3)

## Quick Run

Simply execute the pre-configured script:

```bash
cd /home/willy/macro_marl_ppo
bash experiments/BoxPushing/mac_iac_instructions.sh
```

This will:
- Enable instructions
- Load both `big_box_spot_0` and `big_box_spot_1` instructions
- Randomly assign one instruction per environment
- Train agents on a 4x4 grid
- Apply reward shaping (+20 bonus for compliance, -0.5 penalty for non-compliance)
- Log metrics to WandB including instruction compliance rate

## What Happens

### Instruction Assignment
Each of the 32 parallel environments will be randomly assigned one of the two instructions:
- Some environments get `"big_box_spot_0"` → agents rewarded for using macro-action 2 (GT_BB0)
- Other environments get `"big_box_spot_1"` → agents rewarded for using macro-action 3 (GT_BB1)

### Reward Shaping
When an agent receives an instruction:
- **If agent performs the instructed action**: Base reward + 20
- **If agent performs a different action**: Base reward - 0.5

### Example
```
Environment 0: Instruction = "big_box_spot_0"
  - Agent chooses action 2 (GT_BB0) → Compliance! Bonus +20
  - Agent chooses action 3 (GT_BB1) → Non-compliance! Penalty -0.5

Environment 1: Instruction = "big_box_spot_1"
  - Agent chooses action 3 (GT_BB1) → Compliance! Bonus +20
  - Agent chooses action 2 (GT_BB0) → Non-compliance! Penalty -0.5
```

## Expected Output

### Console Output
You should see messages like:
```
[Env 0] Instruction fetched: 'big_box_spot_0' emb_shape=(768,) at step 0
Agent 0 is following instruction (action 2) - bonus applied
Agent 1 is not following instruction (action 0) - penalty applied
```

### WandB Metrics
- `Returns`: Episode returns (should be higher with instruction compliance)
- `Instruction_Compliance`: Percentage of actions matching the instruction (0.0 to 1.0)
- Tags: `instructions_enabled`

## Customization

### Different Grid Sizes

Edit `experiments/BoxPushing/mac_iac_instructions.sh` and uncomment the desired grid size:

```bash
# For 6x6 grid: uncomment lines 50-69
# For 8x8 grid: uncomment lines 72-99
```

### Different Reward Shaping

Edit `/home/willy/macro_marl_ppo/src/macro_marl/cores/pg_based/mac_iac/envs_runner.py`:

Search for the `_inst_reward` method (around line 357) and modify:
```python
# Current values:
shaped_reward = r[agent_idx].item() + 20   # Compliance bonus
shaped_reward = r[agent_idx].item() - 0.5  # Non-compliance penalty
```

### Test Only One Instruction

If you want to test only one instruction, modify the environment variable:

```bash
export OVERCOOKED_INSTRUCTIONS="big_box_spot_0"
```

Or edit the script directly.

## Verification

To verify the instructions are working:

1. **Check console output** for "Instruction fetched" messages
2. **Monitor WandB** for `Instruction_Compliance` metric (should be > 0)
3. **Check reward shaping** messages showing "bonus applied" or "penalty applied"

## Troubleshooting

### No instruction messages appearing
- Verify `INSTRUCTION_ENABLED=1` is set in the script (line 14)
- Check the `OVERCOOKED_INSTRUCTIONS` variable is set (line 17)

### Compliance rate is 0%
- This is normal at the start of training (agents are random)
- Should increase as training progresses if reward shaping is working
- Check that reward bonuses/penalties are being applied (console output)

### WandB not logging
- Verify WandB API key is correct in `mac_iac.py` (line 201)
- Check network connectivity

## Summary

This setup tests whether agents can learn to follow language-based instructions by:
1. Encoding instructions using BERT
2. Conditioning agent actions on instruction embeddings
3. Applying reward shaping to encourage compliance
4. Measuring compliance during evaluation

The two instructions test whether agents can differentiate between:
- Going to big box spot 0 (action 2)
- Going to big box spot 1 (action 3)

Both are similar tasks but at different locations, making this a good test of instruction following.

