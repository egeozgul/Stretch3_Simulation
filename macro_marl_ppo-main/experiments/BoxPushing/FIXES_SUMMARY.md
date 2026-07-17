# Summary of Fixes: Agent Not Learning Instructions

## Problems Identified

### 1. **Weak Penalty for Non-Compliance**
- **Original**: -0.5 penalty for not following instruction
- **Problem**: Tiny compared to task rewards (big box = +300, agent penalty = -5)
- **Impact**: Agents ignored the instruction penalty

### 2. **Console Print Spam**
- **Original**: Printed on EVERY action (potentially 1000s per episode)
- **Problem**: Overwhelmed console, made debugging impossible
- **Impact**: Couldn't see what was happening

### 3. **No Visibility into Action Selection**
- **Original**: No tracking of which actions agents actually chose
- **Problem**: Couldn't tell if agents were exploring vs exploiting
- **Impact**: No way to know if learning was happening

## Solutions Implemented

### 1. Increased Non-Compliance Penalty
**File**: `src/macro_marl/cores/pg_based/mac_iac/envs_runner.py`

**Change**:
```python
# Old
shaped_reward = r[agent_idx].item() - 0.5

# New  
shaped_reward = r[agent_idx].item() - 5.0
```

**Why**: 
- Penalty of -5.0 is now comparable to the environment's action penalty
- Gives a meaningful signal to the agent that non-compliance is costly
- Balanced with the +50 bonus for compliance

### 2. Removed Per-Action Prints, Added Episode Summaries
**File**: `src/macro_marl/cores/pg_based/mac_iac/envs_runner.py`

**Changes**:
- Removed all `print()` statements from `_inst_reward` method
- Added internal statistics tracking per environment
- Added episode-level summary every 10 episodes

**Output Example**:
```
============================================================
Env 0 | Episode 10 | Return: -5.20
Instruction: 'big_box_spot_0'
Expected: {'allowed_actions': [2]}
Compliance: 5/50 (10.0%)
Action distribution: {0: 15, 1: 12, 2: 5, 3: 8, 4: 10}
============================================================
```

**Benefits**:
- Clean console output
- Can see learning progress at a glance
- Action distribution shows if agents are converging

### 3. Added Action Distribution Tracking
**File**: `src/macro_marl/cores/pg_based/mac_iac/envs_runner.py`

**Added**:
```python
# Track compliance for statistics (per environment)
if not hasattr(self, '_instruction_stats'):
    self._instruction_stats = {}
if env_idx not in self._instruction_stats:
    self._instruction_stats[env_idx] = {
        'compliant': 0, 'non_compliant': 0, 'action_counts': {},
        'instruction': instruction_text, 'expected': expected_behavior
    }

# Track action distribution
if action_value not in self._instruction_stats[env_idx]['action_counts']:
    self._instruction_stats[env_idx]['action_counts'][action_value] = 0
self._instruction_stats[env_idx]['action_counts'][action_value] += 1
```

**Benefits**:
- Can see which actions are being chosen
- Can verify agents are shifting toward instructed actions
- Helps identify if agents are stuck in local optima

### 4. Improved Instruction Loading Messages
**File**: `src/macro_marl/algs/pg_based/mac_iac.py`

**Change**:
```python
# Print instruction texts and embedding shapes
print("\n" + "="*70)
print("LOADED INSTRUCTIONS:")
for i, (text, emb) in enumerate(zip(self.instruction_texts, self.instruction_embeddings)):
    emb_shape = emb.shape if hasattr(emb, 'shape') else 'scalar'
    print(f"  Instruction {i}: '{text}' -> embedding shape: {emb_shape}")
print(f"Total instructions: {len(self.instruction_texts)}")
print(f"Instructions will be randomly assigned to {n_env} environments")
print("="*70 + "\n")
```

**Benefits**:
- Confirms instructions loaded correctly
- Shows embedding shapes (should be (1, 64) for RNN size 64)
- Verifies number of environments

## Current Reward Shaping Setup

### Positive Instructions (do X)
- **Compliance**: base_reward + 50
- **Non-compliance**: base_reward - 5.0

### Negative Instructions (don't do X)  
- **Compliance (avoiding action)**: base_reward + 50
- **Non-compliance (doing prohibited action)**: base_reward - 5.0

### Comparison to Environment Rewards
```
Big box completion:      +300 (sparse)
Small box completion:    +10  (sparse)
Step penalty:            -0.1 (dense)
Agent action penalty:    -5   (when applicable)

Instruction bonus:       +50  (dense, every step)
Instruction penalty:     -5   (dense, every step)
```

## What to Expect During Training

### Early Training (Episodes 0-500)
```
Episode 10:
  Compliance: 8/45 (17.8%)  ← Slightly above random (12.5%)
  Action distribution: {0: 10, 1: 8, 2: 8, 3: 9, 4: 10}  ← Uniform
  Return: -8.5
```
- Agents are exploring
- Compliance slightly above random (12.5% for 8 actions)
- Action distribution roughly uniform

### Mid Training (Episodes 500-5000)
```
Episode 1000:
  Compliance: 22/50 (44.0%)  ← Improving!
  Action distribution: {0: 8, 1: 5, 2: 22, 3: 7, 4: 8}  ← Action 2 increasing
  Return: -3.2  ← Getting better
```
- Compliance increasing  
- Action distribution shifting toward instructed action (2)
- Returns improving

### Late Training (Episodes 5000+)
```
Episode 10000:
  Compliance: 41/50 (82.0%)  ← Strong compliance!
  Action distribution: {0: 3, 1: 2, 2: 41, 3: 1, 4: 3}  ← Heavily favoring action 2
  Return: 45.8  ← Much better
```
- High compliance (>80%)
- Action distribution dominated by instructed action
- Good returns

## How to Run

```bash
cd /home/willy/macro_marl_ppo
bash experiments/BoxPushing/mac_iac_instructions.sh
```

## Monitoring Training

### Console Output
Watch for episode summaries every 10 episodes. Check:
1. **Compliance** increasing over time
2. **Action distribution** concentrating on expected action
3. **Return** improving

### WandB Dashboard
Check metrics:
1. **Returns**: Should trend upward
2. **Instruction_Compliance**: Should increase from ~12% to >70%

## Troubleshooting

### If compliance is NOT increasing after 1000 episodes:

1. **Check instruction loading**:
   - Should see "LOADED INSTRUCTIONS" at startup
   - Verify embedding shapes are correct

2. **Try stronger reward shaping**:
   Edit `envs_runner.py` line ~418:
   ```python
   shaped_reward = r[agent_idx].item() + 100  # Increased from 50
   ```
   
   And line ~422:
   ```python
   shaped_reward = r[agent_idx].item() - 10.0  # Increased from 5.0
   ```

3. **Reduce exploration**:
   Edit the shell script:
   ```bash
   --eps_stable_at=2_000 \  # Down from 4_000
   ```

4. **Increase learning rate**:
   ```bash
   --a_lr=0.001 \  # Up from 0.0005
   ```

### If compliance is high but task performance is low:

This means instruction following is working, but agents need more training for the overall task.

**Solution**: Keep training! The instruction guidance is working, task completion will improve.

### If you want faster results:

Run a smaller test first:
```bash
--total_epi=5_000 \  # Instead of 40_000
--n_env=16 \         # Instead of 32
```

## Files Modified

1. `src/macro_marl/cores/pg_based/mac_iac/envs_runner.py`
   - Increased penalty: -0.5 → -5.0
   - Removed per-action prints
   - Added episode-level statistics tracking
   - Added episode summary logging

2. `src/macro_marl/algs/pg_based/mac_iac.py`
   - Improved instruction loading messages
   - Removed debug breakpoint

3. Documentation created:
   - `DEBUGGING_GUIDE.md` - Comprehensive debugging guide
   - `FIXES_SUMMARY.md` - This file
   - Updated `INSTRUCTIONS_GUIDE.md` and `QUICK_START_TWO_INSTRUCTIONS.md`

## Expected Timeline

With the current settings (40,000 episodes):
- **10 minutes**: Instructions loaded, training started
- **1-2 hours**: Episodes 0-5000, compliance should reach 30-40%
- **3-4 hours**: Episodes 5000-20000, compliance should reach 60-70%
- **6-8 hours**: Full training, compliance should reach 70-85%

## Summary

The main issue was **weak penalty for non-compliance** (-0.5 was too small). With the new penalty of -5.0, agents should learn to follow instructions. The new diagnostics will help you verify this is happening.

**Key success indicators**:
1. ✅ Compliance rate increases from ~12% to >70%
2. ✅ Action distribution shifts to favor instructed action
3. ✅ Episode returns improve over time

Good luck with training!

