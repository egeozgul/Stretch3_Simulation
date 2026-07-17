# Instruction Timing Update

## Change Summary

**Instructions are now provided every 20 timesteps** instead of at macro-action completion.

## Old Behavior

Previously, instructions were refreshed whenever any agent completed a macro-action:
- **Timing**: Variable (depends on when macro-actions complete)
- **Problem**: Inconsistent timing made learning harder
- **Issue**: Some agents might get many instruction changes in a short time, others might keep the same instruction for a long time

## New Behavior

Instructions are now refreshed on a fixed schedule:
- **Timing**: Every 20 timesteps (fixed interval)
- **Benefit**: Consistent learning signal across all environments
- **Benefit**: Agents have time to execute multi-step behaviors under the same instruction

## Implementation Details

### Code Changes

**File**: `src/macro_marl/cores/pg_based/mac_iac/envs_runner.py`

1. **Added instruction timing tracking**:
```python
# Track when instructions were last updated (for periodic refresh)
self.instruction_refresh_interval = 20  # Refresh every 20 timesteps
self.last_instruction_step = [0] * n_envs  # Track last refresh timestep per env
```

2. **Modified instruction fetching logic**:
```python
# Fetch new instruction every 20 timesteps
steps_since_last_instruction = self.step_count[idx] - self.last_instruction_step[idx]

# Fetch instruction if: (1) first time (step 0), or (2) 20 steps have passed
should_fetch = (self.instruction_embs[idx] is None and self.instruction_texts_for_env[idx] is None) or \
               (steps_since_last_instruction >= self.instruction_refresh_interval)

if should_fetch:
    # Get fresh instruction and update timer
    inst = self.instruction_provider(idx, self.step_count[idx])
    # ... process instruction ...
    self.last_instruction_step[idx] = self.step_count[idx]
```

3. **Removed macro-action-based refresh**:
- Deleted the code that refreshed instructions when agents completed macro-actions
- This prevents conflicts with the time-based refresh

4. **Reset timers on episode reset**:
```python
# Reset instruction timer when episode ends
self.last_instruction_step[idx] = 0
```

### Instruction Timeline Example

```
Step 0:   Instruction = "big_box_spot_0"  (new instruction)
Step 1:   Instruction = "big_box_spot_0"  (same)
Step 2:   Instruction = "big_box_spot_0"  (same)
...
Step 19:  Instruction = "big_box_spot_0"  (same)
Step 20:  Instruction = "big_box_spot_1"  (new instruction - 20 steps elapsed)
Step 21:  Instruction = "big_box_spot_1"  (same)
...
Step 39:  Instruction = "big_box_spot_1"  (same)
Step 40:  Instruction = "big_box_spot_0"  (new instruction - 20 steps elapsed)
```

## Why 20 Timesteps?

The interval of 20 timesteps was chosen because:

1. **Macro-action duration**: Most macro-actions in Box Pushing take 5-15 steps to complete
2. **Multi-step behaviors**: 20 steps allows agents to execute 1-3 macro-actions under the same instruction
3. **Learning stability**: Long enough to learn associations, short enough to get diverse experiences
4. **Flexibility**: Can be adjusted by changing `self.instruction_refresh_interval`

## Adjusting the Interval

To change the instruction refresh interval, edit `envs_runner.py`:

```python
# In __init__ method around line 115
self.instruction_refresh_interval = 20  # Change this value

# Examples:
# self.instruction_refresh_interval = 10   # More frequent changes
# self.instruction_refresh_interval = 50   # Less frequent changes
# self.instruction_refresh_interval = 100  # Rare changes
```

### Recommended Values

| Interval | Use Case |
|----------|----------|
| 10 steps | Fast-paced tasks, short macro-actions |
| 20 steps | **Default** - Good balance for Box Pushing |
| 50 steps | Longer-term planning tasks |
| 100 steps | Near-constant instructions, minimal variety |

## Impact on Learning

### Expected Benefits

1. **More consistent learning signal**: Agents see the same instruction for multiple steps
2. **Better credit assignment**: Rewards from following an instruction happen within the same instruction period
3. **Easier to learn multi-step behaviors**: Agents can complete sequences under one instruction

### What to Monitor

Watch the episode statistics to see if learning improves:

```
============================================================
Env 0 | Episode 100 | Return: -2.50
Instruction: 'big_box_spot_0'
Expected: {'allowed_actions': [2]}
Compliance: 18/45 (40.0%)  ← Should be higher with fixed timing
Action distribution: {0: 5, 1: 4, 2: 18, 3: 6, 4: 12}
============================================================
```

Look for:
- **Higher compliance rates** compared to the old variable timing
- **Stronger action preferences** (action distribution more concentrated)
- **Faster learning** (compliance increases earlier in training)

## Comparison with Old Behavior

### Variable Timing (Old)
```
Step 0:   Instruction = "big_box_spot_0"  (agent 0 ready)
Step 3:   Instruction = "big_box_spot_1"  (agent 1 ready)
Step 5:   Instruction = "big_box_spot_0"  (agent 0 ready)
Step 6:   Instruction = "big_box_spot_1"  (agent 1 ready)
Step 12:  Instruction = "big_box_spot_0"  (agent 0 ready)
...
```
**Problem**: Instructions change at unpredictable times, hard for agents to learn

### Fixed Timing (New)
```
Step 0:   Instruction = "big_box_spot_0"
Step 20:  Instruction = "big_box_spot_1"
Step 40:  Instruction = "big_box_spot_0"
Step 60:  Instruction = "big_box_spot_1"
...
```
**Benefit**: Predictable intervals, consistent learning windows

## Backward Compatibility

This change is **backward compatible**:
- Old saved models will still work
- The instruction provider interface hasn't changed
- The memory buffer format is the same

## Testing

To verify the change is working:

1. Run training with debug output:
```bash
bash experiments/BoxPushing/mac_iac_instructions.sh
```

2. Check console output for episode statistics
3. Verify compliance rates are increasing
4. Monitor action distribution shifting toward expected actions

## Future Enhancements

Possible improvements:
1. **Adaptive intervals**: Change interval based on learning progress
2. **Per-environment intervals**: Different intervals for different environments
3. **Task-dependent intervals**: Adjust based on task complexity
4. **Curriculum learning**: Start with long intervals, decrease over time

## Summary

| Aspect | Old | New |
|--------|-----|-----|
| **Timing** | Variable (macro-action based) | Fixed (20 timesteps) |
| **Consistency** | Unpredictable | Predictable |
| **Learning** | Harder | Easier |
| **Configuration** | Not adjustable | Adjustable via `instruction_refresh_interval` |

The time-based instruction refresh provides a more stable learning environment for instruction-following agents.

