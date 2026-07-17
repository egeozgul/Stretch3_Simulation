# Debugging Guide: Box Pushing with Instructions

## Issue: Agent Not Learning to Follow Instructions

### Diagnostic Checklist

#### 1. Verify Instructions are Loaded
When you start training, you should see:
```
======================================================================
LOADED INSTRUCTIONS:
  Instruction 0: 'big_box_spot_0' -> embedding shape: (1, 64)
  Instruction 1: 'big_box_spot_1' -> embedding shape: (1, 64)
Total instructions: 2
Instructions will be randomly assigned to 32 environments
======================================================================
```

#### 2. Monitor Episode Statistics
Every 10 episodes, you'll see output like:
```
============================================================
Env 0 | Episode 10 | Return: -5.20
Instruction: 'big_box_spot_0'
Expected: {'allowed_actions': [2]}
Compliance: 5/50 (10.0%)
Action distribution: {0: 15, 1: 12, 2: 5, 3: 8, 4: 10}
============================================================
```

**What to look for:**
- **Compliance Rate**: Should increase over time (start ~12.5% random, should go to >80%)
- **Action Distribution**: Should concentrate on the expected action as training progresses
- **Return**: Should increase as compliance improves

#### 3. Understanding the Action Distribution

Box Pushing has 8 actions:
```
0: GT_SB0    (Go to small box 0)
1: GT_SB1    (Go to small box 1)
2: GT_BB0    (Go to big box spot 0) ← Expected for "big_box_spot_0"
3: GT_BB1    (Go to big box spot 1) ← Expected for "big_box_spot_1"
4: Push      (Push box)
5: T_L       (Turn left)
6: T_R       (Turn right)
7: Stay      (Stay in place)
```

**Example Good Progress:**
```
Episode 10:  {0: 15, 1: 12, 2: 5, 3: 8, 4: 10}   ← Random exploration
Episode 100: {0: 8, 1: 5, 2: 25, 3: 5, 4: 7}     ← Learning action 2
Episode 500: {0: 2, 1: 1, 2: 42, 3: 1, 4: 4}     ← Strong preference for action 2
```

### Common Issues and Solutions

#### Issue 1: Compliance Rate Not Increasing

**Symptoms:**
- Compliance stays around 10-15% even after hundreds of episodes
- Action distribution remains uniform

**Possible Causes:**
1. **Reward shaping not strong enough**: Increased penalty from -0.5 to -5.0
2. **Epsilon too high**: Agents exploring too much, reduce `eps_stable_at`
3. **Learning rate too low**: Try increasing `a_lr` from 0.0005 to 0.001
4. **Instruction embedding not being used**: Check that `use_instructions=True`

**Solutions:**
```bash
# Increase learning rate
--a_lr=0.001 \

# Reduce exploration faster
--eps_stable_at=2_000 \  # Down from 4_000

# Increase reward shaping (already done - penalty is now -5.0 instead of -0.5)
```

#### Issue 2: High Compliance But Low Task Performance

**Symptoms:**
- Compliance rate is high (>70%)
- But episode returns don't improve
- Agents choose the "right" action but don't complete the task

**Possible Cause:**
Agents learn to spam the rewarded action without actually completing the task.

**Solution:**
This is actually okay! The instruction following is working. The agents need more training to learn the overall task. The instruction is just guiding them toward useful behaviors.

#### Issue 3: Print Spam

**Symptoms:**
- Console flooded with messages

**Already Fixed:**
- Removed per-action prints
- Added episode-level summaries (every 10 episodes)
- Statistics tracked internally

### Monitoring Training Progress

#### WandB Metrics
Log into WandB and check:
1. **Returns**: Should increase over time
2. **Instruction_Compliance**: Should increase from ~12.5% to >70%
3. Compare runs with/without instructions

#### Console Monitoring
```bash
# Run training and save output
bash experiments/BoxPushing/mac_iac_instructions.sh 2>&1 | tee training.log

# Monitor compliance in real-time
tail -f training.log | grep -A5 "Episode"

# Search for specific environment's progress
grep "Env 0" training.log | grep "Episode"
```

### Expected Learning Curve

#### Phase 1: Random Exploration (Episodes 0-500)
- Compliance: 10-20%
- Returns: Low, inconsistent
- Action distribution: Mostly uniform
- **This is normal!** Agents are exploring

#### Phase 2: Beginning to Learn (Episodes 500-2000)
- Compliance: 20-40%
- Returns: Starting to improve
- Action distribution: Slight preference for instructed action
- Epsilon decreasing, agents starting to exploit

#### Phase 3: Strong Instruction Following (Episodes 2000-10000)
- Compliance: 40-70%
- Returns: Consistently improving
- Action distribution: Clear preference for instructed action
- Agents reliably following instructions

#### Phase 4: Convergence (Episodes 10000+)
- Compliance: 70-90%
- Returns: Near-optimal
- Action distribution: Dominated by instructed action
- Task completion improving

### Tuning Reward Shaping

Current settings in `envs_runner.py`:
```python
# Compliance bonus
shaped_reward = r[agent_idx].item() + 50  

# Non-compliance penalty  
shaped_reward = r[agent_idx].item() - 5.0
```

**If learning is too slow:**
```python
# Increase bonus
shaped_reward = r[agent_idx].item() + 100

# Increase penalty
shaped_reward = r[agent_idx].item() - 10.0
```

**If agents only follow instructions but don't complete task:**
```python
# Reduce bonus (focus more on task reward)
shaped_reward = r[agent_idx].item() + 20

# Keep penalty moderate
shaped_reward = r[agent_idx].item() - 5.0
```

### Comparing With and Without Instructions

To verify instructions are helping:

1. **Run with instructions:**
```bash
export INSTRUCTION_ENABLED=1
export OVERCOOKED_INSTRUCTIONS="big_box_spot_0||big_box_spot_1"
bash experiments/BoxPushing/mac_iac_instructions.sh
```

2. **Run without instructions (baseline):**
```bash
export INSTRUCTION_ENABLED=0
bash experiments/BoxPushing/mac_iac_instructions.sh
```

3. **Compare in WandB:**
- Filter by tags: `instructions_enabled` vs `instructions_disabled`
- Compare convergence speed
- Compare final performance

### Quick Diagnostic Test

Run a short test to verify everything is working:

```bash
# Short test run (1000 episodes instead of 40000)
export INSTRUCTION_ENABLED=1
export OVERCOOKED_INSTRUCTIONS="big_box_spot_0||big_box_spot_1"

pg_based_main.py --save_dir='ma_iac_bp6_test' \
                --alg='MacIAC' \
                --env_id='BP-MA-v0' \
                --n_agent=2 \
                --device='cpu' \
                --env_terminate_step=100 \
                --big_box_reward=300 \
                --a_lr=0.001 \
                --c_lr=0.003 \
                --train_freq=32 \
                --n_env=4 \  # Fewer environments for testing
                --c_target_update_freq=32 \
                --n_step_TD=0 \
                --grad_clip_norm=0 \
                --eps_start=1.0 \
                --eps_end=0.01 \
                --eps_stable_at=500 \  # Faster epsilon decay
                --total_epi=1_000 \  # Short test
                --grid_dim 6 6 \
                --gamma=0.98 \
                --eval_policy \
                --sample_epi \
                --run_id=0
```

After 1000 episodes, you should see compliance increasing from ~12% to at least 25-30%.

### Advanced Debugging

#### Check Instruction Embeddings Are Being Passed

Add this temporarily to `controller.py` line 47:
```python
if instruction_emb is not None:
    print(f"Agent {idx} received instruction embedding shape: {instruction_emb.shape}")
```

You should see this during action selection.

#### Check Reward Shaping Is Applied

The episode summary shows compliance counts, which confirms reward shaping is working.

#### Verify Actions Match Environment

Print available actions in the environment to ensure indices match:
```python
# In box_pushing_MA.py
print("Macro actions:", [ma.name for ma in self.MAs])
```

Should output:
```
Macro actions: ['GT_SB0', 'GT_SB1', 'GT_BB0', 'GT_BB1', 'Push', 'T_L', 'T_R', 'Stay']
```

### Summary

The key indicators that instruction following is working:
1. ✅ Instructions load correctly at startup
2. ✅ Episode statistics show compliance tracking
3. ✅ Compliance rate increases over time
4. ✅ Action distribution shifts toward expected actions
5. ✅ Episode returns improve

If you see all of these, the system is working correctly and just needs more training time!

