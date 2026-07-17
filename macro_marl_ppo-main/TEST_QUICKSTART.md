# Quick Test Guide - Hard-Coded Chop Policy

## One-Line Test

```bash
python test_hardcoded_chop_policy.py --episodes 5
```

## What to Look For

The test will output something like:

```
===============================================================================
HARD-CODED CHOP POLICY TEST
===============================================================================
Food on knife steps: 85
Agent 2 chopped when food on knife: 84/85
Success rate: 98.8%

✓ PASS: Hard-coded policy is working correctly!
```

## Success Criteria

| Success Rate | Status | Meaning |
|---|---|---|
| > 90% | ✓ PASS | Policy works correctly |
| 70% - 90% | ⚠ PARTIAL | Policy mostly works |
| < 70% | ✗ FAIL | Policy not working properly |

## Common Issues & Solutions

### Test Can't Find Policies

**Problem:** `Policy not found for agent...`

**Solution:** Specify the policy directory:
```bash
python test_hardcoded_chop_policy.py --policy_dir experiments/Overcooked/policy_nns/mac_iac_overcooked_D_desktop2_stochastic
```

Or train policies first:
```bash
cd experiments/Overcooked
bash mac_iac.sh
```

### "No food appeared on knife" message

**Problem:** The test runs but no food shows up on the knife

**Reason:** The environment randomly spawns food; this is expected variance. Run more episodes:
```bash
python test_hardcoded_chop_policy.py --episodes 20
```

### Success Rate Too Low

**Problem:** Success rate is below 70%

**Cause:** The hard-coded policy may not be triggered correctly. Check:
1. Is agent 2 receiving the instruction "let me do all the chopping"?
2. Is the food detection logic working?

**Debug:** Add logging to `_is_food_on_knife()` in `envs_runner.py`

## Detailed Test Output Example

```
======================================================================
HARD-CODED CHOP POLICY TEST
======================================================================
Map Type: D
Number of Episodes: 5
Number of Agents: 3

Agent 2 will receive instruction: 'let me do all the chopping'
Expected behavior: Agent 2 should perform chop action (10)
                  whenever food is on the knife
======================================================================

Environment created: MacEnvWrapper
Observation sizes: [28, 28, 28]
Action sizes: [20, 20, 20]
Loaded policy for agent 0
Loaded policy for agent 1

======================================================================
Episode 1
======================================================================
Food on knife: 12 steps
Agent 2 chopped: 12/12 (100.0%)

======================================================================
Episode 2
======================================================================
Food on knife: 8 steps
Agent 2 chopped: 7/8 (87.5%)

[... more episodes ...]

======================================================================
TEST SUMMARY
======================================================================
Total episodes: 5
Total steps: 1000
Average steps per episode: 200.0

Food on knife steps: 85
Agent 2 chopped when food on knife: 84/85
Success rate: 98.8%

✓ PASS: Hard-coded policy is working correctly!
======================================================================
```

## Next Steps After Testing

If the policy works (PASS status):
1. Run training with `bash experiments/Overcooked/mac_iac.sh`
2. The hard-coded policy will automatically help agent 2 learn
3. Agents 0 and 1 will learn to collaborate around agent 2's chopping

If the policy doesn't work:
1. Check the implementation in `src/macro_marl/cores/pg_based/mac_iac/envs_runner.py`
2. Verify the instruction "let me do all the chopping" is being passed
3. Check food detection logic in `_is_food_on_knife()`
