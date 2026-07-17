# Per-Agent Instructions Feature

## Overview

The system now supports **per-agent instructions** where each agent can receive a different instruction, enabling better coordination in multi-agent tasks.

## Why Per-Agent Instructions?

For Box Pushing, you need two agents to coordinate:
- **Agent 0**: Go to big box spot 0
- **Agent 1**: Go to big box spot 1  

With per-agent instructions, each agent learns their specific role!

## How It Works

### Automatic Mode Detection

The system automatically detects whether to use per-agent or per-environment mode:

**Per-Agent Mode** (activated when number of instructions == number of agents):
```bash
export OVERCOOKED_INSTRUCTIONS="big_box_spot_0||big_box_spot_1"  # 2 instructions
# With n_agent=2, this triggers PER-AGENT mode
```

**Per-Environment Mode** (when number of instructions ≠ number of agents):
```bash
export OVERCOOKED_INSTRUCTIONS="big_box_spot_0||big_box_spot_1||small_box_0"  # 3 instructions
# With n_agent=2, this triggers PER-ENVIRONMENT mode
```

### Startup Output

When you run training, you'll see:

**Per-Agent Mode:**
```
======================================================================
LOADED INSTRUCTIONS:
  Instruction 0: 'big_box_spot_0' -> embedding shape: (1, 64)
  Instruction 1: 'big_box_spot_1' -> embedding shape: (1, 64)
Total instructions: 2
PER-AGENT ASSIGNMENT MODE:
  Each of 2 agents will get their own fixed instruction
  Agent 0: 'big_box_spot_0'
  Agent 1: 'big_box_spot_1'
======================================================================
```

**Per-Environment Mode:**
```
======================================================================
LOADED INSTRUCTIONS:
  Instruction 0: 'big_box_spot_0' -> embedding shape: (1, 64)
  Instruction 1: 'big_box_spot_1' -> embedding shape: (1, 64)
  Instruction 2: 'small_box_0' -> embedding shape: (1, 64)
Total instructions: 3
PER-ENVIRONMENT MODE:
  Instructions will be randomly assigned to 32 environments
  All agents in an environment share the same instruction
======================================================================
```

## Episode Statistics

With per-agent instructions, episode statistics show each agent's instruction:

```
============================================================
Env 0 | Episode 10 | Return: -13.14
Agent 0 Instruction: 'big_box_spot_0' -> Expected: {'allowed_actions': [2]}
Agent 1 Instruction: 'big_box_spot_1' -> Expected: {'allowed_actions': [3]}
Compliance: 162/200 (81.0%)
Action distribution: {0: 4, 1: 10, 2: 82, 3: 80, 4: 3, 5: 4, 6: 5, 7: 4}
============================================================
```

**What to look for:**
- Agent 0's actions should cluster around action 2 (big_box_spot_0)
- Agent 1's actions should cluster around action 3 (big_box_spot_1)
- Compliance is calculated across both agents

## Implementation Details

### Per-Agent Instruction Storage

Instructions are now stored per-agent:
```python
# OLD: One instruction per environment
self.instruction_embs = [None] * n_envs  # [env]

# NEW: One instruction per agent per environment  
self.instruction_embs = [[None] * n_agent for _ in range(n_envs)]  # [env][agent]
```

### Instruction Provider Signature

The instruction provider now accepts `agent_idx`:
```python
def instruction_provider(env_idx, step, agent_idx=None):
    if per_agent_mode and agent_idx is not None:
        # Return agent-specific instruction
        inst_idx = agent_idx % len(instruction_texts)
        return (instruction_texts[inst_idx], instruction_embeddings[inst_idx])
    else:
        # Return environment-wide instruction (same for all agents)
        inst_idx = env_instruction_indices[env_idx]
        return (instruction_texts[inst_idx], instruction_embeddings[inst_idx])
```

### Controller Changes

The controller now accepts a list of instruction embeddings:
```python
# Pass list of per-agent instructions
instruction_emb=self.instruction_embs[idx]  # This is a list: [emb_agent0, emb_agent1, ...]

# Controller extracts the right embedding for each agent
for idx, agent in enumerate(self.agents):
    if isinstance(instruction_emb, list):
        agent_instruction = instruction_emb[idx]  # Get this agent's instruction
    else:
        agent_instruction = instruction_emb  # Fallback: same for all
```

### Reward Shaping

Reward shaping is now per-agent based on each agent's instruction:
```python
for agent_idx, agent_action in enumerate(a):
    # Get THIS agent's instruction
    agent_instruction_text = inst_texts[agent_idx]
    
    # Get expected behavior for THIS agent
    expected_behavior = self._get_expected_macro_action(agent_instruction_text)
    
    # Shape reward based on THIS agent's compliance
    if action_value in expected_behavior['allowed_actions']:
        shaped_reward = r[agent_idx].item() + 50  # Agent 0 gets +50 for action 2
    else:
        shaped_reward = r[agent_idx].item() - 5.0  # Agent 0 gets -5 for action 3
```

## Expected Behavior

### For Box Pushing with 2 Agents

**Correct Setup:**
```bash
export INSTRUCTION_ENABLED=1
export OVERCOOKED_INSTRUCTIONS="big_box_spot_0||big_box_spot_1"
```

**Expected Results:**
- Agent 0 learns to choose action 2 (go to big box spot 0)
- Agent 1 learns to choose action 3 (go to big box spot 1)
- High compliance rate (>70%) for both agents
- Action distribution shows: `{2: ~80-90 for agent 0, 3: ~80-90 for agent 1}`

### Example Action Distribution

**Good Learning (Per-Agent):**
```
Action distribution: {0: 5, 1: 8, 2: 85, 3: 90, 4: 5, 5: 3, 6: 2, 7: 2}
                                    ↑ Agent 0    ↑ Agent 1
```
- Action 2 chosen ~85 times (Agent 0 following "big_box_spot_0")
- Action 3 chosen ~90 times (Agent 1 following "big_box_spot_1")

**Poor Learning (If both got same instruction):**
```
Action distribution: {0: 10, 1: 15, 2: 150, 3: 5, 4: 10, 5: 5, 6: 3, 7: 2}
                                     ↑ Both agents!
```
- Both agents choosing action 2, ignoring action 3
- No coordination!

## Troubleshooting

### Both agents choosing the same action?

**Problem**: Both agents have high compliance for action 2, but action 3 is rarely chosen.

**Cause**: Instructions not properly assigned per-agent.

**Check**:
1. Verify startup output shows "PER-AGENT ASSIGNMENT MODE"
2. Check episode statistics show different instructions per agent
3. Ensure 2 instructions are provided for 2 agents

### Compliance low for both agents?

**Problem**: Overall compliance is low (<30%).

**Solutions**:
1. Increase instruction bonus (currently +50)
2. Train longer (compliance improves over time)
3. Check that shaped rewards are in buffer (should be after recent fixes)

### One agent learning, other not?

**Problem**: Agent 0 has high compliance, Agent 1 doesn't.

**Possible Causes**:
1. Reward imbalance - one instruction easier to follow
2. Environment dynamics favor one waypoint
3. Network initialization differences

**Solutions**:
1. Check if both instructions are equally feasible
2. Verify both agents' networks are updating
3. Monitor per-agent compliance separately

## Configuration

### For Box Pushing (2 Agents)

**Recommended:**
```bash
export INSTRUCTION_ENABLED=1
export OVERCOOKED_INSTRUCTIONS="big_box_spot_0||big_box_spot_1"

# Run with per-agent mode
bash experiments/BoxPushing/mac_iac_instructions.sh
```

### For Other Environments

**3+ Agents:**
```bash
export OVERCOOKED_INSTRUCTIONS="inst0||inst1||inst2"  # For 3 agents
```

**Single Shared Instruction:**
```bash
export OVERCOOKED_INSTRUCTIONS="push_box"  # All agents get same instruction
```

## Benefits

1. **Better Coordination**: Each agent learns complementary behaviors
2. **Role Specialization**: Agents can specialize in different sub-tasks
3. **Realistic**: Matches real-world scenarios where team members get different instructions
4. **Flexible**: Automatically switches between per-agent and per-environment modes

## Limitations

1. **Fixed Assignment**: Agent X always gets instruction X (no dynamic switching)
2. **Requires N Instructions**: Per-agent mode needs exactly N instructions for N agents
3. **No Inter-Agent Communication**: Agents don't know what other agents were told

## Summary

Per-agent instructions enable true multi-agent coordination by giving each agent a specific role:

| Mode | When | Behavior |
|------|------|----------|
| **Per-Agent** | # instructions == # agents | Each agent gets fixed instruction |
| **Per-Environment** | # instructions ≠ # agents | All agents share random instruction |

For Box Pushing with 2 agents, use per-agent mode to teach:
- Agent 0: "Go to big box spot 0"
- Agent 1: "Go to big box spot 1"

This enables proper coordination for pushing the big box! 🎉

