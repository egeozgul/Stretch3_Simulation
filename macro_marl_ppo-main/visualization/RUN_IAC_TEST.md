# Running MAC_IAC Overcooked Test

## Quick Start

Your MAC_IAC policies are located at:
```
/home/willy/Documents/macro_marl_ppo/experiments/Overcooked/policy_nns/mac_iac_overcooked_A_laptop_iac_sleeper/
```

The test script will automatically find and load these policies!

## Run Tests

### Watch 2 AI Agents Play Together

```bash
cd /home/willy/Documents/macro_marl_ppo/visualization

# Map A with 2 agents (default)
python test_overcooked_iac.py --mapType A

# Slower speed (easier to watch)
python test_overcooked_iac.py --mapType A --render_delay 0.3

# Faster speed
python test_overcooked_iac.py --mapType A --render_delay 0.05
```

### Human Control Mode (Play with AI)

Control one agent yourself while the other agent uses the trained policy:

```bash
# You control Agent 0, Agent 1 uses AI
python test_overcooked_iac.py --mapType A --human_agent_idx 0

# You control Agent 1, Agent 0 uses AI
python test_overcooked_iac.py --mapType A --human_agent_idx 1
```

### Using Instructions

Press `T` during the game to enter instruction mode:
1. Game pauses
2. Type your instruction (e.g., "get tomato", "deliver salad")
3. Press `ENTER` to confirm (instruction persists for 300 steps)
4. Press `ESC` to cancel

Example instructions:
- `"get tomato"`
- `"get lettuce"`
- `"deliver the salad"`
- `"go to the knife"`
- `"chop vegetables"`
- `"don't touch the onion"`

## Available Policies

The script found these trained policies:

```bash
ls experiments/Overcooked/policy_nns/mac_iac_overcooked_A_laptop_iac_sleeper/0_agent_*.pt
```

Output:
- `0_agent_0.pt` - Agent 0's policy (latest)
- `0_agent_1.pt` - Agent 1's policy (latest)
- `0_agent_0_ep3000-4000.pt` - Agent 0's policy (checkpoint from episodes 3000-4000)
- `0_agent_1_ep3000-4000.pt` - Agent 1's policy (checkpoint from episodes 3000-4000)

By default, the script loads the latest policies (`0_agent_0.pt` and `0_agent_1.pt`).

## Command Line Options

```bash
python test_overcooked_iac.py [OPTIONS]

Options:
  --mapType A|B|C         Map type (default: A)
  --n_agent N             Number of agents (default: 2, auto-detects from policies)
  --human_agent_idx N     Human controls agent N (0 or 1), None = all AI (default: None)
  --render_delay FLOAT    Delay between frames in seconds (default: 0.15)
                          Lower = faster, 0.001 = max speed
  --p_id N                Policy run ID to load (default: 0)
  --task N                Task type: 3 or 6 (default: 6)
  --grid_dim H W          Grid dimensions (default: 7 7)
```

## Keyboard Controls (Human Mode Only)

### Macro-Actions
- `0` or `Space`: Stay
- `1`: Get Tomato
- `2`: Get Lettuce
- `3`: Get Onion
- `4`: Get Plate 1
- `5`: Get Plate 2
- `6`: Go to Knife 1
- `7`: Go to Knife 2
- `8` or `D`: Deliver
- `9` or `C`: Chop

### Movement
- Arrow Keys: Move up/down/left/right

### Special
- `T`: Enter instruction mode (pauses game)

## How the Script Works

The test script (`test_overcooked_iac.py`) automatically:

1. **Finds Policies**: Looks in the training output directory:
   ```
   experiments/Overcooked/policy_nns/mac_iac_overcooked_A_laptop_iac_sleeper/
   ```

2. **Detects Available Agents**: Scans for policy files matching:
   - `{p_id}_agent_{idx}.pt` (e.g., `0_agent_0.pt`)
   - `stochastic_policy_agent_{idx}.pt` (alternative naming)

3. **Auto-Adjusts**: If you specify `--n_agent 3` but only 2 policies exist, it automatically adjusts to 2 agents

4. **Loads Models**: Each agent gets its own independent policy network with instruction support

## Using Different Policy Checkpoints

To test the checkpoint from episodes 3000-4000:

```bash
# This requires renaming the checkpoint files temporarily
cd experiments/Overcooked/policy_nns/mac_iac_overcooked_A_laptop_iac_sleeper/

# Backup current files
mv 0_agent_0.pt 0_agent_0_latest_backup.pt
mv 0_agent_1.pt 0_agent_1_latest_backup.pt

# Use checkpoint
cp 0_agent_0_ep3000-4000.pt 0_agent_0.pt
cp 0_agent_1_ep3000-4000.pt 0_agent_1.pt

# Run test
cd ../../../../visualization
python test_overcooked_iac.py --mapType A

# Restore latest (afterwards)
cd ../experiments/Overcooked/policy_nns/mac_iac_overcooked_A_laptop_iac_sleeper/
mv 0_agent_0_latest_backup.pt 0_agent_0.pt
mv 0_agent_1_latest_backup.pt 0_agent_1.pt
```

## Using Custom Policy Directory

Set environment variable to use a different policy directory:

```bash
export MAC_IAC_POLICY_DIR="/path/to/your/policies"
python test_overcooked_iac.py --mapType A
```

## Troubleshooting

### Error: No policy files found

**Check that policies exist:**
```bash
ls experiments/Overcooked/policy_nns/mac_iac_overcooked_A_laptop_iac_sleeper/*.pt
```

**Solution**: The script will show what files it's looking for and what files are available in the directory.

### Error: Dimension mismatch

The script is configured for:
- `input_dim=56` (observation size)
- `output_dim=169` (action space size)

If your trained models used different dimensions, you'll need to update these values in the script.

### Error: TypeError: argument of type 'Actor' is not iterable

This error occurred because policies were saved as complete model objects rather than state dictionaries.

**Solution**: The test script now automatically handles both formats:
- Full model objects (saved with `torch.save(agent.actor_net, PATH)`)
- State dictionaries (saved with `torch.save(agent.actor_net.state_dict(), PATH)`)

Your current policies use the full model format, which is now supported!

### Only 2 Agents Available

Your current training saved policies for 2 agents (Agent 0 and Agent 1), not 3. This is fine! The test script automatically detects and works with 2 agents.

If you want 3 agents:
1. Update training to use 3 agents in the environment
2. Ensure all 3 agent policies are saved during training
3. Check that `save_policies()` is called correctly in the training code

## Performance Tips

- **Slower for watching**: `--render_delay 0.3`
- **Default speed**: `--render_delay 0.15`
- **Faster**: `--render_delay 0.05`
- **Maximum speed**: `--render_delay 0.001`

## Example Session

```bash
$ cd /home/willy/Documents/macro_marl_ppo/visualization
$ python test_overcooked_iac.py --mapType A --human_agent_idx 0

Using policy directory from default: ../experiments/Overcooked/policy_nns/mac_iac_overcooked_A_laptop_iac_sleeper
Loading 2 separate IAC policies from: ../experiments/Overcooked/policy_nns/mac_iac_overcooked_A_laptop_iac_sleeper
Models configured with use_instructions=True

Found 2 agent policies:
  Agent 0: 0_agent_0.pt
  Agent 1: 0_agent_1.pt

Environment observation sizes: [56, 56]
Environment action sizes: [14, 14]

============================================================
      HUMAN CONTROL MODE (PAUSE) - Map A
============================================================
Agent 0 is human-controlled (you!)
Agent 1 uses trained policy

[Controls listed...]
[Game starts]
```

Enjoy testing your MAC_IAC policies! 🎮

