#!/bin/bash

# Run the trained mac_cac policy

# Default values
POLICY_PATH="experiments/BoxPushing/policy_nns/ma_cac_bp6_instructions/0_agent_cen_MacCAC_run_0_['instructions_enabled'].pt"
GRID_DIM="6 6"
N_EPISODES=5
USE_INSTRUCTION=1
INSTRUCTION="don't go to any small box"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --policy)
            POLICY_PATH="$2"
            shift 2
            ;;
        --grid)
            GRID_DIM="$2"
            shift 2
            ;;
        --episodes)
            N_EPISODES="$2"
            shift 2
            ;;
        --no-instruction)
            USE_INSTRUCTION=0
            shift
            ;;
        --instruction)
            INSTRUCTION="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--policy PATH] [--grid \"H W\"] [--episodes N] [--no-instruction] [--instruction TEXT]"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Running Box Pushing mac_cac Policy"
echo "=========================================="
echo "Policy: $(basename "$POLICY_PATH")"
echo "Grid: $GRID_DIM"
echo "Episodes: $N_EPISODES"
if [ "$USE_INSTRUCTION" == "1" ]; then
    echo "Instruction: $INSTRUCTION"
else
    echo "Instruction: DISABLED"
fi
echo "=========================================="
echo ""

cd experiments/BoxPushing

# Run the policy test
PYTHONPATH=/home/willy/macro_marl_ppo/src:$PYTHONPATH python -c "
import sys
sys.path.append('../../visualization')
from test_bp_cac import test

# Set instruction parameters
use_instruction = $USE_INSTRUCTION == 1
instruction_text = \"\$INSTRUCTION\" if use_instruction else \"no instruction\"

print(f'Using instruction: {use_instruction}')
print(f'Instruction text: {instruction_text}')

test(
    policy_path=\"policy_nns/ma_cac_bp6_instructions/0_agent_cen_MacCAC_run_0_['instructions_enabled'].pt\",
    env_id='BP-MA-v0',
    env_terminate_step=100,
    grid_dim=[6, 6],
    n_agent=2,
    n_episode=$N_EPISODES,
    use_instruction=use_instruction,
    instruction_text=\"$INSTRUCTION\"
)"

