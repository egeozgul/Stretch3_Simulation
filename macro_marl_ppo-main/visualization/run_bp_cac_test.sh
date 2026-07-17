#!/bin/bash

# Test Box Pushing mac_cac policy

# Default values
POLICY_PATH="../experiments/BoxPushing/policy_nns/ma_cac_bp6_instructions/0_agent_cen_MacCAC_run_0_['instructions_enabled'].pt"
GRID_DIM="6 6"
N_EPISODES=5
USE_INSTRUCTION="--use_instruction"
INSTRUCTION="--instruction \"don't go to any small box\""

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
        --n_episode)
            N_EPISODES="$2"
            shift 2
            ;;
        --no-instruction)
            USE_INSTRUCTION="--no_instruction"
            shift
            ;;
        --instruction)
            # Store the instruction text as-is (with quotes)
            INSTRUCTION="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--policy PATH] [--grid \"H W\"] [--episodes N] [--n_episode N] [--no-instruction] [--instruction TEXT]"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Testing Box Pushing mac_cac Policy"
echo "=========================================="
echo "Policy: $POLICY_PATH"
echo "Grid: $GRID_DIM"
echo "Episodes: $N_EPISODES"
if [ -n "$USE_INSTRUCTION" ] && [ "$USE_INSTRUCTION" != "--no_instruction" ]; then
    echo "Instruction: $INSTRUCTION"
else
    echo "Instruction: DISABLED"
fi
echo "=========================================="
echo ""

cd "$(dirname "$0")"

if [ -n "$USE_INSTRUCTION" ] && [ "$USE_INSTRUCTION" != "--no_instruction" ]; then
    python test_bp_cac.py \
        --policy_path "$POLICY_PATH" \
        --grid_dim $GRID_DIM \
        --n_episode $N_EPISODES \
        $USE_INSTRUCTION \
        --instruction "$INSTRUCTION"
else
    python test_bp_cac.py \
        --policy_path "$POLICY_PATH" \
        --grid_dim $GRID_DIM \
        --n_episode $N_EPISODES \
        $USE_INSTRUCTION
fi

