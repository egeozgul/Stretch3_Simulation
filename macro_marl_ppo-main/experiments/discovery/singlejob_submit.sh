#!/bin/bash
#SBATCH --job-name=overcooked_sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --partition=short
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --mail-user=lin.wo@northeastern.edu
#SBATCH --mail-type=END,FAIL

# ============================================================================
# Generic single-job driver for Overcooked pg_based sweeps.
# Works for MacIAC, MacCAC, MacIAICC, ACAC (and others) because every algorithm
# now parses OVERCOOKED_INSTRUCTIONS with '||' priority.
#
# Each SLURM job pulls commands from:
#     experiments/discovery/gen_commands/commands_w_args.txt
# and runs up to (cpus_per_task / n_cpus_per_task) training runs concurrently.
# ============================================================================

set -euo pipefail

# ---------------- Conda env ----------------
module load anaconda3/2024.06
source activate /projects/llpr/lin.wo/macro_marl

# ---------------- Thread pinning ----------------
# We run multiple concurrent training processes per job. Without this, each
# PyTorch process spawns 24 BLAS threads and trashes the node.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# ---------------- wandb online mode ----------------
# Force online syncing for sweep jobs (unless user intentionally overrides
# before submission). This avoids silent offline/local-only logging.
unset WANDB_DISABLED
export WANDB_MODE=online

# ---------------- Overcooked instruction config ----------------
# Universally-parsable by all pg_based algs (|| delimiter).
# Honor values inherited from `sbatch --export=ALL,INSTRUCTION_ENABLED=...,
# USE_CHAIN_BREAK=...,USE_VALUE_CANCELLATION=...` (set by
# mac_iac_multijob_submit.sh per variant). Defaults below apply only if the
# script is launched without those overrides (e.g. interactive `bash …`).
export INSTRUCTION_ENABLED="${INSTRUCTION_ENABLED:-1}"
export INSTRUCTION_SWITCH_MODE="${INSTRUCTION_SWITCH_MODE:-stochastic}"
export INSTRUCTION_PROVIDED_PROB="${INSTRUCTION_PROVIDED_PROB:-0.00347}"

# Dual-critic value cancellation + chain-break segmentation.
# Benign segment [0, T-1]: trains V_Psi with r_env, bootstraps with V_Psi.
# Instruct segment [T, end]: trains V_{Psi_delta} with r_env + penalty, bootstraps with V_{Psi_delta}.
# Read by mac_iac / mac_cac / mac_iaicc / acac learners. Set to 0 to disable.
export USE_CHAIN_BREAK="${USE_CHAIN_BREAK:-0}"
export USE_VALUE_CANCELLATION="${USE_VALUE_CANCELLATION:-0}"

# ---------------- Sweep artifact routing ------------------------------------
# Three independent knobs control what gets written to disk and where:
#
#   MARC_DISABLE_POLICY_SAVE  -> skip all save_policies/save_policy/
#                                save_policies_multi calls (the big per-agent
#                                .pt files). One file per agent per run x 400+
#                                runs would blow out $HOME quota, so we leave
#                                these off during sweeps. Wandb has the
#                                learning curves we actually need to pick
#                                hyperparameters.
#
#   MARC_DISABLE_CKPT_SAVE    -> skip all save_checkpoint/save_checkpoint_cent
#                                calls (.tar files containing optimizer state,
#                                episode count, eval returns, env runner RNG).
#                                We KEEP these on so SLURM-preempted or
#                                wall-time-killed jobs can resume from the
#                                last checkpoint instead of restarting.
#
#   MARC_ARTIFACT_ROOT        -> base dir for everything that IS still written
#                                (.tar checkpoints + performance/train/test
#                                pickles created by pg_based_main.py). Points
#                                at /projects/llpr/lin.wo which has the space
#                                for it; $HOME does not.
#
# To keep the .pt policies from a specific winning sweep config, re-launch it
# after the sweep with MARC_DISABLE_POLICY_SAVE=0. MARC_ARTIFACT_ROOT will
# still redirect the output so $HOME stays clean.
export MARC_DISABLE_POLICY_SAVE=1
export MARC_DISABLE_CKPT_SAVE=0
export MARC_ARTIFACT_ROOT=/projects/llpr/lin.wo/marc_sweep_artifacts
mkdir -p "$MARC_ARTIFACT_ROOT/performance" "$MARC_ARTIFACT_ROOT/policy_nns"

# Instructions match experiments/Overcooked/mac_iac.sh (cutting boards + ovens)
OVERCOOKED_INSTRUCTIONS_ARRAY=(
    "don't use the right cutting board"
    "don't use the left cutting board"
    "don't use the right oven"
    "don't use the left oven"
)
export OVERCOOKED_INSTRUCTIONS="$(printf '%s||' "${OVERCOOKED_INSTRUCTIONS_ARRAY[@]}")"

# ---------------- Launch sweep consumer ----------------
echo "[$(date '+%F %T')] Job ${SLURM_JOB_ID:-local} starting on $(hostname)"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK:-?}"
echo "wandb mode:    ${WANDB_MODE:-unset}"
echo "Variant vars:  INSTRUCTION_ENABLED=$INSTRUCTION_ENABLED USE_CHAIN_BREAK=$USE_CHAIN_BREAK USE_VALUE_CANCELLATION=$USE_VALUE_CANCELLATION"
echo "Instructions:  $OVERCOOKED_INSTRUCTIONS"

# Per-variant queue: each variant drains its own commands_<variant>_w_args.txt
# so all three variants can run concurrently. Default keeps prior behavior.
SWEEP_CMD_FILE="${SWEEP_CMD_FILE:-commands_w_args.txt}"
echo "Cmd file:      $SWEEP_CMD_FILE"

# 24 cpus / 8 per task = 3 concurrent training runs per SLURM job.
python run_sweeps_from_cmd_file.py --n-cpus-per-task 8 --cmd-file "$SWEEP_CMD_FILE"

echo "[$(date '+%F %T')] Job ${SLURM_JOB_ID:-local} finished"
