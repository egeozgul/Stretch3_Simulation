#!/bin/bash
#SBATCH --job-name=mac_iac_overcooked
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
# mac_iac Overcooked (Task 9, Map D) — dual-actor / dual-critic training
# Each SLURM job pulls commands from experiments/discovery/gen_commands/commands_w_args.txt
# and runs up to (cpus_per_task / n_cpus_per_task) training runs concurrently.
# ============================================================================

set -euo pipefail

# ---------------- Conda env ----------------
module load anaconda3/2024.06
source activate /projects/llpr/lin.wo/macro_marl

# ---------------- Critical: limit BLAS/OpenMP threads ----------------
# We run multiple concurrent training processes per job (3 by default). If each
# spawns unlimited OMP/MKL threads they trample each other on the CPU. Keep it
# single-threaded at the BLAS layer so torch + env subprocesses play nicely.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
# Prevent tokenizer from forking warnings (BERT embeddings in instruction path)
export TOKENIZERS_PARALLELISM=false

# ---------------- Overcooked instruction config ----------------
# These must match the env-var names read by src/macro_marl/algs/pg_based/mac_iac.py
# Defaults can be overridden via sbatch --export or inline env vars.
: "${INSTRUCTION_ENABLED:=1}"
: "${INSTRUCTION_SWITCH_MODE:=stochastic}"
: "${INSTRUCTION_PROVIDED_PROB:=0.00347}"

# Dual-critic value cancellation + chain-break segmentation.
# Set either to 1 to enable. Defaults keep standard per-step bootstrap selection.
: "${USE_CHAIN_BREAK:=0}"
: "${USE_VALUE_CANCELLATION:=0}"

export INSTRUCTION_ENABLED
export INSTRUCTION_SWITCH_MODE
export INSTRUCTION_PROVIDED_PROB
export USE_CHAIN_BREAK
export USE_VALUE_CANCELLATION

# Instructions match experiments/Overcooked/mac_iac.sh (|| delimiter) unless overridden.
if [[ -z "${OVERCOOKED_INSTRUCTIONS:-}" ]]; then
    OVERCOOKED_INSTRUCTIONS_ARRAY=(
            "don't use the right cutting board"
            "get tomato"
            "go to the right"
    )
    export OVERCOOKED_INSTRUCTIONS="$(printf '%s||' "${OVERCOOKED_INSTRUCTIONS_ARRAY[@]}")"
fi

# ---------------- Launch sweep consumer ----------------
echo "[$(date '+%F %T')] Job ${SLURM_JOB_ID:-local} starting on $(hostname)"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK:-?}"
echo "Instructions:  $OVERCOOKED_INSTRUCTIONS"

# 24 cpus / 8 per task = 3 concurrent training runs per SLURM job.
# Raise --n-cpus-per-task to 12 for bigger n_env, or lower to 6 for tighter packing.
python run_sweeps_from_cmd_file.py --n-cpus-per-task 8

echo "[$(date '+%F %T')] Job ${SLURM_JOB_ID:-local} finished"
