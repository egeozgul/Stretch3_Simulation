#!/bin/bash
# ============================================================================
# All-Algorithm Multijob Submit
#
# Runs the pg_based sweep across MacIAC, MacCAC, MacIAICC, and ACAC on the
# same Overcooked env config (task 9 / map D / n_agent 2) so results are
# directly comparable.
#
# What this script does:
#   1) (Re)generates experiments/discovery/gen_commands/commands_w_args.txt
#      from sweep_yaml/all_algs_overcooked_sweep.yaml (12 commands by default:
#      4 algs x 3 seeds).
#   2) Submits N SLURM jobs of ./singlejob_submit.sh. Jobs cooperatively pull
#      commands via file lock, so any # >= 1 works correctly; more just means
#      more parallelism. When a run finishes, the job grabs the next command.
#
# Usage:
#   bash multijob_submit.sh                    # defaults: 1 YAML, 4 SLURM jobs
#   bash multijob_submit.sh 8                  # override job count
#   bash multijob_submit.sh 4 my_sweep another # append extra YAMLs (basename,
#                                              # no .yaml), merging commands
# ============================================================================

set -euo pipefail

N_JOBS="${1:-4}"
shift || true
EXTRA_YAMLS=("$@")   # optional additional sweep YAML basenames

DEFAULT_YAML="all_algs_overcooked_sweep"
CMD_FILE="gen_commands/commands_w_args.txt"

# ---------------- 1) Regenerate command file ----------------
echo ">>> Regenerating $CMD_FILE from sweep YAMLs"
rm -f "$CMD_FILE"

# Default: the 4-alg merged sweep
python generate_sweeps_yaml.py \
    --config-file "$DEFAULT_YAML" --append False

# Optional: append extra per-alg or tuning YAMLs onto the same queue
for yaml_name in "${EXTRA_YAMLS[@]}"; do
  echo ">>> Appending $yaml_name.yaml"
  python generate_sweeps_yaml.py \
      --config-file "$yaml_name" --append True
done

if [[ ! -s "$CMD_FILE" ]]; then
  echo "ERROR: $CMD_FILE is empty after generation. Check sweep YAML(s)." >&2
  exit 1
fi

N_CMDS=$(wc -l < "$CMD_FILE")
echo ">>> $N_CMDS commands queued. Submitting $N_JOBS SLURM jobs."

# ---------------- 2) Submit N SLURM jobs ----------------
mkdir -p slurm_logs

for j in $(seq 1 "$N_JOBS"); do
  sbatch \
    --job-name="overcooked_all_${j}" \
    --output="slurm_logs/overcooked_all_${j}_%j.out" \
    ./singlejob_submit.sh
done

cat <<EOF

============================================================================
Submitted: $N_JOBS jobs, $N_CMDS commands total.
   Each SLURM job runs up to 3 training procs concurrently (24 cpus / 8 each).
   Effective parallelism: up to $((N_JOBS * 3)) simultaneous training runs.

Monitor:
   squeue -u \$USER
   tail -f slurm_logs/overcooked_all_1_*.out

Cancel all:
   scancel --name=overcooked_all_1 \$USER   # per job index
   # or: squeue -u \$USER -h -o '%i' | xargs -r scancel

Results land in:
   performance/all_algs_overcooked__alg-<AlgName>_.../run_<0..N>/
============================================================================
EOF
