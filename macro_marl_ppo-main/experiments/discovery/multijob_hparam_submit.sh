#!/bin/bash
# ============================================================================
# Per-algorithm hyperparameter sweep — Discovery driver
#
# Launches the full hparam grid for some or all of:
#   MacIAC, MacCAC, MacIAICC, ACAC
# using the 4 per-alg sweep YAMLs in sweep_yaml/. Each YAML is a 108-run grid
# (a_lr x c_lr x n_step_TD x train_freq x 3 seeds). All four = 432 runs.
#
# Usage:
#   bash multijob_hparam_submit.sh                        # all 4 algs, 16 jobs
#   bash multijob_hparam_submit.sh 24                     # all 4 algs, 24 jobs
#   bash multijob_hparam_submit.sh 8 mac_iac              # single alg
#   bash multijob_hparam_submit.sh 16 mac_iac mac_cac     # subset
#
# Alg names (after the job count): mac_iac mac_cac mac_iaicc mac_acac
# ============================================================================

set -euo pipefail

N_JOBS="${1:-16}"
shift || true

# Map alg short names to YAML basenames
declare -A YAML_MAP=(
  # [mac_iac]="mac_iac_overcooked_sweep"
  [mac_cac]="mac_cac_overcooked_sweep"
  [mac_iaicc]="mac_iaicc_overcooked_sweep"
  [mac_acac]="mac_acac_overcooked_sweep"
)

# Default: all 4 algs
if (( $# == 0 )); then
  ALGS=(mac_iac mac_cac mac_iaicc mac_acac)
else
  ALGS=("$@")
fi

CMD_FILE="gen_commands/commands_w_args.txt"

# ---------------- 1) Validate alg names ----------------
for alg in "${ALGS[@]}"; do
  if [[ -z "${YAML_MAP[$alg]:-}" ]]; then
    echo "ERROR: unknown alg '$alg'. Valid: ${!YAML_MAP[*]}" >&2
    exit 2
  fi
done

# ---------------- 2) Regenerate command queue ----------------
echo ">>> Clearing $CMD_FILE"
rm -f "$CMD_FILE"

for alg in "${ALGS[@]}"; do
  yaml="${YAML_MAP[$alg]}"
  echo ">>> Generating commands for $alg -> $yaml.yaml"
  python generate_sweeps_yaml.py --config-file "$yaml" --append True
done

if [[ ! -s "$CMD_FILE" ]]; then
  echo "ERROR: $CMD_FILE is empty after generation." >&2
  exit 1
fi

N_CMDS=$(wc -l < "$CMD_FILE")
echo ">>> Total commands queued: $N_CMDS  (algs: ${ALGS[*]})"

# ---------------- 3) Submit N SLURM jobs ----------------
mkdir -p slurm_logs
for j in $(seq 1 "$N_JOBS"); do
  sbatch \
    --job-name="hparam_${j}" \
    --output="slurm_logs/hparam_${j}_%j.out" \
    ./singlejob_submit.sh
done

cat <<EOF

============================================================================
Submitted: $N_JOBS jobs for $N_CMDS commands.
   Effective parallelism: up to $((N_JOBS * 3)) simultaneous training runs
   (24 cpus-per-task / 8 cpus-per-run = 3 slots per job).

Monitor:
   squeue -u \$USER
   tail -f slurm_logs/hparam_1_*.out

Results per config:
   performance/<save_dir>__alg-<Alg>_a_lr-<lr>_c_lr-<lr>_..._run-<seed>/

Cancel all:
   squeue -u \$USER -h -n hparam_1 -o '%i' | xargs -r scancel
============================================================================
EOF
