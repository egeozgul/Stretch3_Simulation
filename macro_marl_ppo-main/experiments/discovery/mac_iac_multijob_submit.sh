#!/bin/bash
# ============================================================================
# Launch N SLURM jobs that cooperatively drain commands from:
#   experiments/discovery/gen_commands/commands_w_args.txt
#
# Each submitted job runs mac_iac_singlejob_submit.sh, which spawns multiple
# concurrent training runs internally (3 by default with 24 cpus-per-task /
# 8 cpus-per-run). When a run finishes, the job pulls the next command from
# the shared file (file-locked) and starts it — so adding jobs just adds
# throughput without duplicating work.
#
# Usage:
#   # 1) Regenerate the command file from your sweep YAML
#   python generate_sweeps_yaml.py \
#       --config-file <your_config> --append False
#
#   # 2) Submit jobs (variant controls instruction/chain-break settings)
#   bash mac_iac_multijob_submit.sh 8 instr_chainbreak
#   bash mac_iac_multijob_submit.sh 8 instr_nochain
#   bash mac_iac_multijob_submit.sh 8 vanilla
# ============================================================================

set -euo pipefail

N_JOBS="${1:-8}"
VARIANT="${2:-instr_nochain}"

# Per-variant queue file: <variant>_w_args.txt. Lives alongside the legacy
# commands_w_args.txt in gen_commands/ but is fully isolated from the others.
SWEEP_CMD_FILE="${VARIANT}_w_args.txt"

case "$VARIANT" in
  instr_chainbreak)
    EXPORT_VARS="INSTRUCTION_ENABLED=1,USE_CHAIN_BREAK=1,USE_VALUE_CANCELLATION=1,WANDB_PROJECT=Mac_IAC_Overcooked,SWEEP_CMD_FILE=$SWEEP_CMD_FILE"
    ;;
  instr_nochain)
    EXPORT_VARS="INSTRUCTION_ENABLED=1,USE_CHAIN_BREAK=0,USE_VALUE_CANCELLATION=0,WANDB_PROJECT=Mac_IAC_Overcooked,SWEEP_CMD_FILE=$SWEEP_CMD_FILE"
    ;;
  vanilla)
    EXPORT_VARS="INSTRUCTION_ENABLED=0,USE_CHAIN_BREAK=0,USE_VALUE_CANCELLATION=0,WANDB_PROJECT=Mac_IAC_Overcooked,SWEEP_CMD_FILE=$SWEEP_CMD_FILE"
    ;;
  *)
    echo "ERROR: Unknown variant '$VARIANT' (use instr_chainbreak|instr_nochain|vanilla)." >&2
    exit 1
    ;;
esac

CMD_FILE="gen_commands/$SWEEP_CMD_FILE"
if [[ ! -s "$CMD_FILE" ]]; then
  echo "ERROR: $CMD_FILE is missing or empty." >&2
  echo "Run: python generate_sweeps_yaml.py --config-file mac_iac_overcooked_${VARIANT}_20seed --append False --out-name $VARIANT" >&2
  exit 1
fi

N_CMDS=$(wc -l < "$CMD_FILE")
echo "Submitting $N_JOBS SLURM jobs ($VARIANT) to consume $N_CMDS commands from $CMD_FILE"

mkdir -p slurm_logs

for j in $(seq 1 "$N_JOBS"); do
  sbatch \
    --job-name="mac_iac_${j}" \
    --output="slurm_logs/mac_iac_${j}_%j.out" \
    --export=ALL,"$EXPORT_VARS" \
    ./mac_iac_singlejob_submit.sh
done

echo "Done. Monitor with:  squeue -u \$USER"
echo "Kill all with:       scancel --name=mac_iac_<n> \$USER   (per job)"
