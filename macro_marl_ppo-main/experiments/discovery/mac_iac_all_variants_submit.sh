#!/bin/bash
# ============================================================================
# Submit all three MacIAC variants concurrently:
#   1) instr_chainbreak — instructions on, chain-break + value-cancellation on
#   2) instr_nochain    — instructions on, chain-break/value-cancellation off
#   3) vanilla          — instructions off, chain-break/value-cancellation off
#
# Each variant gets its own queue file (gen_commands/<variant>_w_args.txt) so
# all three can run in parallel without pulling each other's commands.
#
# Usage:
#   bash mac_iac_all_variants_submit.sh              # defaults: 16 jobs/variant
#   bash mac_iac_all_variants_submit.sh 16           # 16 jobs per variant (= 48 total)
#   bash mac_iac_all_variants_submit.sh 8            #  8 jobs per variant (= 24 total)
#
# Per-variant queues are regenerated each run (--append False), so re-running
# this script restarts the sweep from scratch for every variant.
# ============================================================================

set -euo pipefail

JOBS_PER_VARIANT="${1:-16}"

VARIANTS=(instr_chainbreak instr_nochain vanilla)
YAMLS=(
  mac_iac_overcooked_chainbreak_20seed
  mac_iac_overcooked_instr_nochain_20seed
  mac_iac_overcooked_vanilla_20seed
)

# ---------------- 1) Generate per-variant queue files ----------------
echo ">>> Generating per-variant queue files"
for i in "${!VARIANTS[@]}"; do
  variant="${VARIANTS[$i]}"
  yaml="${YAMLS[$i]}"
  echo "    [$variant] from sweep_yaml/$yaml.yaml -> gen_commands/${variant}_w_args.txt"
  python generate_sweeps_yaml.py \
      --config-file "$yaml" \
      --append False \
      --out-name "$variant"
done

# ---------------- 2) Submit jobs per variant ----------------
TOTAL=$((JOBS_PER_VARIANT * ${#VARIANTS[@]}))
echo ">>> Submitting $JOBS_PER_VARIANT jobs per variant ($TOTAL total)"
for variant in "${VARIANTS[@]}"; do
  echo "    [$variant]"
  bash mac_iac_multijob_submit.sh "$JOBS_PER_VARIANT" "$variant"
done

cat <<EOF

============================================================================
Submitted $TOTAL jobs across 3 variants ($JOBS_PER_VARIANT per variant).

Monitor:
   squeue -u \$USER
   wc -l gen_commands/{instr_chainbreak,instr_nochain,vanilla}_w_args.txt
   grep "Variant vars" slurm_logs/mac_iac_*.out | sort -u

Cancel everything:
   squeue -u \$USER -h -o '%i' | xargs -r scancel
============================================================================
EOF
