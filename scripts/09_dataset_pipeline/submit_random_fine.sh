#!/bin/bash
# ============================================================================
# Fine test on the 30 pilot scenarios (RT001-RT030): a 1 kW ladder inside each known 5 kW bracket,
# noise seeds 1 and 2. Only the SWEEP step is needed: the no-shed runs and detector files exist.
#
# Bracket of each scenario = the pilot result (tag rnd_g10_w1s, seed 1), read from its result file:
#   pilot first safe offset +5  (+10 collapsed, +5 safe)   -> try +9, +8, +7, +6   (stops at the first safe one)
#   pilot first safe offset +10 (+15 collapsed, +10 safe)  -> try +14, +13, +12, +11
# Seed 2 has no pilot result, so its ladder also includes the bottom of the bracket (+5 or +10), to see
# if the limit with seed 2 falls BELOW the bracket.
# Offset +5 = margin 0 (caps at the network's own prediction). Same sensor as the pilot:
# 10 % Gaussian noise, read every 10 ms, delay 50 ns, 1 s average, staged limiter, network caps.
#
# Four job arrays, one after the other (each waits for the one before it), so only MAXJOBS (default 3)
# simulations run at the same time in total. Fair to the other cluster users.
# Tags: rnd_fine_g10_w1s_s1 and rnd_fine_g10_w1s_s2.
# About 2.5-3 simulations of 11 min per task: expect roughly 12-14 hours in total.
#
# Run from the project root on the cluster (after git pull):
#   DRY=1 bash scripts/09_dataset_pipeline/submit_random_fine.sh     # only print the sbatch lines
#   bash scripts/09_dataset_pipeline/submit_random_fine.sh
# Options: FIRST=1 LAST=30 (which RT numbers), MAXJOBS=3
# Then:  python3 scripts/09_dataset_pipeline/summarize_random_fine.py
# ============================================================================
set -euo pipefail
D=scripts/09_dataset_pipeline
SCRIPT=${D}/run_staged_array.sh
ROOT="${DATASET_OUTPUT_ROOT:-model_outputs/dataset_pipeline}"
MAXJOBS="${MAXJOBS:-3}"
FIRST="${FIRST:-1}"
LAST="${LAST:-30}"
DRY="${DRY:-0}"
COARSE_TAG="rnd_g10_w1s"

grep -q MEAS_MODE "$SCRIPT" || { echo "run_staged_array.sh has no MEAS_MODE option: git pull first." >&2; exit 1; }
grep -q "^RT030," ${D}/scenario_table.csv || { echo "scenario_table.csv has no RT030: git pull first." >&2; exit 1; }
mkdir -p logs

row_of() {   # table row number (SLURM array id) of scenario $1
  awk -F, -v id="$1" 'NR>1 && $1==id {print NR-1}' ${D}/scenario_table.csv
}
off_of() {   # first_safe_offset_kw of the pilot result of scenario $1 (nothing if there is no result)
  local f="${ROOT}/$1/staged_$1_${COARSE_TAG}_result.json"
  [ -f "$f" ] || return 0
  grep -o '"first_safe_offset_kw" *: *-\?[0-9.]*' "$f" | head -1 | sed 's/.*: *//'
}

ROWS_P=""; ROWS_Q=""
for N in $(seq "$FIRST" "$LAST"); do
  ID=$(printf "RT%03d" "$N")
  ROW=$(row_of "$ID")
  [ -n "$ROW" ] || { echo "SKIPPED ${ID}: not in scenario_table.csv"; continue; }
  OFF=$(off_of "$ID" || true)
  case "$OFF" in
    5|5.0|5.00)    ROWS_P="${ROWS_P:+${ROWS_P},}${ROW}" ;;
    10|10.0|10.00) ROWS_Q="${ROWS_Q:+${ROWS_Q},}${ROW}" ;;
    "")            echo "SKIPPED ${ID} (row ${ROW}): no pilot result (tag ${COARSE_TAG}) found" ;;
    *)             echo "SKIPPED ${ID} (row ${ROW}): pilot offset is ${OFF}, not +5 or +10" ;;
  esac
done
echo "pilot offset +5  (rows): ${ROWS_P:-none}"
echo "pilot offset +10 (rows): ${ROWS_Q:-none}"

PREV=""
submit() {   # $1 label  $2 rows  $3 seed  $4 offsets
  local label="$1" rows="$2" seed="$3" offs="$4"
  [ -n "$rows" ] || return 0
  local base="STAGED=1,GROUPS_CSV=staged_cap_groups_nn.csv,MEAS_PERIOD=0.01,MEAS_DELAY=0.00000005,MEAS_FILTER=1,MEAS_NOISE=10,MEAS_SEED=${seed},TAG=rnd_fine_g10_w1s_s${seed},OFFSETS=${offs}"
  local dep=""
  [ -n "$PREV" ] && dep="--dependency=afterany:${PREV}"
  local cmd="sbatch --parsable --array=${rows}%${MAXJOBS} --cpus-per-task=2 --mem=32G ${dep} --export=ALL,${base} ${SCRIPT}"
  if [ "$DRY" = "1" ]; then
    echo "[dry run] ${label}: ${cmd}"
    PREV="<previous job>"
  else
    PREV=$(${cmd})
    printf "%-26s job %s   (rows %s; offsets %s)\n" "$label" "$PREV" "$rows" "$offs"
  fi
}

submit "seed 1, bracket 5-10"   "$ROWS_P" 1 "9:8:7:6"
submit "seed 1, bracket 10-15"  "$ROWS_Q" 1 "14:13:12:11"
submit "seed 2, bracket 5-10"   "$ROWS_P" 2 "9:8:7:6:5"
submit "seed 2, bracket 10-15"  "$ROWS_Q" 2 "14:13:12:11:10"
echo
echo "Watch: squeue -u \$USER    Options check (once a task has started):  grep Options logs/staged_<jobid>_<first row>.log"
echo "Stop all: scancel <the four job ids>   (finished simulations are kept)"
