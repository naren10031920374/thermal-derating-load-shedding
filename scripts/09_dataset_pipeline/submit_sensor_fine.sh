#!/bin/bash
# Finer offset grid (1 kW steps) on the 10 % noise + 1 s average result (2026-10-29), noise seed 1, 24 test + profile scenarios.
# The coarse run (tag sr_g10_w1s) found the largest safe offset on a 5 kW grid. This run looks inside the last step.
# The scenarios are sorted by reading their coarse result, not typed in:
#   coarse offset +5  : +10 failed and +5 was safe  -> try +9, +8, +7, +6   (stops at the first safe one)
#   coarse offset +10 : +15 failed and +10 was safe -> try +14, +13, +12, +11
# If none of the four is safe, the limit is just above the coarse offset (shown in the summary).
# Margin 0 = offset +5. A fine offset of +8 means the cap could be 3 kW higher than margin 0 and still be safe.
# Sensor: 10 % Gaussian noise, read at EVERY 10 ms step, delay 50 ns (= 0 steps), 1 s average, staged limiter, network caps.
# Same tag for both arrays (they cover different scenarios): sr_fine_g10_w1s.  Up to 24 tasks, each up to 4 simulations of about 11 minutes.
# Run from the project root:   bash scripts/09_dataset_pipeline/submit_sensor_fine.sh
# Then:  python3 scripts/09_dataset_pipeline/summarize_sensor_next.py     (after  module load python/3.11.5)
set -euo pipefail
D=scripts/09_dataset_pipeline
SCRIPT=${D}/run_staged_array.sh
grep -q MEAS_MODE "$SCRIPT" || { echo "run_staged_array.sh has no MEAS_MODE option: git pull first." >&2; exit 1; }
ROOT="${DATASET_OUTPUT_ROOT:-model_outputs/dataset_pipeline}"

get_off() {   # prints first_safe_offset_kw of the coarse 10 % + 1 s result of scenario $1 (nothing if there is no result)
  local f="${ROOT}/$1/staged_$1_sr_g10_w1s_result.json"
  [ -f "$f" ] || return 0
  grep -o '"first_safe_offset_kw" *: *-\?[0-9.]*' "$f" | head -1 | sed 's/.*: *//'
}

ROWS_P=""; ROWS_Q=""
for PAIR in S012:12 S013:13 S014:14 S019:19 S020:20 S024:24 S032:32 S034:34 S035:35 \
            H01:127 H02:128 H03:129 H04:130 H05:131 H06:132 H07:133 H08:134 H09:135 H10:136 H11:137 H12:138 H13:139 H14:140 H15:141; do
  ID=${PAIR%%:*}; ROW=${PAIR##*:}
  OFF=$(get_off "$ID" || true)
  case "$OFF" in
    5|5.0|5.00)    ROWS_P="${ROWS_P:+${ROWS_P},}${ROW}" ;;
    10|10.0|10.00) ROWS_Q="${ROWS_Q:+${ROWS_Q},}${ROW}" ;;
    "")            echo "SKIPPED ${ID} (row ${ROW}): no coarse 10 % + 1 s result (tag sr_g10_w1s) found" ;;
    *)             echo "SKIPPED ${ID} (row ${ROW}): coarse offset is ${OFF}, not +5 or +10" ;;
  esac
done
echo "coarse offset +5  (rows): ${ROWS_P:-none}"
echo "coarse offset +10 (rows): ${ROWS_Q:-none}"

mkdir -p logs
BASE="STAGED=1,GROUPS_CSV=staged_cap_groups_nn.csv,MEAS_PERIOD=0.01,MEAS_DELAY=0.00000005,MEAS_FILTER=1,MEAS_NOISE=10,MEAS_SEED=1,TAG=sr_fine_g10_w1s"
if [ -n "${ROWS_P}" ]; then
  J1=$(sbatch --parsable --array="${ROWS_P}%4" --export=ALL,${BASE},OFFSETS=9:8:7:6 "$SCRIPT")
  echo "fine grid +9..+6    job ${J1}   (rows ${ROWS_P})"
fi
if [ -n "${ROWS_Q}" ]; then
  J2=$(sbatch --parsable --array="${ROWS_Q}%4" --export=ALL,${BASE},OFFSETS=14:13:12:11 "$SCRIPT")
  echo "fine grid +14..+11  job ${J2}   (rows ${ROWS_Q})"
fi
echo "Watch: squeue -u \$USER    First check (once a task has started):  grep Options logs/staged_<jobid>_<first row>.log"
