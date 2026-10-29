#!/bin/bash
# Sensor re-run, step 2 (2026-10-29): the 24 test + profile scenarios, two noise seeds.
# Scenarios: S012 S013 S014 S019 S020 S024 S032 S034 S035 (rows 12,13,14,19,20,24,32,34,35) and H01-H15 (rows 127-141).
# Sensor: read at EVERY 10 ms simulation step, delay 50 ns (= 0 steps), 1 s moving average of the readings,
# staged limiter, network caps (staged_cap_groups_nn.csv). Offsets (kW added to all six caps, most permissive first,
# stop at the first safe one): 15:10:5:0:-5:-10.
# Four job arrays (tag, what):
#   sr_b05_w1s      0.5 % bounded (Uniform), seed 1.  H01 and H09 were done in submit_sensor_rerun.sh, so rows 127 and 135 are left out.
#   sr_b05_w1s_s2   0.5 % bounded (Uniform), seed 2.  All 24 scenarios.
#   sr_g10_w1s      10 % Gaussian, seed 1.            Rows 127 and 135 left out (already done).
#   sr_g10_w1s_s2   10 % Gaussian, seed 2.            All 24 scenarios.
# 22 + 24 + 22 + 24 = 92 tasks, each up to 6 simulations of about 11 minutes. At most 6 tasks per array run at once.
# Run from the project root:   bash scripts/09_dataset_pipeline/submit_sensor_rerun2.sh
# Then:  python3 scripts/09_dataset_pipeline/summarize_sensor_rerun2.py     (after  module load python/3.11.5)
set -euo pipefail
D=scripts/09_dataset_pipeline
SCRIPT=${D}/run_staged_array.sh
grep -q MEAS_MODE "$SCRIPT" || { echo "run_staged_array.sh has no MEAS_MODE option: git pull first." >&2; exit 1; }
grep -q meas_noise_mode ${D}/run_staged_limiter.m || { echo "run_staged_limiter.m has no meas_noise_mode option: git pull first." >&2; exit 1; }

# warn (do not stop) if a scenario's earlier outputs are missing: its task would just fail and be skipped by the summary
ROOT="${DATASET_OUTPUT_ROOT:-model_outputs/dataset_pipeline}"
for ID in S012 S013 S014 S019 S020 S024 S032 S034 S035 H01 H02 H03 H04 H05 H06 H07 H08 H09 H10 H11 H12 H13 H14 H15; do
  if [ ! -f "${ROOT}/${ID}/noshed_${ID}_collapse.csv" ] || [ ! -f "${ROOT}/${ID}/detector_triggers_${ID}.json" ]; then
    echo "WARNING: ${ID}: no-shed collapse file or detector-trigger file not found in ${ROOT}/${ID}"
  fi
done

mkdir -p logs
ROWS_ALL="12,13,14,19,20,24,32,34,35,127-141"
ROWS_S1="12,13,14,19,20,24,32,34,35,128,129,130-134,136-141"
BASE="STAGED=1,GROUPS_CSV=staged_cap_groups_nn.csv,OFFSETS=15:10:5:0:-5:-10,MEAS_PERIOD=0.01,MEAS_DELAY=0.00000005,MEAS_FILTER=1"
sub() { J=$(sbatch --parsable --array="$2%6" --export=ALL,${BASE},TAG="$1",$3 "$SCRIPT"); printf "%-14s job %s   (rows %s; %s)\n" "$1" "$J" "$2" "$3"; }

sub sr_b05_w1s    "${ROWS_S1}"  "MEAS_NOISE=0.5,MEAS_MODE=uniform,MEAS_SEED=1"
sub sr_b05_w1s_s2 "${ROWS_ALL}" "MEAS_NOISE=0.5,MEAS_MODE=uniform,MEAS_SEED=2"
sub sr_g10_w1s    "${ROWS_S1}"  "MEAS_NOISE=10,MEAS_SEED=1"
sub sr_g10_w1s_s2 "${ROWS_ALL}" "MEAS_NOISE=10,MEAS_SEED=2"
echo "Watch: squeue -u \$USER    First check (once a task has started):  grep Options logs/staged_<jobid>_12.log"
