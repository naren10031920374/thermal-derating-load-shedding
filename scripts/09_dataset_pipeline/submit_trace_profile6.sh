#!/bin/bash
# Trace runs for the six new load-pattern scenarios (bus voltages and load over time, same figure style as the
# sensor-margin figures). 17 single simulations of the staged limiter (network caps) with SAVE_TRACE=1,
# one cap offset per run, same sensor as the pilot and the fine test:
#   10% Gaussian noise, 1 s moving average, 10 ms read, 50 ns delay, noise seed 1.
# Offset +5 = margin 0.  Every offset below was already run in the pilot or the fine test (same sensor, seed 1),
# so the outcome of each panel is known in advance (S = safe, C = collapse):
#   RT051 (row 192): +5 S   +7 S   +8 C          RT057 (row 198): +5 S   +6 C   +10 C
#   RT071 (row 212): +5 S   +9 S   +10 C         RT041 (row 182): +5 S   +10 S  +11 C
#   RC034 (row 275): +5 S   +10 S  +15 C         RC007 (row 248): +5 S   +10 C
# One array job per offset, 3 tasks at a time, the arrays run one after the other (afterany).
# Run from the project root:   bash scripts/09_dataset_pipeline/submit_trace_profile6.sh
# Then:  python3 scripts/09_dataset_pipeline/make_profile6_graphs.py --dir model_outputs/dataset_pipeline --out profile6_graphs
# Options: DRY=1 (print the sbatch lines, submit nothing)
set -euo pipefail
D=scripts/09_dataset_pipeline
grep -q SAVE_TRACE ${D}/run_staged_array.sh || { echo "run_staged_array.sh has no SAVE_TRACE option: git pull first." >&2; exit 1; }
grep -q "^RT041," ${D}/scenario_table.csv   || { echo "scenario_table.csv has no random rows: git pull first." >&2; exit 1; }
mkdir -p logs
SCRIPT=${D}/run_staged_array.sh
BASE="STAGED=1,GROUPS_CSV=staged_cap_groups_nn.csv,SAVE_TRACE=1"
SENSOR="MEAS_NOISE=10,MEAS_FILTER=1,MEAS_PERIOD=0.01,MEAS_DELAY=0.00000005,MEAS_SEED=1"
RES="--cpus-per-task=2 --mem=32G"
PREV=""
sub() {   # offset rows
  local OFF="$1" ROWS="$2" TAGN="tr_p6_o$1" DEP=""
  [ -n "$PREV" ] && DEP="--dependency=afterany:${PREV}"
  local CMD="sbatch --parsable --array=${ROWS}%3 ${RES} ${DEP} --export=ALL,${BASE},TAG=${TAGN},OFFSETS=${OFF},${SENSOR} ${SCRIPT}"
  if [ "${DRY:-0}" = "1" ]; then echo "[dry run] ${CMD}"; PREV=""; return; fi
  PREV=$(${CMD})
  printf "tag %-12s offset +%-3s rows %-22s job %s\n" "${TAGN}" "${OFF}" "${ROWS}" "${PREV}"
}
sub 5  "182,192,198,212,248,275"   # margin 0 for all six
sub 10 "182,198,212,248,275"       # RT041 S, RT057 C, RT071 C, RC007 C, RC034 S
sub 6  "198"                       # RT057 C
sub 7  "192"                       # RT051 S
sub 8  "192"                       # RT051 C
sub 9  "212"                       # RT071 S
sub 11 "182"                       # RT041 C
sub 15 "275"                       # RC034 C
echo "Watch: squeue -u \$USER     (17 simulations, roughly 15 to 40 minutes each, 3 at a time)"
