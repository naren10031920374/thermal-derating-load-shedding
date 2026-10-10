#!/bin/bash
# ============================================================================
# Submits the whole sensor noise / delay test: 6 settings x the 9 collapsing held-out
# scenarios, all with the causal staged limiter (STAGED=1). Run from the repo root:
#     bash scripts/09_dataset_pipeline/submit_noise_delay.sh
# Check the wiring first with a smoke run (about 3 min):
#     sbatch --array=34 --export=ALL,SMOKE=1,STAGED=1,MEAS_DELAY=30,MEAS_NOISE=5 scripts/09_dataset_pipeline/run_staged_array.sh
# Afterwards:  python3 scripts/09_dataset_pipeline/summarize_noise_delay.py
# ============================================================================
set -euo pipefail
ROWS=12,13,14,19,20,24,32,34,35
S=scripts/09_dataset_pipeline/run_staged_array.sh

sbatch --array=$ROWS --export=ALL,STAGED=1,TAG=nd_d30,MEAS_DELAY=30 $S
sbatch --array=$ROWS --export=ALL,STAGED=1,TAG=nd_d60,MEAS_DELAY=60 $S
sbatch --array=$ROWS --export=ALL,STAGED=1,TAG=nd_d120,MEAS_DELAY=120 $S
sbatch --array=$ROWS --export=ALL,STAGED=1,TAG=nd_n5,MEAS_NOISE=5 $S
sbatch --array=$ROWS --export=ALL,STAGED=1,TAG=nd_n10,MEAS_NOISE=10 $S
sbatch --array=$ROWS --export=ALL,STAGED=1,TAG=nd_n5d30,MEAS_NOISE=5,MEAS_DELAY=30 $S
echo "Submitted 6 arrays x 9 tasks. Watch with: squeue -u \$USER"
