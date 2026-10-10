#!/bin/bash
# ============================================================================
# Sensor noise test WITH a 10 s moving average on the reading (needs the new run_staged_limiter.m).
# 3 settings x the 9 collapsing held-out scenarios, causal staged limiter. From the repo root:
#     bash scripts/09_dataset_pipeline/submit_noise_filter.sh
# Smoke check first (about 3 min):
#     sbatch --array=34 --export=ALL,SMOKE=1,STAGED=1,MEAS_NOISE=5,MEAS_FILTER=10 scripts/09_dataset_pipeline/run_staged_array.sh
# Afterwards:  python3 scripts/09_dataset_pipeline/summarize_noise_delay.py --filtered
# ============================================================================
set -euo pipefail
ROWS=12,13,14,19,20,24,32,34,35
S=scripts/09_dataset_pipeline/run_staged_array.sh

sbatch --array=$ROWS --export=ALL,STAGED=1,TAG=nf_n5f10,MEAS_NOISE=5,MEAS_FILTER=10 $S
sbatch --array=$ROWS --export=ALL,STAGED=1,TAG=nf_n10f10,MEAS_NOISE=10,MEAS_FILTER=10 $S
sbatch --array=$ROWS --export=ALL,STAGED=1,TAG=nf_n5d30f10,MEAS_NOISE=5,MEAS_DELAY=30,MEAS_FILTER=10 $S
echo "Submitted 3 arrays x 9 tasks. Watch with: squeue -u \$USER"
