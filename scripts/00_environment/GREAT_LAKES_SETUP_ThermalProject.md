# Great Lakes HPC Setup — ThermalProject

Tailored to the actual repo (not the generic template), for running the
thermal derating / load shedding pipeline's long MATLAB+Simulink sweeps on
U-M's Great Lakes cluster instead of tying up a laptop for hours.

**First use case this unlocks:** the pending Stage 06 gradual-ramp sweep
(`test_shed_gradual_ramp_corridor_triad_collapse.m`), whose serial runtime
is ~2–2.25 hours. Run in parallel on Great Lakes (see below) it should
finish in well under an hour.

## What's new / what changed

- `test_shed_gradual_ramp_corridor_triad_collapse.m` and
  `compare_sheddable_vs_ramped_corridor_triad_collapse.m` no longer
  hardcode `PROJECT_ROOT = 'D:\naren\Documents\...'`. They now compute it
  from their own file location (`fileparts(fileparts(fileparts(mfilename('fullpath'))))`),
  which resolves correctly on both Windows and Linux — **no other change
  in behavior**, they still run exactly as before if you run them locally.
- `test_shed_gradual_ramp_corridor_triad_collapse.m` now takes an optional
  argument: `test_shed_gradual_ramp_corridor_triad_collapse(duration_idx)`
  runs only that one of the 8 ramp durations and saves a partial result,
  instead of looping over all 8. Calling it with no argument (as before)
  is unchanged — full serial sweep, all 8, one call.
- New `combine_gradual_ramp_results.m` — stitches the 8 partial results
  back into the same summary table and 3 plots the serial run produces.
  Only meaningful after running the array version below.
- New job scripts and this doc, all under `scripts/00_environment/`
  (previously empty — this seemed like the natural home for
  environment/infrastructure files as opposed to pipeline stages).
- **Every other `.m` script in this repo still hardcodes the Windows path**
  and will fail immediately on Linux with "file not found" type errors.
  Apply the same one-line fix before running any of them on Great Lakes:
  replace the hardcoded `PROJECT_ROOT = '...'` line with
  `PROJECT_ROOT = fileparts(fileparts(fileparts(mfilename('fullpath'))));`
  — adjusting the number of `fileparts()` calls to match how many folders
  deep that script sits under the project root (2 for anything directly in
  `scripts/0N_.../`, which is every script in this repo).

## One-time setup

### 1. Confirm your Great Lakes access

You'll need: Great Lakes SSH access + Duo, and your uniqname
(`narenv`, based on your umich.edu email — replace if that's wrong).

### 2. Upload the project

From your local machine (PowerShell or the Windows Terminal), **not**
Great Lakes:

```
scp -r "D:\naren\Documents\Thermal_Derating_Work_Handover\ThermalProject" narenv@greatlakes-xfer.arc-ts.umich.edu:/home/narenv/
```

This will be slow the first time — `model_outputs/`, `data/`, `data_d/`,
and `large_files_to_copy/` likely hold the bulk of the size (parquet/CSV/
.mat files). If upload is too slow or too large for your quota, consider
uploading only what a given job actually reads (e.g. just
`scripts/`, the two `.slx` models needed, and the specific
`model_outputs/thermal_derating_v7/corridor_triad_collapse*` subfolders)
rather than the whole tree.

### 3. Log in and verify

```
ssh narenv@greatlakes.arc-ts.umich.edu
cd ~/ThermalProject
ls -lah scripts/06_gradual_ramp/
```

### 4. Verify MATLAB + Simulink licensing — do this before submitting a real job

This is the biggest unknown in this setup: Great Lakes provides a MATLAB
module, but Simulink is a separate license entitlement, and academic
license pools often cap **concurrent** checkouts. The array-job approach
below opens up to 8 simultaneous Simulink sessions — if your account/pool
doesn't have that many seats, some tasks will hang waiting on a license
or fail.

```
module load matlab
matlab -batch "disp(license('test','simulink'))"
```

If this prints `0`, or you're unsure how many concurrent seats you have,
email `arc-support@umich.edu` before running the array version — the
serial version (1 session at a time) is the safe fallback regardless.

### 5. Python environment (only needed for stages 02–04 / 07, not the MATLAB-only stage 06 sweep)

```
module load python
pip install --user -r scripts/00_environment/requirements.txt
```

`requirements.txt` was built from the actual `import` statements in the
delivered pipeline scripts (not from a real `pip freeze`, since the
device bridge couldn't reach the local venv this session — see the file's
header comment for how to double-check it against the real venv).

## Running Stage 06 (the pending gradual-ramp sweep)

Three options, in `scripts/00_environment/`:

**Option A — simplest, lowest risk:** one job, serial, ~2–2.25 hours,
matches running it locally exactly.
```
cd ~/ThermalProject
sbatch scripts/00_environment/run_gradual_ramp_serial.sh
```

**Option B — parallel, the actual point of using a cluster:** 8 array
tasks (one per ramp duration) run concurrently, each far shorter than the
full sweep, then an automatic combine step reproduces the same summary
table + plots.
```
cd ~/ThermalProject
bash scripts/00_environment/submit_gradual_ramp_array.sh
```
This chains three SLURM jobs (`verify` → `array[1-8]` → `combine`), each
gated on the previous succeeding. Watch progress with `squeue -u narenv`.
The array job defaults to at most 3 concurrent tasks (`--array=1-8%3`) as
a conservative guess at license availability — raise the `%3` in
`run_gradual_ramp_array.sh` once you've confirmed your Simulink seat count
comfortably covers more.

**Option C — resubmit just one failed duration**, after Option B partially
failed:
```
sbatch --array=3 scripts/00_environment/run_gradual_ramp_array.sh   # e.g. duration_idx=3 only
sbatch scripts/00_environment/run_combine_gradual_ramp.sh           # once all 8 partials exist
```

## Monitoring and results

```
squeue -u narenv                                   # job status
tail -f scripts/00_environment/logs/<name>_<jobid>.log   # live output
sacct -j <jobid>                                    # exit status / runtime after completion
```

Results land in the same place they would locally:
`model_outputs/thermal_derating_v7/corridor_triad_collapse_gradual_ramp_sweep/`
— `gradual_ramp_outcome_summary.png`, `gradual_ramp_voltages.png`,
`gradual_ramp_derate_factors.png`, `gradual_ramp_sweep_results.mat` (plus,
in the array path, 8 `gradual_ramp_partial_D*.mat` files you can delete
once `combine_gradual_ramp_results.m` has run successfully).

## Downloading results back

From your **local** machine:
```
scp -r narenv@greatlakes-xfer.arc-ts.umich.edu:/home/narenv/ThermalProject/model_outputs/thermal_derating_v7/corridor_triad_collapse_gradual_ramp_sweep ./
```

## Known unknowns — verify, don't assume

- **Simulink concurrent license count** — see step 4 above. This is the
  one thing most likely to make Option B (array) behave worse than Option
  A (serial) if assumed rather than checked.
- **Exact `module load matlab` version** — Great Lakes may default to a
  specific MATLAB release; if the pipeline needs a specific version, use
  `module avail matlab` to see what's installed and pin it explicitly
  (`module load matlab/R2024b`, etc.) in each `.sh` script.
- **Partition/QoS/allocation name** — `--partition=standard` assumes the
  default; if your account attaches to a group allocation instead
  (common for RA work), you may need `--account=<slurm_account>` on every
  job script, or a different `--partition`. Check with
  `sacctmgr show associations user=narenv` or ask your PI/advisor which
  account/partition to use.
- **Home directory storage quota** — `model_outputs/` for this project
  already runs to hundreds of MB to GB; Great Lakes home directories are
  typically quota-limited (often 80GB), so for heavier future use consider
  `/scratch` space instead of `/home` (ask ARC-TS for your scratch path).
