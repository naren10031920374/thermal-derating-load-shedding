function summary = run_scenario(scenario, opts)
% ==========================================================================
%  RUN_SCENARIO.M   (dataset pipeline, step 2: the NO-SHED run)
%  --------------------------------------------------------------------
%  Reads ONE row of scenario_table.csv, builds that scenario's load profile,
%  runs the full 5000 s simulation with NO shedding, and saves:
%    - the full signal table (same columns as generate_gradual_busC_scenario_v4,
%      so the existing feature-building scripts can read it)
%    - a per-bus collapse table
%    - a JSON summary the later steps read (does it collapse? which buses? when?)
%
%  This one function replaces the per-scenario generator scripts. The model
%  is Grid_modelling_Thermal_V7_ALfix_sheddable_ramped with every ShedFrac
%  block left at 1 (no shed), exactly as the bisection scripts start each run.
%
%  Usage
%    run_scenario('S041')                    % by scenario_id
%    run_scenario(41)                        % by row number (SLURM array task id)
%    run_scenario('S041', struct('smoke_test', true))   % 200 s wiring check
%
%  opts fields (all optional)
%    smoke_test   true -> TEND=200 s, no collapse expected, checks wiring only
%    tend         override stop time in seconds (default 5000)
%    output_root  default <project>/model_outputs/dataset_pipeline
%    model_file   full path to the .slx (default: scripts/06_gradual_ramp copy)
%    format       'csv' (default) or 'parquet' for the big signal table
%
%  VALIDATION TARGET: row S041 is an exact repeat of Bus C v4 (seed 104). Its
%  Bus C collapse should land at about t = 2094.47 s, as in the original run.
% ==========================================================================
    if nargin < 2; opts = struct(); end
    H = scenario_helpers();

    SCRIPT_DIR   = fileparts(mfilename('fullpath'));
    PROJECT_ROOT = fileparts(fileparts(SCRIPT_DIR));   % scripts/09_dataset_pipeline -> project root

    MODEL_NAME = 'Grid_modelling_Thermal_V7_ALfix_sheddable_ramped';
    TS   = 0.01;
    PMIN = 2000;
    PMAX = 220000;
    COLLAPSE_VOLTAGE_V      = 100.0;
    MIN_CONSECUTIVE_SAMPLES = 50;
    EPSILON_W = 500.0;
    WARMUP_TOLERANCE_S = 1.0;

    BUS  = {'A','B','C','D','E','F','G','H','K','L'};
    CONV = {'AB','BC','CD','DE','EF','FG','GH','HK','KL','AL'};
    NB = numel(BUS);

    smoke = get_opt(opts, 'smoke_test', false);
    if smoke
        TEND = get_opt(opts, 'tend', 200);
    else
        TEND = get_opt(opts, 'tend', 5000);
    end
    N_STEPS = round(TEND / TS) + 1;
    out_format = get_opt(opts, 'format', 'csv');

    %% 1. Read the scenario row -------------------------------------------
    tbl = readtable(fullfile(SCRIPT_DIR, 'scenario_table.csv'), 'TextType', 'string');
    row = get_row(tbl, scenario);

    % Output folder: opts.output_root, else the DATASET_OUTPUT_ROOT environment
    % variable (set it to a /scratch path on Great Lakes to keep large files
    % out of your home quota), else <project>/model_outputs/dataset_pipeline.
    env_root = getenv('DATASET_OUTPUT_ROOT');
    if isempty(env_root)
        env_root = fullfile(PROJECT_ROOT, 'model_outputs', 'dataset_pipeline');
    end
    out_root = get_opt(opts, 'output_root', env_root);
    OUTPUT_DIR = fullfile(out_root, row.scenario_id);
    if exist(OUTPUT_DIR, 'dir') ~= 7; mkdir(OUTPUT_DIR); end
    tag = row.scenario_id;
    if smoke; tag = [tag '_smoke']; end

    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('NO-SHED RUN: %s (%s)%s\n', row.scenario_id, row.kind, ternary(smoke, '  [SMOKE TEST]', ''));
    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('Target buses:   %s\n', strjoin(row.targets, ', '));
    fprintf('Boundary buses: %s\n', strjoin(row.boundary, ', '));
    fprintf('Ramp %d-%d s | plateau %.0f W | boundary low %.0f W | seed %d\n', ...
        row.ramp_start_s, row.ramp_end_s, row.plateau_w, row.boundary_low_w, row.seed);
    fprintf('Stop time: %d s\n', TEND);

    %% 2. Locate and load the model ---------------------------------------
    % Model choice. On Linux (Great Lakes, MATLAB R2025b) use the copy that was
    % re-exported for the cluster (export_models_for_greatlakes.m); on Windows
    % use the normal copy. An explicit opts.model_file always wins.
    model_file = get_opt(opts, 'model_file', '');
    if isempty(model_file)
        normal_copy  = fullfile(PROJECT_ROOT, 'scripts', '06_gradual_ramp', [MODEL_NAME '.slx']);
        cluster_copy = fullfile(PROJECT_ROOT, 'scripts', '06_gradual_ramp', 'greatlakes_export', [MODEL_NAME '.slx']);
        if isunix && exist(cluster_copy, 'file') == 2
            model_file = cluster_copy;
        elseif exist(normal_copy, 'file') == 2
            model_file = normal_copy;
        else
            hh = dir(fullfile(PROJECT_ROOT, '**', [MODEL_NAME '.slx']));
            if isempty(hh)
                error('Model not found: %s.slx under %s', MODEL_NAME, PROJECT_ROOT);
            end
            model_file = fullfile(hh(1).folder, hh(1).name);
        end
    end
    fprintf('Model: %s\n', model_file);
    if bdIsLoaded(MODEL_NAME)
        close_system(MODEL_NAME, 0);
    end
    load_system(model_file);
    set_param(MODEL_NAME, 'StopTime', num2str(TEND));

    % No-shed state: every bus's ShedFrac = 1, no slew limit. Same reset the
    % bisection scripts do at the start of every iteration.
    for b = 1:NB
        H.ensure_constant_block(MODEL_NAME, BUS{b}, ...
            sprintf('%s/ShedFrac_%s_default', MODEL_NAME, BUS{b}), 1);
        ramp_blk = sprintf('%s/ShedFrac_%s_ramp', MODEL_NAME, BUS{b});
        if getSimulinkBlockHandle(ramp_blk) ~= -1
            set_param(ramp_blk, 'RisingSlewLimit', 'inf', 'FallingSlewLimit', '-inf');
        end
    end

    %% 3. Build the load profile ------------------------------------------
    fprintf('\nBuilding load profile (seed %d) ...\n', row.seed);
    [load_hist, clip_hi, clip_lo] = build_scenario_loads(row, BUS, N_STEPS, TS, PMIN, PMAX);
    for b = 1:NB
        assignin('base', sprintf('Pload_%s', BUS{b}), load_hist(:, b).');
    end
    fprintf('  Clip-high %% by bus: %s\n', mat2str(round(clip_hi, 1)));
    fprintf('  Clip-low  %% by bus: %s\n', mat2str(round(clip_lo, 1)));

    %% 4. Wire taps, run --------------------------------------------------
    fprintf('\nWiring signal taps ...\n');
    H.wire_all_taps(MODEL_NAME, BUS, CONV);
    derate_var_names   = H.tap_derate_factors(MODEL_NAME, CONV);
    junction_var_names = H.tap_junction_temp(MODEL_NAME, CONV);
    phase_var_names    = H.tap_phase_commands(MODEL_NAME, CONV);

    fprintf('\nRunning simulation ...\n');
    tic;
    ws = warning('off', 'all');
    so = sim(MODEL_NAME, 'ReturnWorkspaceOutputs', 'on');
    warning(ws);
    wall_clock = toc;
    fprintf('Finished in %.1f s wall clock.\n', wall_clock);

    time_vec = (0:N_STEPS-1)' * TS;

    %% 5. Assemble the signal table (same columns as generate_..._v4) ------
    T = table(time_vec, 'VariableNames', {'time'});
    src_pow_matrix = nan(N_STEPS, NB);
    for b = 1:NB
        T.(sprintf('V_Bus_%s', BUS{b})) = H.grab_series(so, sprintf('V_Bus_%s', BUS{b}), TS, N_STEPS);
        src_pow_matrix(:, b) = H.grab_series(so, sprintf('Bus_%s_Src_Pow', BUS{b}), TS, N_STEPS);
        T.(sprintf('Bus_%s_Src_Pow', BUS{b})) = src_pow_matrix(:, b);
        T.(sprintf('Bus_%s_Temp', BUS{b})) = H.grab_series(so, sprintf('Bus_%s_Temp', BUS{b}), TS, N_STEPS);
    end
    for b = 1:NB
        T.(sprintf('CommandedLoad_kW_%s', BUS{b})) = load_hist(:, b) / 1000;
    end

    avg_power = mean(src_pow_matrix, 2, 'omitnan');
    grid_collapsed = avg_power < EPSILON_W;
    for b = 1:NB
        ratio = src_pow_matrix(:, b) ./ avg_power;
        ratio(grid_collapsed) = 0.0;
        T.(sprintf('GEI_%s', BUS{b})) = ratio;
    end
    T.Src_Pow_Avg_recomputed = avg_power;

    for c = 1:numel(CONV)
        T.(sprintf('Phase_%s_cmd_deg', CONV{c})) = H.grab_series(so, phase_var_names{c}, TS, N_STEPS);
        T.(sprintf('DAB_%s_Derate_Factor', CONV{c})) = H.grab_series(so, derate_var_names{c}, TS, N_STEPS) / 100e3;
        T.(sprintf('JunctionTemp_C_%s', CONV{c})) = H.grab_series(so, junction_var_names{c}, TS, N_STEPS);
    end
    hs_conv_order = {'AB','BC','CD','DE','EF','FG','GH','HK','KL','AL'};
    hs_tags = {'DAB_AB_H_Temp','DAB_BC_H_Temp','DAB_CD_H_Temp','DAB_DE_H_Temp', ...
               'DAB_EF_H_Temp','DAB_FG_H_Temp','DAB_GH_Temp','DAB_HK_Temp', ...
               'DAB_KL_Temp','DAB_LA_Temp'};
    for h = 1:numel(hs_tags)
        T.(sprintf('HeatSinkTemp_C_%s', hs_conv_order{h})) = H.grab_series(so, hs_tags{h}, TS, N_STEPS);
    end

    %% 6. NaN/Inf validation ----------------------------------------------
    log_path = fullfile(OUTPUT_DIR, sprintf('noshed_%s_validation_log.txt', tag));
    log_fid = fopen(log_path, 'w');
    data_cols = T.Properties.VariableNames(2:end);
    any_unexpected = false;
    for ci = 1:numel(data_cols)
        v = T.(data_cols{ci});
        bad_here = isnan(v) | isinf(v);
        if any(bad_here)
            bad_times = time_vec(bad_here);
            n_after = sum(bad_times > WARMUP_TOLERANCE_S);
            H.log_both(log_fid, '  %-30s %6d bad samples, %6d after warmup\n', ...
                data_cols{ci}, sum(bad_here), n_after);
            if n_after > 0; any_unexpected = true; end
        end
    end
    if any_unexpected
        H.log_both(log_fid, 'RESULT: FAILED VALIDATION (unexpected NaN/Inf after warmup).\n');
    else
        H.log_both(log_fid, 'RESULT: PASSED VALIDATION.\n');
    end
    fclose(log_fid);

    %% 7. Per-bus collapse check ------------------------------------------
    fprintf('\nPer-bus collapse check (%.0f V, %d-sample debounce):\n', ...
        COLLAPSE_VOLTAGE_V, MIN_CONSECUTIVE_SAMPLES);
    collapsed = false(NB, 1); onset = nan(NB, 1); min_v = nan(NB, 1);
    for b = 1:NB
        [c, on, mv] = H.detect_collapse_summary(time_vec, T.(sprintf('V_Bus_%s', BUS{b})), ...
            COLLAPSE_VOLTAGE_V, MIN_CONSECUTIVE_SAMPLES);
        collapsed(b) = c; onset(b) = on; min_v(b) = mv;
        if c
            fprintf('  Bus %s: COLLAPSES at t=%.2fs (min V=%.2f)\n', BUS{b}, on, mv);
        else
            fprintf('  Bus %s: survives (min V=%.2f)\n', BUS{b}, mv);
        end
    end

    is_target = ismember(BUS(:), row.targets(:));
    target_all_collapse = all(collapsed(is_target));
    nontarget_collapsed = BUS(collapsed & ~is_target);

    % Flag scenarios that need a human look before they are used.
    if strcmp(row.kind, 'mild')
        needs_review = any(collapsed);
    else
        needs_review = ~target_all_collapse || ~isempty(nontarget_collapsed);
    end

    if smoke
        status = 'smoke_test';
    elseif any_unexpected
        status = 'failed_validation';
    elseif any(collapsed)
        status = 'collapses';
    else
        status = 'normal';
    end

    %% 8. Save -------------------------------------------------------------
    summary = struct();
    summary.scenario_id         = row.scenario_id;
    summary.kind                = row.kind;
    summary.status              = status;
    summary.needs_review        = needs_review;
    summary.targets             = row.targets;
    summary.boundary            = row.boundary;
    summary.seed                = row.seed;
    summary.tend_s              = TEND;
    summary.target_all_collapse = target_all_collapse;
    summary.nontarget_collapsed = nontarget_collapsed;
    summary.collapsed_buses     = BUS(collapsed);
    summary.collapse_onset_s    = onset(collapsed)';
    summary.min_voltage_by_bus  = min_v';
    summary.bus_order           = BUS;
    summary.wall_clock_s        = wall_clock;
    summary.model_file          = model_file;

    collapse_tbl = table(BUS(:), collapsed, onset, min_v, is_target, ...
        'VariableNames', {'bus', 'collapsed', 'onset_s', 'min_voltage', 'is_target'});
    writetable(collapse_tbl, fullfile(OUTPUT_DIR, sprintf('noshed_%s_collapse.csv', tag)));

    fid = fopen(fullfile(OUTPUT_DIR, sprintf('noshed_%s_summary.json', tag)), 'w');
    fprintf(fid, '%s', jsonencode(summary));
    fclose(fid);

    if any_unexpected
        close_system(MODEL_NAME, 0);
        error('run_scenario:validation', ...
              '%s FAILED NaN/Inf validation. See %s', row.scenario_id, log_path);
    end

    if strcmp(out_format, 'parquet')
        out_sig = fullfile(OUTPUT_DIR, sprintf('noshed_%s_%ds.parquet', tag, TEND));
        parquetwrite(out_sig, T);
    else
        out_sig = fullfile(OUTPUT_DIR, sprintf('noshed_%s_%ds.csv', tag, TEND));
        writetable(T, out_sig);
    end
    fprintf('\nWrote: %s (%d rows x %d cols)\n', out_sig, height(T), width(T));
    fprintf('STATUS: %s%s\n', status, ternary(needs_review, '   ** NEEDS REVIEW **', ''));

    close_system(MODEL_NAME, 0);
end


%% ---- local helpers ----------------------------------------------------

function row = get_row(tbl, scenario)
    if isnumeric(scenario)
        idx = scenario;
    else
        idx = find(strcmp(string(tbl.scenario_id), string(scenario)), 1);
    end
    if isempty(idx) || idx < 1 || idx > height(tbl)
        error('run_scenario:noRow', 'Scenario %s not found in scenario_table.csv', string(scenario));
    end
    r = tbl(idx, :);
    row = struct();
    row.scenario_id        = char(r.scenario_id);
    row.kind               = char(r.kind);
    row.targets            = strsplit(char(r.targets), ';');
    row.boundary           = strsplit(char(r.boundary), ';');
    row.ramp_start_s       = double(r.ramp_start_s);
    row.ramp_end_s         = double(r.ramp_end_s);
    row.plateau_w          = double(r.plateau_w);
    row.boundary_low_w     = double(r.boundary_low_w);
    row.preheat_w          = double(r.preheat_w);
    row.boundary_preheat_w = double(r.boundary_preheat_w);
    row.jitter_frac        = double(r.jitter_frac);
    row.stepdown_start_s   = double(r.stepdown_start_s);
    row.stepdown_end_s     = double(r.stepdown_end_s);
    row.recovery_w         = double(r.recovery_w);
    row.boundary_recovery_w = double(r.boundary_recovery_w);
    row.seed               = double(r.seed);
end


function v = get_opt(opts, name, default)
    if isfield(opts, name); v = opts.(name); else; v = default; end
end


function out = ternary(cond, a, b)
    if cond; out = a; else; out = b; end
end
