function result = bisect_scenario(scenario, opts)
% ==========================================================================
%  BISECT_SCENARIO.M   (dataset pipeline, step 4: find the minimum safe shed)
%  --------------------------------------------------------------------
%  For ONE scenario that collapses with no shedding, binary-search the mildest
%  shed that keeps ALL 10 buses alive for the FULL 5000 s run.
%
%  Inputs it reads (all written by earlier steps, same output folder):
%    noshed_<ID>_collapse.csv        which buses collapse, and when   (step 2)
%    detector_triggers_<ID>.json     detector trigger time per bus    (step 3)
%
%  What one search iteration does (same recipe as the proven Bus C / Bus F
%  bisection scripts):
%    1. reload the model fresh, reset every ShedFrac_<bus>_default to 1
%    2. put a Step block on ShedFrac_<bus>_default for each SHED bus:
%       value 1 before that bus's trigger time, <fraction> after it
%    3. run the full sim, check all 10 buses for collapse
%    4. all safe -> try a MILDER cut next; any collapse -> try a HARDER cut
%
%  ShedFrac is the fraction of load KEPT: 1 = no shed, 0.30 = keep 30 %.
%  Bigger value = milder cut. Reported shed percent = (1 - ShedFrac) * 100.
%
%  Which buses are shed: every bus that collapses in the no-shed run, at its
%  own detector trigger time. If the detector never fired for such a bus (a
%  MISS), the fallback trigger = onset - 60 s is used and flagged in the
%  result. A detector trigger on a bus that never collapses (false alarm) is
%  recorded but that bus is NOT shed, so the answer stays "the minimum cut of
%  the buses that actually need it".
%
%  cut_mode 'shared' (default, the first batch): all shed buses use the SAME
%  fraction. ('per_bus' is reserved for the later upgrade and errors for now.)
%
%  Usage
%    bisect_scenario('S041')
%    bisect_scenario(41)                              % row number (SLURM)
%    bisect_scenario('S041', struct('verify_monotonic', true))
%
%  opts fields (all optional)
%    grid              ShedFrac candidates, ascending (default 0.05:0.05:0.95)
%    cut_mode          'shared' (default)
%    fallback_lead_s   lead used when the detector missed (default 60)
%    verify_monotonic  true -> after the search run ONE extra sim one grid step
%                      more aggressive than the answer (must also be safe)
%    output_root       default env DATASET_OUTPUT_ROOT, else <project>/model_outputs/dataset_pipeline
%    model_file        override the .slx path
%    tend              stop time, default 5000
%  Resumable: every finished iteration is saved; re-running skips fractions
%  already simulated (delete bisect_<ID>_iterations.json to start over).
% ==========================================================================
    if nargin < 2; opts = struct(); end
    H = scenario_helpers();

    SCRIPT_DIR   = fileparts(mfilename('fullpath'));
    PROJECT_ROOT = fileparts(fileparts(SCRIPT_DIR));

    MODEL_NAME = 'Grid_modelling_Thermal_V7_ALfix_sheddable_ramped';
    TS   = 0.01;
    PMIN = 2000;
    PMAX = 220000;
    COLLAPSE_VOLTAGE_V      = 100.0;
    MIN_CONSECUTIVE_SAMPLES = 50;

    BUS  = {'A','B','C','D','E','F','G','H','K','L'};
    CONV = {'AB','BC','CD','DE','EF','FG','GH','HK','KL','AL'};
    NB = numel(BUS);

    TEND    = get_opt(opts, 'tend', 5000);
    N_STEPS = round(TEND / TS) + 1;
    GRID    = get_opt(opts, 'grid', 0.05:0.05:0.95);
    GRID    = round(GRID(:)' * 1000) / 1000;
    cut_mode = get_opt(opts, 'cut_mode', 'shared');
    FALLBACK_LEAD_S = get_opt(opts, 'fallback_lead_s', 60);
    verify_monotonic = get_opt(opts, 'verify_monotonic', false);
    if ~strcmp(cut_mode, 'shared')
        error('bisect_scenario:cutMode', ...
              'cut_mode ''%s'' is not implemented yet (only ''shared'').', cut_mode);
    end
    if any(diff(GRID) <= 0)
        error('bisect_scenario:grid', 'grid must be strictly ascending.');
    end

    %% 1. Scenario row + earlier-step outputs -------------------------------
    tbl = readtable(fullfile(SCRIPT_DIR, 'scenario_table.csv'), 'TextType', 'string');
    row = get_row(tbl, scenario);
    id  = row.scenario_id;

    env_root = getenv('DATASET_OUTPUT_ROOT');
    if isempty(env_root)
        env_root = fullfile(PROJECT_ROOT, 'model_outputs', 'dataset_pipeline');
    end
    out_root   = get_opt(opts, 'output_root', env_root);
    OUTPUT_DIR = fullfile(out_root, id);

    collapse_csv = fullfile(OUTPUT_DIR, sprintf('noshed_%s_collapse.csv', id));
    trig_json    = fullfile(OUTPUT_DIR, sprintf('detector_triggers_%s.json', id));
    if exist(collapse_csv, 'file') ~= 2
        error('bisect_scenario:missing', 'Run step 2 first: %s not found.', collapse_csv);
    end
    ct = readtable(collapse_csv, 'TextType', 'string');
    ct_bus = cellstr(ct.bus);
    ct_collapsed = logical(ct.collapsed);
    ct_onset = double(ct.onset_s);

    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('BISECTION: %s (%s), cut_mode=%s\n', id, row.kind, cut_mode);
    fprintf('%s\n', repmat('=', 1, 70));

    result = struct();
    result.scenario_id = id;
    result.kind        = row.kind;
    result.cut_mode    = cut_mode;
    result.shedfrac_meaning = 'fraction of load kept (1 = no shed)';

    if ~any(ct_collapsed)
        fprintf('No bus collapses in the no-shed run: nothing to bisect.\n');
        result.status = 'no_collapse';
        result.best_shedfrac = 1;
        result.min_shed_percent = 0;
        write_json(fullfile(OUTPUT_DIR, sprintf('bisect_%s_result.json', id)), result);
        return;
    end
    if exist(trig_json, 'file') ~= 2
        error('bisect_scenario:missing', 'Run step 3 first: %s not found.', trig_json);
    end
    det = jsondecode(fileread(trig_json));

    %% 2. Decide who is shed and when -----------------------------------------
    shed_idx = find(ct_collapsed)';
    shed_bus = ct_bus(shed_idx)';
    n_shed = numel(shed_bus);
    trig_time = nan(1, n_shed);
    trig_source = cell(1, n_shed);
    for k = 1:n_shed
        b = shed_bus{k};
        onset_k = ct_onset(shed_idx(k));
        t_det = [];
        if isfield(det.buses, b) && isfield(det.buses.(b), 'trigger_time')
            t_det = det.buses.(b).trigger_time;
        end
        if ~isempty(t_det) && ~isnan(t_det) && t_det < onset_k
            trig_time(k) = t_det;
            trig_source{k} = 'detector';
        else
            trig_time(k) = max(0, onset_k - FALLBACK_LEAD_S);
            trig_source{k} = 'fallback_onset_minus_lead';
        end
        fprintf('  Shed bus %s: no-shed onset %.2f s, trigger %.2f s (%s)\n', ...
                b, onset_k, trig_time(k), trig_source{k});
    end
    false_alarm_buses = {};
    for b = 1:NB
        nm = BUS{b};
        if ~ismember(nm, shed_bus) && isfield(det.buses, nm) && ...
                isfield(det.buses.(nm), 'trigger_time') && ~isempty(det.buses.(nm).trigger_time)
            false_alarm_buses{end+1} = nm; %#ok<AGROW>
        end
    end
    if ~isempty(false_alarm_buses)
        fprintf('  Detector false alarms (buses that never collapse, NOT shed): %s\n', ...
                strjoin(false_alarm_buses, ', '));
    end

    %% 3. Model + load profile ----------------------------------------------
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

    [load_hist, ~, ~] = build_scenario_loads(row, BUS, N_STEPS, TS, PMIN, PMAX);
    time_vec = (0:N_STEPS-1)' * TS;

    %% 4. Resume cache -------------------------------------------------------
    iter_path = fullfile(OUTPUT_DIR, sprintf('bisect_%s_iterations.json', id));
    iters = struct('grid_index', {}, 'shedfrac', {}, 'all_safe', {}, ...
                   'collapsed_buses', {}, 'onsets', {}, 'min_v', {}, 'wall_clock_s', {});
    if exist(iter_path, 'file') == 2
        prev = jsondecode(fileread(iter_path));
        if isfield(prev, 'trigger_time_by_bus') && isequal(sort(prev.shed_buses(:))', sort(shed_bus(:))') ...
                && max(abs(prev.trigger_time_by_bus(:)' - trig_time)) < 1e-6 && isequal(prev.grid(:)', GRID)
            for q = 1:numel(prev.iterations)
                it = prev.iterations(q);
                iters(end+1) = struct('grid_index', it.grid_index, 'shedfrac', it.shedfrac, ... %#ok<AGROW>
                    'all_safe', logical(it.all_safe), 'collapsed_buses', {as_cellstr(it.collapsed_buses)}, ...
                    'onsets', it.onsets(:)', 'min_v', it.min_v(:)', 'wall_clock_s', it.wall_clock_s);
            end
            fprintf('Resuming: %d iteration(s) already on disk.\n', numel(iters));
        else
            fprintf('Existing iterations file does not match this setup; starting fresh.\n');
        end
    end

    %% 5. Binary search ------------------------------------------------------
    n_cand = numel(GRID);
    lo = 1; hi = n_cand; best_idx = -1;
    t_total = tic;
    it_count = 0;
    while lo <= hi
        mid = floor((lo + hi) / 2);
        [iters, rec] = run_or_reuse(mid);
        it_count = it_count + 1;
        fprintf('\nITERATION %d: range=[%d,%d] test %.3f -> %s\n', it_count, lo, hi, GRID(mid), ...
                ternary(rec.all_safe, 'ALL SAFE (try milder)', ...
                        ['COLLAPSE: ' strjoin(rec.collapsed_buses, ',') ' (try harder)']));
        if rec.all_safe
            best_idx = mid; lo = mid + 1;
        else
            hi = mid - 1;
        end
    end

    mono_ok = [];
    mono_frac = NaN;
    if verify_monotonic && best_idx > 1
        [iters, rec] = run_or_reuse(best_idx - 1);
        mono_frac = GRID(best_idx - 1);
        mono_ok = rec.all_safe;
        fprintf('\nMonotonic check at %.3f (one step harder than the answer): %s\n', ...
                mono_frac, ternary(mono_ok, 'safe (consistent)', 'NOT SAFE (non-monotonic!)'));
    end

    %% 6. Result -------------------------------------------------------------
    if best_idx > 0
        result.status = 'safe_found';
        result.best_shedfrac = GRID(best_idx);
        result.min_shed_percent = (1 - GRID(best_idx)) * 100;
        if best_idx == n_cand
            result.note = 'mildest grid point is already safe; true minimum may be milder than the grid';
        end
        fprintf('\nBEST (mildest safe) ShedFrac = %.3f  (min shed %.1f %%)\n', ...
                result.best_shedfrac, result.min_shed_percent);
    else
        result.status = 'no_safe_in_grid';
        result.best_shedfrac = NaN;
        result.min_shed_percent = NaN;
        fprintf('\nNO SAFE ShedFrac in the grid (even %.3f still collapses).\n', GRID(1));
    end
    result.shed_buses = shed_bus;
    result.trigger_time_by_bus = trig_time;
    result.trigger_source_by_bus = trig_source;
    result.detector_false_alarm_buses = false_alarm_buses;
    result.noshed_onset_by_bus = ct_onset(shed_idx)';
    result.grid = GRID;
    result.sims_run_this_call = it_count;
    result.n_iterations_total = numel(iters);
    result.monotonic_checked = ~isempty(mono_ok);
    if ~isempty(mono_ok)
        result.monotonic_fraction = mono_frac;
        result.monotonic_ok = mono_ok;
    end
    result.wall_clock_s = toc(t_total);
    result.model_file = model_file;
    write_json(fullfile(OUTPUT_DIR, sprintf('bisect_%s_result.json', id)), result);
    fprintf('Wrote %s\n', fullfile(OUTPUT_DIR, sprintf('bisect_%s_result.json', id)));

    if bdIsLoaded(MODEL_NAME); close_system(MODEL_NAME, 0); end

    % ---------------------------------------------------------------------
    %  nested helpers (share the variables above)
    % ---------------------------------------------------------------------
    function [iters_out, rec_out] = run_or_reuse(gi)
        iters_out = iters;
        hit = find([iters_out.grid_index] == gi, 1);
        if ~isempty(hit)
            rec_out = iters_out(hit);
            fprintf('  (grid %d = %.3f already simulated, reusing)\n', gi, GRID(gi));
            return;
        end
        rec_out = simulate_fraction(gi);
        iters_out(end+1) = rec_out;
        save_iterations(iters_out);
    end

    function rec = simulate_fraction(gi)
        frac = GRID(gi);
        fprintf('Simulating ShedFrac = %.3f ...\n', frac);

        if bdIsLoaded(MODEL_NAME); close_system(MODEL_NAME, 0); end
        load_system(model_file);
        set_param(MODEL_NAME, 'StopTime', num2str(TEND));
        ws0 = warning('off', 'all');
        for bb = 1:NB
            H.ensure_constant_block(MODEL_NAME, BUS{bb}, ...
                sprintf('%s/ShedFrac_%s_default', MODEL_NAME, BUS{bb}), 1);
            rb = sprintf('%s/ShedFrac_%s_ramp', MODEL_NAME, BUS{bb});
            if getSimulinkBlockHandle(rb) ~= -1
                set_param(rb, 'RisingSlewLimit', 'inf', 'FallingSlewLimit', '-inf');
            end
        end
        for bb = 1:NB
            assignin('base', sprintf('Pload_%s', BUS{bb}), load_hist(:, bb).');
        end
        H.wire_all_taps(MODEL_NAME, BUS, CONV);
        for kk = 1:n_shed
            H.set_shed_step(MODEL_NAME, shed_bus{kk}, ...
                sprintf('%s/ShedFrac_%s_default', MODEL_NAME, shed_bus{kk}), ...
                trig_time(kk), 1, frac);
        end
        warning(ws0);

        tsim = tic;
        ws1 = warning('off', 'all');
        so = sim(MODEL_NAME, 'ReturnWorkspaceOutputs', 'on');
        warning(ws1);
        wc = toc(tsim);

        coll = {}; ons = []; mvs = nan(1, NB);
        for bb = 1:NB
            v_full = H.grab_series(so, sprintf('V_Bus_%s', BUS{bb}), TS, N_STEPS);
            [c, on, mv] = H.detect_collapse_summary(time_vec, v_full, ...
                COLLAPSE_VOLTAGE_V, MIN_CONSECUTIVE_SAMPLES);
            mvs(bb) = mv;
            if c
                coll{end+1} = BUS{bb}; %#ok<AGROW>
                ons(end+1) = on; %#ok<AGROW>
            end
            fprintf('   %s %s min_v=%.2f%s\n', BUS{bb}, ternary(c, 'COLLAPSE', 'survives'), mv, ...
                    ternary(ismember(BUS{bb}, shed_bus), '  [shed]', ''));
        end
        fprintf('   sim wall clock %.1f s\n', wc);
        rec = struct('grid_index', gi, 'shedfrac', frac, 'all_safe', isempty(coll), ...
                     'collapsed_buses', {coll}, 'onsets', ons, 'min_v', mvs, 'wall_clock_s', wc);
    end

    function save_iterations(it)
        s = struct();
        s.scenario_id = id;
        s.shed_buses = shed_bus;
        s.trigger_time_by_bus = trig_time;
        s.grid = GRID;
        s.bus_order = BUS;
        s.iterations = it;
        write_json(iter_path, s);
    end
end


%% ---- local helpers ----------------------------------------------------

function row = get_row(tbl, scenario)
    if isnumeric(scenario)
        idx = scenario;
    else
        idx = find(strcmp(string(tbl.scenario_id), string(scenario)), 1);
    end
    if isempty(idx) || idx < 1 || idx > height(tbl)
        error('bisect_scenario:noRow', 'Scenario %s not found in scenario_table.csv', string(scenario));
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


function c = as_cellstr(x)
    % jsondecode returns [] , a char vector or a cell array depending on how many
    % buses collapsed; normalise to a 1xN cell array of chars.
    if isempty(x)
        c = {};
    elseif ischar(x)
        c = {x};
    elseif isstring(x)
        c = cellstr(x(:))';
    else
        c = x(:)';
    end
end


function write_json(path, s)
    fid = fopen(path, 'w');
    fprintf(fid, '%s', jsonencode(s));
    fclose(fid);
end
