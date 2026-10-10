function result = run_staged_limiter(scenario, opts)
% ==========================================================================
%  RUN_STAGED_LIMITER.M   (Option B, step 2: closed-loop check of the load limiter)
%  --------------------------------------------------------------------
%  Staged shedding as a LOAD LIMITER. After each shed bus's detector trigger time,
%  that bus's load is held at or below a power cap:
%        load_used(t) = min( load(t), CAP )      for t >= trigger time
%  Below the cap nothing is cut. As the load ramp climbs past the cap the cut grows by
%  itself (ShedFrac = CAP / load). No forecast of the final load is needed.
%
%  THE SIMULINK MODEL IS NOT EDITED. The cap is applied to the Pload_<bus> arrays
%  (the same variables run_scenario.m / bisect_scenario.m already write before every
%  run). Every ShedFrac block stays at 1. The model is closed without saving.
%
%  What one call does for ONE scenario
%    1. (opts.validate = true) two checks against results we already trust:
%         a. cap = Inf  -> must reproduce the no-shed run (same buses collapse)
%         b. the bisection's best constant ShedFrac applied through the arrays
%            (not through a Step block) -> must still be safe. This proves that
%            cutting the load array is equivalent to the ShedFrac block.
%    2. Try the cap from staged_caps.csv (column cap_kw_margin) plus each offset in
%       opts.cap_offsets_kw (default [0 -10 -20 -30] kW), most permissive first,
%       and STOP at the first cap that keeps all 10 buses alive for 5000 s.
%    3. Report shed energy (kWh) of the limiter vs. the bisection's constant cut.
%
%  Needs the same earlier-step files as bisect_scenario.m (same output folder):
%    noshed_<ID>_collapse.csv, detector_triggers_<ID>.json
%    bisect_<ID>_result.json is optional (used for the comparison and check 1b)
%
%  Usage
%    run_staged_limiter('S012')
%    run_staged_limiter(12, struct('validate', true))
%    run_staged_limiter('S012', struct('smoke_test', true))     % 200 s wiring check
%
%  opts (all optional)
%    caps_csv         default <this folder>/staged_caps.csv  (from make_staged_caps.py)
%    cap_offsets_kw   default [0 -10 -20 -30]
%    validate         default false
%    staged           default false. true -> CAUSAL STAGED cap: the cap of every limited bus follows the
%                     number of buses the detector has flagged SO FAR (and whether E or F is among them),
%                     read from staged_cap_groups.csv. Without it the cap is the one for the FINAL group.
%    groups_csv       default <this folder>/staged_cap_groups.csv (used only with staged)
%    tag              default ''. Adds _<tag> to staged_<ID>_result.json / _iterations.json so runs of
%                     different policies do not overwrite each other. smoke_test uses tag 'smoke'.
%    smoke_test       true -> stop time 200 s, one sim, wiring only
%    tend             stop time, default 5000
%    meas_delay_s     default 0. MEASUREMENT TEST: the limiter sees each bus's load this many seconds late.
%    meas_noise_pct   default 0. MEASUREMENT TEST: the limiter sees load*(1+e), e ~ Normal(0, pct/100),
%                     a new value for every sensor sample (see meas_period_s). Plant load is NOT noisy.
%    meas_period_s    default 1. Sensor sample period: the reading is held between samples.
%    meas_seed        default 1. Seed of the noise (each limited bus gets seed+k, so buses differ).
%                     With any of the above on, the limiter works from the MEASURED load m:
%                        load_used = load * min(1, cap / m)      (with m = load this equals min(load, cap))
%                     The true load still drives the plant. Use a different tag for every setting.
%    output_root      default env DATASET_OUTPUT_ROOT, else <project>/model_outputs/dataset_pipeline
%    model_file       override the .slx path
%  Resumable: every finished simulation is written to staged_<ID>_iterations.json;
%  a re-run reuses entries with the same label and cap.
%  Output: staged_<ID>_result.json
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
    FALLBACK_LEAD_S = 60;

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
    OFFSETS = get_opt(opts, 'cap_offsets_kw', [0 -10 -20 -30]);
    do_validate = get_opt(opts, 'validate', false);
    STAGED = get_opt(opts, 'staged', false);
    TAG = get_opt(opts, 'tag', '');
    if smoke && isempty(TAG); TAG = 'smoke'; end
    SUF = ternary(isempty(TAG), '', ['_' TAG]);
    GCAP = [];
    MEAS_D    = get_opt(opts, 'meas_delay_s', 0);
    MEAS_N    = get_opt(opts, 'meas_noise_pct', 0);
    MEAS_P    = get_opt(opts, 'meas_period_s', 1);
    MEAS_SEED = get_opt(opts, 'meas_seed', 1);
    MEAS_ON   = MEAS_D > 0 || MEAS_N > 0;

    %% 1. Scenario row, caps, earlier-step outputs ---------------------------
    tbl = readtable(fullfile(SCRIPT_DIR, 'scenario_table.csv'), 'TextType', 'string');
    row = get_row(tbl, scenario);
    id  = row.scenario_id;

    caps_csv = get_opt(opts, 'caps_csv', fullfile(SCRIPT_DIR, 'staged_caps.csv'));
    if exist(caps_csv, 'file') ~= 2
        error('run_staged_limiter:caps', 'Run make_staged_caps.py first: %s not found.', caps_csv);
    end
    caps = readtable(caps_csv, 'TextType', 'string');
    ci = find(strcmp(string(caps.scenario_id), string(id)), 1);
    if isempty(ci)
        error('run_staged_limiter:caps', '%s not in %s', id, caps_csv);
    end
    CAP_BASE_KW = double(caps.cap_kw_margin(ci));
    if STAGED
        grp_csv = get_opt(opts, 'groups_csv', fullfile(SCRIPT_DIR, 'staged_cap_groups.csv'));
        if exist(grp_csv, 'file') ~= 2
            error('run_staged_limiter:groups', 'staged needs %s', grp_csv);
        end
        gt = readtable(grp_csv);
        GCAP = nan(3, 2);      % rows: 1,2,3 flagged buses; columns: no E/F, with E/F
        for gi = 1:height(gt)
            GCAP(gt.n_target(gi), gt.has_ef(gi) + 1) = gt.cap_kw_margin(gi);
        end
        if any(isnan(GCAP(:)))
            error('run_staged_limiter:groups', 'staged_cap_groups.csv must cover n_target 1..3 x has_ef 0/1');
        end
    end

    env_root = getenv('DATASET_OUTPUT_ROOT');
    if isempty(env_root)
        env_root = fullfile(PROJECT_ROOT, 'model_outputs', 'dataset_pipeline');
    end
    out_root   = get_opt(opts, 'output_root', env_root);
    OUTPUT_DIR = fullfile(out_root, id);

    collapse_csv = fullfile(OUTPUT_DIR, sprintf('noshed_%s_collapse.csv', id));
    trig_json    = fullfile(OUTPUT_DIR, sprintf('detector_triggers_%s.json', id));
    bisect_json  = fullfile(OUTPUT_DIR, sprintf('bisect_%s_result.json', id));
    if exist(collapse_csv, 'file') ~= 2
        error('run_staged_limiter:missing', 'Run step 2 first: %s not found.', collapse_csv);
    end
    ct = readtable(collapse_csv, 'TextType', 'string');
    ct_bus = cellstr(ct.bus);
    ct_collapsed = logical(ct.collapsed);
    ct_onset = double(ct.onset_s);

    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('STAGED LOAD LIMITER: %s (%s)%s\n', id, row.kind, ternary(smoke, '  [SMOKE TEST]', ''));
    fprintf('%s\n', repmat('=', 1, 70));

    result = struct();
    result.scenario_id = id;
    result.kind = row.kind;
    result.policy = 'load limiter: load_used = min(load, cap) after the bus trigger time';
    if STAGED
        result.policy = 'causal staged load limiter: cap follows the number of buses flagged so far';
    end
    result.staged = STAGED;
    result.tag = TAG;
    result.meas_delay_s = MEAS_D;
    result.meas_noise_pct = MEAS_N;
    result.meas_period_s = MEAS_P;
    result.meas_seed = MEAS_SEED;
    if MEAS_ON
        fprintf('MEASUREMENT TEST: limiter sees the load with delay %.1f s, noise %.1f %% (period %.2f s, seed %d)\n', ...
                MEAS_D, MEAS_N, MEAS_P, MEAS_SEED);
    end

    if ~any(ct_collapsed)
        % Nothing collapses without shedding. The limiter must also be harmless: run it
        % once on the buses the detector flagged (or none) -- here we only record it.
        fprintf('No bus collapses in the no-shed run: nothing to protect. (Recorded only.)\n');
        result.status = 'no_collapse';
        write_json(fullfile(OUTPUT_DIR, sprintf('staged_%s%s_result.json', id, SUF)), result);
        return;
    end
    if exist(trig_json, 'file') ~= 2
        error('run_staged_limiter:missing', 'Run step 3 first: %s not found.', trig_json);
    end
    det = jsondecode(fileread(trig_json));

    %% 2. Who is limited and from when (same rule as bisect_scenario.m) ------
    shed_idx = find(ct_collapsed)';
    shed_bus = ct_bus(shed_idx)';
    n_shed = numel(shed_bus);
    trig_time = nan(1, n_shed);
    for k = 1:n_shed
        b = shed_bus{k};
        onset_k = ct_onset(shed_idx(k));
        t_det = [];
        if isfield(det.buses, b) && isfield(det.buses.(b), 'trigger_time')
            t_det = det.buses.(b).trigger_time;
        end
        if ~isempty(t_det) && ~isnan(t_det) && t_det < onset_k
            trig_time(k) = t_det;
        else
            trig_time(k) = max(0, onset_k - FALLBACK_LEAD_S);
        end
        fprintf('  Limited bus %s: trigger %.2f s (no-shed onset %.2f s)\n', b, trig_time(k), onset_k);
    end
    shed_col = cellfun(@(b) find(strcmp(BUS, b)), shed_bus);
    fprintf('Base cap (table value minus margin): %.1f kW per limited bus\n', CAP_BASE_KW);
    if STAGED
        fprintf('STAGED mode: cap per flagged-bus group (kW, rows 1-3 buses; cols no E/F | with E/F):\n');
        disp(GCAP);
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

    best_const = NaN;
    if exist(bisect_json, 'file') == 2
        bj = jsondecode(fileread(bisect_json));
        if isfield(bj, 'best_shedfrac') && ~isempty(bj.best_shedfrac) && ~isnan(bj.best_shedfrac)
            best_const = bj.best_shedfrac;
        end
    end
    result.bisection_best_shedfrac = best_const;
    result.shed_buses = shed_bus;
    result.trigger_time_by_bus = trig_time;
    result.base_cap_kw = CAP_BASE_KW;

    %% 4. Resume cache -------------------------------------------------------
    iter_path = fullfile(OUTPUT_DIR, sprintf('staged_%s%s_iterations.json', id, SUF));
    iters = {};
    if exist(iter_path, 'file') == 2 && ~smoke
        prev = jsondecode(fileread(iter_path));
        if isfield(prev, 'trigger_time_by_bus') && isequal(sort(prev.shed_buses(:))', sort(shed_bus(:))') ...
                && max(abs(prev.trigger_time_by_bus(:)' - trig_time)) < 1e-6 ...
                && ((~isfield(prev, 'tend') && TEND == 5000) || (isfield(prev, 'tend') && prev.tend == TEND)) ...
                && get_opt(prev, 'meas_delay_s', 0) == MEAS_D && get_opt(prev, 'meas_noise_pct', 0) == MEAS_N ...
                && get_opt(prev, 'meas_period_s', 1) == MEAS_P && get_opt(prev, 'meas_seed', 1) == MEAS_SEED
            for q = 1:numel(prev.iterations)
                if iscell(prev.iterations)
                    iters{end+1} = prev.iterations{q}; %#ok<AGROW>
                else
                    iters{end+1} = prev.iterations(q); %#ok<AGROW>
                end
            end
            fprintf('Resuming: %d simulation(s) already on disk.\n', numel(iters));
        end
    end

    %% 5. Optional validation sims -------------------------------------------
    if do_validate && ~smoke
        fprintf('\n--- VALIDATION 1: cap = Inf must reproduce the no-shed run ---\n');
        r1 = run_policy('validate_noshed', Inf, NaN);
        same_set = isequal(sort(r1.collapsed_buses(:))', sort(ct_bus(ct_collapsed))');
        onset_diff = NaN;
        if same_set && ~isempty(r1.onsets)
            [~, o] = sort(r1.collapsed_buses); [~, o2] = sort(ct_bus(ct_collapsed));
            a1 = r1.onsets(o); a2 = ct_onset(ct_collapsed); a2 = a2(o2)';
            onset_diff = max(abs(a1(:)' - a2(:)'));
        end
        result.validate_noshed_same_buses = same_set;
        result.validate_noshed_max_onset_diff_s = onset_diff;
        fprintf('   same collapsing buses: %s | max onset difference %.3f s\n', ...
                ternary(same_set, 'YES', 'NO'), onset_diff);

        if ~isnan(best_const)
            fprintf('\n--- VALIDATION 2: bisection best ShedFrac %.2f applied through the arrays ---\n', best_const);
            r2 = run_policy('validate_const_best', NaN, best_const);
            result.validate_const_best_safe = r2.all_safe;
            fprintf('   %s (the bisection found this fraction safe)\n', ternary(r2.all_safe, 'SAFE (consistent)', 'NOT SAFE -> array cut is NOT equivalent, stop and look'));
        end
    end

    %% 6. The limiter: most permissive cap first, stop at the first safe one --
    result.cap_tried_kw = [];
    result.cap_safe = [];
    first_safe = NaN;
    cap_list = CAP_BASE_KW + OFFSETS(:)';
    cap_list = cap_list(cap_list > 5);
    for q = 1:numel(cap_list)
        cap_kw = cap_list(q);
        lbl = sprintf('%s_%.1fkW', ternary(STAGED, 'staged', 'limiter'), cap_kw);
        fprintf('\n--- LIMITER with cap %.1f kW per limited bus ---\n', cap_kw);
        r = run_policy(lbl, cap_kw * 1000, NaN);
        result.cap_tried_kw(end+1) = cap_kw;
        result.cap_safe(end+1) = r.all_safe;
        result.(matlab.lang.makeValidName(sprintf('shed_kwh_cap_%d', round(cap_kw)))) = r.shed_kwh;
        fprintf('   %s | shed energy %.2f kWh | deepest cut %.1f %% | cap first reached %s | lowest bus voltage %.1f V\n', ...
                ternary(r.all_safe, 'ALL 10 BUSES SURVIVE', ['COLLAPSE: ' strjoin(r.collapsed_buses, ',')]), ...
                r.shed_kwh, r.max_cut_pct, num2str(r.first_cut_s), min(r.min_v));
        if r.all_safe
            first_safe = cap_kw; result.safe_policy = r;
            break;
        end
        if smoke; break; end
    end

    %% 7. Comparison with the constant cut -----------------------------------
    if ~isnan(best_const)
        kwh_const = 0;
        for k = 1:n_shed
            idx = time_vec >= trig_time(k);
            kwh_const = kwh_const + (1 - best_const) * sum(load_hist(idx, shed_col(k))) * TS / 3.6e6;
        end
        result.constant_cut_shed_kwh = kwh_const;
        fprintf('\nConstant cut (bisection, ShedFrac %.2f) sheds %.2f kWh in total.\n', best_const, kwh_const);
    end
    if ~isnan(first_safe)
        result.status = 'safe_found';
        result.first_safe_cap_kw = first_safe;
        result.first_safe_offset_kw = first_safe - CAP_BASE_KW;
        fprintf('RESULT: first safe cap = %.1f kW (offset %+.0f kW from the table value).\n', first_safe, first_safe - CAP_BASE_KW);
    else
        result.status = 'no_safe_cap_tried';
        fprintf('RESULT: none of the tried caps kept all buses alive.\n');
    end
    write_json(fullfile(OUTPUT_DIR, sprintf('staged_%s%s_result.json', id, SUF)), result);
    if bdIsLoaded(MODEL_NAME); close_system(MODEL_NAME, 0); end

    % ---------------------------------------------------------------------
    %  nested helpers
    % ---------------------------------------------------------------------
    function rec = run_policy(label, cap_w, const_frac)
        % cap_w: Inf = no change, finite = limiter. const_frac: NaN = limiter/none,
        % else a constant ShedFrac from each bus's trigger time (array version).
        for qq = 1:numel(iters)
            it = iters{qq};
            if strcmp(it.label, label)
                rec = it;
                rec.collapsed_buses = as_cellstr(rec.collapsed_buses);
                fprintf('   (already simulated: %s, reusing)\n', label);
                return;
            end
        end
        load_run = load_hist;
        cap_series = [];
        if STAGED && isfinite(cap_w) && isnan(const_frac)
            ge_t  = time_vec >= trig_time;                       % N x n_shed: bus j flagged by time t
            n_fl  = max(min(sum(ge_t, 2), 3), 1);
            ef_fl = any(ge_t(:, ismember(shed_bus, {'E', 'F'})), 2);
            lin   = sub2ind(size(GCAP), n_fl, double(ef_fl) + 1);
            cap_series = GCAP(lin) * 1000 + (cap_w - CAP_BASE_KW * 1000);   % W; offset = retry step
        end
        for kk = 1:n_shed
            idx = time_vec >= trig_time(kk);
            col = shed_col(kk);
            if ~isnan(const_frac)
                load_run(idx, col) = load_hist(idx, col) * const_frac;
            elseif isfinite(cap_w)
                if MEAS_ON
                    % the limiter works from the MEASURED load (late and/or noisy); the plant gets the true load times the factor
                    if STAGED
                        capv = cap_series;
                    else
                        capv = cap_w * ones(N_STEPS, 1);
                    end
                    mload = measure_load(load_hist(:, col), MEAS_D, MEAS_N, MEAS_P, MEAS_SEED + kk, TS);
                    fac = min(1, capv ./ max(mload, 1));
                    load_run(idx, col) = load_hist(idx, col) .* fac(idx);
                elseif STAGED
                    load_run(idx, col) = min(load_hist(idx, col), cap_series(idx));
                else
                    load_run(idx, col) = min(load_hist(idx, col), cap_w);
                end
            end
        end
        cut = load_hist(:, shed_col) - load_run(:, shed_col);
        shed_kwh = sum(cut(:)) * TS / 3.6e6;
        frac_cut = cut ./ max(load_hist(:, shed_col), 1);
        max_cut_pct = 100 * max(frac_cut(:));
        first_cut_s = NaN;
        anycut = any(cut > 1, 2);
        if any(anycut); first_cut_s = time_vec(find(anycut, 1)); end

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
            assignin('base', sprintf('Pload_%s', BUS{bb}), load_run(:, bb).');
        end
        H.wire_all_taps(MODEL_NAME, BUS, CONV);
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
                    ternary(ismember(BUS{bb}, shed_bus), '  [limited]', ''));
        end
        fprintf('   sim wall clock %.1f s\n', wc);
        rec = struct('label', label, 'cap_w', cap_w, 'const_frac', const_frac, ...
                     'all_safe', isempty(coll), 'collapsed_buses', {coll}, 'onsets', ons, ...
                     'min_v', mvs, 'shed_kwh', shed_kwh, 'max_cut_pct', max_cut_pct, ...
                     'first_cut_s', first_cut_s, 'wall_clock_s', wc);
        iters{end+1} = rec;
        s = struct('scenario_id', id, 'shed_buses', {shed_bus}, ...
                   'trigger_time_by_bus', trig_time, 'bus_order', {BUS}, 'iterations', {iters}, 'tend', TEND, ...
                   'meas_delay_s', MEAS_D, 'meas_noise_pct', MEAS_N, 'meas_period_s', MEAS_P, 'meas_seed', MEAS_SEED);
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
        error('run_staged_limiter:noRow', 'Scenario %s not found in scenario_table.csv', string(scenario));
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


function m = measure_load(x, delay_s, noise_pct, period_s, seed, ts)
% What a sensor would report for the load column x (one value per simulation step ts):
% a reading is taken every period_s and held until the next one, it arrives delay_s late,
% and it carries a multiplicative Gaussian error of noise_pct percent (one draw per reading).
    n = numel(x);
    step = max(1, round(period_s / ts));
    dsamp = round(delay_s / ts);
    k = (1:n)';
    src = max(1, floor((k - 1 - dsamp) / step) * step + 1);   % index of the reading in force at step k
    m = x(src);
    if noise_pct > 0
        nS = ceil(n / step) + 1;
        rs = RandStream('mt19937ar', 'Seed', seed);
        e = randn(rs, nS, 1);
        m = m .* max(0, 1 + (noise_pct / 100) * e((src - 1) / step + 1));
    end
end


function v = get_opt(opts, name, default)
    if isfield(opts, name); v = opts.(name); else; v = default; end
end


function out = ternary(cond, a, b)
    if cond; out = a; else; out = b; end
end


function c = as_cellstr(x)
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

