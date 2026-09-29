function generate_gradual_busC_scenario_v2()
% ==========================================================================
%  GENERATE_GRADUAL_BUSC_SCENARIO_V2.M
%  --------------------------------------------------------------------
%  Iteration 2 of the fresh gradual-load scenario (v1 = generate_gradual_
%  busC_scenario.m, kept for comparison, not deleted -- same spirit as the
%  earlier abandoned attempt's own documented v1/v2/v3 iteration).
%
%  WHAT V1 FOUND: pushing Bus C alone (a 65,000 W boost on top of ~27,000 W
%  ambient, neighbors B/D left flat) got BC and CD to 164.38 C -- real
%  derating (down to 21% capacity) but short of the 175 C full-cutoff line,
%  so Bus C never got isolated and never collapsed. Notably, 164.38 C is
%  almost identical to the OLD abandoned attempt's own near-miss (163.9 C),
%  despite a larger demand differential here. That similarity, across two
%  different scenario designs, suggests this model's derate curve has a
%  self-limiting feedback loop (as a converter derates, it carries less
%  power, which reduces further heating) that naturally stabilizes around
%  ~164 C for a wide range of moderate differentials -- simply nudging the
%  differential up a little more is unlikely to break through it.
%
%  WHAT CHANGES IN V2, AND WHY:
%   1. Bus C is pushed all the way to this model's hard demand ceiling
%      (PMAX = 100,000 W), using the full remaining headroom from v1.
%   2. Bus B and Bus D (C's direct neighbors) are now pushed DOWN toward
%      the model's hard floor (PMIN = 20,000 W), instead of staying flat --
%      widens the BC/CD differential further than v1 could reach by
%      boosting C alone.
%   3. NEW: Bus A and Bus E (the next buses out, one hop past B and D) get
%      a moderate boost too. This isn't just a bigger push on the same two
%      converters -- it deliberately widens the stressed region to also
%      compound the AB and DE differentials (which v1 showed already
%      derate as a side effect), giving more than one path to a real
%      cutoff instead of resting everything on BC/CD alone crossing 175 C.
%   4. The ramp starts earlier and finishes earlier (500s -> 2000s instead
%      of 800s -> 2800s), which stretches the sustained-heavy hold from
%      v1's 2200s to 3000s within the SAME 5000s total run length -- rules
%      out "it just needed more time to fully heat-soak" as an explanation
%      if this still doesn't cross 175 C, without breaking the 500,001-row
%      / 5000s schema every other scenario in this project uses.
%
%  Buses F, G, H, K, L are untouched (pure ambient), same as v1 -- keeps
%  half the ring as an unaffected control region.
%
%  RUNTIME: one 5000s sim() call, ~30-40 minutes wall clock (v1 took
%  2224.8s = ~37 min; expect similar).
%
%  Run:
%    generate_gradual_busC_scenario_v2
% ==========================================================================

    clc;

    %% 0. CONFIG ----------------------------------------------------------
    PROJECT_ROOT = 'D:\naren\Documents\Thermal_Derating_Work_Handover\ThermalProject';
    MODEL_NAME = 'Grid_modelling_Thermal_V7_ALfix';

    TS   = 0.01;
    TEND = 5000;
    N_STEPS = round(TEND / TS) + 1;
    PMIN = 20000;
    PMAX = 100000;
    SEED = 102;   % new seed for this iteration

    BUS  = {'A','B','C','D','E','F','G','H','K','L'};
    CONV = {'AB','BC','CD','DE','EF','FG','GH','HK','KL','AL'};

    RAMP_START_S = 500;
    RAMP_END_S   = 2000;
    C_BOOST_W    = 73000;   % Bus C: 27000 + 73000 = 100000 = PMAX (full ceiling)
    BD_DROP_W    = 7000;    % Bus B/D: 27000 - 7000  = 20000  = PMIN (full floor)
    AE_BOOST_W   = 15000;   % Bus A/E: 27000 + 15000 = 42000  (moderate, compounds AB/DE)

    OUTPUT_DIR = fullfile(PROJECT_ROOT, 'model_outputs', 'thermal_derating_gradual_busC');
    if exist(OUTPUT_DIR, 'dir') ~= 7; mkdir(OUTPUT_DIR); end
    addpath(PROJECT_ROOT);

    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('GRADUAL LOAD SCENARIO v2: wider Bus-C isolation push\n');
    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('Settle phase:    0 - %ds\n', RAMP_START_S);
    fprintf('Ramp phase:      %d - %ds\n', RAMP_START_S, RAMP_END_S);
    fprintf('Sustained heavy: %d - %ds (%d s hold, up from v1''s 2200s)\n', ...
            RAMP_END_S, TEND, TEND - RAMP_END_S);
    fprintf('Bus C  -> %.0f W (ceiling, PMAX)\n', 27000 + C_BOOST_W);
    fprintf('Bus B,D -> %.0f W (floor, PMIN)\n', 27000 - BD_DROP_W);
    fprintf('Bus A,E -> %.0f W (moderate boost)\n', 27000 + AE_BOOST_W);

    %% 1. Locate and load the model ----------------------------------------
    hh = dir(fullfile(PROJECT_ROOT, '**', [MODEL_NAME '.slx']));
    if isempty(hh)
        error('Base model not found: %s.slx', MODEL_NAME);
    end
    model_file = fullfile(hh(1).folder, hh(1).name);
    fprintf('\nModel: %s\n', model_file);
    if bdIsLoaded(MODEL_NAME)
        close_system(MODEL_NAME, 0);
    end
    load_system(model_file);
    set_param(MODEL_NAME, 'StopTime', num2str(TEND));

    %% 2. Build the new load profile ---------------------------------------
    fprintf('\nBuilding v2 load profile (seed %d) ...\n', SEED);
    [load_hist, clip_hi, clip_lo] = build_gradual_busC_loads_v2( ...
        SEED, N_STEPS, TS, BUS, PMIN, PMAX, RAMP_START_S, RAMP_END_S, ...
        C_BOOST_W, BD_DROP_W, AE_BOOST_W);
    if any(clip_hi > 5) || any(clip_lo > 5)
        fprintf('  NOTE: clipping >5%% of samples on at least one bus (clip_hi max %.1f%%,\n', max(clip_hi));
        fprintf('  clip_lo max %.1f%%) -- EXPECTED here: Bus C is deliberately pinned at\n', max(clip_lo));
        fprintf('  PMAX and Bus B/D at PMIN for most of the heavy phase, not a bug.\n');
    end

    %% 3. Wire taps (proven pilot infrastructure, reused as-is) ------------
    fprintf('\nWiring signal taps ...\n');
    wire_all_taps(MODEL_NAME, BUS, CONV);
    derate_var_names   = tap_derate_factors(MODEL_NAME, CONV);
    junction_var_names = tap_junction_temp(MODEL_NAME, CONV);
    phase_var_names    = tap_phase_commands(MODEL_NAME, CONV);

    %% 4. Run the simulation -------------------------------------------------
    fprintf('\nRunning simulation ...\n');
    tic;
    ws = warning('off', 'all');
    so = sim(MODEL_NAME, 'ReturnWorkspaceOutputs', 'on');
    warning(ws);
    fprintf('Finished in %.1f s wall clock.\n', toc);

    time_vec = (0:N_STEPS-1)' * TS;

    %% 5. Assemble the CSV (schema matches step33's expectations) -----------
    T = table(time_vec, 'VariableNames', {'time'});
    src_pow_matrix = nan(N_STEPS, numel(BUS));
    for b = 1:numel(BUS)
        T.(sprintf('V_Bus_%s', BUS{b})) = grab_series(so, sprintf('V_Bus_%s', BUS{b}), TS, N_STEPS);
        src_pow_matrix(:, b) = grab_series(so, sprintf('Bus_%s_Src_Pow', BUS{b}), TS, N_STEPS);
        T.(sprintf('Bus_%s_Src_Pow', BUS{b})) = src_pow_matrix(:, b);
        T.(sprintf('Bus_%s_Temp', BUS{b})) = grab_series(so, sprintf('Bus_%s_Temp', BUS{b}), TS, N_STEPS);
    end

    for b = 1:numel(BUS)
        T.(sprintf('CommandedLoad_kW_%s', BUS{b})) = load_hist(:, b) / 1000;
    end

    EPSILON_W = 500.0;
    avg_power = mean(src_pow_matrix, 2, 'omitnan');
    grid_collapsed = avg_power < EPSILON_W;
    for b = 1:numel(BUS)
        ratio = src_pow_matrix(:, b) ./ avg_power;
        ratio(grid_collapsed) = 0.0;
        T.(sprintf('GEI_%s', BUS{b})) = ratio;
    end
    T.Src_Pow_Avg_recomputed = avg_power;

    for c = 1:numel(CONV)
        T.(sprintf('Phase_%s_cmd_deg', CONV{c})) = grab_series(so, phase_var_names{c}, TS, N_STEPS);
        T.(sprintf('DAB_%s_Derate_Factor', CONV{c})) = grab_series(so, derate_var_names{c}, TS, N_STEPS) / 100e3;
        T.(sprintf('JunctionTemp_C_%s', CONV{c})) = grab_series(so, junction_var_names{c}, TS, N_STEPS);
    end
    hs_conv_order = {'AB','BC','CD','DE','EF','FG','GH','HK','KL','AL'};
    hs_tags = {'DAB_AB_H_Temp','DAB_BC_H_Temp','DAB_CD_H_Temp','DAB_DE_H_Temp', ...
               'DAB_EF_H_Temp','DAB_FG_H_Temp','DAB_GH_Temp','DAB_HK_Temp', ...
               'DAB_KL_Temp','DAB_LA_Temp'};
    for h = 1:numel(hs_tags)
        T.(sprintf('HeatSinkTemp_C_%s', hs_conv_order{h})) = grab_series(so, hs_tags{h}, TS, N_STEPS);
    end

    %% 6. NaN/Inf validation --------------------------------------------------
    WARMUP_TOLERANCE_S = 1.0;
    log_path = fullfile(OUTPUT_DIR, 'gradual_busC_v2_validation_log.txt');
    log_fid = fopen(log_path, 'w');
    log_both(log_fid, '%s\n', repmat('=', 1, 70));
    log_both(log_fid, 'NaN/Inf VALIDATION: gradual_busC_v2 scenario\n');
    log_both(log_fid, '%s\n', repmat('=', 1, 70));

    data_cols = T.Properties.VariableNames(2:end);
    bad_mask = false(height(T), 1);
    any_unexpected = false;
    for ci = 1:numel(data_cols)
        col = data_cols{ci};
        v = T.(col);
        bad_here = isnan(v) | isinf(v);
        n_bad = sum(bad_here);
        if n_bad > 0
            bad_mask = bad_mask | bad_here;
            bad_times = time_vec(bad_here);
            n_after_warmup = sum(bad_times > WARMUP_TOLERANCE_S);
            log_both(log_fid, '  %-30s %6d bad samples total, %6d after warmup (t>%.1fs)\n', ...
                      col, n_bad, n_after_warmup, WARMUP_TOLERANCE_S);
            if n_after_warmup > 0
                any_unexpected = true;
            end
        end
    end
    if ~any(bad_mask)
        log_both(log_fid, '  No NaN/Inf found in any column. Clean.\n');
    end
    if any_unexpected
        log_both(log_fid, '\n  RESULT: FAILED VALIDATION. Unexpected NaN/Inf beyond warmup.\n');
        fclose(log_fid);
        error('gradual_busC_v2 scenario FAILED NaN/Inf validation. See log:\n%s', log_path);
    else
        log_both(log_fid, '\n  RESULT: PASSED VALIDATION. Safe to proceed.\n');
    end
    fclose(log_fid);

    out_csv = fullfile(OUTPUT_DIR, 'thermal_derating_gradual_busC_v2_5000s.csv');
    writetable(T, out_csv);
    fprintf('\nWrote: %s (%d rows x %d cols)\n', out_csv, height(T), width(T));

    %% 7. Health check + collapse summary -----------------------------------
    fprintf('\n%s\n', repmat('=', 1, 70));
    fprintf('CONVERTER HEALTH CHECK\n');
    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('%-6s %10s %10s %8s %8s %10s %8s\n', 'Conv', 'HS max', 'Tj max', '>125C?', '>175C?', 'Derate min', 'Derated?');
    for c = 1:numel(CONV)
        hs = T.(sprintf('HeatSinkTemp_C_%s', CONV{c}));
        tj = T.(sprintf('JunctionTemp_C_%s', CONV{c}));
        der = T.(sprintf('DAB_%s_Derate_Factor', CONV{c}));
        hs_max = max(hs, [], 'omitnan');
        tj_max = max(tj, [], 'omitnan');
        der_min = min(der, [], 'omitnan');
        over125 = tj_max > 125; over175 = tj_max > 175;
        fprintf('%-6s %9.2fC %9.2fC %8s %8s %10.3f %8s\n', CONV{c}, hs_max, tj_max, ...
            ternary(over125,'YES','no'), ternary(over175,'YES','no'), der_min, ternary(der_min<0.999,'YES','no'));
    end

    fprintf('\nPer-bus voltage collapse check (threshold 100V, 50-sample debounce):\n');
    any_collapse = false;
    for b = 1:numel(BUS)
        v = T.(sprintf('V_Bus_%s', BUS{b}));
        [c, onset, min_v] = detect_collapse_summary(time_vec, v, 100.0, 50);
        if c
            fprintf('  Bus %s: COLLAPSES at t=%.2fs (min V=%.2f)\n', BUS{b}, onset, min_v);
            any_collapse = true;
        else
            fprintf('  Bus %s: survives (min V=%.2f)\n', BUS{b}, min_v);
        end
    end
    if ~any_collapse
        fprintf('\n  Still no collapse. If Tj max on BC/CD is again close to (but under)\n');
        fprintf('  175C, that would strongly confirm the self-limiting-feedback theory --\n');
        fprintf('  at that point the next lever is a structurally different one (e.g.\n');
        fprintf('  stress three consecutive buses instead of one), not just more Watts.\n');
    end

    close_system(MODEL_NAME, 0);
end


%% ---- v2 load profile: wider push, same spirit as v1 --------------------

function [load_hist, clip_hi, clip_lo] = build_gradual_busC_loads_v2( ...
    seed, n_steps, ts, BUS, pmin, pmax, ramp_start_s, ramp_end_s, ...
    c_boost_w, bd_drop_w, ae_boost_w)

    rng(seed);
    t = (0:n_steps-1)' * ts;
    NB = numel(BUS);

    ambient_base = 27000 * ones(1, NB);
    idx_C = find(strcmp(BUS, 'C'));
    idx_B = find(strcmp(BUS, 'B'));
    idx_D = find(strcmp(BUS, 'D'));
    idx_A = find(strcmp(BUS, 'A'));
    idx_E = find(strcmp(BUS, 'E'));

    load_hist = nan(n_steps, NB);
    clip_hi   = nan(1, NB);
    clip_lo   = nan(1, NB);

    ramp_frac = min(1, max(0, (t - ramp_start_s) / (ramp_end_s - ramp_start_s)));

    for k = 1:NB
        phase_shift = 0.4 * k;
        slow_var = 4000 * sin(2*pi*t/1500 + phase_shift);
        noise    = 800  * randn(n_steps, 1);

        raw = ambient_base(k) + slow_var + noise;

        if k == idx_C
            raw = raw + c_boost_w * ramp_frac;
        elseif k == idx_B || k == idx_D
            raw = raw - bd_drop_w * ramp_frac;
        elseif k == idx_A || k == idx_E
            raw = raw + ae_boost_w * ramp_frac;
        end

        clip_hi(k) = mean(raw > pmax) * 100;
        clip_lo(k) = mean(raw < pmin) * 100;

        p = max(pmin, min(pmax, raw));
        load_hist(:, k) = p;
        assignin('base', sprintf('Pload_%s', BUS{k}), p(:).');
    end
end


%% ---- proven pilot tap-wiring infrastructure (reused verbatim) ----

function wire_all_taps(model, BUS, CONV) %#ok<INUSD>
    want_V    = strcat('V_Bus_',   BUS);
    want_P    = strcat('Bus_',     BUS, '_Src_Pow');
    want_T    = strcat('Bus_',     BUS, '_Temp');
    want_HS   = {'DAB_AB_H_Temp','DAB_BC_H_Temp','DAB_CD_H_Temp', ...
                 'DAB_DE_H_Temp','DAB_EF_H_Temp','DAB_FG_H_Temp', ...
                 'DAB_GH_Temp',  'DAB_HK_Temp',  'DAB_KL_Temp', ...
                 'DAB_LA_Temp'};
    all_wanted = [want_V, want_P, want_T, want_HS];

    ws = warning('off','all');
    gt = find_system(model,'FollowLinks','on','LookUnderMasks','all', ...
                     'BlockType','Goto');
    warning(ws);
    existing_tags = cellfun(@(b) get_param(b,'GotoTag'), gt, 'uni', 0);

    for k = 1:numel(all_wanted)
        tg = all_wanted{k};
        if ~ismember(tg, existing_tags)
            fprintf('    [skip] no Goto tag for %s\n', tg);
            continue;
        end
        wire_one_tap(model, tg);
    end
end


function wire_one_tap(model, tag)
    vn = matlab.lang.makeValidName(tag);
    from_name = ['V7F_' vn];
    tw_name   = ['V7L_' vn];

    for nm = {from_name, tw_name}
        ex = find_system(model, 'SearchDepth', 1, 'LookUnderMasks', 'all', 'Name', nm{1});
        if ~isempty(ex); try; delete_block(ex{1}); catch; end; end
    end

    stale = find_system(model, 'FollowLinks', 'on', 'LookUnderMasks', 'all', ...
                        'BlockType', 'ToWorkspace', 'VariableName', vn);
    for s = 1:numel(stale)
        try; delete_block(stale{s}); catch; end
    end

    try
        add_block('simulink/Signal Routing/From', [model '/' from_name], ...
                  'GotoTag', tag);
        add_block('simulink/Sinks/To Workspace', [model '/' tw_name], ...
                  'VariableName', vn, ...
                  'SaveFormat',   'Timeseries', ...
                  'SampleTime',   '0.01');
        pf = get_param([model '/' from_name], 'PortHandles');
        pl = get_param([model '/' tw_name],   'PortHandles');
        add_line(model, pf.Outport(1), pl.Inport(1), 'autorouting', 'on');
    catch ME
        warning('wire_one_tap:%s : %s', tag, ME.message);
    end
end


function var_names = tap_derate_factors(model, CONV)
    var_names = cell(1, numel(CONV));
    for i = 1:numel(CONV)
        var_names{i} = sprintf('V7DER_%s', CONV{i});
        dab_path = [model '/DAB_Model_' CONV{i}];
        if isempty(find_system(dab_path, 'SearchDepth', 1, 'LookUnderMasks', 'all', ...
                               'Name', 'thermal_derater'))
            warning('tap_derate_factors:noSubsystem', ...
                'No thermal_derater subsystem found under DAB_Model_%s.', CONV{i});
            continue;
        end
        derater_path = [dab_path '/thermal_derater'];
        ph = get_param(derater_path, 'PortHandles');
        if ~isfield(ph, 'Outport') || isempty(ph.Outport)
            warning('tap_derate_factors:noOutport', ...
                'thermal_derater under %s has no output port.', CONV{i});
            continue;
        end
        log_name = ['V7DERLOG_' var_names{i}];
        cleanup_and_add_tap(dab_path, log_name, var_names{i}, ph.Outport(1), model);
    end
end


function var_names = tap_junction_temp(model, CONV)
    var_names = cell(1, numel(CONV));
    for i = 1:numel(CONV)
        var_names{i} = sprintf('V7TJ_%s', CONV{i});
        dab_path = [model '/DAB_Model_' CONV{i}];
        outport_blk = find_system(dab_path, 'SearchDepth', 1, 'LookUnderMasks', 'all', ...
                                   'BlockType', 'Outport', 'Name', 'Temp_Junction');
        if numel(outport_blk) ~= 1
            warning('tap_junction_temp:noOutport', ...
                'Expected one Outport named Temp_Junction under DAB_Model_%s, found %d.', ...
                CONV{i}, numel(outport_blk));
            continue;
        end
        ph = get_param(outport_blk{1}, 'PortHandles');
        line = get_param(ph.Inport(1), 'Line');
        if line == -1
            warning('tap_junction_temp:noLine', ...
                'Temp_Junction Outport under DAB_Model_%s has no incoming line.', CONV{i});
            continue;
        end
        src_port = get_param(line, 'SrcPortHandle');
        log_name = ['V7TJLOG_' var_names{i}];
        cleanup_and_add_tap(dab_path, log_name, var_names{i}, src_port, model);
    end
end


function var_names = tap_phase_commands(model, CONV)
    var_names = cell(1, numel(CONV));
    for i = 1:numel(CONV)
        var_names{i} = sprintf('V7PHASE_%s', CONV{i});
        dab_path = [model '/DAB_Model_' CONV{i}];
        inner_inport_path = [dab_path '/phase_shift'];
        if getSimulinkBlockHandle(inner_inport_path) == -1
            warning('tap_phase_commands:noInport', ...
                'No Inport block named phase_shift found under DAB_Model_%s.', CONV{i});
            continue;
        end
        port_num = str2double(get_param(inner_inport_path, 'Port'));
        outer_ph = get_param(dab_path, 'PortHandles');
        if numel(outer_ph.Inport) < port_num
            warning('tap_phase_commands:badPort', ...
                'DAB_Model_%s does not have inport #%d at the top level.', CONV{i}, port_num);
            continue;
        end
        line = get_param(outer_ph.Inport(port_num), 'Line');
        if line == -1
            warning('tap_phase_commands:noLine', ...
                'DAB_Model_%s phase_shift inport has no incoming line at the top level.', CONV{i});
            continue;
        end
        src_port = get_param(line, 'SrcPortHandle');
        log_name = ['V7PHASELOG_' var_names{i}];
        cleanup_and_add_tap(model, log_name, var_names{i}, src_port, model);
    end
end


function cleanup_and_add_tap(parent, log_name, var_name, src_port, model)
    ex = find_system(parent, 'SearchDepth', 1, 'LookUnderMasks', 'all', 'Name', log_name);
    if ~isempty(ex); try; delete_block(ex{1}); catch; end; end

    stale = find_system(model, 'FollowLinks', 'on', 'LookUnderMasks', 'all', ...
                        'BlockType', 'ToWorkspace', 'VariableName', var_name);
    for s = 1:numel(stale)
        try; delete_block(stale{s}); catch; end
    end

    try
        add_block('simulink/Sinks/To Workspace', [parent '/' log_name], ...
            'VariableName', var_name, 'SaveFormat', 'Timeseries', 'SampleTime', '0.01');
        pl = get_param([parent '/' log_name], 'PortHandles');
        add_line(parent, src_port, pl.Inport(1), 'autorouting', 'on');
    catch ME
        warning('cleanup_and_add_tap:%s : %s', var_name, ME.message);
    end
end


function log_both(fid, fmt, varargin)
    fprintf(fmt, varargin{:});
    fprintf(fid, fmt, varargin{:});
end


function [collapsed, onset_t, min_v] = detect_collapse_summary(t, v, threshold, min_run)
    v = v(:);
    valid = ~isnan(v);
    if ~any(valid)
        collapsed = false; onset_t = NaN; min_v = NaN;
        return;
    end
    min_v = min(v(valid));
    below = v < threshold;
    below(~valid) = false;

    d = diff([false; below; false]);
    run_starts = find(d == 1);
    run_ends   = find(d == -1) - 1;
    run_lengths = run_ends - run_starts + 1;

    qualifying = find(run_lengths >= min_run, 1, 'first');
    if isempty(qualifying)
        collapsed = false; onset_t = NaN;
    else
        collapsed = true;
        onset_t = t(run_starts(qualifying));
    end
end


function y = grab_series(so, var_name, ts, n_steps)
    y = nan(n_steps, 1);
    v = try_get(so, matlab.lang.makeValidName(var_name));
    if isempty(v); return; end
    [~, col] = ts_vec(v, ts);
    n = min(n_steps, numel(col));
    y(1:n) = col(1:n);
end


function v = try_get(so, name)
    v = [];
    try; v = so.get(name); if ~isempty(v); return; end; catch; end
    try
        if evalin('base', sprintf('exist(''%s'',''var'')', name)) == 1
            v = evalin('base', name);
        end
    catch
    end
end


function [t, y] = ts_vec(x, ts)
    if isa(x, 'timeseries')
        t = x.Time; y = squeeze(x.Data); y = y(:);
    elseif isnumeric(x) && size(x,2) >= 2
        t = x(:,1); y = x(:,2);
    elseif isnumeric(x)
        y = x(:); t = (0:numel(y)-1)' * ts;
    else
        t = []; y = [];
    end
end


function out = ternary(cond, a, b)
    if cond; out = a; else; out = b; end
end