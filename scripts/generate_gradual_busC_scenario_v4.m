function generate_gradual_busC_scenario_v4()
% ==========================================================================
%  GENERATE_GRADUAL_BUSC_SCENARIO_V4.M
%  --------------------------------------------------------------------
%  v1, v2, and v3 all plateaued at ~164 C on BC/CD -- three different
%  topologies (single-bus, wider multi-bus, fast-ramp single-bus), all
%  saturating this model's assumed PMIN=20,000W / PMAX=100,000W range,
%  all landing on the same number to within 0.5 C. That convergence is
%  strong evidence that the achievable current differential WITHIN that
%  range cannot physically drive a converter past 175 C -- not that
%  gradual scenarios can't collapse a bus in general.
%
%  THE REALIZATION: PMIN/PMAX = 20000/100000 are not a verified Simulink
%  model constraint. They are clipping values chosen in the ORIGINAL
%  pilot dataset-generation script and copied forward, uncritically, into
%  every scenario script since (the old abandoned gradual attempt, this
%  project's stage1 multi-scenario script, and all three of my own v1-v3
%  attempts). Nothing has actually tested whether the model itself
%  rejects power commands outside that range -- it's simply never been
%  tried.
%
%  WHAT V4 TESTS: back to the clean single-bus topology from v1 (only
%  Bus C boosted, Bus B and Bus D dropped -- no A/E, deliberately NOT the
%  3-consecutive-bus shape, to keep this clearly distinct from
%  corridor_triad_collapse, which is being rebuilt separately). Same
%  ramp timing as v1 (800s -> 2800s, held to 5000s). The only thing that
%  changes is magnitude, and this time it goes OUTSIDE the assumed range:
%    Bus C  -> 200,000 W  (double the old "ceiling")
%    Bus B,D -> 3,000 W   (well under the old "floor")
%  The clip bounds themselves are loosened to [2000, 220000] so these
%  targets aren't silently re-clipped back to the old range.
%
%  RISK, STATED UP FRONT: pushing this far outside every prior script's
%  range could cause solver stiffness warnings, or a genuine NaN/Inf
%  failure at Stage 6 (the same hard-validation gate every scenario in
%  this project goes through). If that happens, that is itself a real
%  finding -- it means the true model limit has been found, just not
%  the one everyone assumed.
%
%  RUNTIME: one 5000s sim() call, ~15-20 minutes expected (may differ if
%  the solver struggles with the more extreme inputs).
%
%  Run:
%    generate_gradual_busC_scenario_v4
% ==========================================================================

    clc;

    %% 0. CONFIG ----------------------------------------------------------
    PROJECT_ROOT = 'D:\naren\Documents\Thermal_Derating_Work_Handover\ThermalProject';
    MODEL_NAME = 'Grid_modelling_Thermal_V7_ALfix';

    TS   = 0.01;
    TEND = 5000;
    N_STEPS = round(TEND / TS) + 1;
    PMIN = 2000;      % loosened from the old 20000 -- see header
    PMAX = 220000;    % loosened from the old 100000 -- see header
    SEED = 104;

    BUS  = {'A','B','C','D','E','F','G','H','K','L'};
    CONV = {'AB','BC','CD','DE','EF','FG','GH','HK','KL','AL'};

    RAMP_START_S = 800;
    RAMP_END_S   = 2800;
    C_TARGET_W   = 200000;   % Bus C target (was capped at 100000 in v1-v3)
    BD_TARGET_W  = 3000;     % Bus B/D target (was floored at 20000 in v1-v3)

    OUTPUT_DIR = fullfile(PROJECT_ROOT, 'model_outputs', 'thermal_derating_gradual_busC');
    if exist(OUTPUT_DIR, 'dir') ~= 7; mkdir(OUTPUT_DIR); end
    addpath(PROJECT_ROOT);

    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('GRADUAL LOAD SCENARIO v4: testing beyond the assumed PMIN/PMAX\n');
    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('Settle phase:    0 - %ds\n', RAMP_START_S);
    fprintf('Ramp phase:      %d - %ds\n', RAMP_START_S, RAMP_END_S);
    fprintf('Sustained heavy: %d - %ds\n', RAMP_END_S, TEND);
    fprintf('Bus C  -> %.0f W (was capped at 100000 in v1-v3)\n', C_TARGET_W);
    fprintf('Bus B,D -> %.0f W (was floored at 20000 in v1-v3)\n', BD_TARGET_W);
    fprintf('Clip range loosened to [%.0f, %.0f]\n', PMIN, PMAX);

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
    fprintf('\nBuilding v4 load profile (seed %d) ...\n', SEED);
    [load_hist, clip_hi, clip_lo] = build_gradual_busC_loads_v4( ...
        SEED, N_STEPS, TS, BUS, PMIN, PMAX, RAMP_START_S, RAMP_END_S, ...
        C_TARGET_W, BD_TARGET_W);
    fprintf('  Clip-high %% by bus (expect ~0 except transient overshoot): %s\n', mat2str(round(clip_hi,1)));
    fprintf('  Clip-low  %% by bus: %s\n', mat2str(round(clip_lo,1)));

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
    [msg_before] = lastwarn('');
    so = sim(MODEL_NAME, 'ReturnWorkspaceOutputs', 'on');
    [warn_msg, warn_id] = lastwarn();
    warning(ws);
    fprintf('Finished in %.1f s wall clock.\n', toc);
    if ~isempty(warn_msg) && ~strcmp(warn_msg, msg_before)
        fprintf('  NOTE: a solver-related warning fired during this run (id: %s):\n  %s\n', warn_id, warn_msg);
        fprintf('  Not necessarily fatal -- the NaN/Inf validation below is the real check.\n');
    end

    time_vec = (0:N_STEPS-1)' * TS;

    %% 5. Assemble the CSV ---------------------------------------------------
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
    log_path = fullfile(OUTPUT_DIR, 'gradual_busC_v4_validation_log.txt');
    log_fid = fopen(log_path, 'w');
    log_both(log_fid, '%s\n', repmat('=', 1, 70));
    log_both(log_fid, 'NaN/Inf VALIDATION: gradual_busC_v4 scenario\n');
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
                first_bad_t = bad_times(bad_times > WARMUP_TOLERANCE_S);
                log_both(log_fid, '    FIRST unexpected bad sample at t=%.2fs\n', first_bad_t(1));
            end
        end
    end
    if ~any(bad_mask)
        log_both(log_fid, '  No NaN/Inf found in any column. Clean.\n');
    end
    if any_unexpected
        log_both(log_fid, '\n  RESULT: FAILED VALIDATION. Unexpected NaN/Inf beyond warmup.\n');
        log_both(log_fid, '  This itself is informative: it suggests the model becomes\n');
        log_both(log_fid, '  numerically unstable somewhere between the old PMAX/PMIN and\n');
        log_both(log_fid, '  these v4 targets -- a real limit, just not a thermal one.\n');
        fclose(log_fid);
        error('gradual_busC_v4 scenario FAILED NaN/Inf validation. See log:\n%s', log_path);
    else
        log_both(log_fid, '\n  RESULT: PASSED VALIDATION. Safe to proceed.\n');
    end
    fclose(log_fid);

    out_csv = fullfile(OUTPUT_DIR, 'thermal_derating_gradual_busC_v4_5000s.csv');
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
    if any_collapse
        fprintf('\n  Real collapse achieved. This confirms PMIN/PMAX in every prior\n');
        fprintf('  script was a convention, not a model limit.\n');
    else
        fprintf('\n  Still no collapse. If Tj on BC/CD moved meaningfully ABOVE 164C\n');
        fprintf('  this time (even without crossing 175), that confirms the old\n');
        fprintf('  PMIN/PMAX really was suppressing the achievable temperature, and\n');
        fprintf('  pushing further (or extending the hold time at this new level)\n');
        fprintf('  should keep helping. If Tj is STILL ~164C despite doubling the\n');
        fprintf('  target, that would mean the ceiling is downstream of load\n');
        fprintf('  entirely -- something else in the thermal model itself caps it,\n');
        fprintf('  which would be worth inspecting the thermal_derater subsystem for\n');
        fprintf('  directly rather than iterating on load profiles further.\n');
    end

    close_system(MODEL_NAME, 0);
end


%% ---- v4 load profile: same single-bus topology as v1, new magnitude ----

function [load_hist, clip_hi, clip_lo] = build_gradual_busC_loads_v4( ...
    seed, n_steps, ts, BUS, pmin, pmax, ramp_start_s, ramp_end_s, ...
    c_target_w, bd_target_w)

    rng(seed);
    t = (0:n_steps-1)' * ts;
    NB = numel(BUS);

    ambient_base = 27000 * ones(1, NB);
    idx_C = find(strcmp(BUS, 'C'));
    idx_B = find(strcmp(BUS, 'B'));
    idx_D = find(strcmp(BUS, 'D'));

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
            raw = raw + (c_target_w - ambient_base(k)) * ramp_frac;
        elseif k == idx_B || k == idx_D
            raw = raw - (ambient_base(k) - bd_target_w) * ramp_frac;
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