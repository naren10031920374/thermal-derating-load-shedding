function locate_gradual_lead_time()
% ==========================================================================
%  LOCATE_GRADUAL_LEAD_TIME.M
%  --------------------------------------------------------------------
%  Answers the still-open question: "how early must GRADUAL shedding
%  begin?" locate_cliff_exact.m pinned the instantaneous-step cutoff
%  at t=3640s safe / t=3645s collapse (essentially zero lead time before
%  Bus F's own onset at 3643.4s). That result does NOT carry over to a
%  ramped shed: a Rate Limiter with duration D does not reach the target
%  fraction (0.30) until start_t + D, so a ramp that starts at 3640 does
%  not finish until 3640+D, well after the 3643.4s onset for any D > 3s.
%  The working hypothesis is that the gradual cutoff sits roughly D
%  seconds earlier than the instant cutoff (anchor = 3640, the last
%  confirmed-safe instant start), i.e. cutoff(D) ~= 3640 - D. This script
%  tests that hypothesis directly: for each ramp duration, it sweeps
%  start times centered on 3640-D rather than repeating a full 2D grid,
%  since the search region differs per duration.
%
%  PREREQUISITE: the ramped model must already be verified via
%  compare_sheddable_vs_ramped.m (all buses 0.000e+00 diff, confirmed).
%
%  MECHANISM SIGNALS CAPTURED: Bus E's converter (DE, EF) derate factor
%  AND junction temperature, same as the prior sweeps.
%
%  OUTPUT: shed_gradual_lead_time_results.csv (one row per run), plus a
%  per-duration min-V-vs-start-time plot (not a 2D heatmap, since each
%  duration was swept over a different start-time window).
%
%  RUNTIME: 20 full 5000s sim() calls (4 durations x 5 offsets), at the
%  ~220s/run observed in prior sweeps, expect roughly 20 x 3.7 min ~=
%  73 minutes. This is a COARSE first pass (20s offset steps), expect to
%  need a follow-up fine pass once each duration's bracket is known,
%  same two-stage pattern as the instant-step cliff search.
%
%  Run:
%    locate_gradual_lead_time
% ==========================================================================

    clc;

    %% 0. CONFIG --------------------------------------------------------
    PROJECT_ROOT = 'D:\naren\Documents\Thermal_Derating_Work_Handover\ThermalProject';
    MODEL_NAME = 'Grid_modelling_Thermal_V7_ALfix_sheddable_ramped';

    FIXED_FRACTION = 0.30;

    ANCHOR = 3640;                    % last confirmed-safe instant start (locate_cliff_exact.m)
    RAMP_DURATIONS = [30, 60, 100, 150];
    OFFSETS = [-40, -20, 0, 20, 40];  % around each duration's predicted cutoff (ANCHOR - D)

    % Build the flat (start_t, ramp_dur) run list, one search window per
    % duration rather than a shared 2D grid, since the hypothesis says
    % the right window itself moves with D.
    RUN_START = [];
    RUN_RAMP  = [];
    for d = RAMP_DURATIONS
        predicted_cutoff = ANCHOR - d;
        RUN_START = [RUN_START, predicted_cutoff + OFFSETS]; %#ok<AGROW>
        RUN_RAMP  = [RUN_RAMP,  repmat(d, 1, numel(OFFSETS))]; %#ok<AGROW>
    end
    n_runs_total = numel(RUN_START);

    TS   = 0.01;
    TEND = 5000;
    N_STEPS = round(TEND / TS) + 1;
    LOAD_SEED = 1;
    GRID_LOAD_SCALE = 1.0;
    PMIN = 20000;
    PMAX = 100000;

    COLLAPSE_VOLTAGE_V = 100.0;
    MIN_CONSECUTIVE_SAMPLES = 50;

    BUS  = {'A','B','C','D','E','F','G','H','K','L'};
    CONV = {'AB','BC','CD','DE','EF','FG','GH','HK','KL','AL'};
    E_CONV = {'DE', 'EF'};

    ZOOM_LO = 3200; ZOOM_HI = 4300;   % widened low end since durations push start times earlier

    OUTPUT_DIR = fullfile(PROJECT_ROOT, 'model_outputs', 'thermal_derating_v7', 'shed_gradual_lead_time_sweep');
    if exist(OUTPUT_DIR, 'dir') ~= 7; mkdir(OUTPUT_DIR); end
    addpath(PROJECT_ROOT);

    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('GRADUAL LEAD TIME LOCATE: per-duration start-time windows, Bus F -> %.2f\n', FIXED_FRACTION);
    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('Anchor (instant cutoff): %.0fs\n', ANCHOR);
    for d = RAMP_DURATIONS
        fprintf('  duration %3.0fs: predicted cutoff %.0fs, testing %s\n', ...
                d, ANCHOR - d, mat2str((ANCHOR - d) + OFFSETS));
    end
    fprintf('Total runs: %d (roughly %.1f minutes at ~3.7 min/run)\n', ...
            n_runs_total, n_runs_total * 3.7);

    %% 1. Build the load profile, identical every run ---------------------
    fprintf('\nBuilding load profile (seed %d, scale %.2f) ...\n', LOAD_SEED, GRID_LOAD_SCALE);
    [~, ~, ~] = build_derating_loads(LOAD_SEED, N_STEPS, TS, BUS, GRID_LOAD_SCALE, PMIN, PMAX);

    %% 2. Locate and load the ramped model ------------------------------
    hh = dir(fullfile(PROJECT_ROOT, '**', [MODEL_NAME '.slx']));
    if isempty(hh)
        error(['Ramped model not found: %s.slx\n' ...
               'Run add_gradual_ramp_capability.m first, then ' ...
               'compare_sheddable_vs_ramped.m to verify it before this.'], MODEL_NAME);
    end
    model_file = fullfile(hh(1).folder, hh(1).name);
    fprintf('\nModel: %s\n', model_file);
    if bdIsLoaded(MODEL_NAME)
        close_system(MODEL_NAME, 0);
    end
    load_system(model_file);

    ramp_block = sprintf('%s/ShedFrac_F_ramp', MODEL_NAME);
    if getSimulinkBlockHandle(ramp_block) == -1
        error(['Block not found: %s\nRun add_gradual_ramp_capability.m ' ...
               'on the sheddable model first.'], ramp_block);
    end

    %% 3. Wire taps ONCE -------------------------------------------------
    fprintf('\nWiring signal taps (once, reused across all runs) ...\n');
    wire_all_taps(MODEL_NAME, BUS, CONV);
    derate_var_names   = tap_derate_factors(MODEL_NAME, CONV);
    junction_var_names = tap_junction_temp(MODEL_NAME, CONV);

    set_param(MODEL_NAME, 'StopTime', num2str(TEND));

    %% 4. Sweep -------------------------------------------------------------
    time_vec = (0:N_STEPS-1)' * TS;
    zoom_mask = time_vec >= ZOOM_LO & time_vec <= ZOOM_HI;
    zoom_time = time_vec(zoom_mask); %#ok<NASGU>
    n_zoom = sum(zoom_mask);

    V_F_zoom   = nan(n_zoom, n_runs_total);
    V_E_zoom   = nan(n_zoom, n_runs_total);
    DER_E_zoom = nan(n_zoom, numel(E_CONV), n_runs_total);
    TJ_E_zoom  = nan(n_zoom, numel(E_CONV), n_runs_total);

    results = struct('start_time', {}, 'ramp_duration', {}, ...
                      'f_collapsed', {}, 'f_onset', {}, 'f_min_v', {}, ...
                      'e_collapsed', {}, 'e_onset', {}, 'e_min_v', {});

    for run_idx = 1:n_runs_total
        start_t  = RUN_START(run_idx);
        ramp_dur = RUN_RAMP(run_idx);

        fprintf('\n%s\n', repmat('-', 1, 70));
        fprintf('RUN %d/%d: start_t=%.0fs, ramp_duration=%.0fs (fraction %.2f)\n', ...
                run_idx, n_runs_total, start_t, ramp_dur, FIXED_FRACTION);
        fprintf('%s\n', repmat('-', 1, 70));

        % Reset every bus to constant no-shedding first.
        for b = 1:numel(BUS)
            blk = sprintf('%s/ShedFrac_%s_default', MODEL_NAME, BUS{b});
            ensure_constant_block(MODEL_NAME, BUS{b}, blk, 1);
        end
        % Bus F's default -> Step at start_t, 1.0 -> FIXED_FRACTION.
        f_blk = sprintf('%s/ShedFrac_F_default', MODEL_NAME);
        set_shed_step(MODEL_NAME, 'F', f_blk, start_t, 1, FIXED_FRACTION);

        % Set the ramp rate for this run's duration.
        if ramp_dur <= 0
            set_param(ramp_block, 'RisingSlewLimit', 'inf', 'FallingSlewLimit', '-inf');
        else
            falling_slew = -(1 - FIXED_FRACTION) / ramp_dur;
            set_param(ramp_block, 'RisingSlewLimit', 'inf', ...
                      'FallingSlewLimit', num2str(falling_slew));
        end

        tic;
        ws = warning('off', 'all');
        so = sim(MODEL_NAME, 'ReturnWorkspaceOutputs', 'on');
        warning(ws);
        fprintf('Finished in %.1f s wall clock.\n', toc);

        v_f_full = grab_series(so, 'V_Bus_F', TS, N_STEPS);
        v_e_full = grab_series(so, 'V_Bus_E', TS, N_STEPS);

        V_F_zoom(:, run_idx) = v_f_full(zoom_mask);
        V_E_zoom(:, run_idx) = v_e_full(zoom_mask);

        for c = 1:numel(E_CONV)
            conv_idx = find(strcmp(CONV, E_CONV{c}));
            der_full = grab_series(so, derate_var_names{conv_idx}, TS, N_STEPS) / 100e3;
            tj_full  = grab_series(so, junction_var_names{conv_idx}, TS, N_STEPS);
            DER_E_zoom(:, c, run_idx) = der_full(zoom_mask);
            TJ_E_zoom(:, c, run_idx)  = tj_full(zoom_mask);
        end

        [f_c, f_onset, f_min] = detect_collapse_summary( ...
            time_vec, v_f_full, COLLAPSE_VOLTAGE_V, MIN_CONSECUTIVE_SAMPLES);
        [e_c, e_onset, e_min] = detect_collapse_summary( ...
            time_vec, v_e_full, COLLAPSE_VOLTAGE_V, MIN_CONSECUTIVE_SAMPLES);

        if f_c
            fprintf('  Bus F: COLLAPSES, onset t=%.1fs, min V=%.2f\n', f_onset, f_min);
        else
            fprintf('  Bus F: survives, min V=%.2f\n', f_min);
        end
        if e_c
            fprintf('  Bus E: COLLAPSES, onset t=%.1fs, min V=%.2f\n', e_onset, e_min);
        else
            fprintf('  Bus E: survives, min V=%.2f\n', e_min);
        end

        results(end+1) = struct('start_time', start_t, 'ramp_duration', ramp_dur, ... %#ok<AGROW>
            'f_collapsed', f_c, 'f_onset', f_onset, 'f_min_v', f_min, ...
            'e_collapsed', e_c, 'e_onset', e_onset, 'e_min_v', e_min);
    end

    %% 5. Write the dataset as CSV ----------------------------------------
    fprintf('\n%s\n', repmat('=', 1, 70));
    fprintf('WRITING DATASET\n');
    fprintf('%s\n', repmat('=', 1, 70));

    T = struct2table(results);
    T.both_safe = ~T.f_collapsed & ~T.e_collapsed;
    csv_path = fullfile(OUTPUT_DIR, 'shed_gradual_lead_time_results.csv');
    writetable(T, csv_path);
    fprintf('Wrote dataset: %s (%d rows)\n', csv_path, height(T));

    %% 6. Per-duration min-V vs start-time plot ------------------------------
    fprintf('\nBuilding per-duration plot ...\n');
    fig1 = figure('Position', [100, 100, 1000, 700], 'Visible', 'off');
    colors = lines(numel(RAMP_DURATIONS));
    hold on;
    for di = 1:numel(RAMP_DURATIONS)
        d = RAMP_DURATIONS(di);
        mask = ([results.ramp_duration] == d);
        st = [results(mask).start_time];
        fv = [results(mask).f_min_v];
        [st, order] = sort(st);
        fv = fv(order);
        plot(st, fv, '-o', 'Color', colors(di,:), 'LineWidth', 2, ...
             'DisplayName', sprintf('ramp %.0fs', d));
    end
    yline(0, '--', 'collapse (V=0)', 'Color', [0.8 0.2 0.2]);
    xlabel('shedding start time (s)');
    ylabel('Bus F min voltage (V)');
    title(sprintf('Gradual shedding: min V vs start time, per ramp duration, fraction=%.2f', ...
                   FIXED_FRACTION));
    legend('Location', 'best');
    grid on;
    hold off;
    saveas(fig1, fullfile(OUTPUT_DIR, 'shed_gradual_lead_time_plot.png'));
    close(fig1);

    %% 7. Save full results + zoomed traces ----------------------------------
    save(fullfile(OUTPUT_DIR, 'shed_gradual_lead_time_results.mat'), ...
         'V_F_zoom', 'V_E_zoom', 'DER_E_zoom', 'TJ_E_zoom', 'zoom_time', ...
         'RUN_START', 'RUN_RAMP', 'ANCHOR', 'OFFSETS', 'RAMP_DURATIONS', ...
         'FIXED_FRACTION', 'results', 'E_CONV', 'ZOOM_LO', 'ZOOM_HI', '-v7.3');

    fprintf('\nWrote plots and .mat to: %s\n', OUTPUT_DIR);
    fprintf('\nDone. For each duration, find where f_collapsed flips 0->1 in\n');
    fprintf('shed_gradual_lead_time_results.csv (or the dip to V=0 in the plot).\n');
    fprintf('That is duration''s cutoff, to 20s resolution. Compare each cutoff\n');
    fprintf('to (ANCHOR - duration) to check the hypothesis: if they roughly\n');
    fprintf('match, the rule "start ramp_duration seconds earlier than the\n');
    fprintf('instant cutoff" holds and can be stated as policy. If a duration''s\n');
    fprintf('bracket falls outside the tested +/-40s window, rerun with wider\n');
    fprintf('offsets for that duration specifically.\n');

    close_system(MODEL_NAME, 0);
end


%% ---- block manipulation helpers (from test_shed_lead_time.m) ----

function ensure_constant_block(model, bus, blk_path, value)
    % Ensures blk_path is a plain Constant block with the given value.
    % If a Step block from a previous run is sitting there instead, it is
    % removed and replaced, so every non-target bus is a clean no-shed default.
    if getSimulinkBlockHandle(blk_path) == -1
        return;
    end
    bt = get_param(blk_path, 'BlockType');
    if strcmp(bt, 'Constant')
        set_param(blk_path, 'Value', num2str(value));
        return;
    end
    pos = get_param(blk_path, 'Position');
    out_line = get_param(blk_path, 'PortHandles');
    dst_line = -1;
    if isfield(out_line, 'Outport') && ~isempty(out_line.Outport)
        dst_line = get_param(out_line.Outport(1), 'Line');
    end
    dst_block = ''; dst_port = 1;
    if dst_line ~= -1
        dst_port_handle = get_param(dst_line, 'DstPortHandle');
        dst_block = get_param(dst_port_handle, 'Parent');
        dst_port_info = get_param(dst_port_handle, 'PortNumber');
        dst_port = dst_port_info;
        delete_line(dst_line);
    end
    delete_block(blk_path);
    add_block('simulink/Sources/Constant', blk_path, 'Value', num2str(value), 'Position', pos);
    if ~isempty(dst_block)
        [~, dst_name] = fileparts(dst_block);
        add_line(model, sprintf('%s/1', regexprep(blk_path, '.*/', '')), ...
                 sprintf('%s/%d', dst_name, dst_port), 'autorouting', 'on');
    end
end


function set_shed_step(model, bus, blk_path, step_time, before_val, after_val) %#ok<INUSD>
    % Swaps blk_path (currently a Constant, or a Step from a prior run) for
    % a Step source block: outputs before_val until step_time, after_val
    % from step_time onward. Same name and position, so the single
    % downstream line (into the Rate Limiter) is re-added identically.
    pos = get_param(blk_path, 'Position');
    ph = get_param(blk_path, 'PortHandles');
    dst_block = ''; dst_port = 1;
    if isfield(ph, 'Outport') && ~isempty(ph.Outport)
        line = get_param(ph.Outport(1), 'Line');
        if line ~= -1
            dst_port_handle = get_param(line, 'DstPortHandle');
            dst_block = get_param(dst_port_handle, 'Parent');
            dst_port = get_param(dst_port_handle, 'PortNumber');
            delete_line(line);
        end
    end
    delete_block(blk_path);
    add_block('simulink/Sources/Step', blk_path, ...
        'Time', num2str(step_time), 'Before', num2str(before_val), 'After', num2str(after_val), ...
        'Position', pos);
    if ~isempty(dst_block)
        [~, src_name] = fileparts(blk_path);
        [~, dst_name] = fileparts(dst_block);
        add_line(model, sprintf('%s/1', src_name), sprintf('%s/%d', dst_name, dst_port), 'autorouting', 'on');
    end
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


%% ---- verified verbatim against grid_derating_dataset_5000s_v7_2.m ----

function [load_hist, clip_hi, clip_lo] = build_derating_loads( ...
    seed, n_steps, ts, BUS, scale, pmin, pmax)

    rng(seed);
    t = (0:n_steps-1)' * ts;
    NB = numel(BUS);
    base_load = [35000 30000 25000 32000 28000 24000 30000 26000 22000 34000];
    grid_slow = 25000 * sin(2*pi*t/1800);
    grid_med  = 15000 * sin(2*pi*t/700 + 0.4);

    load_hist = nan(n_steps, NB);
    clip_hi   = nan(1, NB);
    clip_lo   = nan(1, NB);

    for k = 1:NB
        phase_shift = 0.35 * k;
        bus_var  = (10000 + 700*k) * sin(2*pi*t/(800 + 80*k) + phase_shift);
        fast_var = (1800 + 100*k) * sin(2*pi*t/(50 + 3*k) + 0.2*k);
        noise    = 1200 * randn(n_steps,1);

        event = zeros(n_steps,1);
        event(t >= 800  & t < 1600) = 15000 + 1000*k;
        event(t >= 1600 & t < 2600) = -(8000 + 500*k);
        event(t >= 3000 & t < 4200) = 20000 + 1200*k;
        event(t >= 4200)            = -(10000 + 600*k);

        raw = scale * (base_load(k) + grid_slow + 0.8*grid_med + bus_var + fast_var + noise + event);
        clip_hi(k) = mean(raw > pmax) * 100;
        clip_lo(k) = mean(raw < pmin) * 100;

        p = max(pmin, min(pmax, raw));
        load_hist(:,k) = p;
        assignin('base', sprintf('Pload_%s', BUS{k}), p(:).');
    end
end


function wire_all_taps(model, BUS, CONV)
    want_V   = strcat('V_Bus_',   BUS);
    want_P   = strcat('Bus_',     BUS, '_Src_Pow');
    want_T   = strcat('Bus_',     BUS, '_Temp');
    want_HS  = {'DAB_AB_H_Temp','DAB_BC_H_Temp','DAB_CD_H_Temp', ...
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
                ['Expected exactly one Outport block named ''Temp_Junction'' under ' ...
                 'DAB_Model_%s, found %d. This converter''s junction temperature ' ...
                 'will be blank.'], CONV{i}, numel(outport_blk));
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
