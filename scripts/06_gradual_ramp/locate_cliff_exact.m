function locate_cliff_exact()
% ==========================================================================
%  LOCATE_CLIFF_EXACT.M
%  --------------------------------------------------------------------
%  Follow-up to refine_shed_cliff_sweep.m. That 35-run sweep narrowed the
%  cliff to somewhere between 3625s (safe, but thin margin, F min V
%  dropped to 735.03) and 3650s (total collapse, onset t=3643.4s exactly
%  matching the unshed baseline). Every failing run in that sweep
%  reproduced onset t=3643.4s exactly, so the working hypothesis is a
%  hard deadline tied to Bus F's own natural onset time, not a gradually
%  softening effect. This script sweeps every 5 seconds across the
%  narrowed window, at a single ramp duration (0s, since duration showed
%  no effect on outcome in every prior sweep), to pin the exact cutoff
%  cheaply before spending time re-adding the full duration grid there.
%
%  PREREQUISITE: the ramped model must already be verified via
%  compare_sheddable_vs_ramped.m (all buses 0.000e+00 diff, confirmed).
%  No need to rerun that here, this script reuses the same model file.
%
%  MECHANISM SIGNALS CAPTURED: Bus E's converter (DE, EF) derate factor
%  AND junction temperature, same as the prior sweeps, in case the "why"
%  question comes up once the exact cutoff is known.
%
%  RUNTIME: 6 full 5000s sim() calls, at the ~220-230s/run observed in
%  the prior sweeps, expect roughly 6 x 3.8 min ~= 23 minutes.
%
%  Run:
%    locate_cliff_exact
% ==========================================================================

    clc;

    %% 0. CONFIG --------------------------------------------------------
    PROJECT_ROOT = 'D:\naren\Documents\Thermal_Derating_Work_Handover\ThermalProject';
    MODEL_NAME = 'Grid_modelling_Thermal_V7_ALfix_sheddable_ramped';

    FIXED_FRACTION = 0.30;   % known-safe magnitude, held constant, per compare_failing_vs_safe_fraction.m

    % Start times: 5s resolution across the exact window that bracketed
    % the cliff in the prior 35-run sweep (3625 safe/thin, 3650 collapse).
    START_TIMES = [3625, 3630, 3635, 3640, 3645, 3650];

    % Single ramp duration: 0s (instantaneous). Duration made no measurable
    % difference to outcome anywhere in either prior sweep, so this pass
    % is timing-only. Re-add the full duration grid at whichever start
    % time turns out to be the cutoff, if duration sensitivity near the
    % edge (hinted at in the 3625s row of the prior sweep) needs checking.
    RAMP_DURATIONS = [0];

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
    E_CONV = {'DE', 'EF'};   % Bus E's own converters, the mechanism signal

    ZOOM_LO = 3300; ZOOM_HI = 4300;   % window retained at full resolution

    OUTPUT_DIR = fullfile(PROJECT_ROOT, 'model_outputs', 'thermal_derating_v7', 'shed_cliff_exact_sweep');
    if exist(OUTPUT_DIR, 'dir') ~= 7; mkdir(OUTPUT_DIR); end
    addpath(PROJECT_ROOT);

    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('EXACT CLIFF LOCATE: start time (5s steps), Bus F -> %.2f\n', FIXED_FRACTION);
    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('Start times:    %s\n', mat2str(START_TIMES));
    fprintf('Ramp durations: %s\n', mat2str(RAMP_DURATIONS));
    n_runs_total = numel(START_TIMES) * numel(RAMP_DURATIONS);
    fprintf('Total runs: %d (roughly %.1f minutes at ~3.8 min/run)\n', ...
            n_runs_total, n_runs_total * 3.8);

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
    zoom_time = time_vec(zoom_mask);
    n_zoom = sum(zoom_mask);

    n_start = numel(START_TIMES);
    n_ramp  = numel(RAMP_DURATIONS);

    V_F_zoom   = nan(n_zoom, n_start, n_ramp);
    V_E_zoom   = nan(n_zoom, n_start, n_ramp);
    DER_E_zoom = nan(n_zoom, numel(E_CONV), n_start, n_ramp);
    TJ_E_zoom  = nan(n_zoom, numel(E_CONV), n_start, n_ramp);

    results = struct('start_time', {}, 'ramp_duration', {}, ...
                      'f_collapsed', {}, 'f_onset', {}, 'f_min_v', {}, ...
                      'e_collapsed', {}, 'e_onset', {}, 'e_min_v', {});

    run_idx = 0;
    for si = 1:n_start
        start_t = START_TIMES(si);
        for ri = 1:n_ramp
            ramp_dur = RAMP_DURATIONS(ri);
            run_idx = run_idx + 1;

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

            % Set the ramp rate. 0 duration = no rate limiting (matches
            % the original instantaneous-step behavior exactly).
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

            V_F_zoom(:, si, ri) = v_f_full(zoom_mask);
            V_E_zoom(:, si, ri) = v_e_full(zoom_mask);

            for c = 1:numel(E_CONV)
                conv_idx = find(strcmp(CONV, E_CONV{c}));
                der_full = grab_series(so, derate_var_names{conv_idx}, TS, N_STEPS) / 100e3;
                tj_full  = grab_series(so, junction_var_names{conv_idx}, TS, N_STEPS);
                DER_E_zoom(:, c, si, ri) = der_full(zoom_mask);
                TJ_E_zoom(:, c, si, ri)  = tj_full(zoom_mask);
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
    end

    %% 5. Write the dataset as CSV ----------------------------------------
    fprintf('\n%s\n', repmat('=', 1, 70));
    fprintf('WRITING DATASET\n');
    fprintf('%s\n', repmat('=', 1, 70));

    T = struct2table(results);
    T.both_safe = ~T.f_collapsed & ~T.e_collapsed;
    csv_path = fullfile(OUTPUT_DIR, 'shed_cliff_exact_results.csv');
    writetable(T, csv_path);
    fprintf('Wrote dataset: %s (%d rows)\n', csv_path, height(T));

    %% 6. Outcome heatmap ---------------------------------------------------
    fprintf('\nBuilding heatmap ...\n');
    safe_grid = nan(n_start, n_ramp);
    for si = 1:n_start
        for ri = 1:n_ramp
            row = results(([results.start_time] == START_TIMES(si)) & ...
                          ([results.ramp_duration] == RAMP_DURATIONS(ri)));
            safe_grid(si, ri) = ~row.f_collapsed && ~row.e_collapsed;
        end
    end

    fig1 = figure('Position', [100, 100, 900, 600], 'Visible', 'off');
    imagesc(safe_grid);
    colormap([0.85 0.2 0.2; 0.2 0.75 0.2]);
    caxis([0 1]);
    set(gca, 'XTick', 1:n_ramp, 'XTickLabel', arrayfun(@(x) sprintf('%.0fs', x), RAMP_DURATIONS, 'uni', 0));
    set(gca, 'YTick', 1:n_start, 'YTickLabel', arrayfun(@(x) sprintf('%.0fs', x), START_TIMES, 'uni', 0));
    xlabel('ramp duration'); ylabel('shedding start time');
    title(sprintf('Both Bus E and F survive (green) vs at least one collapses (red), fraction=%.2f', ...
                   FIXED_FRACTION));
    for si = 1:n_start
        for ri = 1:n_ramp
            txt = 'FAIL'; if safe_grid(si,ri); txt = 'OK'; end
            text(ri, si, txt, 'HorizontalAlignment', 'center', 'Color', 'w', 'FontWeight', 'bold');
        end
    end
    saveas(fig1, fullfile(OUTPUT_DIR, 'shed_cliff_exact_heatmap.png'));
    close(fig1);

    %% 7. Save full results + zoomed traces ----------------------------------
    save(fullfile(OUTPUT_DIR, 'shed_cliff_exact_results.mat'), ...
         'V_F_zoom', 'V_E_zoom', 'DER_E_zoom', 'TJ_E_zoom', 'zoom_time', ...
         'START_TIMES', 'RAMP_DURATIONS', 'FIXED_FRACTION', 'results', 'E_CONV', ...
         'ZOOM_LO', 'ZOOM_HI', '-v7.3');

    fprintf('\nWrote plots and .mat to: %s\n', OUTPUT_DIR);
    fprintf('\nDone. The row where f_collapsed flips from 0 to 1 in\n');
    fprintf('shed_cliff_exact_results.csv is the exact cutoff, to 5s\n');
    fprintf('resolution: shedding must start AT OR BEFORE that start_time\n');
    fprintf('for Bus F and Bus E to survive. Compare that number against\n');
    fprintf('Bus F''s own onset (t=3643.4s) to get the true minimum lead\n');
    fprintf('time needed, and against the detector''s validated 60s/90s\n');
    fprintf('windows to check whether the detector already gives enough\n');
    fprintf('lead time for this scenario.\n');

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
