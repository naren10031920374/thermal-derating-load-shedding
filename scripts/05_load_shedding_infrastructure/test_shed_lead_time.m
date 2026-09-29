function test_shed_lead_time()
% ==========================================================================
%  TEST_SHED_LEAD_TIME.M
%  --------------------------------------------------------------------
%  Every prior shed-fraction test applied its fraction from t=0 for the
%  whole 5000s run. This answers a different, more important question:
%  how LATE can shedding start and still work? The known-safe fraction
%  (0.30 on Bus F alone) is held fixed; what varies is START_TIME, the
%  moment shedding switches on. Before START_TIME, Bus F carries its
%  normal, unshed load.
%
%  WHY THIS MATTERS: the mechanism found in compare_failing_vs_safe_fraction.m
%  showed Bus E's own converters (DE, EF) need to heat up and self-derate
%  BEFORE the t~4150s crisis window to survive it, in the always-on 0.30
%  case that heating starts around t~3900-4000s, roughly 150-250 seconds
%  of head start. Your early-warning detector's best validated lead time
%  is only ~60-90 seconds (N=60s/90s from the early-warning sweep). If the
%  real requirement is 150-250s, the detector cannot drive this
%  intervention no matter how the control loop is built. This script
%  finds the actual minimum requirement directly, rather than assuming it.
%
%  MECHANISM: ShedFrac_F_default is swapped at runtime from a Constant
%  block to a Step block (Before=1.0, After=0.30, StepTime=START_TIME),
%  same name and position, so nothing downstream needs rewiring. This
%  change is never saved to disk, exactly like the tap-wiring elsewhere
%  in this project, the verified .slx on disk is untouched.
%
%  CAVEAT: a Step is an instantaneous jump, not the gradual ramp a real
%  deployed system would use. This finds a BEST-CASE minimum lead time;
%  a real ramped implementation would need to start even earlier.
%
%  Also worth remembering: Bus F's OWN original collapse mechanism (the
%  one this whole project started from) triggers at t~3643s if F is never
%  shed. If START_TIME is later than that, F may already be on its way to
%  collapse before shedding engages at all, this is checked directly
%  rather than assumed.
%
%  RUNTIME: 10 full 5000s sim() calls, roughly 40-45 minutes based on
%  prior timings. Reduce START_TIMES for a faster first pass.
%
%  Run:
%    test_shed_lead_time
% ==========================================================================

    clc;

    %% 0. CONFIG --------------------------------------------------------
    PROJECT_ROOT = 'D:\naren\Documents\Thermal_Derating_Work_Handover\ThermalProject';
    MODEL_NAME = 'Grid_modelling_Thermal_V7_ALfix_sheddable';

    FIXED_FRACTION = 0.30;   % the known-safe magnitude, held constant across this sweep
    % Sweep from "always on" down toward "very late". Includes t=3643 (Bus
    % F's own original collapse onset) explicitly, since that boundary
    % matters for F's own survival, separately from E's.
    START_TIMES = [0, 3000, 3400, 3600, 3643, 3750, 3900, 4000, 4050, 4100];

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
    E_CONV = {'DE', 'EF'};   % Bus E's own converters, the mechanism from the last comparison

    OUTPUT_DIR = fullfile(PROJECT_ROOT, 'model_outputs', 'thermal_derating_v7', 'shed_lead_time_sweep');
    if exist(OUTPUT_DIR, 'dir') ~= 7; mkdir(OUTPUT_DIR); end
    addpath(PROJECT_ROOT);

    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('LEAD TIME SWEEP: Bus F shed to %.2f, starting at different times\n', FIXED_FRACTION);
    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('\nStart times to test: %s\n', mat2str(START_TIMES));
    fprintf('CAVEAT: instantaneous step, not a ramp. This finds a best-case minimum.\n');

    %% 1. Build the load profile, identical every run ---------------------
    fprintf('\nBuilding load profile (seed %d, scale %.2f) ...\n', LOAD_SEED, GRID_LOAD_SCALE);
    [~, ~, ~] = build_derating_loads(LOAD_SEED, N_STEPS, TS, BUS, GRID_LOAD_SCALE, PMIN, PMAX);

    %% 2. Locate and load the sheddable model ------------------------------
    hh = dir(fullfile(PROJECT_ROOT, '**', [MODEL_NAME '.slx']));
    if isempty(hh)
        error(['Sheddable model not found: %s.slx\n' ...
               'Run add_controllable_load_shedding.m first.'], MODEL_NAME);
    end
    model_file = fullfile(hh(1).folder, hh(1).name);
    fprintf('\nModel: %s\n', model_file);
    if bdIsLoaded(MODEL_NAME)
        close_system(MODEL_NAME, 0);
    end
    load_system(model_file);

    %% 3. Wire taps ONCE -------------------------------------------------
    fprintf('\nWiring signal taps (once, reused across all start times) ...\n');
    wire_all_taps(MODEL_NAME, BUS, CONV);
    derate_var_names   = tap_derate_factors(MODEL_NAME, CONV);
    junction_var_names = tap_junction_temp(MODEL_NAME, CONV); %#ok<NASGU>

    set_param(MODEL_NAME, 'StopTime', num2str(TEND));

    %% 4. Sweep -------------------------------------------------------------
    time_vec = (0:N_STEPS-1)' * TS;
    n_runs = numel(START_TIMES);

    V_F = nan(N_STEPS, n_runs);
    V_E = nan(N_STEPS, n_runs);
    DER_E = nan(N_STEPS, numel(E_CONV), n_runs);   % E's own converters
    results = struct('start_time', {}, 'f_collapsed', {}, 'f_onset', {}, 'f_min_v', {}, ...
                      'e_collapsed', {}, 'e_onset', {}, 'e_min_v', {});

    for r = 1:n_runs
        start_t = START_TIMES(r);
        fprintf('\n%s\n', repmat('-', 1, 70));
        fprintf('RUN %d/%d: shedding starts at t=%.0fs (fraction %.2f after that)\n', ...
                r, n_runs, start_t, FIXED_FRACTION);
        fprintf('%s\n', repmat('-', 1, 70));

        % Reset every bus to constant no-shedding first (safe default).
        for b = 1:numel(BUS)
            blk = sprintf('%s/ShedFrac_%s_default', MODEL_NAME, BUS{b});
            ensure_constant_block(MODEL_NAME, BUS{b}, blk, 1);
        end
        % Swap Bus F's block to a time-triggered Step.
        f_blk = sprintf('%s/ShedFrac_F_default', MODEL_NAME);
        set_shed_step(MODEL_NAME, 'F', f_blk, start_t, 1, FIXED_FRACTION);

        tic;
        ws = warning('off', 'all');
        so = sim(MODEL_NAME, 'ReturnWorkspaceOutputs', 'on');
        warning(ws);
        fprintf('Finished in %.1f s wall clock.\n', toc);

        V_F(:, r) = grab_series(so, 'V_Bus_F', TS, N_STEPS);
        V_E(:, r) = grab_series(so, 'V_Bus_E', TS, N_STEPS);
        for c = 1:numel(E_CONV)
            conv_idx = find(strcmp(CONV, E_CONV{c}));
            der_raw = grab_series(so, derate_var_names{conv_idx}, TS, N_STEPS);
            DER_E(:, c, r) = der_raw / 100e3;   % corrected scale, learned from last time
        end

        [f_c, f_onset, f_min] = detect_collapse_summary( ...
            time_vec, V_F(:, r), COLLAPSE_VOLTAGE_V, MIN_CONSECUTIVE_SAMPLES);
        [e_c, e_onset, e_min] = detect_collapse_summary( ...
            time_vec, V_E(:, r), COLLAPSE_VOLTAGE_V, MIN_CONSECUTIVE_SAMPLES);

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

        results(end+1) = struct('start_time', start_t, ... %#ok<AGROW>
            'f_collapsed', f_c, 'f_onset', f_onset, 'f_min_v', f_min, ...
            'e_collapsed', e_c, 'e_onset', e_onset, 'e_min_v', e_min);
    end

    %% 5. Summary table -------------------------------------------------------
    fprintf('\n%s\n', repmat('=', 1, 70));
    fprintf('SUMMARY\n');
    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('%-12s %-14s %-10s %-14s %-10s\n', 'start_t', 'F outcome', 'F onset', 'E outcome', 'E onset');
    for r = 1:n_runs
        res = results(r);
        f_str = 'survives'; f_onset_str = 'n/a';
        if res.f_collapsed; f_str = 'COLLAPSES'; f_onset_str = sprintf('%.1f', res.f_onset); end
        e_str = 'survives'; e_onset_str = 'n/a';
        if res.e_collapsed; e_str = 'COLLAPSES'; e_onset_str = sprintf('%.1f', res.e_onset); end
        fprintf('%-12.0f %-14s %-10s %-14s %-10s\n', res.start_time, f_str, f_onset_str, e_str, e_onset_str);
    end

    % Find the latest start time that still keeps BOTH buses safe.
    both_safe = arrayfun(@(r) ~r.f_collapsed && ~r.e_collapsed, results);
    if any(both_safe)
        latest_safe = max([results(both_safe).start_time]);
        fprintf('\nLatest tested start time with BOTH buses surviving: t=%.0fs\n', latest_safe);
        fprintf('(i.e. at least %.0fs of lead time before Bus E''s ~4150s crisis window)\n', 4150 - latest_safe);
    else
        fprintf('\nNo tested start time kept both buses safe.\n');
    end

    %% 6. Plots -----------------------------------------------------------------
    fprintf('\nBuilding plots ...\n');
    zoom_lo = 3400; zoom_hi = 4300;
    mask = time_vec >= zoom_lo & time_vec <= zoom_hi;

    % --- Outcome dot chart, same style as the fraction sweep ---
    fig1 = figure('Position', [100, 100, 1200, 400], 'Visible', 'off');
    hold on;
    for r = 1:n_runs
        f_color = [0.85 0.2 0.2]; if ~results(r).f_collapsed; f_color = [0.2 0.75 0.2]; end
        e_color = [0.85 0.2 0.2]; if ~results(r).e_collapsed; e_color = [0.2 0.75 0.2]; end
        plot(results(r).start_time, 2, 'o', 'MarkerSize', 16, 'MarkerFaceColor', f_color, 'MarkerEdgeColor', 'w');
        plot(results(r).start_time, 1, 'o', 'MarkerSize', 16, 'MarkerFaceColor', e_color, 'MarkerEdgeColor', 'w');
    end
    xline(3643, ':k', 'F''s own original onset');
    set(gca, 'YTick', [1 2], 'YTickLabel', {'Bus E', 'Bus F'}, 'YLim', [0.5, 2.5]);
    xlabel('shedding start time (s)');
    title(sprintf('Outcome vs shedding start time (fixed fraction %.2f): how late can it start?', FIXED_FRACTION));
    grid on; hold off;
    saveas(fig1, fullfile(OUTPUT_DIR, 'lead_time_outcome_summary.png'));
    close(fig1);

    % --- Bus E voltage across start times, zoomed ---
    fig2 = figure('Position', [100, 100, 1400, 500], 'Visible', 'off');
    hold on;
    colors = lines(n_runs);
    for r = 1:n_runs
        plot(time_vec(mask), V_E(mask, r), 'Color', colors(r,:), 'LineWidth', 1.2, ...
             'DisplayName', sprintf('start=%.0fs', START_TIMES(r)));
    end
    yline(COLLAPSE_VOLTAGE_V, '--k', '100V threshold');
    xlabel('time (s)'); ylabel('Bus E voltage (V)');
    title('Bus E voltage vs shedding start time, zoomed');
    legend('Location', 'southwest', 'Interpreter', 'none'); grid on; hold off;
    saveas(fig2, fullfile(OUTPUT_DIR, 'lead_time_bus_E_voltage.png'));
    close(fig2);

    % --- E's converter derate factor across start times, zoomed ---
    fig3 = figure('Position', [100, 100, 1400, 500], 'Visible', 'off');
    hold on;
    linestyles = {'-', '--'};
    for r = 1:n_runs
        for c = 1:numel(E_CONV)
            plot(time_vec(mask), DER_E(mask, c, r), 'Color', colors(r,:), 'LineStyle', linestyles{c}, ...
                 'LineWidth', 1.0, 'DisplayName', sprintf('start=%.0fs, %s', START_TIMES(r), E_CONV{c}));
        end
    end
    ylim([0, 1.05]);
    xlabel('time (s)'); ylabel('Derate factor');
    title('Bus E''s own converter (DE, EF) derate factor vs shedding start time');
    legend('Location', 'southwest', 'Interpreter', 'none', 'NumColumns', 2); grid on; hold off;
    saveas(fig3, fullfile(OUTPUT_DIR, 'lead_time_bus_E_derate.png'));
    close(fig3);

    save(fullfile(OUTPUT_DIR, 'lead_time_sweep_results.mat'), ...
         'V_F', 'V_E', 'DER_E', 'time_vec', 'START_TIMES', 'FIXED_FRACTION', 'results', 'E_CONV', '-v7.3');

    fprintf('\nWrote plots to: %s\n', OUTPUT_DIR);
    fprintf('\nDone. Read lead_time_outcome_summary.png first: the latest start time with both\n');
    fprintf('dots green is the minimum required lead time. Compare (4150 - that number)\n');
    fprintf('against the ~60-90s your early-warning detector actually provides, that\n');
    fprintf('comparison answers whether this intervention can be driven by this detector.\n');

    close_system(MODEL_NAME, 0);
end


function ensure_constant_block(model, bus, blk_path, value)
    % Ensures blk_path is a plain Constant block with the given value.
    % If a Step block from a previous run is sitting there instead, it is
    % removed and replaced, so every non-target bus is a clean no-shed default.
    if getSimulinkBlockHandle(blk_path) == -1
        return;   % should not happen if add_controllable_load_shedding.m ran correctly
    end
    bt = get_param(blk_path, 'BlockType');
    if strcmp(bt, 'Constant')
        set_param(blk_path, 'Value', num2str(value));
        return;
    end
    % It's a Step block from a prior run's swap, replace it with a fresh Constant.
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
    % downstream line is re-added identically, nothing else changes.
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

    % run-length encode "below" to find any run >= min_run samples
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


function [load_hist, clip_hi, clip_lo] = build_derating_loads( ...
    seed, n_steps, ts, BUS, scale, pmin, pmax)
    % VERBATIM from grid_derating_dataset_5000s_v7_2.m, cross-checked
    % directly against the uploaded file this time, not reproduced from
    % memory. An earlier version of this script used a fabricated load
    % function that did not match this one, producing a different load
    % schedule and invalid collapse timing (Bus F at t~2150s instead of
    % the established t~3643s). This is the corrected, verified version.
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
    % VERBATIM from grid_derating_dataset_5000s_v7_2.m. This wires V_Bus_<bus>,
    % Bus_<bus>_Src_Pow, Bus_<bus>_Temp, and the heat-sink temps, each of
    % which already has a Goto tag in the base model. GEI is deliberately
    % NOT handled here (it has no Goto tag in this model, it is a named
    % output port on a root Divide block instead) and is not needed for
    % this sweep, since only voltage, derate factor, and junction temp are
    % used below.
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
    % VERBATIM from grid_derating_dataset_5000s_v7_2.m.
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


function var_names = tap_phase_inports(model, CONV)
    var_names = cell(1, numel(CONV));
    for i = 1:numel(CONV)
        var_names{i} = sprintf('V7PH_%s', CONV{i});
        dab_path = [model '/DAB_Model_' CONV{i}];
        blk = find_system(dab_path, 'FollowLinks', 'on', 'LookUnderMasks', 'all', ...
                          'BlockType', 'Inport', 'Name', 'phase_shift');
        if isempty(blk)
            warning('tap_phase_inports:noInport', ...
                'No phase_shift Inport found in %s, this converter will be blank.', dab_path);
            continue;
        end
        parent = get_param(blk{1}, 'Parent');
        ph = get_param(blk{1}, 'PortHandles');
        if ~isfield(ph, 'Outport') || isempty(ph.Outport)
            continue;
        end
        log_name = ['V7LOG_' var_names{i}];
        cleanup_and_add_tap(parent, log_name, var_names{i}, ph.Outport(1), model);
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
