function run_closed_loop_verification_bus_f_only()
% ==========================================================================
%  RUN_CLOSED_LOOP_VERIFICATION_BUS_F_ONLY.M
%  --------------------------------------------------------------------
%  The clean version of the closed-loop test. The all-bus run
%  (run_closed_loop_verification.m + detector_trigger_times_all_buses.json)
%  mixed one genuine result with several artifacts:
%    - Bus F is the ONLY bus that was held out from detector training, so
%      its trigger (t=3592.59s, lead time 50.85s before its 3643.44s
%      onset) is a real generalization test.
%    - Bus E and every other bus were part of the training set. Bus E's
%      trigger (t=183.44s, 99.94% confidence, ~3513s "lead time") is
%      almost certainly the model recognizing training data it already
%      fit, not genuine precursor signal, the confidence and lead time
%      are both physically implausible given everything the shed-cliff
%      sweeps found about how close to onset the real cutoff sits.
%
%  This script isolates the one legitimate number: Bus F sheds at its
%  real, held-out-bus trigger time. Every other bus, INCLUDING Bus E,
%  gets NO shedding at all, not even E's own (spurious) trigger. If E
%  still survives here, that survival is attributable ONLY to Bus F
%  being shed with 50.85s of real lead time, nothing else, which is the
%  actual question worth answering: is 50.85s enough.
%
%  Bus F trigger time is HARDCODED below, not read from the JSON, since
%  this script intentionally ignores every other bus's entry in that file.
%
%  RUNTIME: 1 full 5000s sim() call, roughly 3.5-4 minutes.
%
%  Run:
%    run_closed_loop_verification_bus_f_only
% ==========================================================================

    clc;

    %% 0. CONFIG --------------------------------------------------------
    PROJECT_ROOT = 'D:\naren\Documents\Thermal_Derating_Work_Handover\ThermalProject';
    MODEL_NAME = 'Grid_modelling_Thermal_V7_ALfix_sheddable_ramped';

    % Bus F's real, genuinely-held-out-bus detector trigger, from
    % find_detector_trigger_time.py's debounced (10-sample sustained) run.
    F_TRIGGER_TIME = 3592.59;
    F_ONSET_REFERENCE = 3643.44;   % for display only, not used in the sim

    FIXED_FRACTION = 0.30;
    RAMP_DURATION = 0;   % 0 = instantaneous

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

    ZOOM_LO = 3200; ZOOM_HI = 4300;

    OUTPUT_DIR = fullfile(PROJECT_ROOT, 'model_outputs', 'thermal_derating_v7', 'closed_loop_verification');
    if exist(OUTPUT_DIR, 'dir') ~= 7; mkdir(OUTPUT_DIR); end
    addpath(PROJECT_ROOT);

    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('CLEAN CLOSED-LOOP TEST: Bus F only, real held-out-bus trigger\n');
    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('Bus F trigger time:     %.2fs\n', F_TRIGGER_TIME);
    fprintf('Bus F onset (reference): %.2fs\n', F_ONSET_REFERENCE);
    fprintf('Real lead time:         %.2fs\n', F_ONSET_REFERENCE - F_TRIGGER_TIME);
    fprintf('Shed fraction:          %.2f\n', FIXED_FRACTION);
    fprintf('Ramp duration:          %.0fs\n', RAMP_DURATION);
    fprintf('Every other bus (including E): NO shedding.\n');

    %% 1. Build the load profile, identical to every prior sweep -----------
    fprintf('\nBuilding load profile (seed %d, scale %.2f) ...\n', LOAD_SEED, GRID_LOAD_SCALE);
    [~, ~, ~] = build_derating_loads(LOAD_SEED, N_STEPS, TS, BUS, GRID_LOAD_SCALE, PMIN, PMAX);

    %% 2. Locate and load the ramped model --------------------------------
    hh = dir(fullfile(PROJECT_ROOT, '**', [MODEL_NAME '.slx']));
    if isempty(hh)
        error(['Ramped model not found: %s.slx\n' ...
               'Run add_gradual_ramp_capability.m first, then ' ...
               'compare_sheddable_vs_ramped.m to verify it.'], MODEL_NAME);
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

    %% 3. Wire taps ------------------------------------------------------
    fprintf('\nWiring signal taps ...\n');
    wire_all_taps(MODEL_NAME, BUS, CONV);
    set_param(MODEL_NAME, 'StopTime', num2str(TEND));

    %% 4. Configure: F sheds at its real trigger, EVERYTHING else untouched ---
    for b = 1:numel(BUS)
        default_blk = sprintf('%s/ShedFrac_%s_default', MODEL_NAME, BUS{b});
        ensure_constant_block(MODEL_NAME, BUS{b}, default_blk, 1);
        ramp_blk = sprintf('%s/ShedFrac_%s_ramp', MODEL_NAME, BUS{b});
        if getSimulinkBlockHandle(ramp_blk) ~= -1
            set_param(ramp_blk, 'RisingSlewLimit', 'inf', 'FallingSlewLimit', '-inf');
        end
    end

    f_blk = sprintf('%s/ShedFrac_F_default', MODEL_NAME);
    set_shed_step(MODEL_NAME, 'F', f_blk, F_TRIGGER_TIME, 1, FIXED_FRACTION);
    if RAMP_DURATION <= 0
        set_param(ramp_block, 'RisingSlewLimit', 'inf', 'FallingSlewLimit', '-inf');
    else
        falling_slew = -(1 - FIXED_FRACTION) / RAMP_DURATION;
        set_param(ramp_block, 'RisingSlewLimit', 'inf', 'FallingSlewLimit', num2str(falling_slew));
    end

    % Tap the actual applied shed fraction (post-Saturation, the real
    % signal feeding the Product block), not just the commanded Step, so
    % the plot shows what the actuator actually did, ramp included.
    shedfrac_tap_name = 'V7F_ShedFrac_F_actual';
    shedfrac_log_name = 'V7L_ShedFrac_F_actual';
    for nm = {shedfrac_tap_name, shedfrac_log_name}
        ex = find_system(MODEL_NAME, 'SearchDepth', 1, 'LookUnderMasks', 'all', 'Name', nm{1});
        if ~isempty(ex); try; delete_block(ex{1}); catch; end; end
    end
    sat_outport = get_param(sprintf('%s/ShedFrac_F_limit', MODEL_NAME), 'PortHandles');
    add_block('simulink/Sinks/To Workspace', [MODEL_NAME '/' shedfrac_log_name], ...
        'VariableName', matlab.lang.makeValidName(shedfrac_tap_name), ...
        'SaveFormat', 'Timeseries', 'SampleTime', '0.01');
    tw_ports = get_param([MODEL_NAME '/' shedfrac_log_name], 'PortHandles');
    add_line(MODEL_NAME, sat_outport.Outport(1), tw_ports.Inport(1), 'autorouting', 'on');

    %% 5. Run the single simulation -----------------------------------------
    fprintf('\nRunning clean closed-loop sim (Bus F only) ...\n');
    tic;
    ws = warning('off', 'all');
    so = sim(MODEL_NAME, 'ReturnWorkspaceOutputs', 'on');
    warning(ws);
    fprintf('Finished in %.1f s wall clock.\n', toc);

    time_vec = (0:N_STEPS-1)' * TS;
    zoom_mask = time_vec >= ZOOM_LO & time_vec <= ZOOM_HI;
    zoom_time = time_vec(zoom_mask); %#ok<NASGU>

    fprintf('\n%s\n', repmat('=', 1, 70));
    fprintf('RESULT: ALL 10 BUSES (only F was shed)\n');
    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('%-6s %-10s %-12s %-10s\n', 'Bus', 'Shed?', 'Outcome', 'Min V');

    bus_results = struct('bus', {}, 'shed', {}, 'collapsed', {}, 'onset', {}, 'min_v', {});
    V_zoom = struct();
    all_safe = true;

    for b = 1:numel(BUS)
        bus = BUS{b};
        v_full = grab_series(so, sprintf('V_Bus_%s', bus), TS, N_STEPS);
        V_zoom.(bus) = v_full(zoom_mask);

        [c, onset, min_v] = detect_collapse_summary( ...
            time_vec, v_full, COLLAPSE_VOLTAGE_V, MIN_CONSECUTIVE_SAMPLES);

        shed_str = 'no'; if strcmp(bus, 'F'); shed_str = 'yes'; end
        outcome_str = 'survives'; if c; outcome_str = sprintf('COLLAPSE@%.0fs', onset); end
        fprintf('%-6s %-10s %-12s %-10.2f\n', bus, shed_str, outcome_str, min_v);

        if c; all_safe = false; end

        bus_results(end+1) = struct('bus', bus, 'shed', strcmp(bus, 'F'), ... %#ok<AGROW>
            'collapsed', c, 'onset', onset, 'min_v', min_v);
    end

    fprintf('\n');
    if all_safe
        fprintf('CLEAN TEST PASSED: with ONLY Bus F shed, using its real 50.85s\n');
        fprintf('held-out-bus lead time, every bus survives. This is attributable\n');
        fprintf('to Bus F''s genuine detector trigger alone, not to any spurious\n');
        fprintf('trigger on Bus E or elsewhere, since none of those were used here.\n');
    else
        fprintf('CLEAN TEST FAILED: at least one bus still collapses with only Bus F\n');
        fprintf('shed. If Bus E specifically collapses, that would mean Bus F''s real\n');
        fprintf('50.85s lead time alone is NOT sufficient to save the downstream bus,\n');
        fprintf('and the earlier all-bus "pass" was propped up by Bus E''s spurious\n');
        fprintf('early trigger, not a genuine result. This is an important negative\n');
        fprintf('finding, report it as such, do not rerun with E''s trigger added back\n');
        fprintf('in to make it pass.\n');
    end

    %% 6. Shed fraction actually applied, from the Saturation tap -----------
    shedfrac_full = grab_series(so, shedfrac_tap_name, TS, N_STEPS);
    shedfrac_zoom = shedfrac_full(zoom_mask);

    %% 7. PLOTS -----------------------------------------------------------
    fprintf('\nBuilding plots ...\n');
    collapse_color = [0.85 0.2 0.2];
    trigger_color  = [0.15 0.45 0.85];
    onset_color    = [0.9 0.55 0.1];

    % --- Plot 1: per-bus grid, all 10 buses, individual subplots -----------
    fig1 = figure('Position', [50, 50, 1600, 700], 'Visible', 'off');
    for b = 1:numel(BUS)
        bus = BUS{b};
        subplot(2, 5, b);
        plot(zoom_time, V_zoom.(bus), 'Color', [0.1 0.1 0.1], 'LineWidth', 1.1);
        hold on;
        yline(COLLAPSE_VOLTAGE_V, '--', 'Color', collapse_color, 'LineWidth', 0.8);
        xline(F_TRIGGER_TIME, ':', 'Color', trigger_color, 'LineWidth', 1.2);
        r = bus_results(b);
        if r.collapsed
            xline(r.onset, '-', 'Color', onset_color, 'LineWidth', 1.2);
            title_str = sprintf('%s: COLLAPSE @%.0fs', bus, r.onset);
            title_color = collapse_color;
        else
            title_str = sprintf('%s: survives (min %.0fV)', bus, r.min_v);
            title_color = [0.1 0.5 0.15];
        end
        shed_tag = ''; if strcmp(bus, 'F'); shed_tag = ' [SHED]'; end
        title([title_str shed_tag], 'Color', title_color, 'FontSize', 9);
        xlabel('t (s)', 'FontSize', 8);
        ylabel('V', 'FontSize', 8);
        ylim([0 850]);
        grid on;
        hold off;
    end
    sgtitle(sprintf('Clean closed-loop test: Bus F shed at t=%.2fs (real detector trigger), all others untouched', ...
                     F_TRIGGER_TIME), 'FontSize', 12, 'FontWeight', 'bold');
    per_bus_png = fullfile(OUTPUT_DIR, 'closed_loop_bus_f_only_per_bus_grid.png');
    saveas(fig1, per_bus_png);
    close(fig1);
    fprintf('Wrote: %s\n', per_bus_png);

    % --- Plot 2: overlay, all 10 buses on one axis -------------------------
    fig2 = figure('Position', [100, 100, 1100, 650], 'Visible', 'off');
    hold on;
    colors = lines(numel(BUS));
    for b = 1:numel(BUS)
        bus = BUS{b};
        lw = 1.1; if strcmp(bus, 'F'); lw = 2.2; end
        plot(zoom_time, V_zoom.(bus), 'Color', colors(b,:), 'LineWidth', lw, ...
             'DisplayName', bus);
    end
    yline(COLLAPSE_VOLTAGE_V, '--', 'collapse threshold', 'Color', collapse_color, 'LineWidth', 1);
    xline(F_TRIGGER_TIME, ':', sprintf('F trigger (t=%.1fs)', F_TRIGGER_TIME), ...
          'Color', trigger_color, 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom');
    xline(F_ONSET_REFERENCE, '-.', sprintf('F unshed onset (t=%.1fs)', F_ONSET_REFERENCE), ...
          'Color', onset_color, 'LineWidth', 1.2, 'LabelVerticalAlignment', 'top');
    xlabel('time (s)');
    ylabel('bus voltage (V)');
    title(sprintf('All 10 buses overlaid, only Bus F shed (fraction %.2f, ramp %.0fs)', ...
                   FIXED_FRACTION, RAMP_DURATION));
    legend('Location', 'eastoutside');
    grid on;
    hold off;
    overlay_png = fullfile(OUTPUT_DIR, 'closed_loop_bus_f_only_overlay.png');
    saveas(fig2, overlay_png);
    close(fig2);
    fprintf('Wrote: %s\n', overlay_png);

    % --- Plot 3: shed actuator trace + Bus F/E voltage, stacked -----------
    fig3 = figure('Position', [100, 100, 1100, 700], 'Visible', 'off');

    subplot(3,1,1);
    plot(zoom_time, shedfrac_zoom, 'Color', [0.2 0.3 0.7], 'LineWidth', 1.8);
    hold on;
    xline(F_TRIGGER_TIME, ':', 'Color', trigger_color, 'LineWidth', 1.2);
    ylabel('ShedFrac_F (applied)');
    title('Bus F shed actuator: commanded fraction actually applied (post-ramp, post-saturation)');
    ylim([-0.05 1.05]);
    grid on;
    hold off;

    subplot(3,1,2);
    plot(zoom_time, V_zoom.F, 'Color', [0.1 0.1 0.1], 'LineWidth', 1.3);
    hold on;
    yline(COLLAPSE_VOLTAGE_V, '--', 'Color', collapse_color, 'LineWidth', 0.8);
    xline(F_TRIGGER_TIME, ':', 'Color', trigger_color, 'LineWidth', 1.2);
    xline(F_ONSET_REFERENCE, '-.', 'Color', onset_color, 'LineWidth', 1);
    ylabel('V_{Bus F}');
    title('Bus F voltage (shed bus)');
    ylim([0 850]);
    grid on;
    hold off;

    subplot(3,1,3);
    plot(zoom_time, V_zoom.E, 'Color', [0.1 0.1 0.1], 'LineWidth', 1.3);
    hold on;
    yline(COLLAPSE_VOLTAGE_V, '--', 'Color', collapse_color, 'LineWidth', 0.8);
    xline(F_TRIGGER_TIME, ':', 'Color', trigger_color, 'LineWidth', 1.2);
    ylabel('V_{Bus E}');
    xlabel('time (s)');
    title('Bus E voltage (downstream, NOT shed, no intervention of its own)');
    ylim([0 850]);
    grid on;
    hold off;

    stacked_png = fullfile(OUTPUT_DIR, 'closed_loop_bus_f_only_actuator_and_voltage.png');
    saveas(fig3, stacked_png);
    close(fig3);
    fprintf('Wrote: %s\n', stacked_png);

    %% 8. Save -----------------------------------------------------------
    save(fullfile(OUTPUT_DIR, 'closed_loop_verification_bus_f_only_result.mat'), ...
         'bus_results', 'V_zoom', 'zoom_time', 'shedfrac_zoom', 'all_safe', ...
         'F_TRIGGER_TIME', 'F_ONSET_REFERENCE', 'FIXED_FRACTION', 'RAMP_DURATION', '-v7.3');
    fprintf('\nWrote: %s\n', fullfile(OUTPUT_DIR, 'closed_loop_verification_bus_f_only_result.mat'));
    fprintf('\nAll outputs in: %s\n', OUTPUT_DIR);
    fprintf('  - closed_loop_bus_f_only_per_bus_grid.png       (all 10 buses, individual)\n');
    fprintf('  - closed_loop_bus_f_only_overlay.png            (all 10 buses, one axis)\n');
    fprintf('  - closed_loop_bus_f_only_actuator_and_voltage.png (shed fraction + Bus F + Bus E)\n');
    fprintf('  - closed_loop_verification_bus_f_only_result.mat (raw data, no resim needed for replots)\n');

    close_system(MODEL_NAME, 0);
end


%% ---- block manipulation helpers (from test_shed_lead_time.m) ----

function ensure_constant_block(model, bus, blk_path, value)
    if getSimulinkBlockHandle(blk_path) == -1
        return;
    end
    bt = get_param(blk_path, 'BlockType');
    if strcmp(bt, 'Constant')
        set_param(blk_path, 'Value', num2str(value));
        return;
    end
    pos = get_param(blk_path, 'Position');
    ph = get_param(blk_path, 'PortHandles');
    dst_line = -1;
    if isfield(ph, 'Outport') && ~isempty(ph.Outport)
        dst_line = get_param(ph.Outport(1), 'Line');
    end
    dst_block = ''; dst_port = 1;
    if dst_line ~= -1
        dst_port_handle = get_param(dst_line, 'DstPortHandle');
        dst_block = get_param(dst_port_handle, 'Parent');
        dst_port = get_param(dst_port_handle, 'PortNumber');
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


function wire_all_taps(model, BUS, CONV) %#ok<INUSD>
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
