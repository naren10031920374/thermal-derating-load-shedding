function run_closed_loop_verification_busC_v4_only()
% ==========================================================================
%  RUN_CLOSED_LOOP_VERIFICATION_BUSC_V4_ONLY.M
%  --------------------------------------------------------------------
%  The gradual_busC_v4 analog of run_closed_loop_verification_bus_f_only.m
%  (baseline scenario's finale). Same clean-test philosophy: shed ONLY the
%  bus that actually collapses, using its real cross-scenario detector
%  trigger time, and leave every other bus (including its two structural
%  neighbors B and D, which were already pushed hard in this scenario's
%  own load profile) completely untouched. If everything survives, that
%  survival is attributable ONLY to shedding Bus C with real lead time,
%  nothing else.
%
%  WHERE THE TRIGGER TIME COMES FROM: the cross-scenario generalization
%  test (test_gradual_busC_v4_against_baseline_detectors.py) found that
%  the baseline-trained N=60s detector -- never retrained, never shown
%  this scenario during training -- cleanly detects this new Bus C
%  collapse: at threshold=1e-6, lead time = 1091.97s before the real
%  onset at t=2094.47s, with 0 false alarms on the other 9 buses. That
%  puts the trigger at t = 2094.47 - 1091.97 = 1002.50s. HARDCODED below
%  (not re-read from the JSON), same reasoning as the original Bus F
%  script: this test intentionally only cares about Bus C's own number.
%
%  LOAD PROFILE: reused verbatim from generate_gradual_busC_scenario_v4.m
%  (build_gradual_busC_loads_v4, seed=104, PMIN=2000/PMAX=220000, Bus C
%  ramped to 200,000W and Bus B/D dropped to 3,000W between t=800-2800s).
%  This MUST match exactly, or this wouldn't be the same collapse at all.
%
%  MODEL: reuses Grid_modelling_Thermal_V7_ALfix_sheddable_ramped.slx --
%  the same sheddable+ramp-capable model built once for the baseline
%  scenario. The shed-fraction actuator infrastructure
%  (add_controllable_load_shedding.m) was wired onto ALL 10 buses
%  generically, not just Bus F, so ShedFrac_C_default / ShedFrac_C_ramp
%  already exist in this file and can be driven the same way Bus F's was.
%  RAMP_DURATION=0 here (instantaneous shed), so the ramp capability is
%  present but not actually exercised, same as the Bus F script's default.
%
%  RUNTIME: 1 full 5000s sim() call. generate_gradual_busC_scenario_v4.m's
%  own run of this exact load profile took ~15-20 minutes wall clock; this
%  should be similar (shedding one bus's load doesn't materially change
%  solver difficulty).
%
%  Run:
%    run_closed_loop_verification_busC_v4_only
% ==========================================================================

    clc;

    %% 0. CONFIG --------------------------------------------------------
    PROJECT_ROOT = 'D:\naren\Documents\Thermal_Derating_Work_Handover\ThermalProject';
    MODEL_NAME = 'Grid_modelling_Thermal_V7_ALfix_sheddable_ramped';

    % Bus C's real, cross-scenario-generalization detector trigger, from
    % test_gradual_busC_v4_against_baseline_detectors.py's N=60s sweep
    % (threshold=1e-6, the best clean zero-false-alarm operating point).
    C_TRIGGER_TIME    = 1002.50;
    C_ONSET_REFERENCE = 2094.47;   % for display only, not used in the sim

    FIXED_FRACTION = 0.30;   % same starting point that worked for baseline (Bus F);
                             % NOT yet validated for this scenario's magnitude -- if
                             % Bus C (or B/D) still collapses below, try a larger
                             % fraction (e.g. 0.5, 0.7) before concluding shedding
                             % doesn't work here.
    RAMP_DURATION  = 0;      % 0 = instantaneous

    TS   = 0.01;
    TEND = 5000;
    N_STEPS = round(TEND / TS) + 1;

    % Exact load-profile parameters from generate_gradual_busC_scenario_v4.m --
    % must match exactly, this is what produced the real t=2094.47s collapse.
    SEED = 104;
    PMIN = 2000;
    PMAX = 220000;
    RAMP_START_S = 800;
    RAMP_END_S   = 2800;
    C_TARGET_W   = 200000;
    BD_TARGET_W  = 3000;

    COLLAPSE_VOLTAGE_V = 100.0;
    MIN_CONSECUTIVE_SAMPLES = 50;

    BUS  = {'A','B','C','D','E','F','G','H','K','L'};
    CONV = {'AB','BC','CD','DE','EF','FG','GH','HK','KL','AL'};

    ZOOM_LO = 500; ZOOM_HI = 3000;   % onset t=2094s is much earlier here than
                                      % baseline's t=3643s, so the zoom window
                                      % is shifted correspondingly

    OUTPUT_DIR = fullfile(PROJECT_ROOT, 'model_outputs', 'thermal_derating_gradual_busC', 'closed_loop_verification');
    if exist(OUTPUT_DIR, 'dir') ~= 7; mkdir(OUTPUT_DIR); end
    addpath(PROJECT_ROOT);

    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('CLEAN CLOSED-LOOP TEST: gradual_busC_v4, Bus C only, real cross-scenario trigger\n');
    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('Bus C trigger time:     %.2fs\n', C_TRIGGER_TIME);
    fprintf('Bus C onset (reference): %.2fs\n', C_ONSET_REFERENCE);
    fprintf('Real lead time:         %.2fs\n', C_ONSET_REFERENCE - C_TRIGGER_TIME);
    fprintf('Shed fraction:          %.2f\n', FIXED_FRACTION);
    fprintf('Ramp duration:          %.0fs\n', RAMP_DURATION);
    fprintf('Every other bus (including B and D, both heavily stressed by this scenario''s own load profile): NO shedding.\n');

    %% 1. Build the load profile -- IDENTICAL to generate_gradual_busC_scenario_v4.m
    fprintf('\nBuilding gradual_busC_v4 load profile (seed %d) ...\n', SEED);
    [~, ~, ~] = build_gradual_busC_loads_v4( ...
        SEED, N_STEPS, TS, BUS, PMIN, PMAX, RAMP_START_S, RAMP_END_S, ...
        C_TARGET_W, BD_TARGET_W);

    %% 2. Locate and load the sheddable+ramp model --------------------------
    hh = dir(fullfile(PROJECT_ROOT, '**', [MODEL_NAME '.slx']));
    if isempty(hh)
        error(['Ramped sheddable model not found: %s.slx\n' ...
               'This should already exist from the baseline scenario''s ' ...
               '05_load_shedding_infrastructure / 06_gradual_ramp work.'], MODEL_NAME);
    end
    model_file = fullfile(hh(1).folder, hh(1).name);
    fprintf('\nModel: %s\n', model_file);
    if bdIsLoaded(MODEL_NAME)
        close_system(MODEL_NAME, 0);
    end
    load_system(model_file);

    ramp_block = sprintf('%s/ShedFrac_C_ramp', MODEL_NAME);
    if getSimulinkBlockHandle(ramp_block) == -1
        error(['Block not found: %s\nExpected the shed-fraction ramp actuator ' ...
               'for Bus C to already exist in this model (it was wired onto ' ...
               'all 10 buses generically, not just Bus F). If this errors, ' ...
               'check add_controllable_load_shedding.m / add_gradual_ramp_capability.m ' ...
               'were actually run against all buses.'], ramp_block);
    end

    %% 3. Wire taps ------------------------------------------------------
    fprintf('\nWiring signal taps ...\n');
    wire_all_taps(MODEL_NAME, BUS, CONV);
    set_param(MODEL_NAME, 'StopTime', num2str(TEND));

    %% 4. Configure: C sheds at its real trigger, EVERYTHING else untouched ---
    for b = 1:numel(BUS)
        default_blk = sprintf('%s/ShedFrac_%s_default', MODEL_NAME, BUS{b});
        ensure_constant_block(MODEL_NAME, BUS{b}, default_blk, 1);
        ramp_blk = sprintf('%s/ShedFrac_%s_ramp', MODEL_NAME, BUS{b});
        if getSimulinkBlockHandle(ramp_blk) ~= -1
            set_param(ramp_blk, 'RisingSlewLimit', 'inf', 'FallingSlewLimit', '-inf');
        end
    end

    c_blk = sprintf('%s/ShedFrac_C_default', MODEL_NAME);
    set_shed_step(MODEL_NAME, 'C', c_blk, C_TRIGGER_TIME, 1, FIXED_FRACTION);
    if RAMP_DURATION <= 0
        set_param(ramp_block, 'RisingSlewLimit', 'inf', 'FallingSlewLimit', '-inf');
    else
        falling_slew = -(1 - FIXED_FRACTION) / RAMP_DURATION;
        set_param(ramp_block, 'RisingSlewLimit', 'inf', 'FallingSlewLimit', num2str(falling_slew));
    end

    % Tap the actual applied shed fraction (post-Saturation), same pattern
    % as the Bus F script.
    shedfrac_tap_name = 'V7F_ShedFrac_C_actual';
    shedfrac_log_name = 'V7L_ShedFrac_C_actual';
    for nm = {shedfrac_tap_name, shedfrac_log_name}
        ex = find_system(MODEL_NAME, 'SearchDepth', 1, 'LookUnderMasks', 'all', 'Name', nm{1});
        if ~isempty(ex); try; delete_block(ex{1}); catch; end; end
    end
    sat_outport = get_param(sprintf('%s/ShedFrac_C_limit', MODEL_NAME), 'PortHandles');
    add_block('simulink/Sinks/To Workspace', [MODEL_NAME '/' shedfrac_log_name], ...
        'VariableName', matlab.lang.makeValidName(shedfrac_tap_name), ...
        'SaveFormat', 'Timeseries', 'SampleTime', '0.01');
    tw_ports = get_param([MODEL_NAME '/' shedfrac_log_name], 'PortHandles');
    add_line(MODEL_NAME, sat_outport.Outport(1), tw_ports.Inport(1), 'autorouting', 'on');

    %% 5. Run the single simulation -----------------------------------------
    fprintf('\nRunning clean closed-loop sim (Bus C only) ...\n');
    tic;
    ws = warning('off', 'all');
    so = sim(MODEL_NAME, 'ReturnWorkspaceOutputs', 'on');
    warning(ws);
    fprintf('Finished in %.1f s wall clock.\n', toc);

    time_vec = (0:N_STEPS-1)' * TS;
    zoom_mask = time_vec >= ZOOM_LO & time_vec <= ZOOM_HI;
    zoom_time = time_vec(zoom_mask); %#ok<NASGU>

    fprintf('\n%s\n', repmat('=', 1, 70));
    fprintf('RESULT: ALL 10 BUSES (only C was shed)\n');
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

        shed_str = 'no'; if strcmp(bus, 'C'); shed_str = 'yes'; end
        outcome_str = 'survives'; if c; outcome_str = sprintf('COLLAPSE@%.0fs', onset); end
        fprintf('%-6s %-10s %-12s %-10.2f\n', bus, shed_str, outcome_str, min_v);

        if c; all_safe = false; end

        bus_results(end+1) = struct('bus', bus, 'shed', strcmp(bus, 'C'), ... %#ok<AGROW>
            'collapsed', c, 'onset', onset, 'min_v', min_v);
    end

    fprintf('\n');
    if all_safe
        fprintf('CLEAN TEST PASSED: with ONLY Bus C shed, using its real %.2fs\n', ...
                C_ONSET_REFERENCE - C_TRIGGER_TIME);
        fprintf('cross-scenario detector lead time, every bus survives -- including\n');
        fprintf('B and D, which this scenario''s own load profile pushed hard (dropped\n');
        fprintf('to 3,000W each). This is attributable to Bus C being shed with real\n');
        fprintf('lead time alone, nothing else, since no other bus got any intervention.\n');
    else
        fprintf('CLEAN TEST FAILED: at least one bus still collapses with only Bus C\n');
        fprintf('shed. If B and/or D specifically collapse, that would mean shedding\n');
        fprintf('Bus C alone is not sufficient to protect its structurally stressed\n');
        fprintf('neighbors, and FIXED_FRACTION=0.30 (or the %.2fs lead time itself)\n', ...
                C_ONSET_REFERENCE - C_TRIGGER_TIME);
        fprintf('may need to be revisited. This is an important negative finding,\n');
        fprintf('report it as such -- do not add shedding on B/D just to force a pass.\n');
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
        xline(C_TRIGGER_TIME, ':', 'Color', trigger_color, 'LineWidth', 1.2);
        r = bus_results(b);
        if r.collapsed
            xline(r.onset, '-', 'Color', onset_color, 'LineWidth', 1.2);
            title_str = sprintf('%s: COLLAPSE @%.0fs', bus, r.onset);
            title_color = collapse_color;
        else
            title_str = sprintf('%s: survives (min %.0fV)', bus, r.min_v);
            title_color = [0.1 0.5 0.15];
        end
        shed_tag = ''; if strcmp(bus, 'C'); shed_tag = ' [SHED]'; end
        title([title_str shed_tag], 'Color', title_color, 'FontSize', 9);
        xlabel('t (s)', 'FontSize', 8);
        ylabel('V', 'FontSize', 8);
        ylim([0 850]);
        grid on;
        hold off;
    end
    sgtitle(sprintf('gradual\\_busC\\_v4 clean closed-loop test: Bus C shed at t=%.2fs (real cross-scenario N=60s trigger), all others untouched', ...
                     C_TRIGGER_TIME), 'FontSize', 12, 'FontWeight', 'bold');
    per_bus_png = fullfile(OUTPUT_DIR, 'closed_loop_busC_v4_only_per_bus_grid.png');
    saveas(fig1, per_bus_png);
    close(fig1);
    fprintf('Wrote: %s\n', per_bus_png);

    % --- Plot 2: overlay, all 10 buses on one axis -------------------------
    fig2 = figure('Position', [100, 100, 1100, 650], 'Visible', 'off');
    hold on;
    colors = lines(numel(BUS));
    for b = 1:numel(BUS)
        bus = BUS{b};
        lw = 1.1; if strcmp(bus, 'C'); lw = 2.2; end
        plot(zoom_time, V_zoom.(bus), 'Color', colors(b,:), 'LineWidth', lw, ...
             'DisplayName', bus);
    end
    yline(COLLAPSE_VOLTAGE_V, '--', 'collapse threshold', 'Color', collapse_color, 'LineWidth', 1);
    xline(C_TRIGGER_TIME, ':', sprintf('C trigger (t=%.1fs)', C_TRIGGER_TIME), ...
          'Color', trigger_color, 'LineWidth', 1.5, 'LabelVerticalAlignment', 'bottom');
    xline(C_ONSET_REFERENCE, '-.', sprintf('C unshed onset (t=%.1fs)', C_ONSET_REFERENCE), ...
          'Color', onset_color, 'LineWidth', 1.2, 'LabelVerticalAlignment', 'top');
    xlabel('time (s)');
    ylabel('bus voltage (V)');
    title(sprintf('gradual\\_busC\\_v4: all 10 buses overlaid, only Bus C shed (fraction %.2f, ramp %.0fs)', ...
                   FIXED_FRACTION, RAMP_DURATION));
    legend('Location', 'eastoutside');
    grid on;
    hold off;
    overlay_png = fullfile(OUTPUT_DIR, 'closed_loop_busC_v4_only_overlay.png');
    saveas(fig2, overlay_png);
    close(fig2);
    fprintf('Wrote: %s\n', overlay_png);

    % --- Plot 3: shed actuator trace + Bus C/B/D voltage, stacked ----------
    fig3 = figure('Position', [100, 100, 1100, 900], 'Visible', 'off');

    subplot(4,1,1);
    plot(zoom_time, shedfrac_zoom, 'Color', [0.2 0.3 0.7], 'LineWidth', 1.8);
    hold on;
    xline(C_TRIGGER_TIME, ':', 'Color', trigger_color, 'LineWidth', 1.2);
    ylabel('ShedFrac_C');
    title('Bus C shed actuator: commanded fraction actually applied (post-saturation)');
    ylim([-0.05 1.05]);
    grid on;
    hold off;

    subplot(4,1,2);
    plot(zoom_time, V_zoom.C, 'Color', [0.1 0.1 0.1], 'LineWidth', 1.3);
    hold on;
    yline(COLLAPSE_VOLTAGE_V, '--', 'Color', collapse_color, 'LineWidth', 0.8);
    xline(C_TRIGGER_TIME, ':', 'Color', trigger_color, 'LineWidth', 1.2);
    xline(C_ONSET_REFERENCE, '-.', 'Color', onset_color, 'LineWidth', 1);
    ylabel('V_{Bus C}');
    title('Bus C voltage (shed bus)');
    ylim([0 850]);
    grid on;
    hold off;

    subplot(4,1,3);
    plot(zoom_time, V_zoom.B, 'Color', [0.1 0.1 0.1], 'LineWidth', 1.3);
    hold on;
    yline(COLLAPSE_VOLTAGE_V, '--', 'Color', collapse_color, 'LineWidth', 0.8);
    xline(C_TRIGGER_TIME, ':', 'Color', trigger_color, 'LineWidth', 1.2);
    ylabel('V_{Bus B}');
    title('Bus B voltage (structural neighbor, dropped to 3,000W by this scenario, NOT shed)');
    ylim([0 850]);
    grid on;
    hold off;

    subplot(4,1,4);
    plot(zoom_time, V_zoom.D, 'Color', [0.1 0.1 0.1], 'LineWidth', 1.3);
    hold on;
    yline(COLLAPSE_VOLTAGE_V, '--', 'Color', collapse_color, 'LineWidth', 0.8);
    xline(C_TRIGGER_TIME, ':', 'Color', trigger_color, 'LineWidth', 1.2);
    ylabel('V_{Bus D}');
    xlabel('time (s)');
    title('Bus D voltage (structural neighbor, dropped to 3,000W by this scenario, NOT shed)');
    ylim([0 850]);
    grid on;
    hold off;

    stacked_png = fullfile(OUTPUT_DIR, 'closed_loop_busC_v4_only_actuator_and_voltage.png');
    saveas(fig3, stacked_png);
    close(fig3);
    fprintf('Wrote: %s\n', stacked_png);

    %% 8. Save -----------------------------------------------------------
    save(fullfile(OUTPUT_DIR, 'closed_loop_verification_busC_v4_only_result.mat'), ...
         'bus_results', 'V_zoom', 'zoom_time', 'shedfrac_zoom', 'all_safe', ...
         'C_TRIGGER_TIME', 'C_ONSET_REFERENCE', 'FIXED_FRACTION', 'RAMP_DURATION', '-v7.3');
    fprintf('\nWrote: %s\n', fullfile(OUTPUT_DIR, 'closed_loop_verification_busC_v4_only_result.mat'));
    fprintf('\nAll outputs in: %s\n', OUTPUT_DIR);
    fprintf('  - closed_loop_busC_v4_only_per_bus_grid.png       (all 10 buses, individual)\n');
    fprintf('  - closed_loop_busC_v4_only_overlay.png            (all 10 buses, one axis)\n');
    fprintf('  - closed_loop_busC_v4_only_actuator_and_voltage.png (shed fraction + Bus C/B/D)\n');
    fprintf('  - closed_loop_verification_busC_v4_only_result.mat (raw data, no resim needed for replots)\n');

    close_system(MODEL_NAME, 0);
end


%% ---- gradual_busC_v4's own load-profile function, reused verbatim ----

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


%% ---- block manipulation + tap-wiring helpers (verbatim from the Bus F script) ----

function ensure_constant_block(model, bus, blk_path, value) %#ok<INUSL>
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