function test_shed_gradual_ramp_corridor_triad_collapse(duration_idx)
% ==========================================================================
%  TEST_SHED_GRADUAL_RAMP_CORRIDOR_TRIAD_COLLAPSE.M
%  --------------------------------------------------------------------
%  Converts stage 05's best-case instantaneous-step result into a
%  physically realistic one. test_shed_lead_time_corridor_triad_collapse.m
%  found that an INSTANT step to fraction 0.80 on Bus E+F still prevents
%  collapse even when triggered as late as t=1140s -- only 9.1s before the
%  natural t=1149.1s collapse. That is a best-case bound: no real actuator
%  can jump load instantly, and 9s of margin is thin if a smooth ramp eats
%  into it.
%
%  This script asks the question that actually matters for deployment:
%  the selected early-warning detector (N=30s, threshold 0.1, see
%  scripts\04_detector_evaluation\FINAL_DETECTOR_SELECTION_corridor_triad_collapse.md)
%  gives 30 seconds of warning. If shedding is triggered EXACTLY when that
%  detector would fire -- t = 1149.1 - 30 = 1119.1s -- and the actuator
%  ramps smoothly from 1.0 to 0.80 over some duration D (instead of
%  jumping instantly), how large can D be before the ramp is too slow and
%  collapse happens anyway?
%
%  PREREQUISITE: compare_sheddable_vs_ramped_corridor_triad_collapse.m
%  must already show ALL PASS. Do not trust this script's results
%  otherwise -- it would mean the Rate Limiter itself behaves differently
%  than expected on this scenario's load profile.
%
%  MECHANISM: for each duration D, Bus E and F's ShedFrac_<bus>_default
%  blocks are swapped to Step (Before=1.0, After=0.80, StepTime=1119.1),
%  and their ShedFrac_<bus>_ramp Rate Limiter blocks get
%  FallingSlewLimit = -(1.0-0.80)/D so the Rate Limiter takes exactly D
%  seconds to slew from 1.0 to 0.80 once the Step fires. All other buses
%  stay at Constant=1 (no shedding). None of this is saved to disk; the
%  verified .slx is untouched.
%
%  RUNTIME: 8 full 5000s sim() calls. At this scenario's observed
%  ~800-950s/run, expect roughly 2-2.25 hours total run serially.
%
%  Run (serial, all 8 durations, original behavior -- unchanged):
%    test_shed_gradual_ramp_corridor_triad_collapse
%
%  Run (HPC array-job mode -- one duration per call, e.g. one per SLURM
%  array task, so all 8 run in parallel instead of serially):
%    test_shed_gradual_ramp_corridor_triad_collapse(duration_idx)
%  where duration_idx is 1-8, indexing into RAMP_DURATIONS below. This
%  mode saves a per-duration partial .mat file and returns immediately
%  after that one sim() call -- it does NOT produce the summary table or
%  plots. Once all 8 array tasks finish, run combine_gradual_ramp_results.m
%  once to reproduce the exact same summary table and 3 plots the serial
%  run would have produced.
% ==========================================================================

    if nargin < 1
        duration_idx = [];
    end

    clc;

    %% 0. CONFIG --------------------------------------------------------
    % Portable root: this script always lives at <root>/scripts/06_gradual_ramp/,
    % so walk up two levels from this file's own location. Identical result
    % on Windows (dev machine) and Linux (Great Lakes / any other checkout).
    PROJECT_ROOT = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    MODEL_NAME = 'Grid_modelling_Thermal_V7_ALfix_sheddable_ramped';

    SHED_BUSES = {'E', 'F'};
    FIXED_FRACTION = 0.80;
    NATURAL_COLLAPSE_T = 1149.1;
    DETECTOR_LEAD_TIME = 30;                          % N=30s, the selected detector
    FIXED_START_TIME = NATURAL_COLLAPSE_T - DETECTOR_LEAD_TIME;  % = 1119.1s

    % Ramp durations to test: brackets the DETECTOR_LEAD_TIME itself (30s)
    % plus values well below and somewhat above it, since stage 05 showed
    % the system tolerates reaching full shed later than the naive
    % "instant" intuition would suggest.
    RAMP_DURATIONS = [1, 5, 10, 15, 20, 25, 30, 40];

    TS   = 0.01;
    TEND = 5000;
    N_STEPS = round(TEND / TS) + 1;
    SEED = 7;                          % must match generate_corridor_triad_collapse.m exactly
    PMIN = 20000;
    PMAX = 100000;

    COLLAPSE_VOLTAGE_V = 100.0;
    MIN_CONSECUTIVE_SAMPLES = 50;

    BUS  = {'A','B','C','D','E','F','G','H','K','L'};
    CONV = {'AB','BC','CD','DE','EF','FG','GH','HK','KL','AL'};

    WATCH_BUSES = {'E', 'F', 'G'};
    WATCH_CONV  = {'DE', 'EF', 'FG', 'GH'};

    OUTPUT_DIR = fullfile(PROJECT_ROOT, 'model_outputs', 'thermal_derating_v7', ...
                           'corridor_triad_collapse_gradual_ramp_sweep');
    if exist(OUTPUT_DIR, 'dir') ~= 7; mkdir(OUTPUT_DIR); end
    addpath(PROJECT_ROOT);

    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('CORRIDOR_TRIAD_COLLAPSE: GRADUAL RAMP SWEEP, Bus E+F to %.2f\n', FIXED_FRACTION);
    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('Fixed shed start time: t=%.1fs (= N=30s detector lead time before t=%.1fs collapse)\n', ...
            FIXED_START_TIME, NATURAL_COLLAPSE_T);
    fprintf('Ramp durations to test: %s\n', mat2str(RAMP_DURATIONS));
    fprintf('Question: how long can the actuator take to reach 0.80 and still prevent collapse,\n');
    fprintf('given it starts exactly when the selected N=30s detector would fire?\n');

    %% 1. Locate and load the RAMPED model ------------------------------
    hh = dir(fullfile(PROJECT_ROOT, '**', [MODEL_NAME '.slx']));
    if isempty(hh)
        error(['Ramped model not found: %s.slx\n' ...
               'Run add_gradual_ramp_capability.m first, then ' ...
               'compare_sheddable_vs_ramped_corridor_triad_collapse.m to ' ...
               'verify it before this.'], MODEL_NAME);
    end
    model_file = fullfile(hh(1).folder, hh(1).name);
    fprintf('\nModel: %s\n', model_file);
    if bdIsLoaded(MODEL_NAME)
        close_system(MODEL_NAME, 0);
    end
    load_system(model_file);

    default_blocks = cell(1, numel(SHED_BUSES));
    ramp_blocks    = cell(1, numel(SHED_BUSES));
    for i = 1:numel(SHED_BUSES)
        default_blocks{i} = sprintf('%s/ShedFrac_%s_default', MODEL_NAME, SHED_BUSES{i});
        ramp_blocks{i}    = sprintf('%s/ShedFrac_%s_ramp', MODEL_NAME, SHED_BUSES{i});
        if getSimulinkBlockHandle(default_blocks{i}) == -1
            error('Block not found: %s', default_blocks{i});
        end
        if getSimulinkBlockHandle(ramp_blocks{i}) == -1
            error(['Block not found: %s\nRun add_gradual_ramp_capability.m ' ...
                   'on the sheddable model first.'], ramp_blocks{i});
        end
    end

    %% 2. Wire taps ONCE, reused across all durations ------------------------
    fprintf('\nWiring signal taps (once, reused across all durations) ...\n');
    wire_all_taps(MODEL_NAME, BUS, CONV);
    derate_var_names = tap_derate_factors(MODEL_NAME, CONV);

    set_param(MODEL_NAME, 'StopTime', num2str(TEND));

    %% 3. Sweep -------------------------------------------------------------
    time_vec = (0:N_STEPS-1)' * TS;
    n_runs = numel(RAMP_DURATIONS);

    V = struct();
    for b = 1:numel(WATCH_BUSES)
        V.(WATCH_BUSES{b}) = nan(N_STEPS, n_runs);
    end
    DER = struct();
    for c = 1:numel(WATCH_CONV)
        DER.(WATCH_CONV{c}) = nan(N_STEPS, n_runs);
    end

    results = struct('ramp_duration', {}, 'collapsed', {}, 'onset', {}, 'min_v', {});

    if isempty(duration_idx)
        run_indices = 1:n_runs;
    else
        if duration_idx < 1 || duration_idx > n_runs || duration_idx ~= round(duration_idx)
            error('duration_idx must be an integer between 1 and %d (got %g)', n_runs, duration_idx);
        end
        run_indices = duration_idx;
        fprintf('\nHPC array-job mode: running ONLY duration index %d of %d (D=%.0fs).\n', ...
                duration_idx, n_runs, RAMP_DURATIONS(duration_idx));
    end

    for r = run_indices
        ramp_dur = RAMP_DURATIONS(r);
        fprintf('\n%s\n', repmat('-', 1, 70));
        fprintf('RUN %d/%d: shed starts at t=%.1fs, ramps to %.2f over %.0fs\n', ...
                r, n_runs, FIXED_START_TIME, FIXED_FRACTION, ramp_dur);
        fprintf('%s\n', repmat('-', 1, 70));

        % Reset every bus to constant no-shedding first (safe default).
        for b = 1:numel(BUS)
            blk = sprintf('%s/ShedFrac_%s_default', MODEL_NAME, BUS{b});
            if getSimulinkBlockHandle(blk) ~= -1
                ensure_constant_block(MODEL_NAME, BUS{b}, blk, 1);
            end
        end
        % Also reset every bus's Rate Limiter to no-op, in case a prior run
        % (or a different script) left one non-default.
        for b = 1:numel(BUS)
            rblk = sprintf('%s/ShedFrac_%s_ramp', MODEL_NAME, BUS{b});
            if getSimulinkBlockHandle(rblk) ~= -1
                set_param(rblk, 'RisingSlewLimit', 'inf', 'FallingSlewLimit', '-inf');
            end
        end

        % Swap E and F's default blocks to time-triggered Steps, and set
        % their Rate Limiters to slew over this run's duration.
        falling_slew = -(1 - FIXED_FRACTION) / ramp_dur;
        for i = 1:numel(SHED_BUSES)
            set_shed_step(MODEL_NAME, SHED_BUSES{i}, default_blocks{i}, ...
                           FIXED_START_TIME, 1, FIXED_FRACTION);
            set_param(ramp_blocks{i}, 'RisingSlewLimit', 'inf', ...
                      'FallingSlewLimit', num2str(falling_slew));
        end

        % Re-populate Pload_<bus> workspace vars fresh each run: RNG state
        % must be reset to SEED every time so every run sees the identical
        % underlying load profile.
        build_corridor_triad_loads(SEED, N_STEPS, TS, BUS, PMIN, PMAX);

        tic;
        ws = warning('off', 'all');
        so = sim(MODEL_NAME, 'ReturnWorkspaceOutputs', 'on');
        warning(ws);
        fprintf('Finished in %.1f s wall clock.\n', toc);

        for b = 1:numel(WATCH_BUSES)
            bus = WATCH_BUSES{b};
            V.(bus)(:, r) = grab_series(so, sprintf('V_Bus_%s', bus), TS, N_STEPS);
        end
        for c = 1:numel(WATCH_CONV)
            conv_idx = find(strcmp(CONV, WATCH_CONV{c}));
            der_raw = grab_series(so, derate_var_names{conv_idx}, TS, N_STEPS);
            DER.(WATCH_CONV{c})(:, r) = der_raw / 100e3;
        end

        collapsed = false(1, numel(WATCH_BUSES));
        onsets = nan(1, numel(WATCH_BUSES));
        min_vs = nan(1, numel(WATCH_BUSES));
        for b = 1:numel(WATCH_BUSES)
            bus = WATCH_BUSES{b};
            [c_flag, onset, min_v] = detect_collapse_summary( ...
                time_vec, V.(bus)(:, r), COLLAPSE_VOLTAGE_V, MIN_CONSECUTIVE_SAMPLES);
            collapsed(b) = c_flag; onsets(b) = onset; min_vs(b) = min_v;
            if c_flag
                fprintf('  Bus %s: COLLAPSES, onset t=%.1fs, min V=%.2f\n', bus, onset, min_v);
            else
                fprintf('  Bus %s: survives, min V=%.2f\n', bus, min_v);
            end
        end

        results(end+1) = struct('ramp_duration', ramp_dur, 'collapsed', collapsed, ...
                                 'onset', onsets, 'min_v', min_vs); %#ok<AGROW>
    end

    if ~isempty(duration_idx)
        % HPC array-job mode: save just this duration's slice and stop here.
        % Run combine_gradual_ramp_results.m after all array tasks finish to
        % get the exact same summary table and 3 plots the serial run makes.
        r = duration_idx;
        Vr = struct();
        for b = 1:numel(WATCH_BUSES)
            Vr.(WATCH_BUSES{b}) = V.(WATCH_BUSES{b})(:, r);
        end
        DERr = struct();
        for c = 1:numel(WATCH_CONV)
            DERr.(WATCH_CONV{c}) = DER.(WATCH_CONV{c})(:, r);
        end
        res_r = results(end);
        ramp_dur = RAMP_DURATIONS(r); %#ok<NASGU>
        partial_file = fullfile(OUTPUT_DIR, sprintf('gradual_ramp_partial_D%02d.mat', RAMP_DURATIONS(r)));
        save(partial_file, 'Vr', 'DERr', 'res_r', 'time_vec', 'ramp_dur', ...
             'FIXED_FRACTION', 'FIXED_START_TIME', 'DETECTOR_LEAD_TIME', ...
             'NATURAL_COLLAPSE_T', 'WATCH_BUSES', 'WATCH_CONV', '-v7.3');
        fprintf('\nWrote partial result (duration=%.0fs) to: %s\n', RAMP_DURATIONS(r), partial_file);
        fprintf('Once all %d array tasks have finished, run combine_gradual_ramp_results.m\n', n_runs);
        close_system(MODEL_NAME, 0);
        return;
    end

    %% 4. Summary table -------------------------------------------------------
    fprintf('\n%s\n', repmat('=', 1, 70));
    fprintf('SUMMARY (shed starts t=%.1fs, Bus E+F ramp to %.2f over duration D)\n', ...
            FIXED_START_TIME, FIXED_FRACTION);
    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('%-12s', 'duration_D');
    for b = 1:numel(WATCH_BUSES)
        fprintf(' %-16s', sprintf('Bus %s', WATCH_BUSES{b}));
    end
    fprintf('\n');
    for r = 1:n_runs
        res = results(r);
        fprintf('%-12.0f', res.ramp_duration);
        for b = 1:numel(WATCH_BUSES)
            if res.collapsed(b)
                fprintf(' %-16s', sprintf('COLLAPSE@%.0fs', res.onset(b)));
            else
                fprintf(' %-16s', 'survives');
            end
        end
        fprintf('\n');
    end

    both_safe = arrayfun(@(r) ~r.collapsed(1) && ~r.collapsed(2), results); % E, F both survive
    if any(both_safe)
        max_safe_duration = max([results(both_safe).ramp_duration]);
        fprintf('\nLongest tested ramp duration with BOTH Bus E and Bus F surviving: %.0fs\n', ...
                max_safe_duration);
        fprintf('(triggered at t=%.1fs, exactly N=30s''s lead time before the %.1fs collapse)\n', ...
                FIXED_START_TIME, NATURAL_COLLAPSE_T);
        if max_safe_duration >= DETECTOR_LEAD_TIME
            fprintf('\nThis is >= the detector''s own %.0fs lead time -- meaning a ramp taking as\n', ...
                    DETECTOR_LEAD_TIME);
            fprintf('long as the full warning window itself still worked in this test. Sanity-check\n');
            fprintf('this against the untested durations above %.0fs before trusting it fully.\n', ...
                    max(RAMP_DURATIONS));
        else
            fprintf('\nThis is LESS than the detector''s %.0fs lead time -- meaning a realistic\n', ...
                    DETECTOR_LEAD_TIME);
            fprintf('actuator that takes longer than %.0fs to fully shed would collapse anyway,\n', ...
                    max_safe_duration);
            fprintf('even though the detector fired with a full 30s of warning. The margin is\n');
            fprintf('%.0fs of actuator speed, not %.0fs of warning time.\n', ...
                    max_safe_duration, DETECTOR_LEAD_TIME);
        end
    else
        fprintf('\nNo tested ramp duration kept both Bus E and Bus F safe when triggered at\n');
        fprintf('t=%.1fs. Even the fastest tested ramp (%.0fs) failed -- re-check against\n', ...
                FIXED_START_TIME, min(RAMP_DURATIONS));
        fprintf('stage 05''s instantaneous-step result (which succeeded at this exact start time)\n');
        fprintf('before trusting this; a discrepancy would point to a Rate Limiter wiring issue.\n');
    end

    %% 5. Plots -----------------------------------------------------------------
    fprintf('\nBuilding plots ...\n');
    zoom_lo = 1000; zoom_hi = 1300;
    mask = time_vec >= zoom_lo & time_vec <= zoom_hi;

    % --- Outcome dot chart (duration on x-axis) ---
    fig1 = figure('Position', [100, 100, 1200, 400], 'Visible', 'off');
    hold on;
    for r = 1:n_runs
        for b = 1:numel(WATCH_BUSES)
            color = [0.2 0.75 0.2];
            if results(r).collapsed(b); color = [0.85 0.2 0.2]; end
            plot(results(r).ramp_duration, numel(WATCH_BUSES) - b + 1, 'o', ...
                 'MarkerSize', 16, 'MarkerFaceColor', color, 'MarkerEdgeColor', 'w');
        end
    end
    xline(DETECTOR_LEAD_TIME, ':k', sprintf('N=30s detector lead time (%.0fs)', DETECTOR_LEAD_TIME));
    set(gca, 'YTick', 1:numel(WATCH_BUSES), ...
             'YTickLabel', fliplr(WATCH_BUSES), 'YLim', [0.5, numel(WATCH_BUSES) + 0.5]);
    xlabel('ramp duration D (s)');
    title(sprintf('corridor triad collapse: outcome vs ramp duration (shed starts t=%.1fs, fraction %.2f)', ...
                  FIXED_START_TIME, FIXED_FRACTION));
    grid on; hold off;
    saveas(fig1, fullfile(OUTPUT_DIR, 'gradual_ramp_outcome_summary.png'));
    close(fig1);

    % --- Bus voltages across durations, zoomed ---
    fig2 = figure('Position', [100, 100, 1400, 700], 'Visible', 'off');
    colors = winter(n_runs);
    for b = 1:numel(WATCH_BUSES)
        subplot(numel(WATCH_BUSES), 1, b); hold on;
        bus = WATCH_BUSES{b};
        for r = 1:n_runs
            plot(time_vec(mask), V.(bus)(mask, r), 'Color', colors(r,:), 'LineWidth', 1.1, ...
                 'DisplayName', sprintf('D=%.0fs', RAMP_DURATIONS(r)));
        end
        yline(COLLAPSE_VOLTAGE_V, '--k', '100V threshold');
        xline(FIXED_START_TIME, ':k', 'shed start');
        ylabel(sprintf('Bus %s (V)', bus));
        if b == 1
            title('Bus voltages vs ramp duration, zoomed to the crisis window');
        end
        if b == numel(WATCH_BUSES)
            xlabel('time (s)');
            legend('Location', 'eastoutside', 'Interpreter', 'none', 'FontSize', 7);
        end
        grid on; hold off;
    end
    saveas(fig2, fullfile(OUTPUT_DIR, 'gradual_ramp_voltages.png'));
    close(fig2);

    % --- Derate factors across durations, zoomed ---
    fig3 = figure('Position', [100, 100, 1400, 800], 'Visible', 'off');
    for c = 1:numel(WATCH_CONV)
        subplot(numel(WATCH_CONV), 1, c); hold on;
        conv = WATCH_CONV{c};
        for r = 1:n_runs
            plot(time_vec(mask), DER.(conv)(mask, r), 'Color', colors(r,:), 'LineWidth', 1.0, ...
                 'DisplayName', sprintf('D=%.0fs', RAMP_DURATIONS(r)));
        end
        ylim([0, 1.05]);
        ylabel(conv);
        if c == 1
            title('Converter derate factors vs ramp duration, zoomed');
        end
        if c == numel(WATCH_CONV)
            xlabel('time (s)');
            legend('Location', 'eastoutside', 'Interpreter', 'none', 'FontSize', 7);
        end
        grid on; hold off;
    end
    saveas(fig3, fullfile(OUTPUT_DIR, 'gradual_ramp_derate_factors.png'));
    close(fig3);

    save(fullfile(OUTPUT_DIR, 'gradual_ramp_sweep_results.mat'), ...
         'V', 'DER', 'time_vec', 'RAMP_DURATIONS', 'FIXED_FRACTION', 'FIXED_START_TIME', ...
         'DETECTOR_LEAD_TIME', 'NATURAL_COLLAPSE_T', 'SHED_BUSES', ...
         'WATCH_BUSES', 'WATCH_CONV', 'results', '-v7.3');

    fprintf('\nWrote plots and results to: %s\n', OUTPUT_DIR);
    fprintf('\nDone. Read gradual_ramp_outcome_summary.png first: the longest ramp duration\n');
    fprintf('with both Bus E and Bus F dots green is the answer to "how slow can the real\n');
    fprintf('actuator be and still work, given the detector''s exact 30s of warning".\n');

    close_system(MODEL_NAME, 0);
end


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
    out_line = get_param(blk_path, 'PortHandles');
    dst_line = -1;
    if isfield(out_line, 'Outport') && ~isempty(out_line.Outport)
        dst_line = get_param(out_line.Outport(1), 'Line');
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


function [load_hist, meta] = build_corridor_triad_loads(seed, n_steps, ts, BUS, pmin, pmax)
% VERBATIM from generate_corridor_triad_collapse.m -- same SEED, same
% 4-phase mechanism, so this sweep reproduces the exact validated scenario.
    rng(seed);
    t = (0:n_steps-1)' * ts;
    NB = numel(BUS);

    nominal = [35000 30000 25000 32000 28000 24000 30000 26000 22000 34000];

    idxD = find(strcmp(BUS,'D')); idxE = find(strcmp(BUS,'E'));
    idxF = find(strcmp(BUS,'F')); idxG = find(strcmp(BUS,'G'));
    idxH = find(strcmp(BUS,'H'));

    target_idx     = [idxE idxF idxG];
    boundary_idx   = [idxD idxH];
    background_idx = setdiff(1:NB, [target_idx, boundary_idx]);

    PHASE1_END = 1000;
    PHASE2_END = 1150;
    PHASE3_END = 2800;

    PHASE1_MULT     = 1.08;
    PHASE3_TARGET_W = 95000;
    PHASE4_MULT     = 1.50;
    BOUNDARY_RECOVER_MULT = 1.00;

    NOISE_FRAC   = 0.02;
    BG_SINE_FRAC = 0.04;
    BG_SINE_PERIOD = 900;

    load_hist = nan(n_steps, NB);

    jitter = ones(1, NB);
    for k = target_idx
        jitter(k) = 1 + 0.03 * (2*rand() - 1);
    end

    for k = 1:NB
        nom = nominal(k);

        if ismember(k, target_idx)
            phase1_level = nom * PHASE1_MULT;
            phase3_level = PHASE3_TARGET_W;

            raw = nan(n_steps, 1);
            m1 = t <= PHASE1_END;
            raw(m1) = phase1_level;
            m2 = t > PHASE1_END & t <= PHASE2_END;
            frac = (t(m2) - PHASE1_END) / (PHASE2_END - PHASE1_END);
            raw(m2) = phase1_level + frac .* (phase3_level - phase1_level);
            m3 = t > PHASE2_END & t <= PHASE3_END;
            raw(m3) = phase3_level;
            m4 = t > PHASE3_END;
            raw(m4) = nom * PHASE4_MULT;
            raw = raw * jitter(k);

        elseif ismember(k, boundary_idx)
            lvl = nan(n_steps, 1);
            m123 = t <= PHASE3_END;
            lvl(m123) = PHASE1_MULT;
            m4 = t > PHASE3_END;
            lvl(m4) = BOUNDARY_RECOVER_MULT;
            raw = nom .* lvl;

        else
            raw = nom + BG_SINE_FRAC * nom * sin(2*pi*t/BG_SINE_PERIOD + 0.7*k);
        end

        raw = raw + NOISE_FRAC * nom * randn(n_steps, 1);

        p = max(pmin, min(pmax, raw));
        load_hist(:, k) = p;
        assignin('base', sprintf('Pload_%s', BUS{k}), p(:).');
    end

    meta = struct('target_idx', target_idx, 'boundary_idx', boundary_idx, ...
                  'background_idx', background_idx, 'jitter', jitter, ...
                  'phase_bounds', [PHASE1_END, PHASE2_END, PHASE3_END]);
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
