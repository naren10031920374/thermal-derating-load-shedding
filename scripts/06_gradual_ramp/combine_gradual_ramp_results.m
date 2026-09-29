function combine_gradual_ramp_results()
% ==========================================================================
%  COMBINE_GRADUAL_RAMP_RESULTS.M
%  --------------------------------------------------------------------
%  Companion to test_shed_gradual_ramp_corridor_triad_collapse.m's HPC
%  array-job mode. Run this ONCE, after all 8 SLURM array tasks (duration
%  indices 1-8) have finished and each has written its
%  gradual_ramp_partial_D<duration>.mat file. Reassembles them in the same
%  order as RAMP_DURATIONS and reproduces the exact same summary table,
%  3 plots, and final .mat file that running the sweep serially would have
%  produced -- this script never touches Simulink and runs in seconds.
%
%  If any partial file is missing, this errors out naming which duration
%  is missing rather than silently producing an incomplete summary.
%
%  Run:
%    combine_gradual_ramp_results
% ==========================================================================

    clc;

    %% 0. CONFIG -- MUST match test_shed_gradual_ramp_corridor_triad_collapse.m
    PROJECT_ROOT = fileparts(fileparts(fileparts(mfilename('fullpath'))));

    FIXED_FRACTION = 0.80;
    NATURAL_COLLAPSE_T = 1149.1;
    DETECTOR_LEAD_TIME = 30;
    FIXED_START_TIME = NATURAL_COLLAPSE_T - DETECTOR_LEAD_TIME;  % = 1119.1s
    RAMP_DURATIONS = [1, 5, 10, 15, 20, 25, 30, 40];

    TS   = 0.01;
    TEND = 5000;
    N_STEPS = round(TEND / TS) + 1;

    WATCH_BUSES = {'E', 'F', 'G'};
    WATCH_CONV  = {'DE', 'EF', 'FG', 'GH'};

    OUTPUT_DIR = fullfile(PROJECT_ROOT, 'model_outputs', 'thermal_derating_v7', ...
                           'corridor_triad_collapse_gradual_ramp_sweep');

    n_runs = numel(RAMP_DURATIONS);

    %% 1. Load every partial file, in RAMP_DURATIONS order -------------------
    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('COMBINING %d PARTIAL RESULTS from: %s\n', n_runs, OUTPUT_DIR);
    fprintf('%s\n', repmat('=', 1, 70));

    time_vec = (0:N_STEPS-1)' * TS;

    V = struct();
    for b = 1:numel(WATCH_BUSES)
        V.(WATCH_BUSES{b}) = nan(N_STEPS, n_runs);
    end
    DER = struct();
    for c = 1:numel(WATCH_CONV)
        DER.(WATCH_CONV{c}) = nan(N_STEPS, n_runs);
    end
    results = struct('ramp_duration', {}, 'collapsed', {}, 'onset', {}, 'min_v', {});

    for r = 1:n_runs
        dur = RAMP_DURATIONS(r);
        partial_file = fullfile(OUTPUT_DIR, sprintf('gradual_ramp_partial_D%02d.mat', dur));
        if exist(partial_file, 'file') ~= 2
            error(['Missing partial result for duration=%.0fs: %s\n' ...
                   'Check that the array task for duration_idx=%d actually ' ...
                   'completed before running this.'], dur, partial_file, r);
        end
        s = load(partial_file, 'Vr', 'DERr', 'res_r', 'time_vec', 'ramp_dur');
        if abs(s.ramp_dur - dur) > 1e-9
            error('Partial file %s is for duration=%.0fs, expected %.0fs -- filename/content mismatch.', ...
                  partial_file, s.ramp_dur, dur);
        end
        for b = 1:numel(WATCH_BUSES)
            V.(WATCH_BUSES{b})(:, r) = s.Vr.(WATCH_BUSES{b});
        end
        for c = 1:numel(WATCH_CONV)
            DER.(WATCH_CONV{c})(:, r) = s.DERr.(WATCH_CONV{c});
        end
        results(end+1) = s.res_r; %#ok<AGROW>
        fprintf('  Loaded duration=%.0fs from %s\n', dur, partial_file);
    end

    %% 2. Summary table -- identical to the serial script's section 4 --------
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

    %% 3. Plots -- identical to the serial script's section 5 ----------------
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
        yline(100.0, '--k', '100V threshold');
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
         'DETECTOR_LEAD_TIME', 'NATURAL_COLLAPSE_T', 'WATCH_BUSES', 'WATCH_CONV', 'results', '-v7.3');

    fprintf('\nWrote plots and combined results to: %s\n', OUTPUT_DIR);
    fprintf('\nDone. Read gradual_ramp_outcome_summary.png first: the longest ramp duration\n');
    fprintf('with both Bus E and Bus F dots green is the answer to "how slow can the real\n');
    fprintf('actuator be and still work, given the detector''s exact 30s of warning".\n');

end
