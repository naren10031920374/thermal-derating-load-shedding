function test_shed_multi_bus_sweep()
% ==========================================================================
%  TEST_SHED_MULTI_BUS_SWEEP.M
%  --------------------------------------------------------------------
%  Extends test_shed_fraction_sweep.m in two ways, run in one script so
%  the results are directly comparable (same load profile, same session):
%
%    GROUP 1, "F only": shed Bus F alone, at lower fractions than before
%      (1.0 baseline, 0.7, 0.5, 0.3, 0.1), to see whether collapse is
%      ever actually avoided rather than just delayed.
%
%    GROUP 2, "E+F together": shed Bus E and Bus F simultaneously, at the
%      same fraction, to test whether shedding only F is what caused the
%      earlier finding that E and F started collapsing almost
%      simultaneously (under 1 second apart) instead of E lagging F by
%      the baseline's natural 53 seconds.
%
%  Both groups share the same baseline (fraction=1.0 on every bus is
%  identical regardless of which buses are "in scope" for a group), so
%  it is only simulated once, not twice.
%
%  SAME STATIC-SHEDDING LIMITATION AS BEFORE: each scenario applies its
%  fraction(s) for the WHOLE 5000s run, there is still no time-triggered
%  response. This tests "what if these buses always carried less load,"
%  not "what if we shed load only during the warning window."
%
%  Before each scenario runs, EVERY bus's ShedFrac_<bus>_default is reset
%  to 1 first, then only the buses in that scenario are set to the target
%  fraction. This matters now that different scenarios touch different
%  buses, without the reset a setting from one scenario could leak into
%  the next.
%
%  RUNTIME: baseline + 7 (F-only) + 3 (E+F) = 11 full 5000s sim() calls.
%  Based on prior timings (roughly 215-270s each), expect on the order of
%  40-50 minutes total. This also regenerates the original coarse-fraction
%  results (0.70, 0.50, 0.30, 0.10), since the previous run's data was
%  never saved to disk and is no longer available in the workspace.
%
%  Run:
%    test_shed_multi_bus_sweep
% ==========================================================================

    clc;

    %% 0. CONFIG --------------------------------------------------------
    PROJECT_ROOT = 'D:\naren\Documents\Thermal_Derating_Work_Handover\ThermalProject';
    MODEL_NAME = 'Grid_modelling_Thermal_V7_ALfix_sheddable';

    % Group 1: Bus F alone. Includes the original coarse fractions plus
    % finer values between 0.50 (failed, F still collapsed) and 0.30
    % (succeeded, no collapse), to pin down the minimum safe cut.
    F_ONLY_FRACTIONS = [0.70, 0.50, 0.45, 0.40, 0.35, 0.30, 0.10];

    % Group 2: Bus E and Bus F together, same fraction applied to both.
    EF_TOGETHER_FRACTIONS = [0.70, 0.50, 0.30];

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

    % Buses/converters tracked in every run, regardless of scenario, so the
    % same three buses can be compared across both groups on one chart.
    WATCH_BUSES = {'E', 'F', 'G'};
    WATCH_CONV  = {'EF', 'FG'};

    OUTPUT_DIR = fullfile(PROJECT_ROOT, 'model_outputs', 'thermal_derating_v7', 'shed_multi_bus_sweep');
    if exist(OUTPUT_DIR, 'dir') ~= 7; mkdir(OUTPUT_DIR); end
    addpath(PROJECT_ROOT);

    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('MULTI-BUS SHED SWEEP: F-only vs E+F together\n');
    fprintf('%s\n', repmat('=', 1, 70));
    fprintf(['\nREMINDER: shedding is applied STATICALLY for the whole %.0fs run.\n' ...
             'This tests "always less load," not a time-triggered response.\n'], TEND);

    %% 1. Build scenario list -------------------------------------------
    % Each scenario: label, cell array of buses to shed, fraction applied
    % to all of them. The baseline appears once and is shared by both groups.
    scenarios = struct('label', {}, 'group', {}, 'buses', {}, 'fraction', {});

    scenarios(end+1) = struct('label', 'baseline (no shedding)', 'group', 'baseline', ...
                               'buses', {{}}, 'fraction', 1.0);
    for f = F_ONLY_FRACTIONS
        scenarios(end+1) = struct('label', sprintf('F only, frac=%.2f', f), ...
                                   'group', 'F_only', 'buses', {{'F'}}, 'fraction', f);
    end
    for f = EF_TOGETHER_FRACTIONS
        scenarios(end+1) = struct('label', sprintf('E+F together, frac=%.2f', f), ...
                                   'group', 'EF_together', 'buses', {{'E','F'}}, 'fraction', f);
    end
    n_scen = numel(scenarios);

    %% 2. Build the load profile, identical every run ---------------------
    fprintf('\nBuilding load profile (seed %d, scale %.2f) ...\n', LOAD_SEED, GRID_LOAD_SCALE);
    [~, ~, ~] = build_derating_loads(LOAD_SEED, N_STEPS, TS, BUS, GRID_LOAD_SCALE, PMIN, PMAX);

    %% 3. Locate and load the sheddable model ------------------------------
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

    for b = 1:numel(BUS)
        blk = sprintf('%s/ShedFrac_%s_default', MODEL_NAME, BUS{b});
        if getSimulinkBlockHandle(blk) == -1
            error(['Block not found: %s\n' ...
                   'This model may be missing the shedding splice for bus %s.'], blk, BUS{b});
        end
    end

    %% 4. Wire taps ONCE ------------------------------------------------
    fprintf('\nWiring signal taps (once, reused across all scenarios) ...\n');
    wire_all_taps(MODEL_NAME, BUS, CONV);
    derate_var_names   = tap_derate_factors(MODEL_NAME, CONV);
    junction_var_names = tap_junction_temp(MODEL_NAME, CONV); %#ok<NASGU>

    set_param(MODEL_NAME, 'StopTime', num2str(TEND));

    %% 5. Run every scenario ----------------------------------------------
    time_vec = (0:N_STEPS-1)' * TS;
    V = nan(N_STEPS, numel(WATCH_BUSES), n_scen);   % voltage per watched bus per scenario
    results = struct('label', {}, 'group', {}, 'fraction', {}, 'buses', {}, ...
                      'onset', {}, 'collapsed', {}, 'min_v', {});

    for s = 1:n_scen
        scen = scenarios(s);
        fprintf('\n%s\n', repmat('-', 1, 70));
        fprintf('SCENARIO %d/%d: %s\n', s, n_scen, scen.label);
        fprintf('%s\n', repmat('-', 1, 70));

        % Reset every bus to no-shedding first, then apply this scenario's buses.
        for b = 1:numel(BUS)
            blk = sprintf('%s/ShedFrac_%s_default', MODEL_NAME, BUS{b});
            set_param(blk, 'Value', '1');
        end
        for i = 1:numel(scen.buses)
            blk = sprintf('%s/ShedFrac_%s_default', MODEL_NAME, scen.buses{i});
            set_param(blk, 'Value', num2str(scen.fraction));
        end

        tic;
        ws = warning('off', 'all');
        so = sim(MODEL_NAME, 'ReturnWorkspaceOutputs', 'on');
        warning(ws);
        fprintf('Finished in %.1f s wall clock.\n', toc);

        onset_map = containers.Map();
        collapsed_map = containers.Map();
        minv_map = containers.Map();

        for wb = 1:numel(WATCH_BUSES)
            bus_name = WATCH_BUSES{wb};
            v = grab_series(so, sprintf('V_Bus_%s', bus_name), TS, N_STEPS);
            V(:, wb, s) = v;
            [c, onset_t, min_v] = detect_collapse_summary( ...
                time_vec, v, COLLAPSE_VOLTAGE_V, MIN_CONSECUTIVE_SAMPLES);
            onset_map(bus_name) = onset_t;
            collapsed_map(bus_name) = c;
            minv_map(bus_name) = min_v;

            is_shed = any(strcmp(scen.buses, bus_name));
            role = 'watched';
            if is_shed; role = 'SHED'; end
            if c
                fprintf('  Bus %s (%s): collapses, onset t=%.1fs, min V=%.2f\n', ...
                        bus_name, role, onset_t, min_v);
            else
                fprintf('  Bus %s (%s): stays healthy, min V=%.2f\n', bus_name, role, min_v);
            end
        end

        results(end+1) = struct('label', scen.label, 'group', scen.group, ...
            'fraction', scen.fraction, 'buses', {scen.buses}, ...
            'onset', onset_map, 'collapsed', collapsed_map, 'min_v', minv_map); %#ok<AGROW>
    end

    %% 6. Summary table -------------------------------------------------------
    fprintf('\n%s\n', repmat('=', 1, 70));
    fprintf('SUMMARY\n');
    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('%-28s %-10s %-10s %-10s %-10s\n', 'scenario', 'F onset', 'E onset', 'F min V', 'E min V');
    for s = 1:numel(results)
        r = results(s);
        f_onset = 'no collapse'; if r.collapsed('F'); f_onset = sprintf('%.1f', r.onset('F')); end
        e_onset = 'no collapse'; if r.collapsed('E'); e_onset = sprintf('%.1f', r.onset('E')); end
        fprintf('%-28s %-10s %-10s %-10.2f %-10.2f\n', ...
            r.label, f_onset, e_onset, r.min_v('F'), r.min_v('E'));
    end

    %% 7. Plots -----------------------------------------------------------------
    fprintf('\nBuilding plots ...\n');
    % Wider zoom than before: t=4160.8s (the latest observed collapse onset
    % in this dataset) needs real margin so nothing sits at the frame edge.
    zoom_lo = 3400; zoom_hi = 4300;
    zoom_mask = time_vec >= zoom_lo & time_vec <= zoom_hi;
    full_mask = true(size(time_vec));   % full 0-5000s, no truncation
    f_idx = find(strcmp(WATCH_BUSES, 'F'));
    e_idx = find(strcmp(WATCH_BUSES, 'E'));

    % --- Figure 1a: F-only group, Bus F voltage, ZOOMED (3400-4300s) ---
    f_only_idx = find(strcmp({results.group}, 'F_only') | strcmp({results.group}, 'baseline'));
    fig1 = figure('Position', [100, 100, 1400, 500], 'Visible', 'off');
    hold on;
    colors = lines(numel(f_only_idx));
    for i = 1:numel(f_only_idx)
        s = f_only_idx(i);
        plot(time_vec(zoom_mask), V(zoom_mask, f_idx, s), 'Color', colors(i,:), 'LineWidth', 1.3, ...
             'DisplayName', results(s).label);
    end
    yline(COLLAPSE_VOLTAGE_V, '--k', '100V threshold');
    xlabel('time (s)'); ylabel('Bus F voltage (V)');
    title(sprintf('Group 1: Bus F shed alone, zoomed t=%d-%ds', zoom_lo, zoom_hi));
    legend('Location', 'southwest', 'Interpreter', 'none'); grid on; hold off;
    saveas(fig1, fullfile(OUTPUT_DIR, 'group1_F_only_bus_F_voltage_zoomed.png'));
    close(fig1);

    % --- Figure 1b: F-only group, Bus F voltage, FULL RUN (0-5000s) ---
    fig1b = figure('Position', [100, 100, 1400, 500], 'Visible', 'off');
    hold on;
    for i = 1:numel(f_only_idx)
        s = f_only_idx(i);
        plot(time_vec(full_mask), V(full_mask, f_idx, s), 'Color', colors(i,:), 'LineWidth', 1.0, ...
             'DisplayName', results(s).label);
    end
    yline(COLLAPSE_VOLTAGE_V, '--k', '100V threshold');
    xlabel('time (s)'); ylabel('Bus F voltage (V)');
    title('Group 1: Bus F shed alone, FULL 5000s run, nothing truncated');
    legend('Location', 'southwest', 'Interpreter', 'none'); grid on; hold off;
    saveas(fig1b, fullfile(OUTPUT_DIR, 'group1_F_only_bus_F_voltage_full5000s.png'));
    close(fig1b);

    % --- Figure 2a: E+F group, Bus F voltage, ZOOMED ---
    ef_idx = find(strcmp({results.group}, 'EF_together') | strcmp({results.group}, 'baseline'));
    fig2 = figure('Position', [100, 100, 1400, 500], 'Visible', 'off');
    hold on;
    colors2 = lines(numel(ef_idx));
    for i = 1:numel(ef_idx)
        s = ef_idx(i);
        plot(time_vec(zoom_mask), V(zoom_mask, f_idx, s), 'Color', colors2(i,:), 'LineWidth', 1.3, ...
             'DisplayName', results(s).label);
    end
    yline(COLLAPSE_VOLTAGE_V, '--k', '100V threshold');
    xlabel('time (s)'); ylabel('Bus F voltage (V)');
    title(sprintf('Group 2: Bus E+F shed together, Bus F voltage, zoomed t=%d-%ds', zoom_lo, zoom_hi));
    legend('Location', 'southwest', 'Interpreter', 'none'); grid on; hold off;
    saveas(fig2, fullfile(OUTPUT_DIR, 'group2_EF_together_bus_F_voltage_zoomed.png'));
    close(fig2);

    % --- Figure 2b: E+F group, Bus F voltage, FULL RUN ---
    fig2b = figure('Position', [100, 100, 1400, 500], 'Visible', 'off');
    hold on;
    for i = 1:numel(ef_idx)
        s = ef_idx(i);
        plot(time_vec(full_mask), V(full_mask, f_idx, s), 'Color', colors2(i,:), 'LineWidth', 1.0, ...
             'DisplayName', results(s).label);
    end
    yline(COLLAPSE_VOLTAGE_V, '--k', '100V threshold');
    xlabel('time (s)'); ylabel('Bus F voltage (V)');
    title('Group 2: Bus E+F shed together, Bus F voltage, FULL 5000s run');
    legend('Location', 'southwest', 'Interpreter', 'none'); grid on; hold off;
    saveas(fig2b, fullfile(OUTPUT_DIR, 'group2_EF_together_bus_F_voltage_full5000s.png'));
    close(fig2b);

    % --- Figure 3a: E+F group, Bus E voltage, ZOOMED ---
    fig3 = figure('Position', [100, 100, 1400, 500], 'Visible', 'off');
    hold on;
    for i = 1:numel(ef_idx)
        s = ef_idx(i);
        plot(time_vec(zoom_mask), V(zoom_mask, e_idx, s), 'Color', colors2(i,:), 'LineWidth', 1.3, ...
             'DisplayName', results(s).label);
    end
    yline(COLLAPSE_VOLTAGE_V, '--k', '100V threshold');
    xlabel('time (s)'); ylabel('Bus E voltage (V)');
    title(sprintf('Group 2: Bus E+F shed together, Bus E voltage, zoomed t=%d-%ds', zoom_lo, zoom_hi));
    legend('Location', 'southwest', 'Interpreter', 'none'); grid on; hold off;
    saveas(fig3, fullfile(OUTPUT_DIR, 'group3_EF_together_bus_E_voltage_zoomed.png'));
    close(fig3);

    % --- Figure 3b: E+F group, Bus E voltage, FULL RUN ---
    fig3b = figure('Position', [100, 100, 1400, 500], 'Visible', 'off');
    hold on;
    for i = 1:numel(ef_idx)
        s = ef_idx(i);
        plot(time_vec(full_mask), V(full_mask, e_idx, s), 'Color', colors2(i,:), 'LineWidth', 1.0, ...
             'DisplayName', results(s).label);
    end
    yline(COLLAPSE_VOLTAGE_V, '--k', '100V threshold');
    xlabel('time (s)'); ylabel('Bus E voltage (V)');
    title('Group 2: Bus E+F shed together, Bus E voltage, FULL 5000s run');
    legend('Location', 'southwest', 'Interpreter', 'none'); grid on; hold off;
    saveas(fig3b, fullfile(OUTPUT_DIR, 'group3_EF_together_bus_E_voltage_full5000s.png'));
    close(fig3b);

    % --- Figure 4: onset time vs fraction, both groups on Bus F, plus the
    % E-vs-F onset GAP per scenario (the synchronization question directly) ---
    fig4 = figure('Position', [100, 100, 1300, 500], 'Visible', 'off');

    subplot(1,2,1); hold on;
    f_only_fracs = [results(f_only_idx).fraction];
    f_only_onsets = arrayfun(@(s) get_onset_or_nan(s, 'F'), results(f_only_idx));
    ef_fracs = [results(ef_idx).fraction];
    ef_onsets = arrayfun(@(s) get_onset_or_nan(s, 'F'), results(ef_idx));
    plot(f_only_fracs, f_only_onsets, 'o-', 'LineWidth', 1.5, 'DisplayName', 'F only');
    plot(ef_fracs, ef_onsets, 's-', 'LineWidth', 1.5, 'DisplayName', 'E+F together');
    xlabel('shed fraction'); ylabel('Bus F collapse onset (s)');
    title('Onset delay vs fraction'); legend('Location', 'best'); grid on;
    set(gca, 'XDir', 'reverse'); hold off;

    subplot(1,2,2); hold on;
    f_only_gap = arrayfun(@(s) get_onset_gap(s), results(f_only_idx));
    ef_gap = arrayfun(@(s) get_onset_gap(s), results(ef_idx));
    plot(f_only_fracs, f_only_gap, 'o-', 'LineWidth', 1.5, 'DisplayName', 'F only');
    plot(ef_fracs, ef_gap, 's-', 'LineWidth', 1.5, 'DisplayName', 'E+F together');
    yline(53, '--k', 'baseline gap (53s)');
    xlabel('shed fraction'); ylabel('E onset minus F onset (s)');
    title('Does the E-after-F lag survive shedding?'); legend('Location', 'best'); grid on;
    set(gca, 'XDir', 'reverse'); hold off;

    sgtitle('F-only vs E+F together: delay and cascade-timing effects');
    saveas(fig4, fullfile(OUTPUT_DIR, 'group4_onset_and_gap_comparison.png'));
    close(fig4);

    fprintf('\nWrote plots to: %s\n', OUTPUT_DIR);

    % --- Minimum safe F-only fraction, read directly off this run's results ---
    fprintf('\n--- Minimum safe fraction search (F-only group) ---\n');
    f_only_fracs_sorted = sort(f_only_fracs, 'descend');   % 1.0 down to lowest
    safe_frac = NaN;
    for ff = f_only_fracs_sorted
        idx_this = f_only_idx([results(f_only_idx).fraction] == ff);
        if isempty(idx_this); continue; end
        r = results(idx_this);
        if ~r.collapsed('F') && ~r.collapsed('E')
            safe_frac = ff;   % keep going lower to find the SMALLEST safe fraction
        end
    end
    if isnan(safe_frac)
        fprintf('No tested F-only fraction fully prevented collapse of both F and E.\n');
    else
        fprintf('Smallest tested F-only fraction with NO collapse on F or E: %.2f\n', safe_frac);
        fprintf('(i.e. the largest safe cut is %.0f%% of Bus F''s load)\n', (1-safe_frac)*100);
    end

    % --- Permanent save, so future plot/analysis work never needs a rerun ---
    mat_path = fullfile(OUTPUT_DIR, 'shed_multi_bus_sweep_results.mat');
    fprintf('\nSaving all results to: %s\n', mat_path);
    save(mat_path, 'V', 'results', 'time_vec', 'WATCH_BUSES', 'WATCH_CONV', ...
         'TS', 'N_STEPS', 'COLLAPSE_VOLTAGE_V', 'scenarios', '-v7.3');
    fprintf('This file now contains everything needed to regenerate plots or rerun\n');
    fprintf('the collapse/onset analysis WITHOUT simulating again. Use\n');
    fprintf('regenerate_plots_from_saved.m going forward.\n');

    fprintf('\nDone. Read group1_F_only_bus_F_voltage_zoomed.png and the minimum-safe-\n');
    fprintf('fraction line above to see where between 0.30 and 0.50 the actual safety\n');
    fprintf('threshold sits.\n');

    close_system(MODEL_NAME, 0);
end


function v = get_onset_or_nan(result, bus_name)
    if result.collapsed(bus_name)
        v = result.onset(bus_name);
    else
        v = NaN;
    end
end


function gap = get_onset_gap(result)
    if result.collapsed('E') && result.collapsed('F')
        gap = result.onset('E') - result.onset('F');
    else
        gap = NaN;
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
