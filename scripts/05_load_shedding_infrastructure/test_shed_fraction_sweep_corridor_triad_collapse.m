function test_shed_fraction_sweep_corridor_triad_collapse()
% ==========================================================================
%  TEST_SHED_FRACTION_SWEEP_CORRIDOR_TRIAD_COLLAPSE.M
%  --------------------------------------------------------------------
%  Load-shedding infrastructure stage, adapted for corridor_triad_collapse.
%  This is the FIRST question that has to be answered before any lead-time
%  test makes sense: is there a shed fraction that actually prevents
%  collapse in THIS scenario at all? The baseline project's "known-safe
%  fraction" of 0.30 (used throughout its test_shed_lead_time.m) was
%  discovered specifically for the baseline's own physics via its own
%  sweep (test_shed_multi_bus_sweep_v2.m) -- it is not assumed to transfer
%  here, since this scenario drives Bus E and Bus F to collapse together
%  via a completely different mechanism (regional preheat + a
%  near-ceiling absolute demand target, not the baseline's per-bus
%  independent noise formula).
%
%  WHAT THIS DOES: sheds Bus E and Bus F TOGETHER (the two buses that
%  actually collapse in this scenario; Bus G never collapses on its own,
%  so it is only watched, not shed) at a full sweep of fractions from
%  1.0 (no shedding, the known-collapsing baseline) down to 0.0 (fully
%  shed), in steps of 0.1 -- 11 runs total, per your request to sweep the
%  whole 0-to-1 range rather than a handful of hand-picked points.
%
%  SAME STATIC-SHEDDING LIMITATION AS THE ORIGINAL PROJECT'S SCRIPTS: the
%  fraction is applied for the WHOLE 5000s run, not just a warning window.
%  This answers "does less load on E+F ever avoid collapse," not "can a
%  short, detector-triggered shed avoid it." That second, more realistic
%  question is exactly what a lead-time test (the corridor_triad_collapse
%  analog of test_shed_lead_time.m) would answer next, once we know from
%  THIS script whether a safe fraction even exists and roughly where it
%  sits.
%
%  REQUIRES: Grid_modelling_Thermal_V7_ALfix_sheddable.slx already built
%  (it was, for the baseline -- add_controllable_load_shedding.m spliced a
%  generic ShedFrac_<bus> actuator onto every bus's load path, downstream
%  of whatever script populates the Pload_<bus> workspace variables, so
%  the same sheddable model works unchanged for this scenario's load
%  profile too, no need to rebuild it).
%
%  Load profile: build_corridor_triad_loads, copied verbatim from
%  generate_corridor_triad_collapse.m (same SEED=7, same 4-phase
%  mechanism), so every run here reproduces the exact scenario already
%  validated, just with Bus E/F's load scaled by the shed fraction.
%
%  RUNTIME: 11 full 5000s sim() calls. generate_corridor_triad_collapse.m
%  logged ~220s wall-clock for one full run, so expect roughly 40-50
%  minutes total for the whole sweep.
%
%  Run:
%    test_shed_fraction_sweep_corridor_triad_collapse
% ==========================================================================

    clc;

    %% 0. CONFIG --------------------------------------------------------
    PROJECT_ROOT = 'D:\naren\Documents\Thermal_Derating_Work_Handover\ThermalProject';
    MODEL_NAME = 'Grid_modelling_Thermal_V7_ALfix_sheddable';

    SHED_BUSES = {'E', 'F'};          % the two buses that actually collapse together here
    FRACTIONS  = 0:0.1:1.0;           % full 0-to-1 sweep, per request, 11 points

    TS   = 0.01;
    TEND = 5000;
    N_STEPS = round(TEND / TS) + 1;
    SEED = 7;                          % must match generate_corridor_triad_collapse.m exactly
    PMIN = 20000;
    PMAX = 100000;

    COLLAPSE_VOLTAGE_V = 100.0;
    MIN_CONSECUTIVE_SAMPLES = 50;
    CO_COLLAPSE_WINDOW_S = 300;        % same definition as the scenario's own validation

    BUS  = {'A','B','C','D','E','F','G','H','K','L'};
    CONV = {'AB','BC','CD','DE','EF','FG','GH','HK','KL','AL'};

    % Watched throughout, regardless of fraction: the 3-bus corridor plus
    % its two rescue converters (DE, GH) and the two internal ones (EF, FG).
    WATCH_BUSES = {'E', 'F', 'G'};
    WATCH_CONV  = {'DE', 'EF', 'FG', 'GH'};

    OUTPUT_DIR = fullfile(PROJECT_ROOT, 'model_outputs', 'thermal_derating_v7', ...
                           'corridor_triad_collapse_shed_fraction_sweep');
    if exist(OUTPUT_DIR, 'dir') ~= 7; mkdir(OUTPUT_DIR); end
    addpath(PROJECT_ROOT);

    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('CORRIDOR_TRIAD_COLLAPSE: SHED FRACTION SWEEP, Bus E+F together\n');
    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('Fractions to test: %s\n', mat2str(FRACTIONS));
    fprintf(['REMINDER: shedding is applied STATICALLY for the whole %.0fs run.\n' ...
             'This tests "always less load on E+F," not a time-triggered response.\n'], TEND);

    %% 1. Build the load profile, identical every run ---------------------
    fprintf('\nBuilding corridor_triad_collapse load profile (seed %d) ...\n', SEED);
    [~, meta] = build_corridor_triad_loads(SEED, N_STEPS, TS, BUS, PMIN, PMAX);
    fprintf('  Target buses: %s | jitter applied: %s\n', ...
            strjoin(BUS(meta.target_idx), ','), ...
            mat2str(round(meta.jitter(meta.target_idx), 4)));

    %% 2. Locate and load the sheddable model ------------------------------
    hh = dir(fullfile(PROJECT_ROOT, '**', [MODEL_NAME '.slx']));
    if isempty(hh)
        error(['Sheddable model not found: %s.slx\n' ...
               'It should already exist from the baseline''s ' ...
               'add_controllable_load_shedding.m run.'], MODEL_NAME);
    end
    model_file = fullfile(hh(1).folder, hh(1).name);
    fprintf('\nModel: %s\n', model_file);
    if bdIsLoaded(MODEL_NAME)
        close_system(MODEL_NAME, 0);
    end
    load_system(model_file);

    shed_blocks = cell(1, numel(SHED_BUSES));
    for i = 1:numel(SHED_BUSES)
        shed_blocks{i} = sprintf('%s/ShedFrac_%s_default', MODEL_NAME, SHED_BUSES{i});
        if getSimulinkBlockHandle(shed_blocks{i}) == -1
            error('Block not found: %s', shed_blocks{i});
        end
    end

    %% 3. Wire taps ONCE, including E's own converters (DE, EF) -----------
    fprintf('\nWiring signal taps (once, reused across all fractions) ...\n');
    wire_all_taps(MODEL_NAME, BUS, CONV);
    derate_var_names = tap_derate_factors(MODEL_NAME, CONV);

    set_param(MODEL_NAME, 'StopTime', num2str(TEND));

    %% 4. Sweep -------------------------------------------------------------
    time_vec = (0:N_STEPS-1)' * TS;
    n_runs = numel(FRACTIONS);

    V = struct();
    for b = 1:numel(WATCH_BUSES)
        V.(WATCH_BUSES{b}) = nan(N_STEPS, n_runs);
    end
    DER = struct();
    for c = 1:numel(WATCH_CONV)
        DER.(WATCH_CONV{c}) = nan(N_STEPS, n_runs);
    end

    results = struct('fraction', {}, 'collapsed', {}, 'onset', {}, 'min_v', {}, ...
                      'co_collapse_achieved', {}, 'pairs_within_window', {});

    for r = 1:n_runs
        frac = FRACTIONS(r);
        fprintf('\n%s\n', repmat('-', 1, 70));
        fprintf('RUN %d/%d: ShedFrac_E = ShedFrac_F = %.2f\n', r, n_runs, frac);
        fprintf('%s\n', repmat('-', 1, 70));

        % Reset every bus to no-shedding first, then set only E and F.
        for b = 1:numel(BUS)
            blk = sprintf('%s/ShedFrac_%s_default', MODEL_NAME, BUS{b});
            if getSimulinkBlockHandle(blk) ~= -1
                set_param(blk, 'Value', '1');
            end
        end
        for i = 1:numel(SHED_BUSES)
            set_param(shed_blocks{i}, 'Value', num2str(frac));
        end

        % Re-populate Pload_<bus> workspace vars fresh each run: sim() reads
        % the base workspace, and the RNG state must be reset to SEED every
        % time so every fraction sees the identical underlying load profile.
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
            DER.(WATCH_CONV{c})(:, r) = der_raw / 100e3;   % same scale correction as elsewhere
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

        n_collapsed = sum(collapsed);
        pairs = {};
        if n_collapsed >= 2
            valid_onsets = onsets(collapsed);
            valid_names = WATCH_BUSES(collapsed);
            for i = 1:numel(valid_onsets)
                for j = i+1:numel(valid_onsets)
                    if abs(valid_onsets(i) - valid_onsets(j)) <= CO_COLLAPSE_WINDOW_S
                        pairs{end+1} = sprintf('%s (t=%.1fs) & %s (t=%.1fs)', ...
                            valid_names{i}, valid_onsets(i), valid_names{j}, valid_onsets(j)); %#ok<AGROW>
                    end
                end
            end
        end
        co_collapse_achieved = ~isempty(pairs);
        if n_collapsed == 0
            fprintf('  Co-collapse: N/A, nothing collapsed.\n');
        elseif co_collapse_achieved
            fprintf('  Co-collapse: STILL ACHIEVED (%d bus(es) collapsed, within %ds of each other)\n', ...
                    n_collapsed, CO_COLLAPSE_WINDOW_S);
        else
            fprintf('  Co-collapse: NOT within window (only %d bus(es) collapsed, or too far apart)\n', ...
                    n_collapsed);
        end

        results(end+1) = struct('fraction', frac, 'collapsed', collapsed, 'onset', onsets, ...
                                 'min_v', min_vs, 'co_collapse_achieved', co_collapse_achieved, ...
                                 'pairs_within_window', {pairs}); %#ok<AGROW>
    end

    %% 5. Summary table -------------------------------------------------------
    fprintf('\n%s\n', repmat('=', 1, 70));
    fprintf('SUMMARY (Bus E+F shed together, static, whole run)\n');
    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('%-10s', 'fraction');
    for b = 1:numel(WATCH_BUSES)
        fprintf(' %-16s', sprintf('Bus %s', WATCH_BUSES{b}));
    end
    fprintf('\n');
    for r = 1:n_runs
        res = results(r);
        fprintf('%-10.2f', res.fraction);
        for b = 1:numel(WATCH_BUSES)
            if res.collapsed(b)
                fprintf(' %-16s', sprintf('COLLAPSE@%.0fs', res.onset(b)));
            else
                fprintf(' %-16s', 'survives');
            end
        end
        fprintf('\n');
    end

    all_safe = arrayfun(@(r) ~any(r.collapsed), results);
    if any(all_safe)
        safe_fracs = [results(all_safe).fraction];
        fprintf('\nFractions where ALL THREE watched buses survive: %s\n', mat2str(safe_fracs));
        fprintf('Highest (least aggressive) safe fraction found: %.2f\n', max(safe_fracs));
    else
        fprintf('\nNo tested fraction (down to 0.0) kept all three buses safe.\n');
        fprintf('This would mean shedding E+F alone cannot prevent this scenario''s collapse\n');
        fprintf('at all -- worth checking whether Bus G also needs shedding, or whether the\n');
        fprintf('phase-3 demand target itself would need to be revisited.\n');
    end

    %% 6. Plots -----------------------------------------------------------------
    fprintf('\nBuilding plots ...\n');
    zoom_lo = 900; zoom_hi = 2000;   % this scenario's crisis window, not the baseline's
    mask = time_vec >= zoom_lo & time_vec <= zoom_hi;

    % --- Outcome dot chart ---
    fig1 = figure('Position', [100, 100, 1200, 400], 'Visible', 'off');
    hold on;
    for r = 1:n_runs
        for b = 1:numel(WATCH_BUSES)
            color = [0.2 0.75 0.2];
            if results(r).collapsed(b); color = [0.85 0.2 0.2]; end
            plot(results(r).fraction, numel(WATCH_BUSES) - b + 1, 'o', ...
                 'MarkerSize', 16, 'MarkerFaceColor', color, 'MarkerEdgeColor', 'w');
        end
    end
    set(gca, 'YTick', 1:numel(WATCH_BUSES), ...
             'YTickLabel', fliplr(WATCH_BUSES), 'YLim', [0.5, numel(WATCH_BUSES) + 0.5]);
    xlabel('shed fraction applied to Bus E + Bus F (1.0 = no shedding)');
    title('corridor triad collapse: outcome vs shed fraction (E+F shed together, static)');
    grid on; hold off;
    saveas(fig1, fullfile(OUTPUT_DIR, 'fraction_sweep_outcome_summary.png'));
    close(fig1);

    % --- Bus voltages across fractions, zoomed to the crisis window ---
    fig2 = figure('Position', [100, 100, 1400, 700], 'Visible', 'off');
    colors = winter(n_runs);
    for b = 1:numel(WATCH_BUSES)
        subplot(numel(WATCH_BUSES), 1, b); hold on;
        bus = WATCH_BUSES{b};
        for r = 1:n_runs
            plot(time_vec(mask), V.(bus)(mask, r), 'Color', colors(r,:), 'LineWidth', 1.1, ...
                 'DisplayName', sprintf('frac=%.1f', FRACTIONS(r)));
        end
        yline(COLLAPSE_VOLTAGE_V, '--k', '100V threshold');
        ylabel(sprintf('Bus %s (V)', bus));
        if b == 1
            title('Bus voltages vs shed fraction, zoomed to the corridor triad crisis window');
        end
        if b == numel(WATCH_BUSES)
            xlabel('time (s)');
            legend('Location', 'eastoutside', 'Interpreter', 'none', 'FontSize', 7);
        end
        grid on; hold off;
    end
    saveas(fig2, fullfile(OUTPUT_DIR, 'fraction_sweep_voltages.png'));
    close(fig2);

    % --- Derate factors across fractions, zoomed ---
    fig3 = figure('Position', [100, 100, 1400, 800], 'Visible', 'off');
    for c = 1:numel(WATCH_CONV)
        subplot(numel(WATCH_CONV), 1, c); hold on;
        conv = WATCH_CONV{c};
        for r = 1:n_runs
            plot(time_vec(mask), DER.(conv)(mask, r), 'Color', colors(r,:), 'LineWidth', 1.0, ...
                 'DisplayName', sprintf('frac=%.1f', FRACTIONS(r)));
        end
        ylim([0, 1.05]);
        ylabel(conv);
        if c == 1
            title('Converter derate factors vs shed fraction, zoomed');
        end
        if c == numel(WATCH_CONV)
            xlabel('time (s)');
            legend('Location', 'eastoutside', 'Interpreter', 'none', 'FontSize', 7);
        end
        grid on; hold off;
    end
    saveas(fig3, fullfile(OUTPUT_DIR, 'fraction_sweep_derate_factors.png'));
    close(fig3);

    save(fullfile(OUTPUT_DIR, 'fraction_sweep_results.mat'), ...
         'V', 'DER', 'time_vec', 'FRACTIONS', 'SHED_BUSES', 'WATCH_BUSES', 'WATCH_CONV', ...
         'results', '-v7.3');

    fprintf('\nWrote plots and results to: %s\n', OUTPUT_DIR);
    fprintf('\nDone. Read fraction_sweep_outcome_summary.png first: the highest fraction with\n');
    fprintf('every dot green is the minimum shedding needed to prevent this scenario''s\n');
    fprintf('collapse. If a safe fraction exists, the natural next step is a lead-time test\n');
    fprintf('(the corridor_triad_collapse analog of test_shed_lead_time.m) at that fraction,\n');
    fprintf('to see how close to t~1149s (this scenario''s actual collapse onset) shedding can\n');
    fprintf('start and still work -- then compare that required lead time against what your\n');
    fprintf('early-warning detector actually provides (up to ~90s, per step44/46).\n');

    close_system(MODEL_NAME, 0);
end


function [load_hist, meta] = build_corridor_triad_loads(seed, n_steps, ts, BUS, pmin, pmax)
% VERBATIM from generate_corridor_triad_collapse.m -- same SEED, same
% 4-phase mechanism, so this sweep reproduces the exact validated scenario,
% only with Bus E/F's load scaled downstream by ShedFrac_E / ShedFrac_F.
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
