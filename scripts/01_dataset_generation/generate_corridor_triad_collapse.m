function generate_corridor_triad_collapse()
% ==========================================================================
%  GENERATE_CORRIDOR_TRIAD_COLLAPSE.M
%  --------------------------------------------------------------------
%  New scenario: "corridor_triad_collapse"
%  Companion design doc: NEW_SCENARIO_DESIGN_corridor_triad_collapse.md
%
%  Unlike the 5 scenarios in stage1_generate_and_validate.m (all the same
%  load formula, only seed/scale varied, independent per-bus noise), this
%  scenario deliberately drives THREE ADJACENT BUSES (E, F, G) toward
%  collapse TOGETHER using a correlated, phased mechanism:
%
%    Phase 1 (0        - 1000s):  regional preheat on D,E,F,G,H (1.08x nominal)
%    Phase 2 (1000     - 1150s):  synchronized ramp on E,F,G -> near-ceiling
%    Phase 3 (1150     - 2800s):  sustained plateau on E,F,G near PMAX,
%                                 while D,H (the only rescue paths, via
%                                 converters DE/GH) stay pre-heated and
%                                 thus already partly thermally derated
%    Phase 4 (2800s    - end):    E,F,G step down to 1.5x nominal, D,H
%                                 recover to 1.0x -> tests whether the
%                                 corridor recovers or stays depressed
%
%  REVISION NOTE (after pilot #1): the first pilot used a 2.2x-nominal
%  plateau (52-66 kW) for E/F/G. That successfully drove DAB_DE/DAB_GH
%  derate factors down to ~0.21 (converters lost ~80% of support
%  capacity), confirming the thermal-preheat mechanism works - but
%  V_Bus_E/F/G only dipped to ~760-780V, nowhere near the 100V collapse
%  threshold. Each bus's own local source comfortably covered that demand
%  level on its own, so the bottleneck wasn't converter support, it was
%  raw demand amplitude. This revision pushes the Phase 3 plateau to an
%  ABSOLUTE near-ceiling target (95000W, close to PMAX=100000) instead of
%  a nominal-relative multiplier, to actually stress each bus's own
%  source capacity while the rescue converters are still crippled.
%
%  A NEW validation step (check_co_collapse_window) checks whether at
%  least 2 of the 3 target buses actually collapsed within a shared time
%  window, not just "did any bus collapse at some point."
%
%  PILOT_MODE below runs a short 3000s check first (recommended before
%  committing to the full ~220s-wall-clock, 5000s run) so you can look at
%  V_Bus_E/F/G and the DAB_DE/DAB_GH derate factors before trusting the
%  amplitude/timing numbers below.
% ==========================================================================

    clc;

    %% 0. CONFIG --------------------------------------------------------
    PROJECT_ROOT = 'D:\naren\Documents\Thermal_Derating_Work_Handover\ThermalProject';
    MODEL_NAME = 'Grid_modelling_Thermal_V7_ALfix';

    PILOT_MODE = false;      % <-- keep true until the pilot looks right
    if PILOT_MODE
        TEND = 3000;
        fprintf('*** PILOT_MODE = true: running only %ds. Set false for the full 5000s run. ***\n', TEND);
    else
        TEND = 5000;
    end

    TS      = 0.01;
    N_STEPS = round(TEND / TS) + 1;
    PMIN    = 20000;
    PMAX    = 100000;

    BUS  = {'A','B','C','D','E','F','G','H','K','L'};
    CONV = {'AB','BC','CD','DE','EF','FG','GH','HK','KL','AL'};

    SCENARIO_NAME = 'corridor_triad_collapse';
    SEED = 7;

    TARGET_BUSES        = {'E','F','G'};
    COLLAPSE_V_THRESHOLD = 100.0;
    COLLAPSE_MIN_RUN     = 50;
    CO_COLLAPSE_WINDOW_S = 300;

    OUTPUT_DIR = fullfile(PROJECT_ROOT, 'model_outputs', 'thermal_derating_v7', SCENARIO_NAME);
    if exist(OUTPUT_DIR, 'dir') ~= 7; mkdir(OUTPUT_DIR); end
    addpath(PROJECT_ROOT);

    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('SCENARIO: %s  (seed=%d, PILOT_MODE=%d, TEND=%d)\n', SCENARIO_NAME, SEED, PILOT_MODE, TEND);
    fprintf('Target corridor: %s   Boundary: D, H\n', strjoin(TARGET_BUSES, ','));
    fprintf('%s\n', repmat('=', 1, 70));

    %% 1. Locate the base model -----------------------------------------
    hh = dir(fullfile(PROJECT_ROOT, '**', [MODEL_NAME '.slx']));
    if isempty(hh)
        error('Base model not found: %s.slx under %s', MODEL_NAME, PROJECT_ROOT);
    end
    model_file = fullfile(hh(1).folder, hh(1).name);
    fprintf('\nModel: %s\n', model_file);

    time_vec = (0:N_STEPS-1)' * TS;

    %% 2. Run the scenario -------------------------------------------------
    if bdIsLoaded(MODEL_NAME)
        close_system(MODEL_NAME, 0);
    end
    load_system(model_file);
    set_param(MODEL_NAME, 'StopTime', num2str(TEND));

    fprintf('Building corridor_triad_collapse load profile ...\n');
    [load_hist, meta] = build_corridor_triad_loads(SEED, N_STEPS, TS, BUS, PMIN, PMAX);
    fprintf('  Target buses: %s | jitter applied: %s\n', ...
            strjoin(BUS(meta.target_idx), ','), ...
            mat2str(round(meta.jitter(meta.target_idx), 4)));

    fprintf('Wiring signal taps ...\n');
    wire_all_taps(MODEL_NAME, BUS, CONV);
    derate_var_names   = tap_derate_factors(MODEL_NAME, CONV);
    junction_var_names = tap_junction_temp(MODEL_NAME, CONV);
    phase_var_names    = tap_phase_commands(MODEL_NAME, CONV);

    tic;
    ws = warning('off', 'all');
    so = sim(MODEL_NAME, 'ReturnWorkspaceOutputs', 'on');
    warning(ws);
    fprintf('Finished in %.1f s wall clock.\n', toc);

    % ---- Assemble the raw CSV: time + all wired signals -------------
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

    % ==================================================================
    % HARD VALIDATION #1: NaN/Inf audit.
    % ==================================================================
    WARMUP_TOLERANCE_S = 1.0;
    log_path = fullfile(OUTPUT_DIR, sprintf('scenario_%s_validation_log.txt', SCENARIO_NAME));
    log_fid = fopen(log_path, 'w');

    log_both(log_fid, '\n%s\n', repmat('=', 1, 70));
    log_both(log_fid, 'NaN/Inf VALIDATION: scenario %s\n', SCENARIO_NAME);
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
            log_both(log_fid, '  %-30s %6d bad samples total, %6d after warmup tolerance (t>%.1fs)\n', ...
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

    fig = figure('Position', [100, 100, 1100, 350], 'Visible', 'off');
    bucket_s = 50;
    bucket_idx = floor(time_vec / bucket_s);
    pct_bad = accumarray(bucket_idx + 1, double(bad_mask), [], @mean) * 100;
    bucket_t = (0:numel(pct_bad)-1) * bucket_s;
    plot(bucket_t, pct_bad, 'LineWidth', 1.5, 'Color', [0.8 0.2 0.2]);
    xlabel('time (s)'); ylabel('% rows with NaN/Inf (any column)');
    title(sprintf('NaN/Inf distribution: %s', SCENARIO_NAME), 'Interpreter', 'none');
    grid on;
    nan_plot_path = fullfile(OUTPUT_DIR, sprintf('scenario_%s_nan_distribution.png', SCENARIO_NAME));
    saveas(fig, nan_plot_path);
    close(fig);
    log_both(log_fid, '  Wrote NaN distribution plot: %s\n', nan_plot_path);

    if any_unexpected
        log_both(log_fid, '\n  RESULT: FAILED NaN/Inf VALIDATION.\n');
        fclose(log_fid);
        error(['Scenario %s FAILED NaN/Inf validation. See log:\n%s'], SCENARIO_NAME, log_path);
    else
        log_both(log_fid, '\n  RESULT: PASSED NaN/Inf VALIDATION.\n');
    end

    % ==================================================================
    % VALIDATION #2 (NEW): did E, F, G actually collapse TOGETHER?
    % ==================================================================
    log_both(log_fid, '\n%s\n', repmat('=', 1, 70));
    log_both(log_fid, 'CO-COLLAPSE WINDOW CHECK (target buses: %s, window=%ds)\n', ...
              strjoin(TARGET_BUSES, ','), CO_COLLAPSE_WINDOW_S);
    log_both(log_fid, '%s\n', repmat('=', 1, 70));

    report = check_co_collapse_window(time_vec, T, BUS, TARGET_BUSES, ...
                                       COLLAPSE_V_THRESHOLD, COLLAPSE_MIN_RUN, CO_COLLAPSE_WINDOW_S);

    for i = 1:numel(report.bus_names)
        if report.collapsed(i)
            log_both(log_fid, '  %s: COLLAPSED at t=%.1fs\n', report.bus_names{i}, report.onset_times(i));
        else
            log_both(log_fid, '  %s: did not collapse\n', report.bus_names{i});
        end
    end
    log_both(log_fid, '  Buses collapsed: %d / %d\n', report.n_collapsed, numel(TARGET_BUSES));
    if report.co_collapse_achieved
        log_both(log_fid, '  RESULT: REGIONAL CO-COLLAPSE ACHIEVED. Pairs within window:\n');
        for i = 1:numel(report.pairs_within_window)
            log_both(log_fid, '    - %s\n', report.pairs_within_window{i});
        end
    else
        log_both(log_fid, '  RESULT: regional co-collapse NOT achieved with current parameters.\n');
        log_both(log_fid, '  (Fewer than 2 target buses collapsed within %ds of each other.)\n', CO_COLLAPSE_WINDOW_S);
    end
    fclose(log_fid);

    out_csv = fullfile(OUTPUT_DIR, sprintf('scenario_%s_%ds.csv', SCENARIO_NAME, TEND));
    writetable(T, out_csv);
    fprintf('Wrote: %s (%d rows x %d cols)\n', out_csv, height(T), width(T));

    close_system(MODEL_NAME, 0);

    fprintf('\n%s\n', repmat('=', 1, 70));
    fprintf('DONE. co_collapse_achieved = %d\n', report.co_collapse_achieved);
    if PILOT_MODE
        fprintf('This was a PILOT run (%ds). If Bus_E/F/G_Temp and the DAB_DE/DAB_GH\n', TEND);
        fprintf('derate factors are trending as expected, set PILOT_MODE=false and rerun\n');
        fprintf('for the full 5000s scenario.\n');
    end
    fprintf('%s\n', repmat('=', 1, 70));
end


%% ==================== scenario-specific functions ====================

function [load_hist, meta] = build_corridor_triad_loads(seed, n_steps, ts, BUS, pmin, pmax)
% Implements the 4-phase correlated corridor mechanism from
% NEW_SCENARIO_DESIGN_corridor_triad_collapse.md.
%
% REVISED after pilot #1: Phase 3 now targets an ABSOLUTE near-ceiling
% demand (PHASE3_TARGET_W, close to pmax) for E/F/G instead of a
% nominal-relative multiplier. Pilot #1's 2.2x-nominal plateau (52-66 kW)
% crippled the DE/GH converters (derate factor dropped to ~0.21) but each
% bus's own local source still easily covered that demand, so voltage
% barely moved (min ~760-780V vs. the 100V threshold). Pushing to ~95 kW
% - near the model's PMAX=100000 clip - tests whether that's enough to
% actually overwhelm local generation while rescue support is crippled.

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

    PHASE1_MULT     = 1.08;   % preheat level, D/E/F/G/H (unchanged - already worked)
    PHASE3_TARGET_W = 95000;  % NEW: absolute near-ceiling target for E/F/G
    PHASE4_MULT     = 1.50;   % step-down level, E/F/G (unchanged)
    BOUNDARY_RECOVER_MULT = 1.00;

    NOISE_FRAC   = 0.02;
    BG_SINE_FRAC = 0.04;
    BG_SINE_PERIOD = 900;

    load_hist = nan(n_steps, NB);

    jitter = ones(1, NB);
    for k = target_idx
        jitter(k) = 1 + 0.03 * (2*rand() - 1);  % +/- 3%
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


function report = check_co_collapse_window(time_vec, T, BUS, target_bus_names, threshold, min_run, window_s) %#ok<INUSD>
    n = numel(target_bus_names);
    onsets = nan(1, n);
    collapsed_flags = false(1, n);

    for i = 1:n
        v = T.(sprintf('V_Bus_%s', target_bus_names{i}));
        [c, onset, ~] = detect_collapse_summary(time_vec, v, threshold, min_run);
        collapsed_flags(i) = c;
        if c
            onsets(i) = onset;
        end
    end

    n_collapsed = sum(collapsed_flags);
    report = struct('bus_names', {target_bus_names}, 'collapsed', collapsed_flags, ...
                     'onset_times', onsets, 'n_collapsed', n_collapsed, ...
                     'co_collapse_achieved', false, 'window_s', window_s, ...
                     'pairs_within_window', {{}});

    if n_collapsed < 2
        return;
    end

    valid_onsets = onsets(collapsed_flags);
    valid_names  = target_bus_names(collapsed_flags);
    pairs = {};
    for i = 1:numel(valid_onsets)
        for j = i+1:numel(valid_onsets)
            if abs(valid_onsets(i) - valid_onsets(j)) <= window_s
                pairs{end+1} = sprintf('%s (t=%.1fs) & %s (t=%.1fs)', ...
                    valid_names{i}, valid_onsets(i), valid_names{j}, valid_onsets(j)); %#ok<AGROW>
            end
        end
    end
    report.pairs_within_window = pairs;
    report.co_collapse_achieved = ~isempty(pairs);
end


%% ==================== SHARED MODEL-INTERFACE INFRASTRUCTURE ====================
%  Unchanged - generic plumbing to talk to the same Simulink model.

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
                ['Expected exactly one Outport block named ''Temp_Junction'' under ' ...
                 'DAB_Model_%s, found %d.'], CONV{i}, numel(outport_blk));
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