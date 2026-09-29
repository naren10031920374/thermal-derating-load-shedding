function stage1_generate_and_validate()
% ==========================================================================
%  STAGE1_GENERATE_AND_VALIDATE.M
%  --------------------------------------------------------------------
%  Stage 1 of the multi-scenario pipeline: generates the raw dataset for
%  several deliberately diverse collapse scenarios, then HARD-VALIDATES
%  each one for NaN/Inf before it is trusted for anything downstream.
%
%  This is Stage 1 of a multi-stage pipeline, not a single monolithic
%  script covering dataset generation through closed-loop testing.
%  Simulink automation and Python model training are different runtimes
%  and cannot be one file; splitting into logged, independently-checkable
%  stages is the honest way to build this, matching every other script
%  in this project. See the accompanying explanation for what stages
%  2 onward require and why feature engineering (Stage 3) needs either
%  your real step33 source or an explicit go-ahead to reconstruct it
%  from the verified naming convention alone.
%  scenarios in one batch, by varying LOAD_SEED and GRID_LOAD_SCALE.
%  This is the "create dataset" step for multiple scenarios, run
%  sequentially (each is a full 5000s Simulink sim, ~220s wall clock),
%  but each scenario's output is fully independent, so downstream
%  feature-engineering / open-loop / closed-loop work on different
%  scenarios can proceed in parallel as separate jobs once their raw
%  CSVs exist.
%
%  SCENARIO DESIGN, deliberately not just repeated easy cases:
%    baseline_done   seed=1 scale=1.00  (already validated, not rerun here)
%    moderate_A      seed=2 scale=1.00  different noise, same stress
%    moderate_B      seed=3 scale=1.00  different noise, same stress
%    no_collapse_ctrl seed=1 scale=0.90 lighter loading, likely never
%                     collapses, a false-positive control for the detector
%    aggressive      seed=1 scale=1.10  faster/harder cascade
%    hardest         seed=4 scale=1.15  new noise + heaviest stress,
%                     likely multi-bus or faster onset
%    threshold_probe seed=5 scale=0.95  new noise, probes the untested
%                     gap between no_collapse_ctrl (0.90) and the
%                     baseline stress level (1.00)
%
%  WHAT THIS SCRIPT DOES NOT DO:
%  It does not fix the GEI divide-by-zero issue (that was fix_GEI_v7_ALfix.py
%  in the original pipeline, never available to build this script from).
%  It attempts to tap Bus_<bus>_GEI and Bus_<bus>_Commanded_Load directly
%  from the model in addition to the previously-verified signals, since
%  wire_one_tap silently and safely skips any tag that does not exist,
%  printing a [skip] notice rather than erroring. Check the printed
%  output for [skip] lines on GEI/Commanded_Load, if present, that gap
%  still needs to be closed with the same GEI-fix step used before,
%  applied per scenario, before feature engineering (step33) can run.
%  It does not run feature engineering, open-loop analysis, or the
%  detector, those are separate, independent next steps per scenario.
%
%  RUNTIME: 5 full 5000s sim() calls (excludes the already-done baseline),
%  at ~220s/run, roughly 18 minutes total.
%
%  Run:
%    generate_multi_scenario_datasets
% ==========================================================================

    clc;

    %% 0. CONFIG --------------------------------------------------------
    PROJECT_ROOT = ...
        'D:\naren\Documents\Thermal_Derating_Work_Handover\ThermalProject';
    MODEL_NAME = 'Grid_modelling_Thermal_V7_ALfix';   % the AL-fixed base model, no shedding needed here

    TS   = 0.01;
    TEND = 5000;
    N_STEPS = round(TEND / TS) + 1;
    PMIN = 20000;
    PMAX = 100000;

    BUS  = {'A','B','C','D','E','F','G','H','K','L'};
    CONV = {'AB','BC','CD','DE','EF','FG','GH','HK','KL','AL'};

    OUTPUT_DIR = fullfile(PROJECT_ROOT, 'model_outputs', 'thermal_derating_v7', 'multi_scenario');
    if exist(OUTPUT_DIR, 'dir') ~= 7; mkdir(OUTPUT_DIR); end
    addpath(PROJECT_ROOT);

    % ---- Scenario table: deliberately diverse, not just repeats ----------
    scenarios = struct('name', {}, 'seed', {}, 'scale', {});
    scenarios(end+1) = struct('name', 'moderate_A',       'seed', 2, 'scale', 1.00);
    scenarios(end+1) = struct('name', 'moderate_B',       'seed', 3, 'scale', 1.00);
    scenarios(end+1) = struct('name', 'no_collapse_ctrl', 'seed', 1, 'scale', 0.90);
    scenarios(end+1) = struct('name', 'aggressive',       'seed', 1, 'scale', 1.10);
    scenarios(end+1) = struct('name', 'hardest',          'seed', 4, 'scale', 1.15);
    scenarios(end+1) = struct('name', 'threshold_probe',  'seed', 5, 'scale', 0.95);

    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('MULTI-SCENARIO DATASET GENERATION: %d scenarios\n', numel(scenarios));
    fprintf('%s\n', repmat('=', 1, 70));
    for i = 1:numel(scenarios)
        fprintf('  %d. %-18s seed=%d  scale=%.2f\n', i, scenarios(i).name, ...
                scenarios(i).seed, scenarios(i).scale);
    end

    %% 1. Locate the base model once -----------------------------------
    hh = dir(fullfile(PROJECT_ROOT, '**', [MODEL_NAME '.slx']));
    if isempty(hh)
        error('Base model not found: %s.slx', MODEL_NAME);
    end
    model_file = fullfile(hh(1).folder, hh(1).name);
    fprintf('\nModel: %s\n', model_file);

    time_vec = (0:N_STEPS-1)' * TS;

    %% 2. Run each scenario ------------------------------------------------
    for si = 1:numel(scenarios)
        sc = scenarios(si);
        fprintf('\n%s\n', repmat('-', 1, 70));
        fprintf('SCENARIO %d/%d: %s (seed=%d, scale=%.2f)\n', ...
                si, numel(scenarios), sc.name, sc.seed, sc.scale);
        fprintf('%s\n', repmat('-', 1, 70));

        if bdIsLoaded(MODEL_NAME)
            close_system(MODEL_NAME, 0);
        end
        load_system(model_file);
        set_param(MODEL_NAME, 'StopTime', num2str(TEND));

        fprintf('Building load profile ...\n');
        [load_hist, clip_hi, clip_lo] = build_derating_loads( ...
            sc.seed, N_STEPS, TS, BUS, sc.scale, PMIN, PMAX);
        if any(clip_hi > 5) || any(clip_lo > 5)
            fprintf('  NOTE: this scenario clips >5%% of samples on at least one bus,\n');
            fprintf('  (clip_hi max %.1f%%, clip_lo max %.1f%%), expected for an\n', ...
                    max(clip_hi), max(clip_lo));
            fprintf('  intentionally aggressive/light scenario, not necessarily a bug.\n');
        end

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
        src_pow_matrix = nan(N_STEPS, numel(BUS));   % needed to compute GEI below
        for b = 1:numel(BUS)
            T.(sprintf('V_Bus_%s', BUS{b})) = grab_series(so, sprintf('V_Bus_%s', BUS{b}), TS, N_STEPS);
            src_pow_matrix(:, b) = grab_series(so, sprintf('Bus_%s_Src_Pow', BUS{b}), TS, N_STEPS);
            T.(sprintf('Bus_%s_Src_Pow', BUS{b})) = src_pow_matrix(:, b);
            T.(sprintf('Bus_%s_Temp', BUS{b})) = grab_series(so, sprintf('Bus_%s_Temp', BUS{b}), TS, N_STEPS);
        end

        % CommandedLoad_kW_<bus>: NOT tapped from Simulink at all. This is
        % the load profile WE generated and fed into the model via
        % build_derating_loads (assignin('base','Pload_<bus>',...)),
        % already sitting in load_hist, in Watts. step33 expects kW, hence
        % the /1000. Confirmed there is no Goto tag for this in the model
        % (find_phase_signal_source.m-style search would find none), so
        % there was never anything to tap, only something to write down.
        for b = 1:numel(BUS)
            T.(sprintf('CommandedLoad_kW_%s', BUS{b})) = load_hist(:, b) / 1000;
        end

        % GEI_<bus>: NOT a native model signal (confirmed absent as a Goto
        % tag). This is now the CONFIRMED exact logic from the real
        % fix_GEI_v7_ALfix.py (provided directly, no longer a
        % reconstruction). Src_Pow_Avg = mean power across all 10 buses.
        % Every bus's source power is floored at zero by the model itself
        % (a MinMax block against Constant(0) inside source_bus_model_<bus>),
        % so Src_Pow_Avg can only be near zero if ALL TEN buses are
        % simultaneously near zero, i.e. the whole grid has genuinely
        % collapsed, not one bus behaving oddly relative to the others.
        % When that happens, GEI is set to exactly 0.0 for every bus, NOT
        % 1.0: a GEI of 1.0 would falsely claim "operating at grid average"
        % during a total blackout, 0.0 correctly says "contributing nothing".
        EPSILON_W = 500.0;   % exact value from the real fix_GEI_v7_ALfix.py
        avg_power = mean(src_pow_matrix, 2, 'omitnan');
        grid_collapsed = avg_power < EPSILON_W;
        for b = 1:numel(BUS)
            ratio = src_pow_matrix(:, b) ./ avg_power;
            ratio(grid_collapsed) = 0.0;
            T.(sprintf('GEI_%s', BUS{b})) = ratio;
        end
        % Kept for transparency/debugging, matching the real script's own
        % choice to retain this column, not required by step33 itself.
        T.Src_Pow_Avg_recomputed = avg_power;

        for c = 1:numel(CONV)
            T.(sprintf('Phase_%s_cmd_deg', CONV{c})) = grab_series(so, phase_var_names{c}, TS, N_STEPS);
            T.(sprintf('DAB_%s_Derate_Factor', CONV{c})) = grab_series(so, derate_var_names{c}, TS, N_STEPS) / 100e3;
            % JunctionTemp_C_<conv>, matching step33's real expected raw name.
            T.(sprintf('JunctionTemp_C_%s', CONV{c})) = grab_series(so, junction_var_names{c}, TS, N_STEPS);
        end
        % HeatSinkTemp_C_<conv>, matching step33's real expected raw name
        % (was DAB_<conv>_H_Temp column names previously, tap tag unchanged,
        % only the CSV column name is renamed here).
        hs_conv_order = {'AB','BC','CD','DE','EF','FG','GH','HK','KL','AL'};
        hs_tags = {'DAB_AB_H_Temp','DAB_BC_H_Temp','DAB_CD_H_Temp','DAB_DE_H_Temp', ...
                   'DAB_EF_H_Temp','DAB_FG_H_Temp','DAB_GH_Temp','DAB_HK_Temp', ...
                   'DAB_KL_Temp','DAB_LA_Temp'};
        for h = 1:numel(hs_tags)
            T.(sprintf('HeatSinkTemp_C_%s', hs_conv_order{h})) = grab_series(so, hs_tags{h}, TS, N_STEPS);
        end

        % ==================================================================
        % HARD VALIDATION: NaN/Inf audit before this scenario is trusted.
        % Same principle as step35's original NaN diagnostic: a few warm-up
        % rows near t=0 are expected (lag/rolling features have no history
        % yet); anything beyond that, or anything appearing mid-run, is
        % flagged as a real problem and STOPS this scenario rather than
        % writing a silently-bad CSV forward into feature engineering.
        % ==================================================================
        WARMUP_TOLERANCE_S = 1.0;   % rows before this time are allowed to be NaN
        log_path = fullfile(OUTPUT_DIR, sprintf('scenario_%s_validation_log.txt', sc.name));
        log_fid = fopen(log_path, 'w');

        log_both(log_fid, '\n%s\n', repmat('=', 1, 70));
        log_both(log_fid, 'NaN/Inf VALIDATION: scenario %s\n', sc.name);
        log_both(log_fid, '%s\n', repmat('=', 1, 70));

        data_cols = T.Properties.VariableNames(2:end);   % exclude 'time'
        bad_mask = false(height(T), 1);
        col_bad_counts = struct();
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

        % Time-distribution plot, real data, even when clean (shows a flat
        % zero line, which is itself useful confirmation, not skipped).
        fig = figure('Position', [100, 100, 1100, 350], 'Visible', 'off');
        bucket_s = 50;
        bucket_idx = floor(time_vec / bucket_s);
        pct_bad = accumarray(bucket_idx + 1, double(bad_mask), [], @mean) * 100;
        bucket_t = (0:numel(pct_bad)-1) * bucket_s;
        plot(bucket_t, pct_bad, 'LineWidth', 1.5, 'Color', [0.8 0.2 0.2]);
        xlabel('time (s)'); ylabel('% rows with NaN/Inf (any column)');
        title(sprintf('NaN/Inf distribution: scenario %s', sc.name), 'Interpreter', 'none');
        grid on;
        nan_plot_path = fullfile(OUTPUT_DIR, sprintf('scenario_%s_nan_distribution.png', sc.name));
        saveas(fig, nan_plot_path);
        close(fig);
        log_both(log_fid, '  Wrote NaN distribution plot: %s\n', nan_plot_path);

        if any_unexpected
            log_both(log_fid, '\n  RESULT: FAILED VALIDATION. Unexpected NaN/Inf found beyond warmup.\n');
            log_both(log_fid, '  This scenario''s CSV was written but is NOT safe to use downstream\n');
            log_both(log_fid, '  until this is investigated (likely a GEI-style divide-by-zero, same\n');
            log_both(log_fid, '  class of issue documented for the baseline scenario).\n');
            fclose(log_fid);
            error(['Scenario %s FAILED NaN/Inf validation. See log:\n%s\n' ...
                   'Stopping here rather than proceeding to feature engineering on bad data.'], ...
                   sc.name, log_path);
        else
            log_both(log_fid, '\n  RESULT: PASSED VALIDATION. Safe to proceed to feature engineering.\n');
        end
        fclose(log_fid);

        out_csv = fullfile(OUTPUT_DIR, sprintf('scenario_%s_5000s.csv', sc.name));
        writetable(T, out_csv);
        fprintf('Wrote: %s (%d rows x %d cols)\n', out_csv, height(T), width(T));

        % ---- Quick collapse summary, so you know what you got --------
        fprintf('  Quick per-bus collapse check (threshold 100V, 50-sample debounce):\n');
        any_collapse = false;
        for b = 1:numel(BUS)
            v = T.(sprintf('V_Bus_%s', BUS{b}));
            [c, onset, ~] = detect_collapse_summary(time_vec, v, 100.0, 50);
            if c
                fprintf('    Bus %s: COLLAPSES at t=%.1fs\n', BUS{b}, onset);
                any_collapse = true;
            end
        end
        if ~any_collapse
            fprintf('    No bus collapses in this scenario (expected for no_collapse_ctrl).\n');
        end

        close_system(MODEL_NAME, 0);
    end

    fprintf('\n%s\n', repmat('=', 1, 70));
    fprintf('DONE. %d raw scenario CSVs written to:\n%s\n', numel(scenarios), OUTPUT_DIR);
    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('Next, per scenario: apply the GEI fix (if [skip] appeared above for\n');
    fprintf('GEI columns), then run step33-style feature engineering, then an\n');
    fprintf('open-loop cliff-locate sweep, then the closed-loop test, same as the\n');
    fprintf('baseline scenario. These are independent per scenario and can be run\n');
    fprintf('as separate jobs once each CSV exists.\n');
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
    want_V    = strcat('V_Bus_',   BUS);
    want_P    = strcat('Bus_',     BUS, '_Src_Pow');
    want_T    = strcat('Bus_',     BUS, '_Temp');
    want_HS   = {'DAB_AB_H_Temp','DAB_BC_H_Temp','DAB_CD_H_Temp', ...
                 'DAB_DE_H_Temp','DAB_EF_H_Temp','DAB_FG_H_Temp', ...
                 'DAB_GH_Temp',  'DAB_HK_Temp',  'DAB_KL_Temp', ...
                 'DAB_LA_Temp'};
    % Phase_<conv>_cmd_deg, GEI_<bus>, and CommandedLoad_kW_<bus> are
    % DELIBERATELY NOT in this generic Goto-tag tap list. Confirmed via
    % find_phase_signal_source.m: this model has NO Goto tag, no
    % ToWorkspace block, and no named Outport for any of these three.
    % Phase is tapped directly at its true source line via
    % tap_phase_commands (it is an inport into DAB_Model_<conv>, computed
    % upstream). Commanded_Load is not a model output at all, it is the
    % load profile WE generate and feed in (load_hist), written directly.
    % GEI is not a model output either, it is computed here in MATLAB as
    % Power_bus / Average_Power_across_buses, matching this project's own
    % documented GEI-fix history. See the main loop for all three.

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


function var_names = tap_phase_commands(model, CONV)
    % Confirmed directly via find_phase_signal_source.m: Phase_<conv>_cmd_deg
    % has NO Goto tag, NO To Workspace block, NO named Outport anywhere in
    % this model (0 total ToWorkspace blocks exist). DAB_Model_<conv> has
    % an INPORT named phase_shift, meaning the phase command is computed
    % OUTSIDE this subsystem and fed in as a plain signal line at the top
    % level, never previously logged by any mechanism. This taps that
    % top-level line directly, at its true source, the mirror image of
    % tap_junction_temp (which taps an Outport's incoming line): here the
    % relevant line feeds an Inport, so this walks from the internal
    % Inport block's port NUMBER to the corresponding port on the outer
    % subsystem block, then to the line feeding that port at the top level.
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
        % Port number on the internal Inport block equals the port number
        % on the OUTER subsystem block, queried directly, not assumed from
        % listing order (find_system's order is not guaranteed to be port
        % order, confirmed by find_phase_signal_source.m's raw listing).
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
        % Tapped at the TOP LEVEL (model, not dab_path), since the source
        % signal lives outside DAB_Model_<conv> entirely.
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
    % Prints to both the console and the scenario's log file, so every
    % validation message is both visible live and permanently recorded.
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