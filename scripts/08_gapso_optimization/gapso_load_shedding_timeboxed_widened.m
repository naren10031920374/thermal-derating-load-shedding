function gapso_load_shedding_timeboxed_widened(SCENARIO_NAME_ARG, BUDGET_HOURS_ARG)
% ============================================================================
%  GAPSO_LOAD_SHEDDING_TIMEBOXED_WIDENED.M
%  --------------------------------------------------------------------------
%  Identical to gapso_load_shedding_timeboxed.m in every way EXCEPT ONE
%  constant: EVAL_POST_S, the truncated-evaluation window each candidate is
%  simulated for DURING the search.
%
%    Old value: EVAL_POST_S = 120   (shed time + 120s)
%    New value: EVAL_POST_S = 400   (shed time + 400s)
%
%  WHY: the first time-boxed run (Bus C, three-bus) both landed on 30% shed
%  -- the cheapest allowed answer -- and both FAILED the full 5000s
%  confirmation run. Bus C's confirmation showed it actually collapses at
%  2745s; the three-bus scenario's G/H/K collapse at 1260s. Both of those
%  are well past the old 120s window, so the search genuinely could not see
%  the delayed collapse coming -- 30% looked safe in that short window, so
%  the search (correctly, given what it could see) treated it as a cheap
%  win and never explored higher shed percentages.
%
%  Widening the window to 400s costs more time per candidate (each
%  evaluation now simulates ~3.3x longer), so this version will run FEWER
%  candidates in the same wall-clock budget -- that trade is the whole
%  point: fewer candidates, but each one is actually evaluated honestly.
%  The script's own timing test (Section 6) measures the REAL per-candidate
%  cost at this new window size and re-splits the time budget accordingly,
%  same as before -- nothing else about that logic needed to change.
%
%  Everything else (objective function, DAB feasibility check, decision
%  bounds, PSO->GA algorithm, confirmation run, output format) is verbatim
%  from gapso_load_shedding_timeboxed.m.
%
%  OUTPUT LOCATION: results go to a NEW folder --
%    model_outputs/gapso_widened_2026_09_26/<scenario>/
%  -- so this run's results never overwrite the original 120s-window run
%  (model_outputs/gapso_reset_2026_09_22/<scenario>/), and you can compare
%  the two side by side afterward.
%
%  USAGE (called from a wrapper .sh):
%    gapso_load_shedding_timeboxed_widened('bus_f', 3)
%    gapso_load_shedding_timeboxed_widened('bus_c', 3)
%    gapso_load_shedding_timeboxed_widened('three_bus', 3)
%  Defaults if arguments omitted: SCENARIO_NAME='bus_c', BUDGET_HOURS=3.
% ============================================================================

    clc;

    %% 0. RESOLVE ARGUMENTS ---------------------------------------------------
    if nargin < 1 || isempty(SCENARIO_NAME_ARG); SCENARIO_NAME_ARG = 'bus_c'; end
    if nargin < 2 || isempty(BUDGET_HOURS_ARG);   BUDGET_HOURS_ARG = 3;       end

    SCENARIO_NAME = SCENARIO_NAME_ARG;   % 'bus_f' | 'bus_c' | 'three_bus'
    TOTAL_BUDGET_S    = BUDGET_HOURS_ARG * 3600;
    RESERVE_CONFIRM_S = 1800;   % 30 min flat guess for the final confirmation run

    run_start_tic = tic;

    %% 1. GLOBAL / SHARED CONFIG ----------------------------------------------
    PROJECT_ROOT = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    MODEL_NAME   = 'Grid_modelling_Thermal_V7_ALfix_sheddable_ramped';

    TS   = 0.01;
    TEND_FULL = 5000;
    N_STEPS_FULL = round(TEND_FULL / TS) + 1;

    COLLAPSE_VOLTAGE_V      = 100.0;
    MIN_CONSECUTIVE_SAMPLES = 50;
    HEALTHY_VOLTAGE_V       = 780.0;

    BUS  = {'A','B','C','D','E','F','G','H','K','L'};
    CONV = {'AB','BC','CD','DE','EF','FG','GH','HK','KL','AL'};

    DAB.n  = 24/15;
    DAB.fs = 10e3;
    DAB.Lk = 340e-6;
    DAB.K  = DAB.n*pi / (8*DAB.fs*DAB.Lk);

    SHED_PCT_MIN = 30;
    SHED_PCT_MAX = 60;
    OBJ_WEIGHT_POWER   = 0.70;
    OBJ_WEIGHT_VOLTAGE = 0.30;
    EVAL_POST_S = 400;   % <<< WIDENED from 120s -- the one real change in this file

    OUTPUT_DIR = fullfile(PROJECT_ROOT, 'model_outputs', 'gapso_widened_2026_09_26', SCENARIO_NAME);
    if exist(OUTPUT_DIR, 'dir') ~= 7; mkdir(OUTPUT_DIR); end
    addpath(PROJECT_ROOT);

    %% 2. PER-SCENARIO CONFIG --------------------------------------------------
    cfg = build_scenario_config(SCENARIO_NAME);

    fprintf('%s\n', repmat('=', 1, 78));
    fprintf('GA-PSO LOAD-SHEDDING SEARCH (TIME-BOXED, WIDENED WINDOW) -- scenario: %s\n', SCENARIO_NAME);
    fprintf('%s\n', repmat('=', 1, 78));
    fprintf('Target buses:          %s\n', strjoin(cfg.targetBuses, ', '));
    fprintf('Trigger time(s):       %s\n', mat2str(cfg.triggerTimes, 6));
    fprintf('Shed applied at:       %.2fs (%s)\n', cfg.shedTime, cfg.shedTimeNote);
    fprintf('Search range per bus:  %d%%-%d%% shed\n', SHED_PCT_MIN, SHED_PCT_MAX);
    fprintf('Objective weights:     %.0f%% power-shed, %.0f%% voltage-improvement\n', ...
            OBJ_WEIGHT_POWER*100, OBJ_WEIGHT_VOLTAGE*100);
    fprintf('Truncated eval window: shed time + %.0fs (WIDENED from 120s)\n', EVAL_POST_S);
    fprintf('TOTAL TIME BUDGET:     %.1f hours (%.0f seconds)\n', BUDGET_HOURS_ARG, TOTAL_BUDGET_S);
    fprintf('Confirmation reserve:  %.0f seconds (%.1f min)\n', RESERVE_CONFIRM_S, RESERVE_CONFIRM_S/60);

    nvars = numel(cfg.targetBuses);
    lb = SHED_PCT_MIN * ones(1, nvars);
    ub = SHED_PCT_MAX * ones(1, nvars);

    %% 3. SHARED CONTEXT --------------------------------------------------------
    shared.PROJECT_ROOT             = PROJECT_ROOT;
    shared.MODEL_NAME                = MODEL_NAME;
    shared.TS                        = TS;
    shared.TEND_FULL                 = TEND_FULL;
    shared.N_STEPS_FULL               = N_STEPS_FULL;
    shared.BUS                       = BUS;
    shared.CONV                      = CONV;
    shared.COLLAPSE_VOLTAGE_V         = COLLAPSE_VOLTAGE_V;
    shared.MIN_CONSECUTIVE_SAMPLES     = MIN_CONSECUTIVE_SAMPLES;
    shared.HEALTHY_VOLTAGE_V          = HEALTHY_VOLTAGE_V;
    shared.DAB                       = DAB;
    shared.EVAL_POST_S                = EVAL_POST_S;
    shared.OBJ_WEIGHT_POWER           = OBJ_WEIGHT_POWER;
    shared.OBJ_WEIGHT_VOLTAGE         = OBJ_WEIGHT_VOLTAGE;

    hh = dir(fullfile(PROJECT_ROOT, '**', [MODEL_NAME '.slx']));
    if isempty(hh)
        error('Model not found: %s.slx under %s', MODEL_NAME, PROJECT_ROOT);
    end
    shared.model_file = fullfile(hh(1).folder, hh(1).name);
    fprintf('\nModel: %s\n', shared.model_file);

    %% 4. BUILD THE FULL LOAD PROFILE ONCE --------------------------------------
    fprintf('\nBuilding full %.0fs load profile (seed %d) ...\n', TEND_FULL, cfg.loadSeed);
    cfg.loadBuilderFn(cfg.loadSeed, N_STEPS_FULL, TS, BUS, cfg.pmin, cfg.pmax, cfg.loadBuilderExtraArgs{:});

    %% 5. UNSHED LOAD AT TRIGGER TIME --------------------------------------------
    cfg.unshedLoadW = getUnshedLoadAtTrigger(cfg, shared);
    fprintf('\nUnshed commanded load at shed time, per target bus (sanity-check these):\n');
    for i = 1:nvars
        fprintf('  Bus %s: %.0f W  (scenario pmin=%.0f, pmax=%.0f)\n', ...
                cfg.targetBuses{i}, cfg.unshedLoadW(i), cfg.pmin, cfg.pmax);
    end

    %% 6. SINGLE TIMING-TEST CANDIDATE (not three -- saves time) ----------------
    fprintf('\n%s\n', repmat('-', 1, 78));
    fprintf('SINGLE TIMING-TEST CANDIDATE (measuring real per-candidate cost) ...\n');
    fprintf('%s\n', repmat('-', 1, 78));
    test_candidate = mean([SHED_PCT_MIN, SHED_PCT_MAX]) * ones(1, nvars);
    tic;
    [fit_test, diag_test] = gapsoObjective(test_candidate, cfg, shared);
    per_candidate_s = toc;
    fprintf('  Candidate (%s): %.1fs wall clock, all_safe=%d, fitness=%.4f\n', ...
            mat2str(test_candidate), per_candidate_s, diag_test.all_safe, fit_test);

    %% 7. COMPUTE THE ACTUAL TIME SPLIT ------------------------------------------
    elapsed_so_far_s = toc(run_start_tic);
    remaining_s = TOTAL_BUDGET_S - elapsed_so_far_s - RESERVE_CONFIRM_S;

    if remaining_s < 120
        fprintf('\nWARNING: only %.0fs left for the search after the timing test and\n', remaining_s);
        fprintf('confirmation reserve -- this scenario''s per-candidate cost (%.1fs) is\n', per_candidate_s);
        fprintf('eating most of the %.1f-hour budget on its own. Proceeding with a\n', BUDGET_HOURS_ARG);
        fprintf('minimal search (2 min per phase) rather than skipping it entirely.\n');
        remaining_s = 240;   % floor: 2 min PSO + 2 min GA, so SOMETHING real runs
    end

    pso_maxtime_s = remaining_s / 2;
    ga_maxtime_s  = remaining_s / 2;

    fprintf('\nTime budget split for this scenario:\n');
    fprintf('  Timing test used:        %.1f min\n', elapsed_so_far_s/60);
    fprintf('  Confirmation reserve:    %.1f min\n', RESERVE_CONFIRM_S/60);
    fprintf('  Remaining for search:    %.1f min\n', remaining_s/60);
    fprintf('  -> PSO MaxTime:          %.1f min\n', pso_maxtime_s/60);
    fprintf('  -> GA  MaxTime:          %.1f min\n', ga_maxtime_s/60);
    fprintf('  Estimated candidates in that time (rough, at %.1fs/candidate): ~%d\n', ...
            per_candidate_s, max(1, round(remaining_s / max(per_candidate_s, 1))));

    %% 8. PSO (broad search, time-capped) THEN GA (refinement, time-capped) ----
    % POP_SIZE here just controls how many candidates make up ONE iteration --
    % MaxTime (not MaxIterations) is what actually stops each phase, so a high
    % MaxIterations is set deliberately so it never becomes the binding limit.
    POP_SIZE = 6;

    fprintf('\n%s\n', repmat('-', 1, 78));
    fprintf('PARTICLE SWARM (global search, capped at %.1f min) ...\n', pso_maxtime_s/60);
    fprintf('%s\n', repmat('-', 1, 78));

    psoOpts = optimoptions('particleswarm', ...
        'SwarmSize', POP_SIZE, ...
        'MaxIterations', 100000, ...    % effectively unbounded -- MaxTime governs
        'MaxTime', pso_maxtime_s, ...
        'Display', 'iter', ...
        'UseParallel', false);

    psoObj = @(x) gapsoObjective(x, cfg, shared);
    [x_pso, fval_pso] = particleswarm(psoObj, nvars, lb, ub, psoOpts);
    fprintf('PSO result: %s  (fitness=%.4f)\n', mat2str(x_pso, 4), fval_pso);

    fprintf('\n%s\n', repmat('-', 1, 78));
    fprintf('GENETIC ALGORITHM (refinement, capped at %.1f min) ...\n', ga_maxtime_s/60);
    fprintf('%s\n', repmat('-', 1, 78));

    init_pop = [x_pso; repmat(x_pso, POP_SIZE - 1, 1) + 5*randn(POP_SIZE - 1, nvars)];
    init_pop = min(max(init_pop, SHED_PCT_MIN), SHED_PCT_MAX);

    gaOpts = optimoptions('ga', ...
        'PopulationSize', POP_SIZE, ...
        'MaxGenerations', 100000, ...   % effectively unbounded -- MaxTime governs
        'MaxTime', ga_maxtime_s, ...
        'InitialPopulationMatrix', init_pop, ...
        'Display', 'iter', ...
        'UseParallel', false);

    gaObj = @(x) gapsoObjective(x, cfg, shared);
    [x_best, fval_best] = ga(gaObj, nvars, [], [], [], [], lb, ub, [], gaOpts);

    x_best = round(min(max(x_best, SHED_PCT_MIN), SHED_PCT_MAX), 1);

    fprintf('\n%s\n', repmat('=', 1, 78));
    fprintf('SEARCH COMPLETE (%s)\n', SCENARIO_NAME);
    fprintf('%s\n', repmat('=', 1, 78));
    for i = 1:nvars
        fprintf('  Bus %s: %.1f%% shed\n', cfg.targetBuses{i}, x_best(i));
    end
    fprintf('  Final fitness: %.4f\n', fval_best);
    fprintf('  Total elapsed before confirmation run: %.1f min (budget was %.1f min for this point: %.1f)\n', ...
            toc(run_start_tic)/60, (TOTAL_BUDGET_S - RESERVE_CONFIRM_S)/60, TOTAL_BUDGET_S/60);

    %% 9. FULL, UNTRUNCATED CONFIRMATION RUN ------------------------------------
    fprintf('\n%s\n', repmat('-', 1, 78));
    fprintf('CONFIRMING WINNING VECTOR WITH ONE FULL %.0fs RUN, ALL 10 BUSES ...\n', TEND_FULL);
    fprintf('%s\n', repmat('-', 1, 78));

    confirm_tic = tic;
    full_result = runFullConfirmation(x_best, cfg, shared, OUTPUT_DIR);
    confirm_actual_s = toc(confirm_tic);

    fprintf('\nConfirmation run actually took %.1f min (reserve was %.1f min).\n', ...
            confirm_actual_s/60, RESERVE_CONFIRM_S/60);
    if confirm_actual_s > RESERVE_CONFIRM_S * 1.3
        fprintf('NOTE: confirmation run ran notably longer than reserved -- consider a\n');
        fprintf('bigger RESERVE_CONFIRM_S for the next scenario run with this script.\n');
    end

    if full_result.all_safe
        fprintf('\nCONFIRMED: all 10 buses survive with this shed vector.\n');
    else
        fprintf('\nNOT CONFIRMED: at least one bus still collapses in the full run.\n');
        fprintf('Even with the widened %.0fs window, this scenario''s search did not find\n', EVAL_POST_S);
        fprintf('a fully-safe shed vector within its time budget -- treat this as a\n');
        fprintf('starting point, not final, and consider a longer budget or a wider\n');
        fprintf('window still if time allows.\n');
    end

    total_elapsed_s = toc(run_start_tic);
    fprintf('\nTOTAL WALL-CLOCK TIME FOR THIS SCENARIO: %.1f min (target was %.0f min)\n', ...
            total_elapsed_s/60, BUDGET_HOURS_ARG*60);

    %% 10. SAVE ------------------------------------------------------------------
    result_mat = fullfile(OUTPUT_DIR, sprintf('gapso_result_%s.mat', SCENARIO_NAME));
    save(result_mat, 'SCENARIO_NAME', 'cfg', 'x_best', 'fval_best', 'x_pso', 'fval_pso', ...
         'per_candidate_s', 'full_result', 'SHED_PCT_MIN', 'SHED_PCT_MAX', ...
         'OBJ_WEIGHT_POWER', 'OBJ_WEIGHT_VOLTAGE', 'EVAL_POST_S', 'POP_SIZE', ...
         'BUDGET_HOURS_ARG', 'total_elapsed_s', 'confirm_actual_s', '-v7.3');
    fprintf('\nWrote: %s\n', result_mat);
    fprintf('All outputs in: %s\n', OUTPUT_DIR);

    close_system(MODEL_NAME, 0);
end


%% ============================================================================
%%  SCENARIO CONFIGURATION (verbatim from gapso_load_shedding_timeboxed.m)
%% ============================================================================

function cfg = build_scenario_config(name)
    cfg.name = name;
    switch name
        case 'bus_f'
            cfg.targetBuses  = {'F'};
            cfg.triggerTimes = 3592.59;
            cfg.onsetRef     = 3643.44;
            cfg.shedTime     = cfg.triggerTimes(1);
            cfg.shedTimeNote = 'Bus F''s own real held-out-bus detector trigger';
            cfg.loadSeed = 1;
            cfg.pmin = 20000; cfg.pmax = 100000;
            cfg.loadBuilderFn = @build_derating_loads_bus_f;
            cfg.loadBuilderExtraArgs = {1.0};

        case 'bus_c'
            cfg.targetBuses  = {'C'};
            cfg.triggerTimes = 1002.50;
            cfg.onsetRef     = 2094.47;
            cfg.shedTime     = cfg.triggerTimes(1);
            cfg.shedTimeNote = 'Bus C''s own real cross-scenario detector trigger';
            cfg.loadSeed = 104;
            cfg.pmin = 2000; cfg.pmax = 220000;
            cfg.loadBuilderFn = @build_gradual_busC_loads_v4;
            cfg.loadBuilderExtraArgs = {800, 2800, 200000, 3000};

        case 'three_bus'
            trigG = 1081.96 - 248.6;
            trigH = 1082.05 - 249.1;
            trigK = 1108.70 - 149.0;
            cfg.targetBuses  = {'G','H','K'};
            cfg.triggerTimes = [trigG, trigH, trigK];
            cfg.onsetRef     = [1081.96, 1082.05, 1108.70];
            cfg.shedTime     = min(cfg.triggerTimes);
            cfg.shedTimeNote = 'earliest of G/H/K triggers (Bus H''s)';
            cfg.loadSeed = 201;
            cfg.pmin = 2000; cfg.pmax = 220000;
            cfg.loadBuilderFn = @build_three_bus_collapse_loads;
            cfg.loadBuilderExtraArgs = {800, 1400, 3400, 3700, ...
                27000, 33000, 190000, 45000, 31000, 3000, 27000, 0.03};

        otherwise
            error('Unknown SCENARIO_NAME: %s (expected ''bus_f'', ''bus_c'', or ''three_bus'')', name);
    end
end


%% ============================================================================
%%  OBJECTIVE FUNCTION (verbatim from gapso_load_shedding_timeboxed.m)
%% ============================================================================

function [fitness, diag] = gapsoObjective(x, cfg, shared)
    nvars = numel(cfg.targetBuses);
    shedPct = x(1:nvars);

    stopTime = min(shared.TEND_FULL, cfg.shedTime + shared.EVAL_POST_S);

    [so, n_steps_used] = simulateScenario(cfg, shared, shedPct, stopTime);

    time_vec = (0:n_steps_used-1)' * shared.TS;

    diag.all_safe = true;
    diag.min_v = nan(1, nvars);
    diag.collapsed = false(1, nvars);
    penalty = 0;

    for i = 1:nvars
        bus = cfg.targetBuses{i};
        v = grab_series(so, sprintf('V_Bus_%s', bus), shared.TS, n_steps_used);
        [c, ~, min_v] = detect_collapse_summary( ...
            time_vec, v, shared.COLLAPSE_VOLTAGE_V, shared.MIN_CONSECUTIVE_SAMPLES);
        diag.min_v(i) = min_v;
        diag.collapsed(i) = c;
        if c
            diag.all_safe = false;
            penalty = penalty + 1000;
        end
    end

    shedFracFromPct = shedPct / 100;
    powerTerm = mean(shedFracFromPct);

    voltageDeficit = max(0, (shared.HEALTHY_VOLTAGE_V - diag.min_v) / shared.HEALTHY_VOLTAGE_V);
    voltageDeficit(isnan(voltageDeficit)) = 1;
    voltageTerm = mean(voltageDeficit);

    dabPenalty = dabFeasibilityPenalty(cfg, shared, so, n_steps_used);

    fitness = shared.OBJ_WEIGHT_POWER * powerTerm + ...
              shared.OBJ_WEIGHT_VOLTAGE * voltageTerm + ...
              penalty + dabPenalty;

    diag.fitness = fitness;
    diag.powerTerm = powerTerm;
    diag.voltageTerm = voltageTerm;
    diag.dabPenalty = dabPenalty;

    close_system(shared.MODEL_NAME, 0);
end


%% ============================================================================
%%  SIMULATION HELPER (verbatim from gapso_load_shedding_timeboxed.m)
%% ============================================================================

function [so, n_steps_used] = simulateScenario(cfg, shared, shedPct, stopTime)
    nvars = numel(cfg.targetBuses);
    MODEL_NAME = shared.MODEL_NAME;

    if bdIsLoaded(MODEL_NAME)
        close_system(MODEL_NAME, 0);
    end
    load_system(shared.model_file);

    wire_all_taps(MODEL_NAME, shared.BUS, shared.CONV);
    set_param(MODEL_NAME, 'StopTime', num2str(stopTime));

    for b = 1:numel(shared.BUS)
        bus = shared.BUS{b};
        default_blk = sprintf('%s/ShedFrac_%s_default', MODEL_NAME, bus);
        ensure_constant_block(MODEL_NAME, bus, default_blk, 1);
        ramp_blk = sprintf('%s/ShedFrac_%s_ramp', MODEL_NAME, bus);
        if getSimulinkBlockHandle(ramp_blk) ~= -1
            set_param(ramp_blk, 'RisingSlewLimit', 'inf', 'FallingSlewLimit', '-inf');
        end
    end

    for i = 1:nvars
        bus = cfg.targetBuses{i};
        afterVal = 1 - (shedPct(i) / 100);
        blk = sprintf('%s/ShedFrac_%s_default', MODEL_NAME, bus);
        set_shed_step(MODEL_NAME, bus, blk, cfg.shedTime, 1, afterVal);
        ramp_block = sprintf('%s/ShedFrac_%s_ramp', MODEL_NAME, bus);
        if getSimulinkBlockHandle(ramp_block) ~= -1
            set_param(ramp_block, 'RisingSlewLimit', 'inf', 'FallingSlewLimit', '-inf');
        end
    end

    ws = warning('off', 'all');
    so = sim(MODEL_NAME, 'ReturnWorkspaceOutputs', 'on');
    warning(ws);

    n_steps_used = round(stopTime / shared.TS) + 1;
end


%% ============================================================================
%%  UNSHED LOAD LOOKUP (verbatim from gapso_load_shedding_timeboxed.m)
%% ============================================================================

function unshedLoadW = getUnshedLoadAtTrigger(cfg, shared)
    nvars = numel(cfg.targetBuses);
    unshedLoadW = nan(1, nvars);
    idx = round(cfg.shedTime / shared.TS) + 1;
    for i = 1:nvars
        bus = cfg.targetBuses{i};
        varname = sprintf('Pload_%s', bus);
        if evalin('base', sprintf('exist(''%s'',''var'')', varname)) ~= 1
            error(['getUnshedLoadAtTrigger: %s not found in base workspace. ' ...
                   'The load builder must run (Section 4) before this is called.'], varname);
        end
        p = evalin('base', varname);
        idx_clamped = min(max(idx, 1), numel(p));
        unshedLoadW(i) = p(idx_clamped);
    end
end


%% ============================================================================
%%  DAB CONVERTER FEASIBILITY CHECK (verbatim from gapso_load_shedding_timeboxed.m)
%% ============================================================================

function penalty = dabFeasibilityPenalty(cfg, shared, so, n_steps_used)
    BUS = shared.BUS;
    K = shared.DAB.K;
    penalty = 0;

    time_vec = (0:n_steps_used-1)' * shared.TS; %#ok<NASGU>

    for i = 1:numel(cfg.targetBuses)
        bus = cfg.targetBuses{i};
        bidx = find(strcmp(BUS, bus));
        if isempty(bidx); continue; end

        neighbors = ring_neighbors(BUS, bidx);
        for n = 1:numel(neighbors)
            nb = neighbors{n};
            Vb = grab_series(so, sprintf('V_Bus_%s', bus), shared.TS, n_steps_used);
            Vn = grab_series(so, sprintf('V_Bus_%s', nb),  shared.TS, n_steps_used);
            Pb = grab_series(so, sprintf('Bus_%s_Src_Pow', bus), shared.TS, n_steps_used);

            valid = ~isnan(Vb) & ~isnan(Vn) & ~isnan(Pb) & Vb > 1 & Vn > 1;
            if ~any(valid); continue; end

            Pmax = K * Vb(valid) .* Vn(valid);
            overLimit = abs(Pb(valid)) > Pmax;
            if any(overLimit)
                frac_over = mean(overLimit);
                penalty = penalty + 5 * frac_over;
            end
        end
    end
end


function neighbors = ring_neighbors(BUS, idx)
    n = numel(BUS);
    prev_idx = idx - 1; if prev_idx < 1; prev_idx = n; end
    next_idx = idx + 1; if next_idx > n; next_idx = 1; end
    neighbors = {BUS{prev_idx}, BUS{next_idx}};
end


%% ============================================================================
%%  FULL, UNTRUNCATED CONFIRMATION RUN (verbatim from gapso_load_shedding_timeboxed.m)
%% ============================================================================

function full_result = runFullConfirmation(x_best, cfg, shared, OUTPUT_DIR)
    nvars = numel(cfg.targetBuses);
    [so, n_steps_used] = simulateScenario(cfg, shared, x_best, shared.TEND_FULL);
    time_vec = (0:n_steps_used-1)' * shared.TS;

    fprintf('%-6s %-10s %-12s %-10s\n', 'Bus', 'Shed?', 'Outcome', 'Min V');
    bus_results = struct('bus', {}, 'shed', {}, 'shed_pct', {}, 'collapsed', {}, 'onset', {}, 'min_v', {});
    all_safe = true;

    for b = 1:numel(shared.BUS)
        bus = shared.BUS{b};
        v = grab_series(so, sprintf('V_Bus_%s', bus), shared.TS, n_steps_used);
        [c, onset, min_v] = detect_collapse_summary( ...
            time_vec, v, shared.COLLAPSE_VOLTAGE_V, shared.MIN_CONSECUTIVE_SAMPLES);

        ti = find(strcmp(cfg.targetBuses, bus));
        is_shed = ~isempty(ti);
        shed_pct = 0; if is_shed; shed_pct = x_best(ti); end

        shed_str = 'no'; if is_shed; shed_str = sprintf('%.1f%%', shed_pct); end
        outcome_str = 'survives'; if c; outcome_str = sprintf('COLLAPSE@%.0fs', onset); end
        fprintf('%-6s %-10s %-12s %-10.2f\n', bus, shed_str, outcome_str, min_v);

        if c; all_safe = false; end
        bus_results(end+1) = struct('bus', bus, 'shed', is_shed, 'shed_pct', shed_pct, ... %#ok<AGROW>
            'collapsed', c, 'onset', onset, 'min_v', min_v);
    end

    full_result.all_safe = all_safe;
    full_result.bus_results = bus_results;
    full_result.x_best = x_best;

    zoom_lo = max(0, cfg.shedTime - 200);
    zoom_hi = min(shared.TEND_FULL, max(cfg.onsetRef) + 200);
    zoom_mask = time_vec >= zoom_lo & time_vec <= zoom_hi;
    zoom_time = time_vec(zoom_mask);

    fig = figure('Color', 'white', 'Position', [50, 50, 1600, 700], 'Visible', 'off');
    for b = 1:numel(shared.BUS)
        bus = shared.BUS{b};
        v = grab_series(so, sprintf('V_Bus_%s', bus), shared.TS, n_steps_used);
        ax = subplot(2, 5, b);
        set(ax, 'Color', 'white');
        plot(zoom_time, v(zoom_mask), 'Color', [0.1 0.1 0.1], 'LineWidth', 1.1);
        hold on;
        yline(shared.COLLAPSE_VOLTAGE_V, '--', 'Color', [0.85 0.2 0.2], 'LineWidth', 0.8);
        xline(cfg.shedTime, ':', 'Color', [0.15 0.45 0.85], 'LineWidth', 1.2);
        r = bus_results(b);
        if r.collapsed
            title_str = sprintf('%s: COLLAPSE @%.0fs', bus, r.onset);
            title_color = [0.85 0.2 0.2];
        else
            title_str = sprintf('%s: survives (min %.0fV)', bus, r.min_v);
            title_color = [0.1 0.5 0.15];
        end
        shed_tag = ''; if r.shed; shed_tag = sprintf(' [SHED %.1f%%]', r.shed_pct); end
        title([title_str shed_tag], 'Color', title_color, 'FontSize', 9);
        xlabel('t (s)', 'FontSize', 8); ylabel('V', 'FontSize', 8);
        ylim([0 850]); grid on; hold off;
    end
    sgtitle(sprintf('GA-PSO time-boxed search (%s, widened 400s window): full confirmation run, shed vector = %s', ...
                     cfg.name, mat2str(x_best, 4)), 'FontSize', 12, 'FontWeight', 'bold');
    png_path = fullfile(OUTPUT_DIR, sprintf('gapso_%s_confirmation_grid.png', cfg.name));
    exportgraphics(fig, png_path, 'BackgroundColor', 'white');
    close(fig);
    fprintf('Wrote: %s\n', png_path);

    close_system(shared.MODEL_NAME, 0);
end


%% ============================================================================
%%  LOAD-PROFILE BUILDERS (verbatim from gapso_load_shedding_timeboxed.m)
%% ============================================================================

function [load_hist, clip_hi, clip_lo] = build_derating_loads_bus_f( ...
    seed, n_steps, ts, BUS, pmin, pmax, scale)

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


function [load_hist, clip_hi, clip_lo] = build_three_bus_collapse_loads( ...
    seed, n_steps, ts, BUS, pmin, pmax, ...
    preheat_end_s, surge_end_s, plateau_end_s, stepdown_end_s, ...
    ambient_w, target_preheat_w, target_plateau_w, target_recovery_w, ...
    boundary_preheat_w, boundary_low_w, boundary_recovery_w, jitter_frac)

    rng(seed);
    t = (0:n_steps-1)' * ts;
    NB = numel(BUS);

    TARGET_BUSES   = {'G','H','K'};
    BOUNDARY_BUSES = {'F','L'};

    idx_target = find(ismember(BUS, TARGET_BUSES));

    jitter_by_bus = ones(1, NB);
    jitter_by_bus(idx_target) = 1 + jitter_frac * (2*rand(1, numel(idx_target)) - 1);

    frac_surge    = min(1, max(0, (t - preheat_end_s) / (surge_end_s - preheat_end_s)));
    frac_stepdown = min(1, max(0, (t - plateau_end_s) / (stepdown_end_s - plateau_end_s)));

    load_hist = nan(n_steps, NB);
    clip_hi   = nan(1, NB);
    clip_lo   = nan(1, NB);

    for k = 1:NB
        phase_shift = 0.4 * k;
        slow_var = 4000 * sin(2*pi*t/1500 + phase_shift);
        noise    = 800  * randn(n_steps, 1);

        is_target   = ismember(BUS{k}, TARGET_BUSES);
        is_boundary = ismember(BUS{k}, BOUNDARY_BUSES);

        if is_target
            ji = jitter_by_bus(k);
            preheat_local  = target_preheat_w  * ji;
            mid_local      = target_plateau_w  * ji;
            recovery_local = target_recovery_w * ji;
            raw = preheat_local ...
                + (mid_local - preheat_local) .* frac_surge ...
                + (recovery_local - mid_local) .* frac_stepdown ...
                + slow_var + noise;
        elseif is_boundary
            preheat_local  = boundary_preheat_w;
            mid_local      = boundary_low_w;
            recovery_local = boundary_recovery_w;
            raw = preheat_local ...
                + (mid_local - preheat_local) .* frac_surge ...
                + (recovery_local - mid_local) .* frac_stepdown ...
                + slow_var + noise;
        else
            raw = ambient_w + slow_var + noise;
        end

        clip_hi(k) = mean(raw > pmax) * 100;
        clip_lo(k) = mean(raw < pmin) * 100;

        p = max(pmin, min(pmax, raw));
        load_hist(:, k) = p;
        assignin('base', sprintf('Pload_%s', BUS{k}), p(:).');
    end
end


%% ============================================================================
%%  BLOCK MANIPULATION + TAP-WIRING HELPERS (verbatim from gapso_load_shedding_timeboxed.m)
%% ============================================================================

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
