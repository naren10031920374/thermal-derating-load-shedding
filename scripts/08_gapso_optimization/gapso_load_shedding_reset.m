function gapso_load_shedding_reset()
% ============================================================================
%  GAPSO_LOAD_SHEDDING_RESET.M
%  --------------------------------------------------------------------------
%  Stage 06 GA-PSO load-shedding optimizer -- rebuilt from scratch 2026-09-22
%  after Prof. Bui's direction to drop DVSI (the existing ML detector already
%  supplies "which bus, and when") and use Paper 2's real GA-PSO optimizer
%  (MATLAB's Global Optimization Toolbox is confirmed available on this
%  machine) to answer the one remaining question: how much load to shed per
%  flagged bus.
%
%  DESIGN AGREED WITH NAREN BEFORE ANY CODE WAS WRITTEN (2026-09-22):
%   1. Decision variables: one shed PERCENTAGE per bus flagged by the ML
%      detector for that scenario (1 number for Bus F, 1 for Bus C, 3
%      independent numbers for G/H/K -- not one shared fraction).
%   2. Search range: 30%-60% shed per bus.
%   3. Objective: 70% weight on shedding as little power as possible, 30%
%      weight on improving voltage as much as possible (Paper 2's own
%      weighting).
%   4. Speed: each GA-PSO candidate is scored with a TRUNCATED simulation
%      (StopTime cut short to just past the scenario's shed point, not the
%      full ~5000s run), after a small timing test to confirm this is
%      actually fast enough. The final winning vector is re-confirmed with
%      one full, untruncated 5000s run before being treated as final.
%
%  WHY TRUNCATING STOPTIME (not the load array) IS SAFE: the load profile
%  fed to the model is built once, at full 5000s length, with the exact
%  same seed/formula every already-verified closed-loop script in this
%  project uses for this scenario. Only the simulation's StopTime is cut
%  short during the search. This means a truncated search candidate and
%  the final full-length confirmation run see IDENTICAL load data over the
%  time window they share -- nothing about the physics changes, only how
%  far into it we bother simulating.
%
%  WHAT REPLACES AC POWER-FLOW VALIDITY CHECKING (the original Larik et al.
%  2018 paper checks candidates against AC real/reactive power-flow
%  equations -- this DC ring has no reactive power): the DAB converter
%  power-transfer law, confirmed directly against this project's own
%  Simulink model (stage-06-dc-fvsi-derivation.md, Section 1):
%     P_ij = n * Vi * Vj * phi * (1 - |phi|/pi) / (2*pi*fs*Lk)
%     P_max(Vi,Vj) = K * Vi * Vj,   K = n*pi / (8*fs*Lk)
%  with n=24/15, fs=10kHz, Lk=340uH -- this project's real, confirmed
%  converter values. Used here as a FEASIBILITY CHECK folded into the
%  fitness as a soft penalty (can this converter actually deliver the
%  post-shed demanded power), not as a DVSI-style early-warning score and
%  not as a hard MATLAB nonlcon (kept simple/robust on purpose -- see
%  dabFeasibilityPenalty() below for exactly what it checks).
%
%  WHAT THIS SCRIPT DOES NOT YET FULLY VERIFY (flagged honestly rather than
%  silently assumed -- check this before trusting search results):
%   - getUnshedLoadAtTrigger() reads each target bus's commanded load
%     (Pload_<bus>) at the scenario's shed time from the SAME load-profile
%     builder function every verified closed-loop script in this project
%     already uses for that scenario. This should be correct by
%     construction, but has not been checked against a real MATLAB run
%     yet -- read the printed values in Section 5's output and sanity
%     check them (they should be well within [pmin, pmax] for that
%     scenario) before trusting the objective's power-shed term.
%   - dabFeasibilityPenalty() approximates each target bus's two incident
%     converter lines from the ring order (A-B-C-D-E-F-G-H-K-L-A) and reads
%     their measured Bus_X_Src_Pow / V_Bus_X taps as a proxy for what each
%     converter is being asked to deliver. This is a real, physics-based
%     sanity check, but it is intentionally a SOFT penalty, not a hard
%     constraint -- treat any large recurring penalty in the console output
%     as a flag worth looking into, not an automatic disqualification.
%
%  USAGE:
%    1. Set SCENARIO_NAME below to 'bus_f', 'bus_c', or 'three_bus'.
%    2. Run: gapso_load_shedding_reset
%    3. Section 6 runs a small timing test FIRST and prints an estimated
%       full-search runtime -- read this before letting the full search
%       run unattended.
%    4. After the search converges, Section 8 re-confirms the winning shed
%       vector with one full, untruncated 5000s simulation across all 10
%       buses, same validation standard as every other result in this
%       project -- the search's own output is never the last word.
% ============================================================================

    clc;

    %% 0. CHOOSE SCENARIO ----------------------------------------------------
    SCENARIO_NAME = 'bus_f';   % 'bus_f' | 'bus_c' | 'three_bus'

    %% 1. GLOBAL / SHARED CONFIG ----------------------------------------------
    PROJECT_ROOT = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    MODEL_NAME   = 'Grid_modelling_Thermal_V7_ALfix_sheddable_ramped';

    TS   = 0.01;
    TEND_FULL = 5000;
    N_STEPS_FULL = round(TEND_FULL / TS) + 1;

    COLLAPSE_VOLTAGE_V      = 100.0;
    MIN_CONSECUTIVE_SAMPLES = 50;     % 0.5s debounce @ 10ms sampling
    HEALTHY_VOLTAGE_V       = 780.0;  % typical steady operating voltage, used
                                       % only to normalize the objective's
                                       % voltage-improvement term (0..1 scale)

    BUS  = {'A','B','C','D','E','F','G','H','K','L'};   % ring order matters --
                                                          % used to infer each
                                                          % bus's incident DAB
                                                          % converters below
    CONV = {'AB','BC','CD','DE','EF','FG','GH','HK','KL','AL'};

    % DAB converter constants (confirmed against the Simulink model,
    % stage-06-dc-fvsi-derivation.md Section 1)
    DAB.n  = 24/15;                          % turns ratio
    DAB.fs = 10e3;                           % Hz
    DAB.Lk = 340e-6;                         % H
    DAB.K  = DAB.n*pi / (8*DAB.fs*DAB.Lk);   % P_max = K * Vi * Vj

    % Agreed search design (2026-09-22)
    SHED_PCT_MIN = 30;   % percent
    SHED_PCT_MAX = 60;   % percent
    OBJ_WEIGHT_POWER   = 0.70;
    OBJ_WEIGHT_VOLTAGE = 0.30;

    % Truncated-eval window: how far past the scenario's shed time the
    % search-time simulations run. TENTATIVE -- confirm/adjust after
    % Section 6's timing test prints real wall-clock numbers.
    EVAL_POST_S = 120;

    OUTPUT_DIR = fullfile(PROJECT_ROOT, 'model_outputs', 'gapso_reset_2026_09_22', SCENARIO_NAME);
    if exist(OUTPUT_DIR, 'dir') ~= 7; mkdir(OUTPUT_DIR); end
    addpath(PROJECT_ROOT);

    %% 2. PER-SCENARIO CONFIG --------------------------------------------------
    cfg = build_scenario_config(SCENARIO_NAME);

    fprintf('%s\n', repmat('=', 1, 78));
    fprintf('GA-PSO LOAD-SHEDDING SEARCH -- scenario: %s\n', SCENARIO_NAME);
    fprintf('%s\n', repmat('=', 1, 78));
    fprintf('Target buses:          %s\n', strjoin(cfg.targetBuses, ', '));
    fprintf('Trigger time(s):       %s\n', mat2str(cfg.triggerTimes, 6));
    fprintf('Shed applied at:       %.2fs (%s)\n', cfg.shedTime, cfg.shedTimeNote);
    fprintf('Search range per bus:  %d%%-%d%% shed\n', SHED_PCT_MIN, SHED_PCT_MAX);
    fprintf('Objective weights:     %.0f%% power-shed, %.0f%% voltage-improvement\n', ...
            OBJ_WEIGHT_POWER*100, OBJ_WEIGHT_VOLTAGE*100);

    nvars = numel(cfg.targetBuses);
    lb = SHED_PCT_MIN * ones(1, nvars);
    ub = SHED_PCT_MAX * ones(1, nvars);

    %% 3. SHARED CONTEXT PASSED INTO THE OBJECTIVE/HELPERS ---------------------
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

    %% 4. BUILD THE FULL LOAD PROFILE ONCE (shared by every candidate) ---------
    fprintf('\nBuilding full %.0fs load profile (seed %d) ...\n', TEND_FULL, cfg.loadSeed);
    cfg.loadBuilderFn(cfg.loadSeed, N_STEPS_FULL, TS, BUS, cfg.pmin, cfg.pmax, cfg.loadBuilderExtraArgs{:});

    %% 5. UNSHED LOAD AT TRIGGER TIME (power-shed term input) ------------------
    cfg.unshedLoadW = getUnshedLoadAtTrigger(cfg, shared);
    fprintf('\nUnshed commanded load at shed time, per target bus (sanity-check these):\n');
    for i = 1:nvars
        fprintf('  Bus %s: %.0f W  (scenario pmin=%.0f, pmax=%.0f)\n', ...
                cfg.targetBuses{i}, cfg.unshedLoadW(i), cfg.pmin, cfg.pmax);
    end

    %% 6. TIMING TEST -- before committing to a full search --------------------
    fprintf('\n%s\n', repmat('-', 1, 78));
    fprintf('TIMING TEST: 3 sample candidates, truncated-window evaluation\n');
    fprintf('%s\n', repmat('-', 1, 78));
    test_candidates = [ ...
        SHED_PCT_MIN * ones(1, nvars); ...
        mean([SHED_PCT_MIN, SHED_PCT_MAX]) * ones(1, nvars); ...
        SHED_PCT_MAX * ones(1, nvars)];
    test_times = nan(3,1);
    for i = 1:3
        tic;
        [fit_i, diag_i] = gapsoObjective(test_candidates(i,:), cfg, shared);
        test_times(i) = toc;
        fprintf('  Candidate %d (%s): %.1fs wall clock, all_safe=%d, fitness=%.4f\n', ...
                i, mat2str(test_candidates(i,:)), test_times(i), diag_i.all_safe, fit_i);
    end
    fprintf('\nMean time per candidate: %.1fs\n', mean(test_times));
    fprintf('NOTE: read this before proceeding. If POP_SIZE * MAX_GENERATIONS * 2\n');
    fprintf('(PSO pass + GA pass) times this mean would take too long, reduce\n');
    fprintf('POP_SIZE/MAX_GENERATIONS below or shrink EVAL_POST_S above, then\n');
    fprintf('rerun this script from the top before launching the full search.\n');

    %% 7. PSO (broad search) THEN GA (refinement) -- Larik et al. 2018 ---------
    POP_SIZE         = 20;   % START SMALL -- scale up only after Section 6's
    MAX_GENERATIONS  = 15;   % timing readout says the budget is reasonable

    fprintf('\n%s\n', repmat('-', 1, 78));
    fprintf('PARTICLE SWARM (global search) ...\n');
    fprintf('%s\n', repmat('-', 1, 78));

    psoOpts = optimoptions('particleswarm', ...
        'SwarmSize', POP_SIZE, ...
        'MaxIterations', MAX_GENERATIONS, ...
        'Display', 'iter', ...
        'UseParallel', false);

    psoObj = @(x) gapsoObjective(x, cfg, shared);
    [x_pso, fval_pso] = particleswarm(psoObj, nvars, lb, ub, psoOpts);
    fprintf('PSO result: %s  (fitness=%.4f)\n', mat2str(x_pso, 4), fval_pso);

    fprintf('\n%s\n', repmat('-', 1, 78));
    fprintf('GENETIC ALGORITHM (refinement, seeded from PSO result) ...\n');
    fprintf('%s\n', repmat('-', 1, 78));

    init_pop = [x_pso; repmat(x_pso, POP_SIZE - 1, 1) + 5*randn(POP_SIZE - 1, nvars)];
    init_pop = min(max(init_pop, SHED_PCT_MIN), SHED_PCT_MAX);

    gaOpts = optimoptions('ga', ...
        'PopulationSize', POP_SIZE, ...
        'MaxGenerations', MAX_GENERATIONS, ...
        'InitialPopulationMatrix', init_pop, ...
        'Display', 'iter', ...
        'UseParallel', false);

    gaObj = @(x) gapsoObjective(x, cfg, shared);
    [x_best, fval_best] = ga(gaObj, nvars, [], [], [], [], lb, ub, [], gaOpts);

    % GA can drift fractionally outside bounds due to floating point --
    % clip back in, and round to 1 decimal (no need for more precision
    % on a shed percentage in practice).
    x_best = round(min(max(x_best, SHED_PCT_MIN), SHED_PCT_MAX), 1);

    fprintf('\n%s\n', repmat('=', 1, 78));
    fprintf('SEARCH COMPLETE\n');
    fprintf('%s\n', repmat('=', 1, 78));
    for i = 1:nvars
        fprintf('  Bus %s: %.1f%% shed\n', cfg.targetBuses{i}, x_best(i));
    end
    fprintf('  Final fitness: %.4f\n', fval_best);

    %% 8. FULL, UNTRUNCATED CONFIRMATION RUN ------------------------------------
    fprintf('\n%s\n', repmat('-', 1, 78));
    fprintf('CONFIRMING WINNING VECTOR WITH ONE FULL %.0fs RUN, ALL 10 BUSES ...\n', TEND_FULL);
    fprintf('%s\n', repmat('-', 1, 78));

    full_result = runFullConfirmation(x_best, cfg, shared, OUTPUT_DIR);

    if full_result.all_safe
        fprintf('\nCONFIRMED: all 10 buses survive with this shed vector.\n');
    else
        fprintf('\nNOT CONFIRMED: at least one bus still collapses in the full run.\n');
        fprintf('Do not treat the search result as final. Check full_result.bus_results\n');
        fprintf('and consider whether EVAL_POST_S was too short for the search to see\n');
        fprintf('the real outcome, or whether the search needs a larger population/more\n');
        fprintf('generations, then rerun.\n');
    end

    %% 9. SAVE ------------------------------------------------------------------
    result_mat = fullfile(OUTPUT_DIR, sprintf('gapso_result_%s.mat', SCENARIO_NAME));
    save(result_mat, 'SCENARIO_NAME', 'cfg', 'x_best', 'fval_best', 'x_pso', 'fval_pso', ...
         'test_times', 'full_result', 'SHED_PCT_MIN', 'SHED_PCT_MAX', ...
         'OBJ_WEIGHT_POWER', 'OBJ_WEIGHT_VOLTAGE', 'EVAL_POST_S', '-v7.3');
    fprintf('\nWrote: %s\n', result_mat);
    fprintf('All outputs in: %s\n', OUTPUT_DIR);
end


%% ============================================================================
%%  SCENARIO CONFIGURATION
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
            cfg.loadBuilderExtraArgs = {1.0};   % GRID_LOAD_SCALE

        case 'bus_c'
            cfg.targetBuses  = {'C'};
            cfg.triggerTimes = 1002.50;
            cfg.onsetRef     = 2094.47;
            cfg.shedTime     = cfg.triggerTimes(1);
            cfg.shedTimeNote = 'Bus C''s own real cross-scenario detector trigger';
            cfg.loadSeed = 104;
            cfg.pmin = 2000; cfg.pmax = 220000;
            cfg.loadBuilderFn = @build_gradual_busC_loads_v4;
            cfg.loadBuilderExtraArgs = {800, 2800, 200000, 3000};  % ramp start/end, C target W, B/D target W

        case 'three_bus'
            trigG = 1081.96 - 248.6;   % 833.36
            trigH = 1082.05 - 249.1;   % 832.95
            trigK = 1108.70 - 149.0;   % 959.70
            cfg.targetBuses  = {'G','H','K'};
            cfg.triggerTimes = [trigG, trigH, trigK];
            cfg.onsetRef     = [1081.96, 1082.05, 1108.70];
            % Synchronized at the earliest trigger (Bus H's), same
            % "treat the region as one alarm zone" convention already used
            % and validated by this project's fraction-sweep/bisection
            % closed-loop scripts for this scenario.
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
%%  OBJECTIVE FUNCTION
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
            % Large fixed penalty per collapsing bus -- dominates the
            % objective so a candidate that lets a target bus collapse
            % never scores better than one that keeps every target bus
            % alive, regardless of how little power it shed.
            penalty = penalty + 1000;
        end
    end

    % ---- Power-shed term (70% weight): fraction of each target bus's real
    % unshed load that this candidate cuts, averaged across target buses ----
    shedFracFromPct = shedPct / 100;
    powerTerm = mean(shedFracFromPct);   % 0..~0.6 by construction (bounds)

    % ---- Voltage-improvement term (30% weight): how far below healthy
    % operating voltage each target bus's worst moment in the window was,
    % normalized 0 (fully healthy) to 1 (collapsed) ----
    voltageDeficit = max(0, (shared.HEALTHY_VOLTAGE_V - diag.min_v) / shared.HEALTHY_VOLTAGE_V);
    voltageDeficit(isnan(voltageDeficit)) = 1;   % missing signal treated as worst case
    voltageTerm = mean(voltageDeficit);

    % ---- DAB feasibility soft penalty (see file header) ----
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
%%  SIMULATION HELPER (shared by the objective and the full confirmation run)
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

    % Reset every bus's shed actuator to "no shedding" first.
    for b = 1:numel(shared.BUS)
        bus = shared.BUS{b};
        default_blk = sprintf('%s/ShedFrac_%s_default', MODEL_NAME, bus);
        ensure_constant_block(MODEL_NAME, bus, default_blk, 1);
        ramp_blk = sprintf('%s/ShedFrac_%s_ramp', MODEL_NAME, bus);
        if getSimulinkBlockHandle(ramp_blk) ~= -1
            set_param(ramp_blk, 'RisingSlewLimit', 'inf', 'FallingSlewLimit', '-inf');
        end
    end

    % Apply this candidate's shed percentages to the target buses only, at
    % the scenario's shed time. Instantaneous shed (RAMP_DURATION=0), same
    % default as every verified closed-loop script in this project.
    for i = 1:nvars
        bus = cfg.targetBuses{i};
        afterVal = 1 - (shedPct(i) / 100);   % ShedFrac "after" value
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
%%  UNSHED LOAD LOOKUP (power-shed term input)
%% ============================================================================

function unshedLoadW = getUnshedLoadAtTrigger(cfg, shared)
    % The load builder already ran once in Section 4 and assigned
    % Pload_<bus> vectors to the base workspace via assignin('base', ...).
    % Read each target bus's value at the sample nearest the shed time.
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
%%  DAB CONVERTER FEASIBILITY CHECK (replaces AC power-flow validity check)
%% ============================================================================

function penalty = dabFeasibilityPenalty(cfg, shared, so, n_steps_used)
    % For each target bus, look up its two incident converter lines from
    % the ring order, and check whether the power apparently being moved
    % across that line (approximated from the two endpoint buses' tapped
    % source power and voltage) would require a phase shift beyond the
    % converter's physical +-90 degree limit. This is a real physics-based
    % sanity check (see file header for the formula and confirmed
    % constants), but is intentionally kept as a SOFT, informational
    % penalty -- not a hard constraint -- since it relies on approximating
    % "power this line is carrying" from bus-level taps rather than a
    % direct per-converter power measurement.
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
                % Small penalty proportional to how often/how badly this
                % line's implied demand exceeds its physical ceiling --
                % kept modest on purpose (see note above).
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
%%  FULL, UNTRUNCATED CONFIRMATION RUN
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

    %% Plot -- white background, high-contrast, per this project's standing rule
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
    sgtitle(sprintf('GA-PSO reset (%s): full confirmation run, shed vector = %s', ...
                     cfg.name, mat2str(x_best, 4)), 'FontSize', 12, 'FontWeight', 'bold');
    png_path = fullfile(OUTPUT_DIR, sprintf('gapso_%s_confirmation_grid.png', cfg.name));
    exportgraphics(fig, png_path, 'BackgroundColor', 'white');
    close(fig);
    fprintf('Wrote: %s\n', png_path);

    close_system(shared.MODEL_NAME, 0);
end


%% ============================================================================
%%  LOAD-PROFILE BUILDERS (verbatim from each scenario's own verified,
%%  already-published closed-loop script -- reused exactly, not reinvented,
%%  since the whole point is that the search and the final confirmation run
%%  see the SAME real load data every prior result in this project used)
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
%%  BLOCK MANIPULATION + TAP-WIRING HELPERS
%%  (verbatim, reused across every closed-loop script in this project)
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