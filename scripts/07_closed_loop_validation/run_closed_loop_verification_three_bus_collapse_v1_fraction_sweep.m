function run_closed_loop_verification_three_bus_collapse_v1_fraction_sweep()
% ==========================================================================
%  RUN_CLOSED_LOOP_VERIFICATION_THREE_BUS_COLLAPSE_V1_FRACTION_SWEEP.M
%  --------------------------------------------------------------------
%  Closed-loop shed-and-verify test for three_bus_collapse_v1, the first
%  closed-loop test in this project that has to shed MORE THAN ONE bus at
%  once (the baseline's and gradual_busC_v4's closed-loop tests each only
%  ever shed a single bus).
%
%  TWO EXPLICIT DESIGN DECISIONS (Naren's answers, not inferred):
%
%  1. SHED TIMING -- "synchronized at earliest trigger": G and H's real
%     N=60s cross-scenario detector triggers land almost together
%     (~833.4s and ~833.0s respectively), but K's trigger lands ~127s
%     later (~959.7s), because K collapses ~27s after G/H. Rather than
%     shedding each bus independently at its own trigger time, this
%     script sheds ALL THREE of G, H, K together, at the EARLIEST of the
%     three triggers (Bus H's, ~832.95s) -- i.e. as soon as any one of the
%     three detectors fires, the whole target trio is treated as one
%     regional alarm and shed together. This means K gets shed ~127s
%     before its own detector would have fired on its own -- deliberately
%     generous, per the "treat the region as one alarm zone" framing.
%
%  2. SHED FRACTION -- "go through every shed fraction and find which is
%     best": rather than reusing FIXED_FRACTION=0.30 (the single value
%     that worked for baseline's Bus F and gradual_busC_v4's Bus C,
%     un-validated for this 3-bus scenario), this script sweeps a full
%     range of ShedFrac_<bus> settings and reports which is the mildest
%     (least load cut) that still results in every one of the 10 buses
%     surviving the full run.
%
%  WHAT "ShedFrac" MEANS: per add_controllable_load_shedding.m, each bus's
%  commanded load is multiplied by ShedFrac_<bus> (a Product block spliced
%  into its load path). ShedFrac defaults to 1 (full load, no shedding).
%  Setting it to X after the trigger means the bus's load is cut to X of
%  its commanded value -- i.e. LOWER X = MORE aggressive shedding, HIGHER
%  X (closer to 1) = MILDER shedding. This is the exact same Step-block
%  mechanism (before=1, after=X) used in the baseline and gradual_busC_v4
%  closed-loop scripts; only the swept value and the fact that it's
%  applied to three buses at once are new here.
%
%  WHERE THE TRIGGER TIMES COME FROM: test_three_bus_collapse_v1_against_
%  baseline_detectors.py's N=60s sweep, threshold=1e-6 (the clean,
%  zero-false-alarm operating point already used to select N=60s for the
%  other two scenarios):
%      Bus G: onset=1081.96s, lead=248.6s  -> trigger = 833.36s
%      Bus H: onset=1082.05s, lead=249.1s  -> trigger = 832.95s  <- earliest
%      Bus K: onset=1108.70s, lead=149.0s  -> trigger = 959.70s
%  SHED_TIME below is computed as min(TRIGGER_G, TRIGGER_H, TRIGGER_K),
%  which evaluates to Bus H's ~832.95s, matching the "synchronized at
%  earliest trigger" decision (not hardcoded as a literal, so it stays
%  correct even if the per-bus onset/lead constants above are ever revised).
%
%  LOAD PROFILE: reused verbatim from generate_three_bus_collapse_scenario_
%  v1.m (build_three_bus_collapse_loads, seed=201, PMIN=2000/PMAX=220000,
%  four-phase G/H/K surge + F/L boundary starvation). Boundary buses F and
%  L are NOT shed here, same "clean test" philosophy as the other two
%  scenarios: if everything survives, that survival is attributable ONLY
%  to shedding G/H/K, nothing else.
%
%  MODEL: reuses Grid_modelling_Thermal_V7_ALfix_sheddable_ramped.slx.
%  ShedFrac_G/H/K actuators already exist generically (wired onto all 10
%  buses when built for baseline), so no new Simulink infrastructure is
%  needed.
%
%  RUNTIME WARNING: this runs ONE FULL 5000s sim() PER CANDIDATE FRACTION
%  in FRACTION_CANDIDATES below (9 candidates by default). Each full run
%  has taken ~15-20 minutes wall clock for this model in every prior
%  script this project has run. Budget roughly 2.5-3 HOURS total for the
%  full 9-point sweep as written. Trim FRACTION_CANDIDATES to fewer values
%  first if you want a faster answer -- the loop makes no assumption that
%  candidates must be run in any particular order or count.
%
%  Run:
%    run_closed_loop_verification_three_bus_collapse_v1_fraction_sweep
% ==========================================================================

    clc;

    %% 0. CONFIG --------------------------------------------------------
    PROJECT_ROOT = 'D:\naren\Documents\Thermal_Derating_Work_Handover\ThermalProject';
    MODEL_NAME = 'Grid_modelling_Thermal_V7_ALfix_sheddable_ramped';

    % Real per-bus N=60s cross-scenario detector numbers (threshold=1e-6),
    % from test_three_bus_collapse_v1_against_baseline_detectors.py.
    ONSET_G = 1081.96;  LEAD_G = 248.6;  TRIGGER_G = ONSET_G - LEAD_G;   % 833.36
    ONSET_H = 1082.05;  LEAD_H = 249.1;  TRIGGER_H = ONSET_H - LEAD_H;   % 832.95
    ONSET_K = 1108.70;  LEAD_K = 149.0;  TRIGGER_K = ONSET_K - LEAD_K;   % 959.70

    SHED_TIME = min([TRIGGER_G, TRIGGER_H, TRIGGER_K]);   % synchronized at earliest trigger (Bus H's)

    TARGET_BUSES   = {'G','H','K'};     % shed together at SHED_TIME
    BOUNDARY_BUSES = {'F','L'};         % NOT shed -- clean-test philosophy

    % Sweep of ShedFrac_<bus> "after" values -- 1.0 = no shedding (never
    % tested, that's just the unshed baseline which already collapses),
    % descending toward more aggressive cuts. "Best" = the LARGEST value
    % (mildest cut) that still results in every bus surviving.
    FRACTION_CANDIDATES = [0.90 0.80 0.70 0.60 0.50 0.40 0.30 0.20 0.10];
    RAMP_DURATION = 0;   % instantaneous shed, same default as prior closed-loop tests

    TS   = 0.01;
    TEND = 5000;
    N_STEPS = round(TEND / TS) + 1;

    % Exact load-profile parameters from generate_three_bus_collapse_scenario_v1.m --
    % must match exactly, this is what produced the real regional collapse.
    SEED = 201;
    PMIN = 2000;
    PMAX = 220000;
    PREHEAT_END_S  = 800;
    SURGE_END_S    = 1400;
    PLATEAU_END_S  = 3400;
    STEPDOWN_END_S = 3700;
    AMBIENT_W            = 27000;
    TARGET_PREHEAT_W     = 33000;
    TARGET_PLATEAU_W     = 190000;
    TARGET_RECOVERY_W    = 45000;
    BOUNDARY_PREHEAT_W   = 31000;
    BOUNDARY_LOW_W       = 3000;
    BOUNDARY_RECOVERY_W  = 27000;
    TARGET_JITTER_FRAC   = 0.03;

    COLLAPSE_VOLTAGE_V = 100.0;
    MIN_CONSECUTIVE_SAMPLES = 50;

    BUS  = {'A','B','C','D','E','F','G','H','K','L'};
    CONV = {'AB','BC','CD','DE','EF','FG','GH','HK','KL','AL'};

    ZOOM_LO = 500; ZOOM_HI = 2000;   % covers preheat through the plateau where G/H/K collapse

    OUTPUT_DIR = fullfile(PROJECT_ROOT, 'model_outputs', 'thermal_derating_three_bus_collapse', 'closed_loop_verification');
    if exist(OUTPUT_DIR, 'dir') ~= 7; mkdir(OUTPUT_DIR); end
    addpath(PROJECT_ROOT);

    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('CLOSED-LOOP FRACTION SWEEP: three_bus_collapse_v1, G/H/K shed together\n');
    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('Trigger times (N=60s, threshold=1e-6): G=%.2fs, H=%.2fs, K=%.2fs\n', TRIGGER_G, TRIGGER_H, TRIGGER_K);
    fprintf('SYNCHRONIZED SHED TIME (earliest of the three): %.2fs\n', SHED_TIME);
    fprintf('Buses shed together at that time: %s\n', strjoin(TARGET_BUSES, ', '));
    fprintf('Boundary buses (F, L): NOT shed -- clean-test philosophy.\n');
    fprintf('Fraction candidates (ShedFrac after-value; lower = more aggressive cut):\n  ');
    fprintf('%.2f  ', FRACTION_CANDIDATES); fprintf('\n');
    fprintf('Expected total runtime: roughly %.0f-%.0f minutes for all %d candidates (~15-20 min each).\n', ...
            numel(FRACTION_CANDIDATES)*15, numel(FRACTION_CANDIDATES)*20, numel(FRACTION_CANDIDATES));

    hh = dir(fullfile(PROJECT_ROOT, '**', [MODEL_NAME '.slx']));
    if isempty(hh)
        error(['Ramped sheddable model not found: %s.slx\n' ...
               'This should already exist from the baseline scenario''s ' ...
               '05_load_shedding_infrastructure / 06_gradual_ramp work.'], MODEL_NAME);
    end
    model_file = fullfile(hh(1).folder, hh(1).name);
    fprintf('\nModel: %s\n', model_file);

    %% MAIN SWEEP LOOP ----------------------------------------------------
    n_cand = numel(FRACTION_CANDIDATES);
    sweep_results = struct('fraction', {}, 'all_safe', {}, 'bus_results', {}, 'wall_clock_s', {});

    for fi = 1:n_cand
        frac = FRACTION_CANDIDATES(fi);
        fprintf('\n%s\n', repmat('-', 1, 70));
        fprintf('CANDIDATE %d/%d: ShedFrac after-value = %.2f\n', fi, n_cand, frac);
        fprintf('%s\n', repmat('-', 1, 70));

        %% 1. Build the load profile -- IDENTICAL to generate_three_bus_collapse_scenario_v1.m
        fprintf('Building three_bus_collapse_v1 load profile (seed %d) ...\n', SEED);
        build_three_bus_collapse_loads( ...
            SEED, N_STEPS, TS, BUS, PMIN, PMAX, ...
            PREHEAT_END_S, SURGE_END_S, PLATEAU_END_S, STEPDOWN_END_S, ...
            AMBIENT_W, TARGET_PREHEAT_W, TARGET_PLATEAU_W, TARGET_RECOVERY_W, ...
            BOUNDARY_PREHEAT_W, BOUNDARY_LOW_W, BOUNDARY_RECOVERY_W, TARGET_JITTER_FRAC);

        %% 2. Fresh load of the sheddable+ramp model each iteration -----------
        if bdIsLoaded(MODEL_NAME)
            close_system(MODEL_NAME, 0);
        end
        load_system(model_file);

        for tb = TARGET_BUSES
            ramp_block = sprintf('%s/ShedFrac_%s_ramp', MODEL_NAME, tb{1});
            if getSimulinkBlockHandle(ramp_block) == -1
                error(['Block not found: %s\nExpected the shed-fraction ramp actuator ' ...
                       'for Bus %s to already exist in this model.'], ramp_block, tb{1});
            end
        end

        %% 3. Wire taps ------------------------------------------------------
        wire_all_taps(MODEL_NAME, BUS, CONV);
        set_param(MODEL_NAME, 'StopTime', num2str(TEND));

        %% 4. Reset every bus's shed actuator to "no shedding" first --------
        for b = 1:numel(BUS)
            default_blk = sprintf('%s/ShedFrac_%s_default', MODEL_NAME, BUS{b});
            ensure_constant_block(MODEL_NAME, BUS{b}, default_blk, 1);
            ramp_blk = sprintf('%s/ShedFrac_%s_ramp', MODEL_NAME, BUS{b});
            if getSimulinkBlockHandle(ramp_blk) ~= -1
                set_param(ramp_blk, 'RisingSlewLimit', 'inf', 'FallingSlewLimit', '-inf');
            end
        end

        %% 5. G, H, K shed together at SHED_TIME, boundary/background untouched --
        for i = 1:numel(TARGET_BUSES)
            tb = TARGET_BUSES{i};
            blk = sprintf('%s/ShedFrac_%s_default', MODEL_NAME, tb);
            set_shed_step(MODEL_NAME, tb, blk, SHED_TIME, 1, frac);
            ramp_block = sprintf('%s/ShedFrac_%s_ramp', MODEL_NAME, tb);
            if RAMP_DURATION <= 0
                set_param(ramp_block, 'RisingSlewLimit', 'inf', 'FallingSlewLimit', '-inf');
            else
                falling_slew = -(1 - frac) / RAMP_DURATION;
                set_param(ramp_block, 'RisingSlewLimit', 'inf', 'FallingSlewLimit', num2str(falling_slew));
            end
        end

        %% 6. Run the simulation ---------------------------------------------
        fprintf('Running closed-loop sim (fraction=%.2f) ...\n', frac);
        tic;
        ws = warning('off', 'all');
        so = sim(MODEL_NAME, 'ReturnWorkspaceOutputs', 'on');
        warning(ws);
        wall_s = toc;
        fprintf('Finished in %.1f s wall clock.\n', wall_s);

        time_vec = (0:N_STEPS-1)' * TS;

        bus_results = struct('bus', {}, 'shed', {}, 'collapsed', {}, 'onset', {}, 'min_v', {});
        all_safe = true;
        for b = 1:numel(BUS)
            bus = BUS{b};
            v_full = grab_series(so, sprintf('V_Bus_%s', bus), TS, N_STEPS);
            [c, onset, min_v] = detect_collapse_summary( ...
                time_vec, v_full, COLLAPSE_VOLTAGE_V, MIN_CONSECUTIVE_SAMPLES);
            is_shed = ismember(bus, TARGET_BUSES);
            shed_str = 'no'; if is_shed; shed_str = 'yes'; end
            outcome_str = 'survives'; if c; outcome_str = sprintf('COLLAPSE@%.0fs', onset); end
            fprintf('  %-4s shed=%-4s %-14s min_v=%.2f\n', bus, shed_str, outcome_str, min_v);
            if c; all_safe = false; end
            bus_results(end+1) = struct('bus', bus, 'shed', is_shed, ... %#ok<AGROW>
                'collapsed', c, 'onset', onset, 'min_v', min_v);
        end

        if all_safe
            fprintf('  -> ALL 10 BUSES SURVIVE at fraction=%.2f\n', frac);
        else
            fprintf('  -> AT LEAST ONE BUS STILL COLLAPSES at fraction=%.2f\n', frac);
        end

        % Keep a zoomed voltage trace only for G/H/K + one boundary bus (F),
        % to keep the sweep's memory footprint reasonable across 9 runs --
        % full per-bus plots are generated only for the best candidate below.
        zoom_mask = time_vec >= ZOOM_LO & time_vec <= ZOOM_HI;
        V_zoom_partial = struct();
        for bus = [TARGET_BUSES, {'F'}]
            v_full = grab_series(so, sprintf('V_Bus_%s', bus{1}), TS, N_STEPS);
            V_zoom_partial.(bus{1}) = v_full(zoom_mask);
        end

        sweep_results(end+1) = struct('fraction', frac, 'all_safe', all_safe, ... %#ok<AGROW>
            'bus_results', bus_results, 'wall_clock_s', wall_s);

        % Stash the zoomed traces alongside (kept out of the struct array
        % definition above so all_safe/bus_results stay simple to scan).
        all_zoom_traces{fi} = V_zoom_partial; %#ok<AGROW>

        close_system(MODEL_NAME, 0);
    end

    %% SUMMARY -------------------------------------------------------------
    fprintf('\n%s\n', repmat('=', 1, 70));
    fprintf('FRACTION SWEEP SUMMARY (fraction = ShedFrac after-value; 1.0=no shed, lower=more aggressive)\n');
    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('%-10s %-8s %-40s\n', 'Fraction', 'Safe?', 'Buses still collapsing (if any)');
    best_fraction = NaN;
    best_idx = -1;
    for fi = 1:n_cand
        r = sweep_results(fi);
        collapsing = {};
        for b = 1:numel(r.bus_results)
            if r.bus_results(b).collapsed
                collapsing{end+1} = sprintf('%s@%.0fs', r.bus_results(b).bus, r.bus_results(b).onset); %#ok<AGROW>
            end
        end
        safe_str = 'NO'; if r.all_safe; safe_str = 'YES'; end
        fprintf('%-10.2f %-8s %-40s\n', r.fraction, safe_str, strjoin(collapsing, ', '));
        if r.all_safe && (isnan(best_fraction) || r.fraction > best_fraction)
            best_fraction = r.fraction;
            best_idx = fi;
        end
    end

    fprintf('\n');
    if best_idx > 0
        fprintf('BEST (mildest) FRACTION THAT KEEPS EVERY BUS SAFE: %.2f\n', best_fraction);
        fprintf('(Higher ShedFrac after-value = less load cut = less disruptive intervention.\n');
        fprintf(' This is the least aggressive shed setting, among those tested, that still\n');
        fprintf(' prevented every one of the 10 buses -- including G, H, K themselves and the\n');
        fprintf(' two boundary buses F/L that received no shedding at all -- from collapsing.)\n');
    else
        fprintf('NO CANDIDATE IN THE SWEEP KEPT EVERY BUS SAFE.\n');
        fprintf('Even the most aggressive fraction tested (%.2f) was not enough. This is an\n', min(FRACTION_CANDIDATES));
        fprintf('important negative finding -- report it as such. Consider: shedding the\n');
        fprintf('boundary buses too, an even lower fraction, or shedding earlier than the\n');
        fprintf('synchronized-at-earliest-trigger time used here.\n');
    end

    %% PLOTS -- full detail only for the best (or, if none passed, the most
    %% aggressive) candidate, to keep sweep runtime/plotting cost reasonable.
    plot_idx = best_idx; if plot_idx < 1; plot_idx = n_cand; end
    plot_frac = sweep_results(plot_idx).fraction;
    fprintf('\nBuilding detail plots for fraction=%.2f (%s) ...\n', plot_frac, ...
            ternary(best_idx > 0, 'best safe candidate', 'most aggressive candidate tested, still not fully safe'));

    collapse_color = [0.85 0.2 0.2];
    trigger_color  = [0.15 0.45 0.85];

    zoom_mask = ((0:N_STEPS-1)' * TS) >= ZOOM_LO & ((0:N_STEPS-1)' * TS) <= ZOOM_HI;
    zoom_time = ((0:N_STEPS-1)' * TS);
    zoom_time = zoom_time(zoom_mask);

    Vz = all_zoom_traces{plot_idx};

    fig1 = figure('Position', [50, 50, 1200, 800], 'Visible', 'off');
    plot_buses = [TARGET_BUSES, {'F'}];
    for i = 1:numel(plot_buses)
        bus = plot_buses{i};
        subplot(2, 2, i);
        plot(zoom_time, Vz.(bus), 'Color', [0.1 0.1 0.1], 'LineWidth', 1.2);
        hold on;
        yline(COLLAPSE_VOLTAGE_V, '--', 'Color', collapse_color, 'LineWidth', 0.8);
        xline(SHED_TIME, ':', 'Color', trigger_color, 'LineWidth', 1.4);
        r = sweep_results(plot_idx).bus_results(strcmp({sweep_results(plot_idx).bus_results.bus}, bus));
        if r.collapsed
            title_str = sprintf('%s: COLLAPSE @%.0fs', bus, r.onset);
            title_color = collapse_color;
        else
            title_str = sprintf('%s: survives (min %.0fV)', bus, r.min_v);
            title_color = [0.1 0.5 0.15];
        end
        shed_tag = ''; if ismember(bus, TARGET_BUSES); shed_tag = ' [SHED]'; end
        title([title_str shed_tag], 'Color', title_color, 'FontSize', 10);
        xlabel('t (s)'); ylabel('V');
        ylim([0 850]);
        grid on; hold off;
    end
    sgtitle(sprintf('three\\_bus\\_collapse\\_v1 closed-loop: G/H/K shed together at t=%.2fs, fraction=%.2f', ...
                     SHED_TIME, plot_frac), 'FontSize', 12, 'FontWeight', 'bold');
    per_bus_png = fullfile(OUTPUT_DIR, 'closed_loop_three_bus_collapse_v1_best_fraction_grid.png');
    saveas(fig1, per_bus_png);
    close(fig1);
    fprintf('Wrote: %s\n', per_bus_png);

    fig2 = figure('Position', [100, 100, 900, 500], 'Visible', 'off');
    fractions_plot = [sweep_results.fraction];
    safe_plot = double([sweep_results.all_safe]);
    stem(fractions_plot, safe_plot, 'filled', 'LineWidth', 1.5, 'MarkerSize', 8);
    ylim([-0.2 1.2]);
    yticks([0 1]); yticklabels({'collapse', 'all safe'});
    xlabel('ShedFrac after-value (lower = more aggressive cut)');
    title('three\_bus\_collapse\_v1: fraction sweep outcome (G/H/K shed together at t=832.95s)');
    grid on;
    sweep_png = fullfile(OUTPUT_DIR, 'closed_loop_three_bus_collapse_v1_fraction_sweep_outcomes.png');
    saveas(fig2, sweep_png);
    close(fig2);
    fprintf('Wrote: %s\n', sweep_png);

    %% SAVE ------------------------------------------------------------
    save(fullfile(OUTPUT_DIR, 'closed_loop_verification_three_bus_collapse_v1_fraction_sweep_result.mat'), ...
         'sweep_results', 'all_zoom_traces', 'zoom_time', 'FRACTION_CANDIDATES', ...
         'SHED_TIME', 'TRIGGER_G', 'TRIGGER_H', 'TRIGGER_K', 'best_fraction', 'best_idx', '-v7.3');
    fprintf('\nWrote: %s\n', fullfile(OUTPUT_DIR, 'closed_loop_verification_three_bus_collapse_v1_fraction_sweep_result.mat'));
    fprintf('\nAll outputs in: %s\n', OUTPUT_DIR);
end


%% ---- three_bus_collapse_v1's own load-profile function, reused verbatim ----

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

    idx_target   = find(ismember(BUS, TARGET_BUSES));
    idx_boundary = find(ismember(BUS, BOUNDARY_BUSES)); %#ok<NASGU>

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


%% ---- block manipulation + tap-wiring helpers (verbatim, reused across every script) ----

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


function out = ternary(cond, a, b)
    if cond; out = a; else; out = b; end
end