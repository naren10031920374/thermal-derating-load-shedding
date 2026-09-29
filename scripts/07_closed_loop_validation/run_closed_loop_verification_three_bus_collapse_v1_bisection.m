function run_closed_loop_verification_three_bus_collapse_v1_bisection()
% ==========================================================================
%  RUN_CLOSED_LOOP_VERIFICATION_THREE_BUS_COLLAPSE_V1_BISECTION.M
%  --------------------------------------------------------------------
%  Stage 06 load-shedding optimization -- first working implementation.
%
%  WHAT THIS IS: the exact same closed-loop setup as
%  run_closed_loop_verification_three_bus_collapse_v1_fraction_sweep.m
%  (same model, same load profile, same G/H/K synchronized-shed-at-
%  earliest-trigger design, same collapse detector) -- but the fraction
%  is now found with BINARY SEARCH over the same 9 candidate values that
%  script tested one at a time, instead of a full linear sweep.
%
%  WHY THIS WORKS: the fraction_sweep script already proved the
%  safe/fail pattern across the 9 candidates is monotonic --
%    0.10 0.20 0.30 0.40 0.50  -> all safe
%    0.60 0.70 0.80 0.90       -> all fail
%  i.e. sorted ascending, it's a run of TRUE followed by a run of FALSE
%  with no flip-flopping. That is exactly the precondition binary search
%  needs. This script does NOT assume that result -- it re-derives it by
%  actually running sim() at each fraction binary search asks for -- but
%  because the pattern is monotonic, it should reach the same answer
%  (0.50) in far fewer simulations than the original 9.
%
%  ALGORITHM: classic "find the rightmost TRUE in a sorted TRUE/FALSE
%  array" binary search over FRACTION_CANDIDATES (ascending):
%    lo=1, hi=9, best_idx=-1
%    while lo <= hi:
%        mid = floor((lo+hi)/2)
%        run the closed-loop sim at FRACTION_CANDIDATES(mid)
%        if all 10 buses survive: best_idx = mid; lo = mid + 1   (search milder cuts)
%        else:                                    hi = mid - 1   (search more aggressive cuts)
%    best fraction = FRACTION_CANDIDATES(best_idx)
%
%  EVERYTHING ELSE (model, load profile, actuator wiring, collapse
%  detection, helper functions) is reused VERBATIM from the fraction-
%  sweep script -- only the loop that decides which fraction to try next
%  has changed. See that script's header comments for the full design
%  rationale (shed timing, load profile, etc.) if needed.
%
%  RUNTIME: each full 5000s sim takes ~15-20 minutes wall clock (same as
%  every other script in this project). Binary search over 9 sorted
%  candidates converges in AT MOST 4 simulations (ceil(log2(9))), often
%  fewer -- budget roughly 45-80 minutes total, vs. ~2.5-3 hours for the
%  original 9-point sweep.
%
%  STATUS: this is a first, basic implementation -- the goal right now
%  is to confirm it converges on the same 0.50 answer the manual sweep
%  already found, in fewer runs. Per-bus independent search (finding
%  G, H, and K's own individual minimums instead of one shared fraction
%  for all three) is the natural next step, not attempted here yet.
%
%  Run:
%    run_closed_loop_verification_three_bus_collapse_v1_bisection
% ==========================================================================

    clc;

    %% 0. CONFIG -- identical to the fraction-sweep script ----------------
    PROJECT_ROOT = 'D:\naren\Documents\Thermal_Derating_Work_Handover\ThermalProject';
    MODEL_NAME = 'Grid_modelling_Thermal_V7_ALfix_sheddable_ramped';

    ONSET_G = 1081.96;  LEAD_G = 248.6;  TRIGGER_G = ONSET_G - LEAD_G;   % 833.36
    ONSET_H = 1082.05;  LEAD_H = 249.1;  TRIGGER_H = ONSET_H - LEAD_H;   % 832.95
    ONSET_K = 1108.70;  LEAD_K = 149.0;  TRIGGER_K = ONSET_K - LEAD_K;   % 959.70

    SHED_TIME = min([TRIGGER_G, TRIGGER_H, TRIGGER_K]);   % synchronized at earliest trigger (Bus H's)

    TARGET_BUSES   = {'G','H','K'};     % shed together at SHED_TIME
    BOUNDARY_BUSES = {'F','L'};         % NOT shed -- clean-test philosophy

    % SAME 9 candidates as the original sweep, ascending this time because
    % binary search needs a sorted array to search over. Index 1 = most
    % aggressive cut (0.10), index 9 = mildest (0.90).
    FRACTION_CANDIDATES = [0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90];
    RAMP_DURATION = 0;   % instantaneous shed, same default as prior closed-loop tests

    TS   = 0.01;
    TEND = 5000;
    N_STEPS = round(TEND / TS) + 1;

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

    ZOOM_LO = 500; ZOOM_HI = 2000;

    OUTPUT_DIR = fullfile(PROJECT_ROOT, 'model_outputs', 'thermal_derating_three_bus_collapse', 'closed_loop_verification_bisection');
    if exist(OUTPUT_DIR, 'dir') ~= 7; mkdir(OUTPUT_DIR); end
    addpath(PROJECT_ROOT);

    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('CLOSED-LOOP BISECTION SEARCH: three_bus_collapse_v1, G/H/K shed together\n');
    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('Trigger times (N=60s, threshold=1e-6): G=%.2fs, H=%.2fs, K=%.2fs\n', TRIGGER_G, TRIGGER_H, TRIGGER_K);
    fprintf('SYNCHRONIZED SHED TIME (earliest of the three): %.2fs\n', SHED_TIME);
    fprintf('Buses shed together at that time: %s\n', strjoin(TARGET_BUSES, ', '));
    fprintf('Boundary buses (F, L): NOT shed -- clean-test philosophy.\n');
    fprintf('Candidates being binary-searched (ascending): ');
    fprintf('%.2f  ', FRACTION_CANDIDATES); fprintf('\n');
    fprintf('Worst case: %d simulations (vs. 9 for the original full sweep).\n', ceil(log2(numel(FRACTION_CANDIDATES)+1)));

    hh = dir(fullfile(PROJECT_ROOT, '**', [MODEL_NAME '.slx']));
    if isempty(hh)
        error(['Ramped sheddable model not found: %s.slx\n' ...
               'This should already exist from the baseline scenario''s ' ...
               '05_load_shedding_infrastructure / 06_gradual_ramp work.'], MODEL_NAME);
    end
    model_file = fullfile(hh(1).folder, hh(1).name);
    fprintf('\nModel: %s\n', model_file);

    %% MAIN BISECTION LOOP --------------------------------------------------
    n_cand = numel(FRACTION_CANDIDATES);
    lo = 1; hi = n_cand; best_idx = -1;

    tested_idx        = [];
    tested_fraction    = [];
    tested_all_safe    = [];
    tested_bus_results = {};
    tested_wall_clock  = [];
    tested_zoom        = {};

    iter = 0;
    while lo <= hi
        iter = iter + 1;
        mid = floor((lo + hi) / 2);
        frac = FRACTION_CANDIDATES(mid);

        fprintf('\n%s\n', repmat('-', 1, 70));
        fprintf('ITERATION %d: range=[%d,%d] (fractions %.2f..%.2f) -> testing index %d = %.2f\n', ...
                iter, lo, hi, FRACTION_CANDIDATES(lo), FRACTION_CANDIDATES(hi), mid, frac);
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
            fprintf('  -> ALL 10 BUSES SURVIVE at fraction=%.2f  =>  try MILDER cuts next (search upper half)\n', frac);
        else
            fprintf('  -> AT LEAST ONE BUS STILL COLLAPSES at fraction=%.2f  =>  try MORE AGGRESSIVE cuts next (search lower half)\n', frac);
        end

        zoom_mask = time_vec >= ZOOM_LO & time_vec <= ZOOM_HI;
        V_zoom_partial = struct();
        for bus = [TARGET_BUSES, {'F'}]
            v_full = grab_series(so, sprintf('V_Bus_%s', bus{1}), TS, N_STEPS);
            V_zoom_partial.(bus{1}) = v_full(zoom_mask);
        end

        tested_idx(end+1)         = mid;                 %#ok<AGROW>
        tested_fraction(end+1)    = frac;                 %#ok<AGROW>
        tested_all_safe(end+1)    = all_safe;              %#ok<AGROW>
        tested_bus_results{end+1} = bus_results;           %#ok<AGROW>
        tested_wall_clock(end+1)  = wall_s;                %#ok<AGROW>
        tested_zoom{end+1}        = V_zoom_partial;        %#ok<AGROW>

        close_system(MODEL_NAME, 0);

        %% 7. Update the search range based on this result -------------------
        if all_safe
            best_idx = mid;
            lo = mid + 1;
        else
            hi = mid - 1;
        end
    end

    %% SUMMARY -------------------------------------------------------------
    fprintf('\n%s\n', repmat('=', 1, 70));
    fprintf('BISECTION SEARCH SUMMARY\n');
    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('%-10s %-8s %-8s %-40s\n', 'Fraction', 'Index', 'Safe?', 'Buses still collapsing (if any)');
    for k = 1:numel(tested_idx)
        r = tested_bus_results{k};
        collapsing = {};
        for b = 1:numel(r)
            if r(b).collapsed
                collapsing{end+1} = sprintf('%s@%.0fs', r(b).bus, r(b).onset); %#ok<AGROW>
            end
        end
        safe_str = 'NO'; if tested_all_safe(k); safe_str = 'YES'; end
        fprintf('%-10.2f %-8d %-8s %-40s\n', tested_fraction(k), tested_idx(k), safe_str, strjoin(collapsing, ', '));
    end

    fprintf('\nSimulations run: %d  (original full sweep: %d)\n', numel(tested_idx), n_cand);
    if best_idx > 0
        best_fraction = FRACTION_CANDIDATES(best_idx);
        fprintf('BEST (mildest) FRACTION THAT KEEPS EVERY BUS SAFE: %.2f\n', best_fraction);
        fprintf('(This should match the fraction-sweep script''s answer of 0.50 --\n');
        fprintf(' if it does not, the monotonicity assumption behind this search does not\n');
        fprintf(' hold as cleanly as the sweep data suggested, and that itself is a real\n');
        fprintf(' finding worth reporting, not a bug to hide.)\n');
    else
        best_fraction = NaN;
        fprintf('NO CANDIDATE TESTED KEPT EVERY BUS SAFE.\n');
        fprintf('Every fraction the search tried still resulted in a collapse somewhere.\n');
    end

    %% PLOTS -----------------------------------------------------------------
    if best_idx > 0
        plot_pos = find(tested_idx == best_idx, 1, 'last');
        plot_frac = tested_fraction(plot_pos);
        plot_bus_results = tested_bus_results{plot_pos};
        Vz = tested_zoom{plot_pos};
        plot_label = 'best safe candidate found';
    else
        plot_pos = numel(tested_idx);
        plot_frac = tested_fraction(plot_pos);
        plot_bus_results = tested_bus_results{plot_pos};
        Vz = tested_zoom{plot_pos};
        plot_label = 'last candidate tested, still not fully safe';
    end
    fprintf('\nBuilding detail plots for fraction=%.2f (%s) ...\n', plot_frac, plot_label);

    % NOTE on colors/background: MATLAB can default new figures to whatever
    % desktop theme is active (including a dark theme on recent versions),
    % which silently turns the figure/axes background dark. Combined with a
    % near-black line color that is invisible on a dark background. Every
    % color below is therefore set explicitly, and force_light_theme() +
    % exportgraphics(...,'BackgroundColor','white') at the end pin the
    % exported PNG to a white background regardless of desktop theme.
    voltage_color  = [0.00 0.30 0.75];   % strong blue -- visible on white
    collapse_color = [0.85 0.20 0.20];   % red
    trigger_color  = [0.55 0.35 0.00];   % amber -- distinct from red
    grid_color     = [0.75 0.75 0.75];   % light gray, visible on white

    zoom_mask = ((0:N_STEPS-1)' * TS) >= ZOOM_LO & ((0:N_STEPS-1)' * TS) <= ZOOM_HI;
    zoom_time = ((0:N_STEPS-1)' * TS);
    zoom_time = zoom_time(zoom_mask);

    fig1 = figure('Position', [50, 50, 1200, 800], 'Visible', 'off', 'Color', 'white');
    force_light_theme(fig1);
    plot_buses = [TARGET_BUSES, {'F'}];
    for i = 1:numel(plot_buses)
        bus = plot_buses{i};
        ax = subplot(2, 2, i);
        set(ax, 'Color', 'white');
        plot(zoom_time, Vz.(bus), 'Color', voltage_color, 'LineWidth', 1.6);
        hold on;
        yline(COLLAPSE_VOLTAGE_V, '--', 'Color', collapse_color, 'LineWidth', 1.0);
        xline(SHED_TIME, ':', 'Color', trigger_color, 'LineWidth', 1.6);
        r = plot_bus_results(strcmp({plot_bus_results.bus}, bus));
        if r.collapsed
            title_str = sprintf('%s: COLLAPSE @%.0fs', bus, r.onset);
            title_color = collapse_color;
        else
            title_str = sprintf('%s: survives (min %.0fV)', bus, r.min_v);
            title_color = [0.1 0.5 0.15];
        end
        shed_tag = ''; if ismember(bus, TARGET_BUSES); shed_tag = ' [SHED]'; end
        title([title_str shed_tag], 'Color', title_color, 'FontSize', 10);
        xlabel('t (s)', 'Color', 'black'); ylabel('V', 'Color', 'black');
        ylim([0 850]);
        ax.XColor = 'black'; ax.YColor = 'black';
        ax.GridColor = grid_color; ax.GridAlpha = 0.6;
        grid on; hold off;
    end
    sgtitle(sprintf('three\\_bus\\_collapse\\_v1 bisection search: G/H/K shed together at t=%.2fs, fraction=%.2f', ...
                     SHED_TIME, plot_frac), 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'black');
    per_bus_png = fullfile(OUTPUT_DIR, 'closed_loop_three_bus_collapse_v1_bisection_best_fraction_grid.png');
    exportgraphics(fig1, per_bus_png, 'BackgroundColor', 'white', 'Resolution', 150);
    close(fig1);
    fprintf('Wrote: %s\n', per_bus_png);

    % "Before vs after" plot: all 9 candidates on the x-axis, but only the
    % ones the bisection search actually tested get a marker -- makes the
    % simulation-count savings visually obvious next to the original sweep.
    fig2 = figure('Position', [100, 100, 900, 500], 'Visible', 'off', 'Color', 'white');
    force_light_theme(fig2);
    ax2 = axes(fig2);
    set(ax2, 'Color', 'white');
    hold(ax2, 'on');
    all_x = FRACTION_CANDIDATES;
    tested_mask = false(1, n_cand);
    tested_mask(tested_idx) = true;
    xlim(ax2, [min(all_x) - 0.05, max(all_x) + 0.05]); % fixes axis scale w/o a hidden dummy line
    for k = 1:n_cand
        if tested_mask(k)
            outcome = tested_all_safe(tested_idx == k);
            outcome = outcome(1);
            if outcome
                plot(ax2, all_x(k), 1, 'o', 'MarkerSize', 10, 'MarkerFaceColor', [0.15 0.55 0.2], 'MarkerEdgeColor', 'k');
            else
                plot(ax2, all_x(k), 0, 'o', 'MarkerSize', 10, 'MarkerFaceColor', [0.75 0.15 0.15], 'MarkerEdgeColor', 'k');
            end
        else
            plot(ax2, all_x(k), 0.5, 'x', 'MarkerSize', 8, 'LineWidth', 1.5, 'Color', [0.45 0.45 0.45]);
        end
    end
    ylim(ax2, [-0.3 1.3]);
    yticks(ax2, [0 0.5 1]); yticklabels(ax2, {'collapse', 'not tested', 'all safe'});
    xlabel(ax2, 'ShedFrac after-value (lower = more aggressive cut)', 'Color', 'black');
    ax2.XColor = 'black'; ax2.YColor = 'black';
    ax2.GridColor = grid_color; ax2.GridAlpha = 0.6;
    title(ax2, sprintf('three\\_bus\\_collapse\\_v1 bisection search: %d of %d candidates tested', numel(tested_idx), n_cand), ...
          'Color', 'black');
    grid(ax2, 'on');
    hold(ax2, 'off');
    sweep_png = fullfile(OUTPUT_DIR, 'closed_loop_three_bus_collapse_v1_bisection_outcomes.png');
    exportgraphics(fig2, sweep_png, 'BackgroundColor', 'white', 'Resolution', 150);
    close(fig2);
    fprintf('Wrote: %s\n', sweep_png);

    %% SAVE ------------------------------------------------------------
    save(fullfile(OUTPUT_DIR, 'closed_loop_verification_three_bus_collapse_v1_bisection_result.mat'), ...
         'tested_idx', 'tested_fraction', 'tested_all_safe', 'tested_bus_results', 'tested_wall_clock', ...
         'FRACTION_CANDIDATES', 'SHED_TIME', 'TRIGGER_G', 'TRIGGER_H', 'TRIGGER_K', ...
         'best_idx', 'best_fraction', '-v7.3');
    fprintf('\nWrote: %s\n', fullfile(OUTPUT_DIR, 'closed_loop_verification_three_bus_collapse_v1_bisection_result.mat'));
    fprintf('\nAll outputs in: %s\n', OUTPUT_DIR);
    fprintf('\nSIMULATION COUNT: bisection used %d, original sweep used %d.\n', numel(tested_idx), n_cand);
end


%% ---- three_bus_collapse_v1's own load-profile function, reused verbatim ----
%% (identical to run_closed_loop_verification_three_bus_collapse_v1_fraction_sweep.m)

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


function force_light_theme(fig)
    % Recent MATLAB versions can default a new figure to whatever desktop
    % theme is active, including a dark theme -- which silently makes the
    % figure/axes background dark. Combined with any near-black plot line,
    % that makes the line invisible once exported. This pins the figure to
    % a light theme + white background regardless of the user's desktop
    % theme setting, so exported plots always come out readable.
    try
        fig.Theme = 'light';   % MATLAB R2025a+ only; no-op error on older versions
    catch
        % Older MATLAB with no figure Theme property -- explicit
        % Color/'white' (set by the caller) is sufficient on its own.
    end
    fig.Color = 'white';
end