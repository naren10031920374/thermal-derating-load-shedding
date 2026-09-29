function run_closed_loop_verification_gradual_busC_v4_bisection()
% ==========================================================================
%  RUN_CLOSED_LOOP_VERIFICATION_GRADUAL_BUSC_V4_BISECTION.M
%  --------------------------------------------------------------------
%  The gradual_busC_v4 analog of
%  run_closed_loop_verification_three_bus_collapse_v1_bisection.m.
%
%  WHY THIS SCRIPT EXISTS: run_closed_loop_verification_busC_v4_only.m
%  only ever tested ONE shed fraction on Bus C (0.30, copied over from
%  what worked for the baseline scenario's Bus F) and it passed. That
%  answers "does shedding work here at all" but not "is 0.30 the
%  CHEAPEST fraction that still works" -- Bus C might not need to be cut
%  as hard as 30%, which would mean unnecessary lost power every time
%  this scenario's detector fires for real. This script answers that
%  question directly: binary-search the same 9-point fraction grid used
%  for three_bus_collapse_v1, but for Bus C alone.
%
%  This is also a second, independent test of the bisection algorithm
%  itself (single-bus this time, vs. three_bus_collapse_v1's 3-bus group
%  cut) -- a second confirming data point for whether the safe/fail
%  pattern is a clean single cutoff (required for binary search to be
%  trustworthy) or not, per the earlier discussion about what would
%  actually break it.
%
%  SEARCH DIRECTION, IMPORTANT: bigger ShedFrac value = MILDER cut = less
%  power thrown away. We already know 0.30 (aggressive) is safe. The
%  interesting question is whether something milder -- a bigger fraction,
%  closer to 1.0 -- is ALSO safe. So "safe" pushes the search toward
%  bigger fractions (less shedding, less waste), "still collapses" pushes
%  it toward smaller fractions (more shedding), exactly like the
%  three-bus script. If the answer comes back exactly 0.30 or smaller,
%  0.30 was already close to the true minimum. If it comes back bigger
%  than 0.30, that is a real, reportable finding: Bus C was being
%  over-shed before, wasting power for no safety reason.
%
%  WHAT STAYS IDENTICAL TO THE ORIGINAL BUS-C SCRIPT (so this is testing
%  the shed AMOUNT only, not a different scenario):
%    - Load profile: build_gradual_busC_loads_v4, seed=104,
%      PMIN=2000/PMAX=220000, Bus C ramped to 200,000W, Bus B/D dropped
%      to 3,000W between t=800-2800s -- copied verbatim.
%    - Trigger time: C_TRIGGER_TIME = 1002.50s, the real cross-scenario
%      N=60s detector trigger (same source as the original script).
%    - Every bus except C is left completely untouched (ShedFrac=1) --
%      same "clean test" philosophy: if everything survives, it's
%      attributable to Bus C's own shed amount alone, including its
%      structurally stressed neighbors B and D.
%
%  PLOTTING: figure/axes backgrounds are forced to white and all
%  line/marker colors are chosen for high contrast, regardless of the
%  MATLAB desktop theme -- see force_light_theme() below. This follows
%  the standing project rule adopted after the first bisection script's
%  plots came out with invisible near-black lines on a dark background.
%
%  RUNTIME: same order of magnitude as three_bus_collapse_v1's bisection
%  run -- each sim is a single Bus C's worth of grid dynamics, not three,
%  so if anything this should be a little faster per sim (~15-20 min,
%  per the original Bus-C script's own estimate). Worst case 4
%  simulations -> roughly 60-80 minutes total, vs. running the full
%  9-point grid by hand (~2.5-3 hours).
%
%  Run:
%    run_closed_loop_verification_gradual_busC_v4_bisection
% ==========================================================================

    clc;

    %% 0. CONFIG --------------------------------------------------------
    PROJECT_ROOT = 'D:\naren\Documents\Thermal_Derating_Work_Handover\ThermalProject';
    MODEL_NAME = 'Grid_modelling_Thermal_V7_ALfix_sheddable_ramped';

    % Bus C's real, cross-scenario-generalization detector trigger --
    % identical source/value to run_closed_loop_verification_busC_v4_only.m.
    C_TRIGGER_TIME    = 1002.50;
    C_ONSET_REFERENCE = 2094.47;   % for display only, not used in the sim

    % Same 9-point grid as three_bus_collapse_v1's sweep/bisection scripts,
    % ascending (mild -> aggressive is descending fraction).
    FRACTION_CANDIDATES = [0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90];
    KNOWN_SAFE_FRACTION  = 0.30;   % already confirmed safe by the original script

    RAMP_DURATION = 0;   % 0 = instantaneous, matches the original script

    TS   = 0.01;
    TEND = 5000;
    N_STEPS = round(TEND / TS) + 1;

    % Exact load-profile parameters from generate_gradual_busC_scenario_v4.m --
    % must match exactly, this is what produced the real t=2094.47s collapse.
    SEED = 104;
    PMIN = 2000;
    PMAX = 220000;
    RAMP_START_S = 800;
    RAMP_END_S   = 2800;
    C_TARGET_W   = 200000;
    BD_TARGET_W  = 3000;

    COLLAPSE_VOLTAGE_V = 100.0;
    MIN_CONSECUTIVE_SAMPLES = 50;

    BUS  = {'A','B','C','D','E','F','G','H','K','L'};
    CONV = {'AB','BC','CD','DE','EF','FG','GH','HK','KL','AL'};

    ZOOM_LO = 500; ZOOM_HI = 3000;

    OUTPUT_DIR = fullfile(PROJECT_ROOT, 'model_outputs', 'thermal_derating_gradual_busC', ...
                          'closed_loop_verification_bisection');
    if exist(OUTPUT_DIR, 'dir') ~= 7; mkdir(OUTPUT_DIR); end
    addpath(PROJECT_ROOT);

    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('CLOSED-LOOP BISECTION SEARCH: gradual_busC_v4, Bus C only\n');
    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('Bus C trigger time (real N=60s cross-scenario detector): %.2fs\n', C_TRIGGER_TIME);
    fprintf('Bus C onset (reference, unshed): %.2fs\n', C_ONSET_REFERENCE);
    fprintf('Already known safe (from the original script): fraction=%.2f\n', KNOWN_SAFE_FRACTION);
    fprintf('Candidates being binary-searched (ascending): %s\n', sprintf('%.2f ', FRACTION_CANDIDATES));
    fprintf('Worst case: 4 simulations (vs. 9 for a full hand sweep).\n');
    fprintf('Every other bus (including B and D, both heavily stressed by this\n');
    fprintf('scenario''s own load profile): NO shedding, this whole run.\n');

    %% 1. BINARY SEARCH ---------------------------------------------------
    n_cand = numel(FRACTION_CANDIDATES);
    lo = 1; hi = n_cand; best_idx = -1;
    tested_idx = []; tested_fraction = []; tested_all_safe = [];
    tested_bus_results = {}; tested_wall_clock = []; tested_zoom = {};
    iter = 0;

    hh = dir(fullfile(PROJECT_ROOT, '**', [MODEL_NAME '.slx']));
    if isempty(hh)
        error(['Ramped sheddable model not found: %s.slx\n' ...
               'This should already exist from the baseline scenario''s ' ...
               '05_load_shedding_infrastructure / 06_gradual_ramp work.'], MODEL_NAME);
    end
    model_file = fullfile(hh(1).folder, hh(1).name);
    fprintf('\nModel: %s\n', model_file);

    while lo <= hi
        iter = iter + 1;
        mid = floor((lo + hi) / 2);
        frac = FRACTION_CANDIDATES(mid);

        fprintf('\n%s\n', repmat('-', 1, 70));
        fprintf('ITERATION %d: range=[%d,%d] (fractions %.2f..%.2f) -> testing index %d = %.2f\n', ...
                iter, lo, hi, FRACTION_CANDIDATES(lo), FRACTION_CANDIDATES(hi), mid, frac);
        fprintf('%s\n', repmat('-', 1, 70));

        %% Build the load profile -- identical to the original script
        fprintf('Building gradual_busC_v4 load profile (seed %d) ...\n', SEED);
        [~, ~, ~] = build_gradual_busC_loads_v4( ...
            SEED, N_STEPS, TS, BUS, PMIN, PMAX, RAMP_START_S, RAMP_END_S, ...
            C_TARGET_W, BD_TARGET_W);

        %% Reload the model fresh each iteration (avoids stale block state)
        if bdIsLoaded(MODEL_NAME)
            close_system(MODEL_NAME, 0);
        end
        load_system(model_file);

        ramp_block = sprintf('%s/ShedFrac_C_ramp', MODEL_NAME);
        if getSimulinkBlockHandle(ramp_block) == -1
            error(['Block not found: %s\nExpected the shed-fraction ramp actuator ' ...
                   'for Bus C to already exist in this model.'], ramp_block);
        end

        wire_all_taps(MODEL_NAME, BUS, CONV);
        set_param(MODEL_NAME, 'StopTime', num2str(TEND));

        %% Reset every bus to unshed, then apply Bus C's step at this candidate
        for b = 1:numel(BUS)
            default_blk = sprintf('%s/ShedFrac_%s_default', MODEL_NAME, BUS{b});
            ensure_constant_block(MODEL_NAME, BUS{b}, default_blk, 1);
            ramp_blk = sprintf('%s/ShedFrac_%s_ramp', MODEL_NAME, BUS{b});
            if getSimulinkBlockHandle(ramp_blk) ~= -1
                set_param(ramp_blk, 'RisingSlewLimit', 'inf', 'FallingSlewLimit', '-inf');
            end
        end

        c_blk = sprintf('%s/ShedFrac_C_default', MODEL_NAME);
        set_shed_step(MODEL_NAME, 'C', c_blk, C_TRIGGER_TIME, 1, frac);
        set_param(ramp_block, 'RisingSlewLimit', 'inf', 'FallingSlewLimit', '-inf');

        %% Run
        fprintf('Running closed-loop sim (fraction=%.2f) ...\n', frac);
        tic;
        ws = warning('off', 'all');
        so = sim(MODEL_NAME, 'ReturnWorkspaceOutputs', 'on');
        warning(ws);
        wall_clock = toc;
        fprintf('Finished in %.1f s wall clock.\n', wall_clock);

        time_vec = (0:N_STEPS-1)' * TS;
        zoom_mask = time_vec >= ZOOM_LO & time_vec <= ZOOM_HI;
        zoom_time = time_vec(zoom_mask);

        bus_results = struct('bus', {}, 'shed', {}, 'collapsed', {}, 'onset', {}, 'min_v', {});
        V_zoom = struct();
        all_safe = true;

        for b = 1:numel(BUS)
            bus = BUS{b};
            v_full = grab_series(so, sprintf('V_Bus_%s', bus), TS, N_STEPS);
            V_zoom.(bus) = v_full(zoom_mask);

            [c, onset, min_v] = detect_collapse_summary( ...
                time_vec, v_full, COLLAPSE_VOLTAGE_V, MIN_CONSECUTIVE_SAMPLES);

            shed_str = 'no'; if strcmp(bus, 'C'); shed_str = 'yes'; end
            outcome_str = 'survives'; if c; outcome_str = sprintf('COLLAPSE@%.0fs', onset); end
            fprintf(' %s shed=%s %s min_v=%.2f\n', bus, shed_str, outcome_str, min_v);

            if c; all_safe = false; end

            bus_results(end+1) = struct('bus', bus, 'shed', strcmp(bus, 'C'), ... %#ok<AGROW>
                'collapsed', c, 'onset', onset, 'min_v', min_v);
        end

        tested_idx(end+1) = mid; %#ok<AGROW>
        tested_fraction(end+1) = frac; %#ok<AGROW>
        tested_all_safe(end+1) = all_safe; %#ok<AGROW>
        tested_bus_results{end+1} = bus_results; %#ok<AGROW>
        tested_wall_clock(end+1) = wall_clock; %#ok<AGROW>
        tested_zoom{end+1} = V_zoom; %#ok<AGROW>

        if all_safe
            fprintf('-> ALL 10 BUSES SURVIVE at fraction=%.2f => try MILDER cuts next (search upper half)\n', frac);
            best_idx = mid;
            lo = mid + 1;
        else
            fprintf('-> AT LEAST ONE BUS STILL COLLAPSES at fraction=%.2f => try MORE AGGRESSIVE cuts next (search lower half)\n', frac);
            hi = mid - 1;
        end
    end

    %% 2. SUMMARY -----------------------------------------------------------
    fprintf('\n%s\n', repmat('=', 1, 70));
    fprintf('BISECTION SEARCH SUMMARY\n');
    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('%-10s %-8s %-6s %-30s\n', 'Fraction', 'Index', 'Safe?', 'Buses still collapsing (if any)');
    for i = 1:numel(tested_idx)
        r = tested_bus_results{i};
        bad = {r([r.collapsed]).bus};
        bad_str = '';
        if ~isempty(bad)
            onsets = [r([r.collapsed]).onset];
            parts = arrayfun(@(k) sprintf('%s@%.0fs', bad{k}, onsets(k)), 1:numel(bad), 'uni', 0);
            bad_str = strjoin(parts, ', ');
        end
        safe_str = 'NO'; if tested_all_safe(i); safe_str = 'YES'; end
        fprintf('%-10.2f %-8d %-6s %-30s\n', tested_fraction(i), tested_idx(i), safe_str, bad_str);
    end
    fprintf('\nSimulations run: %d (original full sweep would be: 9)\n', numel(tested_idx));

    if best_idx > 0
        fprintf('BEST (mildest) FRACTION THAT KEEPS EVERY BUS SAFE: %.2f\n', FRACTION_CANDIDATES(best_idx));
        if FRACTION_CANDIDATES(best_idx) > KNOWN_SAFE_FRACTION
            fprintf(['This is MILDER than the %.2f already used in the original script --\n' ...
                     'that means Bus C was being over-shed before, wasting power for no\n' ...
                     'safety reason. Worth reporting as a concrete finding.\n'], KNOWN_SAFE_FRACTION);
        elseif FRACTION_CANDIDATES(best_idx) == KNOWN_SAFE_FRACTION
            fprintf('Matches the %.2f already used in the original script -- that value was\n', KNOWN_SAFE_FRACTION);
            fprintf('already close to the true minimum-shedding safe point.\n');
        else
            fprintf(['This is MORE aggressive than the %.2f already used before, which is\n' ...
                     'unexpected -- the original script''s single test at 0.30 passed, so a\n' ...
                     'smaller "best" here would mean the safe/fail pattern is not a clean\n' ...
                     'single cutoff (non-monotonic). Treat that as a real finding to report,\n' ...
                     'not a bug to hide -- see the table above for exactly which candidate(s)\n' ...
                     'disagree with fraction=%.2f.\n'], KNOWN_SAFE_FRACTION, KNOWN_SAFE_FRACTION);
        end
    else
        fprintf(['NO SAFE FRACTION FOUND IN THE TESTED RANGE. This would directly contradict\n' ...
                 'the original script''s already-passed test at fraction=%.2f -- check for a\n' ...
                 'setup mismatch (trigger time, load profile, taps) before concluding shedding\n' ...
                 'does not work here.\n'], KNOWN_SAFE_FRACTION);
    end

    %% 3. PLOTS -----------------------------------------------------------
    % See force_light_theme() below and the project's standing plotting
    % rule: every figure/axes background is forced white and every
    % line/marker color is chosen for high contrast, regardless of the
    % MATLAB desktop theme.
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

    voltage_color  = [0.00 0.30 0.75];   % strong blue -- visible on white
    collapse_color = [0.85 0.20 0.20];   % red
    trigger_color  = [0.55 0.35 0.00];   % amber -- distinct from red
    grid_color     = [0.75 0.75 0.75];   % light gray, visible on white

    zoom_mask = ((0:N_STEPS-1)' * TS) >= ZOOM_LO & ((0:N_STEPS-1)' * TS) <= ZOOM_HI;
    zoom_time = ((0:N_STEPS-1)' * TS);
    zoom_time = zoom_time(zoom_mask);

    fig1 = figure('Position', [50, 50, 1500, 500], 'Visible', 'off', 'Color', 'white');
    force_light_theme(fig1);
    plot_buses = {'C', 'B', 'D'};   % shed bus + its two structurally stressed neighbors
    for i = 1:numel(plot_buses)
        bus = plot_buses{i};
        ax = subplot(1, 3, i);
        set(ax, 'Color', 'white');
        plot(zoom_time, Vz.(bus), 'Color', voltage_color, 'LineWidth', 1.6);
        hold on;
        yline(COLLAPSE_VOLTAGE_V, '--', 'Color', collapse_color, 'LineWidth', 1.0);
        xline(C_TRIGGER_TIME, ':', 'Color', trigger_color, 'LineWidth', 1.6);
        r = plot_bus_results(strcmp({plot_bus_results.bus}, bus));
        if r.collapsed
            title_str = sprintf('%s: COLLAPSE @%.0fs', bus, r.onset);
            title_color = collapse_color;
        else
            title_str = sprintf('%s: survives (min %.0fV)', bus, r.min_v);
            title_color = [0.1 0.5 0.15];
        end
        shed_tag = ''; if strcmp(bus, 'C'); shed_tag = ' [SHED]'; end
        title([title_str shed_tag], 'Color', title_color, 'FontSize', 10);
        xlabel('t (s)', 'Color', 'black'); ylabel('V', 'Color', 'black');
        ylim([0 850]);
        ax.XColor = 'black'; ax.YColor = 'black';
        ax.GridColor = grid_color; ax.GridAlpha = 0.6;
        grid on; hold off;
    end
    sgtitle(sprintf('gradual\\_busC\\_v4 bisection search: Bus C shed alone at t=%.2fs, fraction=%.2f', ...
                     C_TRIGGER_TIME, plot_frac), 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'black');
    per_bus_png = fullfile(OUTPUT_DIR, 'closed_loop_gradual_busC_v4_bisection_best_fraction_grid.png');
    exportgraphics(fig1, per_bus_png, 'BackgroundColor', 'white', 'Resolution', 150);
    close(fig1);
    fprintf('Wrote: %s\n', per_bus_png);

    % "Before vs after" plot: all 9 candidates on the x-axis, only the ones
    % the bisection search actually tested get a marker.
    fig2 = figure('Position', [100, 100, 900, 500], 'Visible', 'off', 'Color', 'white');
    force_light_theme(fig2);
    ax2 = axes(fig2);
    set(ax2, 'Color', 'white');
    hold(ax2, 'on');
    all_x = FRACTION_CANDIDATES;
    tested_mask = false(1, n_cand);
    tested_mask(tested_idx) = true;
    xlim(ax2, [min(all_x) - 0.05, max(all_x) + 0.05]);
    % Mark the fraction already known-safe from the original script, for reference.
    xline(ax2, KNOWN_SAFE_FRACTION, '--', sprintf('originally used: %.2f', KNOWN_SAFE_FRACTION), ...
          'Color', [0.4 0.4 0.4], 'LineWidth', 1.2, 'LabelVerticalAlignment', 'bottom');
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
    xlabel(ax2, 'ShedFrac after-value on Bus C (lower = more aggressive cut)', 'Color', 'black');
    ax2.XColor = 'black'; ax2.YColor = 'black';
    ax2.GridColor = grid_color; ax2.GridAlpha = 0.6;
    title(ax2, sprintf('gradual\\_busC\\_v4 bisection search: %d of %d candidates tested', numel(tested_idx), n_cand), ...
          'Color', 'black');
    grid(ax2, 'on');
    hold(ax2, 'off');
    outcomes_png = fullfile(OUTPUT_DIR, 'closed_loop_gradual_busC_v4_bisection_outcomes.png');
    exportgraphics(fig2, outcomes_png, 'BackgroundColor', 'white', 'Resolution', 150);
    close(fig2);
    fprintf('Wrote: %s\n', outcomes_png);

    %% 4. SAVE ------------------------------------------------------------
    save(fullfile(OUTPUT_DIR, 'closed_loop_verification_gradual_busC_v4_bisection_result.mat'), ...
         'tested_idx', 'tested_fraction', 'tested_all_safe', 'tested_bus_results', 'tested_wall_clock', ...
         'FRACTION_CANDIDATES', 'C_TRIGGER_TIME', 'C_ONSET_REFERENCE', 'KNOWN_SAFE_FRACTION', 'best_idx', '-v7.3');
    fprintf('\nWrote: %s\n', fullfile(OUTPUT_DIR, 'closed_loop_verification_gradual_busC_v4_bisection_result.mat'));
    fprintf('\nAll outputs in: %s\n', OUTPUT_DIR);
    fprintf('\nSIMULATION COUNT: bisection used %d, a full hand sweep would use %d.\n', numel(tested_idx), n_cand);

    close_system(MODEL_NAME, 0);
end


%% ---- gradual_busC_v4's own load-profile function, reused verbatim ----

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


%% ---- block manipulation + tap-wiring helpers (verbatim from the Bus F / Bus C scripts) ----

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
    % Recent MATLAB versions can default a new figure to whatever
    % desktop/system theme is active, including a dark theme -- which
    % silently makes the figure/axes background dark. This pins the
    % figure to a light theme + white background regardless of the
    % user's desktop theme setting, so exported plots always come out
    % readable. See the project handover doc's "Plot readability" note.
    try
        fig.Theme = 'light';   % MATLAB R2025a+ only; no-op error on older versions
    catch
        % Older MATLAB with no figure Theme property -- explicit
        % Color/'white' (set by the caller) is sufficient on its own.
    end
    fig.Color = 'white';
end
