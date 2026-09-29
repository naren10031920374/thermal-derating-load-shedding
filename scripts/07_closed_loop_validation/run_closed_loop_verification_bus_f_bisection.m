function run_closed_loop_verification_bus_f_bisection()
% ==========================================================================
%  RUN_CLOSED_LOOP_VERIFICATION_BUS_F_BISECTION.M
%  --------------------------------------------------------------------
%  The baseline_v7/Bus F analog of
%  run_closed_loop_verification_gradual_busC_v4_bisection.m.
%
%  WHY THIS SCRIPT EXISTS: run_closed_loop_verification_bus_f_only.m only
%  ever tested ONE shed fraction on Bus F (0.30, aggressive) and it
%  passed. That answers "does shedding work here at all" but not "is 0.30
%  the CHEAPEST fraction that still works" -- same open question the
%  gradual_busC_v4 bisection answered for Bus C (turned out 0.30 was
%  over-shedding there; the true minimum was 0.40). This script answers
%  the same question for Bus F: binary-search the same 9-point fraction
%  grid, single bus.
%
%  SCOPE NOTE (2026-09-21): per the professor's guidance, this project's
%  load-shedding optimization no longer routes through DVSI -- the
%  existing ML collapse detector already identifies which bus is
%  stressed and when (that's exactly what F_TRIGGER_TIME below is), so
%  the only open question per bus is "how much to shed," answered here by
%  a real-simulation-validated search (this project's practical stand-in
%  for Paper 2's GA-PSO optimizer, given no Global Optimization Toolbox
%  license -- see the project doc). No DVSI slice, no analytical proxy.
%
%  SEARCH DIRECTION, IMPORTANT: bigger ShedFrac value = MILDER cut = less
%  power thrown away. We already know 0.30 (aggressive) is safe. The
%  interesting question is whether something milder -- a bigger fraction,
%  closer to 1.0 -- is ALSO safe. So "safe" pushes the search toward
%  bigger fractions (less shedding, less waste), "still collapses" pushes
%  it toward smaller fractions (more shedding).
%
%  WHAT STAYS IDENTICAL TO run_closed_loop_verification_bus_f_only.m (so
%  this is testing the shed AMOUNT only, not a different scenario):
%    - Load profile: build_derating_loads, seed=1, scale=1.0,
%      PMIN=20000/PMAX=100000 -- copied verbatim.
%    - Trigger time: F_TRIGGER_TIME = 3592.59s, the real held-out-bus
%      detector trigger (50.85s lead before the 3643.44s onset).
%    - Every bus except F is left completely untouched (ShedFrac=1),
%      including Bus E (the downstream bus of interest) -- same "clean
%      test" philosophy as the original script: if everything survives,
%      it's attributable to Bus F's own shed amount alone.
%
%  RUNTIME: ~3.5-4 min per sim (per the original Bus-F script's own
%  measurement) -- much faster than the three_bus_collapse_v1 or
%  gradual_busC_v4 scenarios. Worst case 4 simulations -> ~15-20 min
%  total, vs. ~30-35 min for a full 9-point hand sweep.
%
%  Run:
%    run_closed_loop_verification_bus_f_bisection
% ==========================================================================

    clc;

    %% 0. CONFIG --------------------------------------------------------
    PROJECT_ROOT = 'D:\naren\Documents\Thermal_Derating_Work_Handover\ThermalProject';
    MODEL_NAME = 'Grid_modelling_Thermal_V7_ALfix_sheddable_ramped';

    % Bus F's real, held-out-bus detector trigger -- identical
    % source/value to run_closed_loop_verification_bus_f_only.m.
    F_TRIGGER_TIME    = 3592.59;
    F_ONSET_REFERENCE = 3643.44;   % for display only, not used in the sim

    % Same 9-point grid convention as the three_bus_collapse_v1 and
    % gradual_busC_v4 bisection scripts, ascending (mild -> aggressive is
    % descending fraction).
    FRACTION_CANDIDATES = [0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90];
    KNOWN_SAFE_FRACTION  = 0.30;   % already confirmed safe by the original script

    RAMP_DURATION = 0;   % 0 = instantaneous, matches the original script

    TS   = 0.01;
    TEND = 5000;
    N_STEPS = round(TEND / TS) + 1;

    % Exact load-profile parameters from run_closed_loop_verification_bus_f_only.m --
    % must match exactly, this is what produced the real t=3643.44s collapse.
    LOAD_SEED = 1;
    GRID_LOAD_SCALE = 1.0;
    PMIN = 20000;
    PMAX = 100000;

    COLLAPSE_VOLTAGE_V = 100.0;
    MIN_CONSECUTIVE_SAMPLES = 50;

    BUS  = {'A','B','C','D','E','F','G','H','K','L'};
    CONV = {'AB','BC','CD','DE','EF','FG','GH','HK','KL','AL'};

    ZOOM_LO = 3200; ZOOM_HI = 4300;

    OUTPUT_DIR = fullfile(PROJECT_ROOT, 'model_outputs', 'thermal_derating_v7', ...
                          'closed_loop_verification_bisection');
    if exist(OUTPUT_DIR, 'dir') ~= 7; mkdir(OUTPUT_DIR); end
    addpath(PROJECT_ROOT);

    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('CLOSED-LOOP BISECTION SEARCH: baseline_v7, Bus F only\n');
    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('Bus F trigger time (real held-out-bus detector): %.2fs\n', F_TRIGGER_TIME);
    fprintf('Bus F onset (reference, unshed): %.2fs\n', F_ONSET_REFERENCE);
    fprintf('Already known safe (from the original script): fraction=%.2f\n', KNOWN_SAFE_FRACTION);
    fprintf('Candidates being binary-searched (ascending): %s\n', sprintf('%.2f ', FRACTION_CANDIDATES));
    fprintf('Worst case: 4 simulations (vs. 9 for a full hand sweep).\n');
    fprintf('Every other bus (including E, the downstream bus of interest): NO\n');
    fprintf('shedding, this whole run.\n');

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
        fprintf('Building baseline_v7 load profile (seed %d, scale %.2f) ...\n', LOAD_SEED, GRID_LOAD_SCALE);
        [~, ~, ~] = build_derating_loads(LOAD_SEED, N_STEPS, TS, BUS, GRID_LOAD_SCALE, PMIN, PMAX);

        %% Reload the model fresh each iteration (avoids stale block state)
        if bdIsLoaded(MODEL_NAME)
            close_system(MODEL_NAME, 0);
        end
        load_system(model_file);

        ramp_block = sprintf('%s/ShedFrac_F_ramp', MODEL_NAME);
        if getSimulinkBlockHandle(ramp_block) == -1
            error(['Block not found: %s\nExpected the shed-fraction ramp actuator ' ...
                   'for Bus F to already exist in this model.'], ramp_block);
        end

        wire_all_taps(MODEL_NAME, BUS, CONV);
        set_param(MODEL_NAME, 'StopTime', num2str(TEND));

        %% Reset every bus to unshed, then apply Bus F's step at this candidate
        for b = 1:numel(BUS)
            default_blk = sprintf('%s/ShedFrac_%s_default', MODEL_NAME, BUS{b});
            ensure_constant_block(MODEL_NAME, BUS{b}, default_blk, 1);
            ramp_blk = sprintf('%s/ShedFrac_%s_ramp', MODEL_NAME, BUS{b});
            if getSimulinkBlockHandle(ramp_blk) ~= -1
                set_param(ramp_blk, 'RisingSlewLimit', 'inf', 'FallingSlewLimit', '-inf');
            end
        end

        f_blk = sprintf('%s/ShedFrac_F_default', MODEL_NAME);
        set_shed_step(MODEL_NAME, 'F', f_blk, F_TRIGGER_TIME, 1, frac);
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
        zoom_time = time_vec(zoom_mask); %#ok<NASGU>

        bus_results = struct('bus', {}, 'shed', {}, 'collapsed', {}, 'onset', {}, 'min_v', {});
        V_zoom = struct();
        all_safe = true;

        for b = 1:numel(BUS)
            bus = BUS{b};
            v_full = grab_series(so, sprintf('V_Bus_%s', bus), TS, N_STEPS);
            V_zoom.(bus) = v_full(zoom_mask);

            [c, onset, min_v] = detect_collapse_summary( ...
                time_vec, v_full, COLLAPSE_VOLTAGE_V, MIN_CONSECUTIVE_SAMPLES);

            shed_str = 'no'; if strcmp(bus, 'F'); shed_str = 'yes'; end
            outcome_str = 'survives'; if c; outcome_str = sprintf('COLLAPSE@%.0fs', onset); end
            fprintf(' %s shed=%s %s min_v=%.2f\n', bus, shed_str, outcome_str, min_v);

            if c; all_safe = false; end

            bus_results(end+1) = struct('bus', bus, 'shed', strcmp(bus, 'F'), ... %#ok<AGROW>
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
                     'that means Bus F was being over-shed before, wasting power for no\n' ...
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

    fig1 = figure('Position', [50, 50, 1000, 500], 'Visible', 'off', 'Color', 'white');
    force_light_theme(fig1);
    plot_buses = {'F', 'E'};   % shed bus + its downstream bus of interest
    for i = 1:numel(plot_buses)
        bus = plot_buses{i};
        ax = subplot(1, 2, i);
        set(ax, 'Color', 'white');
        plot(zoom_time, Vz.(bus), 'Color', voltage_color, 'LineWidth', 1.6);
        hold on;
        yline(COLLAPSE_VOLTAGE_V, '--', 'Color', collapse_color, 'LineWidth', 1.0);
        xline(F_TRIGGER_TIME, ':', 'Color', trigger_color, 'LineWidth', 1.6);
        r = plot_bus_results(strcmp({plot_bus_results.bus}, bus));
        if r.collapsed
            title_str = sprintf('%s: COLLAPSE @%.0fs', bus, r.onset);
            title_color = collapse_color;
        else
            title_str = sprintf('%s: survives (min %.0fV)', bus, r.min_v);
            title_color = [0.1 0.5 0.15];
        end
        shed_tag = ''; if strcmp(bus, 'F'); shed_tag = ' [SHED]'; end
        title([title_str shed_tag], 'Color', title_color, 'FontSize', 10);
        xlabel('t (s)', 'Color', 'black'); ylabel('V', 'Color', 'black');
        ylim([0 850]);
        ax.XColor = 'black'; ax.YColor = 'black';
        ax.GridColor = grid_color; ax.GridAlpha = 0.6;
        grid on; hold off;
    end
    sgtitle(sprintf('baseline\\_v7 bisection search: Bus F shed alone at t=%.2fs, fraction=%.2f', ...
                     F_TRIGGER_TIME, plot_frac), 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'black');
    per_bus_png = fullfile(OUTPUT_DIR, 'closed_loop_bus_f_bisection_best_fraction_grid.png');
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
    xlabel(ax2, 'ShedFrac after-value on Bus F (lower = more aggressive cut)', 'Color', 'black');
    ax2.XColor = 'black'; ax2.YColor = 'black';
    ax2.GridColor = grid_color; ax2.GridAlpha = 0.6;
    title(ax2, sprintf('baseline\\_v7 bisection search: %d of %d candidates tested', numel(tested_idx), n_cand), ...
          'Color', 'black');
    grid(ax2, 'on');
    hold(ax2, 'off');
    outcomes_png = fullfile(OUTPUT_DIR, 'closed_loop_bus_f_bisection_outcomes.png');
    exportgraphics(fig2, outcomes_png, 'BackgroundColor', 'white', 'Resolution', 150);
    close(fig2);
    fprintf('Wrote: %s\n', outcomes_png);

    %% 4. SAVE ------------------------------------------------------------
    save(fullfile(OUTPUT_DIR, 'closed_loop_verification_bus_f_bisection_result.mat'), ...
         'tested_idx', 'tested_fraction', 'tested_all_safe', 'tested_bus_results', 'tested_wall_clock', ...
         'FRACTION_CANDIDATES', 'F_TRIGGER_TIME', 'F_ONSET_REFERENCE', 'KNOWN_SAFE_FRACTION', 'best_idx', '-v7.3');
    fprintf('\nWrote: %s\n', fullfile(OUTPUT_DIR, 'closed_loop_verification_bus_f_bisection_result.mat'));
    fprintf('\nAll outputs in: %s\n', OUTPUT_DIR);
    fprintf('\nSIMULATION COUNT: bisection used %d, a full hand sweep would use %d.\n', numel(tested_idx), n_cand);

    close_system(MODEL_NAME, 0);
end


%% ---- baseline_v7's own load-profile function, reused verbatim from
%% run_closed_loop_verification_bus_f_only.m ----

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
    try
        fig.Theme = 'light';   % MATLAB R2025a+ only; no-op error on older versions
    catch
    end
    fig.Color = 'white';
end
