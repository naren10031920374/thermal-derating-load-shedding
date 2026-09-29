function run_closed_loop_verification_three_bus_collapse_v1_dvsi_candidate()
% ==========================================================================
%  RUN_CLOSED_LOOP_VERIFICATION_THREE_BUS_COLLAPSE_V1_DVSI_CANDIDATE.M
%  --------------------------------------------------------------------
%  Stage 06 -- validates the GA-PSO/DVSI prototype's specific candidate
%  shed vector against the real closed-loop model, ONE simulation, not a
%  search. Adapted directly from
%  run_closed_loop_verification_three_bus_collapse_v1_bisection.m -- same
%  model, same load profile, same detector trigger times, same collapse
%  detection - only the shedding itself is different:
%
%    - The real bisection script sheds G, H, and K TOGETHER by the same
%      flat fraction (its answer: 50% on all three).
%    - This script sheds G, K, and (from iteration 3 on) H too, each at
%      its own per-bus fraction.
%
%  ITERATION 1 (G=48.5%, K=5.0%, H=0%, DVSI's raw proxy numbers): PARTIAL.
%  G genuinely survived, but K still collapsed (delayed ~23s, not
%  prevented) despite the proxy predicting DVSI_after=0.515 (comfortably
%  resolved).
%
%  ITERATION 2 (G=48.5%, K=50.0%, H=0%): a follow-up "worst-case physical
%  bounds" version of the proxy (Load=plateau target, Gen=0) was tried to
%  fix K's number, but it badly overshot (recommended 93% off G, which
%  iteration 1 already showed is far more than G actually needs) --
%  assuming Load and Gen hit their individual worst values at the same
%  instant isn't realistic, so that approach was dropped. Instead this
%  iteration kept G at its VALIDATED 48.5% and bumped K up to 50% -- the
%  same flat fraction the original bisection search already proved keeps
%  K (and everyone else) up when applied to G/H/K together. RESULT: G and
%  K both survived for real. Bus H still collapsed (delayed further, to
%  t=1200s) -- this RESOLVES the cascade hypothesis (see project doc
%  Section 7/12): saving G+K alone is NOT enough to save H. DVSI never
%  flagging H pre-collapse meant "needs its own shed, unknown amount,"
%  not "safe to leave alone."
%
%  ITERATION 3 (2026-09-19, this version): keep G=48.5% and K=50.0% (both
%  validated together in iteration 2), and add H=50.0% too -- again
%  reusing the one number already proven (the original bisection's flat
%  50%), rather than guessing a new one for H.
%
%  Shed timing: kept IDENTICAL to the bisection script's SHED_TIME (the
%  earliest of the three detector trigger times, ~832.95s) for direct
%  comparability -- the ML detector's job is answering WHEN to act
%  (Section 9), regardless of which specific buses DVSI then says to cut.
%
%  RUNTIME: ~15-20 minutes wall clock (one 5000s sim).
%
%  Run:
%    run_closed_loop_verification_three_bus_collapse_v1_dvsi_candidate
% ==========================================================================

    clc;

    %% 0. CONFIG -- identical to the bisection script, except the shed vector
    PROJECT_ROOT = 'D:\naren\Documents\Thermal_Derating_Work_Handover\ThermalProject';
    MODEL_NAME = 'Grid_modelling_Thermal_V7_ALfix_sheddable_ramped';

    ONSET_G = 1081.96;  LEAD_G = 248.6;  TRIGGER_G = ONSET_G - LEAD_G;   % 833.36
    ONSET_H = 1082.05;  LEAD_H = 249.1;  TRIGGER_H = ONSET_H - LEAD_H;   % 832.95
    ONSET_K = 1108.70;  LEAD_K = 149.0;  TRIGGER_K = ONSET_K - LEAD_K;   % 959.70

    SHED_TIME = min([TRIGGER_G, TRIGGER_H, TRIGGER_K]);   % same synchronized trigger as the bisection run

    % ---- THE THING BEING TESTED (iteration 3): G and K kept at their
    % iteration-2-validated fractions, H added at the bisection's
    % known-good 50% (see header). Values are the REMAINING fraction
    % after shedding (matches this project's ShedFrac convention: 1 = no
    % shedding), so shed 48.5% -> remaining 0.515, shed 50% -> remaining
    % 0.50.
    SHED_BUSES     = {'G', 'K', 'H'};
    SHED_REMAINING = [0.515, 0.50, 0.50]; % G: 48.5% (validated), K: 50.0% (validated), H: 50.0% (iteration 3)
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

    OUTPUT_DIR = fullfile(PROJECT_ROOT, 'model_outputs', 'thermal_derating_three_bus_collapse', 'closed_loop_verification_dvsi_candidate');
    if exist(OUTPUT_DIR, 'dir') ~= 7; mkdir(OUTPUT_DIR); end
    addpath(PROJECT_ROOT);

    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('CLOSED-LOOP VALIDATION: three_bus_collapse_v1, DVSI candidate shed vector\n');
    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('Trigger times (N=60s, threshold=1e-6): G=%.2fs, H=%.2fs, K=%.2fs\n', TRIGGER_G, TRIGGER_H, TRIGGER_K);
    fprintf('SYNCHRONIZED SHED TIME (earliest of the three, same as bisection run): %.2fs\n', SHED_TIME);
    fprintf('Shed vector under test:\n');
    for i = 1:numel(SHED_BUSES)
        fprintf('  Bus %s: remaining=%.3f (shed %.1f%%)\n', SHED_BUSES{i}, SHED_REMAINING(i), 100*(1-SHED_REMAINING(i)));
    end
    fprintf('Compare against: iteration 1 (G=48.5%%, K=5.0%%, H=0%%) -- K, H collapsed.\n');
    fprintf('iteration 2 (G=48.5%%, K=50%%, H=0%%) -- G,K survived, H still collapsed.\n');
    fprintf('Real bisection answer was a flat 50%% on G, H, AND K together.\n');

    hh = dir(fullfile(PROJECT_ROOT, '**', [MODEL_NAME '.slx']));
    if isempty(hh)
        error(['Ramped sheddable model not found: %s.slx\n' ...
               'This should already exist from the baseline scenario''s ' ...
               '05_load_shedding_infrastructure / 06_gradual_ramp work.'], MODEL_NAME);
    end
    model_file = fullfile(hh(1).folder, hh(1).name);
    fprintf('\nModel: %s\n', model_file);

    %% 1. Build the load profile -- IDENTICAL to the bisection/fraction-sweep scripts
    fprintf('\nBuilding three_bus_collapse_v1 load profile (seed %d) ...\n', SEED);
    build_three_bus_collapse_loads( ...
        SEED, N_STEPS, TS, BUS, PMIN, PMAX, ...
        PREHEAT_END_S, SURGE_END_S, PLATEAU_END_S, STEPDOWN_END_S, ...
        AMBIENT_W, TARGET_PREHEAT_W, TARGET_PLATEAU_W, TARGET_RECOVERY_W, ...
        BOUNDARY_PREHEAT_W, BOUNDARY_LOW_W, BOUNDARY_RECOVERY_W, TARGET_JITTER_FRAC);

    %% 2. Load the sheddable+ramp model ------------------------------------
    if bdIsLoaded(MODEL_NAME)
        close_system(MODEL_NAME, 0);
    end
    load_system(model_file);

    for tb = SHED_BUSES
        ramp_block = sprintf('%s/ShedFrac_%s_ramp', MODEL_NAME, tb{1});
        if getSimulinkBlockHandle(ramp_block) == -1
            error(['Block not found: %s\nExpected the shed-fraction ramp actuator ' ...
                   'for Bus %s to already exist in this model.'], ramp_block, tb{1});
        end
    end

    %% 3. Wire taps ---------------------------------------------------------
    wire_all_taps(MODEL_NAME, BUS, CONV);
    set_param(MODEL_NAME, 'StopTime', num2str(TEND));

    %% 4. Reset every bus's shed actuator to "no shedding" first ------------
    for b = 1:numel(BUS)
        default_blk = sprintf('%s/ShedFrac_%s_default', MODEL_NAME, BUS{b});
        ensure_constant_block(MODEL_NAME, BUS{b}, default_blk, 1);
        ramp_blk = sprintf('%s/ShedFrac_%s_ramp', MODEL_NAME, BUS{b});
        if getSimulinkBlockHandle(ramp_blk) ~= -1
            set_param(ramp_blk, 'RisingSlewLimit', 'inf', 'FallingSlewLimit', '-inf');
        end
    end

    %% 5. G, K, and H each get shed, each at its own fraction (iteration 3) -
    for i = 1:numel(SHED_BUSES)
        tb = SHED_BUSES{i};
        remaining = SHED_REMAINING(i);
        blk = sprintf('%s/ShedFrac_%s_default', MODEL_NAME, tb);
        set_shed_step(MODEL_NAME, tb, blk, SHED_TIME, 1, remaining);
        ramp_block = sprintf('%s/ShedFrac_%s_ramp', MODEL_NAME, tb);
        if RAMP_DURATION <= 0
            set_param(ramp_block, 'RisingSlewLimit', 'inf', 'FallingSlewLimit', '-inf');
        else
            falling_slew = -(1 - remaining) / RAMP_DURATION;
            set_param(ramp_block, 'RisingSlewLimit', 'inf', 'FallingSlewLimit', num2str(falling_slew));
        end
    end

    %% 6. Run the simulation --------------------------------------------------
    fprintf('\nRunning closed-loop sim (single run, DVSI candidate vector) ...\n');
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
        is_shed = ismember(bus, SHED_BUSES);
        shed_str = 'no'; if is_shed; shed_str = 'yes'; end
        outcome_str = 'survives'; if c; outcome_str = sprintf('COLLAPSE@%.0fs', onset); end
        fprintf('  %-4s shed=%-4s %-14s min_v=%.2f\n', bus, shed_str, outcome_str, min_v);
        if c; all_safe = false; end
        bus_results(end+1) = struct('bus', bus, 'shed', is_shed, ... %#ok<AGROW>
            'collapsed', c, 'onset', onset, 'min_v', min_v);
    end

    zoom_mask = time_vec >= ZOOM_LO & time_vec <= ZOOM_HI;
    V_zoom = struct();
    % Fixed 4-bus set for plotting (G, K, H = the three under shed, F = the
    % nearest un-shed boundary bus for reference). Kept separate from
    % SHED_BUSES so adding H to SHED_BUSES in iteration 3 doesn't duplicate
    % it here and break the 2x2 subplot grid below.
    PLOT_SET = {'G', 'K', 'H', 'F'};
    for bus = PLOT_SET
        v_full = grab_series(so, sprintf('V_Bus_%s', bus{1}), TS, N_STEPS);
        V_zoom.(bus{1}) = v_full(zoom_mask);
    end

    close_system(MODEL_NAME, 0);

    %% SUMMARY ---------------------------------------------------------------
    fprintf('\n%s\n', repmat('=', 1, 70));
    if all_safe
        fprintf('RESULT: ALL 10 BUSES SURVIVE\n');
        fprintf('%s\n', repmat('=', 1, 70));
        fprintf('The iteration-3 vector (G=48.5%%, K=50.0%%, H=50.0%%) held up in the real\n');
        fprintf('closed-loop model -- all three previously-collapsing buses now survive.\n');
        fprintf('Next: consider trimming H''s (and/or G''s) shed %% downward in small steps,\n');
        fprintf('re-validating each step with a real run (the analytical proxy has been\n');
        fprintf('shown unreliable per-bus -- don''t trust it alone, see iterations 1-2).\n');
    else
        fprintf('RESULT: AT LEAST ONE BUS STILL COLLAPSES\n');
        fprintf('%s\n', repmat('=', 1, 70));
        collapsing = {};
        for b = 1:numel(bus_results)
            if bus_results(b).collapsed
                collapsing{end+1} = sprintf('%s@%.0fs', bus_results(b).bus, bus_results(b).onset); %#ok<AGROW>
            end
        end
        fprintf('Still collapsing: %s\n', strjoin(collapsing, ', '));
        fprintf('50%% on H (the bisection-proven number) was not enough here even though\n');
        fprintf('it worked for K in iteration 2 -- worth checking whether H''s deficit is\n');
        fprintf('simply larger, or whether having G/K/H all shed together changes the\n');
        fprintf('ring''s behavior enough that per-bus fractions can''t be validated fully\n');
        fprintf('independently of each other.\n');
    end

    %% PLOT --------------------------------------------------------------
    voltage_color  = [0.00 0.30 0.75];
    collapse_color = [0.85 0.20 0.20];
    trigger_color  = [0.55 0.35 0.00];
    grid_color     = [0.75 0.75 0.75];

    zoom_time = time_vec(zoom_mask);
    plot_buses = PLOT_SET;

    fig1 = figure('Position', [50, 50, 1200, 800], 'Visible', 'off', 'Color', 'white');
    force_light_theme(fig1);
    for i = 1:numel(plot_buses)
        bus = plot_buses{i};
        ax = subplot(2, 2, i);
        set(ax, 'Color', 'white');
        plot(zoom_time, V_zoom.(bus), 'Color', voltage_color, 'LineWidth', 1.6);
        hold on;
        yline(COLLAPSE_VOLTAGE_V, '--', 'Color', collapse_color, 'LineWidth', 1.0);
        xline(SHED_TIME, ':', 'Color', trigger_color, 'LineWidth', 1.6);
        r = bus_results(strcmp({bus_results.bus}, bus));
        if r.collapsed
            title_str = sprintf('%s: COLLAPSE @%.0fs', bus, r.onset);
            title_color = collapse_color;
        else
            title_str = sprintf('%s: survives (min %.0fV)', bus, r.min_v);
            title_color = [0.1 0.5 0.15];
        end
        shed_tag = ''; if ismember(bus, SHED_BUSES); shed_tag = ' [SHED]'; else; shed_tag = ' [NOT SHED]'; end
        title([title_str shed_tag], 'Color', title_color, 'FontSize', 10);
        xlabel('t (s)', 'Color', 'black'); ylabel('V', 'Color', 'black');
        ylim([0 850]);
        ax.XColor = 'black'; ax.YColor = 'black';
        ax.GridColor = grid_color; ax.GridAlpha = 0.6;
        grid on; hold off;
    end
    sgtitle(sprintf('three\\_bus\\_collapse\\_v1: iteration 3 (G=48.5%%, K=50%%, H=50%%) at t=%.2fs', SHED_TIME), ...
            'FontSize', 12, 'FontWeight', 'bold', 'Color', 'black');
    per_bus_png = fullfile(OUTPUT_DIR, 'closed_loop_three_bus_collapse_v1_dvsi_candidate_grid.png');
    exportgraphics(fig1, per_bus_png, 'BackgroundColor', 'white', 'Resolution', 150);
    close(fig1);
    fprintf('Wrote: %s\n', per_bus_png);

    %% SAVE --------------------------------------------------------------
    save(fullfile(OUTPUT_DIR, 'closed_loop_verification_three_bus_collapse_v1_dvsi_candidate_result.mat'), ...
         'bus_results', 'all_safe', 'SHED_BUSES', 'SHED_REMAINING', 'SHED_TIME', ...
         'TRIGGER_G', 'TRIGGER_H', 'TRIGGER_K', 'wall_s', '-v7.3');
    fprintf('\nWrote: %s\n', fullfile(OUTPUT_DIR, 'closed_loop_verification_three_bus_collapse_v1_dvsi_candidate_result.mat'));
    fprintf('\nAll outputs in: %s\n', OUTPUT_DIR);
end


%% ---- three_bus_collapse_v1's own load-profile function, reused verbatim ----
%% (identical to the bisection / fraction-sweep scripts)

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
    try
        fig.Theme = 'light';   % MATLAB R2025a+ only; no-op error on older versions
    catch
    end
    fig.Color = 'white';
end
