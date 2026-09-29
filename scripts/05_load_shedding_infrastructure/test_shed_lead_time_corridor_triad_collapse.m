function test_shed_lead_time_corridor_triad_collapse()
% ==========================================================================
%  TEST_SHED_LEAD_TIME_CORRIDOR_TRIAD_COLLAPSE.M
%  --------------------------------------------------------------------
%  Picks up exactly where test_shed_fraction_sweep_corridor_triad_collapse.m
%  left off. That script proved shedding Bus E+F together to fraction 0.80,
%  applied STATICALLY for the whole 5000s run, fully prevents this
%  scenario's collapse (0.90 only delays it to t=1236s; 1.00 is the natural
%  t=1149.1s collapse). But "always shed" isn't realistic -- a real system
%  only sheds once a detector raises an alarm, shortly before the actual
%  collapse. This answers the real question: how LATE can that shed start
%  and still work?
%
%  MECHANISM: ShedFrac_E_default and ShedFrac_F_default are each swapped at
%  runtime from a Constant block to a Step block (Before=1.0, After=0.80,
%  StepTime=START_TIME), same name and position, so nothing downstream
%  needs rewiring -- identical technique to the baseline project's own
%  test_shed_lead_time.m. This change is never saved to disk; the verified
%  .slx on disk is untouched.
%
%  START_TIMES bracket this scenario's actual crisis window: Phase 2 (the
%  ramp toward the near-ceiling target) runs 1000-1150s, and collapse hits
%  at t=1149.1s -- a much tighter window than the baseline's, where the
%  gap between the ramp starting and collapse was measured in hundreds of
%  seconds. Here it's undesirably short: from t=1000s (ramp starts) to
%  t=1149.1s (collapse) is only ~149 seconds. That's the whole window this
%  script has to search.
%
%  CAVEAT: a Step is an instantaneous jump, not the gradual ramp a real
%  deployed actuator would use (that refinement is 06_gradual_ramp,
%  explicitly out of scope here). This finds a BEST-CASE minimum lead time;
%  a real ramped implementation would need to start even earlier.
%
%  RUNTIME: 9 full 5000s sim() calls. Observed wall-clock for this
%  scenario's own runs (fraction sweep) was ~750-900s per run, so expect
%  roughly 2-2.25 hours total, not the ~13 min/run the baseline logged --
%  budget your session accordingly.
%
%  Run:
%    test_shed_lead_time_corridor_triad_collapse
% ==========================================================================

    clc;

    %% 0. CONFIG --------------------------------------------------------
    PROJECT_ROOT = 'D:\naren\Documents\Thermal_Derating_Work_Handover\ThermalProject';
    MODEL_NAME = 'Grid_modelling_Thermal_V7_ALfix_sheddable';

    SHED_BUSES = {'E', 'F'};           % shed together, matching the validated fraction sweep
    FIXED_FRACTION = 0.80;             % the safe fraction found by the fraction sweep
    % Brackets this scenario's actual window: Phase 2 ramp starts at 1000s,
    % natural collapse at 1149.1s. Includes 1149 explicitly (one step before
    % the known onset) since that boundary matters directly.
    START_TIMES = [0, 400, 700, 900, 1000, 1050, 1090, 1120, 1140];

    TS   = 0.01;
    TEND = 5000;
    N_STEPS = round(TEND / TS) + 1;
    SEED = 7;                          % must match generate_corridor_triad_collapse.m exactly
    PMIN = 20000;
    PMAX = 100000;

    COLLAPSE_VOLTAGE_V = 100.0;
    MIN_CONSECUTIVE_SAMPLES = 50;

    BUS  = {'A','B','C','D','E','F','G','H','K','L'};
    CONV = {'AB','BC','CD','DE','EF','FG','GH','HK','KL','AL'};

    WATCH_BUSES = {'E', 'F', 'G'};
    WATCH_CONV  = {'DE', 'EF', 'FG', 'GH'};

    OUTPUT_DIR = fullfile(PROJECT_ROOT, 'model_outputs', 'thermal_derating_v7', ...
                           'corridor_triad_collapse_shed_lead_time');
    if exist(OUTPUT_DIR, 'dir') ~= 7; mkdir(OUTPUT_DIR); end
    addpath(PROJECT_ROOT);

    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('CORRIDOR_TRIAD_COLLAPSE: LEAD-TIME SWEEP, Bus E+F shed to %.2f\n', FIXED_FRACTION);
    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('Start times to test: %s\n', mat2str(START_TIMES));
    fprintf('Natural collapse onset (no shedding): t=1149.1s\n');
    fprintf('CAVEAT: instantaneous step, not a ramp. This finds a best-case minimum.\n');

    %% 1. Locate and load the sheddable model ------------------------------
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

    %% 2. Wire taps ONCE, reused across all start times ---------------------
    fprintf('\nWiring signal taps (once, reused across all start times) ...\n');
    wire_all_taps(MODEL_NAME, BUS, CONV);
    derate_var_names = tap_derate_factors(MODEL_NAME, CONV);

    set_param(MODEL_NAME, 'StopTime', num2str(TEND));

    %% 3. Sweep -------------------------------------------------------------
    time_vec = (0:N_STEPS-1)' * TS;
    n_runs = numel(START_TIMES);

    V = struct();
    for b = 1:numel(WATCH_BUSES)
        V.(WATCH_BUSES{b}) = nan(N_STEPS, n_runs);
    end
    DER = struct();
    for c = 1:numel(WATCH_CONV)
        DER.(WATCH_CONV{c}) = nan(N_STEPS, n_runs);
    end

    results = struct('start_time', {}, 'collapsed', {}, 'onset', {}, 'min_v', {});

    for r = 1:n_runs
        start_t = START_TIMES(r);
        fprintf('\n%s\n', repmat('-', 1, 70));
        fprintf('RUN %d/%d: shedding starts at t=%.0fs (E+F to %.2f after that)\n', ...
                r, n_runs, start_t, FIXED_FRACTION);
        fprintf('%s\n', repmat('-', 1, 70));

        % Reset every bus to constant no-shedding first (safe default).
        for b = 1:numel(BUS)
            blk = sprintf('%s/ShedFrac_%s_default', MODEL_NAME, BUS{b});
            if getSimulinkBlockHandle(blk) ~= -1
                ensure_constant_block(MODEL_NAME, BUS{b}, blk, 1);
            end
        end
        % Swap E and F's blocks to time-triggered Steps.
        for i = 1:numel(SHED_BUSES)
            set_shed_step(MODEL_NAME, SHED_BUSES{i}, shed_blocks{i}, start_t, 1, FIXED_FRACTION);
        end

        % Re-populate Pload_<bus> workspace vars fresh each run: RNG state
        % must be reset to SEED every time so every run sees the identical
        % underlying load profile, exactly like the fraction sweep did.
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
            DER.(WATCH_CONV{c})(:, r) = der_raw / 100e3;
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

        results(end+1) = struct('start_time', start_t, 'collapsed', collapsed, ...
                                 'onset', onsets, 'min_v', min_vs); %#ok<AGROW>
    end

    %% 4. Summary table -------------------------------------------------------
    fprintf('\n%s\n', repmat('=', 1, 70));
    fprintf('SUMMARY (Bus E+F shed to %.2f, starting at different times)\n', FIXED_FRACTION);
    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('%-10s', 'start_t');
    for b = 1:numel(WATCH_BUSES)
        fprintf(' %-16s', sprintf('Bus %s', WATCH_BUSES{b}));
    end
    fprintf('\n');
    for r = 1:n_runs
        res = results(r);
        fprintf('%-10.0f', res.start_time);
        for b = 1:numel(WATCH_BUSES)
            if res.collapsed(b)
                fprintf(' %-16s', sprintf('COLLAPSE@%.0fs', res.onset(b)));
            else
                fprintf(' %-16s', 'survives');
            end
        end
        fprintf('\n');
    end

    both_safe = arrayfun(@(r) ~r.collapsed(1) && ~r.collapsed(2), results); % E, F both survive
    if any(both_safe)
        latest_safe = max([results(both_safe).start_time]);
        fprintf('\nLatest tested start time with BOTH Bus E and Bus F surviving: t=%.0fs\n', latest_safe);
        fprintf('(i.e. at least %.0fs of lead time before this scenario''s t=1149.1s collapse)\n', ...
                1149.1 - latest_safe);
        fprintf('\nCompare that lead-time requirement against your early-warning detector''s\n');
        fprintf('actual best lead times (up to ~90s, per step44/46) to see whether the detector\n');
        fprintf('can drive this intervention in time.\n');
    else
        fprintf('\nNo tested start time kept both Bus E and Bus F safe.\n');
        fprintf('This would mean even shedding at t=0 (statically for the whole run, matching\n');
        fprintf('the fraction sweep''s own t=0 case) fails here -- re-check against the fraction\n');
        fprintf('sweep''s own fraction=0.80 result before trusting this.\n');
    end

    %% 5. Plots -----------------------------------------------------------------
    fprintf('\nBuilding plots ...\n');
    zoom_lo = 900; zoom_hi = 1400;   % this scenario's crisis window, not the baseline's
    mask = time_vec >= zoom_lo & time_vec <= zoom_hi;

    % --- Outcome dot chart ---
    fig1 = figure('Position', [100, 100, 1200, 400], 'Visible', 'off');
    hold on;
    for r = 1:n_runs
        for b = 1:numel(WATCH_BUSES)
            color = [0.2 0.75 0.2];
            if results(r).collapsed(b); color = [0.85 0.2 0.2]; end
            plot(results(r).start_time, numel(WATCH_BUSES) - b + 1, 'o', ...
                 'MarkerSize', 16, 'MarkerFaceColor', color, 'MarkerEdgeColor', 'w');
        end
    end
    xline(1149.1, ':k', 'natural collapse onset (t=1149.1s)');
    xline(1000, '--k', 'Phase 2 ramp starts (t=1000s)');
    set(gca, 'YTick', 1:numel(WATCH_BUSES), ...
             'YTickLabel', fliplr(WATCH_BUSES), 'YLim', [0.5, numel(WATCH_BUSES) + 0.5]);
    xlabel('shedding start time (s)');
    title(sprintf('corridor triad collapse: outcome vs shedding start time (fixed fraction %.2f)', FIXED_FRACTION));
    grid on; hold off;
    saveas(fig1, fullfile(OUTPUT_DIR, 'lead_time_outcome_summary.png'));
    close(fig1);

    % --- Bus voltages across start times, zoomed ---
    fig2 = figure('Position', [100, 100, 1400, 700], 'Visible', 'off');
    colors = winter(n_runs);
    for b = 1:numel(WATCH_BUSES)
        subplot(numel(WATCH_BUSES), 1, b); hold on;
        bus = WATCH_BUSES{b};
        for r = 1:n_runs
            plot(time_vec(mask), V.(bus)(mask, r), 'Color', colors(r,:), 'LineWidth', 1.1, ...
                 'DisplayName', sprintf('start=%.0fs', START_TIMES(r)));
        end
        yline(COLLAPSE_VOLTAGE_V, '--k', '100V threshold');
        ylabel(sprintf('Bus %s (V)', bus));
        if b == 1
            title('Bus voltages vs shedding start time, zoomed to the crisis window');
        end
        if b == numel(WATCH_BUSES)
            xlabel('time (s)');
            legend('Location', 'eastoutside', 'Interpreter', 'none', 'FontSize', 7);
        end
        grid on; hold off;
    end
    saveas(fig2, fullfile(OUTPUT_DIR, 'lead_time_voltages.png'));
    close(fig2);

    % --- Derate factors across start times, zoomed ---
    fig3 = figure('Position', [100, 100, 1400, 800], 'Visible', 'off');
    for c = 1:numel(WATCH_CONV)
        subplot(numel(WATCH_CONV), 1, c); hold on;
        conv = WATCH_CONV{c};
        for r = 1:n_runs
            plot(time_vec(mask), DER.(conv)(mask, r), 'Color', colors(r,:), 'LineWidth', 1.0, ...
                 'DisplayName', sprintf('start=%.0fs', START_TIMES(r)));
        end
        ylim([0, 1.05]);
        ylabel(conv);
        if c == 1
            title('Converter derate factors vs shedding start time, zoomed');
        end
        if c == numel(WATCH_CONV)
            xlabel('time (s)');
            legend('Location', 'eastoutside', 'Interpreter', 'none', 'FontSize', 7);
        end
        grid on; hold off;
    end
    saveas(fig3, fullfile(OUTPUT_DIR, 'lead_time_derate_factors.png'));
    close(fig3);

    save(fullfile(OUTPUT_DIR, 'lead_time_sweep_results.mat'), ...
         'V', 'DER', 'time_vec', 'START_TIMES', 'FIXED_FRACTION', 'SHED_BUSES', ...
         'WATCH_BUSES', 'WATCH_CONV', 'results', '-v7.3');

    fprintf('\nWrote plots and results to: %s\n', OUTPUT_DIR);
    fprintf('\nDone. Read lead_time_outcome_summary.png first: the latest start time with both\n');
    fprintf('Bus E and Bus F dots green is the minimum required lead time before this\n');
    fprintf('scenario''s t=1149.1s collapse. Compare that number against the ~90s your\n');
    fprintf('early-warning detector actually provides.\n');

    close_system(MODEL_NAME, 0);
end


function ensure_constant_block(model, bus, blk_path, value) %#ok<INUSL>
    % Ensures blk_path is a plain Constant block with the given value.
    % If a Step block from a previous run is sitting there instead, it is
    % removed and replaced, so every non-target bus is a clean no-shed default.
    if getSimulinkBlockHandle(blk_path) == -1
        return;
    end
    bt = get_param(blk_path, 'BlockType');
    if strcmp(bt, 'Constant')
        set_param(blk_path, 'Value', num2str(value));
        return;
    end
    pos = get_param(blk_path, 'Position');
    out_line = get_param(blk_path, 'PortHandles');
    dst_line = -1;
    if isfield(out_line, 'Outport') && ~isempty(out_line.Outport)
        dst_line = get_param(out_line.Outport(1), 'Line');
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
    % Swaps blk_path (currently a Constant, or a Step from a prior run) for
    % a Step source block: outputs before_val until step_time, after_val
    % from step_time onward. Same name and position, so the single
    % downstream line is re-added identically, nothing else changes.
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


function [load_hist, meta] = build_corridor_triad_loads(seed, n_steps, ts, BUS, pmin, pmax)
% VERBATIM from generate_corridor_triad_collapse.m -- same SEED, same
% 4-phase mechanism, so this sweep reproduces the exact validated scenario.
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
