function compare_failing_vs_safe_fraction()
% ==========================================================================
%  COMPARE_FAILING_VS_SAFE_FRACTION.M
%  --------------------------------------------------------------------
%  Only 2 scenarios, not a resweep: F-only shed fraction 0.45 (Bus E
%  collapses at t=4145.8s) versus 0.30 (Bus E survives). Both come from
%  the multi-bus sweep, but neither run captured Bus E's OWN converter
%  thermal signals (DE and EF), only F's neighbor converters (EF, FG)
%  were tapped. This run adds DE and EF's derate factor and junction
%  temperature, the two converters actually connected to Bus E, so the
%  physical difference between the failing and safe case is visible
%  directly rather than inferred from voltage alone.
%
%  QUESTION THIS ANSWERS: does 0.30 actually relieve E's thermal stress
%  in the 4000-4300s window, or does E's converter look similarly
%  stressed in both cases and 0.30 just happens to squeak by on margin?
%  The first would mean 0.30 is a real fix. The second would mean it is
%  a narrow coincidence specific to this one load scenario, and should
%  not be trusted as a general answer without more collapse scenarios.
%
%  Same STATIC-SHEDDING limitation as every other script in this set:
%  the fraction applies for the whole 5000s run, not just a warning window.
%
%  RUNTIME: 2 full 5000s sim() calls, roughly 8-9 minutes total based on
%  prior timings (~210-280s each).
%
%  Run:
%    compare_failing_vs_safe_fraction
% ==========================================================================

    clc;

    %% 0. CONFIG --------------------------------------------------------
    PROJECT_ROOT = 'D:\naren\Documents\Thermal_Derating_Work_Handover\ThermalProject';
    MODEL_NAME = 'Grid_modelling_Thermal_V7_ALfix_sheddable';

    SCENARIOS = [0.45, 0.30];   % failing case first, then the safe case

    TS   = 0.01;
    TEND = 5000;
    N_STEPS = round(TEND / TS) + 1;
    LOAD_SEED = 1;
    GRID_LOAD_SCALE = 1.0;
    PMIN = 20000;
    PMAX = 100000;

    COLLAPSE_VOLTAGE_V = 100.0;
    MIN_CONSECUTIVE_SAMPLES = 50;

    BUS  = {'A','B','C','D','E','F','G','H','K','L'};
    CONV = {'AB','BC','CD','DE','EF','FG','GH','HK','KL','AL'};

    % E's own two converters, the ones actually missing from every prior run.
    E_CONV = {'DE', 'EF'};

    OUTPUT_DIR = fullfile(PROJECT_ROOT, 'model_outputs', 'thermal_derating_v7', 'shed_failing_vs_safe_compare');
    if exist(OUTPUT_DIR, 'dir') ~= 7; mkdir(OUTPUT_DIR); end
    addpath(PROJECT_ROOT);

    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('COMPARE: F-only fraction 0.45 (fails) vs 0.30 (safe)\n');
    fprintf('Capturing Bus E''s own converters (DE, EF) this time\n');
    fprintf('%s\n', repmat('=', 1, 70));

    %% 1. Build the load profile, identical every run ---------------------
    fprintf('\nBuilding load profile (seed %d, scale %.2f) ...\n', LOAD_SEED, GRID_LOAD_SCALE);
    [~, ~, ~] = build_derating_loads(LOAD_SEED, N_STEPS, TS, BUS, GRID_LOAD_SCALE, PMIN, PMAX);

    %% 2. Locate and load the sheddable model ------------------------------
    hh = dir(fullfile(PROJECT_ROOT, '**', [MODEL_NAME '.slx']));
    if isempty(hh)
        error(['Sheddable model not found: %s.slx\n' ...
               'Run add_controllable_load_shedding.m first.'], MODEL_NAME);
    end
    model_file = fullfile(hh(1).folder, hh(1).name);
    fprintf('\nModel: %s\n', model_file);
    if bdIsLoaded(MODEL_NAME)
        close_system(MODEL_NAME, 0);
    end
    load_system(model_file);

    shed_block = sprintf('%s/ShedFrac_F_default', MODEL_NAME);
    if getSimulinkBlockHandle(shed_block) == -1
        error('Block not found: %s', shed_block);
    end

    %% 3. Wire taps ONCE, this time including E's own converters -----------
    fprintf('\nWiring signal taps (once, includes DE/EF derate + junction temp) ...\n');
    wire_all_taps(MODEL_NAME, BUS, CONV);
    derate_var_names   = tap_derate_factors(MODEL_NAME, CONV);
    junction_var_names = tap_junction_temp(MODEL_NAME, CONV);

    set_param(MODEL_NAME, 'StopTime', num2str(TEND));

    %% 4. Run both scenarios ------------------------------------------------
    time_vec = (0:N_STEPS-1)' * TS;
    n_scen = numel(SCENARIOS);

    V_E = nan(N_STEPS, n_scen);
    V_F = nan(N_STEPS, n_scen);
    DER = nan(N_STEPS, numel(E_CONV), n_scen);
    TJ  = nan(N_STEPS, numel(E_CONV), n_scen);

    labels = cell(1, n_scen);

    for s = 1:n_scen
        frac = SCENARIOS(s);
        labels{s} = sprintf('frac=%.2f', frac);
        fprintf('\n%s\n', repmat('-', 1, 70));
        fprintf('RUN %d/%d: ShedFrac_F = %.2f\n', s, n_scen, frac);
        fprintf('%s\n', repmat('-', 1, 70));

        set_param(shed_block, 'Value', num2str(frac));

        tic;
        ws = warning('off', 'all');
        so = sim(MODEL_NAME, 'ReturnWorkspaceOutputs', 'on');
        warning(ws);
        fprintf('Finished in %.1f s wall clock.\n', toc);

        V_E(:, s) = grab_series(so, 'V_Bus_E', TS, N_STEPS);
        V_F(:, s) = grab_series(so, 'V_Bus_F', TS, N_STEPS);
        for c = 1:numel(E_CONV)
            conv_idx = find(strcmp(CONV, E_CONV{c}));
            der_raw = grab_series(so, derate_var_names{conv_idx}, TS, N_STEPS);
            DER(:, c, s) = der_raw;
            TJ(:, c, s)  = grab_series(so, junction_var_names{conv_idx}, TS, N_STEPS);
        end

        [collapsed_E, onset_E, min_E] = detect_collapse_summary( ...
            time_vec, V_E(:, s), COLLAPSE_VOLTAGE_V, MIN_CONSECUTIVE_SAMPLES);
        if collapsed_E
            fprintf('  Bus E: COLLAPSES, onset t=%.1fs, min V=%.2f\n', onset_E, min_E);
        else
            fprintf('  Bus E: survives, min V=%.2f\n', min_E);
        end
    end

    %% 5. Plots -----------------------------------------------------------------
    fprintf('\nBuilding comparison plots ...\n');
    zoom_lo = 3900; zoom_hi = 4300;
    mask = time_vec >= zoom_lo & time_vec <= zoom_hi;
    colors = {[0.85 0.2 0.2], [0.2 0.65 0.2]};   % red for failing, green for safe

    fig = figure('Position', [100, 100, 1300, 900], 'Visible', 'off');

    % Bus E voltage
    subplot(3,1,1); hold on;
    for s = 1:n_scen
        plot(time_vec(mask), V_E(mask, s), 'Color', colors{s}, 'LineWidth', 1.4, ...
             'DisplayName', labels{s});
    end
    yline(COLLAPSE_VOLTAGE_V, '--k', '100V threshold');
    ylabel('Bus E voltage (V)'); title('Bus E voltage'); legend('Location', 'southwest'); grid on; hold off;

    % E's converters' derate factor
    subplot(3,1,2); hold on;
    linestyles = {'-', '--'};
    for s = 1:n_scen
        for c = 1:numel(E_CONV)
            plot(time_vec(mask), DER(mask, c, s), 'Color', colors{s}, 'LineStyle', linestyles{c}, ...
                 'LineWidth', 1.3, 'DisplayName', sprintf('%s, %s', labels{s}, E_CONV{c}));
        end
    end
    ylabel('Derate factor'); title('DE and EF converter derate factor (1.0 = full power, 0 = fully derated)');
    legend('Location', 'southwest'); grid on; hold off;

    % E's converters' junction temperature
    subplot(3,1,3); hold on;
    for s = 1:n_scen
        for c = 1:numel(E_CONV)
            plot(time_vec(mask), TJ(mask, c, s), 'Color', colors{s}, 'LineStyle', linestyles{c}, ...
                 'LineWidth', 1.3, 'DisplayName', sprintf('%s, %s', labels{s}, E_CONV{c}));
        end
    end
    yline(125, ':k', '125C derating starts');
    yline(175, ':r', '175C full derate');
    xlabel('time (s)'); ylabel('Junction temp (C)'); title('DE and EF junction temperature');
    legend('Location', 'best'); grid on; hold off;

    sgtitle('Failing (0.45, red) vs safe (0.30, green) fraction: what actually differs at Bus E');
    out_path = fullfile(OUTPUT_DIR, 'failing_vs_safe_bus_E_mechanism.png');
    saveas(fig, out_path);
    close(fig);
    fprintf('Wrote: %s\n', out_path);

    save(fullfile(OUTPUT_DIR, 'failing_vs_safe_results.mat'), ...
         'V_E', 'V_F', 'DER', 'TJ', 'time_vec', 'SCENARIOS', 'E_CONV', '-v7.3');

    fprintf('\nDone. Read failing_vs_safe_bus_E_mechanism.png:\n');
    fprintf('  - If the green (0.30) derate/junction-temp traces are clearly lower/cooler\n');
    fprintf('    than red (0.45) throughout t=3900-4300s, shedding F is genuinely relieving\n');
    fprintf('    E''s converters, and 0.30 looks like a real fix.\n');
    fprintf('  - If green and red look similar until right before red collapses, 0.30 is\n');
    fprintf('    likely just a narrow margin call, not a structural fix, and should not be\n');
    fprintf('    trusted without testing on a second, independent collapse scenario.\n');

    close_system(MODEL_NAME, 0);
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

    % run-length encode "below" to find any run >= min_run samples
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


function [load_hist, clip_hi, clip_lo] = build_derating_loads( ...
    seed, n_steps, ts, BUS, scale, pmin, pmax)
    % VERBATIM from grid_derating_dataset_5000s_v7_2.m, cross-checked
    % directly against the uploaded file this time, not reproduced from
    % memory. An earlier version of this script used a fabricated load
    % function that did not match this one, producing a different load
    % schedule and invalid collapse timing (Bus F at t~2150s instead of
    % the established t~3643s). This is the corrected, verified version.
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


function wire_all_taps(model, BUS, CONV)
    % VERBATIM from grid_derating_dataset_5000s_v7_2.m. This wires V_Bus_<bus>,
    % Bus_<bus>_Src_Pow, Bus_<bus>_Temp, and the heat-sink temps, each of
    % which already has a Goto tag in the base model. GEI is deliberately
    % NOT handled here (it has no Goto tag in this model, it is a named
    % output port on a root Divide block instead) and is not needed for
    % this sweep, since only voltage, derate factor, and junction temp are
    % used below.
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
    % VERBATIM from grid_derating_dataset_5000s_v7_2.m.
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


function var_names = tap_phase_inports(model, CONV)
    var_names = cell(1, numel(CONV));
    for i = 1:numel(CONV)
        var_names{i} = sprintf('V7PH_%s', CONV{i});
        dab_path = [model '/DAB_Model_' CONV{i}];
        blk = find_system(dab_path, 'FollowLinks', 'on', 'LookUnderMasks', 'all', ...
                          'BlockType', 'Inport', 'Name', 'phase_shift');
        if isempty(blk)
            warning('tap_phase_inports:noInport', ...
                'No phase_shift Inport found in %s, this converter will be blank.', dab_path);
            continue;
        end
        parent = get_param(blk{1}, 'Parent');
        ph = get_param(blk{1}, 'PortHandles');
        if ~isfield(ph, 'Outport') || isempty(ph.Outport)
            continue;
        end
        log_name = ['V7LOG_' var_names{i}];
        cleanup_and_add_tap(parent, log_name, var_names{i}, ph.Outport(1), model);
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
