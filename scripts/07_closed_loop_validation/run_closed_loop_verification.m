function run_closed_loop_verification()
% ==========================================================================
%  RUN_CLOSED_LOOP_VERIFICATION.M  (all-bus version)
%  --------------------------------------------------------------------
%  Pass 2 of the closed-loop shedding validation, extended to every bus:
%
%    Run simulation normally
%          |
%    Feed live signals into detector, PER BUS    (Pass 1, Python)
%          |
%    Record exact trigger time, PER BUS          (Pass 1, Python)
%          |
%    Automatically start shedding on whichever   <- THIS SCRIPT
%    buses the detector flagged
%          |
%    See whether ALL 10 buses survive             <- THIS SCRIPT
%
%  Reads the per-bus trigger times written by find_detector_trigger_time.py
%  and runs ONE simulation where every bus that got a real trigger has its
%  own shed command fire at that bus's own detector-produced timestamp.
%  Buses with no trigger (never collapse in ground truth and the
%  detector correctly stayed quiet) are left at ShedFrac=1, unshed.
%
%  PREREQUISITE: run find_detector_trigger_time.py first (all-bus
%  version). This script reads its JSON output and will error clearly if
%  it is missing.
%
%  RUNTIME: 1 full 5000s sim() call, roughly 3.5-4 minutes.
%
%  Run:
%    run_closed_loop_verification
% ==========================================================================

    clc;

    %% 0. CONFIG --------------------------------------------------------
    PROJECT_ROOT = 'D:\naren\Documents\Thermal_Derating_Work_Handover\ThermalProject';
    MODEL_NAME = 'Grid_modelling_Thermal_V7_ALfix_sheddable_ramped';

    TRIGGER_JSON = fullfile(PROJECT_ROOT, 'model_outputs', 'thermal_derating_v7', ...
                             'closed_loop_trigger', 'detector_trigger_times_all_buses.json');

    FIXED_FRACTION = 0.30;
    RAMP_DURATION = 0;   % 0 = instantaneous. Applied uniformly to every triggered bus.

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

    ZOOM_LO = 3200; ZOOM_HI = 4300;

    OUTPUT_DIR = fullfile(PROJECT_ROOT, 'model_outputs', 'thermal_derating_v7', 'closed_loop_verification');
    if exist(OUTPUT_DIR, 'dir') ~= 7; mkdir(OUTPUT_DIR); end
    addpath(PROJECT_ROOT);

    %% 1. Read the detector's per-bus trigger times -----------------------
    if ~isfile(TRIGGER_JSON)
        error(['Trigger file not found: %s\n' ...
               'Run find_detector_trigger_time.py first.'], TRIGGER_JSON);
    end
    trig = jsondecode(fileread(TRIGGER_JSON));

    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('CLOSED-LOOP VERIFICATION: detector-triggered shedding, ALL BUSES\n');
    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('%-6s %-12s %-12s %-12s\n', 'Bus', 'GT onset', 'Trigger', 'Lead time');

    trigger_time = containers.Map();
    for b = 1:numel(BUS)
        bus = BUS{b};
        if ~isfield(trig.buses, bus)
            error('Bus %s missing from trigger JSON, rerun find_detector_trigger_time.py.', bus);
        end
        entry = trig.buses.(bus);
        onset_s = 'never'; if ~isempty(entry.collapse_onset); onset_s = sprintf('%.1f', entry.collapse_onset); end
        trig_s  = 'none';  if ~isempty(entry.trigger_time);   trig_s  = sprintf('%.1f', entry.trigger_time); end
        lead_s  = '-';     if ~isempty(entry.lead_time);      lead_s  = sprintf('%.1f', entry.lead_time); end
        fprintf('%-6s %-12s %-12s %-12s\n', bus, onset_s, trig_s, lead_s);

        if ~isempty(entry.trigger_time)
            trigger_time(bus) = entry.trigger_time;
        end
    end
    fprintf('\nBuses with a shed command wired: %s\n', strjoin(keys(trigger_time), ', '));
    fprintf('Shed fraction: %.2f, ramp duration: %.0fs\n', FIXED_FRACTION, RAMP_DURATION);

    %% 2. Build the load profile, identical to every prior sweep -----------
    fprintf('\nBuilding load profile (seed %d, scale %.2f) ...\n', LOAD_SEED, GRID_LOAD_SCALE);
    [~, ~, ~] = build_derating_loads(LOAD_SEED, N_STEPS, TS, BUS, GRID_LOAD_SCALE, PMIN, PMAX);

    %% 3. Locate and load the ramped model --------------------------------
    hh = dir(fullfile(PROJECT_ROOT, '**', [MODEL_NAME '.slx']));
    if isempty(hh)
        error(['Ramped model not found: %s.slx\n' ...
               'Run add_gradual_ramp_capability.m first, then ' ...
               'compare_sheddable_vs_ramped.m to verify it.'], MODEL_NAME);
    end
    model_file = fullfile(hh(1).folder, hh(1).name);
    fprintf('\nModel: %s\n', model_file);
    if bdIsLoaded(MODEL_NAME)
        close_system(MODEL_NAME, 0);
    end
    load_system(model_file);

    %% 4. Wire taps ------------------------------------------------------
    fprintf('\nWiring signal taps ...\n');
    wire_all_taps(MODEL_NAME, BUS, CONV);
    set_param(MODEL_NAME, 'StopTime', num2str(TEND));

    %% 5. Configure every bus: shed at its own detector trigger, or none ---
    for b = 1:numel(BUS)
        bus = BUS{b};
        default_blk = sprintf('%s/ShedFrac_%s_default', MODEL_NAME, bus);
        ramp_blk = sprintf('%s/ShedFrac_%s_ramp', MODEL_NAME, bus);

        if getSimulinkBlockHandle(ramp_blk) == -1
            error(['Block not found: %s\nRun add_gradual_ramp_capability.m ' ...
                   'on the sheddable model first.'], ramp_blk);
        end

        if isKey(trigger_time, bus)
            start_t = trigger_time(bus);
            set_shed_step(MODEL_NAME, bus, default_blk, start_t, 1, FIXED_FRACTION);
            if RAMP_DURATION <= 0
                set_param(ramp_blk, 'RisingSlewLimit', 'inf', 'FallingSlewLimit', '-inf');
            else
                falling_slew = -(1 - FIXED_FRACTION) / RAMP_DURATION;
                set_param(ramp_blk, 'RisingSlewLimit', 'inf', ...
                          'FallingSlewLimit', num2str(falling_slew));
            end
        else
            ensure_constant_block(MODEL_NAME, bus, default_blk, 1);
            set_param(ramp_blk, 'RisingSlewLimit', 'inf', 'FallingSlewLimit', '-inf');
        end
    end

    %% 6. Run the single simulation -----------------------------------------
    fprintf('\nRunning closed-loop verification sim ...\n');
    tic;
    ws = warning('off', 'all');
    so = sim(MODEL_NAME, 'ReturnWorkspaceOutputs', 'on');
    warning(ws);
    fprintf('Finished in %.1f s wall clock.\n', toc);

    time_vec = (0:N_STEPS-1)' * TS;
    zoom_mask = time_vec >= ZOOM_LO & time_vec <= ZOOM_HI;
    zoom_time = time_vec(zoom_mask); %#ok<NASGU>

    fprintf('\n%s\n', repmat('=', 1, 70));
    fprintf('RESULT: ALL 10 BUSES\n');
    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('%-6s %-10s %-12s %-10s\n', 'Bus', 'Shed?', 'Outcome', 'Min V');

    bus_results = struct('bus', {}, 'shed', {}, 'collapsed', {}, 'onset', {}, 'min_v', {});
    V_zoom = struct();
    all_safe = true;

    for b = 1:numel(BUS)
        bus = BUS{b};
        v_full = grab_series(so, sprintf('V_Bus_%s', bus), TS, N_STEPS);
        V_zoom.(bus) = v_full(zoom_mask);

        [c, onset, min_v] = detect_collapse_summary( ...
            time_vec, v_full, COLLAPSE_VOLTAGE_V, MIN_CONSECUTIVE_SAMPLES);

        shed_str = 'no'; if isKey(trigger_time, bus); shed_str = 'yes'; end
        outcome_str = 'survives'; if c; outcome_str = sprintf('COLLAPSE@%.0fs', onset); end
        fprintf('%-6s %-10s %-12s %-10.2f\n', bus, shed_str, outcome_str, min_v);

        if c; all_safe = false; end

        bus_results(end+1) = struct('bus', bus, 'shed', isKey(trigger_time, bus), ... %#ok<AGROW>
            'collapsed', c, 'onset', onset, 'min_v', min_v);
    end

    fprintf('\n');
    if all_safe
        fprintf('CLOSED-LOOP TEST PASSED: every bus survives with detector-triggered\n');
        fprintf('shedding, on this one scenario.\n');
    else
        fprintf('CLOSED-LOOP TEST FAILED: at least one bus still collapses. Check\n');
        fprintf('which bus(es) above, and compare their trigger time against the\n');
        fprintf('measured cutoff from locate_cliff_exact.m / locate_gradual_lead_time.m.\n');
    end

    %% 7. Save -----------------------------------------------------------
    save(fullfile(OUTPUT_DIR, 'closed_loop_verification_all_buses_result.mat'), ...
         'bus_results', 'V_zoom', 'zoom_time', 'all_safe', 'FIXED_FRACTION', 'RAMP_DURATION', '-v7.3');
    fprintf('\nWrote: %s\n', fullfile(OUTPUT_DIR, 'closed_loop_verification_all_buses_result.mat'));

    close_system(MODEL_NAME, 0);
end


%% ---- block manipulation helpers (from test_shed_lead_time.m) ----

function ensure_constant_block(model, bus, blk_path, value)
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
