function step36_generate_gradual_load_dataset_v4()
% ==========================================================================
%  GRADUAL LOAD SCENARIO - DATASET GENERATION (v8, "kicker" experiment,
%  earlier kick to allow thermal runway)
%  --------------------------------------------------------------------
%  Same model, tap-wiring, config, extraction and health-check as the
%  proven pilot generator (grid_derating_dataset_5000s_v7_2.m). Only
%  build_derating_loads_gradual changes.
%
%  WHAT'S NEW IN v8 vs v7
%    v7 added a sudden step at t=4000s, sized at the real pilot's proven
%    collapse magnitude, to test whether transition SPEED (not heat
%    magnitude or duration) is what this model needs to cross 175 C.
%    The result came back numerically IDENTICAL to v6's health-check
%    table, to two decimal places, on every converter. That's not a
%    coincidence: rng(seed) reproduces the exact same noise sequence up
%    through t=4000s (nothing before the kick changed), so the recorded
%    "Tj max" is almost certainly the overshoot spike from the initial
%    ramp around t=600-1000s, not anything the kicker produced. The
%    kicker only had 1000 seconds (t=4000 to t=5000, sim end) to show an
%    effect, and converters have thermal mass -- Tj climbs over some
%    RC-like time constant, it doesn't jump instantly. Given KL/AL never
%    fully settled even over the 2000-4000s ramps used in v3-v6, 1000
%    seconds of runway after a sudden kick may simply not be enough to
%    register a new peak. The experiment likely never finished playing
%    out, so this was inconclusive, not a real result yet.
%
%    v8 makes exactly one change: the kick moves from t=4000s to
%    t=1500s, giving it ~3500 seconds of runway (still comfortably after
%    the initial ramp settles at t=600s) before the simulation ends at
%    t=5000s, enough for the thermal response to actually show whether
%    Tj keeps climbing through 175 C or plateaus below it.
% ==========================================================================

    clc; close all;

    %% 0. CONFIG ------------------------------------------------------------
    PROJECT_ROOT = 'D:\naren\Documents\Thermal_Derating_Work_Handover\ThermalProject';

    MODEL_NAME = 'Grid_modelling_Thermal_V7_ALfix';

    TS         = 0.01;
    TEND       = 5000;
    N_STEPS    = round(TEND / TS) + 1;
    LOAD_SEED  = 1;

    PMIN = 20000;
    PMAX = 100000;

    BUS  = {'A','B','C','D','E','F','G','H','K','L'};
    CONV = {'AB','BC','CD','DE','EF','FG','GH','HK','KL','AL'};
    NB   = numel(BUS);
    NC   = numel(CONV);

    OUTPUT_DIR = fullfile(PROJECT_ROOT, 'model_outputs', 'thermal_derating_gradual');
    if exist(OUTPUT_DIR, 'dir') ~= 7; mkdir(OUTPUT_DIR); end
    addpath(PROJECT_ROOT);

    fprintf('\n===== GRADUAL LOAD SCENARIO (v8, sudden kicker at t=1500s), %.0f s =====\n', TEND);

    %% 1. Build the load profile ---------------------------------------------
    fprintf('Building gradual load profile with late sudden kicker (seed %d) ...\n', LOAD_SEED);
    [load_hist, clip_hi, clip_lo] = build_derating_loads_gradual( ...
        LOAD_SEED, N_STEPS, TS, BUS, PMIN, PMAX);

    fprintf('\n--- clipping check ---\n');
    for i = 1:NB
        fprintf('  Bus %s: %5.1f%% at Pmax, %5.1f%% at Pmin\n', ...
            BUS{i}, clip_hi(i), clip_lo(i));
    end

    %% 2. Locate and load the fixed V7 model -------------------------------------
    hh = dir(fullfile(PROJECT_ROOT, '**', [MODEL_NAME '.slx']));
    if isempty(hh)
        error('Model not found under project root: %s.slx', MODEL_NAME);
    end
    model_file = fullfile(hh(1).folder, hh(1).name);
    fprintf('\nModel: %s\n', model_file);
    if bdIsLoaded(MODEL_NAME)
        close_system(MODEL_NAME, 0);
    end
    load_system(model_file);

    %% 3. Wire taps ---------------------------------------------------------
    fprintf('\nWiring signal taps ...\n');
    wire_all_taps(MODEL_NAME, BUS, CONV);
    gei_var_names       = tap_gei_signals(MODEL_NAME, BUS);
    phase_var_names     = tap_phase_inports(MODEL_NAME, CONV);
    derate_var_names    = tap_derate_factors(MODEL_NAME, CONV);
    junction_var_names  = tap_junction_temp(MODEL_NAME, CONV);

    %% 4. Run, single sim() call for the full duration --------------------------
    set_param(MODEL_NAME, 'StopTime', num2str(TEND));
    fprintf('\nRunning %.0f s, single sim() call ...\n', TEND);
    tic;
    ws = warning('off', 'all');
    so = sim(MODEL_NAME, 'ReturnWorkspaceOutputs', 'on');
    warning(ws);
    wall_s = toc;
    fprintf('Finished in %.1f s wall clock (%.1f min).\n', wall_s, wall_s/60);

    %% 5. Extract signals onto the TS grid ---------------------------------------
    time_vec = (0:N_STEPS-1)' * TS;

    V   = nan(N_STEPS, NB);
    P   = nan(N_STEPS, NB);
    GEI = nan(N_STEPS, NB);
    T   = nan(N_STEPS, NB);
    HS  = nan(N_STEPS, NC);
    PH  = nan(N_STEPS, NC);
    DER = nan(N_STEPS, NC);
    TJ  = nan(N_STEPS, NC);

    for i = 1:NB
        V(:,i)   = grab_series(so, sprintf('V_Bus_%s', BUS{i}),        TS, N_STEPS);
        P(:,i)   = grab_series(so, sprintf('Bus_%s_Src_Pow', BUS{i}),  TS, N_STEPS);
        GEI(:,i) = grab_series(so, gei_var_names{i},                    TS, N_STEPS);
        T(:,i)   = grab_series(so, sprintf('Bus_%s_Temp', BUS{i}),     TS, N_STEPS);
    end

    hs_tags = {'DAB_AB_H_Temp','DAB_BC_H_Temp','DAB_CD_H_Temp', ...
               'DAB_DE_H_Temp','DAB_EF_H_Temp','DAB_FG_H_Temp', ...
               'DAB_GH_Temp',  'DAB_HK_Temp',  'DAB_KL_Temp', ...
               'DAB_LA_Temp'};
    for j = 1:NC
        HS(:,j)  = grab_series(so, hs_tags{j}, TS, N_STEPS);
        PH(:,j)  = grab_series(so, phase_var_names{j}, TS, N_STEPS);
        raw_pos  = grab_series(so, derate_var_names{j}, TS, N_STEPS);
        DER(:,j) = raw_pos / 100e3;
        TJ(:,j)  = grab_series(so, junction_var_names{j}, TS, N_STEPS);
    end

    %% 6. Health check ----------------------------------------------------------
    fprintf('\n--- HEALTH CHECK ---\n');
    fprintf('%-4s %-10s %-10s %-10s %-10s %-14s %-10s\n', ...
        'CONV', 'HS max', 'Tj max', 'Tj>125?', 'Tj>175?', 'Derate min', 'Derated?');
    for j = 1:NC
        hcol = HS(:,j); hcol = hcol(~isnan(hcol));
        tcol = TJ(:,j); tcol = tcol(~isnan(tcol));
        dcol = DER(:,j); dcol = dcol(~isnan(dcol));
        if isempty(hcol) || isempty(tcol)
            fprintf('%-4s  (missing heat-sink or junction data, check taps)\n', CONV{j});
            continue;
        end
        derated = 'no';
        if ~isempty(dcol) && min(dcol) < 0.99
            derated = 'YES';
        end
        over125 = 'no'; if max(tcol) > 125; over125 = 'YES'; end
        over175 = 'no'; if max(tcol) > 175; over175 = 'YES'; end
        fprintf('%-4s %8.2f C %8.2f C %-10s %-10s %12.3f  %-10s\n', ...
            CONV{j}, max(hcol), max(tcol), over125, over175, min(dcol), derated);
    end

    n_derated = sum(arrayfun(@(j) any(DER(:,j) < 0.99), 1:NC));
    n_over175 = sum(arrayfun(@(j) any(TJ(:,j) > 175), 1:NC));
    fprintf('\n%d of %d converters showed derate factor below 1.0 at some point.\n', n_derated, NC);
    fprintf('%d of %d converters crossed 175 C (junction temperature).\n', n_over175, NC);

    %% 7. Assemble and write the dataset -----------------------------------------
    fprintf('\nWriting dataset ...\n');
    out = table();
    out.time = time_vec;
    for i = 1:NB
        out.(sprintf('V_Bus_%s',            BUS{i})) = V(:,i);
        out.(sprintf('Bus_%s_Src_Pow',       BUS{i})) = P(:,i);
        out.(sprintf('GEI_%s',               BUS{i})) = GEI(:,i);
        out.(sprintf('Bus_%s_Temp',          BUS{i})) = T(:,i);
        out.(sprintf('CommandedLoad_kW_%s',  BUS{i})) = load_hist(:,i) / 1000;
    end
    for j = 1:NC
        out.(sprintf('HeatSinkTemp_C_%s',    CONV{j})) = HS(:,j);
        out.(sprintf('JunctionTemp_C_%s',    CONV{j})) = TJ(:,j);
        out.(sprintf('Phase_%s_cmd_deg',     CONV{j})) = PH(:,j);
        out.(sprintf('DAB_%s_Derate_Factor', CONV{j})) = DER(:,j);
    end

    out_csv = fullfile(OUTPUT_DIR, 'thermal_derating_gradual_load_5000s.csv');
    writetable(out, out_csv);
    fprintf('Wrote: %s\n', out_csv);
    d = dir(out_csv);
    fprintf('File size: %.1f MB\n', d.bytes / 1e6);

    bdclose(MODEL_NAME);
    fprintf('Simulink model closed.\n');
end


%% ============================== load profile =============================
function [load_hist, clip_hi, clip_lo] = build_derating_loads_gradual( ...
    seed, n_steps, ts, BUS, pmin, pmax)

    rng(seed);
    t = (0:n_steps-1)' * ts;
    NB = numel(BUS);
    base_load = [35000 30000 25000 32000 28000 24000 30000 26000 22000 34000];
    grid_slow = 25000 * sin(2*pi*t/1800);
    grid_med  = 15000 * sin(2*pi*t/700 + 0.4);

    % Same gradual ramp-in as v6: light until 500s, ramps 500-600s, holds.
    t_light_end = 500;
    t_ramp_end  = 600;
    ramp_frac = zeros(n_steps, 1);
    idx2 = (t >= t_light_end) & (t < t_ramp_end);
    idx3 = t >= t_ramp_end;
    ramp_frac(idx2) = (t(idx2) - t_light_end) / (t_ramp_end - t_light_end);
    ramp_frac(idx3) = 1;

    % Sudden secondary kicker, one timestep, at t=1500s -- after the
    % initial ramp has settled (t=600s) but with ~3500s of runway left
    % before the sim ends, enough for the converter's thermal mass to
    % actually respond before "max" gets recorded. Sized at the real
    % pilot's own proven collapse-triggering magnitude, applied as a true
    % step (no ramp) to test whether transition speed, not added heat, is
    % what this model actually needs to cross 175 C.
    t_kick = 1500;
    kick_frac = zeros(n_steps, 1);
    kick_frac(t >= t_kick) = 1;

    load_hist = nan(n_steps, NB);
    clip_hi = nan(1, NB);
    clip_lo = nan(1, NB);

    for k = 1:NB
        bus_phase = 0.35 * k;
        bus_var  = (10000 + 700*k) * sin(2*pi*t/(800 + 80*k) + bus_phase);
        fast_var = (1800 + 100*k) * sin(2*pi*t/(50 + 3*k) + 0.2*k);
        noise    = 1200 * randn(n_steps, 1);

        event_bias = 20000 + 1200*k;         % real proven magnitude, gradual ramp
        bias = ramp_frac * event_bias;

        kicker_bias = kick_frac * event_bias; % same proven magnitude, SUDDEN step

        raw = base_load(k) + grid_slow + 0.8*grid_med + bus_var + fast_var + ...
              noise + bias + kicker_bias;

        clip_hi(k) = mean(raw > pmax) * 100;
        clip_lo(k) = mean(raw < pmin) * 100;

        p = max(pmin, min(pmax, raw));
        load_hist(:,k) = p;
        assignin('base', sprintf('Pload_%s', BUS{k}), p(:).');
    end
end


%% ============================== helpers (unchanged) ======================
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


function var_names = tap_gei_signals(model, BUS)
    block_map = containers.Map( ...
        {'A','B','C','D','E','F','G','H','K','L'}, ...
        {'Divide3','Divide4','Divide5','Divide13','Divide14', ...
         'Divide15','Divide16','Divide17','Divide18','Divide19'});

    var_names = cell(1, numel(BUS));
    for i = 1:numel(BUS)
        var_names{i} = sprintf('V7GEI_%s', BUS{i});
        blk_name = block_map(BUS{i});
        blk_path = [model '/' blk_name];
        if isempty(find_system(model, 'SearchDepth', 1, 'LookUnderMasks', 'all', ...
                               'Name', blk_name))
            warning('tap_gei_signals:noBlock', ...
                'Block %s not found at model root for bus %s.', blk_name, BUS{i});
            continue;
        end
        ph = get_param(blk_path, 'PortHandles');
        if ~isfield(ph, 'Outport') || isempty(ph.Outport)
            warning('tap_gei_signals:noOutport', '%s has no output port.', blk_name);
            continue;
        end
        log_name = ['V7GEILOG_' var_names{i}];
        cleanup_and_add_tap(model, log_name, var_names{i}, ph.Outport(1), model);
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