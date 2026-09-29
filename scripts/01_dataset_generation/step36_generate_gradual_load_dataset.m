function step36_generate_gradual_load_dataset()
% ==========================================================================
%  THERMAL DERATING DATASET GENERATION, 10-BUS, V7 MODEL, GRADUAL LOAD
%  --------------------------------------------------------------------
%  This is a direct adaptation of grid_derating_dataset_5000s_v7_2.m,
%  the script that generated the real pilot dataset
%  (thermal_derating_v7_ALfix_5000s.csv). Only TWO things changed:
%
%    1. The load profile. The pilot used a deterministic sequence of
%       step "events" (sudden jumps up/down at fixed times) to force
%       a fast thermal stress. This version instead ramps every bus's
%       RAW (pre-clip) demand smoothly from ~30% of Pmax up over
%       t=500s-2500s to ~140% of Pmax, then holds there until the end.
%       Overshooting Pmax is deliberate: the first version of this
%       ramped only to 90% of Pmax and never crossed the derate onset
%       (125 C) -- max junction temp topped out at ~108 C, nothing ever
%       derated. What actually drove derating in the pilot dataset was
%       raw demand exceeding Pmax by a wide margin for extended
%       stretches, so the bus is clipped and pegged continuously at its
%       100 kW ceiling (100% delivered power, sustained), not merely
%       "high but with headroom." Ramping the target past Pmax
%       reproduces that same continuous-saturation condition. See
%       build_derating_loads_gradual() below.
%
%    2. The output filename/folder (writes into
%       model_outputs\thermal_derating_gradual\ instead of
%       model_outputs\thermal_derating_v7\).
%
%  EVERYTHING ELSE — the model used, the signal taps (GEI, phase,
%  derate factor, junction temperature), the extraction, and the
%  output column names — is copied unchanged from the proven pilot
%  script. This matters: a first attempt at this file used a generic
%  out.logsout.getElement(...) approach that silently failed to find
%  GEI, Phase, junction-temp and commanded-load signals (this model
%  does not broadcast them through logsout at all — they have to be
%  tapped directly off internal block ports, which is what the
%  functions below do). That attempt produced a dataset missing the
%  Phase_* columns, i.e. missing the actual collapse ground truth.
%  This version reuses the tap machinery that is already known to
%  work, so the output has the same 992-feature-compatible schema as
%  the pilot dataset.
%
%  Output: thermal_derating_gradual_load_5000s.csv
%          500,001 rows (TS=0.01s, TEND=5000s), same schema as the
%          pilot CSV (time, V_Bus_X, Bus_X_Src_Pow, GEI_X, Bus_X_Temp,
%          CommandedLoad_kW_X, HeatSinkTemp_C_XY, JunctionTemp_C_XY,
%          Phase_XY_cmd_deg, DAB_XY_Derate_Factor).
% ==========================================================================

    clc; close all;

    %% 0. CONFIG ------------------------------------------------------------
    PROJECT_ROOT = 'D:\naren\Documents\Thermal_Derating_Work_Handover\ThermalProject';

    % Same base model as the pilot (Grid_modelling_Thermal_V7_ALfix — the
    % AL-wiring-fixed model, no shedding). We don't need the sheddable/
    % ramped model for this step; that only matters later in Phase 7
    % (closed-loop testing).
    MODEL_NAME = 'Grid_modelling_Thermal_V7_ALfix';

    TS         = 0.01;
    TEND       = 5000;
    N_STEPS    = round(TEND / TS) + 1;
    LOAD_SEED  = 1;              % same seed as pilot; change to get a
                                  % different random noise realization

    PMIN = 20000;                 % 20 kW floor per bus (same as pilot)
    PMAX = 100000;                % 100 kW ceiling per bus (same as pilot)

    WRITE_STRIDE = 1;             % 1 = full 10 ms resolution

    BUS  = {'A','B','C','D','E','F','G','H','K','L'};
    CONV = {'AB','BC','CD','DE','EF','FG','GH','HK','KL','AL'};
    NB   = numel(BUS);
    NC   = numel(CONV);

    OUTPUT_DIR = fullfile(PROJECT_ROOT, 'model_outputs', 'thermal_derating_gradual');
    if exist(OUTPUT_DIR, 'dir') ~= 7; mkdir(OUTPUT_DIR); end
    addpath(PROJECT_ROOT);

    fprintf('\n===== THERMAL DERATING DATASET, V7 MODEL, GRADUAL LOAD, %.0f s =====\n', TEND);

    %% 1. Build the GRADUAL load profile -------------------------------------
    fprintf('Building gradual load profile (seed %d, Pmax %.0f kW) ...\n', ...
        LOAD_SEED, PMAX/1000);
    [load_hist, clip_hi, clip_lo] = build_derating_loads_gradual( ...
        LOAD_SEED, N_STEPS, TS, BUS, PMIN, PMAX);

    fprintf('\n--- clipping check ---\n');
    for i = 1:NB
        fprintf('  Bus %s: %5.1f%% at Pmax, %5.1f%% at Pmin\n', ...
            BUS{i}, clip_hi(i), clip_lo(i));
    end
    fprintf(['  NOTE: clip_hi close to 100%% during the t=2500-5000s heavy phase\n' ...
             '  is expected and correct -- that means the bus is pegged at its\n' ...
             '  100 kW ceiling continuously, the sustained-overload condition\n' ...
             '  this scenario needs to actually cross the 125 C derate onset.\n']);

    %% 2. Locate and load the V7-ALfix model -------------------------------------
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
    DER = nan(N_STEPS, NC);   % derate factor, 0 to 1
    TJ  = nan(N_STEPS, NC);   % junction temperature, degrees C

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
        PH(:,j)  = grab_series(so, phase_var_names{j}, TS, N_STEPS);   % degrees
        raw_pos  = grab_series(so, derate_var_names{j}, TS, N_STEPS);  % 0 to 100e3 W
        DER(:,j) = raw_pos / 100e3;                                     % normalize to 0-1
        TJ(:,j)  = grab_series(so, junction_var_names{j}, TS, N_STEPS); % degrees C
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
    fprintf('\n%d of %d converters showed derate factor below 1.0 at some point.\n', ...
        n_derated, NC);
    fprintf('%d of %d converters crossed 175 C (junction temperature).\n', n_over175, NC);
    if n_derated == 0
        fprintf(['No converter triggered derating in this run. Even with the heavy\n' ...
                 'phase target raised to 140%% of Pmax, that would mean the\n' ...
                 'saturation period was not long/hot enough to cross 125 C --\n' ...
                 'worth checking clip_hi above (should be near 100%% for t>2500s)\n' ...
                 'before moving on to feature extraction.\n']);
    end

    %% 7. Assemble and write the dataset -----------------------------------------
    keep = 1:WRITE_STRIDE:N_STEPS;
    fprintf('\nWriting dataset, %d of %d rows (stride %d) ...\n', ...
        numel(keep), N_STEPS, WRITE_STRIDE);

    out = table();
    out.time = time_vec(keep);
    for i = 1:NB
        out.(sprintf('V_Bus_%s',            BUS{i})) = V(keep,i);
        out.(sprintf('Bus_%s_Src_Pow',       BUS{i})) = P(keep,i);
        out.(sprintf('GEI_%s',               BUS{i})) = GEI(keep,i);
        out.(sprintf('Bus_%s_Temp',          BUS{i})) = T(keep,i);
        out.(sprintf('CommandedLoad_kW_%s',  BUS{i})) = load_hist(keep,i) / 1000;
    end
    for j = 1:NC
        out.(sprintf('HeatSinkTemp_C_%s',    CONV{j})) = HS(keep,j);
        out.(sprintf('JunctionTemp_C_%s',    CONV{j})) = TJ(keep,j);
        out.(sprintf('Phase_%s_cmd_deg',     CONV{j})) = PH(keep,j);
        out.(sprintf('DAB_%s_Derate_Factor', CONV{j})) = DER(keep,j);
    end

    out_csv = fullfile(OUTPUT_DIR, 'thermal_derating_gradual_load_5000s.csv');
    writetable(out, out_csv);
    fprintf('Wrote: %s\n', out_csv);
    d = dir(out_csv);
    fprintf('File size: %.1f MB\n', d.bytes / 1e6);

    fprintf('\nNext step: run fix_GEI_gradual_load.py.py, then rerun\n');
    fprintf('step33_build_gradual_load_features.py (unchanged) on the result.\n');

    close_system(MODEL_NAME, 0);
end


%% ============================== load profile =============================
function [load_hist, clip_hi, clip_lo] = build_derating_loads_gradual( ...
    seed, n_steps, ts, BUS, pmin, pmax)
    % Builds a per-bus load history that ramps RAW (pre-clip) demand from
    % ~30% of Pmax up to ~140% of Pmax, instead of the pilot's sudden step
    % events.
    %
    %   t < 500s          : envelope = 30% of Pmax  (light, steady)
    %   500s <= t < 2500s : envelope ramps linearly 30% -> 140% of Pmax
    %   t >= 2500s        : envelope = 140% of Pmax (heavy, sustained)
    %
    % The envelope is allowed to exceed 100% of Pmax deliberately: once
    % raw demand is above Pmax, the actual DELIVERED power (p, after
    % clipping below) is pegged at the 100 kW ceiling continuously. That
    % continuous saturation -- not merely "high demand with headroom" --
    % is what actually drives junction temperature up past the 125 C
    % derate onset. An earlier version of this function ramped only to
    % 90% of Pmax (no overshoot) and never crossed 125 C.
    %
    % Each bus's raw demand = envelope(t) + a small fixed per-bus offset
    % + slow and fast sinusoidal wobble + noise, so buses stay
    % distinguishable from one another (needed for the ring-imbalance/GEI
    % features) while the ramp itself is what drives the thermal stress,
    % not artificial step events.

    rng(seed);
    t = (0:n_steps-1)' * ts;
    NB = numel(BUS);

    % Per-bus fixed offset, +/- ~13 kW around the ramp, same relative
    % spread as the pilot's base_load list (recentered around 0 so it
    % layers on top of the envelope instead of replacing it).
    base_offset = [35000 30000 25000 32000 28000 24000 30000 26000 22000 34000] - 30000;

    % ---- envelope: fraction of PMAX, piecewise-linear ramp -------------
    t_light_end = 500;
    t_ramp_end  = 2500;
    frac_light  = 0.30;
    frac_heavy  = 1.40;   % CHANGED from 0.90 -- overshoot Pmax so the
                          % heavy phase is continuous saturation at the
                          % ceiling, not just "high but with headroom."

    envelope = zeros(n_steps, 1);
    idx1 = t < t_light_end;
    idx2 = (t >= t_light_end) & (t < t_ramp_end);
    idx3 = t >= t_ramp_end;
    envelope(idx1) = frac_light;
    envelope(idx2) = frac_light + (frac_heavy - frac_light) * ...
        (t(idx2) - t_light_end) / (t_ramp_end - t_light_end);
    envelope(idx3) = frac_heavy;

    target_center = envelope * pmax;   % 30 kW ... 140 kW (pre-clip) per bus

    load_hist = nan(n_steps, NB);
    clip_hi = nan(1, NB);
    clip_lo = nan(1, NB);

    for k = 1:NB
        bus_phase = 0.35 * k;
        bus_slow  = (2500 + 150*k) * sin(2*pi*t/(900 + 60*k) + bus_phase);
        bus_fast  = (700 + 40*k)  * sin(2*pi*t/(45 + 2*k) + 0.2*k);
        noise     = 600 * randn(n_steps, 1);

        raw = target_center + base_offset(k) + bus_slow + bus_fast + noise;

        clip_hi(k) = mean(raw > pmax) * 100;
        clip_lo(k) = mean(raw < pmin) * 100;

        p = max(pmin, min(pmax, raw));   % actual DELIVERED power, capped at Pmax
        load_hist(:,k) = p;

        % The model's Repeating Sequence Stair blocks read Pload_<bus>.
        assignin('base', sprintf('Pload_%s', BUS{k}), p(:).');
    end
end


%% ============================== taps (unchanged from pilot) ==============
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
    % GEI has no Goto tag. Each bus's GEI is a named output port on a
    % root-level Divide block:
    %   A->Divide3  B->Divide4  C->Divide5   D->Divide13 E->Divide14
    %   F->Divide15 G->Divide16 H->Divide17  K->Divide18 L->Divide19
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
                'No phase_shift Inport found in %s.', dab_path);
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
    % Temp_Junction is outport 4 of the parent DAB_Model_<CONV> subsystem.
    % Its external port has no line connected (nothing outside reads it
    % by default), so this taps the line INSIDE the subsystem that feeds
    % the Temp_Junction Outport block, which is always connected.
    var_names = cell(1, numel(CONV));
    for i = 1:numel(CONV)
        var_names{i} = sprintf('V7TJ_%s', CONV{i});
        dab_path = [model '/DAB_Model_' CONV{i}];
        outport_blk = find_system(dab_path, 'SearchDepth', 1, 'LookUnderMasks', 'all', ...
                                   'BlockType', 'Outport', 'Name', 'Temp_Junction');
        if numel(outport_blk) ~= 1
            warning('tap_junction_temp:noOutport', ...
                'Expected one Temp_Junction Outport under DAB_Model_%s, found %d.', ...
                CONV{i}, numel(outport_blk));
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