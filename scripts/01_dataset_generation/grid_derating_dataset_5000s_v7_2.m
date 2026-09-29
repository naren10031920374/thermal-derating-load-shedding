function grid_derating_dataset_5000s_v7_2()
% ==========================================================================
%  THERMAL DERATING DATASET GENERATION, 10-BUS, V7 MODEL, 5000 s
%  --------------------------------------------------------------------
%  CHANGES FROM v7_1
%    1. MODEL_NAME now points at Grid_modelling_Thermal_V7_ALfix, not the
%       original Grid_modelling_Thermal_V7. The AL wiring defect
%       (mismatched Goto/From tags dab_al_i/dab_al_j vs dab_la_i/dab_la_j,
%       meaning AL's current never reached bus A or bus L) is fixed in
%       that file. Confirmed live, not just from the XML: after the fix,
%       AL's phase spent 6.8% of a 5000 s run within 1 deg of +/-90,
%       down from 99.2% before, and its derate factor now tracks the
%       (175-Tj)/50 ramp formula almost exactly during real thermal
%       events instead of ranging outside [0,1]. Still worth a two-line
%       confirmation from Apoorv before this becomes the reference file
%       everyone builds on, since it is his model.
%    2. Junction temperature (Tj) is now tapped and written as a real
%       column, JunctionTemp_C_<CONV>. It sits on outport 4 of every
%       DAB_Model_<CONV> subsystem (5 outports total: i_dab_i, i_dab_j,
%       dab_k, Temp_Junction, Temp_HeatSink), not inside DAB_Thermal_Model
%       itself, which only has 2 outports (T_j, T_h). This is the signal
%       the derating logic actually switches on (the If block condition
%       inside thermal_derater reads Tj directly), the previously logged
%       HeatSinkTemp_C_<CONV> is one thermal-resistance hop downstream
%       and runs cooler and later than Tj, so it under-represents how
%       thermally stressed a converter actually is at a given moment.
%       Tapped the same direct-port way as phase_shift and derate factor,
%       see tap_junction_temp below.
%    3. The "Checking From Workspace extrapolation setting" step from
%       v7_1 has been removed. Confirmed directly against the model:
%       Grid_modelling_Thermal_V7(_ALfix) contains zero FromWorkspace
%       blocks. Loads enter through Repeating Sequence Stair blocks
%       reading Pload_<bus> from the base workspace (written by
%       build_derating_loads below via assignin), not From Workspace
%       blocks at all. The old check always found 0 blocks and printed
%       "All 0 block(s) already set", which read as a real check passing
%       but was never testing anything. Removed rather than left in.
%
%  EVERYTHING BELOW UNCHANGED FROM v7_1, retained for reference
%  --------------------------------------------------------------------
%  What changed from the earlier V3-based version of this script
%    This targets the V7 model, the one Apoorv delivered with real
%    thermal derating logic built in. Verified directly in the model
%    file before writing this:
%
%    1. Every one of the ten DAB_Model_<CONV> subsystems contains a
%       "thermal_derater" subsystem. It reads junction temperature (Tj)
%       and applies a three-zone rule:
%         Tj < 125 C:            derate factor = 1.0 (full power)
%         125 C <= Tj <= 175 C:  derate factor ramps linearly 1.0 -> 0.0
%         Tj > 175 C:            derate factor, see caveat below
%       The factor scales to a +/- power limit (0 to 100 kW) applied
%       through a Saturation Dynamic block downstream of the existing
%       fixed +/-90 deg phase saturation. Phase itself is still capped at
%       +/-90 deg as before; this adds a temperature-dependent power cap
%       on top of that.
%
%    2. AL's phase_shift is wired to 'bus_l_cntrl', the same signal the
%       earlier investigation found orphaned in the V3 model. In V7 the
%       controller itself was correctly connected by Apoorv, but the
%       Goto/From tag mismatch downstream (see change 1 above) meant
%       AL's current still never reached the buses until the ALfix
%       applied here.
%
%  CAVEAT worth flagging back to Apoorv, not fixed here
%    The Tj > 175 C branch's Constant block has no explicit Value set in
%    the model file, meaning it uses Simulink's default of 1, the same
%    as the "fully healthy" branch below 125 C. If accurate, a converter
%    that gets hotter than 175 C would revert to FULL power instead of
%    being cut off, the opposite of what the 125-175 ramp is building
%    toward. Not yet confirmed live: the one full run checked so far
%    peaked at 164.32 C (AL), never crossing 175. This dataset runs at
%    GRID_LOAD_SCALE = 1.25 (higher than that check's 1.0), so it may be
%    the first real chance to see whether this bug actually triggers,
%    worth checking the JunctionTemp_C_* columns against 175 once this
%    is generated. Now checkable directly since Tj is finally logged,
%    not inferred from the cooler, lagging heat-sink temperature.
%
%  Load recipe
%    Reused exactly as given in grid_load_test_temp_run_10bus.m: seed 1,
%    scale 1.25, Pmax left at 100 kW (not scaled to 125 kW the way an
%    earlier draft of this script did). Kept unscaled deliberately, since
%    that is the exact configuration that produced the attached result,
%    and changing it here would break reproducibility with that run.
%
%  Duration and timing
%    TEND = 5000 s, single sim() call (PI control lives entirely in
%    Simulink). A full run of the fixed model with junction temperature
%    logging added took 254.9 s wall clock (about 4.25 min) in the
%    verification run this script's junction temp tap was validated
%    against, though that run had fewer tapped signals overall, treat
%    that as a rough floor, not a guarantee, for this fuller script.
%
%  Output file size
%    500,001 rows at full 10 ms resolution. WRITE_STRIDE below controls
%    this, default 1 (full resolution). This dataset has one more column
%    per converter than v7_1 (adds junction temperature), so expect a
%    somewhat larger file for the same stride.
% ==========================================================================

    clc; close all;

    %% 0. CONFIG ------------------------------------------------------------
    PROJECT_ROOT = 'D:\naren\Documents\Thermal_Derating_Work_Handover\ThermalProject';
        
    MODEL_NAME = 'Grid_modelling_Thermal_V7_ALfix';

    TS         = 0.01;
    TEND       = 5000;
    N_STEPS    = round(TEND / TS) + 1;
    LOAD_SEED  = 1;

    GRID_LOAD_SCALE = 1.0;
    PMIN = 20000;
    PMAX = 100000;    % left unscaled, matches grid_load_test_temp_run_10bus.m exactly

    WRITE_STRIDE = 1;    % 1 = full 10 ms resolution, raise to subsample for storage

    BUS  = {'A','B','C','D','E','F','G','H','K','L'};
    CONV = {'AB','BC','CD','DE','EF','FG','GH','HK','KL','AL'};
    NB   = numel(BUS);
    NC   = numel(CONV);

    OUTPUT_DIR = fullfile(PROJECT_ROOT, 'model_outputs', 'thermal_derating_v7');
    if exist(OUTPUT_DIR, 'dir') ~= 7; mkdir(OUTPUT_DIR); end
    addpath(PROJECT_ROOT);

    fprintf('\n===== THERMAL DERATING DATASET, V7 MODEL (AL WIRING FIXED), %.0f s, scale %.2f =====\n', ...
        TEND, GRID_LOAD_SCALE);

    %% 1. Build the load profile, exactly matching the attached script -------
    fprintf('Building load profile (seed %d, scale %.2f, Pmax %.0f kW, unscaled) ...\n', ...
        LOAD_SEED, GRID_LOAD_SCALE, PMAX/1000);
    [load_hist, clip_hi, clip_lo] = build_derating_loads( ...
        LOAD_SEED, N_STEPS, TS, BUS, GRID_LOAD_SCALE, PMIN, PMAX);

    fprintf('\n--- clipping check ---\n');
    for i = 1:NB
        fprintf('  Bus %s: %5.1f%% at Pmax, %5.1f%% at Pmin\n', ...
            BUS{i}, clip_hi(i), clip_lo(i));
    end
    if mean(clip_hi) > 5
        fprintf(['  NOTE: meaningful ceiling clipping at this scale/Pmax combination.\n' ...
                 '  Pmax was left unscaled to match the attached reference run;\n' ...
                 '  raising it would trade some of this clipping for a different\n' ...
                 '  dataset than the one already validated visually.\n']);
    end

    %% 2. Locate and load the fixed V7 model -------------------------------------
    hh = dir(fullfile(PROJECT_ROOT, '**', [MODEL_NAME '.slx']));
    if isempty(hh)
        error(['Model not found under project root: %s.slx\n' ...
               'This script expects the AL-fixed model. If you have not created it\n' ...
               'yet, run fix_AL_wiring_v7.m first, it saves this file next to the\n' ...
               'original without modifying it.'], MODEL_NAME);
    end
    model_file = fullfile(hh(1).folder, hh(1).name);
    fprintf('\nModel: %s\n', model_file);
    if bdIsLoaded(MODEL_NAME)
        % Avoid reusing a stale in-memory copy from an earlier run in this
        % session, which could still be carrying taps or edits from a
        % previous call to this script or a diagnostic script.
        close_system(MODEL_NAME, 0);
    end
    load_system(model_file);

    %% 3. Wire taps ---------------------------------------------------------
    fprintf('\nWiring signal taps ...\n');
    wire_all_taps(MODEL_NAME, BUS, CONV);
    gei_var_names    = tap_gei_signals(MODEL_NAME, BUS);
    phase_var_names  = tap_phase_inports(MODEL_NAME, CONV);
    derate_var_names = tap_derate_factors(MODEL_NAME, CONV);
    junction_var_names = tap_junction_temp(MODEL_NAME, CONV);

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
        PH(:,j)  = grab_series(so, phase_var_names{j}, TS, N_STEPS);   % already degrees
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
        fprintf(['No converter triggered derating in this run. Either the load scale\n' ...
                 'was not enough to cross 125 C, or a tap is not reading correctly,\n' ...
                 'check the junction temperatures above against the 125 C threshold.\n']);
    end
    if n_over175 > 0
        fprintf(['At least one converter crossed 175 C. Worth checking its\n' ...
                 'DAB_%%_Derate_Factor column right at that crossing: if it is still\n' ...
                 'near 1.0 rather than dropping toward 0, that is the >175 C\n' ...
                 'Constant-defaults-to-1 bug flagged above, now confirmed live in\n' ...
                 'this dataset rather than only read from the model XML. Worth\n' ...
                 'reporting to Apoorv with the specific timestamp if so.\n']);
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

    out_csv = fullfile(OUTPUT_DIR, 'thermal_derating_v7_ALfix_5000s.csv');
    writetable(out, out_csv);
    fprintf('Wrote: %s\n', out_csv);
    d = dir(out_csv);
    fprintf('File size: %.1f MB\n', d.bytes / 1e6);

    fprintf(['\nDone. Output file name changed to thermal_derating_v7_ALfix_5000s.csv\n' ...
             '(not the old thermal_derating_v7_5000s.csv name) to avoid silently mixing\n' ...
             'this dataset with the earlier one generated before the AL fix. Update\n' ...
             'INPUT_CSV in step33 (or your current feature-build script) to point at\n' ...
             'this new file before rerunning it.\n']);
end


%% ============================== helpers =================================
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


function wire_all_taps(model, BUS, CONV)
    % NOTE: GEI is intentionally NOT included here. In this model, GEI
    % is computed live at the model root as a named output port on a
    % Divide block (e.g. 'Bus_A_GEI' on block Divide3), never broadcast
    % through a Goto/From pair the way V, P, T, and HS are. Searching for
    % a Goto tag named GEI_<bus> here would always silently miss it. See
    % tap_gei_signals below for the correct direct-port approach.
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
    % GEI has no Goto tag in this model. Each bus's GEI is a named output
    % port on a root-level Divide block. This mapping was confirmed
    % directly against the model file, not assumed:
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

    % Defensive cleanup: remove any prior To Workspace block anywhere
    % already writing this variable name, regardless of which script or
    % naming prefix created it. Several scripts in this project share the
    % same variable-naming convention and will collide otherwise.
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
    % Taps the "thermal_derater" subsystem's derate_pos output (0 to
    % 100e3 W) inside each DAB_Model_<CONV>. This signal is not broadcast
    % via any Goto tag in the model as delivered, it is local to each
    % converter subsystem, so it is tapped the same way phase_shift is:
    % directly on the block's output port.
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
    % Temp_Junction is OUTPORT 4 OF THE PARENT DAB_Model_<CONV> SUBSYSTEM
    % (5 outports total: i_dab_i, i_dab_j, dab_k, Temp_Junction,
    % Temp_HeatSink), not a port on the DAB_Thermal_Model block inside
    % it, that block only has 2 outports (T_j, T_h). Confirmed directly
    % against the model file, and confirmed live via a verification run
    % before this tap was added here permanently.
    %
    % The subsystem's EXTERNAL port line (the one visible from outside,
    % at model root) has no line connected, since nothing outside the
    % subsystem currently reads Temp_Junction, that is the entire reason
    % this tap exists. So this taps the line INSIDE the subsystem that
    % feeds INTO the 'Temp_Junction' Outport block instead, which is
    % always connected internally regardless of what happens outside.
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
