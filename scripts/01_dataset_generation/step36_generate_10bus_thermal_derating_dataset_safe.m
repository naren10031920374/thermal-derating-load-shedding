function step36_generate_10bus_thermal_derating_dataset_safe(run_mode)
% =========================================================================
% STEP 36: SAFE 10-BUS THERMAL-DERATING DATASET GENERATION
% =========================================================================
% PURPOSE
%   Generate a thermal-derating dataset without modifying Apoorv's original
%   Grid_modelling_Thermal_V7.slx model.
%
% WHY A DATASET-ONLY MODEL COPY IS REQUIRED
%   1. In the delivered V7 model, AL is driven by a tenth independent PI
%      controller. Around the closed ring, the ten adjacent power-error
%      equations contain only nine independent conditions. The extra
%      integral loop can create a redundant circulating control mode and AL
%      has already been observed to saturate. For this dataset-only copy,
%      AL is therefore restored as the 0-degree phase reference.
%
%   2. The delivered thermal derater has two implementation problems:
%        - the Tj > 175 C branch uses the default Constant value 1 rather
%          than 0;
%        - exact Tj = 125 C and Tj = 175 C are not covered by the If logic.
%      This script corrects those items only in the dataset copy:
%        Tj <= 125 C              -> factor 1
%        125 C < Tj < 175 C       -> (175-Tj)/50
%        Tj >= 175 C              -> factor 0
%
% IMPORTANT SCOPE NOTE
%   This is a transparent research workaround for dataset generation while
%   the electrical-model owner is unavailable. It is not presented as an
%   approved permanent control-model change. The original V7 file is never
%   overwritten.
%
% OUTPUTS
%   A. Dataset-only model copy:
%      Grid_modelling_Thermal_V7_DatasetReady.slx
%
%   B. CSV with, for all 10 buses/converters:
%      commanded load, source power, common average source power, GEI,
%      voltage, bus temperature, junction temperature, heat-sink
%      temperature, phase command, derate factor, and thermal-stage ID.
%
%   C. A text quality report. A dataset is marked PASS only when:
%      - no logged numeric signal contains NaN or Inf;
%      - AL remains the 0-degree reference;
%      - the derate-factor values agree with junction temperature;
%      - at least some genuine derating is observed.
%
% RUN ORDER
%   First run a pilot:
%       step36_generate_10bus_thermal_derating_dataset_safe('pilot')
%
%   Review the printed quality result. If PASS, run the full dataset:
%       step36_generate_10bus_thermal_derating_dataset_safe('full')
%
% MATLAB / SIMULINK
%   Written for the model structure supplied in Grid_modelling_Thermal_V7.
% =========================================================================

    if nargin < 1
        run_mode = 'pilot';
    end
    run_mode = lower(string(run_mode));
    if ~ismember(run_mode, ["pilot", "full"])
        error('run_mode must be ''pilot'' or ''full''.');
    end

    clc; close all;

    %% 0. CONFIG ------------------------------------------------------------
    PROJECT_ROOT = 'D:\naren\Documents\Thermal_Derating_Work_Handover\ThermalProject';

    SOURCE_MODEL = 'Grid_modelling_Thermal_V7';
    DATASET_MODEL = 'Grid_modelling_Thermal_V7_DatasetReady';

    TS = 0.01;
    if run_mode == "pilot"
        TEND = 1000;          % checks all taps and all stress stages quickly
        WRITE_STRIDE = 1;
    else
        TEND = 5000;
        WRITE_STRIDE = 1;     % full 10 ms resolution
    end

    LOAD_SEED = 36;
    PMIN = 20000;
    PMAX = 100000;

    BUS  = {'A','B','C','D','E','F','G','H','K','L'};
    CONV = {'AB','BC','CD','DE','EF','FG','GH','HK','KL','AL'};
    NB = numel(BUS);
    NC = numel(CONV);
    N_STEPS = round(TEND / TS) + 1;

    OUTPUT_DIR = fullfile(PROJECT_ROOT, 'model_outputs', ...
        'thermal_derating_dataset_safe');
    if exist(OUTPUT_DIR, 'dir') ~= 7
        mkdir(OUTPUT_DIR);
    end
    addpath(PROJECT_ROOT);

    fprintf('\n============================================================\n');
    fprintf('STEP 36: SAFE THERMAL-DERATING DATASET (%s, %.0f s)\n', ...
        upper(char(run_mode)), TEND);
    fprintf('============================================================\n');

    %% 1. CREATE AND PATCH A COPY OF V7 ------------------------------------
    fprintf('\n[1/7] Creating dataset-only model copy ...\n');
    dataset_model_file = prepare_dataset_model_copy( ...
        PROJECT_ROOT, SOURCE_MODEL, DATASET_MODEL, CONV);
    fprintf('Dataset model: %s\n', dataset_model_file);

    %% 2. BUILD A THERMAL-EXCITATION LOAD PROFILE --------------------------
    fprintf('\n[2/7] Building deterministic thermal-excitation loads ...\n');
    [load_hist, stage_id, clip_hi, clip_lo] = build_thermal_excitation_loads( ...
        LOAD_SEED, N_STEPS, TS, TEND, BUS, PMIN, PMAX);

    fprintf('\nLoad clipping summary:\n');
    for i = 1:NB
        fprintf('  Bus %s: %6.2f%% at Pmax, %6.2f%% at Pmin\n', ...
            BUS{i}, clip_hi(i), clip_lo(i));
    end
    fprintf('  Mean ceiling clipping: %.2f%%\n', mean(clip_hi));

    %% 3. LOAD MODEL AND ADD SIGNAL TAPS -----------------------------------
    fprintf('\n[3/7] Loading dataset model and wiring taps ...\n');
    if bdIsLoaded(DATASET_MODEL)
        close_system(DATASET_MODEL, 0);
    end
    load_system(dataset_model_file);

    force_final_value_hold(DATASET_MODEL);
    wire_goto_taps(DATASET_MODEL, BUS);

    gei_var_names = tap_gei_signals(DATASET_MODEL, BUS);
    phase_var_names = tap_phase_inports(DATASET_MODEL, CONV, TS);
    derate_var_names = tap_derate_factors(DATASET_MODEL, CONV, TS);
    junction_var_names = tap_dab_output_ports( ...
        DATASET_MODEL, CONV, 4, 'V7TJ', TS);
    heat_sink_var_names = tap_dab_output_ports( ...
        DATASET_MODEL, CONV, 5, 'V7HS', TS);

    % The common GEI denominator is essential for diagnosing any future
    % simultaneous GEI failure.
    wire_one_goto_tap(DATASET_MODEL, 'Src_Pow_Avg', TS);
    avg_power_var_name = matlab.lang.makeValidName('Src_Pow_Avg');

    save_system(DATASET_MODEL);

    %% 4. SIMULATE ---------------------------------------------------------
    fprintf('\n[4/7] Running one continuous Simulink simulation ...\n');
    set_param(DATASET_MODEL, 'StopTime', num2str(TEND));

    tic;
    warn_state = warning('off', 'all');
    so = sim(DATASET_MODEL, 'ReturnWorkspaceOutputs', 'on');
    warning(warn_state);
    wall_s = toc;
    fprintf('Simulation completed in %.1f s wall time (%.2f min).\n', ...
        wall_s, wall_s / 60);

    %% 5. EXTRACT SIGNALS --------------------------------------------------
    fprintf('\n[5/7] Extracting signals ...\n');
    time_vec = (0:N_STEPS-1)' * TS;

    V = nan(N_STEPS, NB);
    P = nan(N_STEPS, NB);
    GEI = nan(N_STEPS, NB);
    BT = nan(N_STEPS, NB);
    PH = nan(N_STEPS, NC);
    DER = nan(N_STEPS, NC);
    TJ = nan(N_STEPS, NC);
    HS = nan(N_STEPS, NC);

    PAVG = grab_series(so, avg_power_var_name, TS, N_STEPS);

    for i = 1:NB
        V(:,i) = grab_series(so, sprintf('V_Bus_%s', BUS{i}), TS, N_STEPS);
        P(:,i) = grab_series(so, sprintf('Bus_%s_Src_Pow', BUS{i}), TS, N_STEPS);
        GEI(:,i) = grab_series(so, gei_var_names{i}, TS, N_STEPS);
        BT(:,i) = grab_series(so, sprintf('Bus_%s_Temp', BUS{i}), TS, N_STEPS);
    end

    for j = 1:NC
        PH(:,j) = grab_series(so, phase_var_names{j}, TS, N_STEPS);
        raw_derate_limit = grab_series(so, derate_var_names{j}, TS, N_STEPS);
        DER(:,j) = raw_derate_limit / 100e3;
        TJ(:,j) = grab_series(so, junction_var_names{j}, TS, N_STEPS);
        HS(:,j) = grab_series(so, heat_sink_var_names{j}, TS, N_STEPS);
    end

    %% 6. QUALITY GATE -----------------------------------------------------
    fprintf('\n[6/7] Running dataset quality gate ...\n');
    require_derating = (run_mode == "full");
    report = assess_dataset_quality( ...
        time_vec, PAVG, V, P, GEI, BT, TJ, HS, PH, DER, CONV, ...
        require_derating);
    print_quality_report(report, CONV);

    %% 7. WRITE DATASET AND REPORT ----------------------------------------
    fprintf('\n[7/7] Writing outputs ...\n');
    keep = 1:WRITE_STRIDE:N_STEPS;

    out = table();
    out.time = time_vec(keep);
    out.Thermal_Stage_ID = stage_id(keep);
    out.Src_Pow_Avg = PAVG(keep);

    for i = 1:NB
        out.(sprintf('CommandedLoad_kW_%s', BUS{i})) = load_hist(keep,i) / 1000;
        out.(sprintf('Bus_%s_Src_Pow', BUS{i})) = P(keep,i);
        out.(sprintf('GEI_%s', BUS{i})) = GEI(keep,i);
        out.(sprintf('V_Bus_%s', BUS{i})) = V(keep,i);
        out.(sprintf('Bus_%s_Temp', BUS{i})) = BT(keep,i);
    end

    for j = 1:NC
        out.(sprintf('JunctionTemp_C_%s', CONV{j})) = TJ(keep,j);
        out.(sprintf('HeatSinkTemp_C_%s', CONV{j})) = HS(keep,j);
        out.(sprintf('Phase_%s_cmd_deg', CONV{j})) = PH(keep,j);
        out.(sprintf('DAB_%s_Derate_Factor', CONV{j})) = DER(keep,j);
    end

    status_text = 'REVIEW';
    if report.pass
        status_text = 'PASS';
    end

    base_name = sprintf('thermal_derating_%s_%ds_%s', ...
        char(run_mode), round(TEND), status_text);
    out_csv = fullfile(OUTPUT_DIR, [base_name '.csv']);
    writetable(out, out_csv);

    report_txt = fullfile(OUTPUT_DIR, [base_name '_quality_report.txt']);
    write_quality_report(report_txt, report, run_mode, TEND, TS, ...
        SOURCE_MODEL, DATASET_MODEL, CONV, clip_hi, clip_lo, wall_s);

    fprintf('Dataset:       %s\n', out_csv);
    fprintf('Quality report:%s\n', report_txt);
    file_info = dir(out_csv);
    fprintf('CSV size: %.1f MB\n', file_info.bytes / 1e6);

    if report.pass
        fprintf('\nFINAL STATUS: PASS\n');
        if run_mode == "pilot"
            fprintf(['The pilot passed. You may now run:\n' ...
                '  step36_generate_10bus_thermal_derating_dataset_safe(''full'')\n']);
        else
            fprintf('The full dataset passed the mandatory quality checks.\n');
        end
    else
        fprintf('\nFINAL STATUS: REVIEW, DO NOT TRAIN ON THIS CSV.\n');
        fprintf('Read the quality report and resolve the failed checks first.\n');
    end

    close_system(DATASET_MODEL, 0);
end


%% =========================================================================
% MODEL PREPARATION
% =========================================================================
function dataset_model_file = prepare_dataset_model_copy( ...
    project_root, source_model, dataset_model, CONV)

    src = dir(fullfile(project_root, '**', [source_model '.slx']));
    if isempty(src)
        error('Could not find %s.slx under %s.', source_model, project_root);
    end
    source_model_file = fullfile(src(1).folder, src(1).name);
    dataset_model_file = fullfile(src(1).folder, [dataset_model '.slx']);

    if bdIsLoaded(source_model)
        close_system(source_model, 0);
    end
    if bdIsLoaded(dataset_model)
        close_system(dataset_model, 0);
    end

    % Copy the package byte-for-byte first. All modifications are applied
    % only after the copy is loaded under the destination filename.
    if exist(dataset_model_file, 'file') == 2
        delete(dataset_model_file);
    end
    [copy_ok, copy_msg] = copyfile(source_model_file, dataset_model_file, 'f');
    if ~copy_ok
        error('Could not create dataset model copy: %s', copy_msg);
    end

    load_system(dataset_model_file);

    % Confirm the model loaded using the destination name.
    loaded_roots = find_system('SearchDepth', 0, 'Type', 'block_diagram');
    if ~ismember(dataset_model, loaded_roots)
        close_system('all', 0);
        error(['The dataset copy was saved, but Simulink did not load it as %s. ' ...
            'Open %s manually once, save it, and rerun.'], ...
            dataset_model, dataset_model_file);
    end

    patch_deraters(dataset_model, CONV);
    restore_al_reference(dataset_model);

    set_param(dataset_model, 'Description', sprintf([ ...
        'DATASET-ONLY COPY created by step36. Original V7 preserved.\n' ...
        'AL held at 0 deg as ring reference. Thermal derater corrected to ' ...
        '1 below/equal 125 C, linear 125-175 C, and 0 above/equal 175 C.']));

    save_system(dataset_model);
    close_system(dataset_model, 0);
end


function patch_deraters(model, CONV)
    fprintf('  Correcting thermal deraters in all converters ...\n');

    for i = 1:numel(CONV)
        derater = [model '/DAB_Model_' CONV{i} '/thermal_derater'];
        if getSimulinkBlockHandle(derater) == -1
            error('Missing thermal_derater: %s', derater);
        end

        if_block = [derater '/If'];
        set_param(if_block, ...
            'IfExpression', 'u1 <= 125', ...
            'ElseIfExpressions', 'u1 > 125 & u1 < 175, u1 >= 175', ...
            'ShowElse', 'off');

        actions = find_system(derater, 'SearchDepth', 1, ...
            'LookUnderMasks', 'all', 'BlockType', 'SubSystem');

        found_healthy = false;
        found_hot = false;
        found_linear = false;

        for a = 1:numel(actions)
            action_ports = find_system(actions{a}, 'SearchDepth', 1, ...
                'LookUnderMasks', 'all', 'BlockType', 'ActionPort');
            if isempty(action_ports)
                continue;
            end
            label = get_param(action_ports{1}, 'ActionPortLabel');
            constants = find_system(actions{a}, 'SearchDepth', 1, ...
                'LookUnderMasks', 'all', 'BlockType', 'Constant');
            inports = find_system(actions{a}, 'SearchDepth', 1, ...
                'LookUnderMasks', 'all', 'BlockType', 'Inport');

            if contains(label, '< 125') && isempty(inports)
                if isempty(constants)
                    error('Healthy action has no Constant in %s.', actions{a});
                end
                set_param(constants{1}, 'Value', '1');
                found_healthy = true;

            elseif contains(label, '> 175') && isempty(inports)
                if isempty(constants)
                    error('Hot action has no Constant in %s.', actions{a});
                end
                set_param(constants{1}, 'Value', '0');
                found_hot = true;

            elseif ~isempty(inports)
                % Linear branch. Set threshold constants explicitly.
                tderate = find_system(actions{a}, 'SearchDepth', 1, ...
                    'LookUnderMasks', 'all', 'BlockType', 'Constant', ...
                    'Name', 'Tderate');
                tmax = find_system(actions{a}, 'SearchDepth', 1, ...
                    'LookUnderMasks', 'all', 'BlockType', 'Constant', ...
                    'Name', 'Tmax');
                if isempty(tderate) || isempty(tmax)
                    error('Could not find Tderate/Tmax in %s.', actions{a});
                end
                set_param(tderate{1}, 'Value', '125');
                set_param(tmax{1}, 'Value', '175');
                found_linear = true;
            end
        end

        if ~(found_healthy && found_linear && found_hot)
            error(['Failed to identify all three derating branches for %s. ' ...
                'healthy=%d linear=%d hot=%d'], ...
                CONV{i}, found_healthy, found_linear, found_hot);
        end

        fprintf('    %s: corrected\n', CONV{i});
    end
end


function restore_al_reference(model)
    fprintf('  Restoring AL as the 0-degree ring reference ...\n');

    al_block = [model '/DAB_Model_AL'];
    ph_al = get_param(al_block, 'PortHandles');
    if isempty(ph_al.Inport)
        error('DAB_Model_AL has no input ports.');
    end
    al_phase_in = ph_al.Inport(1);

    old_line = get_param(al_phase_in, 'Line');
    if old_line ~= -1
        delete_line(old_line);
    end

    const_name = 'AL_Phase_Reference_0';
    const_path = [model '/' const_name];
    existing = find_system(model, 'SearchDepth', 1, ...
        'LookUnderMasks', 'all', 'Name', const_name);
    if ~isempty(existing)
        delete_block(existing{1});
    end

    from_blocks = find_system(model, 'SearchDepth', 1, ...
        'LookUnderMasks', 'all', 'BlockType', 'From', ...
        'GotoTag', 'bus_l_cntrl');
    pos = [-2090, 3745, -1990, 3775];
    if ~isempty(from_blocks)
        old_pos = get_param(from_blocks{1}, 'Position');
        pos = old_pos;
    end

    add_block('simulink/Sources/Constant', const_path, ...
        'Value', '0', 'Position', pos);
    ph_const = get_param(const_path, 'PortHandles');
    add_line(model, ph_const.Outport(1), al_phase_in, 'autorouting', 'on');

    fprintf('    AL phase input now uses Constant(0).\n');
end


%% =========================================================================
% LOAD GENERATION
% =========================================================================
function [load_hist, stage_id, clip_hi, clip_lo] = ...
    build_thermal_excitation_loads(seed, n_steps, ts, tend, BUS, pmin, pmax)

    rng(seed);
    t = (0:n_steps-1)' * ts;
    NB = numel(BUS);

    % Moderately different baselines preserve bus diversity.
    base_load = [41000 38000 35000 40000 37000 34000 39000 36000 33000 40500];

    common_slow = 6500 * sin(2*pi*t/700);
    common_med  = 4000 * sin(2*pi*t/180 + 0.4);

    % Six stages are expressed as fractions of TEND, so pilot and full runs
    % exercise the same sequence. Stages 2 and 4 are sustained heating;
    % stages 3 and 5 provide recovery data.
    q = t / tend;
    stage_id = ones(n_steps, 1);
    stage_id(q >= 0.10 & q < 0.34) = 2;
    stage_id(q >= 0.34 & q < 0.48) = 3;
    stage_id(q >= 0.48 & q < 0.76) = 4;
    stage_id(q >= 0.76 & q < 0.90) = 5;
    stage_id(q >= 0.90) = 6;

    load_hist = nan(n_steps, NB);
    clip_hi = nan(1, NB);
    clip_lo = nan(1, NB);

    for k = 1:NB
        bus_phase = 0.31 * k;
        bus_slow = (4500 + 250*k) * sin(2*pi*t/(520 + 35*k) + bus_phase);
        bus_fast = (1200 + 60*k) * sin(2*pi*t/(42 + 2*k) + 0.16*k);
        noise = 650 * randn(n_steps, 1);

        stage_bias = zeros(n_steps, 1);
        stage_bias(stage_id == 1) = 0;
        stage_bias(stage_id == 2) = 33000 + 550*k;
        stage_bias(stage_id == 3) = -9000 + 150*k;
        stage_bias(stage_id == 4) = 40500 + 450*k;
        stage_bias(stage_id == 5) = -7000 + 100*k;
        stage_bias(stage_id == 6) = 23000 + 350*k;

        % Small converter-location diversity avoids all buses following an
        % identical thermal trajectory while keeping every load positive.
        asymmetry = 2500 * sin(2*pi*(k-1)/NB) .* double(stage_id == 4);

        raw = base_load(k) + common_slow + 0.8*common_med + ...
            bus_slow + bus_fast + noise + stage_bias + asymmetry;

        clip_hi(k) = mean(raw > pmax) * 100;
        clip_lo(k) = mean(raw < pmin) * 100;
        p = max(pmin, min(pmax, raw));
        load_hist(:,k) = p;

        % The model's Repeating Sequence Stair blocks read Pload_X.'
        assignin('base', sprintf('Pload_%s', BUS{k}), p(:).');
    end
end


%% =========================================================================
% SIGNAL TAPS
% =========================================================================
function force_final_value_hold(model)
    fw_blocks = find_system(model, 'FollowLinks', 'on', ...
        'LookUnderMasks', 'all', 'BlockType', 'FromWorkspace');
    for i = 1:numel(fw_blocks)
        try
            set_param(fw_blocks{i}, 'OutputAfterFinalValue', 'Holding final value');
        catch
            % Some source blocks in this model are Repeating Sequence Stair,
            % not From Workspace. They do not expose this parameter.
        end
    end
end


function wire_goto_taps(model, BUS)
    wanted = [ ...
        strcat('V_Bus_', BUS), ...
        strcat('Bus_', BUS, '_Src_Pow'), ...
        strcat('Bus_', BUS, '_Temp')];

    for i = 1:numel(wanted)
        wire_one_goto_tap(model, wanted{i}, 0.01);
    end
end


function wire_one_goto_tap(model, tag, ts)
    vn = matlab.lang.makeValidName(tag);
    from_name = ['S36F_' vn];
    log_name = ['S36L_' vn];

    cleanup_named_block(model, from_name);
    cleanup_named_block(model, log_name);
    cleanup_workspace_variable_taps(model, vn);

    all_goto = find_system(model, 'FollowLinks', 'on', ...
        'LookUnderMasks', 'all', 'BlockType', 'Goto', 'GotoTag', tag);
    if isempty(all_goto)
        warning('No Goto tag found for %s.', tag);
        return;
    end

    add_block('simulink/Signal Routing/From', [model '/' from_name], ...
        'GotoTag', tag);
    add_block('simulink/Sinks/To Workspace', [model '/' log_name], ...
        'VariableName', vn, 'SaveFormat', 'Timeseries', ...
        'SampleTime', num2str(ts));

    pf = get_param([model '/' from_name], 'PortHandles');
    pl = get_param([model '/' log_name], 'PortHandles');
    add_line(model, pf.Outport(1), pl.Inport(1), 'autorouting', 'on');
end


function var_names = tap_gei_signals(model, BUS)
    block_map = containers.Map( ...
        {'A','B','C','D','E','F','G','H','K','L'}, ...
        {'Divide3','Divide4','Divide5','Divide13','Divide14', ...
         'Divide15','Divide16','Divide17','Divide18','Divide19'});

    var_names = cell(1, numel(BUS));
    for i = 1:numel(BUS)
        var_names{i} = sprintf('S36GEI_%s', BUS{i});
        block_path = [model '/' block_map(BUS{i})];
        if getSimulinkBlockHandle(block_path) == -1
            error('GEI block not found: %s', block_path);
        end
        ph = get_param(block_path, 'PortHandles');
        add_direct_tap(model, ['S36LOG_' var_names{i}], ...
            var_names{i}, ph.Outport(1), model, 0.01);
    end
end


function var_names = tap_phase_inports(model, CONV, ts)
    var_names = cell(1, numel(CONV));
    for i = 1:numel(CONV)
        var_names{i} = sprintf('S36PH_%s', CONV{i});
        dab_path = [model '/DAB_Model_' CONV{i}];
        phase_port = find_system(dab_path, 'FollowLinks', 'on', ...
            'LookUnderMasks', 'all', 'BlockType', 'Inport', ...
            'Name', 'phase_shift');
        if isempty(phase_port)
            error('phase_shift Inport not found in %s.', dab_path);
        end
        parent = get_param(phase_port{1}, 'Parent');
        ph = get_param(phase_port{1}, 'PortHandles');
        add_direct_tap(parent, ['S36LOG_' var_names{i}], ...
            var_names{i}, ph.Outport(1), model, ts);
    end
end


function var_names = tap_derate_factors(model, CONV, ts)
    var_names = cell(1, numel(CONV));
    for i = 1:numel(CONV)
        var_names{i} = sprintf('S36DER_%s', CONV{i});
        dab_path = [model '/DAB_Model_' CONV{i}];
        derater = [dab_path '/thermal_derater'];
        ph = get_param(derater, 'PortHandles');
        add_direct_tap(dab_path, ['S36LOG_' var_names{i}], ...
            var_names{i}, ph.Outport(1), model, ts);
    end
end


function var_names = tap_dab_output_ports(model, CONV, port_index, prefix, ts)
    var_names = cell(1, numel(CONV));
    for i = 1:numel(CONV)
        var_names{i} = sprintf('%s_%s', prefix, CONV{i});
        dab_path = [model '/DAB_Model_' CONV{i}];
        ph = get_param(dab_path, 'PortHandles');
        if numel(ph.Outport) < port_index
            error('%s does not have output port %d.', dab_path, port_index);
        end
        add_direct_tap(model, ['S36LOG_' var_names{i}], ...
            var_names{i}, ph.Outport(port_index), model, ts);
    end
end


function add_direct_tap(parent, block_name, var_name, src_port, model, ts)
    cleanup_named_block(parent, block_name);
    cleanup_workspace_variable_taps(model, var_name);

    add_block('simulink/Sinks/To Workspace', [parent '/' block_name], ...
        'VariableName', var_name, 'SaveFormat', 'Timeseries', ...
        'SampleTime', num2str(ts));
    ph = get_param([parent '/' block_name], 'PortHandles');
    add_line(parent, src_port, ph.Inport(1), 'autorouting', 'on');
end


function cleanup_named_block(parent, name)
    existing = find_system(parent, 'SearchDepth', 1, ...
        'LookUnderMasks', 'all', 'Name', name);
    for i = 1:numel(existing)
        try
            delete_block(existing{i});
        catch
        end
    end
end


function cleanup_workspace_variable_taps(model, var_name)
    stale = find_system(model, 'FollowLinks', 'on', ...
        'LookUnderMasks', 'all', 'BlockType', 'ToWorkspace', ...
        'VariableName', var_name);
    for i = 1:numel(stale)
        try
            delete_block(stale{i});
        catch
        end
    end
end


%% =========================================================================
% EXTRACTION
% =========================================================================
function y = grab_series(so, var_name, ts, n_steps)
    y = nan(n_steps, 1);
    v = try_get(so, matlab.lang.makeValidName(var_name));
    if isempty(v)
        return;
    end
    [~, col] = series_to_vector(v, ts);
    n = min(n_steps, numel(col));
    y(1:n) = col(1:n);
end


function v = try_get(so, name)
    v = [];
    try
        v = so.get(name);
        if ~isempty(v)
            return;
        end
    catch
    end
    try
        if evalin('base', sprintf('exist(''%s'',''var'')', name)) == 1
            v = evalin('base', name);
        end
    catch
    end
end


function [t, y] = series_to_vector(x, ts)
    if isa(x, 'timeseries')
        t = x.Time;
        y = squeeze(x.Data);
        y = y(:);
    elseif isnumeric(x) && size(x,2) >= 2
        t = x(:,1);
        y = x(:,2);
    elseif isnumeric(x)
        y = x(:);
        t = (0:numel(y)-1)' * ts;
    else
        t = [];
        y = [];
    end
end


%% =========================================================================
% QUALITY ASSESSMENT
% =========================================================================
function report = assess_dataset_quality( ...
    time_vec, PAVG, V, P, GEI, BT, TJ, HS, PH, DER, CONV, ...
    require_derating)

    report = struct();

    all_numeric = [time_vec, PAVG, V, P, GEI, BT, TJ, HS, PH, DER];
    report.nonfinite_count = sum(~isfinite(all_numeric), 'all');

    finite_pavg = PAVG(isfinite(PAVG));
    if isempty(finite_pavg)
        report.min_abs_avg_power_W = NaN;
    else
        report.min_abs_avg_power_W = min(abs(finite_pavg));
    end

    finite_gei = GEI(isfinite(GEI));
    if isempty(finite_gei)
        report.max_abs_gei = NaN;
    else
        report.max_abs_gei = max(abs(finite_gei));
    end

    report.al_max_abs_phase_deg = max(abs(PH(:,end)), [], 'omitnan');

    report.phase_sat_pct = nan(1, numel(CONV));
    for j = 1:numel(CONV)
        valid = isfinite(PH(:,j));
        if any(valid)
            report.phase_sat_pct(j) = mean(abs(PH(valid,j)) >= 89) * 100;
        end
    end

    expected = ones(size(DER));
    middle = TJ > 125 & TJ < 175;
    expected(middle) = (175 - TJ(middle)) / 50;
    expected(TJ >= 175) = 0;

    valid_thermal = isfinite(TJ) & isfinite(DER);
    if any(valid_thermal, 'all')
        report.max_derate_formula_error = max( ...
            abs(DER(valid_thermal) - expected(valid_thermal)));
    else
        report.max_derate_formula_error = NaN;
    end

    report.min_derate_factor = min(DER, [], 'all', 'omitnan');
    report.max_junction_temp_C = max(TJ, [], 'all', 'omitnan');
    report.max_heat_sink_temp_C = max(HS, [], 'all', 'omitnan');

    valid_der = isfinite(DER);
    n_valid_der = nnz(valid_der);
    if n_valid_der > 0
        report.derated_pct = 100 * nnz(DER(valid_der) < 0.99) / n_valid_der;
        report.transition_pct = 100 * nnz( ...
            DER(valid_der) > 0.05 & DER(valid_der) < 0.95) / n_valid_der;
        report.cutoff_pct = 100 * nnz(DER(valid_der) <= 0.01) / n_valid_der;
    else
        report.derated_pct = NaN;
        report.transition_pct = NaN;
        report.cutoff_pct = NaN;
    end

    report.fail_nonfinite = report.nonfinite_count > 0;
    report.fail_al_reference = ~isfinite(report.al_max_abs_phase_deg) || ...
        report.al_max_abs_phase_deg > 1e-6;
    report.fail_derate_formula = ~isfinite(report.max_derate_formula_error) || ...
        report.max_derate_formula_error > 1e-6;
    report.fail_no_derating = require_derating && ...
        (~isfinite(report.derated_pct) || report.derated_pct <= 0);

    report.warn_no_derating = ~isfinite(report.derated_pct) || ...
        report.derated_pct <= 0;
    report.warn_low_avg_power = ~isfinite(report.min_abs_avg_power_W) || ...
        report.min_abs_avg_power_W < 1000;
    report.warn_extreme_gei = ~isfinite(report.max_abs_gei) || ...
        report.max_abs_gei > 10;
    active_sat = report.phase_sat_pct(1:end-1);
    report.warn_high_phase_saturation = any( ...
        isfinite(active_sat) & active_sat > 20);
    report.warn_limited_transition_coverage = ...
        ~isfinite(report.transition_pct) || report.transition_pct < 0.1;

    report.pass = ~(report.fail_nonfinite || ...
        report.fail_al_reference || ...
        report.fail_derate_formula || ...
        report.fail_no_derating);
end


function print_quality_report(report, CONV)
    fprintf('\nMandatory checks:\n');
    fprintf('  Nonfinite values:               %d\n', report.nonfinite_count);
    fprintf('  AL maximum absolute phase:      %.9f deg\n', ...
        report.al_max_abs_phase_deg);
    fprintf('  Maximum derate-formula error:   %.3e\n', ...
        report.max_derate_formula_error);
    fprintf('  Rows/signals showing derating:  %.4f%%\n', report.derated_pct);

    fprintf('\nCoverage and stability indicators:\n');
    fprintf('  Minimum derate factor:          %.6f\n', report.min_derate_factor);
    fprintf('  Transition-region coverage:     %.4f%%\n', report.transition_pct);
    fprintf('  Full-cutoff coverage:           %.4f%%\n', report.cutoff_pct);
    fprintf('  Maximum junction temperature:   %.3f C\n', ...
        report.max_junction_temp_C);
    fprintf('  Maximum heat-sink temperature:  %.3f C\n', ...
        report.max_heat_sink_temp_C);
    fprintf('  Minimum |average source power|: %.3f W\n', ...
        report.min_abs_avg_power_W);
    fprintf('  Maximum |GEI|:                  %.6f\n', report.max_abs_gei);

    fprintf('\nPhase saturation percentages:\n');
    for j = 1:numel(CONV)
        fprintf('  %-3s: %8.3f%%\n', CONV{j}, report.phase_sat_pct(j));
    end

    if report.warn_no_derating
        fprintf('  WARNING: no derating occurred in this run.\n');
    end
    if report.warn_low_avg_power
        fprintf('  WARNING: common average source power approaches zero.\n');
    end
    if report.warn_extreme_gei
        fprintf('  WARNING: unusually large GEI values were observed.\n');
    end
    if report.warn_high_phase_saturation
        fprintf('  WARNING: at least one active phase is saturated for >20%%.\n');
    end
    if report.warn_limited_transition_coverage
        fprintf('  WARNING: very little 0.05-to-0.95 derating transition data.\n');
    end
end


function write_quality_report(filename, report, run_mode, tend, ts, ...
    source_model, dataset_model, CONV, clip_hi, clip_lo, wall_s)

    fid = fopen(filename, 'w');
    if fid == -1
        warning('Could not open quality report for writing: %s', filename);
        return;
    end
    cleanup = onCleanup(@() fclose(fid));

    fprintf(fid, 'SAFE THERMAL-DERATING DATASET QUALITY REPORT\n');
    fprintf(fid, '=============================================\n');
    fprintf(fid, 'Run mode: %s\n', upper(char(run_mode)));
    fprintf(fid, 'Duration: %.0f s\n', tend);
    fprintf(fid, 'Sample time: %.6f s\n', ts);
    fprintf(fid, 'Wall time: %.3f s\n', wall_s);
    fprintf(fid, 'Original model: %s\n', source_model);
    fprintf(fid, 'Dataset-only model: %s\n', dataset_model);
    fprintf(fid, ['Dataset model changes: AL=0 degree reference; thermal derater ' ...
        'corrected to <=125:1, 125-175:linear, >=175:0.\n\n']);

    fprintf(fid, 'FINAL STATUS: %s\n\n', ternary(report.pass, 'PASS', 'REVIEW'));

    fprintf(fid, 'Mandatory checks\n');
    fprintf(fid, '  nonfinite_count: %d\n', report.nonfinite_count);
    fprintf(fid, '  al_max_abs_phase_deg: %.12g\n', report.al_max_abs_phase_deg);
    fprintf(fid, '  max_derate_formula_error: %.12g\n', ...
        report.max_derate_formula_error);
    fprintf(fid, '  derated_pct: %.12g\n', report.derated_pct);

    fprintf(fid, '\nCoverage\n');
    fprintf(fid, '  min_derate_factor: %.12g\n', report.min_derate_factor);
    fprintf(fid, '  transition_pct: %.12g\n', report.transition_pct);
    fprintf(fid, '  cutoff_pct: %.12g\n', report.cutoff_pct);
    fprintf(fid, '  max_junction_temp_C: %.12g\n', report.max_junction_temp_C);
    fprintf(fid, '  max_heat_sink_temp_C: %.12g\n', ...
        report.max_heat_sink_temp_C);
    fprintf(fid, '  min_abs_avg_power_W: %.12g\n', ...
        report.min_abs_avg_power_W);
    fprintf(fid, '  max_abs_gei: %.12g\n', report.max_abs_gei);

    fprintf(fid, '\nPhase saturation\n');
    for j = 1:numel(CONV)
        fprintf(fid, '  %s: %.12g percent\n', CONV{j}, ...
            report.phase_sat_pct(j));
    end

    fprintf(fid, '\nLoad clipping\n');
    for j = 1:numel(CONV)
        fprintf(fid, '  bus_position_%d: high=%.6f%% low=%.6f%%\n', ...
            j, clip_hi(j), clip_lo(j));
    end

    fprintf(fid, '\nWarnings\n');
    fprintf(fid, '  no_derating: %d\n', report.warn_no_derating);
    fprintf(fid, '  low_average_power: %d\n', report.warn_low_avg_power);
    fprintf(fid, '  extreme_gei: %d\n', report.warn_extreme_gei);
    fprintf(fid, '  high_phase_saturation: %d\n', ...
        report.warn_high_phase_saturation);
    fprintf(fid, '  limited_transition_coverage: %d\n', ...
        report.warn_limited_transition_coverage);
end


function out = ternary(condition, value_if_true, value_if_false)
    if condition
        out = value_if_true;
    else
        out = value_if_false;
    end
end
