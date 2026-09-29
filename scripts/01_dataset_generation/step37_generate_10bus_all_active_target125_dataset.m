function step37_generate_10bus_all_active_target125_dataset(run_mode)
% =========================================================================
% STEP 37: ALL-ACTIVE 10-BUS THERMAL-DERATING DATASET, TARGET 125 C
% =========================================================================
% PURPOSE
%   Generate an experimental dataset-only model in which:
%     1. AL is active and is not fixed at zero.
%     2. All ten DAB phase commands are checked for sustained saturation.
%     3. Junction temperature rises from 25 C toward and through the
%        125 C thermal-derating threshold.
%
% CONTROL CONFIGURATION USED ONLY IN THE DATASET COPY
%   The ten adjacent ring-control errors contain nine independent integral
%   directions. To keep AL active without restoring the redundant tenth
%   integrator, PID Controller9, which drives AL, is changed to proportional
%   only by setting I = 0. The other nine controllers retain their original
%   PI gains. Built-in back-calculation anti-windup is enabled on the nine
%   PI controllers. This is an experimental dataset configuration, not an
%   approved permanent replacement for Apoorv's control design.
%
% THERMAL EXCITATION
%   Previous runs reached only 79.8 C even while DE, EF, and FG were heavily
%   saturated. This script therefore does not force more electrical stress.
%   Instead, it simulates a documented degraded-cooling condition by
%   increasing heat-sink-to-ambient thermal resistance R_ha in the copied
%   model. Pilot mode automatically calibrates R_ha so the hottest junction
%   reaches approximately 128 C. Full mode runs five independent 1000 s
%   episodes around the calibrated cooling resistance for broader coverage.
%
% ORIGINAL MODEL SAFETY
%   Grid_modelling_Thermal_V7.slx is never overwritten. The script creates:
%     Grid_modelling_Thermal_V7_AllActive_Target125.slx
%
% RUN ORDER
%   1. Pilot and automatic R_ha calibration:
%        step37_generate_10bus_all_active_target125_dataset('pilot')
%
%   2. Run full only after the pilot quality report is acceptable:
%        step37_generate_10bus_all_active_target125_dataset('full')
%
% TRAINING RULE
%   Train only on a file ending in _PASS.csv. A _REVIEW.csv file is a
%   diagnostic output and must not be used for model training.
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
    DATASET_MODEL = 'Grid_modelling_Thermal_V7_AllActive_Target125';

    TS = 0.01;
    EPISODE_TEND = 1000;
    N_EPISODE_STEPS = round(EPISODE_TEND / TS) + 1;

    if run_mode == "pilot"
        N_EPISODES = 1;
    else
        N_EPISODES = 5;
    end

    TARGET_MAX_TJ_C = 128;
    ACCEPTABLE_MAX_TJ_LOW_C = 125;
    ACCEPTABLE_MAX_TJ_HIGH_C = 150;
    MAX_ALLOWED_PHASE_SAT_PCT = 5;
    MIN_AL_PHASE_RANGE_DEG = 0.10;

    PMIN = 25000;
    PMAX = 80000;

    BUS  = {'A','B','C','D','E','F','G','H','K','L'};
    CONV = {'AB','BC','CD','DE','EF','FG','GH','HK','KL','AL'};
    NB = numel(BUS);
    NC = numel(CONV);

    OUTPUT_DIR = fullfile(PROJECT_ROOT, 'model_outputs', ...
        'thermal_derating_all_active_target125');
    if exist(OUTPUT_DIR, 'dir') ~= 7
        mkdir(OUTPUT_DIR);
    end
    CALIBRATION_FILE = fullfile(OUTPUT_DIR, 'step37_rha_calibration.mat');
    addpath(PROJECT_ROOT);

    fprintf('\n============================================================\n');
    fprintf('STEP 37: ALL-ACTIVE TARGET-125 DATASET (%s)\n', ...
        upper(char(run_mode)));
    fprintf('============================================================\n');

    %% 1. CREATE EXPERIMENTAL MODEL COPY -----------------------------------
    fprintf('\n[1/8] Creating experimental dataset model copy ...\n');
    dataset_model_file = prepare_dataset_model_copy( ...
        PROJECT_ROOT, SOURCE_MODEL, DATASET_MODEL, CONV);
    fprintf('Dataset model: %s\n', dataset_model_file);

    %% 2. LOAD MODEL AND WIRE TAPS -----------------------------------------
    fprintf('\n[2/8] Loading model and wiring signal taps ...\n');
    if bdIsLoaded(DATASET_MODEL)
        close_system(DATASET_MODEL, 0);
    end
    load_system(dataset_model_file);
    force_final_value_hold(DATASET_MODEL);
    wire_goto_taps(DATASET_MODEL, BUS, TS);

    gei_var_names = tap_gei_signals(DATASET_MODEL, BUS, TS);
    phase_var_names = tap_phase_inports(DATASET_MODEL, CONV, TS);
    derate_var_names = tap_derate_factors(DATASET_MODEL, CONV, TS);
    junction_var_names = tap_dab_output_ports( ...
        DATASET_MODEL, CONV, 4, 'S37TJ', TS);
    heat_sink_var_names = tap_dab_output_ports( ...
        DATASET_MODEL, CONV, 5, 'S37HS', TS);
    wire_one_goto_tap(DATASET_MODEL, 'Src_Pow_Avg', TS);
    avg_power_var_name = matlab.lang.makeValidName('Src_Pow_Avg');
    save_system(DATASET_MODEL);

    %% 3. CALIBRATE OR LOAD COOLING RESISTANCE -----------------------------
    fprintf('\n[3/8] Resolving degraded-cooling resistance for target temperature ...\n');
    if run_mode == "pilot"
        calibration = calibrate_rha( ...
            DATASET_MODEL, CONV, BUS, TS, EPISODE_TEND, ...
            N_EPISODE_STEPS, PMIN, PMAX, TARGET_MAX_TJ_C, ...
            MAX_ALLOWED_PHASE_SAT_PCT, phase_var_names, ...
            junction_var_names, 3701);
        save(CALIBRATION_FILE, 'calibration');
        fprintf('Saved calibration: %s\n', CALIBRATION_FILE);
    else
        if exist(CALIBRATION_FILE, 'file') ~= 2
            close_system(DATASET_MODEL, 0);
            error(['Pilot calibration file is missing. Run first:\n' ...
                '  step37_generate_10bus_all_active_target125_dataset(''pilot'')']);
        end
        loaded = load(CALIBRATION_FILE, 'calibration');
        calibration = loaded.calibration;
    end

    fprintf('Selected base R_ha: %.6f K/W\n', calibration.selected_rha_K_per_W);
    fprintf('Pilot maximum Tj:   %.3f C\n', calibration.selected_max_tj_C);
    fprintf('Pilot max phase saturation: %.3f%%\n', ...
        calibration.selected_max_phase_sat_pct);

    %% 4. RUN DATASET EPISODES ---------------------------------------------
    fprintf('\n[4/8] Running %d independent dataset episode(s) ...\n', N_EPISODES);

    % Full mode spans mild through stronger degraded-cooling conditions.
    if N_EPISODES == 1
        episode_rha_multipliers = 1.0;
    else
        episode_rha_multipliers = [0.86, 0.93, 1.00, 1.07, 1.14];
    end

    % Small converter-to-converter variation creates staggered thermal
    % activation without requiring large electrical imbalance.
    conv_rha_multipliers = [0.95, 0.98, 1.00, 1.03, 1.06, ...
                            1.04, 1.01, 0.97, 1.05, 0.96];

    pieces = cell(N_EPISODES, 1);
    episode_reports = repmat(struct(), N_EPISODES, 1);
    total_wall_s = 0;

    for ep = 1:N_EPISODES
        fprintf('\n  Episode %d/%d\n', ep, N_EPISODES);
        base_rha = calibration.selected_rha_K_per_W * ...
            episode_rha_multipliers(ep);
        rha_values = base_rha * conv_rha_multipliers;
        rha_values = max(0.50, min(5.00, rha_values));

        set_cooling_resistance(DATASET_MODEL, CONV, rha_values);
        reset_thermal_initial_conditions(DATASET_MODEL, CONV, 25, 25);

        [load_hist, stage_id, clip_hi, clip_lo] = ...
            build_balanced_excitation_loads( ...
                3700 + ep, N_EPISODE_STEPS, TS, EPISODE_TEND, ...
                BUS, PMIN, PMAX, ep);

        fprintf('    R_ha range: %.4f to %.4f K/W\n', ...
            min(rha_values), max(rha_values));
        fprintf('    Mean load high clipping: %.3f%%\n', mean(clip_hi));
        fprintf('    Mean load low clipping:  %.3f%%\n', mean(clip_lo));

        tic;
        so = sim(DATASET_MODEL, 'StopTime', num2str(EPISODE_TEND), ...
            'ReturnWorkspaceOutputs', 'on');
        wall_s = toc;
        total_wall_s = total_wall_s + wall_s;
        fprintf('    Simulation wall time: %.2f s\n', wall_s);

        data = extract_episode( ...
            so, TS, N_EPISODE_STEPS, BUS, CONV, ...
            avg_power_var_name, gei_var_names, phase_var_names, ...
            derate_var_names, junction_var_names, heat_sink_var_names);

        % Drop the initial algebraic-startup sample. Earlier runs showed one
        % undefined GEI value per bus at t = 0 because average source power
        % is exactly zero before the electrical states initialize.
        keep = 2:N_EPISODE_STEPS;
        global_time_offset = (ep - 1) * EPISODE_TEND;

        piece = table();
        piece.time = global_time_offset + data.time(keep);
        piece.Scenario_Time = data.time(keep);
        piece.Scenario_ID = repmat(ep, numel(keep), 1);
        piece.Thermal_Stage_ID = stage_id(keep);
        piece.Src_Pow_Avg = data.PAVG(keep);
        piece.Base_Cooling_Rha_K_per_W = repmat(base_rha, numel(keep), 1);

        for i = 1:NB
            piece.(sprintf('CommandedLoad_kW_%s', BUS{i})) = ...
                load_hist(keep,i) / 1000;
            piece.(sprintf('Bus_%s_Src_Pow', BUS{i})) = data.P(keep,i);
            piece.(sprintf('GEI_%s', BUS{i})) = data.GEI(keep,i);
            piece.(sprintf('V_Bus_%s', BUS{i})) = data.V(keep,i);
            piece.(sprintf('Bus_%s_Temp', BUS{i})) = data.BT(keep,i);
        end

        for j = 1:NC
            piece.(sprintf('Cooling_Rha_K_per_W_%s', CONV{j})) = ...
                repmat(rha_values(j), numel(keep), 1);
            piece.(sprintf('JunctionTemp_C_%s', CONV{j})) = data.TJ(keep,j);
            piece.(sprintf('HeatSinkTemp_C_%s', CONV{j})) = data.HS(keep,j);
            piece.(sprintf('Phase_%s_cmd_deg', CONV{j})) = data.PH(keep,j);
            piece.(sprintf('DAB_%s_Derate_Factor', CONV{j})) = data.DER(keep,j);
        end

        pieces{ep} = piece;
        episode_reports(ep) = assess_dataset_quality( ...
            data.time(keep), data.PAVG(keep), data.V(keep,:), ...
            data.P(keep,:), data.GEI(keep,:), data.BT(keep,:), ...
            data.TJ(keep,:), data.HS(keep,:), data.PH(keep,:), ...
            data.DER(keep,:), CONV, ACCEPTABLE_MAX_TJ_LOW_C, ...
            ACCEPTABLE_MAX_TJ_HIGH_C, MAX_ALLOWED_PHASE_SAT_PCT, ...
            MIN_AL_PHASE_RANGE_DEG);

        fprintf('    Episode max Tj: %.3f C\n', ...
            episode_reports(ep).max_junction_temp_C);
        fprintf('    Episode derated coverage: %.4f%%\n', ...
            episode_reports(ep).derated_pct);
        fprintf('    Episode maximum phase saturation: %.3f%%\n', ...
            max(episode_reports(ep).phase_sat_pct));
    end

    %% 5. COMBINE DATA -----------------------------------------------------
    fprintf('\n[5/8] Combining episode data ...\n');
    out = vertcat(pieces{:});
    fprintf('Combined rows: %d\n', height(out));

    %% 6. FINAL QUALITY GATE -----------------------------------------------
    fprintf('\n[6/8] Running final dataset quality gate ...\n');
    [V, P, GEI, BT, TJ, HS, PH, DER, PAVG] = ...
        table_to_quality_arrays(out, BUS, CONV);

    final_report = assess_dataset_quality( ...
        out.time, PAVG, V, P, GEI, BT, TJ, HS, PH, DER, CONV, ...
        ACCEPTABLE_MAX_TJ_LOW_C, ACCEPTABLE_MAX_TJ_HIGH_C, ...
        MAX_ALLOWED_PHASE_SAT_PCT, MIN_AL_PHASE_RANGE_DEG);
    print_quality_report(final_report, CONV);

    %% 7. WRITE OUTPUTS ----------------------------------------------------
    fprintf('\n[7/8] Writing dataset and report ...\n');
    status_text = 'REVIEW';
    if final_report.pass
        status_text = 'PASS';
    end

    base_name = sprintf('thermal_derating_all_active_%s_%deps_%s', ...
        char(run_mode), N_EPISODES, status_text);
    out_csv = fullfile(OUTPUT_DIR, [base_name '.csv']);
    writetable(out, out_csv);

    report_txt = fullfile(OUTPUT_DIR, [base_name '_quality_report.txt']);
    write_quality_report(report_txt, final_report, episode_reports, ...
        run_mode, N_EPISODES, EPISODE_TEND, TS, SOURCE_MODEL, ...
        DATASET_MODEL, CONV, calibration, total_wall_s);

    file_info = dir(out_csv);
    fprintf('Dataset:        %s\n', out_csv);
    fprintf('Quality report: %s\n', report_txt);
    fprintf('CSV size: %.1f MB\n', file_info.bytes / 1e6);

    %% 8. FINAL STATUS -----------------------------------------------------
    fprintf('\n[8/8] Final status ...\n');
    if final_report.pass
        fprintf('FINAL STATUS: PASS\n');
        if run_mode == "pilot"
            fprintf(['AL is active, phase saturation is within the quality ' ...
                'limit, and the thermal threshold was reached. You may run:\n' ...
                '  step37_generate_10bus_all_active_target125_dataset(''full'')\n']);
        else
            fprintf('The full all-active thermal-derating dataset passed.\n');
        end
    else
        fprintf('FINAL STATUS: REVIEW, DO NOT TRAIN ON THIS CSV.\n');
        fprintf('Use the quality report to identify the remaining failed check.\n');
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

    if exist(dataset_model_file, 'file') == 2
        delete(dataset_model_file);
    end
    [ok, msg] = copyfile(source_model_file, dataset_model_file, 'f');
    if ~ok
        error('Could not create experimental model copy: %s', msg);
    end

    load_system(dataset_model_file);
    loaded_roots = find_system('SearchDepth', 0, 'Type', 'block_diagram');
    if ~ismember(dataset_model, loaded_roots)
        close_system('all', 0);
        error('Dataset model copy did not load using name %s.', dataset_model);
    end

    patch_deraters(dataset_model, CONV);
    configure_all_active_controllers(dataset_model);
    reset_thermal_initial_conditions(dataset_model, CONV, 25, 25);

    set_param(dataset_model, 'Description', sprintf([ ...
        'EXPERIMENTAL DATASET-ONLY COPY created by step37. Original V7 preserved.\n' ...
        'AL remains active but PID9 uses P-only control so there are nine ' ...
        'integral loops. Other PIs use back-calculation anti-windup.\n' ...
        'Thermal derater corrected to <=125:1, 125-175:linear, >=175:0.\n' ...
        'R_ha is adjusted only to simulate documented degraded cooling.']));

    save_system(dataset_model);
    close_system(dataset_model, 0);
end


function patch_deraters(model, CONV)
    fprintf('  Correcting thermal deraters ...\n');
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
        found_linear = false;
        found_hot = false;

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
                set_param(constants{1}, 'Value', '1');
                found_healthy = true;
            elseif contains(label, '> 175') && isempty(inports)
                set_param(constants{1}, 'Value', '0');
                found_hot = true;
            elseif ~isempty(inports)
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
            error('Failed to patch all derating branches for %s.', CONV{i});
        end
    end
end


function configure_all_active_controllers(model)
    fprintf('  Configuring all-active controller copy ...\n');

    pid_names = {'PID Controller','PID Controller1','PID Controller2', ...
        'PID Controller3','PID Controller4','PID Controller5', ...
        'PID Controller6','PID Controller7','PID Controller8', ...
        'PID Controller9'};

    for i = 1:numel(pid_names)
        pid_path = [model '/' pid_names{i}];
        if getSimulinkBlockHandle(pid_path) == -1
            error('Missing controller block: %s', pid_path);
        end

        % Keep the original PI gains for the first nine controllers. Limit
        % output and unwind their integrators if the power command reaches
        % its bound. Balanced loads are still required so these limits are
        % rarely active.
        set_param(pid_path, ...
            'LimitOutput', 'on', ...
            'UpperSaturationLimit', '60e3', ...
            'LowerSaturationLimit', '-60e3');

        if i < 10
            set_param(pid_path, ...
                'P', '0.2', 'I', '0.1', ...
                'AntiWindupMode', 'back-calculation', 'Kb', '1');
        else
            % AL remains active, but its redundant integral state is
            % removed. Its phase can move through proportional action.
            set_param(pid_path, ...
                'P', '0.2', 'I', '0', ...
                'AntiWindupMode', 'none');
        end
    end

    % Explicitly reconnect AL to bus_l_cntrl in case the source model was
    % previously opened or edited by another dataset script.
    al_block = [model '/DAB_Model_AL'];
    ph_al = get_param(al_block, 'PortHandles');
    al_phase_in = ph_al.Inport(1);
    old_line = get_param(al_phase_in, 'Line');
    if old_line ~= -1
        delete_line(old_line);
    end

    from_name = 'S37_AL_bus_l_cntrl';
    cleanup_named_block(model, from_name);
    from_path = [model '/' from_name];
    add_block('simulink/Signal Routing/From', from_path, ...
        'GotoTag', 'bus_l_cntrl', ...
        'Position', [-2095, 3745, -1995, 3775]);
    ph_from = get_param(from_path, 'PortHandles');
    add_line(model, ph_from.Outport(1), al_phase_in, 'autorouting', 'on');

    fprintf('    AL is active through bus_l_cntrl with P-only control.\n');
    fprintf('    Nine PI controllers use back-calculation anti-windup.\n');
end


function reset_thermal_initial_conditions(model, CONV, tj0, th0)
    for i = 1:numel(CONV)
        thermal = [model '/DAB_Model_' CONV{i} '/DAB_Thermal_Model'];
        set_param([thermal '/Integrator'], 'InitialCondition', num2str(tj0));
        set_param([thermal '/Integrator1'], 'InitialCondition', num2str(th0));
    end
end


function set_cooling_resistance(model, CONV, rha_values)
    if isscalar(rha_values)
        rha_values = repmat(rha_values, 1, numel(CONV));
    end
    if numel(rha_values) ~= numel(CONV)
        error('rha_values must be scalar or contain one value per converter.');
    end

    for i = 1:numel(CONV)
        if rha_values(i) <= 0
            error('R_ha must be positive.');
        end
        block_path = [model '/DAB_Model_' CONV{i} ...
            '/DAB_Thermal_Model/R_ha'];
        set_param(block_path, 'Gain', sprintf('1/(%.12g)', rha_values(i)));
    end
end


%% =========================================================================
% R_HA CALIBRATION
% =========================================================================
function calibration = calibrate_rha( ...
    model, CONV, BUS, ts, tend, n_steps, pmin, pmax, target_tj, ...
    max_allowed_phase_sat_pct, phase_var_names, junction_var_names, seed)

    fprintf('  Automatic R_ha calibration toward %.1f C ...\n', target_tj);
    low = 0.50;
    high = 5.00;
    n_iterations = 5;

    candidate_rha = nan(n_iterations, 1);
    candidate_max_tj = nan(n_iterations, 1);
    candidate_max_sat = nan(n_iterations, 1);
    candidate_score = inf(n_iterations, 1);

    for it = 1:n_iterations
        rha = (low + high) / 2;
        set_cooling_resistance(model, CONV, rha);
        reset_thermal_initial_conditions(model, CONV, 25, 25);
        build_balanced_excitation_loads( ...
            seed, n_steps, ts, tend, BUS, pmin, pmax, 1);

        tic;
        so = sim(model, 'StopTime', num2str(tend), ...
            'ReturnWorkspaceOutputs', 'on');
        wall_s = toc;

        max_tj = -inf;
        max_sat = 0;
        for j = 1:numel(CONV)
            tj = grab_series(so, junction_var_names{j}, ts, n_steps);
            ph = grab_series(so, phase_var_names{j}, ts, n_steps);
            tj = tj(2:end);
            ph = ph(2:end);
            max_tj = max(max_tj, max(tj, [], 'omitnan'));
            valid = isfinite(ph);
            if any(valid)
                max_sat = max(max_sat, mean(abs(ph(valid)) >= 89) * 100);
            end
        end

        saturation_penalty = max(0, max_sat - max_allowed_phase_sat_pct);
        score = abs(max_tj - target_tj) + 20 * saturation_penalty;

        candidate_rha(it) = rha;
        candidate_max_tj(it) = max_tj;
        candidate_max_sat(it) = max_sat;
        candidate_score(it) = score;

        fprintf(['    Iteration %d: R_ha=%.5f K/W, max Tj=%.3f C, ' ...
            'max phase saturation=%.3f%%, wall=%.1f s\n'], ...
            it, rha, max_tj, max_sat, wall_s);

        if max_tj < target_tj
            low = rha;
        else
            high = rha;
        end
    end

    valid_sat = candidate_max_sat <= max_allowed_phase_sat_pct;
    if any(valid_sat)
        valid_scores = candidate_score;
        valid_scores(~valid_sat) = inf;
        [~, best] = min(valid_scores);
    else
        [~, best] = min(candidate_score);
    end

    calibration = struct();
    calibration.target_max_tj_C = target_tj;
    calibration.selected_rha_K_per_W = candidate_rha(best);
    calibration.selected_max_tj_C = candidate_max_tj(best);
    calibration.selected_max_phase_sat_pct = candidate_max_sat(best);
    calibration.candidate_rha_K_per_W = candidate_rha;
    calibration.candidate_max_tj_C = candidate_max_tj;
    calibration.candidate_max_phase_sat_pct = candidate_max_sat;
    calibration.candidate_score = candidate_score;
end


%% =========================================================================
% BALANCED LOAD GENERATION
% =========================================================================
function [load_hist, stage_id, clip_hi, clip_lo] = ...
    build_balanced_excitation_loads( ...
    seed, n_steps, ts, tend, BUS, pmin, pmax, scenario_id)

    rng(seed);
    t = (0:n_steps-1)' * ts;
    NB = numel(BUS);

    % Common demand heats the grid without creating a large static bus-to-
    % bus mismatch. Rotating zero-sum components create moderate DAB power
    % transfer and diverse phase commands without the monotonic imbalance
    % that previously drove DE, EF, and FG to the rails.
    stage_id = ones(n_steps, 1);
    stage_id(t >= 100 & t < 650) = 2;
    stage_id(t >= 650 & t < 820) = 3;
    stage_id(t >= 820) = 4;

    common = 42000 * ones(n_steps, 1);
    common(stage_id == 2) = 62000;
    common(stage_id == 3) = 50000;
    common(stage_id == 4) = 38000;

    fixed_offsets = [4000, 3000, 2000, 1000, 0, ...
                    -1000, -2000, -3000, -4000, 0];
    fixed_offsets = circshift(fixed_offsets, scenario_id - 1);

    load_hist = nan(n_steps, NB);
    clip_hi = nan(1, NB);
    clip_lo = nan(1, NB);

    for k = 1:NB
        phase = 2*pi*(k-1)/NB + 0.20*scenario_id;
        rotating_slow = 7500 * sin(2*pi*t/260 + phase);
        rotating_fast = 2200 * sin(2*pi*t/70 + 2*phase);
        local_wave = 900 * sin(2*pi*t/(43 + k) + 0.3*k);
        noise = 300 * randn(n_steps, 1);

        raw = common + fixed_offsets(k) + rotating_slow + ...
            rotating_fast + local_wave + noise;

        clip_hi(k) = mean(raw > pmax) * 100;
        clip_lo(k) = mean(raw < pmin) * 100;
        p = max(pmin, min(pmax, raw));
        load_hist(:,k) = p;
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
            set_param(fw_blocks{i}, 'OutputAfterFinalValue', ...
                'Holding final value');
        catch
        end
    end
end


function wire_goto_taps(model, BUS, ts)
    wanted = [strcat('V_Bus_', BUS), ...
              strcat('Bus_', BUS, '_Src_Pow'), ...
              strcat('Bus_', BUS, '_Temp')];
    for i = 1:numel(wanted)
        wire_one_goto_tap(model, wanted{i}, ts);
    end
end


function wire_one_goto_tap(model, tag, ts)
    vn = matlab.lang.makeValidName(tag);
    from_name = ['S37F_' vn];
    log_name = ['S37L_' vn];

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


function var_names = tap_gei_signals(model, BUS, ts)
    block_map = containers.Map( ...
        {'A','B','C','D','E','F','G','H','K','L'}, ...
        {'Divide3','Divide4','Divide5','Divide13','Divide14', ...
         'Divide15','Divide16','Divide17','Divide18','Divide19'});

    var_names = cell(1, numel(BUS));
    for i = 1:numel(BUS)
        var_names{i} = sprintf('S37GEI_%s', BUS{i});
        block_path = [model '/' block_map(BUS{i})];
        ph = get_param(block_path, 'PortHandles');
        add_direct_tap(model, ['S37LOG_' var_names{i}], ...
            var_names{i}, ph.Outport(1), model, ts);
    end
end


function var_names = tap_phase_inports(model, CONV, ts)
    var_names = cell(1, numel(CONV));
    for i = 1:numel(CONV)
        var_names{i} = sprintf('S37PH_%s', CONV{i});
        dab_path = [model '/DAB_Model_' CONV{i}];
        phase_port = find_system(dab_path, 'FollowLinks', 'on', ...
            'LookUnderMasks', 'all', 'BlockType', 'Inport', ...
            'Name', 'phase_shift');
        parent = get_param(phase_port{1}, 'Parent');
        ph = get_param(phase_port{1}, 'PortHandles');
        add_direct_tap(parent, ['S37LOG_' var_names{i}], ...
            var_names{i}, ph.Outport(1), model, ts);
    end
end


function var_names = tap_derate_factors(model, CONV, ts)
    var_names = cell(1, numel(CONV));
    for i = 1:numel(CONV)
        var_names{i} = sprintf('S37DER_%s', CONV{i});
        dab_path = [model '/DAB_Model_' CONV{i}];
        derater = [dab_path '/thermal_derater'];
        ph = get_param(derater, 'PortHandles');
        add_direct_tap(dab_path, ['S37LOG_' var_names{i}], ...
            var_names{i}, ph.Outport(1), model, ts);
    end
end


function var_names = tap_dab_output_ports(model, CONV, port_index, prefix, ts)
    var_names = cell(1, numel(CONV));
    for i = 1:numel(CONV)
        var_names{i} = sprintf('%s_%s', prefix, CONV{i});
        dab_path = [model '/DAB_Model_' CONV{i}];
        ph = get_param(dab_path, 'PortHandles');
        add_direct_tap(model, ['S37LOG_' var_names{i}], ...
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
function data = extract_episode( ...
    so, ts, n_steps, BUS, CONV, avg_power_var_name, gei_var_names, ...
    phase_var_names, derate_var_names, junction_var_names, ...
    heat_sink_var_names)

    NB = numel(BUS);
    NC = numel(CONV);
    data = struct();
    data.time = (0:n_steps-1)' * ts;
    data.PAVG = grab_series(so, avg_power_var_name, ts, n_steps);
    data.V = nan(n_steps, NB);
    data.P = nan(n_steps, NB);
    data.GEI = nan(n_steps, NB);
    data.BT = nan(n_steps, NB);
    data.PH = nan(n_steps, NC);
    data.DER = nan(n_steps, NC);
    data.TJ = nan(n_steps, NC);
    data.HS = nan(n_steps, NC);

    for i = 1:NB
        data.V(:,i) = grab_series(so, sprintf('V_Bus_%s', BUS{i}), ts, n_steps);
        data.P(:,i) = grab_series(so, sprintf('Bus_%s_Src_Pow', BUS{i}), ts, n_steps);
        data.GEI(:,i) = grab_series(so, gei_var_names{i}, ts, n_steps);
        data.BT(:,i) = grab_series(so, sprintf('Bus_%s_Temp', BUS{i}), ts, n_steps);
    end

    for j = 1:NC
        data.PH(:,j) = grab_series(so, phase_var_names{j}, ts, n_steps);
        raw_derate = grab_series(so, derate_var_names{j}, ts, n_steps);
        data.DER(:,j) = raw_derate / 100e3;
        data.TJ(:,j) = grab_series(so, junction_var_names{j}, ts, n_steps);
        data.HS(:,j) = grab_series(so, heat_sink_var_names{j}, ts, n_steps);
    end
end


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
    acceptable_tj_low, acceptable_tj_high, max_phase_sat_pct, ...
    min_al_phase_range)

    report = struct();
    all_numeric = [time_vec, PAVG, V, P, GEI, BT, TJ, HS, PH, DER];
    report.nonfinite_count = sum(~isfinite(all_numeric), 'all');

    finite_pavg = PAVG(isfinite(PAVG));
    report.min_abs_avg_power_W = min(abs(finite_pavg), [], 'omitnan');
    finite_gei = GEI(isfinite(GEI));
    report.max_abs_gei = max(abs(finite_gei), [], 'omitnan');

    report.phase_sat_pct = nan(1, numel(CONV));
    report.phase_range_deg = nan(1, numel(CONV));
    for j = 1:numel(CONV)
        valid = isfinite(PH(:,j));
        if any(valid)
            p = PH(valid,j);
            report.phase_sat_pct(j) = mean(abs(p) >= 89) * 100;
            report.phase_range_deg(j) = max(p) - min(p);
        end
    end

    report.al_phase_range_deg = report.phase_range_deg(end);
    report.al_max_abs_phase_deg = max(abs(PH(:,end)), [], 'omitnan');

    expected = ones(size(DER));
    middle = TJ > 125 & TJ < 175;
    expected(middle) = (175 - TJ(middle)) / 50;
    expected(TJ >= 175) = 0;
    valid_thermal = isfinite(TJ) & isfinite(DER);
    report.max_derate_formula_error = max( ...
        abs(DER(valid_thermal) - expected(valid_thermal)), [], 'omitnan');

    report.min_derate_factor = min(DER, [], 'all', 'omitnan');
    report.max_junction_temp_C = max(TJ, [], 'all', 'omitnan');
    report.max_heat_sink_temp_C = max(HS, [], 'all', 'omitnan');

    valid_der = isfinite(DER);
    n_valid_der = nnz(valid_der);
    report.derated_pct = 100 * nnz(DER(valid_der) < 0.99) / n_valid_der;
    report.transition_pct = 100 * nnz( ...
        DER(valid_der) > 0.05 & DER(valid_der) < 0.95) / n_valid_der;
    report.cutoff_pct = 100 * nnz(DER(valid_der) <= 0.01) / n_valid_der;

    report.fail_nonfinite = report.nonfinite_count > 0;
    report.fail_al_inactive = ~isfinite(report.al_phase_range_deg) || ...
        report.al_phase_range_deg < min_al_phase_range;
    report.fail_phase_saturation = any( ...
        ~isfinite(report.phase_sat_pct) | ...
        report.phase_sat_pct > max_phase_sat_pct);
    report.fail_temperature_target = ...
        ~isfinite(report.max_junction_temp_C) || ...
        report.max_junction_temp_C < acceptable_tj_low || ...
        report.max_junction_temp_C > acceptable_tj_high;
    report.fail_derate_formula = ...
        ~isfinite(report.max_derate_formula_error) || ...
        report.max_derate_formula_error > 1e-6;
    report.fail_no_derating = ...
        ~isfinite(report.derated_pct) || report.derated_pct <= 0;

    report.warn_low_avg_power = ...
        ~isfinite(report.min_abs_avg_power_W) || ...
        report.min_abs_avg_power_W < 1000;
    report.warn_extreme_gei = ...
        ~isfinite(report.max_abs_gei) || report.max_abs_gei > 10;
    report.warn_limited_transition_coverage = ...
        ~isfinite(report.transition_pct) || report.transition_pct < 0.1;

    report.pass = ~(report.fail_nonfinite || ...
        report.fail_al_inactive || ...
        report.fail_phase_saturation || ...
        report.fail_temperature_target || ...
        report.fail_derate_formula || ...
        report.fail_no_derating);
end


function print_quality_report(report, CONV)
    fprintf('\nMandatory checks:\n');
    fprintf('  Nonfinite values:               %d\n', report.nonfinite_count);
    fprintf('  AL phase range:                 %.6f deg\n', ...
        report.al_phase_range_deg);
    fprintf('  AL maximum absolute phase:      %.6f deg\n', ...
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

    fprintf('\nPhase saturation and range:\n');
    for j = 1:numel(CONV)
        fprintf('  %-3s: saturation=%8.3f%% range=%10.3f deg\n', ...
            CONV{j}, report.phase_sat_pct(j), report.phase_range_deg(j));
    end

    if report.fail_al_inactive
        fprintf('  FAILED: AL did not move enough to be considered active.\n');
    end
    if report.fail_phase_saturation
        fprintf('  FAILED: at least one phase exceeded allowed saturation coverage.\n');
    end
    if report.fail_temperature_target
        fprintf('  FAILED: junction temperature did not remain in target window.\n');
    end
    if report.fail_no_derating
        fprintf('  FAILED: no actual derating occurred.\n');
    end
    if report.warn_low_avg_power
        fprintf('  WARNING: common average source power approaches zero.\n');
    end
    if report.warn_extreme_gei
        fprintf('  WARNING: unusually large GEI values were observed.\n');
    end
    if report.warn_limited_transition_coverage
        fprintf('  WARNING: very little partial-derating transition data.\n');
    end
end


function [V, P, GEI, BT, TJ, HS, PH, DER, PAVG] = ...
    table_to_quality_arrays(out, BUS, CONV)

    n = height(out);
    V = nan(n, numel(BUS));
    P = nan(n, numel(BUS));
    GEI = nan(n, numel(BUS));
    BT = nan(n, numel(BUS));
    TJ = nan(n, numel(CONV));
    HS = nan(n, numel(CONV));
    PH = nan(n, numel(CONV));
    DER = nan(n, numel(CONV));
    PAVG = out.Src_Pow_Avg;

    for i = 1:numel(BUS)
        V(:,i) = out.(sprintf('V_Bus_%s', BUS{i}));
        P(:,i) = out.(sprintf('Bus_%s_Src_Pow', BUS{i}));
        GEI(:,i) = out.(sprintf('GEI_%s', BUS{i}));
        BT(:,i) = out.(sprintf('Bus_%s_Temp', BUS{i}));
    end
    for j = 1:numel(CONV)
        TJ(:,j) = out.(sprintf('JunctionTemp_C_%s', CONV{j}));
        HS(:,j) = out.(sprintf('HeatSinkTemp_C_%s', CONV{j}));
        PH(:,j) = out.(sprintf('Phase_%s_cmd_deg', CONV{j}));
        DER(:,j) = out.(sprintf('DAB_%s_Derate_Factor', CONV{j}));
    end
end


function write_quality_report(filename, report, episode_reports, ...
    run_mode, n_episodes, episode_tend, ts, source_model, dataset_model, ...
    CONV, calibration, total_wall_s)

    fid = fopen(filename, 'w');
    if fid == -1
        warning('Could not open quality report: %s', filename);
        return;
    end
    cleanup = onCleanup(@() fclose(fid));

    fprintf(fid, 'ALL-ACTIVE TARGET-125 THERMAL DATASET QUALITY REPORT\n');
    fprintf(fid, '====================================================\n');
    fprintf(fid, 'Run mode: %s\n', upper(char(run_mode)));
    fprintf(fid, 'Episodes: %d\n', n_episodes);
    fprintf(fid, 'Episode duration: %.0f s\n', episode_tend);
    fprintf(fid, 'Sample time: %.6f s\n', ts);
    fprintf(fid, 'Total wall time: %.3f s\n', total_wall_s);
    fprintf(fid, 'Original model: %s\n', source_model);
    fprintf(fid, 'Dataset model: %s\n', dataset_model);
    fprintf(fid, ['Controller changes: AL active through bus_l_cntrl; ' ...
        'PID9 P-only; nine PI controllers use back-calculation anti-windup; ' ...
        'PID output limits +/-60 kW.\n']);
    fprintf(fid, ['Thermal changes: corrected derating formula; degraded cooling ' ...
        'represented by increased R_ha in the dataset-only copy.\n']);
    fprintf(fid, 'Selected base R_ha: %.12g K/W\n', ...
        calibration.selected_rha_K_per_W);
    fprintf(fid, 'Calibration max Tj: %.12g C\n', ...
        calibration.selected_max_tj_C);
    fprintf(fid, 'Calibration max phase saturation: %.12g percent\n\n', ...
        calibration.selected_max_phase_sat_pct);

    fprintf(fid, 'FINAL STATUS: %s\n\n', ...
        ternary(report.pass, 'PASS', 'REVIEW'));

    fprintf(fid, 'Combined mandatory checks\n');
    fprintf(fid, '  nonfinite_count: %d\n', report.nonfinite_count);
    fprintf(fid, '  al_phase_range_deg: %.12g\n', report.al_phase_range_deg);
    fprintf(fid, '  max_derate_formula_error: %.12g\n', ...
        report.max_derate_formula_error);
    fprintf(fid, '  derated_pct: %.12g\n', report.derated_pct);
    fprintf(fid, '  max_junction_temp_C: %.12g\n', ...
        report.max_junction_temp_C);
    fprintf(fid, '  transition_pct: %.12g\n', report.transition_pct);
    fprintf(fid, '  cutoff_pct: %.12g\n', report.cutoff_pct);

    fprintf(fid, '\nCombined phase results\n');
    for j = 1:numel(CONV)
        fprintf(fid, '  %s: saturation=%.12g percent, range=%.12g deg\n', ...
            CONV{j}, report.phase_sat_pct(j), report.phase_range_deg(j));
    end

    fprintf(fid, '\nEpisode summaries\n');
    for ep = 1:numel(episode_reports)
        er = episode_reports(ep);
        fprintf(fid, ['  Episode %d: max_Tj=%.6f C, derated=%.6f%%, ' ...
            'transition=%.6f%%, max_phase_sat=%.6f%%\n'], ...
            ep, er.max_junction_temp_C, er.derated_pct, ...
            er.transition_pct, max(er.phase_sat_pct));
    end
end


function out = ternary(condition, value_if_true, value_if_false)
    if condition
        out = value_if_true;
    else
        out = value_if_false;
    end
end
