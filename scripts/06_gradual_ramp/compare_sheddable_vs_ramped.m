%% COMPARE_SHEDDABLE_VS_RAMPED.M
% =========================================================================
% Sanity check for add_gradual_ramp_capability.m, same role as the
% original-vs-sheddable check that verified add_controllable_load_shedding.m.
%
% Runs Grid_modelling_Thermal_V7_ALfix_sheddable.slx and
% Grid_modelling_Thermal_V7_ALfix_sheddable_ramped.slx with IDENTICAL
% inputs and every actuator left at default (ShedFrac = 1, RisingSlewLimit
% = inf, FallingSlewLimit = -inf). Since the Rate Limiter is a no-op at
% infinite slew, the two models must produce identical bus voltages,
% to floating point noise only. If they do not, do not use the ramped
% model for anything until this is resolved.
%
% Do NOT proceed to generate_gradual_ramp_dataset.m until this passes.
%
% RUNTIME: 2 full 5000s sim() calls, roughly 8-9 minutes total.
%
% Run:
%   compare_sheddable_vs_ramped
% =========================================================================

function compare_sheddable_vs_ramped()

clc;

%% 0. CONFIG --------------------------------------------------------
PROJECT_ROOT = 'D:\naren\Documents\Thermal_Derating_Work_Handover\ThermalProject';
MODEL_A = 'Grid_modelling_Thermal_V7_ALfix_sheddable';         % baseline
MODEL_B = 'Grid_modelling_Thermal_V7_ALfix_sheddable_ramped';  % candidate

TS   = 0.01;
TEND = 5000;
N_STEPS = round(TEND / TS) + 1;
LOAD_SEED = 1;
GRID_LOAD_SCALE = 1.0;
PMIN = 20000;
PMAX = 100000;

BUS  = {'A','B','C','D','E','F','G','H','K','L'};
CONV = {'AB','BC','CD','DE','EF','FG','GH','HK','KL','AL'};

% Tolerance for "identical". add_controllable_load_shedding.m's own check
% found ~9.55e-15 floating point noise between original and sheddable, so
% this uses the same order of magnitude with margin.
TOL = 1e-9;

OUTPUT_DIR = fullfile(PROJECT_ROOT, 'model_outputs', 'thermal_derating_v7', 'ramp_verification');
if exist(OUTPUT_DIR, 'dir') ~= 7; mkdir(OUTPUT_DIR); end
addpath(PROJECT_ROOT);

fprintf('%s\n', repmat('=', 1, 70));
fprintf('COMPARE: %s  vs  %s\n', MODEL_A, MODEL_B);
fprintf('%s\n', repmat('=', 1, 70));

%% 1. Build the load profile, identical for both runs ---------------------
fprintf('\nBuilding load profile (seed %d, scale %.2f) ...\n', LOAD_SEED, GRID_LOAD_SCALE);
[~, ~, ~] = build_derating_loads(LOAD_SEED, N_STEPS, TS, BUS, GRID_LOAD_SCALE, PMIN, PMAX);

%% 2. Run model A (sheddable, no rate limiter) -----------------------------
V_A = run_model_get_voltages(MODEL_A, PROJECT_ROOT, TEND, TS, N_STEPS, BUS, CONV);

%% 3. Run model B (sheddable + rate limiter, defaults = no-op) -------------
V_B = run_model_get_voltages(MODEL_B, PROJECT_ROOT, TEND, TS, N_STEPS, BUS, CONV);

%% 4. Compare ---------------------------------------------------------------
fprintf('\n%s\n', repmat('=', 1, 70));
fprintf('COMPARISON (tolerance = %.1e)\n', TOL);
fprintf('%s\n', repmat('=', 1, 70));
fprintf('%-8s %-16s %-10s\n', 'Bus', 'max |diff| (V)', 'Result');

all_pass = true;
max_diffs = nan(1, numel(BUS));
for b = 1:numel(BUS)
    a = V_A(:, b);
    c = V_B(:, b);
    valid = ~isnan(a) & ~isnan(c);
    if ~any(valid)
        fprintf('%-8s %-16s %-10s\n', BUS{b}, 'n/a (no data)', 'SKIPPED');
        continue;
    end
    d = max(abs(a(valid) - c(valid)));
    max_diffs(b) = d;
    result = 'PASS';
    if d > TOL
        result = 'FAIL';
        all_pass = false;
    end
    fprintf('%-8s %-16.3e %-10s\n', BUS{b}, d, result);
end

fprintf('\n');
if all_pass
    fprintf('ALL BUSES PASS. The ramped model is a verified no-op at default\n');
    fprintf('settings. Safe to proceed to generate_gradual_ramp_dataset.m.\n');
else
    fprintf('AT LEAST ONE BUS FAILED. Do not trust the ramped model until this\n');
    fprintf('is resolved, check the Rate Limiter wiring for the failing bus(es)\n');
    fprintf('with add_gradual_ramp_capability.m before proceeding.\n');
end

save(fullfile(OUTPUT_DIR, 'compare_sheddable_vs_ramped_results.mat'), ...
     'V_A', 'V_B', 'BUS', 'max_diffs', 'TOL', 'all_pass', '-v7.3');
fprintf('\nWrote: %s\n', fullfile(OUTPUT_DIR, 'compare_sheddable_vs_ramped_results.mat'));

end


function V = run_model_get_voltages(model_name, project_root, tend, ts, n_steps, BUS, CONV)
    hh = dir(fullfile(project_root, '**', [model_name '.slx']));
    if isempty(hh)
        error('Model not found: %s.slx', model_name);
    end
    model_file = fullfile(hh(1).folder, hh(1).name);
    fprintf('\nModel: %s\n', model_file);
    if bdIsLoaded(model_name)
        close_system(model_name, 0);
    end
    load_system(model_file);
    set_param(model_name, 'StopTime', num2str(tend));

    % CRITICAL: V_Bus_<bus> is not captured automatically. It only lands
    % in the sim's workspace output if wired to a To Workspace block via
    % wire_all_taps first, same as every other script in this project.
    % Skipping this call is what silently produced "n/a (no data)" for
    % every bus on the first run of this script, a false pass, not a
    % real comparison.
    fprintf('Wiring voltage taps ...\n');
    wire_all_taps(model_name, BUS, CONV);

    tic;
    ws = warning('off', 'all');
    so = sim(model_name, 'ReturnWorkspaceOutputs', 'on');
    warning(ws);
    fprintf('Finished in %.1f s wall clock.\n', toc);

    V = nan(n_steps, numel(BUS));
    for b = 1:numel(BUS)
        V(:, b) = grab_series(so, sprintf('V_Bus_%s', BUS{b}), ts, n_steps);
    end

    n_missing = sum(all(isnan(V), 1));
    if n_missing > 0
        error(['%d of %d buses returned no data after wiring taps. ' ...
               'Do not trust a pass/fail result from this run, something ' ...
               'is wrong with the model or the tap wiring, not just this ' ...
               'script. Investigate before rerunning.'], n_missing, numel(BUS));
    end

    close_system(model_name, 0);
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
