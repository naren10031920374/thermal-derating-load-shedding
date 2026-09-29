function compare_sheddable_vs_ramped_corridor_triad_collapse()
% ==========================================================================
%  COMPARE_SHEDDABLE_VS_RAMPED_CORRIDOR_TRIAD_COLLAPSE.M
%  --------------------------------------------------------------------
%  Same role as the baseline's compare_sheddable_vs_ramped.m, but using
%  corridor_triad_collapse's own load profile (build_corridor_triad_loads,
%  SEED=7) instead of the baseline's build_derating_loads. This is a
%  mandatory prerequisite before trusting
%  Grid_modelling_Thermal_V7_ALfix_sheddable_ramped.slx for anything on
%  this scenario -- the baseline's own check only proves the Rate Limiter
%  is a no-op under ITS load profile, not this one.
%
%  Runs Grid_modelling_Thermal_V7_ALfix_sheddable.slx and
%  Grid_modelling_Thermal_V7_ALfix_sheddable_ramped.slx with IDENTICAL
%  corridor_triad_collapse inputs and every actuator left at default
%  (ShedFrac = 1, RisingSlewLimit = inf, FallingSlewLimit = -inf). Since
%  the Rate Limiter is a no-op at infinite slew, the two models must
%  produce identical bus voltages, to floating point noise only. If they
%  do not, do NOT use the ramped model for the lead-time sweep until this
%  is resolved.
%
%  RUNTIME: 2 full 5000s sim() calls. At this scenario's observed
%  ~800-950s/run, expect roughly 30-35 minutes total.
%
%  Run:
%    compare_sheddable_vs_ramped_corridor_triad_collapse
% ==========================================================================

clc;

%% 0. CONFIG --------------------------------------------------------
% Portable root: this script always lives at <root>/scripts/06_gradual_ramp/,
% so walk up two levels from this file's own location. Identical result on
% Windows (dev machine) and Linux (Great Lakes / any other checkout) --
% no hardcoded drive letter or path separator.
PROJECT_ROOT = fileparts(fileparts(fileparts(mfilename('fullpath'))));
MODEL_A = 'Grid_modelling_Thermal_V7_ALfix_sheddable';         % baseline (no rate limiter)
MODEL_B = 'Grid_modelling_Thermal_V7_ALfix_sheddable_ramped';  % candidate (rate limiter, default = no-op)

TS   = 0.01;
TEND = 5000;
N_STEPS = round(TEND / TS) + 1;
SEED = 7;                          % must match generate_corridor_triad_collapse.m exactly
PMIN = 20000;
PMAX = 100000;

BUS  = {'A','B','C','D','E','F','G','H','K','L'};
CONV = {'AB','BC','CD','DE','EF','FG','GH','HK','KL','AL'};

% Tolerance for "identical". The baseline's own check found ~9.55e-15
% floating point noise between original and sheddable, so this uses the
% same order of magnitude with margin.
TOL = 1e-9;

OUTPUT_DIR = fullfile(PROJECT_ROOT, 'model_outputs', 'thermal_derating_v7', 'corridor_triad_collapse_ramp_verification');
if exist(OUTPUT_DIR, 'dir') ~= 7; mkdir(OUTPUT_DIR); end
addpath(PROJECT_ROOT);

fprintf('%s\n', repmat('=', 1, 70));
fprintf('COMPARE (corridor_triad_collapse loads): %s  vs  %s\n', MODEL_A, MODEL_B);
fprintf('%s\n', repmat('=', 1, 70));

%% 1. Run model A (sheddable, no rate limiter) -----------------------------
V_A = run_model_get_voltages(MODEL_A, PROJECT_ROOT, TEND, TS, N_STEPS, BUS, CONV, SEED, PMIN, PMAX);

%% 2. Run model B (sheddable + rate limiter, defaults = no-op) -------------
V_B = run_model_get_voltages(MODEL_B, PROJECT_ROOT, TEND, TS, N_STEPS, BUS, CONV, SEED, PMIN, PMAX);

%% 3. Compare ---------------------------------------------------------------
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
    fprintf('settings for corridor_triad_collapse. Safe to proceed to the\n');
    fprintf('ramp-duration lead-time sweep.\n');
else
    fprintf('AT LEAST ONE BUS FAILED. Do not trust the ramped model for this\n');
    fprintf('scenario until this is resolved -- check the Rate Limiter wiring\n');
    fprintf('for the failing bus(es) with add_gradual_ramp_capability.m before\n');
    fprintf('proceeding to any ramp-duration sweep.\n');
end

save(fullfile(OUTPUT_DIR, 'compare_sheddable_vs_ramped_corridor_triad_collapse_results.mat'), ...
     'V_A', 'V_B', 'BUS', 'max_diffs', 'TOL', 'all_pass', '-v7.3');
fprintf('\nWrote: %s\n', fullfile(OUTPUT_DIR, 'compare_sheddable_vs_ramped_corridor_triad_collapse_results.mat'));

% Nonzero exit on failure -- lets a shell/SLURM wrapper (e.g.
% run_gradual_ramp_serial.sh) refuse to proceed to the ramp sweep instead
% of silently continuing after a failed verification.
if ~all_pass
    error('compare_sheddable_vs_ramped_corridor_triad_collapse:FAIL', ...
          'Ramped model FAILED the no-op verification -- see comparison table above. Do not run the ramp-duration sweep until this is fixed.');
end

end


function V = run_model_get_voltages(model_name, project_root, tend, ts, n_steps, BUS, CONV, seed, pmin, pmax)
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

    fprintf('Wiring voltage taps ...\n');
    wire_all_taps(model_name, BUS, CONV);

    % Reset RNG and reload Pload_<bus> workspace vars fresh for this run,
    % exactly as every other corridor_triad_collapse script does, so both
    % models see the identical underlying load profile.
    build_corridor_triad_loads(seed, n_steps, ts, BUS, pmin, pmax);

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


%% ---- verbatim from generate_corridor_triad_collapse.m ----

function [load_hist, meta] = build_corridor_triad_loads(seed, n_steps, ts, BUS, pmin, pmax)
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
