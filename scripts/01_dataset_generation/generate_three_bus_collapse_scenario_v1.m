function generate_three_bus_collapse_scenario_v1()
% ==========================================================================
%  GENERATE_THREE_BUS_COLLAPSE_SCENARIO_V1.M
%  --------------------------------------------------------------------
%  "three_bus_collapse" -- the simple working name for what the project
%  docs call corridor_triad_collapse. Third distinct collapse scenario in
%  this project, built from scratch .
%
%  GOAL: get at least 2, ideally all 3, buses in a contiguous three-bus
%  corridor to collapse within a shared time window -- a genuine regional
%  collapse, not three unrelated single-bus events that happen to occur in
%  the same run. Baseline gave us 1 bus collapsing (with a downstream
%  neighbor following). gradual_busC_v4 gave us 1 bus collapsing via a
%  slow ramp. This scenario is meant to be a third, structurally distinct
%  mechanism: correlated multi-bus stress with thermal preconditioning.
%
%  WHY THE TARGET TRIO IS G, H, K (not E, F, G):
%  An earlier design proposal for this scenario (never coded, found in
%  this project's own docs) targeted E, F, G. But E and F already collapse
%  together in the baseline scenario via a totally different mechanism
%  (sudden spike cascade). Reusing E/F/G here would make this look like a
%  variant of baseline rather than a genuinely separate scenario, so this
%  version targets G, H, K instead -- a trio no other scenario touches.
%  Ring order: A-B-C-D-E-F-G-H-K-L-(A). Boundary buses (the only "rescue"
%  paths into the corridor) are F and L; boundary converters are FG and
%  KL; internal converters linking the trio are GH and HK. Everything
%  else (A, B, C, D, E) is left as ordinary mild background load.
%
%  WHY THE LOAD RANGE IS ALREADY LOOSENED (not starting conservative):
%  gradual_busC_v4 spent 3 iterations (v1-v3) plateauing at ~164C before
%  discovering that PMIN=20000W/PMAX=100000W -- used as a hard clip in
%  literally every scenario script in this project, including the never-
%  run corridor_triad_collapse design proposal -- was never a real
%  Simulink model limit, just an inherited script convention. This script
%  starts with PMIN=2000/PMAX=220000 from the outset and targets
%  magnitudes in the range that actually produced a real collapse for
%  Bus C (150,000-200,000W territory), rather than repeating the same
%  wasted iterations.
%
%  MECHANISM (kept from the original design proposal, which had two
%  genuinely good ideas worth preserving even though it was never run):
%    1. CORRELATED stress: G, H, K ramp together on nearly the same
%       shared curve (small +/-3% per-bus jitter on the target magnitude,
%       not independent noise), so they're pushed toward their limits in
%       the same window, not scattered by chance.
%    2. THERMAL PRECONDITIONING: a preheat phase before the real event
%       uses up some of the converters' derating headroom in advance, so
%       the surge lands on already-warm converters, not cold ones -- every
%       other scenario in this project starts cold.
%    3. A CO-COLLAPSE WINDOW check (new validation metric, not used
%       anywhere else in this project): after the run, checks whether at
%       least 2 of the 3 target buses' collapse onset times fall within a
%       shared window of each other (300s), rather than just checking "did
%       any bus collapse" the way every other scenario's health check does.
%
%  FOUR PHASES (TEND = 5000s, unchanged):
%    Phase 1 - Preheat        (0    - 800s):  F,G,H,K,L mildly elevated.
%    Phase 2 - Synchronized surge (800 - 1400s, 600s ramp): G,H,K ramp UP
%              together toward a high plateau; F,L ramp DOWN toward a
%              starved low value (the only rescue paths choked off) --
%              this dual boost+starve mechanic is exactly what worked for
%              Bus C in gradual_busC_v4.
%    Phase 3 - Sustained plateau (1400 - 3400s, ~2000s hold): held at the
%              stressed configuration -- this is where the collapse(s), if
%              any, should happen.
%    Phase 4 - Step-down / recovery check (3400 - 3700s ramp, held to
%              5000s): eases back toward a moderate level, to see whether
%              affected buses recover or stay latched down from residual
%              thermal derating -- a question none of the other scenarios
%              ask.
%
%  RISK, STATED UP FRONT: this pushes 5 of 10 buses away from ambient at
%  once (3 boosted, 2 starved), a bigger simultaneous demand differential
%  than gradual_busC_v4's single-bus case. Solver stiffness warnings or a
%  genuine NaN/Inf failure are both plausible and both informative if they
%  happen (same "not necessarily fatal, the validation gate is the real
%  check" stance used in gradual_busC_v4).
%
%  RUNTIME: one 5000s sim() call. Expect a similar order of magnitude to
%  gradual_busC_v4's ~15-20 minutes, possibly longer given more buses are
%  away from ambient simultaneously.
%
%  Run:
%    generate_three_bus_collapse_scenario_v1
% ==========================================================================

    clc;

    %% 0. CONFIG ----------------------------------------------------------
    PROJECT_ROOT = 'D:\naren\Documents\Thermal_Derating_Work_Handover\ThermalProject';
    MODEL_NAME = 'Grid_modelling_Thermal_V7_ALfix';   % pure dataset generation, no shedding needed yet

    TS   = 0.01;
    TEND = 5000;
    N_STEPS = round(TEND / TS) + 1;
    PMIN = 2000;      % loosened from the start -- see header
    PMAX = 220000;
    SEED = 201;

    BUS  = {'A','B','C','D','E','F','G','H','K','L'};
    CONV = {'AB','BC','CD','DE','EF','FG','GH','HK','KL','AL'};

    TARGET_BUSES   = {'G','H','K'};
    BOUNDARY_BUSES = {'F','L'};

    PREHEAT_END_S  = 800;
    SURGE_END_S    = 1400;
    PLATEAU_END_S  = 3400;
    STEPDOWN_END_S = 3700;

    AMBIENT_W            = 27000;    % background buses (A,B,C,D,E) stay near here
    TARGET_PREHEAT_W     = 33000;    % G,H,K during Phase 1
    TARGET_PLATEAU_W     = 190000;   % G,H,K during Phase 3 (real-collapse territory per v4)
    TARGET_RECOVERY_W    = 45000;    % G,H,K during Phase 4 (above preheat, well below plateau)
    BOUNDARY_PREHEAT_W   = 31000;    % F,L during Phase 1
    BOUNDARY_LOW_W       = 3000;     % F,L during Phase 3 (starved rescue paths, per v4's B/D)
    BOUNDARY_RECOVERY_W  = 27000;    % F,L during Phase 4 (back to ambient)
    TARGET_JITTER_FRAC   = 0.03;     % +/-3% per-bus jitter on G/H/K's shared curve

    CO_COLLAPSE_WINDOW_S = 300;      % new validation metric, see header

    OUTPUT_DIR = fullfile(PROJECT_ROOT, 'model_outputs', 'thermal_derating_three_bus_collapse');
    if exist(OUTPUT_DIR, 'dir') ~= 7; mkdir(OUTPUT_DIR); end
    addpath(PROJECT_ROOT);

    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('THREE_BUS_COLLAPSE v1 (aka corridor_triad_collapse, rebuilt from scratch)\n');
    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('Target trio (boosted together): %s\n', strjoin(TARGET_BUSES, ', '));
    fprintf('Boundary buses (starved):       %s\n', strjoin(BOUNDARY_BUSES, ', '));
    fprintf('Preheat:    0 - %ds\n', PREHEAT_END_S);
    fprintf('Surge ramp: %d - %ds\n', PREHEAT_END_S, SURGE_END_S);
    fprintf('Plateau:    %d - %ds\n', SURGE_END_S, PLATEAU_END_S);
    fprintf('Step-down:  %d - %ds, held to %ds\n', PLATEAU_END_S, STEPDOWN_END_S, TEND);
    fprintf('Clip range: [%.0f, %.0f]\n', PMIN, PMAX);

    %% 1. Locate and load the model ----------------------------------------
    hh = dir(fullfile(PROJECT_ROOT, '**', [MODEL_NAME '.slx']));
    if isempty(hh)
        error('Base model not found: %s.slx', MODEL_NAME);
    end
    model_file = fullfile(hh(1).folder, hh(1).name);
    fprintf('\nModel: %s\n', model_file);
    if bdIsLoaded(MODEL_NAME)
        close_system(MODEL_NAME, 0);
    end
    load_system(model_file);
    set_param(MODEL_NAME, 'StopTime', num2str(TEND));

    %% 2. Build the load profile ---------------------------------------------
    fprintf('\nBuilding three_bus_collapse v1 load profile (seed %d) ...\n', SEED);
    [load_hist, clip_hi, clip_lo] = build_three_bus_collapse_loads( ...
        SEED, N_STEPS, TS, BUS, PMIN, PMAX, ...
        PREHEAT_END_S, SURGE_END_S, PLATEAU_END_S, STEPDOWN_END_S, ...
        AMBIENT_W, TARGET_PREHEAT_W, TARGET_PLATEAU_W, TARGET_RECOVERY_W, ...
        BOUNDARY_PREHEAT_W, BOUNDARY_LOW_W, BOUNDARY_RECOVERY_W, TARGET_JITTER_FRAC);
    fprintf('  Clip-high %% by bus (expect ~0 except transient overshoot): %s\n', mat2str(round(clip_hi,1)));
    fprintf('  Clip-low  %% by bus: %s\n', mat2str(round(clip_lo,1)));

    %% 3. Wire taps (proven pilot infrastructure, reused as-is) ------------
    fprintf('\nWiring signal taps ...\n');
    wire_all_taps(MODEL_NAME, BUS, CONV);
    derate_var_names   = tap_derate_factors(MODEL_NAME, CONV);
    junction_var_names = tap_junction_temp(MODEL_NAME, CONV);
    phase_var_names    = tap_phase_commands(MODEL_NAME, CONV);

    %% 4. Run the simulation -------------------------------------------------
    fprintf('\nRunning simulation ...\n');
    tic;
    ws = warning('off', 'all');
    [msg_before] = lastwarn('');
    so = sim(MODEL_NAME, 'ReturnWorkspaceOutputs', 'on');
    [warn_msg, warn_id] = lastwarn();
    warning(ws);
    fprintf('Finished in %.1f s wall clock.\n', toc);
    if ~isempty(warn_msg) && ~strcmp(warn_msg, msg_before)
        fprintf('  NOTE: a solver-related warning fired during this run (id: %s):\n  %s\n', warn_id, warn_msg);
        fprintf('  Not necessarily fatal -- the NaN/Inf validation below is the real check.\n');
    end

    time_vec = (0:N_STEPS-1)' * TS;

    %% 5. Assemble the CSV ---------------------------------------------------
    T = table(time_vec, 'VariableNames', {'time'});
    src_pow_matrix = nan(N_STEPS, numel(BUS));
    for b = 1:numel(BUS)
        T.(sprintf('V_Bus_%s', BUS{b})) = grab_series(so, sprintf('V_Bus_%s', BUS{b}), TS, N_STEPS);
        src_pow_matrix(:, b) = grab_series(so, sprintf('Bus_%s_Src_Pow', BUS{b}), TS, N_STEPS);
        T.(sprintf('Bus_%s_Src_Pow', BUS{b})) = src_pow_matrix(:, b);
        T.(sprintf('Bus_%s_Temp', BUS{b})) = grab_series(so, sprintf('Bus_%s_Temp', BUS{b}), TS, N_STEPS);
    end

    for b = 1:numel(BUS)
        T.(sprintf('CommandedLoad_kW_%s', BUS{b})) = load_hist(:, b) / 1000;
    end

    EPSILON_W = 500.0;
    avg_power = mean(src_pow_matrix, 2, 'omitnan');
    grid_collapsed = avg_power < EPSILON_W;
    for b = 1:numel(BUS)
        ratio = src_pow_matrix(:, b) ./ avg_power;
        ratio(grid_collapsed) = 0.0;
        T.(sprintf('GEI_%s', BUS{b})) = ratio;
    end
    T.Src_Pow_Avg_recomputed = avg_power;

    for c = 1:numel(CONV)
        T.(sprintf('Phase_%s_cmd_deg', CONV{c})) = grab_series(so, phase_var_names{c}, TS, N_STEPS);
        T.(sprintf('DAB_%s_Derate_Factor', CONV{c})) = grab_series(so, derate_var_names{c}, TS, N_STEPS) / 100e3;
        T.(sprintf('JunctionTemp_C_%s', CONV{c})) = grab_series(so, junction_var_names{c}, TS, N_STEPS);
    end
    hs_conv_order = {'AB','BC','CD','DE','EF','FG','GH','HK','KL','AL'};
    hs_tags = {'DAB_AB_H_Temp','DAB_BC_H_Temp','DAB_CD_H_Temp','DAB_DE_H_Temp', ...
               'DAB_EF_H_Temp','DAB_FG_H_Temp','DAB_GH_Temp','DAB_HK_Temp', ...
               'DAB_KL_Temp','DAB_LA_Temp'};
    for h = 1:numel(hs_tags)
        T.(sprintf('HeatSinkTemp_C_%s', hs_conv_order{h})) = grab_series(so, hs_tags{h}, TS, N_STEPS);
    end

    %% 6. NaN/Inf validation --------------------------------------------------
    WARMUP_TOLERANCE_S = 1.0;
    log_path = fullfile(OUTPUT_DIR, 'three_bus_collapse_v1_validation_log.txt');
    log_fid = fopen(log_path, 'w');
    log_both(log_fid, '%s\n', repmat('=', 1, 70));
    log_both(log_fid, 'NaN/Inf VALIDATION: three_bus_collapse v1 scenario\n');
    log_both(log_fid, '%s\n', repmat('=', 1, 70));

    data_cols = T.Properties.VariableNames(2:end);
    bad_mask = false(height(T), 1);
    any_unexpected = false;
    for ci = 1:numel(data_cols)
        col = data_cols{ci};
        v = T.(col);
        bad_here = isnan(v) | isinf(v);
        n_bad = sum(bad_here);
        if n_bad > 0
            bad_mask = bad_mask | bad_here;
            bad_times = time_vec(bad_here);
            n_after_warmup = sum(bad_times > WARMUP_TOLERANCE_S);
            log_both(log_fid, '  %-30s %6d bad samples total, %6d after warmup (t>%.1fs)\n', ...
                      col, n_bad, n_after_warmup, WARMUP_TOLERANCE_S);
            if n_after_warmup > 0
                any_unexpected = true;
                first_bad_t = bad_times(bad_times > WARMUP_TOLERANCE_S);
                log_both(log_fid, '    FIRST unexpected bad sample at t=%.2fs\n', first_bad_t(1));
            end
        end
    end
    if ~any(bad_mask)
        log_both(log_fid, '  No NaN/Inf found in any column. Clean.\n');
    end
    if any_unexpected
        log_both(log_fid, '\n  RESULT: FAILED VALIDATION. Unexpected NaN/Inf beyond warmup.\n');
        log_both(log_fid, '  This itself is informative: pushing 5 buses away from ambient at\n');
        log_both(log_fid, '  once (3 boosted, 2 starved) may have hit a genuine numerical limit,\n');
        log_both(log_fid, '  not a thermal one -- consider a milder plateau target or a slower\n');
        log_both(log_fid, '  surge ramp before concluding the mechanism itself does not work.\n');
        fclose(log_fid);
        error('three_bus_collapse v1 scenario FAILED NaN/Inf validation. See log:\n%s', log_path);
    else
        log_both(log_fid, '\n  RESULT: PASSED VALIDATION. Safe to proceed.\n');
    end
    fclose(log_fid);

    out_csv = fullfile(OUTPUT_DIR, 'thermal_derating_three_bus_collapse_v1_5000s.csv');
    writetable(T, out_csv);
    fprintf('\nWrote: %s (%d rows x %d cols)\n', out_csv, height(T), width(T));

    %% 7. Health check + collapse summary -----------------------------------
    fprintf('\n%s\n', repmat('=', 1, 70));
    fprintf('CONVERTER HEALTH CHECK\n');
    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('%-6s %10s %10s %8s %8s %10s %8s\n', 'Conv', 'HS max', 'Tj max', '>125C?', '>175C?', 'Derate min', 'Derated?');
    for c = 1:numel(CONV)
        hs = T.(sprintf('HeatSinkTemp_C_%s', CONV{c}));
        tj = T.(sprintf('JunctionTemp_C_%s', CONV{c}));
        der = T.(sprintf('DAB_%s_Derate_Factor', CONV{c}));
        hs_max = max(hs, [], 'omitnan');
        tj_max = max(tj, [], 'omitnan');
        der_min = min(der, [], 'omitnan');
        over125 = tj_max > 125; over175 = tj_max > 175;
        fprintf('%-6s %9.2fC %9.2fC %8s %8s %10.3f %8s\n', CONV{c}, hs_max, tj_max, ...
            ternary(over125,'YES','no'), ternary(over175,'YES','no'), der_min, ternary(der_min<0.999,'YES','no'));
    end

    fprintf('\nPer-bus voltage collapse check (threshold 100V, 50-sample debounce):\n');
    any_collapse = false;
    for b = 1:numel(BUS)
        v = T.(sprintf('V_Bus_%s', BUS{b}));
        [c, onset, min_v] = detect_collapse_summary(time_vec, v, 100.0, 50);
        if c
            fprintf('  Bus %s: COLLAPSES at t=%.2fs (min V=%.2f)\n', BUS{b}, onset, min_v);
            any_collapse = true;
        else
            fprintf('  Bus %s: survives (min V=%.2f)\n', BUS{b}, min_v);
        end
    end

    %% 7b. NEW: co-collapse window check for the target trio -----------------
    fprintf('\n%s\n', repmat('=', 1, 70));
    fprintf('CO-COLLAPSE WINDOW CHECK (target trio %s, window=%.0fs)\n', strjoin(TARGET_BUSES, ','), CO_COLLAPSE_WINDOW_S);
    fprintf('%s\n', repmat('=', 1, 70));
    target_onsets = [];
    for i = 1:numel(TARGET_BUSES)
        tb = TARGET_BUSES{i};
        v = T.(sprintf('V_Bus_%s', tb));
        [c, onset, ~] = detect_collapse_summary(time_vec, v, 100.0, 50);
        if c
            fprintf('  Bus %s: collapsed at t=%.2fs\n', tb, onset);
            target_onsets(end+1) = onset; %#ok<AGROW>
        else
            fprintf('  Bus %s: did not collapse\n', tb);
        end
    end
    if numel(target_onsets) >= 2
        window_span = max(target_onsets) - min(target_onsets);
        if window_span <= CO_COLLAPSE_WINDOW_S
            fprintf('\n  REGIONAL COLLAPSE ACHIEVED: %d of 3 target buses collapsed within\n', numel(target_onsets));
            fprintf('  %.2fs of each other (allowed window: %.0fs). This is a genuine\n', window_span, CO_COLLAPSE_WINDOW_S);
            fprintf('  regional/correlated collapse, not isolated single-bus events.\n');
        else
            fprintf('\n  Collapsed but NOT simultaneous: %d of 3 target buses collapsed, but\n', numel(target_onsets));
            fprintf('  %.2fs apart (exceeds the %.0fs window). Logged as NOT a regional\n', window_span, CO_COLLAPSE_WINDOW_S);
            fprintf('  collapse -- the correlated-stress mechanism produced separate\n');
            fprintf('  failures rather than a shared one.\n');
        end
    elseif numel(target_onsets) == 1
        fprintf('\n  Only 1 of 3 target buses collapsed -- an isolated event, not a\n');
        fprintf('  regional collapse. The correlated-stress mechanism alone was not\n');
        fprintf('  enough to bring down more than one of the three.\n');
    else
        fprintf('\n  No target bus collapsed this run. If Tj on GH/HK moved meaningfully\n');
        fprintf('  toward 175C (even without crossing it), the mechanism is working but\n');
        fprintf('  underpowered -- push TARGET_PLATEAU_W higher or extend PLATEAU_END_S.\n');
        fprintf('  If Tj barely moved at all, reconsider the boundary starve magnitude\n');
        fprintf('  (BOUNDARY_LOW_W) or the preheat duration before iterating on magnitude.\n');
    end

    if any_collapse
        fprintf('\n  (At least one bus collapsed somewhere in this run -- see per-bus check\n');
        fprintf('  above for which. The co-collapse result above is the one that actually\n');
        fprintf('  matters for judging whether this scenario succeeded as designed.)\n');
    end

    close_system(MODEL_NAME, 0);
end


%% ---- three_bus_collapse v1's own load-profile function ----

function [load_hist, clip_hi, clip_lo] = build_three_bus_collapse_loads( ...
    seed, n_steps, ts, BUS, pmin, pmax, ...
    preheat_end_s, surge_end_s, plateau_end_s, stepdown_end_s, ...
    ambient_w, target_preheat_w, target_plateau_w, target_recovery_w, ...
    boundary_preheat_w, boundary_low_w, boundary_recovery_w, jitter_frac)

    rng(seed);
    t = (0:n_steps-1)' * ts;
    NB = numel(BUS);

    TARGET_BUSES   = {'G','H','K'};
    BOUNDARY_BUSES = {'F','L'};

    idx_target   = find(ismember(BUS, TARGET_BUSES));
    idx_boundary = find(ismember(BUS, BOUNDARY_BUSES));

    % Fixed per-bus jitter for the target trio, drawn once (not
    % time-varying) -- this is what makes G/H/K "correlated but not
    % identical," matching the design's shared-ramp-with-small-jitter
    % intent, rather than three independent noise streams.
    jitter_by_bus = ones(1, NB);
    jitter_by_bus(idx_target) = 1 + jitter_frac * (2*rand(1, numel(idx_target)) - 1);

    % Shared timing fractions -- identical shape for every target/boundary
    % bus, only the jitter (target only) and the preheat/mid/recovery
    % anchor values (target vs. boundary) differ.
    frac_surge    = min(1, max(0, (t - preheat_end_s) / (surge_end_s - preheat_end_s)));
    frac_stepdown = min(1, max(0, (t - plateau_end_s) / (stepdown_end_s - plateau_end_s)));

    load_hist = nan(n_steps, NB);
    clip_hi   = nan(1, NB);
    clip_lo   = nan(1, NB);

    for k = 1:NB
        phase_shift = 0.4 * k;
        slow_var = 4000 * sin(2*pi*t/1500 + phase_shift);
        noise    = 800  * randn(n_steps, 1);

        is_target   = ismember(BUS{k}, TARGET_BUSES);
        is_boundary = ismember(BUS{k}, BOUNDARY_BUSES);

        if is_target
            ji = jitter_by_bus(k);
            preheat_local  = target_preheat_w  * ji;
            mid_local      = target_plateau_w  * ji;   % Phase 3 plateau
            recovery_local = target_recovery_w * ji;
            raw = preheat_local ...
                + (mid_local - preheat_local) .* frac_surge ...
                + (recovery_local - mid_local) .* frac_stepdown ...
                + slow_var + noise;
        elseif is_boundary
            preheat_local  = boundary_preheat_w;
            mid_local      = boundary_low_w;           % Phase 3: starved
            recovery_local = boundary_recovery_w;
            raw = preheat_local ...
                + (mid_local - preheat_local) .* frac_surge ...
                + (recovery_local - mid_local) .* frac_stepdown ...
                + slow_var + noise;
        else
            % background bus: ordinary mild fluctuation around ambient,
            % no phase structure at all
            raw = ambient_w + slow_var + noise;
        end

        clip_hi(k) = mean(raw > pmax) * 100;
        clip_lo(k) = mean(raw < pmin) * 100;

        p = max(pmin, min(pmax, raw));
        load_hist(:, k) = p;
        assignin('base', sprintf('Pload_%s', BUS{k}), p(:).');
    end
end


%% ---- proven pilot tap-wiring infrastructure (reused verbatim from gradual_busC_v4) ----

function wire_all_taps(model, BUS, CONV) %#ok<INUSD>
    want_V    = strcat('V_Bus_',   BUS);
    want_P    = strcat('Bus_',     BUS, '_Src_Pow');
    want_T    = strcat('Bus_',     BUS, '_Temp');
    want_HS   = {'DAB_AB_H_Temp','DAB_BC_H_Temp','DAB_CD_H_Temp', ...
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


function var_names = tap_junction_temp(model, CONV)
    var_names = cell(1, numel(CONV));
    for i = 1:numel(CONV)
        var_names{i} = sprintf('V7TJ_%s', CONV{i});
        dab_path = [model '/DAB_Model_' CONV{i}];
        outport_blk = find_system(dab_path, 'SearchDepth', 1, 'LookUnderMasks', 'all', ...
                                   'BlockType', 'Outport', 'Name', 'Temp_Junction');
        if numel(outport_blk) ~= 1
            warning('tap_junction_temp:noOutport', ...
                'Expected one Outport named Temp_Junction under DAB_Model_%s, found %d.', ...
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


function var_names = tap_phase_commands(model, CONV)
    var_names = cell(1, numel(CONV));
    for i = 1:numel(CONV)
        var_names{i} = sprintf('V7PHASE_%s', CONV{i});
        dab_path = [model '/DAB_Model_' CONV{i}];
        inner_inport_path = [dab_path '/phase_shift'];
        if getSimulinkBlockHandle(inner_inport_path) == -1
            warning('tap_phase_commands:noInport', ...
                'No Inport block named phase_shift found under DAB_Model_%s.', CONV{i});
            continue;
        end
        port_num = str2double(get_param(inner_inport_path, 'Port'));
        outer_ph = get_param(dab_path, 'PortHandles');
        if numel(outer_ph.Inport) < port_num
            warning('tap_phase_commands:badPort', ...
                'DAB_Model_%s does not have inport #%d at the top level.', CONV{i}, port_num);
            continue;
        end
        line = get_param(outer_ph.Inport(port_num), 'Line');
        if line == -1
            warning('tap_phase_commands:noLine', ...
                'DAB_Model_%s phase_shift inport has no incoming line at the top level.', CONV{i});
            continue;
        end
        src_port = get_param(line, 'SrcPortHandle');
        log_name = ['V7PHASELOG_' var_names{i}];
        cleanup_and_add_tap(model, log_name, var_names{i}, src_port, model);
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


function log_both(fid, fmt, varargin)
    fprintf(fmt, varargin{:});
    fprintf(fid, fmt, varargin{:});
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


function out = ternary(cond, a, b)
    if cond; out = a; else; out = b; end
end