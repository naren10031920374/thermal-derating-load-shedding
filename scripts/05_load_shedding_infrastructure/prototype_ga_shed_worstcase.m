%% PROTOTYPE_GA_SHED_WORSTCASE.M
% =========================================================================
% Second-pass GA-PSO shed-amount search (project doc, Section 12), built
% after the first real closed-loop validation showed a gap: Bus G's
% empirical-snapshot-derived shed % (48.5%) genuinely worked, but Bus K's
% (5.0%) did NOT - K still collapsed (delayed ~23s, but not prevented),
% even though the proxy predicted DVSI_after=0.515, comfortably resolved.
%
% WHY THE FIRST VERSION (prototype_ga_shed.m) UNDERSTATED K's NEED:
% It sized every bus's shed % against ONE empirical snapshot: the single
% row, within a narrow pre-collapse extraction window, where that bus's
% own (Load - Gen) deficit happened to be largest IN THE ORIGINAL, UNSHED
% simulation. Two things make that snapshot an unreliable sizing basis:
%   1. Bus_X_Src_Pow (Gen) swings between 0W and a ~95kW ceiling within
%      the scenario window (Section 10) - the snapshot search picks
%      whichever row had the worst deficit it happened to SEE, but Gen
%      could plausibly be even lower elsewhere, off-camera.
%   2. The real closed-loop run applies shedding continuously starting at
%      t=832.95s (long before K's own original unshed collapse at
%      t=1108.70s), which changes the system's whole forward trajectory
%      from that point on. The scenario's load also keeps climbing
%      through a "surge" phase out to t=1400s (per the bisection script's
%      own CONFIG: PREHEAT_END_S=800, SURGE_END_S=1400,
%      TARGET_PLATEAU_W=190000). A snapshot taken from the UNSHED
%      trajectory, in a window that ends before that climb is over, can
%      simply never have seen the worst deficit K actually has to survive
%      once shedding starts early.
%
% THIS VERSION: instead of trusting one empirical Load/Gen reading, size
% every gated bus against the scenario's own KNOWN WORST-CASE PHYSICAL
% BOUNDS, deliberately conservative in both directions:
%   - Load = TARGET_PLATEAU_W (the scenario's own designed worst-case
%     sustained load for a target bus - G, H, K are all "target" buses in
%     this scenario's load-profile config), not whatever Load happened to
%     be logged at one sampled row.
%   - Gen  = 0 (pessimistic - Gen was empirically observed swinging all
%     the way to 0W, so assume it could be there exactly when Load peaks).
%   - V, Vn unchanged from the same worst-case-row search as before - kept
%     because voltage doesn't swing anywhere near as wildly as Gen does,
%     and Section 12's real run already showed G's voltage-based term was
%     fine (G's number DID validate for real).
%
% This is DELIBERATELY pessimistic (worst load, together with worst
% generation, at the same instant) - expected to recommend MORE shedding
% than strictly necessary, which is the safe direction to be wrong in
% after the first version was shown to under-shed. Still not a substitute
% for the real closed-loop run - see the NEXT STEP note at the bottom.
%
% USAGE: same as prototype_ga_shed.m - edit SLICE_CSV if needed, then run:
%   prototype_ga_shed_worstcase
% =========================================================================

clear; clc;

SLICE_CSV = fullfile('..', '..', 'Claude outputs', 'dvsi_validation', ...
                      'three_bus_collapse_v1_dvsi_slice.csv');
CANDIDATE_BUSES = {'G', 'H', 'K'};

% From the bisection verification script's own CONFIG section
% (run_closed_loop_verification_three_bus_collapse_v1_bisection.m) -
% the scenario's designed worst-case sustained load for a TARGET bus
% (G, H, K are all target buses in this scenario).
TARGET_PLATEAU_KW = 190.0;   % TARGET_PLATEAU_W = 190000 W

fprintf('%s\n', repmat('=', 1, 70));
fprintf('GA-PSO SHED SEARCH - WORST-CASE PHYSICAL BOUNDS (v2)\n');
fprintf('%s\n', repmat('=', 1, 70));

if ~isfile(SLICE_CSV)
    error(['Slice not found: %s\nRe-run extract_dvsi_slices.py first ' ...
           '(it now also pulls CommandedLoad_kW_<bus> columns).'], SLICE_CSV);
end

T = readtable(SLICE_CSV);
[T, bus_flags] = compute_dvsi(T);

% ---- Same worst-case-row search as prototype_ga_shed.m, used ONLY to
% pick each bus's stressed line (G/H/K's bottleneck converter) and a
% representative V/Vn pair. Load and Gen from this row are printed for
% comparison but then OVERRIDDEN below with the worst-case physical
% bounds - see file header. ----
CONV_FOR_BUS = containers.Map( ...
    {'G','H','K'}, {{'FG','GH'}, {'GH','HK'}, {'HK','KL'}});

gated_buses = {};
worst_idx = containers.Map('KeyType', 'char', 'ValueType', 'double');
for i = 1:numel(CANDIDATE_BUSES)
    bus = CANDIDATE_BUSES{i};
    gcol = sprintf('DVSI_%s_gated', bus);
    vcol = sprintf('V_Bus_%s', bus);
    load_col = sprintf('CommandedLoad_kW_%s', bus);
    gen_col  = sprintf('Bus_%s_Src_Pow', bus);
    needed = {gcol, vcol, load_col, gen_col};
    if ~all(ismember(needed, T.Properties.VariableNames))
        continue
    end

    valid = T.(gcol) & (T.(vcol) > 100);

    lines_i = CONV_FOR_BUS(bus);
    for j = 1:2
        ln = lines_i{j};
        if ln(1) == bus, nb = ln(2); else, nb = ln(1); end
        nb_col = sprintf('V_Bus_%s', nb);
        if ismember(nb_col, T.Properties.VariableNames)
            valid = valid & (T.(nb_col) > 100);
        end
    end

    if ~any(valid)
        continue
    end

    deficit = T.(load_col) - T.(gen_col)/1000;
    deficit(~valid) = -inf;
    [~, idx_bus] = max(deficit);

    gated_buses{end+1} = bus; %#ok<AGROW>
    worst_idx(bus) = idx_bus;
end

if isempty(gated_buses)
    fprintf('No buses have a valid pre-collapse gated window in this slice.\n');
    return
end
fprintf('Gated buses: %s\n\n', strjoin(gated_buses, ', '));

% ---- Build the per-bus struct, using worst-case PHYSICAL bounds for
% Load/Gen instead of the empirical snapshot row. ----
snap = struct('bus', {}, 'Load', {}, 'Gen', {}, 'V', {}, 'Vn', {});
for i = 1:numel(gated_buses)
    bus = gated_buses{i};
    idx = worst_idx(bus);
    lines = CONV_FOR_BUS(bus);

    dvsi_vals = zeros(1, 2);
    for j = 1:2
        col = sprintf('DVSI_%s', lines{j});
        dvsi_vals(j) = T.(col)(idx);
    end
    [~, best] = max(dvsi_vals);
    stressed_line = lines{best};

    if stressed_line(1) == bus
        neighbor = stressed_line(2);
    else
        neighbor = stressed_line(1);
    end

    load_col = sprintf('CommandedLoad_kW_%s', bus);
    gen_col  = sprintf('Bus_%s_Src_Pow', bus);
    v_col    = sprintf('V_Bus_%s', bus);
    vn_col   = sprintf('V_Bus_%s', neighbor);

    if ~ismember(vn_col, T.Properties.VariableNames)
        error(['Column %s not in this slice. Re-run extract_dvsi_slices.py ' ...
               '(updated to include CommandedLoad_kW_<bus>) and try again.'], vn_col);
    end

    empirical_load = T.(load_col)(idx);
    empirical_gen  = T.(gen_col)(idx) / 1000;

    % ---- THE ACTUAL CHANGE vs. prototype_ga_shed.m: override Load/Gen
    % with the scenario's known worst-case physical bounds. V/Vn still
    % come from the empirical worst-case row (see file header). ----
    snap(end+1) = struct( ...          %#ok<AGROW>
        'bus', bus, ...
        'Load', TARGET_PLATEAU_KW, ...
        'Gen', 0, ...
        'V', T.(v_col)(idx), ...
        'Vn', T.(vn_col)(idx));

    fprintf(['  Bus %s: stressed line %s (neighbor %s), V=%.1fV, V_%s=%.1fV\n' ...
             '           empirical row had Load=%.1f kW, Gen=%.1f kW (t=%.2fs)\n' ...
             '           WORST-CASE bound used instead: Load=%.1f kW, Gen=0 kW\n'], ...
            bus, stressed_line, neighbor, snap(end).V, neighbor, snap(end).Vn, ...
            empirical_load, empirical_gen, T.time(idx), TARGET_PLATEAU_KW);
end

% ---- Same grid search + constrained-feasibility formulation as
% prototype_ga_shed.m (Section 11/12 of the project doc) - only the
% snap() inputs changed above. ----
LOWER_BOUND = 0.05;
UB_MAX = 0.95;                  % widened further - worst-case Load/Gen can
                                 % require much more than 60% for some buses
GRID_RESOLUTION = 0.005;
N = numel(snap);
GRID_STEPS = round((UB_MAX - LOWER_BOUND) / GRID_RESOLUTION) + 1;
grid_vals = linspace(LOWER_BOUND, UB_MAX, GRID_STEPS);

fprintf('\nRunning brute-force grid search over %d bus(es), %d points each ', N, GRID_STEPS);
fprintf('([%.0f%%, %.0f%%], %d combinations)...\n\n', ...
        100*LOWER_BOUND, 100*UB_MAX, GRID_STEPS^N);

RESOLVED_THRESHOLD = 0.9;

num_combos = GRID_STEPS^N;
best_total_shed = inf;
s_opt = [];
info_opt = [];

best_worst_dvsi = inf;
s_closest = grid_vals(1) * ones(1, N);

for combo_idx = 0:num_combos-1
    v = combo_idx;
    digits = zeros(1, N);
    for k = 1:N
        digits(k) = mod(v, GRID_STEPS);
        v = floor(v / GRID_STEPS);
    end
    s_try = grid_vals(digits + 1);
    [~, info_try] = ga_shed_objective(s_try, snap);
    dvsi_vals = [info_try.DVSI_after];

    worst_dvsi = max(dvsi_vals);
    if worst_dvsi < best_worst_dvsi
        best_worst_dvsi = worst_dvsi;
        s_closest = s_try;
    end

    if all(dvsi_vals < RESOLVED_THRESHOLD)
        total_shed = sum([info_try.power_shed_kW]);
        if total_shed < best_total_shed
            best_total_shed = total_shed;
            s_opt = s_try;
            info_opt = info_try;
        end
    end
end

fprintf('\n%s\n', repmat('=', 1, 70));
if isempty(s_opt)
    fprintf('NO FEASIBLE COMBINATION FOUND\n');
    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('No combination within [%.0f%%, %.0f%%] per bus resolves every gated bus\n', ...
            100*LOWER_BOUND, 100*UB_MAX);
    fprintf('at the same time under worst-case Load/Gen. Closest attempt\n');
    fprintf('(smallest worst-case DVSI_after=%.3f):\n', best_worst_dvsi);
    [~, info_closest] = ga_shed_objective(s_closest, snap);
    for k = 1:N
        fprintf('  Bus %s: shed %.1f%% (%.1f kW) -> DVSI_after=%.3f\n', ...
                info_closest(k).bus, 100*info_closest(k).shed_frac, ...
                info_closest(k).power_shed_kW, info_closest(k).DVSI_after);
    end
else
    fprintf('RESULT: least total shedding that resolves every gated bus under\n');
    fprintf('worst-case Load/Gen bounds (%.1f kW total)\n', best_total_shed);
    fprintf('%s\n', repmat('=', 1, 70));
    for k = 1:N
        fprintf('  Bus %s: shed %.1f%% (%.1f kW) -> DVSI_after=%.3f (resolved)\n', ...
                info_opt(k).bus, 100*info_opt(k).shed_frac, info_opt(k).power_shed_kW, ...
                info_opt(k).DVSI_after);
    end
    fprintf('\nCompare against the first (empirical-snapshot) result:\n');
    fprintf('  Bus G: 48.5%%   Bus K: 5.0%%   -- G validated for real, K did not (still\n');
    fprintf('  collapsed in the real closed-loop run). Expect this worst-case version to\n');
    fprintf('  recommend AT LEAST as much shedding for both, likely more for K.\n');
    fprintf('\nNEXT STEP: run the closed-loop validation again\n');
    fprintf('(run_closed_loop_verification_three_bus_collapse_v1_dvsi_candidate.m) with\n');
    fprintf('SHED_BUSES/SHED_REMAINING updated to this result, and check whether Bus K\n');
    fprintf('(and then Bus H, still left unshed to test the cascade hypothesis) survive.\n');
end
