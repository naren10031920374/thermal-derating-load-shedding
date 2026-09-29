%% PROTOTYPE_GA_SHED.M
% =========================================================================
% First runnable prototype of the GA-PSO shed-amount stage (project doc,
% Section 11). Uses compute_dvsi.m to find which buses are gated at a
% pre-collapse snapshot, then searches for per-bus shed fractions (bounded
% 5-20%, Paper 2) that minimize ga_shed_objective.m's weighted cost
% (0.7 power shed + 0.3 remaining stress). Searches via plain brute-force
% grid search (no Global Optimization Toolbox license needed/available on
% this machine) - fine at this variable count (2-3 buses in the validated
% scenarios).
%
% This is a PROTOTYPE: ga_shed_objective.m evaluates candidates with a
% fast analytical proxy, not a real Simulink re-run (see that file's
% header for why, and the caveat about validating the winner for real).
%
% USAGE: edit SLICE_CSV below to point at one of the three DVSI
% validation slices (re-extract first if it doesn't yet have the
% CommandedLoad_kW_<bus> columns - see extract_dvsi_slices.py, updated to
% pull them). Then just run:  prototype_ga_shed
% =========================================================================

clear; clc;

SLICE_CSV = fullfile('..', '..', 'Claude outputs', 'dvsi_validation', ...
                      'three_bus_collapse_v1_dvsi_slice.csv');
% Buses in this scenario whose local Load/Gen columns are available.
% (Only buses with BOTH incident converters present in the slice get a
% bus-level gate from compute_dvsi - here that's G, H, K.)
CANDIDATE_BUSES = {'G', 'H', 'K'};

fprintf('%s\n', repmat('=', 1, 70));
fprintf('PROTOTYPE GA-PSO SHED SEARCH\n');
fprintf('%s\n', repmat('=', 1, 70));

if ~isfile(SLICE_CSV)
    error(['Slice not found: %s\nRe-run extract_dvsi_slices.py first ' ...
           '(it now also pulls CommandedLoad_kW_<bus> columns).'], SLICE_CSV);
end

T = readtable(SLICE_CSV);
[T, bus_flags] = compute_dvsi(T);

% ---- Pick each bus's OWN worst-case moment, not one shared snapshot. ----
% IMPORTANT LESSON (found by checking the actual data after the first
% prototype run looked wrong): Bus_X_Src_Pow is NOT a steady value - in
% this window alone it swings from 0 W up to a ~95kW ceiling. A single
% shared "last pre-collapse sample" can land on a comfortable moment
% (generation happens to be high right then) and badly understate how
% much shedding is really needed - that's exactly what happened the
% first time (it picked a moment where Bus K's generation was near its
% 95kW ceiling, so the apparent deficit was tiny). Instead, for each
% gated bus, use the moment WITHIN ITS OWN GATED WINDOW where its
% deficit (Load - Gen) is largest - the true worst case that a shed
% fraction actually has to survive.
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

    valid = T.(gcol) & (T.(vcol) > 100);   % gated, and not yet collapsed

    % Also require BOTH of this bus's neighbor voltages to still be
    % healthy at that row. Otherwise a row can get picked as "worst
    % case" where a neighbor has already collapsed (V=0) - the
    % DVSI_after formula divides by V*Vn, so a collapsed neighbor makes
    % that denominator ~0 and DVSI_after blows up to 1.000 regardless of
    % shed fraction. That's a divide-by-zero artifact, not a real
    % finding that shedding can't help - exactly what happened to Bus K
    % the first time this ran (its worst-deficit row landed right at
    % Bus H's own collapse instant, V_H=0V).
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

    deficit = T.(load_col) - T.(gen_col)/1000;   % Gen converted W -> kW
    deficit(~valid) = -inf;
    [~, idx_bus] = max(deficit);

    gated_buses{end+1} = bus; %#ok<AGROW>
    worst_idx(bus) = idx_bus;
end

if isempty(gated_buses)
    fprintf('No buses have a valid pre-collapse gated window in this slice.\n');
    return
end
fprintf('Gated buses (worst-case moment found within each one''s own gated window): %s\n\n', ...
        strjoin(gated_buses, ', '));

% ---- Build the per-bus baseline struct ga_shed_objective.m needs,
% each bus using ITS OWN worst-case row. ----
snap = struct('bus', {}, 'Load', {}, 'Gen', {}, 'V', {}, 'Vn', {});
for i = 1:numel(gated_buses)
    bus = gated_buses{i};
    idx = worst_idx(bus);
    lines = CONV_FOR_BUS(bus);

    % Pick whichever of this bus's two lines has the higher DVSI at this
    % bus's own worst-case moment - that's the actual bottleneck line.
    dvsi_vals = zeros(1, 2);
    for j = 1:2
        col = sprintf('DVSI_%s', lines{j});
        dvsi_vals(j) = T.(col)(idx);
    end
    [~, best] = max(dvsi_vals);
    stressed_line = lines{best};

    % Neighbor bus = the letter in the line code that isn't this bus
    % (works because bus codes here are single letters).
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

    % NOTE: Bus_X_Src_Pow is logged in WATTS (grid_derating_dataset_5000s_v7_2.m
    % never divides it by 1000 the way it does for CommandedLoad_kW), so it
    % must be converted here to match Load's kW units.
    snap(end+1) = struct( ...          %#ok<AGROW>
        'bus', bus, ...
        'Load', T.(load_col)(idx), ...
        'Gen', T.(gen_col)(idx) / 1000, ...
        'V', T.(v_col)(idx), ...
        'Vn', T.(vn_col)(idx));

    fprintf('  Bus %s: worst case at t=%.2fs, stressed line %s (neighbor %s), Load=%.1f kW, Gen=%.1f kW, V=%.1fV, V_%s=%.1fV\n', ...
            bus, T.time(idx), stressed_line, neighbor, snap(end).Load, snap(end).Gen, snap(end).V, neighbor, snap(end).Vn);
end

% ---- Run the search ----
% No Global Optimization Toolbox license on this machine, so ga()/
% particleswarm() aren't available. For this few variables (one per gated
% bus - at most 2-3 in the validated scenarios) a plain brute-force grid
% search over the bounded range is just as good, needs no toolbox, and is
% fully deterministic/reproducible.
%
% UB_MAX (2026-09-19, diagnostic widening): the previous run showed Bus G
% is "STILL STRESSED" even at Paper 2's stated 20% cap - by hand, it
% needs ~47% to actually resolve, suspiciously close to the real
% bisection search's 50%. That's the same pattern as Section 4's FVSI
% 0.25 threshold: a number carried over from Paper 2 that didn't actually
% fit this system. Widened here to test that directly: does a single-bus
% search (still no neighbor-shedding) converge near the real 50% once
% it's allowed to? If yes, the fix is just a looser bound, not added
% multi-bus complexity. LOWER_BOUND stays at Paper 2's 5% floor.
LOWER_BOUND = 0.05;
UB_MAX = 0.60;                  % widened from Paper 2's 20% - see above
GRID_RESOLUTION = 0.005;        % keep ~0.5% resolution even at the wider range
N = numel(snap);
GRID_STEPS = round((UB_MAX - LOWER_BOUND) / GRID_RESOLUTION) + 1;
grid_vals = linspace(LOWER_BOUND, UB_MAX, GRID_STEPS);

fprintf('\nRunning brute-force grid search over %d bus(es), %d points each ', N, GRID_STEPS);
fprintf('([%.0f%%, %.0f%%] - widened above Paper 2''s 20%% cap to test Bus G, %d combinations)...\n\n', ...
        100*LOWER_BOUND, 100*UB_MAX, GRID_STEPS^N);

% ---- CONSTRAINT-BASED search (2026-09-19), replacing the weighted-sum
% objective. Why: the first constrained-bound run showed the 70/30
% weighted sum letting Bus G stay fully stressed (DVSI_after=1.0) because
% paying to actually fix it cost more in the power-shed term than it
% saved in the (heavily discounted) stress term - a real result, but the
% wrong question to optimize. DVSI is near-binary (Section 7): a bus is
% either going to hold or it's going to collapse, there's no partial
% credit for "70% likely to survive." So instead of trading the two off,
% require every gated bus to actually be resolved (DVSI_after below the
% same 0.9 threshold used everywhere else in this project), and among
% only the combinations that satisfy that for every bus, pick the one
% that sheds the least total power. This directly answers the real
% question: the smallest amount of shedding that actually prevents
% collapse, rather than a smallest-regret compromise that lets a bus
% collapse anyway.
RESOLVED_THRESHOLD = 0.9;

num_combos = GRID_STEPS^N;
best_total_shed = inf;
s_opt = [];
info_opt = [];

% Track the least-bad infeasible point too (smallest worst-case
% DVSI_after across buses), purely for diagnosis if nothing is feasible.
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
    fprintf('at the same time. Closest attempt (smallest worst-case DVSI_after=%.3f):\n', best_worst_dvsi);
    [~, info_closest] = ga_shed_objective(s_closest, snap);
    for k = 1:N
        fprintf('  Bus %s: shed %.1f%% (%.1f kW) -> DVSI_after=%.3f\n', ...
                info_closest(k).bus, 100*info_closest(k).shed_frac, ...
                info_closest(k).power_shed_kW, info_closest(k).DVSI_after);
    end
    fprintf('\nThis is real evidence multi-bus shedding (a healthy neighbor too) may be\n');
    fprintf('needed - single-bus shedding alone, even up to %.0f%%, cannot resolve\n', 100*UB_MAX);
    fprintf('every gated bus simultaneously in this scenario.\n');
else
    fprintf('RESULT: least total shedding that resolves every gated bus (%.1f kW total)\n', best_total_shed);
    fprintf('%s\n', repmat('=', 1, 70));
    for k = 1:N
        fprintf('  Bus %s: shed %.1f%% (%.1f kW) -> DVSI_after=%.3f (resolved)\n', ...
                info_opt(k).bus, 100*info_opt(k).shed_frac, info_opt(k).power_shed_kW, ...
                info_opt(k).DVSI_after);
    end
    fprintf('\nSanity check vs. the existing manual bisection numbers:\n');
    fprintf('  Bisection used 50%% flat on G/H/K. If the shed %% above lands close to\n');
    fprintf('  50%% for the buses that needed it, that''s good evidence this proxy and\n');
    fprintf('  the constrained formulation are both on the right track.\n');
    fprintf('\nNEXT STEP before trusting this: validate the shed %% above with one real\n');
    fprintf('closed-loop Simulink run (set the ShedFrac_<bus> constants from\n');
    fprintf('add_controllable_load_shedding.m to 1-shed_frac and confirm every bus\n');
    fprintf('actually stays up) - this proxy is analytical, not simulated.\n');
end
