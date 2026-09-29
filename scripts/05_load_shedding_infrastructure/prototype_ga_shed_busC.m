%% PROTOTYPE_GA_SHED_BUSC.M
% =========================================================================
% gradual_busC_v4 analog of prototype_ga_shed.m (project doc Section 11).
% Same DVSI-based analytical proxy and constrained grid search, applied to
% this scenario's own gated bus: Bus C only (per project doc Section 7,
% Bus C's two incident lines BC/CD are both chronically pinned at ±90%
% for the whole extraction window -- a strong, early-warning gate, unlike
% three_bus_collapse_v1's mix of chronic and reactive-only buses).
%
% KNOWN GROUND TRUTH TO CROSS-CHECK AGAINST (from the real closed-loop
% bisection search, run_closed_loop_verification_gradual_busC_v4_bisection.m):
% Bus C alone survives at ShedFrac remaining=0.60 (shed 40%), and fails at
% remaining=0.70 (shed 30%) -- so the true minimum shed is somewhere in
% (30%, 40%], with 40% being the confirmed-safe value found by bisection.
% If this proxy's answer lands anywhere close to 40%, that's good
% cross-validation, the same way three_bus_collapse_v1's Bus G proxy
% number (48.5%) lined up with its own real bisection answer (50%).
%
% USAGE: run directly:  prototype_ga_shed_busC
% =========================================================================

clear; clc;

SLICE_CSV = fullfile('..', '..', 'Claude outputs', 'dvsi_validation', ...
                      'gradual_busC_v4_busC_dvsi_slice.csv');
CANDIDATE_BUSES = {'C'};

fprintf('%s\n', repmat('=', 1, 70));
fprintf('PROTOTYPE GA-PSO SHED SEARCH -- gradual_busC_v4, Bus C\n');
fprintf('%s\n', repmat('=', 1, 70));

if ~isfile(SLICE_CSV)
    error(['Slice not found: %s\nRe-run extract_dvsi_slices.py first.'], SLICE_CSV);
end

T = readtable(SLICE_CSV);
[T, bus_flags] = compute_dvsi(T);

% ---- Bus C's own worst-case moment, within its own gated window. ----
CONV_FOR_BUS = containers.Map({'C'}, {{'BC', 'CD'}});

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

% ---- Build the per-bus struct ga_shed_objective.m needs. ----
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
        error('Column %s not in this slice.', vn_col);
    end

    snap(end+1) = struct( ...          %#ok<AGROW>
        'bus', bus, ...
        'Load', T.(load_col)(idx), ...
        'Gen', T.(gen_col)(idx) / 1000, ...
        'V', T.(v_col)(idx), ...
        'Vn', T.(vn_col)(idx));

    fprintf('  Bus %s: worst case at t=%.2fs, stressed line %s (neighbor %s), Load=%.1f kW, Gen=%.1f kW, V=%.1fV, V_%s=%.1fV\n', ...
            bus, T.time(idx), stressed_line, neighbor, snap(end).Load, snap(end).Gen, snap(end).V, neighbor, snap(end).Vn);
end

% ---- Constrained grid search, same as prototype_ga_shed.m Section 11/12. ----
LOWER_BOUND = 0.05;
UB_MAX = 0.90;
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
    [~, info_closest] = ga_shed_objective(s_closest, snap);
    for k = 1:N
        fprintf('  Bus %s: shed %.1f%% (%.1f kW) -> DVSI_after=%.3f\n', ...
                info_closest(k).bus, 100*info_closest(k).shed_frac, ...
                info_closest(k).power_shed_kW, info_closest(k).DVSI_after);
    end
else
    fprintf('RESULT: least total shedding that resolves every gated bus (%.1f kW total)\n', best_total_shed);
    fprintf('%s\n', repmat('=', 1, 70));
    for k = 1:N
        fprintf('  Bus %s: shed %.1f%% (%.1f kW) -> DVSI_after=%.3f (resolved)\n', ...
                info_opt(k).bus, 100*info_opt(k).shed_frac, info_opt(k).power_shed_kW, ...
                info_opt(k).DVSI_after);
    end
    fprintf('\nCompare against the real closed-loop bisection answer for this scenario:\n');
    fprintf('  Bus C alone survives at 40%% shed (remaining=0.60), fails at 30%% shed\n');
    fprintf('  (remaining=0.70). If the number above lands near 40%%, that''s good\n');
    fprintf('  cross-validation of the proxy -- same pattern as Bus G in\n');
    fprintf('  three_bus_collapse_v1 (proxy 48.5%% vs. real bisection 50%%).\n');
    fprintf('\nNEXT STEP: validate this number with one real closed-loop Simulink run\n');
    fprintf('before trusting it -- three_bus_collapse_v1''s lesson was that even a\n');
    fprintf('proxy number that LOOKS resolved (DVSI_after<0.9) can still fail for real.\n');
end
