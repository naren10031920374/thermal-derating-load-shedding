function [gated_buses, snap] = extract_gated_snapshot(T, CANDIDATE_BUSES, CONV_FOR_BUS)
%% EXTRACT_GATED_SNAPSHOT.M
% =========================================================================
% Shared gating + worst-case-snapshot logic, factored out of the original
% prototype_ga_shed.m / prototype_ga_shed_busC.m so every scenario's
% GA-PSO driver (run_ga_pso_shed_search.m) uses the identical, already-
% validated per-bus worst-case selection (project doc Section 11):
%
%   - a bus is a candidate only if BOTH its incident converter columns
%     are present in the slice
%   - "gated" = compute_dvsi's DVSI_<bus>_gated flag true, and this bus's
%     own voltage still healthy (not yet collapsed)
%   - ALSO require both of the bus's neighbor voltages still healthy, to
%     avoid a divide-by-zero-adjacent artifact where a collapsed
%     neighbor's V~0 makes DVSI_after blow up to 1.000 regardless of shed
%     fraction (project doc Section 11, bug #4)
%   - among valid rows, pick the one where this bus's own deficit
%     (Load - Gen) is largest -- its true worst case, not one shared
%     snapshot time (Bus_X_Src_Pow swings 0W to ~95kW within the same
%     window, so a shared snapshot can catch a bus at a comfortable
%     moment -- project doc Section 11, design correction)
%
% INPUTS:
%   T                - table, already run through compute_dvsi(T)
%   CANDIDATE_BUSES  - cell array of bus letters to check, e.g. {'G','H','K'}
%   CONV_FOR_BUS     - containers.Map, bus letter -> {line1, line2}, e.g.
%                      containers.Map({'G'}, {{'FG','GH'}})
%
% OUTPUTS:
%   gated_buses - cell array of bus letters that had a valid gated window
%   snap        - struct array (bus, Load, Gen, V, Vn), one entry per
%                 gated bus, ready for ga_shed_objective.m /
%                 ga_shed_fitness_power.m / ga_shed_nonlcon.m
% =========================================================================

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

snap = struct('bus', {}, 'Load', {}, 'Gen', {}, 'V', {}, 'Vn', {});
if isempty(gated_buses)
    return
end

for i = 1:numel(gated_buses)
    bus = gated_buses{i};
    idx = worst_idx(bus);
    lines = CONV_FOR_BUS(bus);

    % Pick whichever of this bus's two lines has the higher DVSI at this
    % bus's own worst-case moment -- that's the actual bottleneck line.
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

    % NOTE: Bus_X_Src_Pow is logged in WATTS, unlike CommandedLoad_kW_<bus>
    % which is already in kW -- must convert here to match units.
    snap(end+1) = struct( ...          %#ok<AGROW>
        'bus', bus, ...
        'Load', T.(load_col)(idx), ...
        'Gen', T.(gen_col)(idx) / 1000, ...
        'V', T.(v_col)(idx), ...
        'Vn', T.(vn_col)(idx));

    fprintf('  Bus %s: worst case at t=%.2fs, stressed line %s (neighbor %s), Load=%.1f kW, Gen=%.1f kW, V=%.1fV, V_%s=%.1fV\n', ...
            bus, T.time(idx), stressed_line, neighbor, snap(end).Load, snap(end).Gen, snap(end).V, neighbor, snap(end).Vn);
end

end
