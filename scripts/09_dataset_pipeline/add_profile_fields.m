function row = add_profile_fields(row, script_dir)
% ==========================================================================
%  ADD_PROFILE_FIELDS.M
%  If row.scenario_id is listed in profile_scenarios.csv (next to this file),
%  adds row.profile_file and row.profile_x0_s so build_scenario_loads.m uses the
%  example-profile envelope. For every other scenario the row is returned
%  unchanged, so S0xx / T0xx / X / Y scenarios behave exactly as before.
% ==========================================================================
    f = fullfile(script_dir, 'profile_scenarios.csv');
    if ~isfile(f); return; end
    T = readtable(f, 'TextType', 'string');
    i = find(strcmp(T.scenario_id, string(row.scenario_id)), 1);
    if isempty(i); return; end
    row.profile_file = char(T.profile_file(i));
    row.profile_x0_s = double(T.profile_x0_s(i));
end
