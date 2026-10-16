% ==========================================================================
%  TEST_PROFILE_LOADS.M   (no Simulink needed, runs in a few seconds)
%  Builds the target-bus load of S034 (plain ramp) and H01-H03 (example profile)
%  with build_scenario_loads and prints the load at a few times, so you can compare
%  with the numbers below. The real builder adds a slow sine (+-4 kW) and 0.8 kW
%  noise, so allow about +-6 kW.
%
%  Expected (kW, target bus):
%     S034  t=0 27   t=1850 205   t=3150 205   t=5000 205     (plateau, unchanged)
%     H01   t=0 27   t=1850 201   t=3150  89   t=5000 186
%     H02   t=0 27   t=1565 187   t=2860  99   t=5000 202
%     H03   t=0 27   t=2964 175   t=4150  78   t=5000 176
%  Run from scripts/09_dataset_pipeline:   test_profile_loads
% ==========================================================================
here = fileparts(mfilename('fullpath'));
tbl  = readtable(fullfile(here, 'scenario_table.csv'), 'TextType', 'string');
BUS  = {'A','B','C','D','E','F','G','H','K','L'};
TS = 0.01; N = round(5000/TS) + 1;
times = struct('S034', [0 1850 3150 5000], 'H01', [0 1850 3150 5000], ...
               'H02', [0 1565 2860 5000], 'H03', [0 2964 4150 5000]);
for id = ["S034", "H01", "H02", "H03"]
    r = tbl(strcmp(tbl.scenario_id, id), :);
    row = struct('scenario_id', char(r.scenario_id), ...
        'targets', {strsplit(char(r.targets), ';')}, 'boundary', {strsplit(char(r.boundary), ';')}, ...
        'ramp_start_s', double(r.ramp_start_s), 'ramp_end_s', double(r.ramp_end_s), ...
        'plateau_w', double(r.plateau_w), 'boundary_low_w', double(r.boundary_low_w), ...
        'preheat_w', double(r.preheat_w), 'boundary_preheat_w', double(r.boundary_preheat_w), ...
        'jitter_frac', double(r.jitter_frac), 'stepdown_start_s', double(r.stepdown_start_s), ...
        'stepdown_end_s', double(r.stepdown_end_s), 'recovery_w', double(r.recovery_w), ...
        'boundary_recovery_w', double(r.boundary_recovery_w), 'seed', double(r.seed));
    row = add_profile_fields(row, here);
    L = build_scenario_loads(row, BUS, N, TS, 2000, 220000);
    k = find(strcmp(BUS, row.targets{1}));
    tt = times.(char(id));
    fprintf('%-5s target %s  profile=%d | ', id, row.targets{1}, isfield(row, 'profile_file'));
    for j = 1:numel(tt)
        fprintf('t=%4d: %6.1f kW   ', tt(j), L(round(tt(j)/TS) + 1, k) / 1000);
    end
    fprintf('\n');
end
