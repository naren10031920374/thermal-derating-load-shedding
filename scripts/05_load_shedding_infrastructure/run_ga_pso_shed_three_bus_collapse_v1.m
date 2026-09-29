%% RUN_GA_PSO_SHED_THREE_BUS_COLLAPSE_V1.M
% =========================================================================
% Real GA-PSO shed-amount search (MATLAB ga()/particleswarm(), GADS
% Toolbox confirmed licensed 2026-09-22) for the three_bus_collapse_v1
% scenario -- buses G, H, K stressed together. Jointly optimizes all
% three buses' shed fractions at once (unlike the earlier iterative
% single-bus approach used to find the real closed-loop answer).
%
% USAGE: run directly:  run_ga_pso_shed_three_bus_collapse_v1
% =========================================================================

clear; clc;

SLICE_CSV = fullfile('..', '..', 'Claude outputs', 'dvsi_validation', ...
                      'three_bus_collapse_v1_dvsi_slice.csv');
CANDIDATE_BUSES = {'G', 'H', 'K'};
CONV_FOR_BUS = containers.Map( ...
    {'G', 'H', 'K'}, {{'FG', 'GH'}, {'GH', 'HK'}, {'HK', 'KL'}});

KNOWN_REFERENCE = [ ...
    'Real closed-loop iterative search already found a working vector: ' ...
    'G=48.5%, H=50.0%, K=50.0% -- all ten buses survive ' ...
    '(project doc Section 12, Iteration 3). If ga()/particleswarm() land ' ...
    'near these numbers jointly, that''s good cross-validation of both the ' ...
    'DVSI-based fitness/constraint and the earlier iterative result.'];

[ga_result, pso_result] = run_ga_pso_shed_search('three_bus_collapse_v1', ...
    SLICE_CSV, CANDIDATE_BUSES, CONV_FOR_BUS, KNOWN_REFERENCE); %#ok<NASGU>
