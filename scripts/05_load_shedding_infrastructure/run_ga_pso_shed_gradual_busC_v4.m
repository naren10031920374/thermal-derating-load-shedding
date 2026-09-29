%% RUN_GA_PSO_SHED_GRADUAL_BUSC_V4.M
% =========================================================================
% Real GA-PSO shed-amount search (MATLAB ga()/particleswarm(), GADS
% Toolbox confirmed licensed 2026-09-22) for the gradual_busC_v4 scenario
% -- Bus C alone.
%
% USAGE: run directly:  run_ga_pso_shed_gradual_busC_v4
% =========================================================================

clear; clc;

SLICE_CSV = fullfile('..', '..', 'Claude outputs', 'dvsi_validation', ...
                      'gradual_busC_v4_busC_dvsi_slice.csv');
CANDIDATE_BUSES = {'C'};
CONV_FOR_BUS = containers.Map({'C'}, {{'BC', 'CD'}});

KNOWN_REFERENCE = [ ...
    'Real closed-loop bisection search already found Bus C survives at ' ...
    '40% shed (remaining=0.60), fails at 30% shed (remaining=0.70) -- ' ...
    'true minimum is somewhere in (30%, 40%]. If ga()/particleswarm() ' ...
    'land near 40%, that''s good cross-validation.'];

[ga_result, pso_result] = run_ga_pso_shed_search('gradual_busC_v4', ...
    SLICE_CSV, CANDIDATE_BUSES, CONV_FOR_BUS, KNOWN_REFERENCE); %#ok<NASGU>
