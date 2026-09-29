%% RUN_GA_PSO_SHED_BUS_F.M
% =========================================================================
% Real GA-PSO shed-amount search (MATLAB ga()/particleswarm(), GADS
% Toolbox confirmed licensed 2026-09-22) for the baseline_v7 scenario --
% Bus F alone. This is the one scenario with no real closed-loop minimum
% found yet (only a single fixed 30% shed has been tested, and it
% passed) -- run_closed_loop_verification_bus_f_bisection.m is the
% real-simulation search-based alternative already built for this same
% gap; this script is the DVSI-based ga()/particleswarm() alternative.
%
% USAGE: run directly:  run_ga_pso_shed_bus_f
% =========================================================================

clear; clc;

SLICE_CSV = fullfile('..', '..', 'Claude outputs', 'dvsi_validation', ...
                      'baseline_v7_busF_dvsi_slice.csv');
CANDIDATE_BUSES = {'F'};
CONV_FOR_BUS = containers.Map({'F'}, {{'EF', 'FG'}});

KNOWN_REFERENCE = [ ...
    'Only a single fixed 30% shed has been tested for real on Bus F so ' ...
    'far, and it passed -- no true minimum has been found yet, by any ' ...
    'method. Whatever ga()/particleswarm() suggest here still needs real ' ...
    'closed-loop validation before it can be trusted or compared.'];

[ga_result, pso_result] = run_ga_pso_shed_search('baseline_v7 (Bus F)', ...
    SLICE_CSV, CANDIDATE_BUSES, CONV_FOR_BUS, KNOWN_REFERENCE); %#ok<NASGU>
