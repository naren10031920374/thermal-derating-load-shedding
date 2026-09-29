function [ga_result, pso_result] = run_ga_pso_shed_search(SCENARIO_NAME, SLICE_CSV, ...
    CANDIDATE_BUSES, CONV_FOR_BUS, KNOWN_REFERENCE_TEXT)
%% RUN_GA_PSO_SHED_SEARCH.M
% =========================================================================
% Shared driver for the REAL Paper-2-style GA-PSO shed-amount search, now
% that the Global Optimization Toolbox is confirmed licensed
% (license('test','GADS_Toolbox') == 1). Replaces the earlier brute-force
% grid-search prototypes (prototype_ga_shed.m / prototype_ga_shed_busC.m)
% with MATLAB's own ga() and particleswarm() functions.
%
% SCOPE NOTE (2026-09-22): per the professor's guidance, DVSI is no longer
% used to decide WHICH bus is stressed or WHEN (the existing ML collapse
% detector already does that). It is still used HERE, internally, as the
% fast analytical stand-in that lets ga()/particleswarm() score thousands
% of candidate shed-vectors without a real Simulink run per candidate --
% this is a deliberate, acknowledged reuse of paper 1's physics as an
% implementation detail inside paper 2's optimization technique, not a
% reintroduction of paper 1 as the "which bus/when" driver. Flag this to
% the professor explicitly (see project doc Section 13/14).
%
% Every solver's winning shed vector is analytical (DVSI-based), NOT
% simulated -- exactly like the earlier grid-search prototypes, it still
% requires one real closed-loop Simulink validation run before being
% trusted (project doc Section 12, Iteration 1 is the concrete lesson why:
% the DVSI proxy nailed Bus G's number but was badly wrong for Bus K).
%
% INPUTS:
%   SCENARIO_NAME     : string, for display only, e.g. 'three_bus_collapse_v1'
%   SLICE_CSV         : path to this scenario's DVSI validation slice CSV
%   CANDIDATE_BUSES   : cell array of bus letters to check, e.g. {'G','H','K'}
%   CONV_FOR_BUS      : containers.Map, bus letter -> {line1, line2}
%   KNOWN_REFERENCE_TEXT : string, printed at the end for cross-checking
%                          against the scenario's real closed-loop result
%
% OUTPUTS:
%   ga_result, pso_result : structs with fields x, fval, exitflag, snap
% =========================================================================

LOWER_BOUND = 0.05;   % Paper 2's shed-fraction floor
UB_MAX      = 0.90;   % widened per project doc Section 12: Paper 2's 20%
                       % cap does not fit this system -- real numbers here
                       % run 40-50%+.

fprintf('%s\n', repmat('=', 1, 70));
fprintf('GA-PSO SHED SEARCH -- %s (MATLAB ga()/particleswarm(), GADS Toolbox)\n', SCENARIO_NAME);
fprintf('%s\n', repmat('=', 1, 70));

if ~isfile(SLICE_CSV)
    error('Slice not found: %s', SLICE_CSV);
end
T = readtable(SLICE_CSV);
[T, ~] = compute_dvsi(T);

[gated_buses, snap] = extract_gated_snapshot(T, CANDIDATE_BUSES, CONV_FOR_BUS);
if isempty(gated_buses)
    fprintf('No buses have a valid pre-collapse gated window in this slice.\n');
    ga_result = []; pso_result = [];
    return
end
fprintf('Gated buses (worst-case moment found within each one''s own gated window): %s\n\n', ...
        strjoin(gated_buses, ', '));

N = numel(snap);
lb = LOWER_BOUND * ones(1, N);
ub = UB_MAX * ones(1, N);
fitnessfcn = @(x) ga_shed_fitness_power(x, snap);
nonlcon    = @(x) ga_shed_nonlcon(x, snap);

%% ---- Genetic Algorithm (ga()) ----
ga_opts = optimoptions('ga', 'Display', 'iter', 'PopulationSize', 50, ...
    'MaxGenerations', 100, 'FunctionTolerance', 1e-6, 'ConstraintTolerance', 1e-4);
rng(42, 'twister');   % reproducible run
[x_ga, fval_ga, exitflag_ga] = ga(fitnessfcn, N, [], [], [], [], lb, ub, nonlcon, ga_opts);

fprintf('\n--- ga() result ---\n');
print_shed_result(x_ga, snap, fval_ga, exitflag_ga);

%% ---- Particle Swarm Optimization (particleswarm()), cross-check vs. ga() ----
% particleswarm() has no native nonlinear-constraint argument, so the same
% DVSI<0.9 constraint from ga_shed_nonlcon.m is folded in as a large
% penalty on any candidate that violates it, keeping both solvers on the
% exact same underlying constrained problem.
pso_opts = optimoptions('particleswarm', 'Display', 'iter', ...
    'SwarmSize', 50, 'MaxIterations', 100, 'FunctionTolerance', 1e-6);
penalty_fcn = @(x) pso_penalized_objective(x, snap);
rng(42, 'twister');
[x_pso, fval_pso_penalized, exitflag_pso] = particleswarm(penalty_fcn, N, lb, ub, pso_opts);

fprintf('\n--- particleswarm() result ---\n');
fval_pso_true = ga_shed_fitness_power(x_pso, snap);   % true power-shed value, unpenalized
print_shed_result(x_pso, snap, fval_pso_true, exitflag_pso);

fprintf('\nCross-check vs. known reference:\n%s\n', KNOWN_REFERENCE_TEXT);
fprintf('\nNEXT STEP before trusting either result: validate the winning shed vector\n');
fprintf('with one real closed-loop Simulink run (same "trust but verify" step used\n');
fprintf('for every other number in this project) -- this search is analytical\n');
fprintf('(DVSI-based), not simulated, and DVSI-based predictions have been wrong\n');
fprintf('per-bus before (project doc Section 12, Iteration 1: right for Bus G,\n');
fprintf('wrong for Bus K).\n');

ga_result  = struct('x', x_ga,  'fval', fval_ga,        'exitflag', exitflag_ga,  'snap', snap);
pso_result = struct('x', x_pso, 'fval', fval_pso_true,  'exitflag', exitflag_pso, 'snap', snap);

end

% ---- Local helper functions ----

function print_shed_result(x, snap, fval, exitflag)
[~, info] = ga_shed_objective(x, snap);
for k = 1:numel(snap)
    status = 'resolved';
    if info(k).DVSI_after >= 0.9
        status = 'STILL STRESSED';
    end
    fprintf('  Bus %s: shed %.1f%% (%.1f kW) -> DVSI_after=%.3f (%s)\n', ...
            info(k).bus, 100*info(k).shed_frac, info(k).power_shed_kW, ...
            info(k).DVSI_after, status);
end
fprintf('  Total power shed: %.1f kW | exitflag=%d\n', fval, exitflag);
end

function J = pso_penalized_objective(x, snap)
[c, ~] = ga_shed_nonlcon(x, snap);
base = ga_shed_fitness_power(x, snap);
penalty = 1e6 * sum(max(c, 0));   % heavy penalty for any bus left unresolved
J = base + penalty;
end
