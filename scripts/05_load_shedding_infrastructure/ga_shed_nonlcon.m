function [c, ceq] = ga_shed_nonlcon(shed_fracs, snap)
%% GA_SHED_NONLCON.M
% =========================================================================
% Nonlinear inequality constraint for ga()/particleswarm(): every gated
% bus's predicted DVSI_after (the same DAB-physics-based stress proxy
% derived in the project doc, Section 3, and implemented in
% ga_shed_objective.m) must drop below the project's standard "resolved"
% threshold, 0.9 -- the same threshold used everywhere else in this
% project (compute_dvsi.m's gate condition, the original grid-search
% prototypes' feasibility check, etc.).
%
% ga() requires constraints in the form c(x) <= 0, so:
%   c(k) = DVSI_after(k) - 0.9
%
% particleswarm() has no native nonlinear-constraint argument -- for that
% solver this function's [c] output is instead folded into a penalty term
% by the caller (see run_ga_pso_shed_search.m's pso_penalized_objective),
% so both solvers are optimizing against the exact same physical
% constraint.
%
% Reuses ga_shed_objective.m purely for its DVSI_after computation (its
% own weighted-sum J output is unused here) -- keeps the DVSI physics in
% one place rather than duplicating it.
%
% INPUTS:
%   shed_fracs : 1xN row vector, candidate shed fraction per gated bus
%   snap       : 1xN struct array from extract_gated_snapshot.m
%
% OUTPUTS:
%   c   : 1xN vector, DVSI_after(k) - 0.9 for each gated bus (<=0 = OK)
%   ceq : [] (no equality constraints)
% =========================================================================

RESOLVED_THRESHOLD = 0.9;
[~, info] = ga_shed_objective(shed_fracs, snap);
c = [info.DVSI_after] - RESOLVED_THRESHOLD;
ceq = [];

end
