function J = ga_shed_fitness_power(shed_fracs, snap)
%% GA_SHED_FITNESS_POWER.M
% =========================================================================
% Objective function for ga()/particleswarm(): total power shed across all
% gated buses, in kW, to be MINIMIZED.
%
% DVSI is deliberately NOT part of this objective -- it is enforced
% separately as a hard constraint (see ga_shed_nonlcon.m), per the
% constrained-search design already validated in the original
% prototype_ga_shed.m / prototype_ga_shed_busC.m (project doc Section 11):
% DVSI is near-binary (a bus either holds or collapses -- no partial
% credit for "70% likely to survive"), so trading shed amount off against
% a weighted stress term is the wrong shape of problem. This function
% answers the real question directly: minimize power shed, subject to
% every gated bus actually being resolved.
%
% INPUTS:
%   shed_fracs : 1xN row vector, candidate shed fraction per gated bus
%   snap       : 1xN struct array from extract_gated_snapshot.m (fields:
%                bus, Load, Gen, V, Vn)
%
% OUTPUT:
%   J : scalar, total power shed in kW (lower is better)
% =========================================================================

N = numel(snap);
power_shed = zeros(1, N);
for k = 1:N
    power_shed(k) = shed_fracs(k) * snap(k).Load;
end
J = sum(power_shed);

end
