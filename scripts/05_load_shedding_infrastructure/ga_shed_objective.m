%% GA_SHED_OBJECTIVE.M
% =========================================================================
% Fitness function for the GA-PSO load-shedding prototype (Paper 2's
% weighted objective: 70% minimize power shed, 30% minimize remaining
% stress), evaluated with a FAST ANALYTICAL PROXY instead of a full
% closed-loop Simulink re-run per candidate.
%
% WHY A PROXY, NOT A REAL SIMULINK RUN PER CANDIDATE:
% A population-based search (ga/particleswarm) evaluates dozens of
% candidate shed-vectors per generation, for many generations. Re-running
% the 5000s closed-loop model that many times is not practical for a
% prototype. Instead this reuses DVSI itself (already derived and
% validated - see the project doc, Sections 1-8) as the stress metric,
% which lets stress be estimated algebraically instead of simulated:
%
%   1. A bus's "stressed" line carries approximately its unmet local
%      demand:  D = Load - Gen  (Gen = Bus_X_Src_Pow, local generation).
%      This is a first-order approximation (ignores ring losses and how
%      the OTHER incident line might pick up some of the slack) - it is
%      deliberately the CONSERVATIVE direction: assuming the whole
%      shortfall routes through the one line already known to be pinned.
%   2. Shedding a fraction s of that bus's load reduces D to
%      D' = Load*(1-s) - Gen.
%   3. Inverting the DVSI relationship derived in the project doc
%      (P_ij = K' * Vi * Vj * f(phi), DVSI = f(phi)/f_max) gives the
%      resulting stress directly:  DVSI_after = D' / (K' * Vi * Vj * f_max)
%      holding voltages at their pre-shed snapshot values (voltages would
%      actually recover somewhat too - this proxy is conservative there
%      as well, since it does not credit that recovery).
%
% This is a PROTOTYPE evaluator, meant to get the optimizer's mechanics
% (bounds, weighting, convergence) working and testable today. Before
% treating any result as a real recommendation, the winning shed vector
% must be validated with one real closed-loop Simulink run (exactly the
% same "trust but verify" step already used for the bisection-search
% numbers) - see prototype_ga_shed.m's final printout for this reminder.
%
% INPUTS:
%   shed_fracs : 1xN row vector, shed fraction per gated bus, each
%                expected in [0.05, 0.20] per Paper 2's bound (enforced
%                by the caller's lb/ub, not inside this function).
%   snap       : 1xN struct array, one entry per gated bus, fields:
%                  .bus  - bus letter, e.g. 'G'
%                  .Load - commanded load at the snapshot, kW
%                  .Gen  - local source power at the snapshot, kW
%                  .V    - this bus's voltage at the snapshot, V
%                  .Vn   - neighbor bus's voltage (far end of this bus's
%                          stressed line) at the snapshot, V
%
% OUTPUTS:
%   J    : scalar fitness (lower is better) = 0.7*norm_power_shed +
%          0.3*mean(DVSI_after), both terms in [0,1].
%   info : 1xN struct array with per-bus diagnostics (power_shed_kW,
%          DVSI_after), for printing/inspection after ga() finishes.
% =========================================================================

function [J, info] = ga_shed_objective(shed_fracs, snap)

% ---- Same fixed DAB constants as compute_dvsi.m ----
Lk = 340e-6; fs = 10e3; n = 24/15;
Kp = n / (2*pi*fs*Lk);              % P_ij = Kp * Vi * Vj * f(phi)
f_max = (pi/2) * (1 - 0.5);         % = pi/4

N = numel(snap);
power_shed = zeros(1, N);
dvsi_after = zeros(1, N);
total_load = 0;

for k = 1:N
    s = shed_fracs(k);
    Load = snap(k).Load;
    Gen  = snap(k).Gen;
    V    = snap(k).V;
    Vn   = snap(k).Vn;

    total_load = total_load + Load;
    power_shed(k) = s * Load;

    D_after_kW = Load*(1 - s) - Gen;               % remaining net demand, kW
    % Kp*V*Vn*f_max evaluates in WATTS (it's the DAB power law with V in
    % volts, Lk in henries, fs in Hz - same as compute_dvsi.m's physics),
    % but D_after_kW is in kW (Load/Gen are both kW here). Must convert
    % before dividing, or this understates the deficit by 1000x and makes
    % shedding look ~1000x more effective than it really is - exactly the
    % bug that made the very first working runs recommend only 5% shed
    % when the real bisection search needed 50%.
    dvsi_raw = (D_after_kW * 1000) / (Kp * V * Vn * f_max);
    dvsi_after(k) = min(max(dvsi_raw, 0), 1);      % clip to [0,1]
end

norm_power_shed = sum(power_shed) / max(total_load, eps);
norm_remaining_stress = mean(dvsi_after);

J = 0.7*norm_power_shed + 0.3*norm_remaining_stress;

if nargout > 1
    info = struct('bus', {snap.bus}, ...
                   'shed_frac', num2cell(shed_fracs), ...
                   'power_shed_kW', num2cell(power_shed), ...
                   'DVSI_after', num2cell(dvsi_after));
end

end
