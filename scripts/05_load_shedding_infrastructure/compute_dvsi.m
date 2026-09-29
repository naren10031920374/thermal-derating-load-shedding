%% COMPUTE_DVSI.M
% =========================================================================
% Computes the DC Voltage Stability Index (DVSI) per converter link and per
% bus from a scenario's logged simulation table (the same table structure
% grid_derating_dataset_5000s_v7_2.m writes: columns V_Bus_<X>,
% Phase_<CONV>_cmd_deg for every bus/converter).
%
% BACKGROUND (see project doc stage-06-dc-fvsi-derivation.md for the full
% derivation): every converter in this ring is a phase-shift DAB with fixed
% parameters (Lk=340uH, fs=10kHz, n=24/15), saturated to +/-90 degrees.
% Because delivered power and max deliverable power scale identically with
% V_i*V_j, the stress ratio collapses to a pure function of the phase
% command:
%
%     DVSI_ij = f(phi_ij) / f(pi/2),   f(phi) = phi*(1 - |phi|/pi)
%
% DVSI = 1 means that link is at its physical ceiling (+/-90 deg) and
% cannot deliver more power no matter what the controller does.
%
% VALIDATED FINDING (see the doc, Section 7): DVSI behaves close to
% bang-bang in this system rather than rising smoothly, so this script
% implements the GATE form recommended after validation: a link only
% counts as "stressed" once it has been pinned near +/-90 deg for a
% sustained dwell time, reusing the project's own collapse-debounce
% convention (0.5s = 50 samples at 10ms sampling) rather than inventing a
% new threshold. This matched Bus K and Bus C's known collapse events with
% 10-94s of lead time, and Bus F's shorter (~0.1-0.3s) genuine runaway.
%
% WHAT THIS DOES NOT DO (deliberately, pending a design decision):
% It does not yet compute a magnitude/ranking score for how far PAST the
% ceiling a bus's actual demand is once gated (useful for prioritizing
% among several simultaneously-stressed buses for the GA-PSO stage). That
% needs a decision on how to combine a bus's two incident DAB feeds with
% its own local source power into one "unmet demand" number, which is an
% open modeling question — see the doc's Section 7/8 before adding it.
%
% USAGE:
%   T = readtable('thermal_derating_three_bus_collapse_v1_5000s.csv');
%   [T, bus_flags] = compute_dvsi(T);
%   % T now has DVSI_<CONV> and DVSI_<CONV>_gated columns per converter,
%   % and DVSI_<BUS>_gated per bus.
%   % bus_flags is a table of just the bus-level gate columns, for a quick look.
% =========================================================================

function [T, bus_flags] = compute_dvsi(T)

% ---- Fixed DAB parameters, confirmed directly from the .slx model ----
Lk = 340e-6;        % leakage inductance, H
fs = 10e3;          % switching frequency, Hz
n  = 24/15;         % transformer turns ratio
PHI_MAX = pi/2;     % phase-shift saturation limit, rad (matches model)
f_max = PHI_MAX * (1 - abs(PHI_MAX)/pi);   % = pi/4

% ---- Ring topology: bus -> its two incident converters ----
BUSES = {'A','B','C','D','E','F','G','H','K','L'};
CONVERTERS = {'AB','BC','CD','DE','EF','FG','GH','HK','KL','AL'};
BUS_LINES = containers.Map( ...
    BUSES, ...
    {{'AL','AB'}, {'AB','BC'}, {'BC','CD'}, {'CD','DE'}, {'DE','EF'}, ...
     {'EF','FG'}, {'FG','GH'}, {'GH','HK'}, {'HK','KL'}, {'KL','AL'}});

% ---- Gate: sustained pinning, matching the project's own collapse
%      debounce convention (0.5s at 10ms sampling = 50 samples) ----
GATE_DEG_THRESHOLD = 88;     % "near" +/-90, leaves a couple degrees margin
GATE_MIN_SAMPLES   = 50;     % 0.5s at 10ms

fprintf('%s\n', repmat('=', 1, 70));
fprintf('COMPUTE DVSI\n');
fprintf('%s\n', repmat('=', 1, 70));

n_conv_found = 0;
for i = 1:numel(CONVERTERS)
    conv = CONVERTERS{i};
    col = sprintf('Phase_%s_cmd_deg', conv);
    if ~ismember(col, T.Properties.VariableNames)
        fprintf('  SKIPPED converter %s: column %s not in table\n', conv, col);
        continue
    end
    n_conv_found = n_conv_found + 1;

    phi_deg = T.(col);
    phi_rad = deg2rad(phi_deg);
    f_phi = phi_rad .* (1 - abs(phi_rad)/pi);
    dvsi = abs(f_phi) / f_max;

    pinned = abs(phi_deg) >= GATE_DEG_THRESHOLD;
    gated = sustained_run(pinned, GATE_MIN_SAMPLES);

    T.(sprintf('DVSI_%s', conv)) = dvsi;
    T.(sprintf('DVSI_%s_gated', conv)) = gated;

    fprintf('  %s: max DVSI=%.3f, %.1f%% of samples gated\n', ...
            conv, max(dvsi), 100*mean(gated));
end
fprintf('Found %d of %d converters in this table.\n', n_conv_found, numel(CONVERTERS));

% ---- Bus-level: gated if EITHER incident line is gated ----
fprintf('\nBus-level gate (either incident line pinned >=%.0fs):\n', GATE_MIN_SAMPLES*0.01);
bus_flags = table();
bus_flags.time = T.time;
for i = 1:numel(BUSES)
    bus = BUSES{i};
    lines = BUS_LINES(bus);
    col_a = sprintf('DVSI_%s_gated', lines{1});
    col_b = sprintf('DVSI_%s_gated', lines{2});
    if ~ismember(col_a, T.Properties.VariableNames) || ~ismember(col_b, T.Properties.VariableNames)
        continue
    end
    bus_gate = T.(col_a) | T.(col_b);
    T.(sprintf('DVSI_%s_gated', bus)) = bus_gate;
    bus_flags.(sprintf('DVSI_%s_gated', bus)) = bus_gate;
    if any(bus_gate)
        first_idx = find(bus_gate, 1, 'first');
        fprintf('  Bus %s: first gated at t=%.2fs (%.1f%% of samples)\n', ...
                bus, T.time(first_idx), 100*mean(bus_gate));
    else
        fprintf('  Bus %s: never gated in this table\n', bus);
    end
end

end

% -------------------------------------------------------------------
% True only where `flag` has been continuously true for at least
% `min_samples` samples up to and including that sample (trailing
% window, via movsum — base MATLAB, no toolbox required).
% -------------------------------------------------------------------
function out = sustained_run(flag, min_samples)
    flag = double(flag(:));
    window_sum = movsum(flag, [min_samples - 1, 0]);
    out = window_sum >= min_samples;
end
