function [load_hist, clip_hi, clip_lo] = build_scenario_loads(row, BUS, n_steps, ts, pmin, pmax)
% ==========================================================================
%  BUILD_SCENARIO_LOADS.M
%  --------------------------------------------------------------------
%  ONE load-profile builder for every scenario in scenario_table.csv.
%  It generalizes the two recipes already used in this project:
%    - build_gradual_busC_loads_v4      (single target bus, boundary buses starved)
%    - build_three_bus_collapse_loads   (G/H/K target arc, F/L starved,
%                                        preheat -> surge -> plateau -> step-down)
%
%  Recipe per bus k  (same sine + noise as the originals for every bus):
%      slow_var = 4000*sin(2*pi*t/1500 + 0.4*k)
%      noise    = 800*randn                       (drawn per bus, in bus order)
%    target bus   : preheat_w -> plateau_w over [ramp_start_s, ramp_end_s],
%                   optionally -> recovery_w over [stepdown_start_s, stepdown_end_s]
%    boundary bus : boundary_preheat_w -> boundary_low_w over the same ramp,
%                   optionally -> boundary_recovery_w over the step-down window
%    other bus    : 27000 W ambient + slow_var + noise
%  then clipped to [pmin, pmax].
%
%  Reproducing the known scenarios exactly (same random-draw order):
%    - Bus C v4:  row S041 (seed 104, jitter 0, no step-down)
%    - G/H/K:     row S045 (seed 201, jitter 0.03, with step-down)
%  When jitter_frac > 0, ONE rand() draw for the target buses happens before
%  the per-bus randn draws, exactly as in the three-bus script.
%
%  Inputs
%    row  struct with fields: targets (cellstr), boundary (cellstr), seed,
%         ramp_start_s, ramp_end_s, plateau_w, boundary_low_w, preheat_w,
%         boundary_preheat_w, jitter_frac, stepdown_start_s, stepdown_end_s,
%         recovery_w, boundary_recovery_w   (step-down fields NaN = none)
%  Outputs
%    load_hist  [n_steps x NB] commanded load in W
%    clip_hi / clip_lo  percent of samples clipped at pmax / pmin, per bus
% ==========================================================================
    AMBIENT_W = 27000;   % background buses, same as both original scripts

    rng(row.seed);
    t  = (0:n_steps-1)' * ts;
    NB = numel(BUS);

    is_target   = ismember(BUS, row.targets);
    is_boundary = ismember(BUS, row.boundary) & ~is_target;

    % Per-bus jitter: only draws random numbers when jitter_frac > 0, so the
    % random stream matches the Bus C v4 script (which has no jitter draw).
    jitter_by_bus = ones(1, NB);
    if row.jitter_frac > 0
        idx_t = find(is_target);
        jitter_by_bus(idx_t) = 1 + row.jitter_frac * (2*rand(1, numel(idx_t)) - 1);
    end

    frac_up = min(1, max(0, (t - row.ramp_start_s) / (row.ramp_end_s - row.ramp_start_s)));
    has_step = ~isnan(row.stepdown_start_s) && ~isnan(row.stepdown_end_s);
    if has_step
        frac_dn = min(1, max(0, (t - row.stepdown_start_s) / (row.stepdown_end_s - row.stepdown_start_s)));
    else
        frac_dn = zeros(size(t));
    end

    load_hist = nan(n_steps, NB);
    clip_hi   = nan(1, NB);
    clip_lo   = nan(1, NB);

    for k = 1:NB
        phase_shift = 0.4 * k;
        slow_var = 4000 * sin(2*pi*t/1500 + phase_shift);
        noise    = 800  * randn(n_steps, 1);

        if is_target(k)
            ji  = jitter_by_bus(k);
            pre = row.preheat_w  * ji;
            mid = row.plateau_w  * ji;
            raw = pre + (mid - pre) .* frac_up + slow_var + noise;
            if has_step
                rec = row.recovery_w * ji;
                raw = raw + (rec - mid) .* frac_dn;
            end
        elseif is_boundary(k)
            pre = row.boundary_preheat_w;
            mid = row.boundary_low_w;
            raw = pre + (mid - pre) .* frac_up + slow_var + noise;
            if has_step
                rec = row.boundary_recovery_w;
                raw = raw + (rec - mid) .* frac_dn;
            end
        else
            raw = AMBIENT_W + slow_var + noise;
        end

        clip_hi(k) = mean(raw > pmax) * 100;
        clip_lo(k) = mean(raw < pmin) * 100;
        load_hist(:, k) = max(pmin, min(pmax, raw));
    end
end
