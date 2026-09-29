%% ADD_GRADUAL_RAMP_CAPABILITY.M
% =========================================================================
% Inserts a Rate Limiter into every bus's shed-fraction command path in
% Grid_modelling_Thermal_V7_ALfix_sheddable.slx, then saves the result as
% a new file. The input .slx is never modified in place.
%
% WHY THIS EXISTS:
% add_controllable_load_shedding.m gave every bus an instantaneous
% shed-fraction actuator:
%     Constant/Step (ShedFrac_<bus>_default) -> Saturation (..._limit) -> Goto
% That is fine for a policy DECISION (a controller deciding "shed now" is
% legitimately a discrete event) but wrong for the physical RESPONSE (a
% real shedding actuator cannot jump load instantly). This script splits
% those two roles apart by inserting a Rate Limiter between the command
% source and the existing Saturation block:
%     Constant/Step -> [Rate Limiter] -> Saturation -> Goto
%                            ^
%                 slews the command smoothly to its target instead of
%                 jumping, at a configurable rate
%
% Nothing downstream of the existing Saturation block is touched. The
% Saturation block stays exactly where it is as the final hard 0..1 clamp.
%
% DEFAULT BEHAVIOR (no-op by design):
% Every Rate Limiter is added with RisingSlewLimit = inf and
% FallingSlewLimit = -inf, meaning "no rate limit". Running the model
% right after this script produces IDENTICAL simulation results to the
% input model, exactly like add_controllable_load_shedding.m's own
% default-Constant=1 no-op guarantee. The capability is added, no ramp
% rate is decided here.
%
% WHAT THIS DOES NOT DO (deliberately):
% It does not pick a ramp duration. That is a policy magnitude, same
% category as shed fraction and start time, decided by Apoorv, tested via
% sweep like everything else in this project. To test a candidate ramp
% duration manually: set a bus's ShedFrac_<bus>_ramp block's
% FallingSlewLimit to -(before_val - after_val) / ramp_seconds.
%
% USAGE:
%   1. Place this script in the same folder as
%      Grid_modelling_Thermal_V7_ALfix_sheddable.slx (or edit MODEL_PATH
%      below to point at it).
%   2. Run in MATLAB:  add_gradual_ramp_capability
%   3. Output: Grid_modelling_Thermal_V7_ALfix_sheddable_ramped.slx in the
%      same folder.
%   4. Open the new file and simulate it exactly as before to confirm
%      identical results (see the sanity-check note at the end). This is
%      the same check add_controllable_load_shedding.m used and it is
%      just as essential here: run it before trusting this file.
% =========================================================================

function add_gradual_ramp_capability()

MODEL_PATH = '..\05_load_shedding_infrastructure\Grid_modelling_Thermal_V7_ALfix_sheddable.slx';
OUTPUT_NAME = 'Grid_modelling_Thermal_V7_ALfix_sheddable_ramped';

BUSES = {'A','B','C','D','E','F','G','H','K','L'};

fprintf('%s\n', repmat('=', 1, 70));
fprintf('ADD GRADUAL RAMP CAPABILITY\n');
fprintf('%s\n', repmat('=', 1, 70));

if ~isfile(MODEL_PATH)
    error(['Model not found: %s\nEdit MODEL_PATH at the top of this ' ...
           'script, or run add_controllable_load_shedding.m first if ' ...
           'the sheddable model does not exist yet.'], MODEL_PATH);
end

[~, mdl, ~] = fileparts(MODEL_PATH);
fprintf('Loading model: %s\n', MODEL_PATH);
load_system(MODEL_PATH);

n_modified = 0;
n_skipped = 0;

for i = 1:numel(BUSES)
    bus = BUSES{i};

    default_block = sprintf('%s/ShedFrac_%s_default', mdl, bus);
    limit_block   = sprintf('%s/ShedFrac_%s_limit', mdl, bus);

    if getSimulinkBlockHandle(default_block) == -1 || getSimulinkBlockHandle(limit_block) == -1
        fprintf('  SKIPPED Bus %s: expected blocks not found (%s / %s).\n', ...
                bus, default_block, limit_block);
        fprintf('           Run add_controllable_load_shedding.m first.\n');
        n_skipped = n_skipped + 1;
        continue
    end

    % -----------------------------------------------------------------
    % Idempotency check: if this bus was already processed by a prior
    % run of this script, skip it rather than double-inserting blocks.
    % -----------------------------------------------------------------
    ramp_block_name = sprintf('ShedFrac_%s_ramp', bus);
    ramp_block_path = sprintf('%s/%s', mdl, ramp_block_name);
    if getSimulinkBlockHandle(ramp_block_path) ~= -1
        fprintf('  SKIPPED Bus %s: %s already exists, script already run on this model\n', ...
                bus, ramp_block_name);
        n_skipped = n_skipped + 1;
        continue
    end

    % -----------------------------------------------------------------
    % Step 1: find the existing line from _default to _limit, and
    % delete it. We are splicing the Rate Limiter into this single line.
    % -----------------------------------------------------------------
    default_ports = get_param(default_block, 'PortHandles');
    if isempty(default_ports.Outport)
        error('Bus %s: %s has no outport, cannot locate the default-to-limit line.', ...
              bus, default_block);
    end
    existing_line = get_param(default_ports.Outport(1), 'Line');
    if existing_line == -1
        error('Bus %s: %s output is unconnected, cannot splice a rate limiter in.', ...
              bus, default_block);
    end

    fprintf('  Bus %s: splicing rate limiter between "%s" and "%s"\n', ...
            bus, sprintf('ShedFrac_%s_default', bus), sprintf('ShedFrac_%s_limit', bus));

    delete_line(existing_line);

    % -----------------------------------------------------------------
    % Step 2: add the Rate Limiter block, positioned at the midpoint
    % between the default source and the saturation limit block.
    % -----------------------------------------------------------------
    default_pos = get_param(default_block, 'Position');
    limit_pos   = get_param(limit_block, 'Position');

    mid_x = round((default_pos(3) + limit_pos(1)) / 2);
    mid_y = round((default_pos(2) + default_pos(4)) / 2);
    ramp_w = 50; ramp_h = 30;
    ramp_pos = [mid_x - ramp_w/2, mid_y - ramp_h/2, mid_x + ramp_w/2, mid_y + ramp_h/2];

    add_block('simulink/Discontinuities/Rate Limiter', ramp_block_path, ...
               'RisingSlewLimit', 'inf', 'FallingSlewLimit', '-inf', ...
               'Position', ramp_pos);

    % -----------------------------------------------------------------
    % Step 3: reconnect default -> ramp -> limit.
    % -----------------------------------------------------------------
    add_line(mdl, sprintf('ShedFrac_%s_default/1', bus), sprintf('%s/1', ramp_block_name));
    add_line(mdl, sprintf('%s/1', ramp_block_name), sprintf('ShedFrac_%s_limit/1', bus));

    n_modified = n_modified + 1;
end

fprintf('\nModified %d buses, skipped %d.\n', n_modified, n_skipped);

if n_modified > 0
    out_file = [OUTPUT_NAME, '.slx'];
    fprintf('Saving as: %s (input file left untouched)\n', out_file);
    save_system(mdl, out_file);
else
    fprintf('Nothing modified, not saving a new file.\n');
end

fprintf('\nSANITY CHECK before trusting this model:\n');
fprintf('  1. Open %s in Simulink.\n', [OUTPUT_NAME, '.slx']);
fprintf('  2. Run the existing 5000s simulation exactly as before.\n');
fprintf('  3. Confirm the output CSV is IDENTICAL to a run of the input\n');
fprintf('     sheddable model, since every Rate Limiter defaults to\n');
fprintf('     unlimited slew (+inf / -inf), i.e. no rate limiting at all.\n');
fprintf('     If anything differs, do not use this file until that is resolved.\n');
fprintf('  4. To test a candidate ramp duration later, set a bus''s\n');
fprintf('     "ShedFrac_<bus>_ramp" block''s FallingSlewLimit to\n');
fprintf('     -(before_val - after_val) / ramp_seconds (negative because\n');
fprintf('     the fraction is decreasing). Leave RisingSlewLimit at inf\n');
fprintf('     unless testing load restoration, which is not in scope yet.\n');

close_system(mdl);

end
