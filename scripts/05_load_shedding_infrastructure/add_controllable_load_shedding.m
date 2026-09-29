%% ADD_CONTROLLABLE_LOAD_SHEDDING.M
% =========================================================================
% Adds a controllable load-shedding actuator on every bus's load path in
% Grid_modelling_Thermal_V7_ALfix.slx, then saves the result as a new file.
% The original .slx is never modified in place.
%
% WHAT THIS DOES, PER BUS (confirmed by inspecting the actual model file):
% Every bus's load currently flows as:
%     Repeating Sequence Stair (load profile)  --->  source_bus_model_<bus> / Pload
% This script splices a Product block into that single line:
%     Repeating Sequence Stair ---> [x] ---> source_bus_model_<bus> / Pload
%                                    ^
%                                    |
%                          ShedFraction_<bus>  (Saturate 0..1) <-- From tag
%
% ShedFraction_<bus> is read via a Goto/From tag pair, NOT wired directly,
% so the actual shedding policy (when, how much) can be swapped in later
% without touching this plumbing again. By default, every ShedFraction is
% a Constant = 1 (meaning "no shedding, pass load through unchanged"), so
% running the model right after this script produces IDENTICAL simulation
% results to the original model. The capability is added, the policy is
% not decided, per the "infrastructure now, policy later" plan.
%
% WHAT THIS DOES NOT DO (deliberately):
% It does not decide how much load to shed or when. That still needs
% Apoorv's numbers. Setting a ShedFraction_<bus> Constant block to
% something other than 1 (e.g. 0.7 to shed 30%) is how you would test a
% candidate policy manually, once one exists.
%
% USAGE:
%   1. Place this script in the same folder as Grid_modelling_Thermal_V7_ALfix.slx
%      (or edit MODEL_PATH below to point at it).
%   2. Run in MATLAB:  add_controllable_load_shedding
%   3. Output: Grid_modelling_Thermal_V7_ALfix_sheddable.slx in the same folder.
%   4. Open the new file and simulate it exactly as before to confirm
%      identical results (see the sanity-check note at the end).
% =========================================================================

function add_controllable_load_shedding()

MODEL_PATH = 'D:\naren\Documents\Thermal_Derating_Work_Handover\ThermalProject\code\Grid_modelling_Thermal_V7_ALfix.slx';
OUTPUT_NAME = 'Grid_modelling_Thermal_V7_ALfix_sheddable';

BUSES = {'A','B','C','D','E','F','G','H','K','L'};

fprintf('%s\n', repmat('=', 1, 70));
fprintf('ADD CONTROLLABLE LOAD SHEDDING\n');
fprintf('%s\n', repmat('=', 1, 70));

if ~isfile(MODEL_PATH)
    error('Model not found: %s\nEdit MODEL_PATH at the top of this script.', MODEL_PATH);
end

[~, mdl, ~] = fileparts(MODEL_PATH);
fprintf('Loading model: %s\n', MODEL_PATH);
load_system(MODEL_PATH);

% -------------------------------------------------------------------
% Layout area for the new Goto/From "shed command" plumbing, kept in
% one visually separate region so it does not overlap existing blocks.
% -------------------------------------------------------------------
COMMAND_AREA_X = -3400;
COMMAND_AREA_Y_START = 1800;
ROW_HEIGHT = 90;

n_modified = 0;
n_skipped = 0;

for i = 1:numel(BUSES)
    bus = BUSES{i};
    target_block = sprintf('%s/source_bus_model_%s', mdl, bus);

    if ~bdIsLoaded(mdl) || getSimulinkBlockHandle(target_block) == -1
        fprintf('  SKIPPED Bus %s: block not found (%s)\n', bus, target_block);
        n_skipped = n_skipped + 1;
        continue
    end

    % -----------------------------------------------------------------
    % Idempotency check: if this bus was already processed by a prior
    % run of this script, skip it rather than double-inserting blocks.
    % -----------------------------------------------------------------
    shed_block_name = sprintf('LoadShed_%s', bus);
    shed_block_path = sprintf('%s/%s', mdl, shed_block_name);
    if getSimulinkBlockHandle(shed_block_path) ~= -1
        fprintf('  SKIPPED Bus %s: %s already exists, script already run on this model\n', ...
                bus, shed_block_name);
        n_skipped = n_skipped + 1;
        continue
    end

    % -----------------------------------------------------------------
    % Step 1: find the existing line feeding this bus's Pload inport,
    % and the block currently supplying it (dynamically, not by
    % assuming a fixed name, since this must be robust to model edits).
    % -----------------------------------------------------------------
    port_handles = get_param(target_block, 'PortHandles');
    if isempty(port_handles.Inport)
        error('Bus %s: source_bus_model_%s has no inport, cannot locate Pload.', bus, bus);
    end
    pload_port_handle = port_handles.Inport(1);   % Pload is port 1, confirmed in the model

    existing_line = get_param(pload_port_handle, 'Line');
    if existing_line == -1
        error('Bus %s: Pload inport is unconnected, cannot splice a shedding block in.', bus);
    end

    src_port_handle = get_param(existing_line, 'SrcPortHandle');
    src_block = get_param(src_port_handle, 'Parent');

    src_pos = get_param(src_block, 'Position');
    dst_pos = get_param(target_block, 'Position');

    fprintf('  Bus %s: splicing between "%s" and "source_bus_model_%s"\n', ...
            bus, get_param(src_block, 'Name'), bus);

    % -----------------------------------------------------------------
    % Step 2: remove the direct line, we are inserting a block into it.
    % -----------------------------------------------------------------
    delete_line(existing_line);

    % -----------------------------------------------------------------
    % Step 3: add the Product block that will apply the shed fraction.
    % Placed at the midpoint between the load source and the bus model.
    % -----------------------------------------------------------------
    mid_x = round((src_pos(3) + dst_pos(1)) / 2);
    mid_y = round((src_pos(2) + src_pos(4)) / 2);
    shed_w = 40; shed_h = 30;
    shed_pos = [mid_x - shed_w/2, mid_y - shed_h/2, mid_x + shed_w/2, mid_y + shed_h/2];

    add_block('simulink/Math Operations/Product', shed_block_path, ...
               'Position', shed_pos, 'Inputs', '2');

    % -----------------------------------------------------------------
    % Step 4: add the Goto/From/Saturate/Constant plumbing for this
    % bus's shed fraction, defaulting to 1 (no shedding). Kept in a
    % dedicated column on the canvas, one row per bus.
    % -----------------------------------------------------------------
    row_y = COMMAND_AREA_Y_START + (i - 1) * ROW_HEIGHT;
    goto_tag = sprintf('ShedFrac_%s', bus);

    const_path = sprintf('%s/ShedFrac_%s_default', mdl, bus);
    add_block('simulink/Sources/Constant', const_path, ...
               'Value', '1', ...
               'Position', [COMMAND_AREA_X, row_y, COMMAND_AREA_X + 50, row_y + 30]);

    sat_path = sprintf('%s/ShedFrac_%s_limit', mdl, bus);
    add_block('simulink/Discontinuities/Saturation', sat_path, ...
               'UpperLimit', '1', 'LowerLimit', '0', ...
               'Position', [COMMAND_AREA_X + 90, row_y, COMMAND_AREA_X + 130, row_y + 30]);

    goto_path = sprintf('%s/Goto_%s', mdl, goto_tag);
    add_block('simulink/Signal Routing/Goto', goto_path, ...
               'GotoTag', goto_tag, ...
               'Position', [COMMAND_AREA_X + 170, row_y, COMMAND_AREA_X + 220, row_y + 30]);

    from_path = sprintf('%s/From_%s', mdl, goto_tag);
    add_block('simulink/Signal Routing/From', from_path, ...
               'GotoTag', goto_tag, ...
               'Position', [mid_x - shed_w/2 - 60, mid_y + 40, mid_x - shed_w/2 - 20, mid_y + 60]);

    % Wire: Constant -> Saturation -> Goto
    add_line(mdl, sprintf('ShedFrac_%s_default/1', bus), sprintf('ShedFrac_%s_limit/1', bus));
    add_line(mdl, sprintf('ShedFrac_%s_limit/1', bus), sprintf('Goto_%s/1', goto_tag));

    % -----------------------------------------------------------------
    % Step 5: reconnect the main load path through the Product block,
    % with the shed fraction as its second input.
    % -----------------------------------------------------------------
    add_line(mdl, sprintf('%s/1', get_param(src_block, 'Name')), sprintf('%s/1', shed_block_name));
    add_line(mdl, sprintf('From_%s/1', goto_tag), sprintf('%s/2', shed_block_name));
    add_line(mdl, sprintf('%s/1', shed_block_name), sprintf('source_bus_model_%s/1', bus));

    n_modified = n_modified + 1;
end

fprintf('\nModified %d buses, skipped %d.\n', n_modified, n_skipped);

if n_modified > 0
    out_file = [OUTPUT_NAME, '.slx'];
    fprintf('Saving as: %s (original file left untouched)\n', out_file);
    save_system(mdl, out_file);
else
    fprintf('Nothing modified, not saving a new file.\n');
end

fprintf('\nSANITY CHECK before trusting this model:\n');
fprintf('  1. Open %s in Simulink.\n', out_file);
fprintf('  2. Run the existing 5000s simulation exactly as before.\n');
fprintf('  3. Confirm the output CSV is IDENTICAL to a run of the original\n');
fprintf('     model, since every ShedFraction defaults to 1 (no shedding).\n');
fprintf('     If anything differs, do not use this file until that is resolved.\n');
fprintf('  4. To test a candidate shedding policy later, change one of the\n');
fprintf('     "ShedFrac_<bus>_default" Constant blocks to a value below 1\n');
fprintf('     (e.g. 0.7 sheds 30%% of that bus''s load) and rerun.\n');

close_system(mdl);

end
