function compare_original_vs_sheddable()
% ==========================================================================
%  COMPARE_ORIGINAL_VS_SHEDDABLE.M
%  --------------------------------------------------------------------
%  Sanity check for the load-shedding splice added by
%  add_controllable_load_shedding.m. Every ShedFraction_<bus> in the
%  sheddable model defaults to 1 (no shedding), so a correct splice
%  should produce output IDENTICAL to the original model, same seed,
%  same scale, same everything else.
%
%  This script compares the two output CSVs column by column and tells
%  you plainly: identical, floating-point-noise identical (fine), or
%  genuinely different, in which case it reports exactly which column
%  and row first diverges and by how much, so the wiring can be
%  checked against that specific signal rather than guessed at.
%
%  EDIT THE TWO PATHS BELOW to point at your actual files, then run:
%    compare_original_vs_sheddable
% ==========================================================================

    ORIGINAL_CSV = fullfile( ...
        'D:\ms-subjects\ms-subjects\Research Assistantship\Prof. Van Hai Bui\Hayla.ai\may-26-2026', ...
        'model_outputs', 'thermal_derating_v7', ...
        'thermal_derating_v7_ALfix_GEIfix_5000s.csv');

    SHEDDABLE_CSV = fullfile( ...
        'D:\ms-subjects\ms-subjects\Research Assistantship\Prof. Van Hai Bui\Hayla.ai\may-26-2026', ...
        'model_outputs', 'thermal_derating_v7', ...
        'thermal_derating_v7_ALfix_sheddable_5000s.csv');

    NOISE_TOLERANCE = 1e-9;   % differences below this are treated as floating-point noise

    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('COMPARE ORIGINAL vs SHEDDABLE MODEL OUTPUT\n');
    fprintf('%s\n', repmat('=', 1, 70));

    if ~isfile(ORIGINAL_CSV)
        error('Original CSV not found:\n%s\nEdit ORIGINAL_CSV at the top of this script.', ORIGINAL_CSV);
    end
    if ~isfile(SHEDDABLE_CSV)
        error(['Sheddable CSV not found:\n%s\n' ...
               'Run grid_derating_dataset_5000s_v7_2_sheddable first, then edit\n' ...
               'SHEDDABLE_CSV at the top of this script if the filename differs.'], SHEDDABLE_CSV);
    end

    fprintf('\nOriginal:   %s\n', ORIGINAL_CSV);
    fprintf('Sheddable:  %s\n', SHEDDABLE_CSV);

    fprintf('\nLoading both CSVs (this can take a moment for a full 500k-row file) ...\n');
    orig = readtable(ORIGINAL_CSV);
    shed = readtable(SHEDDABLE_CSV);

    % ----------------------------------------------------------------------
    % 1. Shape check
    % ----------------------------------------------------------------------
    fprintf('\n--- Shape ---\n');
    fprintf('  Original:  %d rows, %d columns\n', height(orig), width(orig));
    fprintf('  Sheddable: %d rows, %d columns\n', height(shed), width(shed));

    if height(orig) ~= height(shed)
        warning(['Row counts differ (%d vs %d). Comparing only the first %d rows.\n' ...
                 'A row-count mismatch on its own is suspicious, since both runs use\n' ...
                 'the same TEND/TS/WRITE_STRIDE, worth checking the simulation logs\n' ...
                 'for an early stop or solver issue before trusting anything else here.'], ...
                height(orig), height(shed), min(height(orig), height(shed)));
    end
    n_rows = min(height(orig), height(shed));

    % ----------------------------------------------------------------------
    % 2. Column check
    % ----------------------------------------------------------------------
    orig_cols = orig.Properties.VariableNames;
    shed_cols = shed.Properties.VariableNames;

    missing_in_shed = setdiff(orig_cols, shed_cols);
    missing_in_orig = setdiff(shed_cols, orig_cols);
    if ~isempty(missing_in_shed)
        fprintf('\n  Columns in ORIGINAL but missing from SHEDDABLE: %s\n', strjoin(missing_in_shed, ', '));
    end
    if ~isempty(missing_in_orig)
        fprintf('  Columns in SHEDDABLE but missing from ORIGINAL: %s\n', strjoin(missing_in_orig, ', '));
    end
    common_cols = intersect(orig_cols, shed_cols, 'stable');
    fprintf('  Comparing %d shared columns.\n', numel(common_cols));

    % ----------------------------------------------------------------------
    % 3. Value-by-value comparison, numeric columns only
    % ----------------------------------------------------------------------
    fprintf('\n--- Value comparison ---\n');

    worst_col = '';
    worst_row = -1;
    worst_diff = -Inf;
    any_numeric_compared = false;
    per_col_max_diff = containers.Map('KeyType', 'char', 'ValueType', 'double');

    for c = 1:numel(common_cols)
        col_name = common_cols{c};
        v1 = orig.(col_name)(1:n_rows);
        v2 = shed.(col_name)(1:n_rows);

        if ~isnumeric(v1) || ~isnumeric(v2)
            continue;   % skip non-numeric columns (shouldn't be any here, but be safe)
        end
        any_numeric_compared = true;

        d = abs(double(v1) - double(v2));
        % NaNs should appear in the same places in both files if the taps
        % behaved identically; treat NaN-vs-NaN at the same row as a match.
        both_nan = isnan(v1) & isnan(v2);
        d(both_nan) = 0;

        [col_max, row_idx] = max(d);
        per_col_max_diff(col_name) = col_max;

        if col_max > worst_diff
            worst_diff = col_max;
            worst_col = col_name;
            worst_row = row_idx;
        end
    end

    if ~any_numeric_compared
        error('No numeric columns were compared, check that both CSVs loaded correctly.');
    end

    % ----------------------------------------------------------------------
    % 4. Verdict
    % ----------------------------------------------------------------------
    fprintf('\n%s\n', repmat('-', 1, 70));
    if worst_diff == 0
        fprintf('RESULT: IDENTICAL. Every shared numeric column matches exactly.\n');
        fprintf('The load-shedding splice is verified safe to build on.\n');
    elseif worst_diff < NOISE_TOLERANCE
        fprintf('RESULT: Effectively identical (max difference %.3g, below the %.3g\n', ...
                worst_diff, NOISE_TOLERANCE);
        fprintf('floating-point noise tolerance). Worst column: %s.\n', worst_col);
        fprintf('The load-shedding splice is verified safe to build on.\n');
    else
        fprintf('RESULT: DIFFERS. Max difference %.6g in column "%s", row %d.\n', ...
                worst_diff, worst_col, worst_row);
        fprintf('DO NOT use the sheddable model for anything else until this is explained.\n');

        fprintf('\n--- Top 10 columns by max difference ---\n');
        keys_list = per_col_max_diff.keys;
        vals_list = cell2mat(per_col_max_diff.values);
        [sorted_vals, order] = sort(vals_list, 'descend');
        sorted_keys = keys_list(order);
        n_show = min(10, numel(sorted_keys));
        for i = 1:n_show
            fprintf('  %-30s max diff = %.6g\n', sorted_keys{i}, sorted_vals(i));
        end

        % Show the actual values around the worst mismatch for direct inspection
        fprintf('\n--- Context around the worst mismatch (%s, rows %d-%d) ---\n', ...
                worst_col, max(1, worst_row-3), min(n_rows, worst_row+3));
        lo = max(1, worst_row - 3);
        hi = min(n_rows, worst_row + 3);
        fprintf('%-8s %-18s %-18s %-18s\n', 'row', 'time', 'original', 'sheddable');
        for r = lo:hi
            marker = '';
            if r == worst_row; marker = '  <-- worst'; end
            fprintf('%-8d %-18.4f %-18.6g %-18.6g%s\n', ...
                r, orig.time(r), orig.(worst_col)(r), shed.(worst_col)(r), marker);
        end
    end
    fprintf('%s\n', repmat('-', 1, 70));

end
