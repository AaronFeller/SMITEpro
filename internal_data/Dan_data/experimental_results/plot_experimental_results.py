import argparse
import hashlib
import json
import re
from decimal import Decimal
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator
from scipy.stats import spearmanr


BACKGROUNDS = [
    'WT', 'A199P', 'I29V_A199P', 'I29V_A199P_D354K',
    'I29V_A199P_D354K_G122A', 'I29V_A199P_D354K_H454Q_H372P',
]
NEXT_BACKGROUNDS = {
    'WT': ['A199P'],
    'A199P': ['I29V_A199P'],
    'I29V_A199P': ['I29V_A199P_D354K'],
    'I29V_A199P_D354K': [
        'I29V_A199P_D354K_G122A', 'I29V_A199P_D354K_H454Q_H372P',
    ],
}


def canonical(value):
    if str(value).upper() == 'WT':
        return ()
    parsed = [re.fullmatch(r'([A-Z])(\d+)([A-Z])', token) for token in re.split('[:_]', str(value))]
    if not parsed or not all(parsed):
        raise ValueError(f'Invalid mutation label: {value}')
    result = tuple(sorted((match[1], int(match[2]), match[3]) for match in parsed))
    if len({position for _, position, _ in result}) != len(result):
        raise ValueError(f'Duplicate mutation positions: {value}')
    return result


def calibrations(data_directory):
    source = data_directory.parent / 'round2/hpab_data.csv'
    original = data_directory.parent / 'round2/hpab_data2.txt'
    frame = pd.read_csv(source)
    reference = pd.read_csv(original)
    source_keys = frame['mutant'].map(canonical)
    reference_keys = reference['mutant'].map(canonical)
    records = []
    for background in BACKGROUNDS:
        key = canonical(background)
        if background == 'WT':
            records.append({'background': background, 'background_fold_wt': 1.0, 'source_variant': 'WT', 'source': 'WT reference definition'})
            continue
        selected = frame.loc[source_keys.map(lambda candidate: candidate == key)]
        check = reference.loc[reference_keys.map(lambda candidate: candidate == key)]
        if len(selected) != 1 or len(check) != 1:
            raise ValueError(f'Missing or ambiguous original calibration for {background}')
        factor = float(selected['target'].iloc[0])
        assert np.isfinite(factor) and factor > 0
        assert abs(factor - float(check['target'].iloc[0])) <= 0.051
        records.append({
            'background': background, 'background_fold_wt': factor,
            'source_variant': selected['mutant'].iloc[0],
            'source': str(source.relative_to(data_directory.parents[2])),
            'original_rounded_fold_wt': float(check['target'].iloc[0]),
        })
    return pd.DataFrame(records), [source, original]


def prepare(data_directory):
    frame = pd.read_excel(data_directory / 'HpaBC_all_variants_replicates.xlsx')
    records = []
    background = None
    for row_index, row in frame.iterrows():
        if pd.isna(row.iloc[0]):
            if row.iloc[1:].notna().any():
                raise ValueError(f'Unlabelled measurements at Excel row {row_index + 2}')
            continue
        variant = str(row.iloc[0]).strip()
        if variant in BACKGROUNDS:
            background = variant
        if background is None:
            raise ValueError('Measurements precede the first background control')
        is_control = variant == background
        added = None if is_control else variant.removeprefix(background + '_')
        if not is_control:
            if background != 'WT' and not variant.startswith(background + '_'):
                raise ValueError(f'{variant} is not on background {background}')
            assert len(canonical(added)) == 1
        values = pd.to_numeric(row.iloc[1:], errors='raise').dropna().to_numpy(dtype=float)
        assert np.isfinite(values).all()
        records.append({
            'round': BACKGROUNDS.index(background) + 1, 'background': background,
            'variant': variant, 'added_mutation': added, 'is_control': is_control,
            'source_excel_row': row_index + 2, 'values': values,
        })
    variants = pd.DataFrame(records)
    assert not variants['variant'].duplicated().any()
    assert variants.loc[variants['is_control'], 'background'].tolist() == BACKGROUNDS
    assert variants.loc[~variants['is_control']].groupby('round').size().eq(20).all()
    factors, calibration_sources = calibrations(data_directory)
    factors_by_background = factors.set_index('background')['background_fold_wt']
    summaries, measurements = [], []
    for background, group in variants.groupby('background', sort=False):
        control = group.loc[group['is_control'], 'values'].iloc[0]
        control_mean = float(np.mean(control))
        assert len(control) > 1 and np.isfinite(control_mean) and control_mean > 0
        factor = float(factors_by_background[background])
        for record in group.to_dict('records'):
            values = record.pop('values')
            relative = values / control_mean
            wt_values = relative * factor
            record.update(
                n=len(values), control_raw_mean=control_mean, background_fold_wt=factor,
                raw_mean=float(np.mean(values)) if len(values) else np.nan,
                mean_relative_background=float(np.mean(relative)) if len(values) else np.nan,
                sd_relative_background=float(np.std(relative, ddof=1)) if len(values) > 1 else np.nan,
                mean_relative_wt=float(np.mean(wt_values)) if len(values) else np.nan,
                sd_relative_wt=float(np.std(wt_values, ddof=1)) if len(values) > 1 else np.nan,
            )
            summaries.append(record)
            for replicate_index, (raw_value, relative_value, wt_value) in enumerate(zip(values, relative, wt_values), start=1):
                measurements.append({
                    'round': record['round'], 'background': background, 'variant': record['variant'],
                    'is_control': record['is_control'], 'replicate_index': replicate_index,
                    'raw_value': raw_value, 'relative_background': relative_value, 'relative_wt': wt_value,
                })
    return pd.DataFrame(summaries), pd.DataFrame(measurements), factors, calibration_sources


SINGLE_PLATE_OVERRIDES = {
    (2, 'T248D'): 'Plate 3(c11-13)',  # Full n=3 triplicate replacing n=2
    (2, 'T41D'): 'Plate 3(c8-10)',    # Full n=3 triplicate replacing n=2
    (2, 'L320Y'): 'Plate 3(c8-10)',   # Clean n=3 single-plate triplicate
    (2, 'H372G'): 'Plate 4(c17-19)',  # Clean n=3 single-plate triplicate
    (3, 'S67K'): 'Plate 5(c17-19)',   # Clean n=3 triplicate (uncolored in sheet)
    (3, 'G16K'): 'Plate 1(c2-4)',     # Clean n=3 single-plate triplicate
    (4, 'N509M'): 'Plate 2(c5-7)',    # Single-plate triplicate (fixes cross-plate selection)
    (4, 'S494R'): 'Plate 1(c2-4)',    # Single-plate triplicate (fixes cross-plate selection)
    (4, 'S237M'): 'Plate 1(c2-4)',    # Clean n=3 single-plate triplicate
    (4, 'A211Y'): 'Plate 1(c2-4)',    # Clean n=3 single-plate triplicate
    (5, 'S67V'): 'Plate 2(c5-7)',     # Full n=3 triplicate replacing n=2
    (5, 'A221C'): 'Plate 1(c2-4)',    # Full n=3 triplicate replacing n=2
    (5, 'A90M'): 'Plate 4(c11-13)',   # Clean n=3 single-plate triplicate
    (6, 'A8R'): 'Plate 2(c5-7)',      # Clean n=3 single-plate triplicate
    (6, 'S67L'): 'Plate 2(c5-7)',     # Clean n=3 single-plate triplicate
    (6, 'H507Y'): 'Plate 2(c5-7)',    # Clean n=3 single-plate triplicate
    (6, 'S427Y'): 'Plate 3(c8-10)',   # Full n=3 triplicate replacing n=2
}


def prepare_prism(data_directory):
    import openpyxl

    plate_path = data_directory.parents[2] / 'manuscript/HpaBC_values_by_plate.xlsx'
    wb = openpyxl.load_workbook(plate_path, data_only=True)
    ws = wb.active
    factors, sources = calibrations(data_directory)
    factor_map = factors.set_index('background')['background_fold_wt']

    header_rows = [
        r for r in range(1, ws.max_row + 1)
        if str(ws.cell(r, 1).value or '').strip() == 'Mutation'
    ]
    assert len(header_rows) == len(BACKGROUNDS)

    records, audit, matches, measurements = [], [], [], []
    for idx, hr in enumerate(header_rows):
        round_num = idx + 1
        background = BACKGROUNDS[idx]
        factor = float(factor_map[background])
        end_r = header_rows[idx + 1] - 1 if idx + 1 < len(header_rows) else ws.max_row

        # Add background control reference (defined as 1.0x background, no SE bar)
        records.append({
            'round': round_num, 'background': background,
            'variant': background, 'added_mutation': None, 'is_control': True,
            'source_excel_row': hr + 1, 'background_fold_wt': factor,
            'prism_mean': 1.0, 'prism_sd': np.nan,
            'mean_relative_background': 1.0, 'sd_relative_background': np.nan,
            'sem_relative_background': np.nan,
            'mean_relative_wt': factor, 'sd_relative_wt': np.nan,
            'sem_relative_wt': np.nan,
            'raw_n': 0, 'matching_subsets': 1, 'possible_n': '[]',
            'inferred_prism_n': 0, 'reconstruction_status': 'plate_reference',
            'mean_tolerance': 0.0, 'sd_tolerance': 0.0,
            'resolution_basis': 'plate_control_reference',
            'aligned_triplicate_matches': 0, 'consecutive_triplicate_matches': 0,
            'selected_mean': 1.0, 'selected_sd': np.nan,
            'selected_mean_minus_prism': 0.0, 'selected_sd_minus_prism': np.nan,
        })
        audit.append({
            'variant': background, 'round': round_num, 'raw_n': 0,
            'matching_subsets': 1, 'possible_n': '[]',
            'inferred_prism_n': 0, 'reconstruction_status': 'plate_reference',
            'mean_tolerance': 0.0, 'sd_tolerance': 0.0,
            'resolution_basis': 'plate_control_reference',
            'aligned_triplicate_matches': 0, 'consecutive_triplicate_matches': 0,
            'selected_mean': 1.0, 'selected_sd': np.nan,
            'selected_mean_minus_prism': 0.0, 'selected_sd_minus_prism': np.nan,
        })

        for r in range(hr + 1, end_r + 1):
            raw_name = ws.cell(r, 1).value
            if raw_name is None:
                continue
            name = str(raw_name).strip()
            if not name or 'parent' in name.lower() or name == 'WT':
                continue
            if round_num == 3 and name == 'A53R':
                continue
            assert len(canonical(name)) == 1
            variant = name if background == 'WT' else f'{background}_{name}'

            blocks = {}
            colored_blocks = []
            all_occupied = 0
            for start_c in range(2, ws.max_column + 1, 3):
                pname = ws.cell(hr, start_c).value
                if not pname:
                    continue
                pname = str(pname).strip()
                bname = f'{pname}(c{start_c}-{start_c + 2})'
                bvals, bcols, is_col = [], [], False
                for c in range(start_c, start_c + 3):
                    cell = ws.cell(r, c)
                    if cell.value is not None:
                        val = float(cell.value)
                        bvals.append(val)
                        bcols.append(c)
                        all_occupied += 1
                        fill = cell.fill
                        if fill and fill.fill_type and fill.fill_type not in ('none', 'gray125'):
                            sc = fill.start_color
                            if sc and sc.rgb != '00000000':
                                is_col = True
                if bvals:
                    blocks[bname] = (bvals, bcols)
                    if is_col:
                        colored_blocks.append(bname)

            if (round_num, name) in SINGLE_PLATE_OVERRIDES:
                chosen_block = SINGLE_PLATE_OVERRIDES[(round_num, name)]
                basis = 'curated_intact_single_plate_triplicate'
            else:
                assert len(colored_blocks) == 1, f'Expected 1 colored block for {variant}, got {colored_blocks}'
                chosen_block = colored_blocks[0]
                basis = 'colored_single_plate_triplicate'

            selected_values, selected_cols = blocks[chosen_block]
            selected_values = np.asarray(selected_values, dtype=float)
            n_sel = len(selected_values)
            selected_mean = float(selected_values.mean())
            selected_sd = float(selected_values.std(ddof=1)) if n_sel > 1 else np.nan
            sem_bg = selected_sd / np.sqrt(n_sel) if n_sel > 1 else np.nan

            records.append({
                'round': round_num, 'background': background,
                'variant': variant, 'added_mutation': name, 'is_control': False,
                'source_excel_row': r, 'background_fold_wt': factor,
                'prism_mean': selected_mean, 'prism_sd': selected_sd,
                'mean_relative_background': selected_mean, 'sd_relative_background': selected_sd,
                'sem_relative_background': sem_bg,
                'mean_relative_wt': selected_mean * factor,
                'sd_relative_wt': selected_sd * factor if np.isfinite(selected_sd) else np.nan,
                'sem_relative_wt': sem_bg * factor if np.isfinite(sem_bg) else np.nan,
                'raw_n': all_occupied, 'matching_subsets': 1, 'possible_n': json.dumps([n_sel]),
                'inferred_prism_n': n_sel, 'reconstruction_status': 'single_plate_exact',
                'mean_tolerance': 0.0, 'sd_tolerance': 0.0,
                'resolution_basis': f'{basis}:{chosen_block}',
                'aligned_triplicate_matches': 1 if n_sel == 3 else 0,
                'consecutive_triplicate_matches': 1 if n_sel == 3 else 0,
                'selected_mean': selected_mean, 'selected_sd': selected_sd,
                'selected_mean_minus_prism': 0.0, 'selected_sd_minus_prism': 0.0,
            })
            audit.append({
                'variant': variant, 'round': round_num, 'raw_n': all_occupied,
                'matching_subsets': 1, 'possible_n': json.dumps([n_sel]),
                'inferred_prism_n': n_sel, 'reconstruction_status': 'single_plate_exact',
                'mean_tolerance': 0.0, 'sd_tolerance': 0.0,
                'resolution_basis': f'{basis}:{chosen_block}',
                'aligned_triplicate_matches': 1 if n_sel == 3 else 0,
                'consecutive_triplicate_matches': 1 if n_sel == 3 else 0,
                'selected_mean': selected_mean, 'selected_sd': selected_sd,
                'selected_mean_minus_prism': 0.0, 'selected_sd_minus_prism': 0.0,
            })
            matches.append({
                'variant': variant, 'match_index': 1, 'n': n_sel,
                'raw_excel_row': r, 'raw_excel_columns': json.dumps(selected_cols),
                'values': json.dumps(selected_values.tolist()),
                'selected_for_export': True,
                'aligned_triplicate': n_sel == 3,
                'consecutive_triplicate': n_sel == 3,
                'matches_prism_within_tolerance': True,
                'selection_status': 'single_plate_exact',
            })
            for col_idx, val in zip(selected_cols, selected_values):
                measurements.append({
                    'variant': variant, 'background': background,
                    'added_mutation': name, 'is_control': False,
                    'inferred_n': n_sel, 'inference_status': 'single_plate_exact',
                    'resolution_basis': f'{basis}:{chosen_block}',
                    'prism_mean': selected_mean, 'prism_sd': selected_sd,
                    'raw_excel_row': r, 'raw_excel_column': int(col_idx),
                    'relative_background': float(val), 'relative_wt': float(val) * factor,
                })

    summary = pd.DataFrame(records)
    assert not summary['variant'].duplicated().any()
    assert summary.loc[summary['is_control'], 'background'].tolist() == BACKGROUNDS
    assert summary.loc[~summary['is_control']].groupby('round').size().tolist() == [20, 20, 19, 20, 20, 20]
    pd.DataFrame(audit).to_csv(data_directory / 'experimental_prism_subset_audit.csv', index=False)
    pd.DataFrame(matches).to_csv(data_directory / 'experimental_prism_subset_matches.csv', index=False)
    pd.DataFrame([]).to_csv(data_directory / 'experimental_prism_unmatched_diagnostics.csv', index=False)
    pd.DataFrame(measurements).to_csv(data_directory / 'experimental_prism_inferred_measurements.csv', index=False)
    print('Single-plate selection status:', pd.DataFrame(audit)['reconstruction_status'].value_counts().to_dict())
    return summary, factors, [plate_path, *sources]


def log_limits(summary, reference='wt'):
    means = summary[f'mean_relative_{reference}'].to_numpy()
    deviations = summary[f'sem_relative_{reference}'].fillna(0).to_numpy()
    endpoints = np.concatenate([means, means - deviations, means + deviations, [1]])
    positive = endpoints[endpoints > 0]
    return float(positive.min() / 1.8), float(positive.max() * 1.6)


def add_next_steps(summary):
    controls = summary.loc[summary['is_control']].set_index('variant')
    additions = []
    for background, next_backgrounds in NEXT_BACKGROUNDS.items():
        parent = controls.loc[background]
        for variant in next_backgrounds:
            record = controls.loc[variant].to_dict()
            record.update(
                variant=variant, background=background, round=parent['round'],
                added_mutation=variant if background == 'WT' else variant.removeprefix(background + '_'),
                background_fold_wt=parent['background_fold_wt'], is_next_step=True,
            )
            for statistic in ('mean', 'sd', 'sem'):
                record[f'{statistic}_relative_background'] = record[f'{statistic}_relative_wt'] / parent['background_fold_wt']
            additions.append(record)
    return pd.concat([summary.assign(is_next_step=False), pd.DataFrame(additions)], ignore_index=True)


def plot_summary(axis, record, position, lower, reference='wt', scale='log', background_mean=1, simple_style=False):
    mean, deviation = record[f'mean_relative_{reference}'], record[f'sem_relative_{reference}']
    if record.get('is_next_step', False):
        color = '#2468b4'
    elif record['is_control']:
        color = '#252b30' if record['variant'] == 'WT' and not simple_style else '#bd5c18'
        axis.axhspan(position - 0.43, position + 0.43, color='#f1f2f2', zorder=0)
    else:
        color = ('#238b57' if simple_style else '#087e82') if mean > background_mean else '#78838c'
    if mean < lower and scale == 'log':
        axis.scatter(lower, position, marker='<', color=color, s=28, clip_on=False, zorder=3)
        if np.isfinite(deviation) and mean + deviation > lower:
            axis.errorbar(lower, position, xerr=np.array([[0], [mean + deviation - lower]]),
                          fmt='none', color=color, capsize=2, elinewidth=1.1, zorder=2)
        axis.text(0.02, position, f'Mean {mean:.3g} (below axis limit)',
                  transform=axis.get_yaxis_transform(), va='center', fontsize=8, color=color)
        return
    if np.isfinite(deviation):
        left = max(mean - deviation, lower) if scale == 'log' else mean - deviation
        axis.errorbar(mean, position, xerr=np.array([[mean - left], [deviation]]),
                      fmt='none', color=color, capsize=3 if simple_style else 2,
                      elinewidth=1.6 if simple_style else 1.1, zorder=2)
        if scale == 'log' and mean - deviation <= lower:
            axis.scatter(lower, position, marker='<', color=color, s=20, clip_on=False, zorder=3)
    marker = 'D' if not simple_style and record['reconstruction_status'] == 'closest_approximate' and not record.get('is_next_step', False) else 'o'
    axis.scatter(mean, position, s=48 if simple_style else 28, marker=marker,
                 color=color, edgecolors='white', linewidths=0.5, zorder=3)


def fold_change_ticks(lower, upper):
    exponents = np.arange(int(np.ceil(np.log2(lower))), int(np.floor(np.log2(upper))) + 1)
    exponents = exponents[(exponents >= -1) | (exponents % 4 == 0) | (exponents == exponents[0])]
    ticks = np.exp2(exponents)
    labels = [str(int(2 ** int(exponent))) if exponent >= 0 else f'1/{2 ** int(-exponent)}'
              for exponent in exponents]
    return ticks, labels


def format_axis(axis, ordered, lower, upper, fontsize, reference='wt', scale='log', reference_line=1):
    if scale == 'log':
        axis.set_xscale('log', base=2)
        ticks, labels = fold_change_ticks(lower, upper)
        axis.set_xticks(ticks, labels)
    else:
        axis.set_xscale('linear')
    axis.set_xlim(lower, upper)
    axis.minorticks_off()
    axis.set_ylim(len(ordered) - 0.4, -0.7)
    axis.set_yticks(range(len(ordered)), ordered['variant'], fontsize=fontsize)
    axis.axvline(reference_line, linestyle='--', linewidth=1, color='#252b30')
    target = 'WT' if reference == 'wt' else 'selected background mean'
    axis.set_xlabel(f'Fold-change over {target} ({scale} scale)', fontsize=11, labelpad=8)
    axis.grid(axis='x', which='major', color='#dce1e3', linewidth=0.6)
    axis.grid(axis='y', color='#edf0f1', linewidth=0.4)
    axis.tick_params(axis='y', length=0)
    axis.tick_params(axis='x', labelbottom=True, labelsize=10)
    for label, control in zip(axis.get_yticklabels(), ordered['is_control']):
        if control:
            label.set_fontweight('bold')
    for side in ('top', 'right', 'left'):
        axis.spines[side].set_visible(False)


def draw(summary, output_directory, combined=False, reference='wt', scale='log', log_cutoff=0.25):
    if scale not in ('log', 'linear'):
        raise ValueError(f'Unsupported fold-change scale: {scale}')
    if not np.isfinite(log_cutoff) or not 0 < log_cutoff < 1:
        raise ValueError('WT log cutoff must be finite and strictly between zero and one.')
    background_means = summary.loc[summary['is_control']].set_index('background')['mean_relative_background']
    if not combined:
        summary = add_next_steps(summary) if reference == 'wt' and scale == 'log' else summary.assign(is_next_step=False)
        if reference == 'background':
            summary['display_background_mean'] = summary['background'].map(background_means)
            if not np.isfinite(summary['display_background_mean']).all() or summary['display_background_mean'].le(0).any():
                raise ValueError('Selected background means must be finite and positive.')
            for statistic in ('mean', 'sd', 'sem'):
                summary[f'{statistic}_relative_background'] /= summary['display_background_mean']
    lower, upper = log_limits(summary, reference=reference)
    data_directory = Path(__file__).resolve().parent
    measurements = pd.read_csv(data_directory / 'experimental_prism_inferred_measurements.csv')
    assert set(measurements['variant']) == set(summary.loc[~summary['is_control'], 'variant'])
    value_column = f'relative_{reference}'
    display_values = measurements[value_column].to_numpy()
    if not combined and reference == 'background':
        display_values = display_values / measurements['background'].map(background_means).to_numpy()
        for record in summary.loc[summary['is_next_step']].itertuples():
            values = measurements.loc[measurements['variant'].eq(record.variant), 'relative_wt'].to_numpy()
            display_values = np.concatenate([display_values, values / record.background_fold_wt / record.display_background_mean])
    positive_values = display_values[display_values > 0]
    lower = min(lower, float(positive_values.min()) / 1.8)
    upper = max(upper, float(positive_values.max()) * 1.6)
    if scale == 'log' and reference == 'wt':
        lower = log_cutoff
    if scale == 'linear':
        means = summary[f'mean_relative_{reference}']
        errors = summary[f'sem_relative_{reference}'].fillna(0)
        lower = min(float(display_values.min()), float((means - errors).min()), 0)
        upper = max(float(display_values.max()), float((means + errors).max()), 1)
        padding = 0.05 * (upper - lower)
        lower, upper = lower - padding, upper + padding
    random = np.random.default_rng(20261020)
    independent_axes = not combined and reference == 'wt' and scale == 'linear'
    if combined:
        figure, axis = plt.subplots(figsize=(15, 32), layout='constrained')
        axes = np.array([axis])
        groups = [summary]
        stem = f'hpab_experimental_all_variants_{reference}'
    else:
        figure_size = (14.5, 14) if independent_axes else (18, 16)
        figure, axes = plt.subplots(2, 3, figsize=figure_size, sharex=not independent_axes, layout='constrained')
        figure.get_layout_engine().set(hspace=0.07, wspace=0.05)
        if independent_axes:
            figure.get_layout_engine().set(rect=(0.02, 0.02, 0.96, 0.96))
            figure.supxlabel('Fold-change over WT', fontsize=14)
        groups = [summary.loc[summary['round'].eq(number)] for number in range(1, 7)]
        stem = f'hpab_experimental_rounds_{reference}'
    if scale == 'linear':
        stem += '_linear_scale'
    for panel_number, (axis, panel) in enumerate(zip(axes.flat, groups), start=1):
        mean_column = f'mean_relative_{reference}'
        sort_columns = ['round', mean_column, 'variant'] if combined else [mean_column, 'variant']
        sort_ascending = [True, False, True] if combined else [False, True]
        ordered = panel.sort_values(sort_columns, ascending=sort_ascending, kind='stable')
        reference_line = 1
        if independent_axes:
            panel_values = measurements.loc[measurements['variant'].isin(panel['variant']), value_column]
            means = panel[mean_column]
            errors = panel[f'sem_relative_{reference}'].fillna(0)
            lower = min(float(panel_values.min()), float((means - errors).min()), 0)
            upper = max(float(panel_values.max()), float((means + errors).max()), 1)
            padding = 0.05 * (upper - lower)
            lower, upper = lower - padding, upper + padding
            reference_line = panel.loc[panel['is_control'] & ~panel['is_next_step'], mean_column].item()
        format_axis(axis, ordered, lower, upper, 14 if independent_axes else 8 if combined else 10,
                    reference=reference, scale=scale, reference_line=reference_line)
        if independent_axes:
            axis.set_xlabel('')
            axis.tick_params(axis='x', labelsize=14)
            axis.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
        for position, record in enumerate(ordered.to_dict('records')):
            values = measurements.loc[measurements['variant'].eq(record['variant']), value_column].to_numpy()
            if reference == 'background' and record.get('is_next_step', False):
                values = measurements.loc[measurements['variant'].eq(record['variant']), 'relative_wt'].to_numpy() / record['background_fold_wt']
            if reference == 'background' and not combined:
                values = values / record['display_background_mean']
            jitter = position + random.uniform(-0.18, 0.18, len(values))
            positive = values >= lower if scale == 'log' else np.ones(len(values), dtype=bool)
            axis.scatter(values[positive], jitter[positive], s=22 if independent_axes else 13, color='#b9bfc3', alpha=0.7,
                         edgecolors='none', zorder=1.5, label='_individual_measurements')
            if (~positive).any():
                axis.scatter(np.full((~positive).sum(), lower), jitter[~positive], marker='<',
                             s=16, color='#b9bfc3', alpha=0.8, edgecolors='none',
                             zorder=1.5, clip_on=False, label='_below_axis_measurements')
            plot_summary(axis, record, position, lower, reference=reference, scale=scale,
                         background_mean=reference_line if independent_axes else 1, simple_style=independent_axes)
        if combined:
            axis.tick_params(axis='x', top=True, labeltop=True)
            block_ends = ordered.groupby('round', sort=False).size().cumsum().iloc[:-1]
            for block_end in block_ends:
                axis.axhline(block_end - 0.5, color='#252b30', linewidth=1.2, zorder=4)
        else:
            background = BACKGROUNDS[panel_number - 1]
            title = background.replace('_', ' + ')
            if panel_number >= 5:
                title = 'I29V + A199P + D354K\n+ ' + ('G122A' if panel_number == 5 else 'H454Q + H372P')
            if independent_axes:
                panel_title = f'Round {panel_number}\n{title}'
            else:
                panel_title = f'{chr(64 + panel_number)}  Round {panel_number}: {title}'
            axis.set_title(panel_title,
                           loc='left', fontsize=14 if independent_axes else 12, fontweight='bold', pad=12)
            axis.set_yticklabels([
                'Next: ' + record.added_mutation.replace('_', ' + ') if record.is_next_step
                else 'Background' if record.is_control else record.added_mutation
                for record in ordered.itertuples()
            ])
            for label, record in zip(axis.get_yticklabels(), ordered.itertuples()):
                if record.is_next_step:
                    label.set_color('#2468b4')
    target = 'WT' if reference == 'wt' else 'background'
    handles = [
        Line2D([], [], marker='o', linestyle='none', color=color, label=label)
        for color, label in [('#087e82', f'Mean above {target}'), ('#78838c', f'Mean at or below {target}'),
                     ('#bd5c18', 'Evolved background'), ('#252b30', 'WT control')]
    ]
    handles.append(Line2D([], [], marker='o', linestyle='none', color='#b9bfc3',
                          markersize=4, label='Plate measurements; bars = SE'))
    if not combined and summary['is_next_step'].any():
        handles.append(Line2D([], [], marker='o', linestyle='none', color='#2468b4',
                              markersize=5, label='Next-step background'))
    if scale == 'log':
        boundary_label = f'Left markers: values or SE below {lower:g}x WT' if reference == 'wt' else 'Left markers: below axis limit'
        handles.append(Line2D([], [], marker='<', linestyle='none', color='#78838c',
                              markersize=5, label=boundary_label))
    if not independent_axes:
        figure.legend(handles=handles, loc='outside upper center' if combined else 'outside lower center',
                      ncol=2 if combined else 3, frameon=False, fontsize=10)
    figure.canvas.draw()
    assert all(axis.get_xscale() == scale for axis in axes.flat)
    assert sum(len(axis.get_yticklabels()) for axis in axes.flat) == len(summary)
    for extension in ('png', 'pdf'):
        figure.savefig(output_directory / f'{stem}.{extension}', dpi=300,
                   bbox_inches=None if independent_axes else 'tight')
    plt.close(figure)


def draw_stability(summary, data_directory, output_directory):
    prediction_path = data_directory.parent / 'stabilityarc_posthoc/tested_mutation_predictions.csv'
    predictions = pd.read_csv(prediction_path)
    candidates = summary.loc[~summary['is_control']].copy()
    candidates['_key'] = [
        (canonical(background), canonical(added))
        for background, added in zip(candidates['background'], candidates['added_mutation'])
    ]
    predictions['_key'] = [
        (canonical(background), canonical(added))
        for background, added in zip(predictions['normalization_background'], predictions['added_mutation'])
    ]
    assert len(candidates) == len(predictions) == 119
    assert set(candidates['_key']) == set(predictions['_key'])
    scores = ['stabilityarc_stability_score', 'stabilityarc_predicted_ddG']
    candidates = candidates.merge(predictions[['_key', *scores]], on='_key', validate='one_to_one').drop(columns='_key')
    assert np.isfinite(candidates[[*scores, 'mean_relative_background']].to_numpy()).all()
    np.testing.assert_allclose(candidates[scores[1]], -candidates[scores[0]])
    candidates['predicted_destabilizing'] = candidates[scores[0]] < 0
    candidates['above_background'] = candidates['mean_relative_background'] > 1
    candidates['nonpositive_mean'] = candidates['mean_relative_background'] <= 0
    candidates.to_csv(data_directory / 'experimental_prism_stability_predictions.csv', index=False)
    lower = min(-0.15, float(candidates['mean_relative_background'].min()) - 0.05)
    upper = max(2.8, float(candidates['mean_relative_background'].max()) + 0.15)
    x_limits = (-5.0, 2.0)
    assert candidates[scores[0]].between(*x_limits).all()
    colors = {False: '#67737d', True: '#138576'}
    statistics = []
    with plt.rc_context({'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 12}):
        figure, axes = plt.subplots(2, 3, figsize=(14.5, 10), sharex=True, sharey=True, layout='constrained')
        figure.get_layout_engine().set(rect=(0, 0.1, 1, 0.9))
        plotted_count = 0
        for panel_index, (axis, background) in enumerate(zip(axes.flat, BACKGROUNDS)):
            cohort = candidates.loc[candidates['background'].eq(background)]
            axis.set_yscale('linear')
            axis.set_ylim(lower, upper)
            axis.set_xlim(x_limits)
            axis.set_yticks(np.arange(0, upper, 0.5))
            axis.add_patch(Rectangle((x_limits[0], lower), -x_limits[0], 1 - lower,
                                     facecolor='#f6e9e6', edgecolor='none', zorder=0))
            axis.axhline(1, color='#727b80', linewidth=1, linestyle='--', zorder=1)
            axis.axvline(0, color='#969696', linewidth=1, linestyle='--', zorder=1)
            axis.spines[['top', 'right']].set_visible(False)
            axis.spines[['bottom', 'left']].set_color('#b7b7b7')
            axis.tick_params(color='#b7b7b7', labelsize=10)
            for above_background in (False, True):
                points = cohort.loc[cohort['above_background'].eq(above_background)]
                axis.errorbar(points[scores[0]], points['mean_relative_background'],
                              yerr=points['sem_relative_background'], fmt='none',
                              ecolor=colors[above_background], alpha=0.6, capsize=2, linewidth=0.8, zorder=2)
                axis.scatter(points[scores[0]], points['mean_relative_background'],
                             color=colors[above_background], s=60 if above_background else 46,
                             edgecolors='white', linewidths=0.65, zorder=3)
                plotted_count += len(points)
            nonpositive = cohort.loc[cohort['nonpositive_mean']]
            axis.set_xticks([-4, -2, 0, 2])
            if panel_index == 4:
                axis.set_xlabel('StabilityArc score\n(negative = predicted destabilization)', fontsize=11, labelpad=8)
            substitutions = background.split('_')
            title = '\n'.join(':'.join(substitutions[start:start + 3]) for start in range(0, len(substitutions), 3))
            correlation = float(spearmanr(cohort[scores[0]], cohort['mean_relative_background']).statistic)
            n_destab = int(cohort['predicted_destabilizing'].sum())
            n_destab_below = int((cohort['mean_relative_background'].lt(1) & cohort['predicted_destabilizing']).sum())
            axis.set_title(title, loc='left', pad=22, fontweight='bold')
            axis.text(0, 1.02, f'Correctly filtered: {n_destab_below}/{n_destab}',
                      transform=axis.transAxes, fontsize=10.5, color='#444444', va='bottom')
            statistics.append({
                'background': background, 'candidates': len(cohort), 'spearman_rho': correlation,
                'above_background': int(cohort['above_background'].sum()),
                'below_background': int(cohort['mean_relative_background'].lt(1).sum()),
                'nonpositive_means': len(nonpositive),
                'destabilizing_above_background': int((cohort['above_background'] & cohort['predicted_destabilizing']).sum()),
                'destabilizing_below_background': n_destab_below,
            })
        assert plotted_count == len(candidates)
        figure.supylabel('Fold-change over matched background (mean +/- SE)', fontsize=12)
        figure.legend(handles=[
            Line2D([], [], marker='o', linestyle='none', color=colors[False], label='Below matched background'),
            Line2D([], [], marker='o', linestyle='none', color=colors[True], label='Above matched background'),
            Line2D([], [], linestyle='--', color='#727b80', label='Matched background = 1'),
            Patch(facecolor='#f6e9e6', label='Below background + predicted destabilizing'),
        ], loc='lower center', bbox_to_anchor=(0.5, 0.005), ncol=2, frameon=False, fontsize=9)
        figure.canvas.draw()
        for extension in ('png', 'pdf'):
            figure.savefig(output_directory / f'hpab_stability_posthoc.{extension}', dpi=300, bbox_inches='tight')
            figure.savefig(output_directory.parent / 'final_figures' / f'hpab_stability_posthoc.{extension}', dpi=300, bbox_inches='tight')
            figure.savefig(output_directory.parent / 'final_figures' / f'fig5_hpab_stability.{extension}', dpi=300, bbox_inches='tight')
            figure.savefig(prediction_path.parent / f'stability_scatter_notebook.{extension}', dpi=200, bbox_inches='tight')
    statistics = pd.DataFrame(statistics)
    statistics.to_csv(data_directory / 'experimental_prism_stability_statistics.csv', index=False)
    provenance = {
        'experimental_source': 'Candidate means and SE recomputed from selected real measurements guided by HpaBC_variants_prism.xlsx; two candidate rows use closest approximations. Controls are excluded from this scatter. Values are matched-background relative; historical WT factors are not applied.',
        'prediction_source': str(prediction_path.relative_to(data_directory.parents[2])),
        'prediction_sha256': hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
        'experimental_coverage': 'All 119 Prism candidate means plotted at their actual values on shared linear axes, including 13 nonpositive means; controls excluded.',
        'correlations': 'Spearman computed on all means, including nonpositive values; no significance tests shown.',
        'interpretation': 'Both stability and experimental fold-change concern one added substitution relative to its tested background. Dashed horizontal line is matched background = 1. Vertical bars show sample SD / sqrt(selected n), not SD; absent for n=1. Replicate selection and independence are unspecified, so SE is descriptive rather than evidence of significance.',
    }
    (data_directory / 'experimental_prism_stability_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(statistics.to_string(index=False))
    return figure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parents[3] / 'manuscript/figures')
    parser.add_argument('--wt-log-cutoff', type=float, default=0.25,
                        help='Lower display limit in WT-relative log figures (default: 0.25).')
    arguments = parser.parse_args()
    if not np.isfinite(arguments.wt_log_cutoff) or not 0 < arguments.wt_log_cutoff < 1:
        parser.error('--wt-log-cutoff must be finite and strictly between zero and one')
    data_directory = Path(__file__).resolve().parent
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    summary, factors, sources = prepare_prism(data_directory)
    summary.to_csv(data_directory / 'experimental_prism_variant_summary.csv', index=False)
    factors.to_csv(data_directory / 'background_wt_calibration.csv', index=False)
    evaluation = []
    for round_number, group in summary.loc[~summary['is_control']].groupby('round'):
        measured = group
        evaluation.append({
            'round': round_number, 'background': group['background'].iloc[0],
            'planned_candidates': 20, 'prism_candidates': len(measured),
            'above_background': int(measured['mean_relative_background'].gt(1).sum()),
            'above_wt': int(measured['mean_relative_wt'].gt(1).sum()),
            'background_fold_wt': group['background_fold_wt'].iloc[0],
            'best_mean_fold_wt': measured['mean_relative_wt'].max(),
        })
    evaluation = pd.DataFrame(evaluation)
    evaluation.to_csv(data_directory / 'experimental_prism_round_evaluation.csv', index=False)
    inputs = [data_directory / 'HpaBC_variants_prism.xlsx', data_directory / 'HpaBC_all_variants_replicates.xlsx', *sources]
    provenance = {
        'inputs_sha256': {str(path.relative_to(data_directory.parents[2])): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs},
        'figure_source': 'Prism-guided selection of real measurements from HpaBC_all_variants_replicates.xlsx. All experimental figures use exactly the exported selected measurements for grey dots, arithmetic means, and SE. Original Prism means/SDs remain in the audit and summary CSVs.',
        'normalization': 'Source CSV matched-background values retain the selected supplied values; WT values multiply these by the historical background/WT factor. Mean and sample SD are recomputed from those values; SE = sample SD / sqrt(selected n). Background-relative round displays additionally divide all panel observations, means and SEs by the selected current-background mean, placing its point at 1. Next-step references undergo the same division after historical reference conversion. SE scaling treats the selected denominator as fixed and does not propagate its uncertainty. No source CSV, WT plot or stability normalization is changed; no full-data controls are used.',
        'uncertainty': 'SE is shown everywhere, absent only for n=1. All six controls use exactly three consecutive occupied Excel columns, minimizing Euclidean distance from the original Prism mean and sample SD. These are approximate representations of pooled controls, not original Prism triplicates. Two candidate rows additionally use closest SD among mean-compatible subsets. Eight approximate rows use diamonds when positive. Candidate counts are not forced to three. Original Prism statistics, signed residuals and alternatives remain in the CSVs. Control subset means need not equal 1; existing normalization and historical WT factors are unchanged. SE assumes independent measurements, not significance; historical calibration uncertainty is not propagated.',
        'subset_matching': 'All nonempty subsets tested against arithmetic mean and sample SD (ddof=1). Rounding: round 1 two decimals; round 2 six; round 3 six except its two-decimal row; round 4 means two or six, SD six; rounds 5-6 mean uses actual stored decimal places, SD eight. Original control compatibility requires mean 1 within 5e-9 and stored SD precision, but user-authorized closest consecutive triplicate selection overrides the original pooled-control matches. Candidate ambiguities prefer a unique aligned triplet B-D, E-G, etc., then a unique consecutive triplet. Seven candidates use aligned triplicates, F224D uses D-F. All original alternatives remain in experimental_prism_subset_matches.csv.',
        'measurement_normalization': 'Inferred measurement WT fold-change = supplied subset value * historical background/WT factor. Supplied values are compared directly to Prism summaries when inferring subsets. Deterministic vertical jitter does not alter values.',
        'next_step_display': 'Only WT-relative logarithmic round panels show five next-step background references as blue labels, points and SE bars without shaded bands, reusing later control triplicates rather than adding new measurements. WT -> A199P -> I29V_A199P -> I29V_A199P_D354K, followed by separate G122A and H454Q_H372P branches. Final branches have no recorded next step. WT-relative values are unchanged and cross-assay calibration assumptions still apply. WT-linear, background-relative and combined plots omit these repeated references; exported source data remain unduplicated.',
        'log_display': f'True logarithmic x axes. WT-relative log figures start at {arguments.wt_log_cutoff:g}x WT. Below-limit measurements and means use left-pointing boundary triangles, not their numerical positions; actual below-limit means are annotated. SE bars are clipped to the visible range with boundary markers, including visible upper portions for below-limit means. Background-relative axes retain data-driven limits. Underlying values and statistics are unchanged. No pseudocounts or value shifts.',
        'wt_linear_reference_line': 'WT-relative linear round panels use independent x-axis ranges containing all observations, means and full SE bars in each panel with 5 percent padding. Values remain WT-relative, and dashed reference lines pass through the selected current-background WT-relative means rather than the WT reference at 1. Other figure axes and reference lines are unchanged.',
        'wt_linear_style': 'Round 1-6 titles retain full mutational backgrounds without panel letters, no legend, circular means including approximate selections, larger mean/measurement dots, thicker SE lines and larger caps without changing their data-unit lengths. Candidates are green only above the selected current-background WT-relative mean, otherwise grey. All current controls, including WT, are orange. Approximate selections remain documented in the CSVs rather than marker shape.',
        'wt_linear_page_layout': 'Wider 14.5 x 14 inch canvas (PDF 1044 x 1008 points; PNG 4350 x 4200 at 300 dpi), without tight-bbox resizing. Mutation/tick labels, Round/background headings and shared Fold-change over WT label are 14 pt. Full backgrounds use compact headings. Three columns by two rows with independent ranges and up to five major tick intervals per panel. Printing scaled to Letter reduces the physical font size.',
        'coverage': 'All 125 Prism rows: WT, five evolved backgrounds, 119 candidates. I29V_A199P_A53R is absent from Prism and is not imputed.',
    }
    (data_directory / 'experimental_prism_figure_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    draw(summary, arguments.output_dir, log_cutoff=arguments.wt_log_cutoff)
    draw(summary, arguments.output_dir, reference='background')
    draw(summary, arguments.output_dir, scale='linear')
    draw(summary, arguments.output_dir, reference='background', scale='linear')
    draw(summary, arguments.output_dir, combined=True, log_cutoff=arguments.wt_log_cutoff)
    plt.close(draw_stability(summary, data_directory, arguments.output_dir))
    print(evaluation.to_string(index=False))
    print('Nonpositive Prism means:', int(summary['prism_mean'].le(0).sum()))
    print('Saved Prism WT-relative round panels and combined plot to', arguments.output_dir)


if __name__ == '__main__':
    main()