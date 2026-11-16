import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent
MUTATION = re.compile(r'([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])')


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def mutate(sequence, substitutions):
    residues = list(sequence)
    positions = set()
    for substitution in substitutions:
        match = MUTATION.fullmatch(substitution)
        if match is None:
            raise ValueError(f'Invalid substitution: {substitution}')
        wildtype, position, mutant = match.groups()
        position = int(position) - 1
        if position in positions or not 0 <= position < len(residues):
            raise ValueError(f'Duplicate or out-of-range position: {substitution}')
        if residues[position] != wildtype or mutant == wildtype:
            raise ValueError(f'Reference residue mismatch: {substitution}')
        positions.add(position)
        residues[position] = mutant
    return ''.join(residues)


def prepare_inputs(input_path, fasta_path):
    frame = pd.read_csv(input_path)
    if frame[['normalization_background', 'variant']].duplicated().any():
        raise ValueError('Duplicate experimental variant within a background')
    if not pd.api.types.is_bool_dtype(frame['is_background_control']):
        raise ValueError('Control indicator must contain booleans')
    sequences = [line.strip() for line in fasta_path.read_text().splitlines() if line.strip()]
    if sum(line.startswith('>') for line in sequences) != 1:
        raise ValueError('Expected a single reference FASTA record')
    reference = ''.join(line for line in sequences if not line.startswith('>'))
    if len(reference) != 520:
        raise ValueError('Unexpected HpaB reference length')
    backgrounds = {}
    for background, cohort in frame.groupby('normalization_background', sort=False):
        sequence = mutate(reference, [] if background == 'WT' else background.split('_'))
        controls = cohort.loc[cohort['is_background_control']]
        if len(controls) != 1 or controls.iloc[0]['variant'] != background:
            raise ValueError(f'Expected one matching control for {background}')
        control_mean = float(controls.iloc[0]['normalized_fluorescence_fold_change'])
        if not np.isfinite(control_mean) or abs(control_mean - 1) > 0.005:
            raise ValueError(f'Invalid normalized control for {background}')
        for record in cohort.loc[~cohort['is_background_control']].to_dict('records'):
            added = record['added_mutation']
            if MUTATION.fullmatch(added) is None:
                raise ValueError(f'Expected one added mutation: {record["variant"]}')
            candidate = mutate(sequence, [added])
            if candidate != mutate(reference, record['variant'].split('_')):
                raise ValueError(f'Background/candidate identity mismatch: {record["variant"]}')
        backgrounds[background] = {'sequence': sequence, 'control_mean': control_mean}
    if len(backgrounds) != 6 or len(frame.loc[~frame['is_background_control']]) != 119:
        raise ValueError('Expected six backgrounds and 119 tested candidates')
    if not np.isfinite(frame['normalized_fluorescence_fold_change']).all():
        raise ValueError('Nonfinite experimental measurement')
    return frame, backgrounds


def association(cohort):
    score = cohort['stabilityarc_stability_score']
    activity = cohort['experimental_relative_to_control']
    if len(cohort) < 3 or score.nunique() < 2 or activity.nunique() < 2:
        return np.nan, np.nan
    result = spearmanr(score, activity)
    return float(result.statistic), float(result.pvalue)


def summarize(candidates, output_dir, seed, permutations):
    records = []
    for background, cohort in [('ALL (descriptive pooled)', candidates), *candidates.groupby('normalization_background', sort=False)]:
        correlation, probability = association(cohort)
        below = cohort['experimental_outcome'].eq('below_background')
        above = cohort['experimental_outcome'].eq('above_background')
        negative = cohort['predicted_destabilizing']
        table = [[int((below & negative).sum()), int((below & ~negative).sum())],
                 [int((above & negative).sum()), int((above & ~negative).sum())]]
        odds, fisher_probability = fisher_exact(table) if below.any() and above.any() else (np.nan, np.nan)
        records.append({
            'background': background, 'candidates': len(cohort),
            'below_background': int(below.sum()), 'above_background': int(above.sum()),
            'near_background': int(cohort['experimental_outcome'].eq('near_background').sum()),
            'predicted_destabilizing': int(negative.sum()),
            'below_and_destabilizing': int((below & negative).sum()),
            'above_and_destabilizing': int((above & negative).sum()),
            'fraction_destabilizing_below': float(negative[below].mean()) if below.any() else np.nan,
            'fraction_destabilizing_above': float(negative[above].mean()) if above.any() else np.nan,
            'spearman_stability_vs_experiment': correlation, 'spearman_p_two_sided': probability,
            'fisher_odds_ratio_below_vs_above': float(odds), 'fisher_p_two_sided': float(fisher_probability),
        })
    summary = pd.DataFrame(records)
    summary.to_csv(output_dir / 'background_summary.csv', index=False)
    rank_frame = candidates.groupby('normalization_background')[
        ['stabilityarc_stability_score', 'experimental_relative_to_control']
    ].rank(method='average')
    rank_frame -= rank_frame.groupby(candidates['normalization_background']).transform('mean')
    stability_ranks = rank_frame['stabilityarc_stability_score'].to_numpy()
    activity_ranks = rank_frame['experimental_relative_to_control'].to_numpy()
    observed = float(pearsonr(stability_ranks, activity_ranks).statistic)
    rng = np.random.default_rng(seed)
    indices = list(candidates.groupby('normalization_background', sort=False).indices.values())
    extreme = 0
    for draw in range(permutations):
        shuffled = activity_ranks.copy()
        for positions in indices:
            shuffled[positions] = rng.permutation(activity_ranks[positions])
        statistic = float(np.dot(stability_ranks, shuffled) / np.sqrt(
            np.dot(stability_ranks, stability_ranks) * np.dot(shuffled, shuffled)
        ))
        extreme += abs(statistic) >= abs(observed)
    statistics = {
        'within_background_centered_rank_correlation': observed,
        'within_background_permutation_p_two_sided': (extreme + 1) / (permutations + 1),
        'permutations': permutations, 'seed': seed,
        'interpretation': 'Exploratory association, not causation; no multiple-testing correction for per-background descriptive tests.',
    }
    (output_dir / 'association_statistics.json').write_text(json.dumps(statistics, indent=2) + '\n')
    return summary, statistics


def plot_results(candidates, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 2, figsize=(13, 12), layout='constrained')
    colors = {'below_background': '#b33c3c', 'above_background': '#227c69', 'near_background': '#777777'}
    for axis, (background, cohort) in zip(axes.flat, candidates.groupby('normalization_background', sort=False)):
        for outcome, color in colors.items():
            group = cohort.loc[cohort['experimental_outcome'].eq(outcome)]
            axis.scatter(group['stabilityarc_stability_score'], group['experimental_relative_to_control'],
                         color=color, label=outcome.replace('_', ' '), s=40, alpha=0.85)
        axis.axvline(0, color='#777777', linestyle=':', linewidth=1)
        axis.axhline(1, color='#777777', linestyle='--', linewidth=1)
        correlation, probability = association(cohort)
        axis.set_title(f'{background.replace("_", ":")}\nn = {len(cohort)}, Spearman rho = {correlation:.2f}', fontsize=10)
        axis.set_xlabel('StabilityArc score (higher = more stable)')
        axis.set_ylabel('Experimental fold change / matched control')
    axes.flat[0].legend(fontsize=8, frameon=False)
    figure.suptitle('Post-hoc HpaB stability predictions versus experimental outcome', fontsize=14)
    figure.savefig(output_dir / 'stability_vs_experiment.png', dpi=180)
    figure.savefig(output_dir / 'stability_vs_experiment.pdf')
    plt.close(figure)
    plot_variant_overview(candidates, output_dir)


def plot_variant_overview(candidates, output_dir):
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    if candidates[['normalization_background', 'added_mutation']].duplicated().any():
        raise ValueError('Duplicate background/mutation in visualization input')
    cohorts = list(candidates.groupby('normalization_background', sort=False))
    assert len(cohorts) == 6
    cmap = LinearSegmentedColormap.from_list('effect_direction', ['#b64049', '#fffdfb', '#216b8b'])
    activity_norm = TwoSlopeNorm(
        vmin=min(0.0, candidates['experimental_relative_to_control'].min()),
        vcenter=1.0, vmax=max(2.0, candidates['experimental_relative_to_control'].max()),
    )
    stability_norm = TwoSlopeNorm(
        vmin=min(-1.0, candidates['stabilityarc_stability_score'].min()),
        vcenter=0.0, vmax=max(1.0, candidates['stabilityarc_stability_score'].max()),
    )
    figure = plt.figure(figsize=(16, 16), layout='constrained')
    grid = figure.add_gridspec(4, 3, height_ratios=[1, 1, 0.09, 0.32], hspace=0.08)
    plotted_count = 0
    for panel_index, (background, cohort) in enumerate(cohorts):
        axis = figure.add_subplot(grid[panel_index // 3, panel_index % 3])
        ordered = cohort.sort_values('experimental_relative_to_control', ascending=False, kind='stable')
        assert ordered['experimental_relative_to_control'].is_monotonic_decreasing
        activity = ordered['experimental_relative_to_control'].to_numpy()
        stability = ordered['stabilityarc_stability_score'].to_numpy()
        assert np.isfinite(activity).all() and np.isfinite(stability).all()
        rgba = np.stack([cmap(activity_norm(activity)), cmap(stability_norm(stability))], axis=1)
        axis.imshow(rgba, aspect='auto', interpolation='nearest')
        axis.set_xticks([0, 1], ['Experiment\nfold / control', 'StabilityArc\nscore'])
        axis.xaxis.tick_top()
        axis.tick_params(axis='x', length=0, labelsize=10, pad=8)
        axis.set_yticks(np.arange(len(ordered)), ordered['added_mutation'], fontsize=9)
        axis.tick_params(axis='y', length=0, pad=6)
        axis.set_xticks([-0.5, 0.5, 1.5], minor=True)
        axis.set_yticks(np.arange(len(ordered) + 1) - 0.5, minor=True)
        axis.grid(which='minor', color='white', linewidth=0.8)
        axis.tick_params(which='minor', bottom=False, left=False)
        for row_index, (activity_value, stability_value) in enumerate(zip(activity, stability)):
            for column_index, (value, norm) in enumerate([(activity_value, activity_norm), (stability_value, stability_norm)]):
                color = cmap(norm(value))
                luminance = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]
                axis.text(column_index, row_index, f'{value:.2f}', ha='center', va='center',
                          fontsize=9, color='white' if luminance < 0.48 else '#242424')
        above_count = int(ordered['experimental_outcome'].eq('above_background').sum())
        if 0 < above_count < len(ordered):
            axis.axhline(above_count - 0.5, color='#333333', linewidth=1.3)
        substitutions = background.split('_')
        background_label = '\n'.join(':'.join(substitutions[start:start + 3]) for start in range(0, len(substitutions), 3))
        correlation, probability = association(cohort)
        axis.set_title(f'{background_label}\nn = {len(cohort)}; Spearman rho = {correlation:.2f}', fontsize=11, pad=46)
        for spine in axis.spines.values():
            spine.set_visible(False)
        plotted_count += len(ordered)
    assert plotted_count == len(candidates)
    for column_index, (norm, label) in enumerate([
        (activity_norm, 'Experimental fold change / matched control (1 = control)'),
        (stability_norm, 'Predicted stability score (0 = neutral; not kcal/mol)'),
    ]):
        color_axis = figure.add_subplot(grid[2, column_index])
        colorbar = figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=color_axis, orientation='horizontal')
        colorbar.set_label(label, fontsize=9)
        colorbar.ax.tick_params(labelsize=9)
    note_axis = figure.add_subplot(grid[2, 2])
    note_axis.axis('off')
    note_axis.text(0, 0.5, 'Red: below control / destabilizing\nBlue: above control / stabilizing', fontsize=10, va='center')
    summary_axis = figure.add_subplot(grid[3, :])
    for row_index, outcome in enumerate(['below_background', 'above_background']):
        group = candidates.loc[candidates['experimental_outcome'].eq(outcome)]
        total = len(group)
        if total == 0:
            continue
        destabilizing = int(group['predicted_destabilizing'].sum())
        nonnegative = total - destabilizing
        negative_width = 100 * destabilizing / total
        summary_axis.barh(row_index, negative_width, color='#b64049', height=0.48,
                          label='Predicted destabilizing' if row_index == 0 else None)
        summary_axis.barh(row_index, 100 - negative_width, left=negative_width, color='#216b8b', height=0.48,
                          label='Predicted non-destabilizing' if row_index == 0 else None)
        if destabilizing:
            summary_axis.text(negative_width / 2, row_index, f'{destabilizing}/{total}', color='white',
                              ha='center', va='center', fontsize=12)
        if nonnegative:
            summary_axis.text((negative_width + 100) / 2, row_index, f'{nonnegative}/{total}', color='white',
                              ha='center', va='center', fontsize=12)
    summary_axis.set_yticks([0, 1], ['Below matched control', 'Above matched control'], fontsize=11)
    summary_axis.invert_yaxis()
    summary_axis.set_xlim(0, 100)
    summary_axis.set_xlabel('Fraction of tested variants (%)')
    summary_axis.set_title('Predicted destabilization is more frequent among below-control variants', fontsize=12, pad=12)
    summary_axis.legend(loc='upper center', bbox_to_anchor=(0.5, -0.32), ncol=2, frameon=False)
    summary_axis.spines[['top', 'right', 'left']].set_visible(False)
    summary_axis.tick_params(axis='y', length=0)
    figure.suptitle('Experimental outcome and predicted stability of the same HpaB mutations\n'
                   'Each substitution is scored on its tested background; rows sorted by experimental outcome', fontsize=15)
    figure.savefig(output_dir / 'stability_mutant_overview.png', dpi=180)
    figure.savefig(output_dir / 'stability_mutant_overview.pdf')
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=DATA / 'HpaB_aaron_variants_digitized.csv')
    parser.add_argument('--fasta', type=Path, default=DATA / 'round1/HpaBC Data/Fastas/HpaB_WT.fasta')
    parser.add_argument('--stabilityarc-repo', type=Path, default=ROOT / 'StabilityArc')
    parser.add_argument('--output-dir', type=Path, default=DATA / 'stabilityarc_posthoc')
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--seed', type=int, default=20261010)
    parser.add_argument('--permutations', type=int, default=10000)
    parser.add_argument('--outcome-tolerance', type=float, default=0.005)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    if args.permutations < 1 or args.outcome_tolerance < 0:
        raise ValueError('Invalid permutation count or outcome tolerance')
    if args.plot_only:
        candidates = pd.read_csv(args.output_dir / 'tested_mutation_predictions.csv')
        plot_results(candidates, args.output_dir)
        print(f'Rendered plots from {len(candidates)} saved predictions; no inference rerun.')
        return
    frame, backgrounds = prepare_inputs(args.input, args.fasta)
    print(f'Validated {len(backgrounds)} backgrounds and 119 single-added-mutation candidates.', flush=True)
    if args.validate_only:
        return
    import torch
    sys.path.insert(0, str(args.stabilityarc_repo.resolve()))
    from stabilityarc.inference import StabilityArcPredictor

    torch.manual_seed(args.seed)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no silent CPU fallback')
    checkpoint = args.stabilityarc_repo / 'checkpoints/stabilityarc.pt'
    predictor = StabilityArcPredictor(checkpoint, device=args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    landscapes_dir = args.output_dir / 'landscapes'
    landscapes_dir.mkdir(exist_ok=True)
    prediction_records = []
    background_records = []
    for background, information in backgrounds.items():
        sequence = information['sequence']
        print(f'Scoring background {background} ({len(sequence)} residues)...', flush=True)
        landscape = pd.DataFrame(predictor.predict(sequence))
        if len(landscape) != 19 * len(sequence) or landscape['mutation'].duplicated().any():
            raise ValueError(f'Incomplete or duplicate landscape for {background}')
        if not np.isfinite(landscape[['stabilityarc_stability_score', 'stabilityarc_predicted_ddG']]).all().all():
            raise ValueError(f'Nonfinite predictions for {background}')
        np.testing.assert_allclose(landscape['stabilityarc_predicted_ddG'], -landscape['stabilityarc_stability_score'])
        landscape.to_csv(landscapes_dir / f'{background}.csv', index=False)
        lookup = landscape.set_index('mutation')
        cohort = frame.loc[frame['normalization_background'].eq(background) & ~frame['is_background_control']]
        for added in cohort['added_mutation']:
            prediction_records.append({'normalization_background': background, 'added_mutation': added,
                                       **lookup.loc[added].to_dict()})
        background_records.append({'background': background, 'sequence': sequence,
                                   'sequence_sha256': hashlib.sha256(sequence.encode()).hexdigest(),
                                   'experimental_control_mean': information['control_mean'],
                                   'tested_candidates': len(cohort), 'landscape_rows': len(landscape)})
    pd.DataFrame(background_records).to_csv(args.output_dir / 'background_sequences.csv', index=False)
    candidates = frame.loc[~frame['is_background_control']].merge(
        pd.DataFrame(prediction_records), on=['normalization_background', 'added_mutation'], how='left', validate='one_to_one',
    )
    if len(candidates) != 119 or candidates['stabilityarc_stability_score'].isna().any():
        raise ValueError('Missing predictions or incorrect candidate join')
    candidates['experimental_control_mean'] = candidates['normalization_background'].map(
        {background: information['control_mean'] for background, information in backgrounds.items()}
    )
    candidates['experimental_relative_to_control'] = candidates['normalized_fluorescence_fold_change'] / candidates['experimental_control_mean']
    change = candidates['experimental_relative_to_control'] - 1
    candidates['experimental_outcome'] = np.select(
        [change < -args.outcome_tolerance, change > args.outcome_tolerance],
        ['below_background', 'above_background'], default='near_background',
    )
    candidates['predicted_destabilizing'] = candidates['stabilityarc_stability_score'] < 0
    candidates.to_csv(args.output_dir / 'tested_mutation_predictions.csv', index=False)
    controls = frame.loc[frame['is_background_control']].copy()
    controls['stabilityarc_stability_score'] = 0.0
    controls['stabilityarc_predicted_ddG'] = 0.0
    controls['prediction_status'] = 'reference self-comparison; zero by definition, not model inference'
    controls.to_csv(args.output_dir / 'background_controls.csv', index=False)
    summary, statistics = summarize(candidates, args.output_dir, args.seed, args.permutations)
    plot_results(candidates, args.output_dir)
    provenance = {
        'analysis': 'post-hoc StabilityArc association with background-relative experimental outcome',
        'input': str(args.input.relative_to(ROOT)), 'input_sha256': digest(args.input),
        'reference_sha256': digest(args.fasta), 'checkpoint_sha256': digest(checkpoint),
        'model_repository': 'aaronfeller/StabilityArc', 'encoder': 'ESMC-600M',
        'torch_version': torch.__version__, 'device': args.device, 'seed': args.seed,
        'outcome_tolerance': args.outcome_tolerance,
        'experimental_values': 'Approximate vector-digitized normalized fluorescence fold changes; error-bar statistic unspecified; negative values retained.',
        'failure_definition': 'Mean outcome below its matched control by more than tolerance; not a significance test.',
        'stability_direction': 'Score < 0 is predicted destabilization; ddG = -score, not calibrated kcal/mol.',
        'comparison': 'Effect of one added substitution on each reconstructed background, not total stability relative to WT.',
    }
    (args.output_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(statistics, indent=2), flush=True)
    print(f'Outputs: {args.output_dir.relative_to(ROOT)}', flush=True)


if __name__ == '__main__':
    main()