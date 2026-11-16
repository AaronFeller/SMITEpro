#!/usr/bin/env python3
"""Generate:
1) A new Figure 4 variant (`fig4_hpab_experiment_with_top_prior.png/.pdf`) that includes the
   top-1 prior training single mutation on each parental background in a distinct color and label.
2) Supplementary figures (one per stage + a 6-panel overview) comparing all prior training
   single mutations on each parental background alongside the SMITEpro-EVOLVEpro ensemble
   prospective candidates.
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


ROOT = Path('/stor/work/Wilke/alf3564/smitepro')
EXP_DIR = ROOT / 'internal_data/Dan_data/experimental_results'
OUT_DIRS = [
    ROOT / 'manuscript/final_figures',
    ROOT / 'manuscript/figures',
    ROOT / 'SMITEpro_github_repo/manuscript/final_figures',
    ROOT / 'SMITEpro_github_repo/manuscript/figures',
]

BACKGROUNDS = [
    'WT',
    'A199P',
    'I29V_A199P',
    'I29V_A199P_D354K',
    'I29V_A199P_D354K_G122A',
    'I29V_A199P_D354K_H454Q_H372P',
]

PRIOR_COLOR = '#2468b4'        # Royal blue for prior training single mutant(s)
ABOVE_COLOR = '#238b57'        # Green for ensemble candidates above background
BELOW_COLOR = '#78838c'        # Slate grey for ensemble candidates at/below background
CONTROL_COLOR = '#bd5c18'      # Orange-brown for parental background control
REP_DOT_COLOR = '#b9bfc3'      # Light grey for individual replicate dots


def canonical_mut(m):
    if pd.isna(m) or not str(m).strip() or str(m).strip() == 'WT':
        return frozenset()
    parts = str(m).strip().replace('+', ':').replace('_', ':').split(':')
    norm = []
    for p in parts:
        p = p.strip()
        if not p or p == 'WT':
            continue
        m_obj = re.match(r'^([A-Z])0*(\d+)([A-Z])$', p)
        if m_obj:
            norm.append(f"{m_obj.group(1)}{int(m_obj.group(2))}{m_obj.group(3)}")
        else:
            norm.append(p)
    return frozenset(norm)


def load_data():
    summary = pd.read_csv(EXP_DIR / 'experimental_prism_variant_summary.csv')
    measurements = pd.read_csv(EXP_DIR / 'experimental_prism_inferred_measurements.csv')
    df1 = pd.read_csv(ROOT / 'internal_data/Dan_data/round2/hpab_data.csv')
    df2 = pd.read_csv(ROOT / 'internal_data/Dan_data/round2/hpab_data2.csv')
    wt_seq = df2['wt_sequence'].iloc[0]

    df1_wt = df1[df1['wt_sequence'] == wt_seq].copy()
    df1_wt['mut_set'] = df1_wt['mutant'].apply(canonical_mut)
    unrounded_lookup = {row['mut_set']: float(row['target']) for _, row in df1_wt.iterrows()}

    df2['mut_set'] = df2['mutant'].apply(canonical_mut)
    df2['fold_wt_rounded'] = df2['target'] + 1.0
    df2['fold_wt_exact'] = df2['mut_set'].apply(lambda s: unrounded_lookup.get(s, np.nan))
    df2['fold_wt'] = df2['fold_wt_exact'].fillna(df2['fold_wt_rounded'])

    prior_by_round = {}
    for r_num, bg_name in enumerate(BACKGROUNDS, 1):
        bg_set = canonical_mut(bg_name)
        singles = df2[df2['mut_set'].apply(lambda s: bg_set.issubset(s) and len(s - bg_set) == 1)].copy()
        singles['added_mutation'] = singles['mut_set'].apply(lambda s: list(s - bg_set)[0])
        singles = singles.sort_values(['fold_wt', 'added_mutation'], ascending=[False, True]).reset_index(drop=True)
        prior_by_round[r_num] = singles

    return summary, measurements, prior_by_round


def make_fig4_with_top_prior(summary, measurements, prior_by_round):
    """Create new Figure 4 with the top-1 prior training single mutation in each background."""
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 14.5), sharex=False, layout='constrained')
    fig.get_layout_engine().set(hspace=0.07, wspace=0.05, rect=(0.02, 0.04, 0.96, 0.96))
    fig.supxlabel('Fold-change over WT', fontsize=14, y=0.03)

    rng = np.random.default_rng(20261020)

    for round_num, axis in enumerate(axes.flat, start=1):
        panel = summary.loc[summary['round'].eq(round_num)].copy()
        panel['is_prior'] = False

        # Add top-1 prior single mutant if available for this background
        priors = prior_by_round[round_num]
        if len(priors) > 0:
            top1 = priors.iloc[0]
            bg_row = panel.loc[panel['is_control']].iloc[0]
            prior_row = {
                'round': round_num,
                'background': bg_row['background'],
                'variant': f"PRIOR_{round_num}_{top1['added_mutation']}",
                'added_mutation': f"Prior: {top1['added_mutation']}",
                'is_control': False,
                'is_prior': True,
                'background_fold_wt': bg_row['background_fold_wt'],
                'mean_relative_wt': float(top1['fold_wt']),
                'sem_relative_wt': np.nan,
                'mean_relative_background': float(top1['fold_wt']) / float(bg_row['background_fold_wt']),
                'sem_relative_background': np.nan,
                'reconstruction_status': 'prior_training_data',
            }
            panel = pd.concat([panel, pd.DataFrame([prior_row])], ignore_index=True)

        ordered = panel.sort_values(['mean_relative_wt', 'variant'], ascending=[False, True], kind='stable').reset_index(drop=True)

        panel_values = measurements.loc[measurements['variant'].isin(ordered['variant']), 'relative_wt']
        means = ordered['mean_relative_wt']
        errors = ordered['sem_relative_wt'].fillna(0)
        lower = min(float(panel_values.min()), float((means - errors).min()), 0)
        upper = max(float(panel_values.max()), float((means + errors).max()), 1)
        padding = 0.05 * (upper - lower)
        lower, upper = lower - padding, upper + padding
        reference_line = ordered.loc[ordered['is_control'], 'mean_relative_wt'].item()

        axis.set_xscale('linear')
        axis.set_xlim(lower, upper)
        axis.minorticks_off()
        axis.set_ylim(len(ordered) - 0.4, -0.7)
        axis.set_yticks(range(len(ordered)))
        axis.set_yticklabels([
            'Background' if r.is_control else r.added_mutation
            for r in ordered.itertuples()
        ], fontsize=13.5)
        axis.axvline(reference_line, linestyle='--', linewidth=1, color='#252b30')
        axis.grid(axis='x', which='major', color='#dce1e3', linewidth=0.6)
        axis.grid(axis='y', color='#edf0f1', linewidth=0.4)
        axis.tick_params(axis='y', length=0)
        axis.tick_params(axis='x', labelbottom=True, labelsize=14)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
        for side in ('top', 'right', 'left'):
            axis.spines[side].set_visible(False)

        for label, row in zip(axis.get_yticklabels(), ordered.itertuples()):
            if row.is_control:
                label.set_fontweight('bold')
            elif row.is_prior:
                label.set_color(PRIOR_COLOR)
                label.set_fontweight('bold')

        for position, record in enumerate(ordered.to_dict('records')):
            if record['is_prior']:
                axis.axhspan(position - 0.43, position + 0.43, color='#eaf1f8', zorder=0)
                axis.scatter(record['mean_relative_wt'], position, s=64, marker='D',
                             color=PRIOR_COLOR, edgecolors='white', linewidths=0.6, zorder=4)
                continue

            values = measurements.loc[measurements['variant'].eq(record['variant']), 'relative_wt'].to_numpy()
            jitter = position + rng.uniform(-0.18, 0.18, len(values))
            axis.scatter(values, jitter, s=22, color=REP_DOT_COLOR, alpha=0.7,
                         edgecolors='none', zorder=1.5)

            mean = record['mean_relative_wt']
            deviation = record['sem_relative_wt']
            if record['is_control']:
                color = CONTROL_COLOR
                axis.axhspan(position - 0.43, position + 0.43, color='#f1f2f2', zorder=0)
            else:
                color = ABOVE_COLOR if mean > reference_line else BELOW_COLOR

            if np.isfinite(deviation):
                axis.errorbar(mean, position, xerr=deviation,
                              fmt='none', color=color, capsize=3, elinewidth=1.6, zorder=2)
            axis.scatter(mean, position, s=48, marker='o',
                         color=color, edgecolors='white', linewidths=0.5, zorder=3)

        background = BACKGROUNDS[round_num - 1]
        title = background.replace('_', ' + ')
        if round_num >= 5:
            title = 'I29V + A199P + D354K\n+ ' + ('G122A' if round_num == 5 else 'H454Q + H372P')
        axis.set_title(f'Round {round_num}\n{title}', loc='left', fontsize=14, fontweight='bold', pad=12)

    legend_handles = [
        Line2D([], [], marker='o', linestyle='none', color=ABOVE_COLOR, markersize=7,
               label='Ensemble candidate > background'),
        Line2D([], [], marker='o', linestyle='none', color=BELOW_COLOR, markersize=7,
               label='Ensemble candidate <= background'),
        Line2D([], [], marker='o', linestyle='none', color=CONTROL_COLOR, markersize=7,
               label='Parental background control'),
        Line2D([], [], marker='D', linestyle='none', color=PRIOR_COLOR, markersize=7,
               label='Top-1 prior training single mutant'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, 0.002),
               ncol=4, frameon=False, fontsize=11.5)

    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        for ext in ('png', 'pdf'):
            fig.savefig(d / f'fig4_hpab_experiment_with_top_prior.{ext}', dpi=300)
    plt.close(fig)
    print('Saved fig4_hpab_experiment_with_top_prior.png & .pdf')


def plot_single_stage_panel(axis, round_num, summary, measurements, priors, rng, is_standalone=True):
    """Plot a single stage showing all prior training single mutants alongside ensemble candidates."""
    panel = summary.loc[summary['round'].eq(round_num)].copy()
    panel['source_group'] = np.where(panel['is_control'], 'control', 'ensemble')
    bg_row = panel.loc[panel['is_control']].iloc[0]
    reference_line = float(bg_row['mean_relative_wt'])

    # For Round 1 (WT), there are 329 prior singles. In a dot-and-bar caterpillar plot,
    # if len(priors) > 35, show all prior singles that are >= WT (or top 25) as labeled rows
    # OR if standalone Round 1, we can show a two-panel view (all 329 sorted distribution + top subset)
    prior_rows = []
    for _, pr in priors.iterrows():
        prior_rows.append({
            'round': round_num,
            'background': bg_row['background'],
            'variant': f"PRIOR_{round_num}_{pr['added_mutation']}",
            'added_mutation': f"Prior: {pr['added_mutation']}",
            'raw_mutation': pr['added_mutation'],
            'is_control': False,
            'source_group': 'prior',
            'background_fold_wt': bg_row['background_fold_wt'],
            'mean_relative_wt': float(pr['fold_wt']),
            'sem_relative_wt': np.nan,
        })
    if prior_rows:
        panel = pd.concat([panel, pd.DataFrame(prior_rows)], ignore_index=True)

    ordered = panel.sort_values(['mean_relative_wt', 'added_mutation'], ascending=[False, True], kind='stable').reset_index(drop=True)

    panel_values = measurements.loc[measurements['variant'].isin(ordered['variant']), 'relative_wt']
    means = ordered['mean_relative_wt']
    errors = ordered['sem_relative_wt'].fillna(0)
    lower = min(float(panel_values.min()) if len(panel_values) else 0, float((means - errors).min()), 0)
    upper = max(float(panel_values.max()) if len(panel_values) else 1, float((means + errors).max()), 1)
    padding = 0.05 * (upper - lower)
    lower, upper = lower - padding, upper + padding

    axis.set_xscale('linear')
    axis.set_xlim(lower, upper)
    axis.minorticks_off()
    axis.set_ylim(len(ordered) - 0.4, -0.7)
    axis.set_yticks(range(len(ordered)))
    fs = 11.5 if len(ordered) <= 30 else 8.5
    axis.set_yticklabels([
        'Background' if r.is_control else r.added_mutation
        for r in ordered.itertuples()
    ], fontsize=fs)
    axis.axvline(reference_line, linestyle='--', linewidth=1.1, color='#252b30')
    axis.grid(axis='x', which='major', color='#dce1e3', linewidth=0.6)
    axis.grid(axis='y', color='#edf0f1', linewidth=0.4)
    axis.tick_params(axis='y', length=0)
    axis.tick_params(axis='x', labelbottom=True, labelsize=12)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=4))
    for side in ('top', 'right', 'left'):
        axis.spines[side].set_visible(False)

    for label, row in zip(axis.get_yticklabels(), ordered.itertuples()):
        if row.is_control:
            label.set_fontweight('bold')
        elif row.source_group == 'prior':
            label.set_color(PRIOR_COLOR)
            label.set_fontweight('bold')

    for position, record in enumerate(ordered.to_dict('records')):
        if record['source_group'] == 'prior':
            axis.axhspan(position - 0.43, position + 0.43, color='#eaf1f8', zorder=0)
            axis.scatter(record['mean_relative_wt'], position, s=52, marker='D',
                         color=PRIOR_COLOR, edgecolors='white', linewidths=0.6, zorder=4)
            continue

        values = measurements.loc[measurements['variant'].eq(record['variant']), 'relative_wt'].to_numpy()
        jitter = position + rng.uniform(-0.18, 0.18, len(values))
        axis.scatter(values, jitter, s=20, color=REP_DOT_COLOR, alpha=0.7,
                     edgecolors='none', zorder=1.5)

        mean = record['mean_relative_wt']
        deviation = record['sem_relative_wt']
        if record['is_control']:
            color = CONTROL_COLOR
            axis.axhspan(position - 0.43, position + 0.43, color='#f1f2f2', zorder=0)
        else:
            color = ABOVE_COLOR if mean > reference_line else BELOW_COLOR

        if np.isfinite(deviation):
            axis.errorbar(mean, position, xerr=deviation,
                          fmt='none', color=color, capsize=2.8, elinewidth=1.5, zorder=2)
        axis.scatter(mean, position, s=44, marker='o',
                     color=color, edgecolors='white', linewidths=0.5, zorder=3)


def make_supplement_stage_plots(summary, measurements, prior_by_round):
    """Make one standalone plot per stage (Rounds 1-6) plus a combined 6-panel supplementary figure."""
    rng = np.random.default_rng(20261020)

    # 1. Individual stage plots: supp_hpab_stage1_WT .. supp_hpab_stage6_...
    for round_num, bg_name in enumerate(BACKGROUNDS, start=1):
        priors = prior_by_round[round_num]
        bg_title = bg_name.replace('_', ' + ')

        if round_num == 1:
            # Round 1 has 329 prior WT single mutants!
            # Create a two-panel figure for Stage 1:
            # Left (A): Full distribution of all 329 prior WT single mutants + 20 Round 1 ensemble candidates
            # Right (B): Detailed labeled comparison of the 20 Round 1 ensemble candidates + all 37 prior WT single mutants > WT (>= 1.0x)
            fig, (ax_all, ax_detail) = plt.subplots(1, 2, figsize=(15.5, 13.5), layout='constrained')
            fig.get_layout_engine().set(wspace=0.08, rect=(0.01, 0.04, 0.99, 0.95))

            # Left panel: all 329 prior singles + 20 ensemble candidates + WT control
            p1 = summary.loc[summary['round'].eq(1)].copy()
            p1['source_group'] = np.where(p1['is_control'], 'control', 'ensemble')
            pr_df = pd.DataFrame([{
                'round': 1, 'background': 'WT',
                'variant': f"PRIOR_1_{r['added_mutation']}",
                'added_mutation': f"Prior: {r['added_mutation']}",
                'is_control': False, 'source_group': 'prior',
                'background_fold_wt': 1.0,
                'mean_relative_wt': float(r['fold_wt']),
                'sem_relative_wt': np.nan,
            } for _, r in priors.iterrows()])
            comb_all = pd.concat([p1, pr_df], ignore_index=True).sort_values(
                ['mean_relative_wt', 'added_mutation'], ascending=[False, True], kind='stable'
            ).reset_index(drop=True)

            y_idx = np.arange(len(comb_all))
            is_pr = comb_all['source_group'].eq('prior').to_numpy()
            is_ens = comb_all['source_group'].eq('ensemble').to_numpy()
            is_ctrl = comb_all['is_control'].to_numpy()

            ax_all.scatter(comb_all.loc[is_pr, 'mean_relative_wt'], y_idx[is_pr],
                           s=14, marker='D', color=PRIOR_COLOR, alpha=0.65, label='Prior training single mutants (n = 329)', zorder=2)
            for pos in y_idx[is_ens]:
                rec = comb_all.iloc[pos]
                c = ABOVE_COLOR if rec['mean_relative_wt'] > 1.0 else BELOW_COLOR
                if np.isfinite(rec['sem_relative_wt']):
                    ax_all.errorbar(rec['mean_relative_wt'], pos, xerr=rec['sem_relative_wt'],
                                    fmt='none', color=c, capsize=2, elinewidth=1.2, zorder=3)
                ax_all.scatter(rec['mean_relative_wt'], pos, s=36, marker='o', color=c,
                               edgecolors='white', linewidths=0.4, zorder=4)
            ctrl_pos = y_idx[is_ctrl][0]
            ax_all.scatter(1.0, ctrl_pos, s=55, marker='o', color=CONTROL_COLOR, edgecolors='white', linewidths=0.6, zorder=5)
            ax_all.axvline(1.0, linestyle='--', linewidth=1.1, color='#252b30')
            ax_all.set_ylim(len(comb_all) + 2, -5)
            ax_all.set_ylabel('Rank across all 349 single mutants on WT (sorted by fold-change)', fontsize=12)
            ax_all.set_xlabel('Fold-change over WT', fontsize=12)
            ax_all.set_title('Stage 1 (WT): All 329 Prior Singles + 20 Ensemble Candidates', loc='left', fontsize=13, fontweight='bold', pad=10)
            ax_all.grid(axis='x', color='#dce1e3', linewidth=0.6)
            for side in ('top', 'right'):
                ax_all.spines[side].set_visible(False)

            # Annotate top 5 prior mutants on left panel with clean non-overlapping offsets
            annot_offsets = [(-0.85, 2), (-0.85, 11), (0.12, -2), (0.12, 8), (0.12, 16)]
            for pos, (dx, dy) in zip(range(5), annot_offsets):
                rec = comb_all.iloc[pos]
                ax_all.annotate(rec['added_mutation'].replace('Prior: ', ''),
                                xy=(rec['mean_relative_wt'], pos),
                                xytext=(rec['mean_relative_wt'] + dx, pos + dy),
                                fontsize=9.5, color=PRIOR_COLOR, fontweight='bold')

            # Right panel: Top 20 prior singles on WT alongside the 20 Round 1 ensemble candidates
            top20_priors = priors.head(20)
            plot_single_stage_panel(ax_detail, 1, summary, measurements, top20_priors, rng, is_standalone=True)
            ax_detail.set_title('Stage 1 (WT): Top 20 Prior Singles vs. 20 Ensemble Candidates', loc='left', fontsize=13, fontweight='bold', pad=10)
            ax_detail.set_xlabel('Fold-change over WT', fontsize=12)

            legend_handles = [
                Line2D([], [], marker='o', linestyle='none', color=ABOVE_COLOR, markersize=7,
                       label='Ensemble candidate > background'),
                Line2D([], [], marker='o', linestyle='none', color=BELOW_COLOR, markersize=7,
                       label='Ensemble candidate <= background'),
                Line2D([], [], marker='o', linestyle='none', color=CONTROL_COLOR, markersize=7,
                       label='Parental background control'),
                Line2D([], [], marker='D', linestyle='none', color=PRIOR_COLOR, markersize=7,
                       label='Prior training single mutant on same background'),
            ]
            fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, 0.002),
                       ncol=4, frameon=False, fontsize=11)
        else:
            fig, axis = plt.subplots(figsize=(7.5, 8.5), layout='constrained')
            fig.get_layout_engine().set(rect=(0.02, 0.06, 0.98, 0.94))
            plot_single_stage_panel(axis, round_num, summary, measurements, priors, rng, is_standalone=True)
            n_pr = len(priors)
            sub_note = f"({n_pr} prior training single mutant{'s' if n_pr != 1 else ''} on this background)"
            axis.set_title(f"Stage {round_num}: {bg_title}\n{sub_note}", loc='left', fontsize=13, fontweight='bold', pad=10)
            axis.set_xlabel('Fold-change over WT', fontsize=12)

            legend_handles = [
                Line2D([], [], marker='o', linestyle='none', color=ABOVE_COLOR, markersize=6.5,
                       label='Ensemble candidate > background'),
                Line2D([], [], marker='o', linestyle='none', color=BELOW_COLOR, markersize=6.5,
                       label='Ensemble candidate <= background'),
                Line2D([], [], marker='o', linestyle='none', color=CONTROL_COLOR, markersize=6.5,
                       label='Parental background'),
                Line2D([], [], marker='D', linestyle='none', color=PRIOR_COLOR, markersize=6.5,
                       label='Prior training single mutant'),
            ]
            fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, 0.002),
                       ncol=2, frameon=False, fontsize=10)

        for d in OUT_DIRS:
            d.mkdir(parents=True, exist_ok=True)
            for ext in ('png', 'pdf'):
                fig.savefig(d / f'supp_hpab_stage{round_num}_{bg_name}.{ext}', dpi=300)
        plt.close(fig)
        print(f'Saved supp_hpab_stage{round_num}_{bg_name}.png & .pdf')

    # 2. Combined 6-panel Supplementary Figure showing all prior singles on Rounds 2-6 and Top-10 prior singles on Round 1
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 16.5), sharex=False, layout='constrained')
    fig.get_layout_engine().set(hspace=0.07, wspace=0.05, rect=(0.02, 0.04, 0.96, 0.96))
    fig.supxlabel('Fold-change over WT', fontsize=14, y=0.03)

    for round_num, axis in enumerate(axes.flat, start=1):
        priors = prior_by_round[round_num]
        if round_num == 1:
            # Show top 10 prior WT singles in the 6-panel grid so labels remain readable
            priors_to_plot = priors.head(10)
            note = 'Top 10 of 329 prior singles shown'
        else:
            priors_to_plot = priors
            note = f'All {len(priors)} prior singles shown' if len(priors) > 0 else '0 prior singles on this background'

        plot_single_stage_panel(axis, round_num, summary, measurements, priors_to_plot, rng, is_standalone=False)
        axis.set_xlabel('')
        bg_name = BACKGROUNDS[round_num - 1]
        title = bg_name.replace('_', ' + ')
        if round_num >= 5:
            title = 'I29V + A199P + D354K\n+ ' + ('G122A' if round_num == 5 else 'H454Q + H372P')
        axis.set_title(f'Round {round_num}: {title}\n({note})', loc='left', fontsize=12.5, fontweight='bold', pad=10)

    legend_handles = [
        Line2D([], [], marker='o', linestyle='none', color=ABOVE_COLOR, markersize=7,
               label='Ensemble candidate > background'),
        Line2D([], [], marker='o', linestyle='none', color=BELOW_COLOR, markersize=7,
               label='Ensemble candidate <= background'),
        Line2D([], [], marker='o', linestyle='none', color=CONTROL_COLOR, markersize=7,
               label='Parental background control'),
        Line2D([], [], marker='D', linestyle='none', color=PRIOR_COLOR, markersize=7,
               label='Prior training single mutant on same background'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, 0.002),
               ncol=4, frameon=False, fontsize=11.5)

    for d in OUT_DIRS:
        for ext in ('png', 'pdf'):
            fig.savefig(d / f'supp_hpab_all_stages_with_priors.{ext}', dpi=300)
    plt.close(fig)
    print('Saved supp_hpab_all_stages_with_priors.png & .pdf')


if __name__ == '__main__':
    summary, measurements, prior_by_round = load_data()
    make_fig4_with_top_prior(summary, measurements, prior_by_round)
    make_supplement_stage_plots(summary, measurements, prior_by_round)
