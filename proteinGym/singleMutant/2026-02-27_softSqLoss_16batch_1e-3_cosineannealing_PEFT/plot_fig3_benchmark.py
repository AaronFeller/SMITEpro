#!/usr/bin/env python3
"""
Generate clean, publication-grade Figure 3 (manuscript/final_figures/benchmark_bubble_matrix.png & .pdf)
showing Top-N Variant Discovery across 106 ProteinGym DMS Assays for:
  - Mean Selected Fitness (Normalized Target)
  - Top 1% Enrichment (Fold over Random)
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

workspace_root = Path('/stor/work/Wilke/alf3564/smitepro')
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

hist_dir = workspace_root / 'proteinGym/singleMutant/2026-02-27_softSqLoss_16batch_1e-3_cosineannealing_PEFT'
cache_csv = hist_dir / 'benchmark_106_assays_per_assay_7methods.csv'

if not cache_csv.is_file():
    nb = json.load(open(hist_dir / '01_table_analysis.ipynb'))
    ns = {'display': lambda *args, **kwargs: None}
    for cell_idx in [23, 24]:
        src = ''.join(nb['cells'][cell_idx]['source'])
        src = src.replace('plt.show()', 'pass').replace('plt.savefig', '# plt.savefig')
        exec(src, ns)

    budget_plot_metrics = ns['budget_plot_metrics']
    comparison_shots = sorted(ns['comparison_shots'])
    comparison_budgets = sorted(ns['comparison_budgets'])
    method_labels = ns['method_labels']
    budget_display_methods = ns['budget_display_methods']

    bubble_metrics = ['Mean Top-N Fitness', 'Top 1% Precision', 'Enrichment Factor']
    bubble_labels = {
        **method_labels,
        'Ensemble': 'SMITEpro + EVOLVEpro',
    }
    bubble_methods = [
        m for m in budget_display_methods
        if 'mlde' not in f'{m} {bubble_labels.get(m, m)}'.lower()
    ]
    for m in bubble_methods:
        if 'kermut' in f'{m} {bubble_labels.get(m, m)}'.lower():
            bubble_labels[m] = 'Gaussian Process'

    bubble_source = budget_plot_metrics.loc[
        budget_plot_metrics['metric'].isin(bubble_metrics)
        & budget_plot_metrics['train_size'].isin(comparison_shots)
        & budget_plot_metrics['N'].isin(comparison_budgets)
        & budget_plot_metrics['method'].isin(bubble_methods)
    ].copy()
    expected = len(bubble_methods) * len(bubble_metrics) * len(comparison_shots) * len(comparison_budgets)
    cov = bubble_source.groupby('assay').size()
    complete_assays = cov.index[cov.eq(expected)]
    assert len(complete_assays) == 106
    bubble_source = bubble_source.loc[bubble_source['assay'].isin(complete_assays)].copy()
    bubble_source['display_method'] = bubble_source['method'].map(lambda m: bubble_labels.get(m, m))
    bubble_source.to_csv(cache_csv, index=False)
    print(f'Saved per-assay cache ({len(bubble_source)} rows) to {cache_csv}')
else:
    bubble_source = pd.read_csv(cache_csv)
    bubble_source['display_method'] = bubble_source['display_method'].fillna(bubble_source['method'])
    print(f'Loaded per-assay cache ({len(bubble_source)} rows) from {cache_csv}')

# Method order and muted, desaturated Nature-style scientific palette (shared with Figure 2)
methods_order = [
    'SMITEpro + EVOLVEpro',
    'SMITEpro',
    'EVOLVEpro',
    'Gaussian Process',
    'MSA PSSM',
    'Zero-shot ESMC',
]

line_methods = methods_order

# Balanced medium-soft bar fills paired with crisp saturation-matched line/edge tones
palette_bar = {
    'SMITEpro + EVOLVEpro': '#6FAF8B',  # Balanced medium-soft sage green (shared with Fig 2B)
    'SMITEpro': '#5B8DB8',              # Balanced medium-soft slate-sky blue (shared with Fig 2B)
    'EVOLVEpro': '#EB9E4C',             # Balanced medium-soft warm amber-apricot (shared with Fig 2B)
    'Gaussian Process': '#66B5A9',      # Balanced medium-soft aqua-teal
    'MSA PSSM': '#E37C6B',              # Balanced medium-soft salmon-terracotta
    'Zero-shot ESMC': '#9E97C4',        # Balanced medium-soft periwinkle-lavender
}

palette_line = {
    'SMITEpro + EVOLVEpro': '#2F6E4A',  # Crisp sage-emerald for lines & bar borders
    'SMITEpro': '#2B5C8F',              # Crisp sky-cobalt
    'EVOLVEpro': '#B86614',             # Crisp warm amber
    'Gaussian Process': '#2B7A6E',      # Crisp teal
    'MSA PSSM': '#B04132',              # Crisp coral-terracotta
    'Zero-shot ESMC': '#5E548E',        # Crisp periwinkle-violet
}

markers = {
    'SMITEpro + EVOLVEpro': 'D',
    'SMITEpro': 'o',
    'EVOLVEpro': 's',
    'Gaussian Process': '^',
    'MSA PSSM': 'v',
    'Zero-shot ESMC': 'P',
}

linestyles = {
    'SMITEpro + EVOLVEpro': '-',
    'SMITEpro': '-',
    'EVOLVEpro': '-',
    'Gaussian Process': '-',
    'MSA PSSM': '--',
    'Zero-shot ESMC': ':',
}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 10.5,
    'axes.labelsize': 11.0,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

fig = plt.figure(figsize=(14.0, 8.5), dpi=300)
gs = fig.add_gridspec(
    2, 2,
    width_ratios=[1.24, 1.0],
    height_ratios=[1.0, 1.0],
    wspace=0.22,
    hspace=0.30,
)

# Top row (A, B) = Top 1% Enrichment; Bottom row (C, D) = Mean Selected Fitness
metrics_cfg = [
    {
        'metric': 'Enrichment Factor',
        'ylabel': 'Top 1% Enrichment\n(Fold over Random)',
        'ylim_bar': (0.0, 6.8),
        'ylim_line': (0.5, 6.3),
        'panel_left': 'A',
        'panel_right': 'B',
        'random_baseline': 1.0,
    },
    {
        'metric': 'Mean Top-N Fitness',
        'ylabel': 'Mean Selected Fitness\n(Normalized Target)',
        'ylim_bar': (0.0, 0.57),
        'ylim_line': (0.0, 0.55),
        'panel_left': 'C',
        'panel_right': 'D',
        'random_baseline': None,  # computed dynamically from the overall Random expectation
    },
]

shots = [16, 32, 64, 128]
budgets = [10, 20, 30, 40]
n_methods = len(methods_order)
bar_width = 0.128
offsets = (np.arange(n_methods) - (n_methods - 1) / 2.0) * bar_width

for row_idx, cfg in enumerate(metrics_cfg):
    metric_name = cfg['metric']
    sub_df = bubble_source.loc[bubble_source['metric'] == metric_name]

    # Exact K- and N-invariant Random expectation (1.0 for Enrichment; overall held-out mean for Fitness)
    rand_const = cfg['random_baseline']
    if rand_const is None:
        rand_const = float(sub_df.loc[sub_df['display_method'] == 'Random', 'value'].mean())

    # --- Left Panel: Grouped Bar Plot across K = 16, 32, 64, 128 at N = 20 ---
    ax_bar = fig.add_subplot(gs[row_idx, 0])
    n20_df = sub_df.loc[sub_df['N'] == 20]
    stats_n20 = (
        n20_df.groupby(['train_size', 'display_method'])['value']
        .agg(['mean', 'sem'])
        .reset_index()
    )

    x_centers = np.arange(len(shots))
    for m_idx, method in enumerate(methods_order):
        m_stats = stats_n20.loc[stats_n20['display_method'] == method].set_index('train_size').reindex(shots)
        means = m_stats['mean'].to_numpy()
        sems = m_stats['sem'].to_numpy()
        x_pos = x_centers + offsets[m_idx]

        is_hero = method in ('SMITEpro + EVOLVEpro', 'SMITEpro')
        ax_bar.bar(
            x_pos,
            means,
            width=bar_width * 0.90,
            color=palette_bar[method],
            edgecolor=palette_line[method],
            linewidth=1.05 if is_hero else 0.85,
            alpha=0.95,
            label=method,
            zorder=3,
        )
        ax_bar.errorbar(
            x_pos,
            means,
            yerr=sems,
            fmt='none',
            ecolor='#2B2B2B',
            elinewidth=0.85,
            capsize=2.2,
            capthick=0.8,
            zorder=4,
        )

    for sep in range(len(shots) - 1):
        ax_bar.axvline(sep + 0.5, color='#CCCCCC', linestyle='--', linewidth=0.8, zorder=1)

    ax_bar.axhline(rand_const, color='#555555', linestyle=':', linewidth=1.5, zorder=5)

    ax_bar.set_xticks(x_centers)
    ax_bar.set_xticklabels([f'K = {k}' for k in shots], fontweight='medium')
    ax_bar.set_xlabel('Training Set Size, K (at Budget N = 20)')
    ax_bar.set_ylabel(cfg['ylabel'])
    ax_bar.set_ylim(*cfg['ylim_bar'])
    ax_bar.set_xlim(-0.52, len(shots) - 0.48)
    ax_bar.grid(axis='y', linestyle=':', alpha=0.5, zorder=0)
    ax_bar.text(
        -0.11, 1.03, cfg['panel_left'],
        transform=ax_bar.transAxes,
        fontsize=14, fontweight='normal', va='bottom', ha='left'
    )

    # --- Right Panel: Budget Scaling Trajectory across N = 10, 20, 30, 40 ---
    ax_line = fig.add_subplot(gs[row_idx, 1])
    per_assay_budget = (
        sub_df.groupby(['assay', 'display_method', 'N'])['value']
        .mean()
        .reset_index()
    )
    stats_budget = (
        per_assay_budget.groupby(['display_method', 'N'])['value']
        .agg(['mean', 'sem'])
        .reset_index()
    )

    # Plot constant horizontal dotted line for Random expectation (budget-invariant)
    ax_line.axhline(
        rand_const,
        color='#555555',
        linestyle=':',
        linewidth=1.5,
        zorder=2,
    )
    ax_line.text(
        39.6,
        rand_const + (0.14 if metric_name == 'Enrichment Factor' else 0.013),
        'Random expectation',
        color='#555555',
        fontsize=8.8,
        ha='right',
        va='bottom',
        style='italic',
    )

    # Monotonic left-to-right cascade matching legend/rank order (green -> blue -> orange -> teal -> coral -> purple)
    line_jitters = {
        'SMITEpro + EVOLVEpro': -0.55,
        'SMITEpro': -0.33,
        'EVOLVEpro': -0.11,
        'Gaussian Process': 0.11,
        'MSA PSSM': 0.33,
        'Zero-shot ESMC': 0.55,
    }

    for method in line_methods:
        m_b = stats_budget.loc[stats_budget['display_method'] == method].set_index('N').reindex(budgets)
        means = m_b['mean'].to_numpy()
        sems = m_b['sem'].to_numpy()
        is_hero = method in ('SMITEpro + EVOLVEpro', 'SMITEpro')
        x_vals = np.array(budgets, dtype=float) + line_jitters[method]

        # Plot ± s.e.m. variability for ALL methods (light ribbon for all 6 + crisp error bars)
        ax_line.fill_between(
            budgets,
            means - sems,
            means + sems,
            color=palette_bar[method],
            alpha=0.22 if is_hero else 0.12,
            zorder=2,
        )
        ax_line.errorbar(
            x_vals,
            means,
            yerr=sems,
            color=palette_line[method],
            linestyle=linestyles[method],
            linewidth=2.2 if is_hero else 1.65,
            marker=markers[method],
            markersize=6.5 if is_hero else 5.4,
            markerfacecolor=palette_bar[method],
            markeredgecolor=palette_line[method],
            markeredgewidth=1.1,
            elinewidth=0.95,
            capsize=2.3,
            capthick=0.85,
            label=method,
            zorder=5 if is_hero else 3,
        )

    ax_line.set_xticks(budgets)
    ax_line.set_xticklabels([f'N = {n}' for n in budgets], fontweight='medium')
    ax_line.set_xlabel('Screening Budget, N (Variants per Round)')
    ax_line.set_ylabel(cfg['ylabel'])
    ax_line.set_ylim(*cfg['ylim_line'])
    ax_line.set_xlim(7.5, 42.5)
    ax_line.grid(axis='both', linestyle=':', alpha=0.5, zorder=0)
    ax_line.text(
        -0.14, 1.03, cfg['panel_right'],
        transform=ax_line.transAxes,
        fontsize=14, fontweight='normal', va='bottom', ha='left'
    )

# Shared top legend across the entire figure (6 methods as patches + Random expectation as dotted line)
legend_handles = [
    mpatches.Patch(
        facecolor=palette_bar[m],
        edgecolor=palette_line[m],
        linewidth=1.0,
        label=m,
    )
    for m in methods_order
] + [
    mlines.Line2D(
        [], [],
        color='#555555',
        linestyle=':',
        linewidth=1.6,
        label='Random expectation',
    )
]
fig.legend(
    handles=legend_handles,
    loc='upper center',
    bbox_to_anchor=(0.5, 0.99),
    ncol=7,
    frameon=True,
    facecolor='white',
    edgecolor='#CCCCCC',
    columnspacing=1.15,
    handlelength=1.6,
    handletextpad=0.5,
)

fig.subplots_adjust(top=0.90, bottom=0.08, left=0.08, right=0.98)

out_dirs = [
    workspace_root / 'manuscript/final_figures',
    workspace_root / 'manuscript/figures',
]
for d in out_dirs:
    d.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(d / f'benchmark_bubble_matrix.{ext}', dpi=300, bbox_inches='tight')
        fig.savefig(d / f'fig3_enrichment.{ext}', dpi=300, bbox_inches='tight')
print('Successfully generated redesigned Figure 3 in manuscript/final_figures/fig3_enrichment.png & .pdf')
