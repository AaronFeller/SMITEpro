import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.stats import spearmanr

peft_dir = '/stor/work/Wilke/alf3564/smitepro/proteinGym/singleMutant/2026-02-27_softSqLoss_16batch_1e-3_cosineannealing_PEFT'
orig_dir = '/stor/work/Wilke/alf3564/smitepro/datasets/DMS_ProteinGym_substitutions/cv_folds_singles_substitutions'

def safe_spearman(x, y):
    if len(x) < 10 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    r, _ = spearmanr(x, y)
    return r

def calc_ndcg(pred, target):
    """Chance-adjusted Normalized Discounted Cumulative Gain (NDCG) over the full candidate pool.
    Logarithmically discounts ranks by 1 / log2(rank + 1) so top-ranked selections dominate,
    and normalizes against the random expectation E[DCG_rand] so 0 = random ordering and 1 = ideal ordering
    independently of candidate pool size.
    """
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    n = len(pred)
    if n < 10 or np.std(pred) < 1e-12 or np.std(target) < 1e-12:
        return np.nan
    t_min, t_max = np.min(target), np.max(target)
    if t_max - t_min < 1e-12:
        return np.nan
    gains = (target - t_min) / (t_max - t_min)
    order = np.argsort(-pred, kind='mergesort')
    discounts = 1.0 / np.log2(np.arange(2, n + 2))
    dcg = np.sum(gains[order] * discounts)
    idcg = np.sum(np.sort(gains)[::-1] * discounts)
    rdcg = np.mean(gains) * np.sum(discounts)
    if idcg - rdcg < 1e-12:
        return np.nan
    return (dcg - rdcg) / (idcg - rdcg)

rows = []
for K in [16, 32, 64, 128]:
    files = sorted(glob.glob(os.path.join(peft_dir, f'*_final_predictions_shot{K}.csv')))
    for f in files:
        assay = os.path.basename(f).replace(f'_final_predictions_shot{K}.csv', '')
        orig_f = f'{orig_dir}/{assay}.csv'
        if not os.path.exists(orig_f):
            continue
        orig_df = pd.read_csv(orig_f, usecols=['mutant', 'mutated_sequence'])
        df = pd.read_csv(f, usecols=['mutant', 'target', 'delta_esm', 'pred_my_method', 'pred_epro'])

        s_min, s_max = df['pred_my_method'].min(), df['pred_my_method'].max()
        e_min, e_max = df['pred_epro'].min(), df['pred_epro'].max()
        df['smite_norm'] = (df['pred_my_method'] - s_min) / (s_max - s_min + 1e-12)
        df['epro_norm'] = (df['pred_epro'] - e_min) / (e_max - e_min + 1e-12)
        df['pred_ens'] = 0.5 * (df['smite_norm'] + df['epro_norm'])

        train_muts = set(orig_df['mutant']) - set(df['mutant'])
        train_sites = {int(m[1:-1]) for m in train_muts}
        df['site'] = df['mutant'].apply(lambda m: int(m[1:-1]))
        df['visited'] = df['site'].isin(train_sites)

        vis = df[df['visited']]
        unvis = df[~df['visited']]
        plen = len(orig_df['mutated_sequence'].iloc[0])
        nsites = orig_df['mutant'].apply(lambda m: int(m[1:-1])).nunique()

        rows.append({
            'K': K,
            'assay': assay,
            'plen': plen,
            'nsites': nsites,
            'cov': K / nsites,
            'vis_rho_smite': safe_spearman(vis['pred_my_method'], vis['target']),
            'vis_rho_epro': safe_spearman(vis['pred_epro'], vis['target']),
            'vis_rho_ens': safe_spearman(vis['pred_ens'], vis['target']),
            'unvis_rho_smite': safe_spearman(unvis['pred_my_method'], unvis['target']),
            'unvis_rho_epro': safe_spearman(unvis['pred_epro'], unvis['target']),
            'unvis_rho_ens': safe_spearman(unvis['pred_ens'], unvis['target']),
            'all_rho_smite': safe_spearman(df['pred_my_method'], df['target']),
            'all_rho_epro': safe_spearman(df['pred_epro'], df['target']),
            'all_rho_ens': safe_spearman(df['pred_ens'], df['target']),
            'vis_ndcg_smite': calc_ndcg(vis['pred_my_method'], vis['target']),
            'vis_ndcg_epro': calc_ndcg(vis['pred_epro'], vis['target']),
            'vis_ndcg_ens': calc_ndcg(vis['pred_ens'], vis['target']),
            'unvis_ndcg_smite': calc_ndcg(unvis['pred_my_method'], unvis['target']),
            'unvis_ndcg_epro': calc_ndcg(unvis['pred_epro'], unvis['target']),
            'unvis_ndcg_ens': calc_ndcg(unvis['pred_ens'], unvis['target']),
            'all_ndcg_smite': calc_ndcg(df['pred_my_method'], df['target']),
            'all_ndcg_epro': calc_ndcg(df['pred_epro'], df['target']),
            'all_ndcg_ens': calc_ndcg(df['pred_ens'], df['target']),
        })

df_all = pd.DataFrame(rows)
df106 = df_all[(df_all['nsites'] >= 100) & (df_all['assay'] != 'A0A140D2T1_ZIKV_Sourisseau_2019')].copy()

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#222222'
plt.rcParams['axes.linewidth'] = 0.9

COLORS_BAR = {
    'SMITEpro + EVOLVEpro': '#6FAF8B',  # Balanced medium-soft sage green
    'SMITEpro': '#5B8DB8',              # Balanced medium-soft slate-sky blue
    'EVOLVEpro': '#EB9E4C',             # Balanced medium-soft warm amber-apricot
}
COLORS_EDGE = {
    'SMITEpro + EVOLVEpro': '#26593B',
    'SMITEpro': '#234B6E',
    'EVOLVEpro': '#9E5812',
}

methods = [
    ('SMITEpro + EVOLVEpro', 'ens'),
    ('SMITEpro', 'smite'),
    ('EVOLVEpro', 'epro'),
]

def make_fig2(metric_key, ylabel_B, ylim_B, out_stem):
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.2, 4.9), dpi=300, gridspec_kw={'width_ratios': [1.0, 1.32]})
    plt.subplots_adjust(wspace=0.23, left=0.075, right=0.985, top=0.92, bottom=0.17)

    # PANEL A: Unseen-Site Advantage vs. Protein Length (SMITEpro vs EVOLVEpro)
    u32 = df_all[df_all['K'] == 32].dropna(subset=['unvis_rho_smite', 'unvis_rho_epro']).copy()
    u32['diff_unvis'] = u32['unvis_rho_smite'] - u32['unvis_rho_epro']

    pt_colors = ['#377EB8' if d >= 0 else '#FFBF69' for d in u32['diff_unvis']]
    axA.axhline(0, color='#D62828', linestyle='--', linewidth=1.2, alpha=0.85, zorder=2)
    axA.scatter(
        u32['plen'],
        u32['diff_unvis'],
        c=pt_colors,
        s=38,
        alpha=0.85,
        edgecolors='#333333',
        linewidths=0.6,
        zorder=3
    )

    log_x = np.log10(u32['plen'])
    slope, intercept = np.polyfit(log_x, u32['diff_unvis'], 1)
    x_line = np.logspace(np.log10(u32['plen'].min()), np.log10(u32['plen'].max()), 200)
    y_line = slope * np.log10(x_line) + intercept
    r_val, p_val = spearmanr(u32['plen'], u32['diff_unvis'])

    axA.plot(x_line, y_line, color='#1D4E89', linestyle='-', linewidth=2.1, zorder=4,
             label=f'Log-linear fit (ρ = {r_val:.2f}, p = {p_val:.1e})')

    axA.set_xscale('log')
    axA.set_xlim(30, 2200)
    axA.set_ylim(-0.62, 0.62)
    axA.set_xticks([50, 100, 250, 500, 1000, 2000])
    axA.get_xaxis().set_major_formatter(mtick.ScalarFormatter())
    axA.tick_params(axis='both', labelsize=10.0)

    axA.set_xlabel("Protein Sequence Length (Amino Acids)", fontsize=11.0)
    axA.set_ylabel("Unseen-Site Δ Spearman's ρ\n(SMITEpro − EVOLVEpro)", fontsize=11.0)
    axA.grid(True, linestyle=':', alpha=0.5, zorder=0)
    axA.spines['top'].set_visible(False)
    axA.spines['right'].set_visible(False)
    axA.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=9.2)
    axA.text(-0.14, 1.03, 'A', transform=axA.transAxes, fontsize=14, fontweight='normal', va='bottom', ha='left')

    # PANEL B: Unseen Sites (K=16, 32, 64) -> Seen Sites (K=64) -> All Held-Out (K=64)
    sub_k16 = df106[df106['K'] == 16]
    sub_k32 = df106[df106['K'] == 32]
    sub_k64 = df106[df106['K'] == 64]

    groups = [
        ('Unseen Sites\n(K = 16)', sub_k16.rename(columns={f'unvis_{metric_key}_ens': 'ens', f'unvis_{metric_key}_smite': 'smite', f'unvis_{metric_key}_epro': 'epro'})),
        ('Unseen Sites\n(K = 32)', sub_k32.rename(columns={f'unvis_{metric_key}_ens': 'ens', f'unvis_{metric_key}_smite': 'smite', f'unvis_{metric_key}_epro': 'epro'})),
        ('Unseen Sites\n(K = 64)', sub_k64.rename(columns={f'unvis_{metric_key}_ens': 'ens', f'unvis_{metric_key}_smite': 'smite', f'unvis_{metric_key}_epro': 'epro'})),
        ('Seen Sites\n(K = 64)', sub_k64.rename(columns={f'vis_{metric_key}_ens': 'ens', f'vis_{metric_key}_smite': 'smite', f'vis_{metric_key}_epro': 'epro'})),
        ('All Held-Out\n(K = 64)', sub_k64.rename(columns={f'all_{metric_key}_ens': 'ens', f'all_{metric_key}_smite': 'smite', f'all_{metric_key}_epro': 'epro'})),
    ]

    x = np.arange(len(groups))
    width = 0.24
    offsets = [-width, 0.0, width]

    axB.axvline(2.5, color='#666666', linestyle='--', linewidth=1.0, zorder=1)
    axB.axvline(3.5, color='#666666', linestyle='--', linewidth=1.0, zorder=1)

    for (m_label, m_col), offset in zip(methods, offsets):
        means = []
        sems = []
        for _, gdf in groups:
            vals = gdf[m_col].dropna().to_numpy()
            means.append(np.mean(vals))
            sems.append(np.std(vals, ddof=1) / np.sqrt(len(vals)))

        axB.bar(
            x + offset,
            means,
            width=width,
            label=m_label,
            color=COLORS_BAR[m_label],
            edgecolor=COLORS_EDGE[m_label],
            linewidth=0.95,
            alpha=0.95,
            yerr=sems,
            capsize=3,
            error_kw={'elinewidth': 1.0, 'ecolor': '#2B2B2B'},
            zorder=3
        )

    axB.set_xticks(x)
    axB.set_xticklabels([g[0] for g in groups], fontsize=9.4)
    axB.set_ylabel(ylabel_B, fontsize=11.0)
    axB.set_ylim(0, ylim_B)
    axB.set_xlim(-0.52, len(groups) - 0.48)
    axB.tick_params(axis='y', labelsize=10.0)
    axB.grid(axis='y', linestyle=':', alpha=0.5, zorder=0)
    axB.spines['top'].set_visible(False)
    axB.spines['right'].set_visible(False)
    axB.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=9.2)
    axB.text(-0.11, 1.03, 'B', transform=axB.transAxes, fontsize=14, fontweight='normal', va='bottom', ha='left')

    out_png = f'/stor/work/Wilke/alf3564/smitepro/manuscript/final_figures/{out_stem}.png'
    out_pdf = f'/stor/work/Wilke/alf3564/smitepro/manuscript/final_figures/{out_stem}.pdf'
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    print("Saved", out_png, "and", out_pdf)

def make_supp_fig(metric_key, ylabel_S, ylim_S, out_stem):
    figS, axesS = plt.subplots(1, 3, figsize=(13.5, 4.4), dpi=300, sharey=True)
    plt.subplots_adjust(wspace=0.08, left=0.065, right=0.985, top=0.88, bottom=0.16)

    panel_configs = [
        ('A', 'Unseen Sites (Absent from Training Set)', f'unvis_{metric_key}'),
        ('B', 'Seen Sites (Mutated in Training Set)', f'vis_{metric_key}'),
        ('C', 'All Held-Out Sites (Complete Library)', f'all_{metric_key}'),
    ]

    k_vals = [16, 32, 64, 128]
    x_k = np.arange(len(k_vals))
    width_s = 0.24
    offsets_s = [-width_s, 0.0, width_s]

    for ax_idx, (p_label, p_desc, prefix) in enumerate(panel_configs):
        ax = axesS[ax_idx]
        for (m_label, m_suffix), offset in zip(methods, offsets_s):
            col = f'{prefix}_{m_suffix}'
            means = []
            sems = []
            for kv in k_vals:
                vals = df106[df106['K'] == kv][col].dropna().to_numpy()
                means.append(np.mean(vals))
                sems.append(np.std(vals, ddof=1) / np.sqrt(len(vals)))
            ax.bar(
                x_k + offset,
                means,
                width=width_s,
                label=m_label if ax_idx == 0 else None,
                color=COLORS_BAR[m_label],
                edgecolor=COLORS_EDGE[m_label],
                linewidth=0.95,
                alpha=0.95,
                yerr=sems,
                capsize=3,
                error_kw={'elinewidth': 1.0, 'ecolor': '#2B2B2B'},
                zorder=3
            )
        ax.set_xticks(x_k)
        ax.set_xticklabels([f'K = {kv}' for kv in k_vals], fontsize=10.0)
        ax.set_xlabel(p_desc, fontsize=10.8, labelpad=8)
        ax.set_ylim(0, ylim_S)
        ax.set_xlim(-0.52, len(k_vals) - 0.48)
        ax.tick_params(axis='y', labelsize=10.0)
        ax.grid(axis='y', linestyle=':', alpha=0.5, zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(-0.08 if ax_idx > 0 else -0.14, 1.03, p_label, transform=ax.transAxes,
                fontsize=14, fontweight='normal', va='bottom', ha='left')

    axesS[0].set_ylabel(ylabel_S, fontsize=11.0)
    axesS[0].legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=9.2)

    supp_png = f'/stor/work/Wilke/alf3564/smitepro/manuscript/final_figures/{out_stem}.png'
    supp_pdf = f'/stor/work/Wilke/alf3564/smitepro/manuscript/final_figures/{out_stem}.pdf'
    figS.savefig(supp_png, dpi=300)
    figS.savefig(supp_pdf)
    plt.close(figS)
    print("Saved", supp_png, "and", supp_pdf)

# Save NDCG versions
make_fig2('ndcg', "Held-Out NDCG (Mean ± s.e.m.)", 0.38, 'fig2_ndcg')
make_supp_fig('ndcg', "Held-Out NDCG (Mean ± s.e.m.)", 0.42, 'supp_site_scaling_k_ndcg')

# Save Spearman versions (both as fig2_spearman and as default fig2)
make_fig2('rho', "Held-Out Spearman's ρ (Mean ± s.e.m.)", 0.34, 'fig2_spearman')
make_fig2('rho', "Held-Out Spearman's ρ (Mean ± s.e.m.)", 0.34, 'fig2')
make_supp_fig('rho', "Held-Out Spearman's ρ (Mean ± s.e.m.)", 0.36, 'supp_site_scaling_k_spearman')
make_supp_fig('rho', "Held-Out Spearman's ρ (Mean ± s.e.m.)", 0.36, 'supp_site_scaling_k')
