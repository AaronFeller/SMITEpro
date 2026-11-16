# SMITEpro Manuscript Narrative & Benchmark Organization Guide

This document summarizes the manuscript structure, figure lineup, and benchmark story for the *Nature Machine Intelligence* submission (`manuscript/manuscript.tex`).

---

## 1. Final 5-Figure Lineup (`manuscript/final_figures/`)

1. **Figure 1 (`final_figures/fig1_diagram.png`) — Section 2.1**:
   - Conceptual overview of the SMITEpro framework: masking the mutated site, extracting the site-level amino acid probability distribution from ESMC-300M, applying LoRA + squared-target softplus margin-ranking loss ($\mathcal{L}_{\text{rank}} - \beta \mathcal{H}$) with 15% contextual token dropout, and scoring the unscreened variant library via masked marginal log-odds $\log(P(\text{mut})/P(\text{WT}))$.
   - Supported by the 34-assay developmental loss ablation table in Appendix A (`Table A1`).

2. **Figure 2 (`final_figures/fig2_spearman.png` / `.pdf` [Spearman] and `final_figures/fig2_ndcg.png` / `.pdf` [NDCG]) — Section 2.2**:
   - **Direct site-level evaluation at Unseen vs. Seen sequence positions**:
     - **Panel A**: Unseen-site $\Delta\text{Spearman's }\rho$ ($\text{SMITEpro} - \text{EVOLVEpro}$ at $K = 32$) plotted against protein sequence length $L$ (log scale, $n = 197$ assays), demonstrating that SMITEpro's advantage on unseen sites grows with protein length ($\rho = 0.45$, $p = 4.1 \times 10^{-11}$) as $1/L$ mean pooling dilutes single-residue changes in full-length proteins.
     - **Panel B**: Held-out **Spearman's $\rho$** (`fig2_spearman.png`, with chance-adjusted **NDCG** also saved in `fig2_ndcg.png`) across the 106 benchmark assays ($\ge 100$ mutated positions) partitioned into **Unseen Sites** ($K = 16, 32,$ and $64$), **Seen Sites ($K = 64$)**, and **All Held-Out ($K = 64$)** for `SMITEpro + EVOLVEpro`, `SMITEpro`, and `EVOLVEpro` (with full $K \in \{16, 32, 64, 128\}$ scaling across all three site partitions provided in Supplementary Figure `final_figures/supp_site_scaling_k.png` / `.pdf` [Spearman] and `final_figures/supp_site_scaling_k_ndcg.png` / `.pdf` [NDCG], Appendix B).
       - Shows that on **Unseen Sites**, `EVOLVEpro` drops sharply at low $K$ ($\rho = 0.08$ at $K=16$ and $0.10$ at $K=32$), whereas `SMITEpro` retains strong ranking accuracy ($\rho = 0.21$ at $K=16$, $p = 4.9 \times 10^{-10}$; $0.18$ at $K=32$, $p = 2.5 \times 10^{-4}$; $0.15$ vs. $0.13$ at $K=64$).
       - Shows that on **Seen Sites ($K = 64$)** and **All Held-Out ($K = 64$)**, standalone `SMITEpro` ($\rho = 0.24$ on both) outperforms `EVOLVEpro` ($\rho = 0.20$ on seen sites, $0.15$ on all held-out), while the `SMITEpro + EVOLVEpro` ensemble achieves the highest correlation ($\rho = 0.27$ on seen sites and $\rho = 0.25$ across all held-out sites at $K=64$).

3. **Figure 3 (`final_figures/fig3_enrichment.png` / `.pdf`) — Section 2.3**:
   - **Comprehensive Top-$N$ Discovery Benchmark across 106 ProteinGym DMS Assays** ($\ge 100$ mutated positions, $K \in \{16, 32, 64, 128\}$, $N \in \{10, 20, 30, 40\}$):
     - **Panels A & C (Left)**: Grouped bar comparisons ($\text{Mean} \pm \text{s.e.m.}$ across 106 assays) at the experimental screening budget $N = 20$ across training set sizes $K \in \{16, 32, 64, 128\}$ for Top 1% Enrichment (A) and Mean Selected Fitness (C).
     - **Panels B & D (Right)**: Screening budget scaling trajectories across $N \in \{10, 20, 30, 40\}$ ($\text{Mean} \pm \text{s.e.m.}$) for Top 1% Enrichment (B) and Mean Selected Fitness (D), with horizontal dotted lines marking the budget-invariant random expectation.
     - Compares 7 methods across 32 $(K, N, \text{metric})$ settings:
       1. **`SMITEpro + EVOLVEpro` (Ensemble)**: Mean rank **1.03** (Rank #1 overall)
       2. **`SMITEpro`**: Mean rank **2.12** (Rank #2 overall, #1 standalone method)
       3. **`EVOLVEpro`** & **`Gaussian Process` (`Kermut`)**: Tied at mean rank **4.12**
       5. **`MSA PSSM`**: Mean rank **4.31**
       6. **`Zero-shot ESMC`**: Mean rank **5.28**
       7. **`Random`**: Mean rank **7.00**

4. **Figure 4 (`final_figures/fig4_hpab_experiment.png` / `.pdf`) — Section 2.4**:
   - **Prospective 6-Round Wet-Lab Evolution of HpaBC for L-DOPA Biosynthesis**:
     - Displays the 119 experimentally tested candidates across Rounds 1–6 on WT-normalized linear axes using clean single-plate biological triplicates from `manuscript/HpaBC_values_by_plate.xlsx` ($n=3$ for 112/119 variants).
     - Demonstrates a **42.49-fold** improvement over wild-type HpaBC in Round 6 (`I29V+A199P+D354K+H454Q+H372P+A8R`), with 28/119 (23.5%) exceeding their plate-matched parental control and 67/119 (56.3%) exceeding wild-type HpaBC.

5. **Figure 5 (`final_figures/fig5_hpab_stability.png` / `.pdf`) — Section 2.5**:
   - **Post-Hoc Thermodynamic Stability Analysis (`StabilityArc`) Across Six HpaB Backgrounds**:
     - Shows that 38 of 41 (92.7%) predicted destabilizing substitutions ($\text{score} < 0$) on evolved backgrounds (Rounds 2–6) failed to improve over their parental control, with positive Spearman correlations ($\rho = 0.28\text{--}0.66$) on all five evolved backgrounds vs. $\rho = -0.02$ on the buffered wild-type enzyme.
