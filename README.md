# SMITEpro: Site-Masked Inference Tuning for Engineering Proteins

Code, evaluation harnesses, figure generation scripts, and processed benchmark/experimental datasets accompanying the paper:

> **Guiding low-$N$ protein evolution by uniting site-masked inference tuning in SMITEpro with sequence-level predictors**  
> Aaron L. Feller, Isa Madrigal Harrison, Phillip R. Woolley, Daniel Gutierrez, Andrew D. Ellington, and Claus O. Wilke

---

## Overview

**SMITEpro** adapts a pretrained protein language model (`ESMC-300M`) to a target functional assay in the low-$N$ regime ($K \in \{16, 32, 64, 128\}$ variants) by masking the mutated residue position and fine-tuning the site-level amino acid probability distribution using low-rank adaptation (`LoRA`) and a squared-target dynamic-margin ranking loss with Shannon entropy regularization and 15% contextual token dropout:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{rank}} - \beta \mathcal{H}$$

Candidate mutations are scored directly from the masked marginal log-odds ratio at the mutated site:

$$\Delta_i = \sum_{k \in M_i} \left( \log P(\text{mut}_{i,k} \mid \tilde{x}_i) - \log P(\text{WT}_{i,k} \mid \tilde{x}_i) \right)$$

Pairing site-specific **SMITEpro** ranking with sequence-level **EVOLVEpro** regression (Random Forest on mean-pooled ESMC embeddings) in an equal-weight min–max normalized ensemble (**SMITEpro + EVOLVEpro**) combines zero-shot-informed extrapolation at unseen sequence positions with local interpolation at previously sampled positions.

---

## Repository Structure

```text
SMITEpro_github_repo/
├── proteinGym/
│   ├── rebuild_eval_pipeline_PEFT.py              # Core SMITEpro + EVOLVEpro ProteinGym benchmark pipeline
│   ├── rebuild_eval_pipeline_PEFT_contiguous.py   # Contiguous 5-fold CV benchmark pipeline
│   └── singleMutant/2026-02-27_softSqLoss_16batch_1e-3_cosineannealing_PEFT/
│       ├── 01_table_analysis.ipynb                # Benchmark aggregation & statistical analysis notebook
│       ├── 01_main_results_for_man.ipynb          # Summary results notebook
│       ├── plot_fig2_new_sites.py                 # Generates Figure 2 and Supplementary Figure S1
│       ├── plot_fig3_benchmark.py                 # Generates Figure 3
│       ├── fig2_site_partition_metrics.csv        # Precomputed unseen/seen site correlations across assays
│       ├── benchmark_106_assays_per_assay_7methods.csv  # Precomputed 106-assay Top-N discovery metrics
│       └── evaluation_metrics_combined.csv        # Combined evaluation metrics across ProteinGym assays
├── baselines/                                     # Matched-split comparator harness (GP/Kermut, MSA PSSM, MLDE, Random)
│   ├── README.md                                  # Detailed documentation for running baseline comparators
│   ├── protocol.py                                # Checksum-validated split recovery & midpoint target scaling
│   ├── models.py                                  # Sequence-kernel GP, one-hot RF, and random baselines
│   ├── evolution.py                               # Alignment-based MSA PSSM baseline
│   ├── features.py                                # Frozen ESMC-300M token mean-pooled embedding extractor
│   ├── ensembles.py                               # Equal-weight min-max ensemble evaluation
│   ├── metrics.py                                 # Top-1% enrichment, precision, mean fitness, and NDCG metrics
│   ├── run.py / run.sh / setup.sh                 # CLI entry points and smoke-test harness
│   └── splits/                                    # Saved K = 16, 32, 64, 128 training split manifests
├── internal_data/                                 # Prospective wet-lab campaign scripts & HpaBC data
│   ├── internal_peft_ensemble.py                  # Standalone SMITEpro + EVOLVEpro ensemble runner for custom proteins
│   ├── internal_peft_multi_background.py          # Multi-background ensemble runner used across HpaBC rounds
│   ├── internal_new.py                            # Core masked-marginal scoring & composite loss utilities
│   ├── enumerate_substeps.py                      # Mutational substep enumeration for multi-mutant trajectories
│   ├── run_internal_peft_ensemble.sh              # Shell launcher for custom experimental datasets
│   └── Dan_data/
│       ├── hpab_stabilityarc_posthoc.py           # Post-hoc StabilityArc scoring & analysis script
│       ├── round1/HpaBC Data/Fastas/HpaB_WT.fasta # Wild-type E. coli HpaB reference sequence (520 aa)
│       ├── round2/                                # HpaB calibration data & multi-round prediction launchers
│       ├── experimental_results/                  # Plate-normalized HpaBC biosensor data & Figure 4/5 script
│       │   ├── plot_experimental_results.py       # Generates Figure 4 (HpaBC rounds) & Figure 5 (StabilityArc)
│       │   ├── HpaBC_all_variants_replicates.xlsx # Raw biological replicate measurements across Rounds 1-6
│       │   └── HpaBC_variants_prism.xlsx          # Curated plate-matched summary workbook
│       └── stabilityarc_posthoc/
│           └── tested_mutation_predictions.csv    # Saved StabilityArc scores for all 119 tested HpaB variants
├── manuscript/
│   ├── HpaBC_values_by_plate.xlsx                 # Plate-organized HpaBC biological triplicates (Rounds 1-6)
│   └── export_figures.py                          # Additional figure export helper
├── SMITEpro_table_analysis_v4.sh                  # Batch launcher for full ProteinGym K-shot benchmark
├── pyproject.toml / uv.lock                       # Python environment & dependency specifications
└── LICENSE.txt                                    # MIT License
```

---

## Quick Start

### 1. Environment Setup

Requires Python $\ge 3.12$ and PyTorch with CUDA support for `ESMC-300M` training/inference. Using [`uv`](https://github.com/astral-sh/uv) or `pip`:

```bash
uv sync
# or with pip:
pip install -e .
```

### 2. Running the SMITEpro + EVOLVEpro Ensemble on a Custom Dataset

To train SMITEpro (LoRA + site-masked ranking loss) and EVOLVEpro (Random Forest on frozen mean-pooled ESMC embeddings) on your own experimental dataset and rank all unscreened single-amino-acid substitutions (or a custom candidate CSV):

```bash
bash internal_data/run_internal_peft_ensemble.sh \
    path/to/measured_variants.csv \
    YourProteinName \
    output/your_protein_run
```

**Input CSV columns:**
- `wt_sequence`: Wild-type (or current parental background) amino acid sequence ($\le 2046$ aa).
- `mutant`: Substitution string(s), e.g. `A199P` for single mutants or `I29V:A199P` for multi-mutants (1-based indexing).
- `target`: Normalized functional score (or provide `DMS_score` and `DMS_score_bin` for automatic midpoint scaling).

The script outputs `<protein>_allSingles_scores_peft_ensemble.csv` and `<protein>_perSite_scores_peft_ensemble.csv` containing:
- `pred_smitepro` (fine-tuned site-masked log-odds)
- `pred_evolvepro` (Random Forest regression on frozen ESMC embeddings)
- `pred_esmc_zero_shot` (pretrained zero-shot ESMC log-odds)
- `pred_ensemble` (equal-weight average of min–max normalized `SMITEpro` and `EVOLVEpro` scores)

---

## Reproducing Manuscript Figures from Included Data

All processed benchmark tables and HpaBC wet-lab measurements needed to regenerate the manuscript data figures (**Figures 2–5** and **Supplementary Figure S1**) are included directly in this repository—no GPU inference is required:

```bash
# Figure 2 (Unseen vs. Seen site Spearman correlation & length scaling) + Supplementary Figure S1
python proteinGym/singleMutant/2026-02-27_softSqLoss_16batch_1e-3_cosineannealing_PEFT/plot_fig2_new_sites.py

# Figure 3 (106-assay Top-N variant discovery across K = 16..128 and budgets N = 10..40)
python proteinGym/singleMutant/2026-02-27_softSqLoss_16batch_1e-3_cosineannealing_PEFT/plot_fig3_benchmark.py

# Figure 4 (6-round HpaBC wet-lab evolution) & Figure 5 (Post-hoc StabilityArc analysis)
python internal_data/Dan_data/experimental_results/plot_experimental_results.py \
    --output-dir manuscript/final_figures
```

---

## Re-running Full ProteinGym Benchmarks from Scratch

1. Download the ProteinGym single-substitution DMS assays (`cv_folds_singles_substitutions/*.csv`) from [proteingym.org](https://proteingym.org) into `datasets/DMS_ProteinGym_substitutions/cv_folds_singles_substitutions/`.
2. Launch the SMITEpro + EVOLVEpro benchmark across all assays:
   ```bash
   bash SMITEpro_table_analysis_v4.sh
   ```
3. To run the matched-split baseline comparators (Sequence-kernel Gaussian Process, MSA PSSM, One-Hot RF, and Random), see the instructions in [`baselines/README.md`](baselines/README.md):
   ```bash
   bash baselines/setup.sh
   bash baselines/run.sh smoke
   ```

---

## License

This project is licensed under the MIT License — see [`LICENSE.txt`](LICENSE.txt) for details.
