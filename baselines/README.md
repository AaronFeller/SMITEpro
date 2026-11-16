# Single-round manuscript baselines

The authoritative run is
`proteinGym/singleMutant/2026-02-27_softSqLoss_16batch_1e-3_cosineannealing_PEFT`,
as identified by its `01_table_analysis.ipynb`. Training observations are recovered
by subtracting the saved unscreened prediction IDs from each original singles-only
assay. No fresh sampling, iterative rounds, or label-guided selection occurs.
Every method consumes the same immutable, checksum-validated manifest for
K = 16, 32, 64, 128. Historical targets are checked against the original midpoint
normalization before a manifest is accepted.

## Implementations

- `protocol.py`: CSV loading, historical target normalization, split recovery and
  manifests. Novel targets have a separate explicit zero-shot position-diverse
  selector; it is never used to replace missing manuscript results.
- `models.py`: Kermut's actual `SequenceKernel` (RBF), PyTorch/GPyTorch exact GP,
  trained on `target - delta_esm` and predicting `delta_esm + posterior residual`.
  This is a fixed zero-shot prior mean, not a learned linear mean. No ProteinMPNN,
  coordinates, structural kernel, or SMITEpro weights are used or modified.
- `features.py`: frozen ESMC-300M embeddings, benchmark-compatible token mean
  pooling and sequence-keyed NPZ caching. No cache files were found in the previous
  benchmark directories; its script computed embeddings in memory. Cached ESMC
  model weights do exist, but model weights are not embedding caches.
- MLDE is a modern classical adapter using the same sklearn
  `RandomForestRegressor(n_estimators=100)` backend as the cloned MLDE RF wrapper,
  with standard full-sequence 20-AA one-hot features. It does not import MLDE's
  obsolete TensorFlow/Keras orchestration or run its ensemble/model-selection
  workflow. Sparse storage avoids dense full-length one-hot matrices.
- Random assigns a seeded random permutation of unscreened variants. Repeat
  seeds for a null distribution rather than interpreting one random realization.
- `evolution.py`: site-independent PSSM log-probability differences, summing
  substitution scores for multis. Removes A3M lowercase insertions and dots,
  projects query-gap columns out, deduplicates homologs, validates the reference,
  and applies a positive pseudocount per AA. This is not EVmutation or an
  epistatic model. Cropped alignments report per-variant coverage; uncovered sites
  contribute zero, with warnings. Do not compare coverage-filtered variants to a
  different evaluation pool.

## Setup And Smoke Test

From the repository root:

```bash
bash baselines/setup.sh
bash baselines/run.sh prepare
bash baselines/run.sh smoke
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest baselines.test_protocol -v
```

Setup installs only pinned GP dependencies into `baselines/.deps`, reusing the
existing `.venv` torch/esm/numpy/pandas/scipy/sklearn/requests packages. It does not
upgrade that environment. `PYTHON=/path/to/python` overrides the interpreter.

The smoke run uses the real SDA assay and its exact February K=16 selection,
20 unscreened observations, a deterministic random encoder, and a toy A3M. It
trains the real GP for five optimization steps and the real RF, then executes
inference, metrics and logging for all four methods. No ESMC inference or network
query occurs. Synthetic encodings are restricted to this CPU smoke configuration.
If a complete matching default ESMC cache exists, smoke reuses it instead; an
explicit `--embedding-cache PATH` also uses cached features without model inference.
`--msa PATH --msa-start 1` can replace the toy alignment with a genuine local MSA.

Run just one smoke adapter with `bash baselines/run.sh smoke --method kermut`
(other choices: `mlde`, `random`, `msa`). Outputs are under
`baselines/results/smoke/SDA_BACSU_Tsuboyama_2023_1PV0/`, including per-method CSVs,
JSON metadata, GP objective history, and `run.log`. Synthetic-resource flags and
`smoke_test` metadata distinguish these checks from publishable predictions.

The verified audit contains 208 complete assays, five partial assays and four
without February outputs. `baselines/splits/preparation_report.json` records exact
availability. Missing budgets are never silently substituted. Run only the
available budgets listed for a partial assay.

## Direct GPU Runs

Edit `config.json` for paths, threads and hyperparameters. Run directly, without
Slurm, for example:

```bash
bash baselines/run.sh run \
  --assay datasets/DMS_ProteinGym_substitutions/cv_folds_singles_substitutions/SDA_BACSU_Tsuboyama_2023_1PV0.csv \
  --shots 16 32 64 128 --method kermut --device cuda:1
```

The same command with `--method all` runs every baseline, provided a local
ProteinGym MSA and reference mapping are installed. RF, Random and PSSM remain
CPU algorithms; GPU accelerates ESMC extraction and GP fitting, not prediction
quality inherently. Uncached ESMC extraction refuses a CPU fallback. Features are
cached in `baselines/cache/esmc/<assay>.npz` and reused across shot budgets. Do not
run simultaneous writers against the same embedding cache or output directory.
Use distinct `--output-dir` values for different `--seed` runs.

An existing cache can be supplied with `--embedding-cache PATH`. The NPZ schema
is `sequences` (unique strings), `embeddings` (row-aligned vectors), scalar
`model='esmc_300m'` and scalar `pooling='benchmark_token_mean'`. An unkeyed matrix
must not be reused by presumed CSV order. Feature normalization is fit on the
K training embeddings only. The GP prior requires all selected and test mutants
in the `delta_esm` column of the zero-shot CSV (`--zero-shot PATH` overrides it).

## Alignment Resources

ProteinGym targets use local resources, with no automatic API fallback. Set
`msa.reference_csv` and `msa.directory` in `config.json`. Reference fields are
`DMS_id`, `MSA_filename`, and 1-based `MSA_start`. ProteinGym provides the actual
alignments as its `DMS_msa_files.zip` resource, rather than checking them into Git.
The archive is about 5.2 GB uncompressed and was not downloaded during smoke tests.
Resource acquisition commands, to run when needed:

```bash
mkdir -p datasets/ProteinGym_reference_files datasets/ProteinGym_MSAs
curl -fL --retry 3 \
  https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/reference_files/DMS_substitutions.csv \
  -o datasets/ProteinGym_reference_files/DMS_substitutions.csv
curl -fL --retry 3 \
  https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_msa_files.zip \
  -o datasets/DMS_msa_files.zip
unzip datasets/DMS_msa_files.zip -d datasets/ProteinGym_MSAs
```

If extraction creates a nested directory, set `msa.directory` to the directory
that directly contains the reference's MSA filenames. `--msa PATH` can bypass the
mapping for an explicitly supplied precomputed alignment.

For novel targets, first freeze a preset from a cached zero-shot CSV containing
`mutant`, `score` (selection ranking) and `delta_esm` (GP prior). Then opt in to the
MMseqs2 API, which transmits the reference protein sequence externally:

```bash
bash baselines/run.sh prepare --novel --assay /path/novel.csv \
  --zero-shot /path/novel_zero_shot.csv --manifest /path/novel_split.json
bash baselines/run.sh run --novel --assay /path/novel.csv \
  --manifest /path/novel_split.json --zero-shot /path/novel_zero_shot.csv \
  --shots 16 32 64 128 --device cuda:1
```

MSAs are cached in `baselines/cache/msas`; cached query identity is checked.
API ticket handling and tar/A3M parsing were tested with a mocked response, not a
live submission. GPU ESMC inference was not run during this task.

## Output Compatibility

Each method writes `<assay>_final_predictions_shot<K>.csv` with original assay
fields, `target`, `pred_score`, `pred_<method>`, `is_train=False`, split checksum,
seed and resource provenance. Combined prediction files retain the February
SMITEpro/EVOLVEpro/zero-shot columns alongside the new scores. Long-form
`evaluation_metrics.csv` has the notebook's `system, train_size, method, metric,
value, N, num_sites` columns, so it can be concatenated with existing metric tables
without labeling the new baselines as SMITEpro or EVOLVEpro.

Ground truth is defined over each K's unscreened library, matching the notebook:
Top-1% uses the 99th percentile including ties; Top-5 Hit Rate is recall of the
true top five including ties; Normalized Max Score is
`max(0, (best_found - library_median)/(library_max - library_median))`.
The notebook's Top-10 Recall is also retained, alongside enrichment, mean fitness
and NDCG. N is 10, 20, 30, or 40. When fewer than N test variants exist, values are
NA with `budget_supported=False`; the 20-test smoke does not claim N=30/40 results.
Apply the notebook's `num_sites >= 100` filter when assembling the manuscript
figure. Changing a Top-10 panel to Top-5 requires an explicit figure update, not
merely relabeling the old measurements.