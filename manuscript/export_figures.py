import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch


ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / "proteinGym/singleMutant/2026-02-27_softSqLoss_16batch_1e-3_cosineannealing_PEFT"
FIGURES = ROOT / "manuscript/figures"


def save_figure(figure, name):
    for extension in ("pdf", "png"):
        figure.savefig(FIGURES / f"{name}.{extension}", dpi=300, bbox_inches="tight")


def workflow():
    figure, axis = plt.subplots(figsize=(11, 3.8))
    axis.set_xlim(0, 11)
    axis.set_ylim(0, 4)
    axis.axis("off")
    stages = [
        (1.3, 2, "Fixed initial screen\nK = 16, 32, 64, 128\nmeasured variants", "#f0f0f0"),
        (4.1, 3, "SMITEpro\nMasked-site log-odds\nLoRA + sequence head", "#e0eef0"),
        (4.1, 1, "EVOLVEpro comparator\nFrozen pooled embeddings\nRandom forest", "#f8e9d8"),
        (7.2, 2, "Same held-out pool\nScale each prediction\nEqual-weight ensemble", "#e4efdf"),
        (10, 2, "Select top N\nN = 10, 20, 30, 40\nEvaluate assay outcomes", "#f0f0f0"),
    ]
    for horizontal, vertical, text, color in stages:
        axis.text(horizontal, vertical, text, ha="center", va="center", fontsize=10,
                  bbox={"boxstyle": "square,pad=0.7", "facecolor": color, "edgecolor": "#555555"})
    for start, end in [((2.4, 2.2), (2.85, 2.9)), ((2.4, 1.8), (2.85, 1.1)),
                       ((5.35, 2.9), (5.85, 2.2)), ((5.35, 1.1), (5.85, 1.8)),
                       ((8.45, 2), (8.7, 2))]:
        axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, color="#555555"))
    save_figure(figure, "site_masked_workflow")
    plt.close(figure)


def benchmark_curves():
    notebook = json.loads((HISTORY / "01_table_analysis.ipynb").read_text())
    matches = [
        "".join(cell.get("source", [])) for cell in notebook["cells"]
        if "comparison_figure, comparison_axes" in "".join(cell.get("source", []))
        and "matched_metrics" in "".join(cell.get("source", []))
    ]
    if len(matches) != 1:
        raise ValueError("Expected one authoritative matched-comparison notebook cell")
    namespace = {"display": lambda *args, **kwargs: None}
    exec(compile(matches[0], str(HISTORY / "01_table_analysis.ipynb") + ":matched_comparison", "exec"), namespace)
    matched = namespace["matched_metrics"]
    counts = matched.groupby("train_size").assay.nunique().to_dict()
    if counts != {16: 107, 32: 106, 64: 106, 128: 106}:
        raise ValueError(f"Unexpected benchmark coverage: {counts}")
    figure = namespace["comparison_figure"]
    save_figure(figure, "benchmark_curves")
    plt.close(figure)


def paired_differences():
    fitness = pd.read_csv(HISTORY / "mean_top_n_ensemble_head_to_head.csv").set_index("comparator")
    precision = pd.read_csv(HISTORY / "top_1pct_precision_ensemble_head_to_head.csv").set_index("comparator")
    order = fitness.index.tolist()
    precision = precision.loc[order]
    if not fitness.assays.eq(106).all() or not precision.assays.eq(106).all():
        raise ValueError("Paired comparison tables do not contain the complete 106-assay cohort")
    names = {
        "MLDE One-Hot RF": "One-hot RF", "Kermut ESMC Sequence GP": "Kermut ESMC sequence GP",
        "Zero-Shot ESM": "Zero-shot ESMC",
    }
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharey=True, layout="constrained")
    for axis, frame, column, lower, upper, title, label, color in [
        (axes[0], fitness, "mean_delta", "ci_low", "ci_high", "A  Mean selected fitness", "Difference in normalized fitness", "#007c91"),
        (axes[1], precision, "mean_delta_pp", "ci_low_pp", "ci_high_pp", "B  Top 1% Precision", "Difference in percentage points", "#c65032"),
    ]:
        positions = np.arange(len(order))
        means = frame[column].to_numpy()
        intervals = np.vstack([means - frame[lower].to_numpy(), frame[upper].to_numpy() - means])
        axis.errorbar(means, positions, xerr=intervals, fmt="o", color=color, capsize=4)
        axis.axvline(0, color="#777777", linestyle="--", linewidth=1)
        axis.set_title(title, loc="left", fontsize=12)
        axis.set_xlabel(label)
        axis.set_yticks(positions, [names.get(method, method) for method in order])
        axis.grid(axis="x", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].invert_yaxis()
    save_figure(figure, "benchmark_paired_differences")
    plt.close(figure)


def main():
    FIGURES.mkdir(exist_ok=True)
    workflow()
    benchmark_curves()
    paired_differences()
    print("Exported three publication figures as PDF and PNG; no model inference performed.")


if __name__ == "__main__":
    main()