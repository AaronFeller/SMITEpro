from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


BACKBONES = [
    "L303H_A199P_I29V",
    "R0094H_S0210D_A0211R_Q0212K",
    "A199P_H454Q_H372P",
]

PREF_RUNS = [1, 2, 3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate triplicate per-site score runs into a consensus shortlist."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("internal_data/Dan_data/round2"),
        help="Directory containing predict_runs_enumerated_singlePref[1/2/3].",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=40,
        help="Number of top mutants to retain per backbone.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("internal_data/Dan_data/round2/triplicate_perSite_summary_requested_backgrounds"),
        help="Directory where summary CSVs will be written.",
    )
    parser.add_argument(
        "--protein",
        default="hpab",
        help="Protein prefix used in the per-site output filenames.",
    )
    parser.add_argument(
        "--backbone",
        action="append",
        default=None,
        help="Backbone directory to summarize. Repeat to provide multiple backbones.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def safe_float(value: str) -> float:
    return float(value) if value not in {"", None} else math.nan


def build_candidate_summary(rows_by_run: list[list[dict[str, str]]], top_n: int) -> list[dict[str, object]]:
    candidate_stats: dict[str, dict[str, object]] = defaultdict(
        lambda: {
            "sequence_name": "",
            "mutant": "",
            "support_runs": 0,
            "support_fraction": 0.0,
            "mean_rank": 0.0,
            "rank_sd": 0.0,
            "best_rank": 0,
            "mean_final_score": 0.0,
            "final_score_sd": 0.0,
            "mean_pred_ensemble": 0.0,
            "mean_pred_smitepro": 0.0,
            "mean_pred_esmc_zero_shot": 0.0,
            "mean_esmc_zero_shot_entropy": 0.0,
            "mean_esmc_zero_shot_mut_logp": 0.0,
            "mean_esmc_zero_shot_wt_logp": 0.0,
            "mean_pred_evolvepro": 0.0,
            "priority_score": 0.0,
        }
    )

    rank_lists: dict[str, list[int]] = defaultdict(list)
    final_score_lists: dict[str, list[float]] = defaultdict(list)
    pred_ensemble_lists: dict[str, list[float]] = defaultdict(list)
    pred_smitepro_lists: dict[str, list[float]] = defaultdict(list)
    pred_esmc_zero_shot_lists: dict[str, list[float]] = defaultdict(list)
    esmc_zero_shot_entropy_lists: dict[str, list[float]] = defaultdict(list)
    esmc_zero_shot_mut_logp_lists: dict[str, list[float]] = defaultdict(list)
    esmc_zero_shot_wt_logp_lists: dict[str, list[float]] = defaultdict(list)
    pred_evolvepro_lists: dict[str, list[float]] = defaultdict(list)

    for rows in rows_by_run:
        for rank, row in enumerate(rows[:top_n], start=1):
            mutant = row["mutant"]
            stats = candidate_stats[mutant]
            stats["sequence_name"] = row["sequence_name"]
            stats["mutant"] = mutant
            stats["support_runs"] = int(stats["support_runs"]) + 1

            rank_lists[mutant].append(rank)
            final_score_lists[mutant].append(safe_float(row["final_score"]))
            pred_ensemble_lists[mutant].append(safe_float(row["pred_ensemble"]))
            pred_smitepro_lists[mutant].append(safe_float(row["pred_smitepro"]))
            if "pred_esmc_zero_shot" in row:
                pred_esmc_zero_shot_lists[mutant].append(safe_float(row["pred_esmc_zero_shot"]))
            if "esmc_zero_shot_entropy" in row:
                esmc_zero_shot_entropy_lists[mutant].append(safe_float(row["esmc_zero_shot_entropy"]))
            if "esmc_zero_shot_mut_logp" in row:
                esmc_zero_shot_mut_logp_lists[mutant].append(safe_float(row["esmc_zero_shot_mut_logp"]))
            if "esmc_zero_shot_wt_logp" in row:
                esmc_zero_shot_wt_logp_lists[mutant].append(safe_float(row["esmc_zero_shot_wt_logp"]))
            pred_evolvepro_lists[mutant].append(safe_float(row["pred_evolvepro"]))

    for mutant, stats in candidate_stats.items():
        ranks = rank_lists[mutant]
        final_scores = final_score_lists[mutant]
        support_fraction = len(ranks) / len(rows_by_run)
        mean_rank = mean(ranks)
        rank_sd = pstdev(ranks) if len(ranks) > 1 else 0.0
        mean_final_score = mean(final_scores)
        final_score_sd = pstdev(final_scores) if len(final_scores) > 1 else 0.0

        rank_stability = 1.0 - min(rank_sd / max(top_n - 1, 1), 1.0)
        priority_score = 100.0 * (
            0.5 * support_fraction + 0.4 * mean_final_score + 0.1 * rank_stability
        )

        stats.update(
            {
                "support_fraction": round(support_fraction, 4),
                "mean_rank": round(mean_rank, 2),
                "rank_sd": round(rank_sd, 4),
                "best_rank": min(ranks),
                "mean_final_score": round(mean_final_score, 6),
                "final_score_sd": round(final_score_sd, 6),
                "mean_pred_ensemble": round(mean(pred_ensemble_lists[mutant]), 6),
                "mean_pred_smitepro": round(mean(pred_smitepro_lists[mutant]), 6),
                "mean_pred_esmc_zero_shot": round(mean(pred_esmc_zero_shot_lists[mutant]), 6),
                "mean_esmc_zero_shot_entropy": round(mean(esmc_zero_shot_entropy_lists[mutant]), 6),
                "mean_esmc_zero_shot_mut_logp": round(mean(esmc_zero_shot_mut_logp_lists[mutant]), 6),
                "mean_esmc_zero_shot_wt_logp": round(mean(esmc_zero_shot_wt_logp_lists[mutant]), 6),
                "mean_pred_evolvepro": round(mean(pred_evolvepro_lists[mutant]), 6),
                "priority_score": round(priority_score, 3),
            }
        )

    return sorted(
        candidate_stats.values(),
        key=lambda row: (
            -float(row["priority_score"]),
            -int(row["support_runs"]),
            float(row["mean_rank"]),
            -float(row["mean_final_score"]),
        ),
    )[:top_n]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def resolve_run_dir(base_dir: Path, pref: int) -> Path:
    requested_background_dirs = {
        1: base_dir / "predict_runs_requested_backgrounds",
        2: base_dir / "predict_runs_requested_backgrounds_seed2",
        3: base_dir / "predict_runs_requested_backgrounds_seed3",
    }
    requested_seed_dirs = {
        1: base_dir / "predict_runs_requested_backgrounds_seed1",
        2: base_dir / "predict_runs_requested_backgrounds_seed2",
        3: base_dir / "predict_runs_requested_backgrounds_seed3",
    }
    legacy_single_pref_dir = base_dir / f"predict_runs_enumerated_singlePref{pref}"

    if requested_seed_dirs[pref].exists():
        return requested_seed_dirs[pref]
    if requested_background_dirs[pref].exists():
        return requested_background_dirs[pref]
    if legacy_single_pref_dir.exists():
        return legacy_single_pref_dir

    raise FileNotFoundError(f"Could not resolve triplicate run directory for seed index {pref} under {base_dir}")


def summarize_backbone(base_dir: Path, backbone: str, top_n: int, protein: str) -> list[dict[str, object]]:
    rows_by_run = []
    for pref in PREF_RUNS:
        csv_path = (
            resolve_run_dir(base_dir, pref)
            / backbone
            / f"{protein}_perSite_scores_peft_ensemble.csv"
        )
        rows_by_run.append(read_rows(csv_path))
    return build_candidate_summary(rows_by_run, top_n)


def collect_diagnostics(base_dir: Path, backbone: str, top_n: int, protein: str) -> dict[str, int]:
    top_mutants = []
    for pref in PREF_RUNS:
        csv_path = (
            resolve_run_dir(base_dir, pref)
            / backbone
            / f"{protein}_perSite_scores_peft_ensemble.csv"
        )
        rows = read_rows(csv_path)
        top_mutants.append([row["mutant"] for row in rows[:top_n]])

    flattened = [mutant for run in top_mutants for mutant in run]
    unique_mutants = sorted(set(flattened))
    shared_by_all_runs = sum(
        all(mutant in run_mutants for run_mutants in top_mutants) for mutant in unique_mutants
    )
    return {
        "unique_top_n_union": len(unique_mutants),
        "shared_by_all_runs": shared_by_all_runs,
        "replicate_count": len(PREF_RUNS),
    }


def build_markdown_summary(
    backbone_rows: dict[str, list[dict[str, object]]],
    diagnostics: dict[str, dict[str, int]],
    top_n: int,
) -> str:
    lines = [
        "# Triplicate Per-Site Consensus Summary",
        "",
        f"Top-N cutoff: {top_n}",
        "",
        "Consensus score:",
        "priority_score = 100 x (0.5 x support_fraction + 0.4 x mean_final_score + 0.1 x rank_stability)",
        "",
        "Term definitions:",
        "support_fraction = fraction of triplicate runs where a mutant appears in the top-N list",
        "mean_final_score = average of the exported final_score across triplicates",
        "mean_pred_esmc_zero_shot = average of the exported pred_esmc_zero_shot across triplicates",
        "rank_stability = 1 - min(rank_sd / (top_n - 1), 1)",
        "",
        "This weighting makes recurrence across runs the dominant signal, keeps the model score visible, and lightly rewards stable ranking order across replicates.",
        "",
    ]

    for backbone, rows in backbone_rows.items():
        diag = diagnostics[backbone]
        lines.extend(
            [
                f"## {backbone}",
                "",
                f"Replicate diagnostic: {diag['shared_by_all_runs']} of {diag['unique_top_n_union']} unique top-{top_n} mutants were present in all {diag['replicate_count']} runs.",
                "",
                "Top 5 mutants:",
            ]
        )
        for row in rows[:5]:
            lines.append(
                f"- {row['mutant']}: priority_score={row['priority_score']}, support_runs={row['support_runs']}, mean_final_score={row['mean_final_score']}, mean_pred_esmc_zero_shot={row['mean_pred_esmc_zero_shot']}, mean_rank={row['mean_rank']}"
            )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    backbones = args.backbone or BACKBONES

    combined_rows: list[dict[str, object]] = []
    backbone_rows: dict[str, list[dict[str, object]]] = {}
    diagnostics: dict[str, dict[str, int]] = {}
    for backbone in backbones:
        summary_rows = summarize_backbone(args.base_dir, backbone, args.top_n, args.protein)
        backbone_rows[backbone] = summary_rows
        diagnostics[backbone] = collect_diagnostics(args.base_dir, backbone, args.top_n, args.protein)
        for row in summary_rows:
            row["backbone"] = backbone
        backbone_slug = backbone.replace(":", "_")
        write_csv(args.output_dir / f"{backbone_slug}_top{args.top_n}.csv", summary_rows)
        combined_rows.extend(summary_rows)

    combined_rows.sort(
        key=lambda row: (
            -float(row["priority_score"]),
            row["backbone"],
            float(row["mean_rank"]),
        )
    )
    write_csv(args.output_dir / f"combined_top{args.top_n}.csv", combined_rows)
    write_text(
        args.output_dir / "stakeholder_summary.md",
        build_markdown_summary(backbone_rows, diagnostics, args.top_n),
    )


if __name__ == "__main__":
    main()