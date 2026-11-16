#!/usr/bin/env python3
import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import internal_peft_ensemble as peft_runner
from internal_peft_multi_background import fit_evolvepro_regressor, score_with_prefit_evolvepro, write_outputs


SCORE_COLUMNS = {
    "zero_shot": "pred_esmc_zero_shot",
    "smitepro": "pred_smitepro",
    "evolvepro": "pred_evolvepro",
    "ensemble": "pred_ensemble",
}
COMPARISON_METHODS = ("smitepro", "evolvepro", "ensemble")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", required=True)
    parser.add_argument("--protein", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--model", default="esmc_300m")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train_batch_size", type=int, default=None)
    parser.add_argument("--inference_batch_size", type=int, default=16)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--positive_threshold", type=float, default=0.0)
    parser.add_argument("--num_splits", type=int, default=10)
    parser.add_argument("--holdout_families_per_split", type=int, default=5)
    parser.add_argument("--max_splits", type=int, default=None)
    return parser.parse_args()


def sanitize_label(label: str) -> str:
    return str(label).replace(":", "__")


def is_single_mutant(mutant: str) -> bool:
    return isinstance(mutant, str) and mutant.count(":") == 0 and len(mutant) >= 3


def mutation_position(mutant: str) -> int:
    return int("".join(ch for ch in str(mutant) if ch.isdigit()))


def select_literal_single_holdouts(train_df: pd.DataFrame) -> pd.DataFrame:
    holdout_df = train_df.copy()
    holdout_df = holdout_df[holdout_df["mutant"].apply(is_single_mutant)].copy()
    if "derivation_depth" in holdout_df.columns:
        holdout_df = holdout_df[holdout_df["derivation_depth"].fillna(-1).astype(int) == 1].copy()
    elif "derivation_type" in holdout_df.columns:
        holdout_df = holdout_df[holdout_df["derivation_type"].fillna("") == "enumerated_single_step"].copy()
    holdout_df = holdout_df[holdout_df["source_mutant"].notna()].copy()
    holdout_df["expected_direction"] = np.where(holdout_df["target"] > 0, "positive", "negative")
    return holdout_df.sort_values(["source_mutant", "parent_mutant", "target", "mutant"], ascending=[True, True, False, True]).reset_index(drop=True)


def resolve_background_mutant(row: pd.Series) -> str:
    if pd.notna(row.get("parent_mutant")) and str(row["parent_mutant"]).strip():
        return str(row["parent_mutant"]).strip()
    if pd.notna(row.get("source_mutant")) and str(row["source_mutant"]).strip():
        source_mutant = str(row["source_mutant"]).strip()
        if ":" in source_mutant:
            return ":".join(source_mutant.split(":")[:-1])
    raise ValueError(f"Could not resolve background for holdout mutant {row['mutant']}")


def build_split_plan(holdout_df: pd.DataFrame, num_splits: int, holdout_families_per_split: int, seed: int) -> pd.DataFrame:
    family_summary = []
    for source_mutant, family in holdout_df.groupby("source_mutant", sort=True):
        family_summary.append(
            {
                "source_mutant": str(source_mutant),
                "family_size": int(len(family)),
                "positive_count": int((family["expected_direction"] == "positive").sum()),
                "negative_count": int((family["expected_direction"] == "negative").sum()),
            }
        )

    family_df = pd.DataFrame(family_summary).sort_values("source_mutant").reset_index(drop=True)
    families = family_df["source_mutant"].tolist()
    if not families:
        raise ValueError("No source_mutant families available for holdout sampling.")

    sample_size = min(holdout_families_per_split, len(families))
    overall_has_positive = bool((holdout_df["expected_direction"] == "positive").any())
    overall_has_negative = bool((holdout_df["expected_direction"] == "negative").any())

    rng = np.random.default_rng(seed)
    plan_rows = []
    for split_index in range(num_splits):
        chosen = None
        for _ in range(200):
            candidate = sorted(rng.choice(families, size=sample_size, replace=False).tolist())
            split_rows = holdout_df[holdout_df["source_mutant"].astype(str).isin(candidate)]
            if overall_has_positive and not (split_rows["expected_direction"] == "positive").any():
                continue
            if overall_has_negative and not (split_rows["expected_direction"] == "negative").any():
                continue
            chosen = candidate
            break
        if chosen is None:
            chosen = sorted(rng.choice(families, size=sample_size, replace=False).tolist())

        split_label = f"split_{split_index + 1:02d}"
        for source_mutant in chosen:
            family_row = family_df.loc[family_df["source_mutant"] == source_mutant].iloc[0]
            plan_rows.append(
                {
                    "split_index": split_index,
                    "split_label": split_label,
                    "source_mutant": source_mutant,
                    "family_size": int(family_row["family_size"]),
                    "positive_count": int(family_row["positive_count"]),
                    "negative_count": int(family_row["negative_count"]),
                }
            )
    return pd.DataFrame(plan_rows)


def build_train_subset(train_df: pd.DataFrame, split_rows: pd.DataFrame) -> pd.DataFrame:
    heldout_source_mutants = set(split_rows["source_mutant"].dropna().astype(str))
    heldout_sequences = set(split_rows["mutated_sequence"].astype(str))
    subset = train_df[
        ~train_df["mutated_sequence"].astype(str).isin(heldout_sequences)
        & ~train_df["mutant"].astype(str).isin(heldout_source_mutants)
        & ~train_df["source_mutant"].fillna("").astype(str).isin(heldout_source_mutants)
    ].copy()
    return subset.reset_index(drop=True)


def fit_split_models(train_subset: pd.DataFrame, args, device):
    base_model, model_ft, tokenizer = peft_runner.fit_peft_model(train_subset, args, device)
    regressor = fit_evolvepro_regressor(train_subset, base_model, tokenizer, device, args.inference_batch_size)
    return base_model, model_ft, tokenizer, regressor


def build_site_specific_candidates(model_ft, tokenizer, wt_sequence: str, device, inference_batch_size: int, target_positions) -> pd.DataFrame:
    site_scores = peft_runner.internal_pipeline.predict_all_site_deltas(
        model=model_ft,
        tokenizer=tokenizer,
        wt_seq=wt_sequence,
        device=device,
        batch_size=inference_batch_size,
    )

    rows = []
    for position in sorted(set(target_positions)):
        index = int(position) - 1
        wt_aa = wt_sequence[index]
        entropy = float(site_scores["entropy"][index])
        for mut_aa in peft_runner.internal_pipeline.AA20:
            if mut_aa == wt_aa:
                continue
            mutant = f"{wt_aa}{position}{mut_aa}"
            rows.append(
                {
                    "mutant": mutant,
                    "wt_sequence": wt_sequence,
                    "mutated_sequence": wt_sequence[:index] + mut_aa + wt_sequence[index + 1 :],
                    "pred_smitepro": float(site_scores["delta20"][index][peft_runner.internal_pipeline.AA20.index(mut_aa)]),
                    "smitepro_entropy": entropy,
                }
            )

    return pd.DataFrame(rows)


def build_scored_candidates(
    train_subset: pd.DataFrame,
    reference_wt_sequence: str,
    background_mutant: str,
    target_positions,
    args,
    device,
    base_model,
    model_ft,
    tokenizer,
    regressor,
):
    prediction_wt_sequence = peft_runner.resolve_prediction_wt_sequence(
        reference_wt_sequence,
        SimpleNamespace(predict_from_mutant=background_mutant, predict_wt_sequence=None),
    )

    candidate_df = build_site_specific_candidates(
        model_ft,
        tokenizer,
        prediction_wt_sequence,
        device,
        args.inference_batch_size,
        target_positions,
    )
    candidate_df = candidate_df[~candidate_df["mutated_sequence"].isin(train_subset["mutated_sequence"])].copy()
    candidate_df = peft_runner.score_with_esmc_zero_shot(
        base_model,
        tokenizer,
        prediction_wt_sequence,
        candidate_df,
        device,
        args.inference_batch_size,
    )
    candidate_df = peft_runner.score_with_smitepro(
        model_ft,
        tokenizer,
        prediction_wt_sequence,
        candidate_df,
        device,
        args.inference_batch_size,
    )
    candidate_df = score_with_prefit_evolvepro(
        candidate_df,
        regressor,
        base_model,
        tokenizer,
        device,
        args.inference_batch_size,
    )
    candidate_df = peft_runner.add_normalized_ensemble(candidate_df)
    candidate_df = candidate_df.sort_values("pred_ensemble", ascending=False).reset_index(drop=True)
    sequence_name = peft_runner.internal_pipeline.seq_diff_to_mutants(reference_wt_sequence, prediction_wt_sequence)
    candidate_df.insert(0, "sequence_name", sequence_name)
    candidate_df.insert(1, "candidate_scope", "heldout_sites_only")
    return candidate_df


def extract_rank_metrics(candidate_df: pd.DataFrame, mutant: str, mutated_sequence: str, score_col: str) -> dict:
    ranked = candidate_df.sort_values(score_col, ascending=False).reset_index(drop=True)
    matches = ranked[(ranked["mutant"] == mutant) | (ranked["mutated_sequence"] == mutated_sequence)].copy()
    if matches.empty:
        raise ValueError(f"Mutant {mutant} not found when ranking by {score_col}")

    position = mutation_position(mutant)
    site_df = ranked[ranked["mutant"].str.extract(r"(\d+)").astype(int)[0] == position].copy().reset_index(drop=True)
    site_match = site_df[(site_df["mutant"] == mutant) | (site_df["mutated_sequence"] == mutated_sequence)].copy()
    site_rank = int(site_match.index[0]) + 1
    return {
        "site_rank": site_rank,
        "site_rank_percentile": 1.0 - ((site_rank - 1) / max(len(site_df) - 1, 1)),
        "site_candidate_count": int(len(site_df)),
        "evaluated_candidate_count": int(len(ranked)),
    }


def directional_outcome(expected_direction: str, delta: int) -> str:
    if delta == 0:
        return "unchanged"
    if expected_direction == "positive":
        return "better" if delta > 0 else "worse"
    return "better" if delta < 0 else "worse"


def summarize_hit(split_label: str, holdout_row: pd.Series, candidate_df: pd.DataFrame, positive_threshold: float) -> dict:
    mutant = str(holdout_row["mutant"])
    mutated_sequence = str(holdout_row["mutated_sequence"])
    hit = candidate_df[(candidate_df["mutant"] == mutant) | (candidate_df["mutated_sequence"] == mutated_sequence)].iloc[0]
    expected_direction = "positive" if float(holdout_row["target"]) > positive_threshold else "negative"

    summary_row = {
        "split_label": split_label,
        "holdout_mutant": mutant,
        "background_mutant": resolve_background_mutant(holdout_row),
        "target": float(holdout_row["target"]),
        "expected_direction": expected_direction,
        "candidate_scope": "heldout_sites_only",
        "parent_mutant": holdout_row.get("parent_mutant"),
        "source_mutant": holdout_row.get("source_mutant"),
        "pred_esmc_zero_shot": float(hit["pred_esmc_zero_shot"]),
        "pred_smitepro": float(hit["pred_smitepro"]),
        "pred_evolvepro": float(hit["pred_evolvepro"]),
        "pred_ensemble": float(hit["pred_ensemble"]),
    }

    rank_metrics = {
        method: extract_rank_metrics(candidate_df, mutant, mutated_sequence, score_col)
        for method, score_col in SCORE_COLUMNS.items()
    }
    zero_shot = rank_metrics["zero_shot"]
    summary_row["site_candidate_count"] = zero_shot["site_candidate_count"]
    summary_row["evaluated_candidate_count"] = zero_shot["evaluated_candidate_count"]

    for method, metrics in rank_metrics.items():
        summary_row[f"{method}_site_rank"] = metrics["site_rank"]
        summary_row[f"{method}_site_rank_percentile"] = metrics["site_rank_percentile"]

    for method in COMPARISON_METHODS:
        site_delta = zero_shot["site_rank"] - rank_metrics[method]["site_rank"]
        outcome = directional_outcome(expected_direction, int(site_delta))
        summary_row[f"{method}_site_rank_delta_vs_zero_shot"] = int(site_delta)
        summary_row[f"{method}_directional_outcome_vs_zero_shot"] = outcome
        summary_row[f"{method}_better_vs_zero_shot"] = outcome == "better"

    return summary_row


def build_metrics(summary_df: pd.DataFrame) -> pd.DataFrame:
    metrics = {
        "holdout_count": int(len(summary_df)),
        "positive_count": int((summary_df["expected_direction"] == "positive").sum()),
        "negative_count": int((summary_df["expected_direction"] == "negative").sum()),
    }

    for method in COMPARISON_METHODS:
        outcome_col = f"{method}_directional_outcome_vs_zero_shot"
        delta_col = f"{method}_site_rank_delta_vs_zero_shot"
        metrics[f"{method}_better_count"] = int((summary_df[outcome_col] == "better").sum())
        metrics[f"{method}_worse_count"] = int((summary_df[outcome_col] == "worse").sum())
        metrics[f"{method}_unchanged_count"] = int((summary_df[outcome_col] == "unchanged").sum())
        metrics[f"{method}_success_fraction"] = float((summary_df[outcome_col] == "better").mean()) if len(summary_df) else np.nan
        metrics[f"{method}_mean_site_rank_delta_vs_zero_shot"] = float(summary_df[delta_col].mean()) if len(summary_df) else np.nan

        positive_mask = summary_df["expected_direction"] == "positive"
        negative_mask = summary_df["expected_direction"] == "negative"
        metrics[f"{method}_positive_success_fraction"] = float((summary_df.loc[positive_mask, outcome_col] == "better").mean()) if positive_mask.any() else np.nan
        metrics[f"{method}_negative_success_fraction"] = float((summary_df.loc[negative_mask, outcome_col] == "better").mean()) if negative_mask.any() else np.nan
        metrics[f"{method}_positive_mean_site_rank_delta"] = float(summary_df.loc[positive_mask, delta_col].mean()) if positive_mask.any() else np.nan
        metrics[f"{method}_negative_mean_site_rank_delta"] = float(summary_df.loc[negative_mask, delta_col].mean()) if negative_mask.any() else np.nan

    return pd.DataFrame([metrics])


def main():
    args = parse_args()
    os.makedirs(args.output_root, exist_ok=True)

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    train_df, reference_wt_sequence = peft_runner.prepare_training_dataframe(pd.read_csv(args.data_file), args.protein)
    holdout_df = select_literal_single_holdouts(train_df)
    if holdout_df.empty:
        raise ValueError("No literal single-mutant holdouts were found.")

    split_plan_df = build_split_plan(holdout_df, args.num_splits, args.holdout_families_per_split, args.seed)
    split_plan_df.to_csv(os.path.join(args.output_root, "random_family_holdout_split_plan.csv"), index=False)

    split_labels = split_plan_df["split_label"].drop_duplicates().tolist()
    if args.max_splits is not None:
        split_labels = split_labels[: args.max_splits]

    summary_rows = []
    for split_label in split_labels:
        split_families = split_plan_df.loc[split_plan_df["split_label"] == split_label, "source_mutant"].astype(str).tolist()
        split_rows = holdout_df[holdout_df["source_mutant"].astype(str).isin(split_families)].copy().reset_index(drop=True)
        train_subset = build_train_subset(train_df, split_rows)
        if train_subset.empty:
            raise ValueError(f"Training subset is empty after removing {split_label}")

        print(f"Training {split_label} with {len(split_rows)} held-out singles across {len(split_families)} source families")

        base_model, model_ft, tokenizer, regressor = fit_split_models(train_subset, args, device)
        scored_by_background = {}
        unique_backgrounds = sorted({resolve_background_mutant(row) for _, row in split_rows.iterrows()})
        for background_mutant in unique_backgrounds:
            background_positions = sorted(
                {mutation_position(mutant) for mutant in split_rows.loc[split_rows["parent_mutant"].fillna("").astype(str) == background_mutant, "mutant"].astype(str)}
            )
            if not background_positions:
                background_positions = sorted(
                    {
                        mutation_position(row["mutant"])
                        for _, row in split_rows.iterrows()
                        if resolve_background_mutant(row) == background_mutant
                    }
                )
            candidate_df = build_scored_candidates(
                train_subset,
                reference_wt_sequence,
                background_mutant,
                background_positions,
                args,
                device,
                base_model,
                model_ft,
                tokenizer,
                regressor,
            )
            scored_by_background[background_mutant] = candidate_df

            background_dir = os.path.join(args.output_root, split_label, sanitize_label(background_mutant))
            write_outputs(candidate_df, background_dir, args.protein)

        for _, holdout_row in split_rows.iterrows():
            background_mutant = resolve_background_mutant(holdout_row)
            summary_row = summarize_hit(split_label, holdout_row, scored_by_background[background_mutant], args.positive_threshold)
            summary_rows.append(summary_row)
            print(
                f"{split_label}: {background_mutant} -> {summary_row['holdout_mutant']} | "
                f"zero_shot={summary_row['zero_shot_site_rank']} | "
                f"smitepro={summary_row['smitepro_site_rank']} | "
                f"evolvepro={summary_row['evolvepro_site_rank']} | "
                f"ensemble={summary_row['ensemble_site_rank']}"
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(args.output_root, "random_family_holdout_summary.csv"), index=False)
    build_metrics(summary_df).to_csv(os.path.join(args.output_root, "random_family_holdout_metrics.csv"), index=False)


if __name__ == "__main__":
    main()