#!/usr/bin/env python3
import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import internal_peft_ensemble as peft_runner
import proteinGym.rebuild_eval_pipeline_PEFT as peft_pipeline


def parse_background_specs(specs: list[str]) -> list[tuple[str, str]]:
    parsed = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Background spec must be LABEL=MUTANT, got: {spec}")
        label, mutant = spec.split("=", 1)
        label = label.strip()
        mutant = mutant.strip()
        if not label:
            raise ValueError(f"Background spec must be LABEL=MUTANT, got: {spec}")
        parsed.append((label, mutant))
    return parsed


def fit_evolvepro_regressor(train_df: pd.DataFrame, base_model, tokenizer, device, inference_batch_size: int):
    train_embed_df = train_df.copy()
    train_embed_df["sequence"] = train_embed_df["mutated_sequence"]

    x_train = peft_pipeline.extract_mean_embeddings(
        train_embed_df,
        tokenizer,
        base_model,
        device,
        bs=inference_batch_size,
    )
    y_train = peft_pipeline.normalize_np(train_df["target"].values)
    regressor = RandomForestRegressor(n_estimators=100, criterion="friedman_mse", n_jobs=40, random_state=1)
    regressor.fit(x_train, y_train)
    return regressor


def score_with_prefit_evolvepro(candidate_df: pd.DataFrame, regressor, base_model, tokenizer, device, inference_batch_size: int) -> pd.DataFrame:
    candidate_embed_df = candidate_df.copy()
    candidate_embed_df["sequence"] = candidate_embed_df["mutated_sequence"]

    x_pred = peft_pipeline.extract_mean_embeddings(
        candidate_embed_df,
        tokenizer,
        base_model,
        device,
        bs=inference_batch_size,
    )

    scored = candidate_df.copy()
    scored["pred_evolvepro"] = regressor.predict(x_pred)
    return scored


def write_outputs(candidate_df: pd.DataFrame, output_dir: str, protein: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    out_path = os.path.join(output_dir, f"{protein}_allSingles_scores_peft_ensemble.csv")
    candidate_df.to_csv(out_path, index=False)

    per_site = candidate_df.copy()
    per_site["position"] = per_site["mutant"].str.extract(r"(\d+)").astype(int)
    per_site = per_site.loc[per_site.groupby("position")["pred_ensemble"].idxmax()].copy()
    per_site = per_site.sort_values("pred_ensemble", ascending=False).reset_index(drop=True)
    per_site.drop(columns=["position"], inplace=True)
    best_out = os.path.join(output_dir, f"{protein}_perSite_scores_peft_ensemble.csv")
    per_site.to_csv(best_out, index=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", required=True)
    parser.add_argument("--protein", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--background", action="append", required=True, help="LABEL=MUTANT")
    parser.add_argument("--model", default="esmc_300m")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train_batch_size", type=int, default=None)
    parser.add_argument("--inference_batch_size", type=int, default=16)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    backgrounds = parse_background_specs(args.background)

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    train_df, reference_wt_sequence = peft_runner.prepare_training_dataframe(pd.read_csv(args.data_file), args.protein)
    base_model, model_ft, tokenizer = peft_runner.fit_peft_model(train_df, args, device)
    regressor = fit_evolvepro_regressor(train_df, base_model, tokenizer, device, args.inference_batch_size)

    for label, background_mutant in backgrounds:
        if background_mutant in {"", "WT"}:
            if reference_wt_sequence is None:
                raise ValueError(
                    f"Background {label!r} requests WT, but the training data does not resolve to a single reference WT sequence."
                )
            prediction_wt_sequence = reference_wt_sequence
        else:
            prediction_wt_sequence = peft_runner.resolve_prediction_wt_sequence(
                reference_wt_sequence,
                SimpleNamespace(predict_from_mutant=background_mutant, predict_wt_sequence=None),
            )

        candidate_df = peft_runner.generate_single_mutant_candidates(
            model_ft,
            tokenizer,
            prediction_wt_sequence,
            device,
            args.inference_batch_size,
        )
        candidate_df = candidate_df[~candidate_df["mutated_sequence"].isin(train_df["mutated_sequence"])].copy()
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

        if reference_wt_sequence is None:
            sequence_name = "custom_background"
        else:
            sequence_name = peft_runner.internal_pipeline.seq_diff_to_mutants(reference_wt_sequence, prediction_wt_sequence)
        candidate_df.insert(0, "sequence_name", sequence_name)

        output_dir = os.path.join(args.output_root, label)
        write_outputs(candidate_df, output_dir, args.protein)
        print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()