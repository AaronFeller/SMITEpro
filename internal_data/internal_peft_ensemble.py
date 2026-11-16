#!/usr/bin/env python3
import argparse
import os
import sys
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from torch.utils.data import DataLoader


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import internal_new as internal_pipeline
import proteinGym.rebuild_eval_pipeline_PEFT as peft_pipeline


os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


def minmax_normalize(values: pd.Series) -> np.ndarray:
    array = values.to_numpy(dtype=np.float32)
    return (array - array.min()) / (array.max() - array.min() + 1e-8)


def normalize_training_targets(data: pd.DataFrame) -> pd.DataFrame:
    normalized = data.copy()
    normalized["target_raw"] = normalized["target"].astype(np.float32)

    if normalized["target_raw"].abs().max() <= 1.5:
        return normalized

    scale = float(max(normalized["target_raw"].abs().quantile(0.95), 1e-8))
    normalized["target"] = np.clip(normalized["target_raw"] / scale, -1.0, 1.0)
    normalized["target_scale"] = scale
    return normalized


def resolve_train_batch_size(num_examples: int, requested_batch_size: int | None) -> int:
    if requested_batch_size is not None:
        return max(1, requested_batch_size)
    return max(4, min(num_examples // 2, 16))


def deduplicate_by_mutated_sequence(data: pd.DataFrame) -> pd.DataFrame:
    key_columns = [column for column in ("wt_sequence", "mutant") if column in data.columns]
    if not key_columns and "mutated_sequence" in data.columns:
        key_columns = ["mutated_sequence"]

    if not key_columns:
        return data

    if "target" in data.columns:
        aggregations = {column: "first" for column in data.columns if column not in set(key_columns + ["target"])}
        aggregations["target"] = "mean"
        deduplicated = data.groupby(key_columns, as_index=False).agg(aggregations)
        column_order = [column for column in data.columns if column in deduplicated.columns]
        return deduplicated[column_order]

    return data.drop_duplicates(subset=key_columns).reset_index(drop=True)


def build_mutated_sequence(wt_sequence: str, mutant: str) -> str:
    sequence = list(wt_sequence)
    for token in [part.strip() for part in str(mutant).split(":") if str(part).strip()]:
        wt_aa = token[0]
        mut_aa = token[-1]
        position = int(token[1:-1]) - 1
        if sequence[position] != wt_aa:
            raise ValueError(
                f"WT mismatch for {token}: expected {wt_aa}, found {sequence[position]} at {position + 1}"
            )
        sequence[position] = mut_aa
    return "".join(sequence)


def normalize_mutant_against_wt(wt_sequence: str, mutant: str) -> str:
    normalized_tokens = []
    for token in [part.strip() for part in str(mutant).split(":") if str(part).strip()]:
        wt_aa = token[0]
        mut_aa = token[-1]
        position = int(token[1:-1]) - 1
        actual_wt = wt_sequence[position]
        if actual_wt == mut_aa:
            continue
        if actual_wt != wt_aa:
            wt_aa = actual_wt
        normalized_tokens.append(f"{wt_aa}{position + 1}{mut_aa}")
    return ":".join(normalized_tokens)


def resolve_prediction_wt_sequence(reference_wt_sequence: str | None, args) -> str:
    if args.predict_wt_sequence and args.predict_from_mutant:
        raise ValueError("Use only one of --predict_wt_sequence or --predict_from_mutant.")

    if args.predict_wt_sequence:
        prediction_wt_sequence = str(args.predict_wt_sequence)
        if reference_wt_sequence is not None and len(prediction_wt_sequence) != len(reference_wt_sequence):
            raise ValueError(
                "--predict_wt_sequence must have the same length as the reference WT sequence."
            )
        return prediction_wt_sequence

    if args.predict_from_mutant:
        if reference_wt_sequence is None:
            raise ValueError(
                "Training data contains multiple WT backgrounds. Use --predict_wt_sequence or include a single source_wt_sequence."
            )
        normalized_mutant = normalize_mutant_against_wt(reference_wt_sequence, args.predict_from_mutant)
        if not normalized_mutant:
            raise ValueError("--predict_from_mutant resolves to the reference WT sequence; provide at least one real mutation.")
        return build_mutated_sequence(reference_wt_sequence, normalized_mutant)

    if reference_wt_sequence is None:
        raise ValueError(
            "Training data contains multiple WT backgrounds. Use --predict_wt_sequence or --predict_from_mutant."
        )
    return reference_wt_sequence


def mutant_from_sequences(wt_sequence: str, mutated_sequence: str) -> str:
    if len(wt_sequence) != len(mutated_sequence):
        raise ValueError(
            f"Sequence length mismatch: WT has {len(wt_sequence)} aa, candidate has {len(mutated_sequence)} aa"
        )
    mutations = []
    for index, (wt_aa, mut_aa) in enumerate(zip(wt_sequence, mutated_sequence), start=1):
        if wt_aa != mut_aa:
            mutations.append(f"{wt_aa}{index}{mut_aa}")
    if not mutations:
        raise ValueError("Candidate sequence matches WT exactly; at least one mutation is required.")
    return ":".join(mutations)


def prepare_training_dataframe(df: pd.DataFrame, protein: str) -> tuple[pd.DataFrame, str | None]:
    data = df.copy()

    if "wt_sequence" not in data.columns:
        if {"mutant", "mutated_sequence"}.issubset(data.columns):
            wt_sequence = peft_pipeline.restore_wt_sequence(
                data["mutant"].iloc[0],
                data["mutated_sequence"].iloc[0],
            )
            data["wt_sequence"] = wt_sequence
        else:
            raise ValueError("Training data must include wt_sequence, or both mutant and mutated_sequence.")

    data["wt_sequence"] = data["wt_sequence"].astype(str)
    data["mutant"] = data.apply(
        lambda row: normalize_mutant_against_wt(row["wt_sequence"], row["mutant"]),
        axis=1,
    )
    data = data[data["mutant"].astype(str).str.len() > 0].copy()

    if "mutated_sequence" not in data.columns or data["mutated_sequence"].isnull().any():
        data["mutated_sequence"] = data.apply(
            lambda row: build_mutated_sequence(row["wt_sequence"], row["mutant"]),
            axis=1,
        )

    max_wt_length = data["wt_sequence"].str.len().max()
    if max_wt_length > 2046:
        raise ValueError(f"{protein}: WT length {max_wt_length} exceeds the ESMC limit of 2046 tokens.")

    if "target" not in data.columns:
        if {"DMS_score", "DMS_score_bin"}.issubset(data.columns):
            data["target"] = data["DMS_score"]
            internal_pipeline.midpoint_scale_whole_df(
                data,
                target_col="target",
                bin_col="DMS_score_bin",
                out_col="target_norm",
            )
            data["target"] = data["target_norm"]
            data.drop(columns=["target_norm"], inplace=True, errors="ignore")
        else:
            raise ValueError(
                "Training data must include target, or both DMS_score and DMS_score_bin for normalization."
            )

    data = deduplicate_by_mutated_sequence(data)
    data = normalize_training_targets(data)

    reference_wt_sequence = None
    if data["wt_sequence"].nunique() == 1:
        reference_wt_sequence = str(data["wt_sequence"].iloc[0])
    elif "source_wt_sequence" in data.columns and data["source_wt_sequence"].dropna().astype(str).nunique() == 1:
        reference_wt_sequence = str(data["source_wt_sequence"].dropna().astype(str).iloc[0])

    return data, reference_wt_sequence


def prepare_candidate_dataframe(candidate_file: str, wt_sequence: str) -> pd.DataFrame:
    data = pd.read_csv(candidate_file).copy()

    if "mutated_sequence" not in data.columns and "sequence" in data.columns:
        data["mutated_sequence"] = data["sequence"]

    if "mutant" not in data.columns:
        if "mutated_sequence" not in data.columns:
            raise ValueError("Candidate file must include mutant, mutated_sequence, or sequence.")
        data["mutant"] = data["mutated_sequence"].astype(str).apply(
            lambda sequence: mutant_from_sequences(wt_sequence, sequence)
        )
    else:
        data["mutant"] = data["mutant"].apply(lambda mutant: normalize_mutant_against_wt(wt_sequence, mutant))

    data = data[data["mutant"].astype(str).str.len() > 0].copy()

    if "wt_sequence" not in data.columns:
        data["wt_sequence"] = wt_sequence
    else:
        data["wt_sequence"] = data["wt_sequence"].fillna(wt_sequence)

    if "mutated_sequence" not in data.columns or data["mutated_sequence"].isnull().any():
        data["mutated_sequence"] = data["mutant"].apply(lambda mutant: build_mutated_sequence(wt_sequence, mutant))

    if "target" not in data.columns and {"DMS_score", "DMS_score_bin"}.issubset(data.columns):
        data["target"] = data["DMS_score"]
        internal_pipeline.midpoint_scale_whole_df(
            data,
            target_col="target",
            bin_col="DMS_score_bin",
            out_col="target_norm",
        )
        data["target"] = data["target_norm"]
        data.drop(columns=["target_norm"], inplace=True, errors="ignore")

    return deduplicate_by_mutated_sequence(data)


def generate_single_mutant_candidates(model, tokenizer, wt_sequence: str, device, inference_batch_size: int) -> pd.DataFrame:
    site_scores = internal_pipeline.predict_all_site_deltas(
        model=model,
        tokenizer=tokenizer,
        wt_seq=wt_sequence,
        device=device,
        batch_size=inference_batch_size,
    )

    rows = []
    for index, wt_aa in enumerate(wt_sequence):
        entropy = float(site_scores["entropy"][index])
        for mut_aa in internal_pipeline.AA20:
            if mut_aa == wt_aa:
                continue
            mutant = f"{wt_aa}{index + 1}{mut_aa}"
            rows.append(
                {
                    "mutant": mutant,
                    "wt_sequence": wt_sequence,
                    "mutated_sequence": wt_sequence[:index] + mut_aa + wt_sequence[index + 1 :],
                    "pred_smitepro": float(site_scores["delta20"][index][internal_pipeline.AA20.index(mut_aa)]),
                    "smitepro_entropy": entropy,
                }
            )

    return pd.DataFrame(rows)


def fit_peft_model(train_df: pd.DataFrame, args, device):
    base_model, tokenizer = peft_pipeline.load_model(args.model)
    base_model.to(device).eval()
    peft_pipeline.tokenizer = tokenizer

    vocab = tokenizer.get_vocab()
    train_data = peft_pipeline.prep_data_from_mutant(train_df, tokenizer, None, vocab)
    train_ds = peft_pipeline.ProteinDataset(train_data)

    effective_batch_size = resolve_train_batch_size(len(train_df), args.train_batch_size)
    train_loader = DataLoader(
        train_ds,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    model_ft = deepcopy(base_model).to(device)
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["layernorm_qkv.1", "query", "value", "q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=None,
    )
    model_ft = get_peft_model(model_ft, peft_config)

    for parameter in model_ft.base_model.model.sequence_head.parameters():
        parameter.requires_grad = True

    optimizer = torch.optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model_ft.parameters()),
        lr=args.lr,
        weight_decay=0.05,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.steps,
        eta_min=1e-5,
    )

    peft_pipeline.train_fixed_steps(
        model_ft,
        train_loader,
        optimizer,
        scheduler,
        device,
        total_steps=args.steps,
        kwargs={"beta": args.beta, "margin": args.margin},
    )

    return base_model, model_ft, tokenizer


def score_with_smitepro(
    model_ft,
    tokenizer,
    wt_sequence: str,
    candidate_df: pd.DataFrame,
    device,
    inference_batch_size: int,
) -> pd.DataFrame:
    scored = candidate_df.copy()
    if "pred_smitepro" not in scored.columns:
        deltas, entropy, _, _ = peft_pipeline.predict_deltas(
            model_ft,
            tokenizer,
            wt_sequence,
            scored,
            device,
            batch_size=inference_batch_size,
        )
        scored["pred_smitepro"] = deltas
        scored["smitepro_entropy"] = entropy
    return scored


def score_with_esmc_zero_shot(
    base_model,
    tokenizer,
    wt_sequence: str,
    candidate_df: pd.DataFrame,
    device,
    inference_batch_size: int,
) -> pd.DataFrame:
    scored = candidate_df.copy()
    deltas, entropy, mut_logp, wt_logp = peft_pipeline.predict_deltas(
        base_model,
        tokenizer,
        wt_sequence,
        scored,
        device,
        batch_size=inference_batch_size,
    )
    scored["pred_esmc_zero_shot"] = deltas
    scored["esmc_zero_shot_entropy"] = entropy
    scored["esmc_zero_shot_mut_logp"] = mut_logp
    scored["esmc_zero_shot_wt_logp"] = wt_logp
    return scored


def score_with_evolvepro(
    train_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    base_model,
    tokenizer,
    device,
    inference_batch_size: int,
) -> pd.DataFrame:
    train_embed_df = train_df.copy()
    train_embed_df["sequence"] = train_embed_df["mutated_sequence"]

    candidate_embed_df = candidate_df.copy()
    candidate_embed_df["sequence"] = candidate_embed_df["mutated_sequence"]

    x_train = peft_pipeline.extract_mean_embeddings(
        train_embed_df,
        tokenizer,
        base_model,
        device,
        bs=inference_batch_size,
    )
    x_pred = peft_pipeline.extract_mean_embeddings(
        candidate_embed_df,
        tokenizer,
        base_model,
        device,
        bs=inference_batch_size,
    )

    y_train = peft_pipeline.normalize_np(train_df["target"].values)
    regressor = RandomForestRegressor(n_estimators=100, criterion="friedman_mse", n_jobs=40, random_state=1)
    regressor.fit(x_train, y_train)

    scored = candidate_df.copy()
    scored["pred_evolvepro"] = regressor.predict(x_pred)
    return scored


def add_normalized_ensemble(candidate_df: pd.DataFrame) -> pd.DataFrame:
    scored = candidate_df.copy()
    scored["norm_smitepro"] = minmax_normalize(scored["pred_smitepro"])
    scored["norm_evolvepro"] = minmax_normalize(scored["pred_evolvepro"])
    if "pred_esmc_zero_shot" in scored.columns:
        scored["norm_esmc_zero_shot"] = minmax_normalize(scored["pred_esmc_zero_shot"])
    scored["pred_ensemble"] = (scored["norm_smitepro"] + scored["norm_evolvepro"]) / 2.0
    scored["final_score"] = scored["pred_ensemble"]
    return scored


def write_metrics(candidate_df: pd.DataFrame, output_path: str) -> None:
    if "target" not in candidate_df.columns or candidate_df["target"].nunique() <= 1:
        return

    metrics = {
        "smitepro_spearman": spearmanr(candidate_df["pred_smitepro"], candidate_df["target"]).correlation,
        "evolvepro_spearman": spearmanr(candidate_df["pred_evolvepro"], candidate_df["target"]).correlation,
        "ensemble_spearman": spearmanr(candidate_df["pred_ensemble"], candidate_df["target"]).correlation,
    }
    if "pred_esmc_zero_shot" in candidate_df.columns:
        metrics["esmc_zero_shot_spearman"] = spearmanr(
            candidate_df["pred_esmc_zero_shot"],
            candidate_df["target"],
        ).correlation
    pd.DataFrame([metrics]).to_csv(output_path, index=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", required=True)
    parser.add_argument("--protein", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--candidate_file", default=None)
    parser.add_argument("--model", default="esmc_300m")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train_batch_size", type=int, default=None)
    parser.add_argument("--inference_batch_size", type=int, default=16)
    parser.add_argument("--batch_size", dest="inference_batch_size", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--predict_from_mutant", default=None)
    parser.add_argument("--predict_wt_sequence", default=None)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    train_df, reference_wt_sequence = prepare_training_dataframe(pd.read_csv(args.data_file), args.protein)
    prediction_wt_sequence = resolve_prediction_wt_sequence(reference_wt_sequence, args)
    base_model, model_ft, tokenizer = fit_peft_model(train_df, args, device)

    if args.candidate_file:
        candidate_df = prepare_candidate_dataframe(args.candidate_file, prediction_wt_sequence)
        candidate_df = score_with_esmc_zero_shot(
            base_model,
            tokenizer,
            prediction_wt_sequence,
            candidate_df,
            device,
            args.inference_batch_size,
        )
        candidate_df = score_with_smitepro(
            model_ft,
            tokenizer,
            prediction_wt_sequence,
            candidate_df,
            device,
            args.inference_batch_size,
        )
    else:
        candidate_df = generate_single_mutant_candidates(
            model_ft,
            tokenizer,
            prediction_wt_sequence,
            device,
            args.inference_batch_size,
        )
        candidate_df = candidate_df[~candidate_df["mutated_sequence"].isin(train_df["mutated_sequence"])].copy()
        candidate_df = score_with_esmc_zero_shot(
            base_model,
            tokenizer,
            prediction_wt_sequence,
            candidate_df,
            device,
            args.inference_batch_size,
        )

    candidate_df = score_with_evolvepro(
        train_df,
        candidate_df,
        base_model,
        tokenizer,
        device,
        args.inference_batch_size,
    )
    candidate_df = add_normalized_ensemble(candidate_df)
    candidate_df = candidate_df.sort_values("pred_ensemble", ascending=False).reset_index(drop=True)
    if reference_wt_sequence is None:
        sequence_name = "custom_background"
    else:
        sequence_name = internal_pipeline.seq_diff_to_mutants(reference_wt_sequence, prediction_wt_sequence)
    candidate_df.insert(0, "sequence_name", sequence_name)

    if args.candidate_file:
        out_path = os.path.join(args.output_dir, f"{args.protein}_experimental_scores_peft_ensemble.csv")
        candidate_df.to_csv(out_path, index=False)
        metrics_path = os.path.join(args.output_dir, f"{args.protein}_experimental_metrics_peft_ensemble.csv")
        write_metrics(candidate_df, metrics_path)
    else:
        out_path = os.path.join(args.output_dir, f"{args.protein}_allSingles_scores_peft_ensemble.csv")
        candidate_df.to_csv(out_path, index=False)

        per_site = candidate_df.copy()
        per_site["position"] = per_site["mutant"].str.extract(r"(\d+)").astype(int)
        per_site = per_site.loc[per_site.groupby("position")["pred_ensemble"].idxmax()].copy()
        per_site = per_site.sort_values("pred_ensemble", ascending=False).reset_index(drop=True)
        per_site.drop(columns=["position"], inplace=True)
        best_out = os.path.join(args.output_dir, f"{args.protein}_perSite_scores_peft_ensemble.csv")
        per_site.to_csv(best_out, index=False)

    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
