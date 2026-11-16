"""Evaluate score ensembles on identical historical held-out variants."""

import json
from pathlib import Path
import re
import zipfile

import numpy as np
import pandas as pd

from baselines.metrics import evaluate


def combine_predictions(first, second, first_column, second_column, *, third=None, third_column=None):
    inputs = [(first, first_column), (second, second_column)]
    if (third is None) != (third_column is None):
        raise ValueError("Third ensemble input and score column must be provided together")
    if third is not None:
        inputs.append((third, third_column))
    for frame, column in inputs:
        if frame.empty or frame.mutant.isna().any() or frame.mutant.duplicated().any():
            raise ValueError("Ensemble inputs must have unique, nonempty variant identities")
        if not np.isfinite(frame[["target", column]].to_numpy(dtype=float)).all():
            raise ValueError("Ensemble inputs contain missing or nonfinite targets/scores")
    output = first[["mutant", "target", first_column]].copy()
    normalized_scores = []
    for frame, column in inputs:
        if set(first.mutant) != set(frame.mutant):
            raise ValueError("Ensemble inputs have different held-out variant sets")
        aligned = frame.set_index("mutant").loc[first.mutant]
        if not np.allclose(first.target.to_numpy(), aligned.target.to_numpy(), rtol=0, atol=1e-7):
            raise ValueError("Ensemble inputs have different evaluation targets")
        output[column] = aligned[column].to_numpy()
        scores = output[column]
        normalized_scores.append((scores - scores.min()) / (scores.max() - scores.min() + 1e-8))
    output["pred_score"] = sum(normalized_scores) / len(inputs)
    return output


def evaluate_ensemble2(archive_path, historical_directory, budgets=(10, 20, 30, 40), min_sites=100):
    return _evaluate_ensemble(archive_path, historical_directory, budgets, min_sites, "Ensemble2")


def evaluate_ensemble3(archive_path, historical_directory, budgets=(10, 20, 30, 40), min_sites=100):
    return _evaluate_ensemble(archive_path, historical_directory, budgets, min_sites, "Ensemble3")


def _evaluate_ensemble(archive_path, historical_directory, budgets, min_sites, method):
    historical_directory = Path(historical_directory)
    records = []
    seen = set()
    with zipfile.ZipFile(archive_path) as archive:
        for member in sorted(archive.namelist()):
            if "/mlde/" not in member:
                continue
            match = re.fullmatch(r"(.+)_final_predictions_shot(\d+)\.csv", Path(member).name)
            if match is None:
                continue
            assay, shots_text = match.groups()
            shots = int(shots_text)
            if shots not in {16, 32, 64, 128}:
                continue
            if (assay, shots) in seen:
                raise ValueError(f"Duplicate MLDE prediction table for {assay}, K={shots}")
            seen.add((assay, shots))
            path = historical_directory / Path(member).name
            if not path.is_file():
                continue
            historical = pd.read_csv(path, usecols=lambda column: column in {
                "mutant", "target", "pred_my_method", "is_train",
                *( ["pred_epro"] if method == "Ensemble3" else [] )
            })
            if "is_train" in historical:
                flags = historical.is_train.astype(str).str.lower()
                if not flags.isin(["true", "false", "1", "0"]).all():
                    raise ValueError(f"Invalid historical training flags: {path.name}")
                historical = historical.loc[~flags.isin(["true", "1"])].copy()
            sites = {token[1:-1] for mutant in historical.mutant for token in mutant.split(":")}
            if len(sites) < min_sites:
                continue
            mlde = pd.read_csv(archive.open(member), usecols=lambda column: column in {
                "mutant", "target", "pred_mlde", "seed", "split_id", "is_train",
                "synthetic_encoder", "synthetic_msa"
            })
            for column in ["is_train", "synthetic_encoder", "synthetic_msa"]:
                if column not in mlde or not mlde[column].astype(str).str.lower().isin(["false", "0"]).all():
                    raise ValueError(f"MLDE table contains training/synthetic or invalid flags: {member}")
            if not mlde.seed.eq(101).all():
                raise ValueError(f"Expected seed-101 MLDE predictions: {member}")
            manifest = json.loads(archive.read(f"baselines/splits/{assay}.json"))
            selection = manifest["selections"][shots_text]
            if not mlde.split_id.eq(selection["split_id"]).all():
                raise ValueError(f"MLDE split identity does not match its manifest: {member}")
            if len(selection["train_mutants"]) != shots or set(selection["train_mutants"]) & set(historical.mutant):
                raise ValueError(f"Training variants overlap the historical held-out pool: {member}")
            extra_input = {"third": historical, "third_column": "pred_epro"} if method == "Ensemble3" else {}
            predictions = combine_predictions(historical, mlde, "pred_my_method", "pred_mlde", **extra_input)
            metrics = evaluate(predictions, assay, shots, method, budgets=budgets)
            metrics["seed"] = 101
            metrics["split_id"] = selection["split_id"]
            metrics["source_archive"] = Path(archive_path).name
            records.append(metrics)
    if not records:
        raise ValueError(f"No matched historical/MLDE prediction pools support {method}")
    return pd.concat(records, ignore_index=True)