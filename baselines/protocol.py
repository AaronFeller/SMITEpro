"""Assay loading and recovery of previously evaluated fixed-shot splits."""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

AA = "ACDEFGHIKLMNPQRSTVWY"
SHOTS = (16, 32, 64, 128)


def load_assay(path):
    frame = pd.read_csv(path)
    if frame["mutant"].isna().any() or frame["mutant"].duplicated().any():
        raise ValueError("Assay mutant IDs must be present and unique")
    if "target" not in frame:
        if not {"DMS_score", "DMS_score_bin"} <= set(frame):
            raise ValueError("Need target or DMS_score and DMS_score_bin")
        positive = frame.loc[frame.DMS_score_bin == 1, "DMS_score"]
        negative = frame.loc[frame.DMS_score_bin == 0, "DMS_score"]
        midpoint = (positive.min() + negative.max()) / 2 if not (
            positive.empty or negative.empty) else frame.DMS_score.median()
        values = frame.DMS_score.to_numpy(dtype=float)
        negative_scale = max(float(midpoint - (negative.quantile(0.05)
                             if not negative.empty else values.min())), 1e-8)
        positive_scale = max(float((positive.quantile(0.95)
                             if not positive.empty else values.max()) - midpoint), 1e-8)
        frame["target"] = np.clip(np.where(values <= midpoint,
            (values - midpoint) / negative_scale, (values - midpoint) / positive_scale), -1, 1)
    if not np.isfinite(frame.target.to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite fitness targets")
    if "mutated_sequence" not in frame:
        if "wt_sequence" not in frame:
            raise ValueError("Need mutated_sequence or wt_sequence")
        frame["mutated_sequence"] = [mutate(wt, mutant) for wt, mutant in
                                      zip(frame.wt_sequence, frame.mutant)]
    return frame


def mutate(sequence, mutant):
    sequence = list(sequence)
    for token in mutant.split(":"):
        match = re.fullmatch(r"([A-Z])(\d+)([A-Z])", token)
        if match is None:
            raise ValueError(f"Invalid substitution: {token}")
        old, position, new = match.groups()
        index = int(position) - 1
        if index < 0 or index >= len(sequence) or sequence[index] != old:
            raise ValueError(f"Reference mismatch: {token}")
        sequence[index] = new
    return "".join(sequence)


def reference_sequence(frame):
    if "wt_sequence" in frame:
        values = frame.wt_sequence.dropna().unique()
        if len(values) == 1:
            return values[0]
    row = frame.iloc[0]
    sequence = list(row.mutated_sequence)
    for token in row.mutant.split(":"):
        index = int(token[1:-1]) - 1
        if sequence[index] != token[-1]:
            raise ValueError("Cannot restore reference sequence")
        sequence[index] = token[0]
    return "".join(sequence)


def recover_split(frame, predictions, shots):
    if shots not in SHOTS:
        raise ValueError(f"Supported shots: {SHOTS}")
    if frame.mutant.str.contains(":").any():
        raise ValueError("Historical exclusion recovery requires a singles-only assay")
    if predictions.mutant.duplicated().any():
        raise ValueError("Duplicate historical prediction IDs")
    available = set(frame.mutant)
    evaluation = set(predictions.mutant)
    if not evaluation <= available:
        raise ValueError("Historical predictions contain variants outside this assay")
    training = available - evaluation
    if len(training) != shots:
        raise ValueError(f"Expected {shots} omitted training variants, found {len(training)}")
    return sorted(training)


def split_id(training):
    return hashlib.sha256("\n".join(sorted(training)).encode()).hexdigest()


def write_manifest(assay_path, historical_dir, output):
    frame = load_assay(assay_path)
    assay = Path(assay_path).stem
    selections = {}
    missing = []
    for shots in SHOTS:
        source = Path(historical_dir) / f"{assay}_final_predictions_shot{shots}.csv"
        if not source.exists():
            missing.append(shots)
            continue
        historical = pd.read_csv(source)
        training = recover_split(frame, historical, shots)
        if "target" in historical:
            normalized = frame.set_index("mutant").target.reindex(historical.mutant).to_numpy()
            if not np.allclose(normalized, historical.target.to_numpy(), atol=1e-7):
                raise ValueError(f"Historical fitness normalization mismatch for K={shots}")
        selections[str(shots)] = {"train_mutants": training, "split_id": split_id(training),
                                  "source": str(source.resolve())}
    if not selections:
        raise FileNotFoundError(f"No February manuscript predictions for {assay}")
    manifest = {"assay": assay, "assay_sha256": hashlib.sha256(
        Path(assay_path).read_bytes()).hexdigest(), "selections": selections}
    if missing:
        manifest["unavailable_shots"] = missing
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and json.loads(output.read_text()) != manifest:
        raise ValueError("Refusing to overwrite a different split manifest")
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def write_novel_manifest(assay_path, prior_path, output):
    frame = load_assay(assay_path)
    prior = pd.read_csv(prior_path)
    if prior.mutant.duplicated().any() or not {"score", "delta_esm"} <= set(prior):
        raise ValueError("Novel selections require unique mutant IDs, score and delta_esm")
    eligible = set(frame.loc[~frame.mutant.str.contains(":"), "mutant"])
    ranked = prior.loc[prior.mutant.isin(eligible)].sort_values("score", ascending=False)
    if set(ranked.mutant) != eligible or not np.isfinite(ranked[["score", "delta_esm"]]).all().all():
        raise ValueError("Incomplete/nonfinite novel zero-shot ranking")
    remaining = ranked.mutant.tolist()
    ordering = []
    while remaining:
        positions = set()
        deferred = []
        for mutant in remaining:
            position = int(mutant[1:-1])
            if position in positions:
                deferred.append(mutant)
            else:
                positions.add(position)
                ordering.append(mutant)
        remaining = deferred
    selections = {}
    for shots in SHOTS:
        if len(ordering) >= shots:
            training = sorted(ordering[:shots])
            selections[str(shots)] = {"train_mutants": training, "split_id": split_id(training),
                                      "source": str(Path(prior_path).resolve())}
    if not selections:
        raise ValueError("Novel assay has fewer than 16 eligible training variants")
    manifest = {"assay": Path(assay_path).stem, "assay_sha256": hashlib.sha256(
        Path(assay_path).read_bytes()).hexdigest(), "selections": selections,
        "provenance": "novel-only position-diverse cached zero-shot selection"}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and json.loads(output.read_text()) != manifest:
        raise ValueError("Refusing to overwrite a different novel manifest")
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def read_split(frame, assay_path, manifest_path, shots, test_limit=None):
    manifest = json.loads(Path(manifest_path).read_text())
    digest = hashlib.sha256(Path(assay_path).read_bytes()).hexdigest()
    if digest != manifest["assay_sha256"]:
        raise ValueError("Assay does not match the frozen manifest")
    if str(shots) not in manifest["selections"]:
        raise ValueError(f"K={shots} has no frozen historical selection for this assay")
    selection = manifest["selections"][str(shots)]
    training = selection["train_mutants"]
    if len(training) != shots or len(set(training)) != shots:
        raise ValueError("Invalid training budget in manifest")
    if split_id(training) != selection["split_id"]:
        raise ValueError("Training selection checksum mismatch")
    if not set(training) <= set(frame.mutant):
        raise ValueError("Missing selected training variants")
    train = frame.set_index("mutant", drop=False).loc[training].reset_index(drop=True)
    test = frame.loc[~frame.mutant.isin(training)].copy()
    if test_limit is not None:
        test = test.head(test_limit)
    if test.empty:
        raise ValueError("No unscreened variants")
    if set(train.mutated_sequence) & set(test.mutated_sequence):
        raise ValueError("Train/test sequence overlap")
    return train, test.reset_index(drop=True), selection["split_id"]