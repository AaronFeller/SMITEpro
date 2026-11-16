"""Direct, fixed-shot baseline runner; no rounds or cluster submission."""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "baselines/.deps"), str(ROOT / "kermut")]

import numpy as np
import pandas as pd

from .evolution import fetch_mmseqs2, proteingym_msa, pssm_predict
from .features import esmc_embeddings, synthetic_embeddings
from .metrics import evaluate
from .models import kermut_predict, mlde_predict, random_predict
from .protocol import (SHOTS, load_assay, read_split, reference_sequence, write_manifest,
                       write_novel_manifest)

METHODS = {"kermut": "Kermut ESMC Sequence GP", "mlde": "MLDE One-Hot RF",
           "random": "Random", "msa": "MSA PSSM"}


def resource(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def prepare(config, assay_path):
    manifest = resource(config["manifest_dir"]) / f"{Path(assay_path).stem}.json"
    write_manifest(assay_path, resource(config["historical_dir"]), manifest)
    return manifest


def run_assay(config, args):
    assay_path = resource(args.assay)
    assay = assay_path.stem
    frame = load_assay(assay_path)
    manifest = resource(args.manifest) if args.manifest else prepare(config, assay_path)
    methods = list(METHODS) if args.method == "all" else [args.method]
    output_dir = resource(args.output_dir or config["output_dir"]) / assay
    output_dir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("baselines")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    log.addHandler(logging.StreamHandler())
    handler = logging.FileHandler(output_dir / "run.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    log.addHandler(handler)
    device = args.device or config["device"]
    seed = config["seed"] if args.seed is None else args.seed
    if "kermut" in methods:
        gp = config["gp"]
        if gp["use_structure_kernel"] or gp["coordinates"] is not None or \
                not gp["use_sequence_kernel"] or gp["prior"] != "delta_esm":
            raise ValueError("Only sequence kernels with delta_esm prior and no coordinates are allowed")
        import torch
        torch.set_num_threads(config["threads"])
    if args.synthetic_encoder and (args.shots != [16] or args.test_limit != 20 or device != "cpu"):
        raise ValueError("Synthetic encoding is restricted to K=16, 20-test CPU smoke runs")
    prior_path = resource(args.zero_shot) if args.zero_shot else \
        resource(config["zero_shot_dir"]) / f"{assay}_zero_shot_scores.csv"
    prior = pd.read_csv(prior_path) if "kermut" in methods else None
    if prior is not None and prior.mutant.duplicated().any():
        raise ValueError("Duplicate zero-shot mutant IDs")
    msa_path = None
    msa_start = args.msa_start
    if "msa" in methods:
        if args.msa:
            msa_path = resource(args.msa)
        elif args.novel:
            msa_path = fetch_mmseqs2(reference_sequence(frame),
                resource(config["msa"]["cache_dir"]) / f"{assay}.a3m", config["msa"]["api_url"])
        else:
            msa_path, msa_start = proteingym_msa(assay,
                resource(config["msa"]["reference_csv"]), resource(config["msa"]["directory"]))
    all_metrics = []
    for shots in args.shots:
        train, test, selection_id = read_split(frame, assay_path, manifest, shots, args.test_limit)
        log.info("assay=%s K=%s train=%s test=%s split=%s device=%s synthetic_encoder=%s",
                 assay, shots, len(train), len(test), selection_id, device, args.synthetic_encoder)
        combined = test.copy()
        for method in methods:
            started = time.monotonic()
            losses = []
            if method == "kermut":
                sequences = train.mutated_sequence.tolist() + test.mutated_sequence.tolist()
                features = synthetic_embeddings(sequences, seed) if args.synthetic_encoder else \
                    esmc_embeddings(sequences,
                        resource(args.embedding_cache) if args.embedding_cache else
                        resource(config["embedding_cache_dir"]) / f"{assay}.npz",
                        device, config["model"], config["embedding_batch_size"])
                prior_values = prior.set_index("mutant").delta_esm.reindex(
                    train.mutant.tolist() + test.mutant.tolist()).to_numpy(dtype=float)
                if not np.isfinite(prior_values).all():
                    raise ValueError("Missing or nonfinite zero-shot log-odds")
                scores, losses = kermut_predict(features[:shots], train.target.to_numpy(),
                    features[shots:], prior_values[:shots], prior_values[shots:], seed, device,
                    args.gp_steps if args.gp_steps is not None else config["gp"]["steps"],
                    config["gp"]["learning_rate"], config["gp"]["prediction_batch_size"])
                combined["delta_esm"] = prior_values[shots:]
            elif method == "mlde":
                scores = mlde_predict(train.mutated_sequence.tolist(), train.target.to_numpy(),
                    test.mutated_sequence.tolist(), seed, config["rf"]["n_estimators"], config["threads"])
            elif method == "random":
                scores = random_predict(len(test), seed)
            else:
                scores, coverage = pssm_predict(msa_path, reference_sequence(frame), test.mutant,
                                                msa_start, config["msa"]["pseudocount"])
                combined["msa_coverage"] = coverage
                if np.any(coverage < 1):
                    log.warning("MSA covers only %s/%s test variants fully; uncovered sites score zero",
                                int((coverage == 1).sum()), len(test))
            if len(scores) != len(test) or not np.isfinite(scores).all():
                raise ValueError(f"Invalid predictions from {method}")
            column = f"pred_{method}"
            combined[column] = scores
            predictions = test.copy()
            predictions[column] = scores
            predictions["pred_score"] = scores
            predictions["is_train"] = False
            predictions["split_id"] = selection_id
            predictions["seed"] = seed
            predictions["synthetic_encoder"] = args.synthetic_encoder and method == "kermut"
            predictions["synthetic_msa"] = args.synthetic_msa and method == "msa"
            if method == "msa":
                predictions["msa_coverage"] = coverage
            method_dir = output_dir / method
            method_dir.mkdir(exist_ok=True)
            filename = f"{assay}_final_predictions_shot{shots}.csv"
            predictions.to_csv(method_dir / filename, index=False)
            metrics = evaluate(predictions, assay, shots, METHODS[method])
            metrics["split_id"] = selection_id
            metrics["seed"] = seed
            metrics["smoke_test"] = args.test_limit is not None
            metrics["synthetic_resources"] = args.synthetic_encoder or args.synthetic_msa
            metrics.to_csv(method_dir / f"metrics_shot{shots}.csv", index=False)
            all_metrics.append(metrics)
            metadata = {"method": method, "shots": shots, "seed": seed, "split_id": selection_id,
                "train_mutants": train.mutant.tolist(), "test_mutants": test.mutant.tolist(),
                "device": device if method == "kermut" else "cpu", "gp_loss": losses,
                "synthetic_encoder": args.synthetic_encoder and method == "kermut",
                "synthetic_msa": args.synthetic_msa and method == "msa",
                "manifest": str(manifest), "assay": str(assay_path), "config": config,
                "zero_shot": str(prior_path) if method == "kermut" else None,
                "msa": str(msa_path) if method == "msa" else None,
                "elapsed_seconds": time.monotonic() - started}
            (method_dir / f"run_shot{shots}.json").write_text(json.dumps(metadata, indent=2) + "\n")
            log.info("%s completed: %s finite predictions in %.2fs", method, len(scores),
                     metadata["elapsed_seconds"])
        historical_path = resource(config["historical_dir"]) / f"{assay}_final_predictions_shot{shots}.csv"
        if historical_path.exists() and not args.novel:
            historical = pd.read_csv(historical_path).set_index("mutant")
            previous = historical.reindex(combined.mutant)
            if not np.allclose(previous.target.to_numpy(), combined.target.to_numpy(), atol=1e-7):
                raise ValueError("Historical evaluation targets differ from current normalized targets")
            for column in ["pred_my_method", "pred_epro", "delta_esm"]:
                if column in historical:
                    combined[column] = previous[column].to_numpy()
        combined["split_id"] = selection_id
        combined["is_train"] = False
        combined["synthetic_encoder"] = args.synthetic_encoder
        combined["synthetic_msa"] = args.synthetic_msa
        combined.to_csv(output_dir / f"{assay}_final_predictions_shot{shots}.csv", index=False)
    pd.concat(all_metrics, ignore_index=True).to_csv(output_dir / "evaluation_metrics.csv", index=False)
    handler.close()
    log.removeHandler(handler)
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run", "smoke"])
    parser.add_argument("--config", default="baselines/config.json")
    parser.add_argument("--assay")
    parser.add_argument("--manifest")
    parser.add_argument("--method", choices=["all", *METHODS], default="all")
    parser.add_argument("--shots", nargs="+", type=int, choices=SHOTS, default=list(SHOTS))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--gp-steps", type=int)
    parser.add_argument("--zero-shot")
    parser.add_argument("--embedding-cache")
    parser.add_argument("--output-dir")
    parser.add_argument("--msa")
    parser.add_argument("--msa-start", type=int, default=1)
    parser.add_argument("--novel", action="store_true",
                        help="Opt in to submitting a novel target sequence to MMseqs2")
    parser.add_argument("--synthetic-encoder", action="store_true")
    args = parser.parse_args()
    args.synthetic_msa = False
    config = json.loads(resource(args.config).read_text())
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.command == "prepare":
        if args.novel:
            if not args.assay or not args.zero_shot:
                parser.error("Novel prepare requires --assay and --zero-shot")
            output = resource(args.manifest) if args.manifest else \
                resource(config["manifest_dir"]) / f"{Path(args.assay).stem}.json"
            write_novel_manifest(resource(args.assay), resource(args.zero_shot), output)
            print(f"Novel preset manifest: {output}")
            return
        paths = [resource(args.assay)] if args.assay else sorted(resource(config["assay_dir"]).glob("*.csv"))
        report = []
        for path in paths:
            try:
                manifest = prepare(config, path)
                content = json.loads(manifest.read_text())
                report.append({"assay": path.stem,
                    "status": "ready" if len(content["selections"]) == 4 else "partial",
                    "available_shots": list(map(int, content["selections"])), "manifest": str(manifest)})
            except FileNotFoundError as error:
                report.append({"assay": path.stem, "status": "unavailable", "reason": str(error)})
            except ValueError as error:
                report.append({"assay": path.stem, "status": "blocked", "reason": str(error)})
        directory = resource(config["manifest_dir"])
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "preparation_report.json").write_text(json.dumps(report, indent=2) + "\n")
        ready = sum(row["status"] == "ready" for row in report)
        print(f"All four budgets: {ready}/{len(report)} assays; "
              f"partial: {sum(row['status'] == 'partial' for row in report)}; "
              f"historically unavailable: {sum(row['status'] == 'unavailable' for row in report)}")
        if any(row["status"] == "blocked" for row in report):
            raise SystemExit("Incompatible assay data; inspect preparation_report.json")
        return
    if args.command == "smoke":
        args.assay = args.assay or str(resource(config["assay_dir"]) / "SDA_BACSU_Tsuboyama_2023_1PV0.csv")
        args.shots, args.test_limit, args.device = [16], 20, "cpu"
        args.gp_steps = 5
        args.output_dir = args.output_dir or "baselines/results/smoke"
        args.synthetic_encoder = not bool(args.embedding_cache)
        default_cache = resource(config["embedding_cache_dir"]) / f"{Path(args.assay).stem}.npz"
        if not args.embedding_cache and default_cache.exists():
            frame = load_assay(resource(args.assay))
            manifest = prepare(config, resource(args.assay))
            train, test, _ = read_split(frame, resource(args.assay), manifest, 16, 20)
            try:
                esmc_embeddings(train.mutated_sequence.tolist() + test.mutated_sequence.tolist(),
                                default_cache, "cpu", config["model"])
                args.embedding_cache, args.synthetic_encoder = str(default_cache), False
            except ValueError:
                print("Existing ESMC cache is incomplete/incompatible; smoke uses synthetic features")
        if not args.msa:
            frame = load_assay(resource(args.assay))
            reference = reference_sequence(frame)
            directory = resource(args.output_dir)
            directory.mkdir(parents=True, exist_ok=True)
            alignment = directory / "synthetic_smoke.a3m"
            homolog = ("A" if reference[0] != "A" else "C") + reference[1:]
            alignment.write_text(f">query\n{reference}\n>homolog1\n{homolog}\n"
                                 f">homolog2\n{reference[:2]}aa{reference[2:]}\n")
            args.msa, args.synthetic_msa = str(alignment), True
    if not args.assay:
        parser.error("--assay is required for run")
    if args.test_limit is not None and args.test_limit <= 0:
        parser.error("--test-limit must be positive")
    if args.gp_steps is not None and args.gp_steps <= 0:
        parser.error("--gp-steps must be positive")
    if args.novel and not args.manifest:
        parser.error("Novel run requires --manifest from prepare --novel")
    print(f"Outputs: {run_assay(config, args)}")


if __name__ == "__main__":
    main()