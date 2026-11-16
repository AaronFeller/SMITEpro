"""Selection metrics using the manuscript notebook's unscreened-library scope."""

import numpy as np
import pandas as pd


def evaluate(predictions, assay, shots, method, budgets=(10, 20, 30, 40)):
    targets = predictions.target
    top_one = set(predictions.loc[targets >= targets.quantile(0.99), "mutant"])
    top_five = set(predictions.nlargest(5, "target", keep="all").mutant)
    top_ten = set(predictions.nlargest(10, "target", keep="all").mutant)
    median = targets.median()
    denominator = max(targets.max() - median, 1e-8)
    prevalence = len(top_one) / len(predictions)
    sites = {token[1:-1] for mutant in predictions.mutant for token in mutant.split(":")}
    records = []
    for budget in budgets:
        selected = predictions.nlargest(budget, "pred_score")
        found = set(selected.mutant)
        supported = len(predictions) >= budget
        precision = len(found & top_one) / len(selected)
        shifted = selected.target.to_numpy() - targets.min()
        discounts = np.log2(np.arange(2, len(selected) + 2))
        ideal = predictions.nlargest(budget, "target").target.to_numpy() - targets.min()
        ideal_gain = np.sum(ideal / discounts)
        values = {"Top 1% Precision": precision,
                  "Top 5 Hit Rate": len(found & top_five) / len(top_five),
                  "Top 10 Recall": len(found & top_ten) / len(top_ten),
                  "Normalized Max Score": max(0, (selected.target.max() - median) / denominator),
                  "Enrichment Factor": precision / prevalence,
                  "Mean Top-N Fitness": selected.target.mean(),
                  "NDCG": np.sum(shifted / discounts) / ideal_gain if ideal_gain > 0 else 0.0}
        for name, value in values.items():
            records.append({"system": f"{assay}_final_predictions_shot{shots}", "train_size": shots,
                            "method": method, "metric": name, "value": value if supported else np.nan,
                            "N": budget, "actual_n": len(selected), "budget_supported": supported,
                            "num_sites": len(sites)})
    return pd.DataFrame(records)