import unittest
import tempfile
from pathlib import Path
import io
import tarfile

import numpy as np
import pandas as pd

from baselines.protocol import load_assay, recover_split, split_id
from baselines.features import one_hot, synthetic_embeddings
from baselines.models import mlde_predict, random_predict
from baselines.metrics import evaluate
from baselines.ensembles import combine_predictions
from baselines.evolution import fetch_mmseqs2, parse_a3m, pssm_predict


class ProtocolTests(unittest.TestCase):
    def test_three_way_ensemble_has_equal_component_weights(self):
        first = pd.DataFrame({"mutant": ["A1C", "A2C", "A3C"], "target": [0.0, 1.0, -1.0],
                              "pred_my_method": [1.0, 3.0, 2.0]})
        second = pd.DataFrame({"mutant": ["A3C", "A1C", "A2C"], "target": [-1.0, 0.0, 1.0],
                               "pred_mlde": [10.0, 30.0, 20.0]})
        third = pd.DataFrame({"mutant": ["A2C", "A3C", "A1C"], "target": [1.0, -1.0, 0.0],
                              "pred_epro": [100.0, 200.0, 0.0]})
        result = combine_predictions(first, second, "pred_my_method", "pred_mlde",
                                     third=third, third_column="pred_epro")
        self.assertEqual(result.mutant.tolist(), first.mutant.tolist())
        np.testing.assert_allclose(result.pred_score, [1 / 3, 2 / 3, 0.5])
        np.testing.assert_equal(result.pred_epro.to_numpy(), [0, 100, 200])
        third["pred_epro"] = 4.0
        constant_result = combine_predictions(first, second, "pred_my_method", "pred_mlde",
                                              third=third, third_column="pred_epro")
        np.testing.assert_allclose(constant_result.pred_score, [1 / 3, 0.5, 1 / 6])

    def test_three_way_ensemble_rejects_invalid_third_component(self):
        first = pd.DataFrame({"mutant": ["A1C", "A2C"], "target": [0.0, 1.0], "pred_my_method": [0.0, 1.0]})
        second = first.rename(columns={"pred_my_method": "pred_mlde"})
        third = first.rename(columns={"pred_my_method": "pred_epro"})
        for column, value in [("mutant", "A3C"), ("mutant", "A2C"), ("target", 0.5), ("pred_epro", np.nan)]:
            invalid = third.copy()
            invalid.loc[0, column] = value
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                combine_predictions(first, second, "pred_my_method", "pred_mlde",
                                    third=invalid, third_column="pred_epro")

    def test_ensemble_matches_variants_not_row_order(self):
        first = pd.DataFrame({"mutant": ["A1C", "A2C", "A3C"], "target": [0, 1, -1],
                              "pred_my_method": [1, 3, 2]})
        second = pd.DataFrame({"mutant": ["A3C", "A1C", "A2C"], "target": [-1, 0, 1],
                               "pred_mlde": [10, 30, 20]})
        result = combine_predictions(first, second, "pred_my_method", "pred_mlde")
        self.assertEqual(result.mutant.tolist(), first.mutant.tolist())
        np.testing.assert_allclose(result.pred_score, [0.5, 0.75, 0.25])

    def test_ensemble_rejects_mismatched_or_invalid_inputs(self):
        first = pd.DataFrame({"mutant": ["A1C", "A2C"], "target": [0.0, 1.0], "pred_my_method": [0.0, 1.0]})
        second = first.rename(columns={"pred_my_method": "pred_mlde"})
        for column, value in [("mutant", "A3C"), ("target", 0.5), ("pred_mlde", np.nan)]:
            invalid = second.copy()
            invalid.loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                combine_predictions(first, invalid, "pred_my_method", "pred_mlde")

    def test_ensemble_constant_scores_remain_finite(self):
        first = pd.DataFrame({"mutant": ["A1C", "A2C"], "target": [0, 1], "pred_my_method": [4, 4]})
        second = first.rename(columns={"pred_my_method": "pred_mlde"})
        result = combine_predictions(first, second, "pred_my_method", "pred_mlde")
        np.testing.assert_equal(result.pred_score.to_numpy(), [0, 0])

    def test_historical_recovery_is_independent_of_row_order(self):
        frame = pd.DataFrame({"mutant": [f"A{index}C" for index in range(1, 37)]})
        predictions = frame.iloc[16:].sample(frac=1, random_state=101)
        training = recover_split(frame.sample(frac=1, random_state=202), predictions, 16)
        self.assertEqual(set(training), set(frame.iloc[:16].mutant))
        self.assertEqual(split_id(training), split_id(training[::-1]))

    def test_truncated_historical_predictions_are_rejected(self):
        frame = pd.DataFrame({"mutant": [f"A{index}C" for index in range(1, 37)]})
        with self.assertRaises(ValueError):
            recover_split(frame, frame.iloc[17:], 16)

    def test_midpoint_normalization_matches_benchmark(self):
        frame = pd.DataFrame({"mutant": ["A1C", "A1D", "A1E", "A1F"],
                              "wt_sequence": ["A"] * 4,
                              "DMS_score": [-4, -2, 2, 6], "DMS_score_bin": [0, 0, 1, 1]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "assay.csv"
            frame.to_csv(path, index=False)
            normalized = load_assay(path)
        np.testing.assert_allclose(normalized.target, [-1, -2 / 3.9, 2 / 5.8, 1])

    def test_encodings_and_classical_predictions(self):
        sequences = ["AC", "AD", "AE", "AF"]
        encoded = one_hot(sequences)
        self.assertEqual(encoded.shape, (4, 40))
        np.testing.assert_equal(encoded.sum(axis=1), np.full((4, 1), 2))
        forward = synthetic_embeddings(sequences, 101)
        reverse = synthetic_embeddings(sequences[::-1], 101)
        np.testing.assert_equal(forward, reverse[::-1])
        predictions = mlde_predict(sequences[:2], [0, 1], sequences[2:], 101)
        self.assertTrue(np.isfinite(predictions).all())
        np.testing.assert_equal(random_predict(20, 101), random_predict(20, 101))
        self.assertEqual(len(set(random_predict(20, 101))), 20)

    def test_unsupported_budgets_are_not_reported_as_real_results(self):
        frame = pd.DataFrame({"mutant": [f"A{index}C" for index in range(1, 21)],
                              "target": np.arange(20), "pred_score": np.arange(20)})
        metrics = evaluate(frame, "test", 16, "Random")
        self.assertTrue(metrics.loc[metrics.N <= 20, "value"].notna().all())
        self.assertTrue(metrics.loc[metrics.N > 20, "value"].isna().all())

    def test_a3m_insertions_gaps_and_crop_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "msa.a3m"
            path.write_text(">query\nA-C\n>homolog\nAvv-\nD\n>duplicate\nA-C\n")
            self.assertEqual(parse_a3m(path), ["A-C", "A-D"])
            scores, coverage = pssm_predict(path, "MACF", ["C3D", "M1A"], start=2)
            self.assertEqual(scores[0], 0)
            np.testing.assert_equal(coverage, [1, 0])
            with self.assertRaises(ValueError):
                pssm_predict(path, "MADF", ["D3A"], start=2)

    def test_sequence_only_gp_is_finite_and_uses_prior(self):
        import torch
        torch.set_num_threads(1)
        from baselines.models import kermut_predict
        features = synthetic_embeddings([f"A{index}" for index in range(20)], 101, dimension=4)
        prior = np.linspace(-1, 1, 20)
        predictions, losses = kermut_predict(features[:16], prior[:16], features[16:],
                                             prior[:16], prior[16:], 101, steps=2)
        self.assertTrue(np.isfinite(losses).all())
        np.testing.assert_allclose(predictions, prior[16:], atol=1e-8)

    def test_mmseqs2_api_contract_without_network(self):
        archive_buffer = io.BytesIO()
        alignment = b">101\nAC\n>hit\nAD\n"
        with tarfile.open(fileobj=archive_buffer, mode="w:gz") as archive:
            member = tarfile.TarInfo("uniref.a3m")
            member.size = len(alignment)
            archive.addfile(member, io.BytesIO(alignment))

        class Response:
            content = archive_buffer.getvalue()

            def raise_for_status(self):
                pass

            def json(self):
                return {"status": "COMPLETE", "id": "test-ticket"}

        class Session:
            def post(self, url, data, timeout):
                self.query = data["q"]
                return Response()

            def get(self, url, timeout):
                return Response()

        session = Session()
        with tempfile.TemporaryDirectory() as directory:
            path = fetch_mmseqs2("AC", Path(directory) / "msa.a3m", session=session)
            self.assertEqual(parse_a3m(path), ["AC", "AD"])
            self.assertEqual(session.query, ">101\nAC\n")


if __name__ == "__main__":
    unittest.main()