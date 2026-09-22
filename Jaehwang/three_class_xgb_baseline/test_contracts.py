"""Contract tests use synthetic data only; they do not select research settings."""
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from data import DOMAIN_FEATURES, forbidden_reason, inspect_input, load_dataset, derive_wearable_features
from evaluation import metrics, bootstrap_intervals
from run import make_splits, training_weights, fit_fold, aggregate_subjects, preprocess_fold


class ResearchContracts(unittest.TestCase):
    def test_forbidden_features(self):
        for col in ["MMSE", "MMSE_TOTAL", "TOTAL", "Q1", "Q30", "Q13_2", "DIAG_NM",
                    "DIAG_SEQ", "DOCTOR_NM", "EMAIL", "SAMPLE_EMAIL", "original_label", "label", "patient_id"]:
            self.assertIsNotNone(forbidden_reason(col), col)
        self.assertTrue(all(forbidden_reason(c) is None for c in DOMAIN_FEATURES))

    def test_unique_subjects_use_exact_stratified_split(self):
        from sklearn.model_selection import StratifiedKFold
        y = np.repeat([0, 1, 2], 10)
        groups = np.array([f"s{i}" for i in range(len(y))])
        splits, name = make_splits(y, groups)
        self.assertEqual(name, "StratifiedKFold")
        expected = StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y)
        for (tr, te), (etr, ete) in zip(splits, expected):
            np.testing.assert_array_equal(tr, etr)
            np.testing.assert_array_equal(te, ete)

    def test_repeated_subjects_remain_disjoint(self):
        y = np.repeat(np.repeat([0, 1, 2], 50), 2)
        groups = np.repeat([f"s{i:03d}" for i in range(150)], 2)
        splits, name = make_splits(y, groups)
        self.assertEqual(name, "StratifiedGroupKFold")
        for tr, te in splits:
            self.assertFalse(set(groups[tr]) & set(groups[te]))

    def test_train_only_balanced_weights(self):
        y = np.array([0, 0, 0, 1, 1, 2])
        weights, counts = training_weights(y, np.array(list("abcdef")))
        self.assertEqual(counts, {"0": 3, "1": 2, "2": 1})
        for label in [0, 1, 2]:
            self.assertAlmostEqual(weights[y == label].sum(), 2.)

    def test_missing_features_fail_without_authorization(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "synthetic.csv"
            fields = ["EMAIL", "original_label", "MMSE", "DIAG_NM"] + DOMAIN_FEATURES[:-2]
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for i in range(30):
                    row = {c: 1 for c in fields}
                    row.update(EMAIL=f"s{i}", original_label=i % 3, MMSE=30, DIAG_NM="forbidden")
                    writer.writerow(row)
            audit = inspect_input(path)
            self.assertEqual(audit["missing_domain_features"], DOMAIN_FEATURES[-2:])
            self.assertIn("MMSE", [r["column"] for r in audit["excluded_columns"]])
            with self.assertRaisesRegex(ValueError, "Missing fixed features"):
                load_dataset(path)

    def test_test_values_do_not_change_fitted_model_or_imputer(self):
        rng = np.random.default_rng(11)
        train = pd.DataFrame(rng.normal(size=(30, 14)), columns=DOMAIN_FEATURES)
        train["sleep_hr_average"] = rng.uniform(55, 75, size=30)
        train["sleep_hr_lowest"] = rng.uniform(40, 50, size=30)
        train.iloc[0, 0] = np.nan
        y = np.tile([0, 1, 2], 10)
        groups = np.array([f"s{i}" for i in range(30)])
        test = train.iloc[:3].copy()
        _, m1, a1 = fit_fold(train, y, groups, test)
        _, m2, a2 = fit_fold(train, y, groups, test.fillna(1e8) + 1e8)
        self.assertEqual(a1["imputer_medians"], a2["imputer_medians"])
        self.assertEqual(m1.get_booster().save_raw(), m2.get_booster().save_raw())
        self.assertAlmostEqual(a1["imputer_medians"][DOMAIN_FEATURES[0]], train.iloc[:, 0].median())

    def test_derivation_is_row_local_and_ignores_forbidden_columns(self):
        raw = pd.DataFrame({c: [1., 2., 3.] for c in DOMAIN_FEATURES[:-2]})
        raw["sleep_hr_average"] = [60., 70., 80.]
        raw["sleep_hr_lowest"] = [45., 50., 55.]
        raw["original_label"] = [0, 1, 2]
        raw["MMSE"] = [30, 20, 10]
        raw["DIAG_NM"] = ["CN", "MCI", "Dem"]
        whole, audit = derive_wearable_features(raw, 1, "train")
        for index in raw.index:
            single, _ = derive_wearable_features(raw.loc[[index]], 1, "test")
            pd.testing.assert_frame_equal(single, whole.loc[[index]])
        changed = raw.copy()
        changed["original_label"] = 99
        changed["MMSE"] = -100
        changed["DIAG_NM"] = "anything"
        changed_features, _ = derive_wearable_features(changed, 1, "test")
        pd.testing.assert_frame_equal(whole, changed_features)
        self.assertEqual(list(whole), DOMAIN_FEATURES)
        np.testing.assert_allclose(whole.HR_drop_ratio, np.array([15., 20., 25.]) / (np.array([60., 70., 80.]) + 1e-5))
        np.testing.assert_allclose(whole.Circadian_Strain, np.array([1., 2., 3.]) / (np.array([1., 2., 3.]) + 1e-5))
        self.assertEqual(len(audit), 6)

    def test_invalid_derivations_are_nan_then_train_median_imputed(self):
        raw = pd.DataFrame({c: np.ones(6) for c in DOMAIN_FEATURES[:-2]})
        raw["sleep_hr_average"] = [60., np.nan, -1e-5, np.inf, 70., 80.]
        raw["sleep_hr_lowest"] = [40., 40., 40., 40., np.inf, 50.]
        raw["circadian_IV"] = [1., 1., 1., 1., np.inf, 1.]
        raw["circadian_IS"] = [2., np.nan, -1e-5, np.inf, 2., 4.]
        features, audit = derive_wearable_features(raw, 2, "test")
        self.assertFalse(np.isinf(features.to_numpy()).any())
        self.assertEqual(features.HR_drop_ratio.isna().sum(), 4)
        self.assertEqual(features.Circadian_Strain.isna().sum(), 4)
        hr_audit = next(r for r in audit if r["feature"] == "HR_drop_ratio")
        self.assertEqual(hr_audit["inf_before_cleaning"], 2)
        self.assertEqual(hr_audit["missing_after_cleaning"], 4)
        train = raw.iloc[[0, 5]].copy()
        _, transformed_test, imputer, _ = preprocess_fold(train, raw.iloc[[1, 2, 3, 4]], fold=2)
        for name in ("HR_drop_ratio", "Circadian_Strain"):
            expected = features.iloc[[0, 5]][name].median()
            np.testing.assert_allclose(transformed_test[name], expected)

    def test_subject_aggregation_and_metrics(self):
        y = np.repeat([0, 1, 2], 2)
        p = np.eye(3)[y] * .8 + .2 / 3
        df = pd.DataFrame({"row_index": range(6), "subject_index": [0, 0, 1, 1, 2, 2],
                           "fold": [1, 1, 2, 2, 3, 3], "y_true": y,
                           "p_CN": p[:, 0], "p_MCI": p[:, 1], "p_Dementia": p[:, 2]})
        subject = aggregate_subjects(df)
        score, cm = metrics(subject.y_true.to_numpy(), subject[["p_CN", "p_MCI", "p_Dementia"]].to_numpy())
        self.assertEqual(score["macro_f1"], 1.)
        np.testing.assert_array_equal(cm, np.eye(3))
        ci = bootstrap_intervals(np.array([0, 1, 2]), np.eye(3), repetitions=100)
        self.assertTrue(all(r["ci_low"] == 1 and r["ci_high"] == 1 for r in ci))


if __name__ == "__main__":
    unittest.main()
