import unittest
from pathlib import Path

from sport_pipeline.evaluation import evaluate_predictions, load_target_registry, validate_prediction_rows
from sport_pipeline.evaluation.predictions import PredictionValidationError
from sport_pipeline.models.context import ConstantContextBaseline


ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "configs/targets/target_registry_v1.yaml"


class EvaluationContractsTest(unittest.TestCase):
    def test_target_registry_loads_expected_heads(self) -> None:
        targets = load_target_registry(REGISTRY_PATH)
        self.assertEqual(targets["ev"].column, "launch_speed")
        self.assertEqual(targets["la"].level, "event")
        self.assertFalse(targets["xba"].required)
        self.assertFalse(targets["xwoba"].required)
        self.assertEqual(targets["ops"].level, "player_season")
        self.assertTrue(targets["ops"].requires_pa_manifest)

    def test_predictions_schema_rejects_event_level_ops(self) -> None:
        row = self._prediction_row(target_name="ops", prediction_level="event")
        with self.assertRaises(PredictionValidationError):
            validate_prediction_rows([row])

    def test_predictions_schema_accepts_integer_valued_float_for_parquet_ints(self) -> None:
        row = self._prediction_row()
        row["n_prior_clips"] = 2.0
        validate_prediction_rows([row])

    def test_predictions_schema_rejects_non_integer_float_for_int_columns(self) -> None:
        row = self._prediction_row()
        row["n_prior_clips"] = 2.5
        with self.assertRaises(PredictionValidationError):
            validate_prediction_rows([row])

    def test_missing_optional_targets_are_skipped(self) -> None:
        targets = load_target_registry(REGISTRY_PATH)
        rows = [
            self._prediction_row(target_name="ev", y_true=100.0, y_pred=98.0),
            self._prediction_row(
                target_name="xba",
                y_true=None,
                y_pred=None,
                target_available=False,
                label_missing_reason="statcast_expected_outcome_missing",
                target_source="estimated_ba_using_speedangle",
                head_kind="probability",
                loss_name="mse",
            ),
        ]
        metrics = evaluate_predictions(rows, targets, run_id="test")
        self.assertIn("ev", metrics["metrics"]["event"])
        self.assertNotIn("xba", metrics["metrics"].get("event", {}))
        self.assertEqual(metrics["label_availability"]["xba"]["missing"], 1)
        self.assertEqual(
            metrics["skipped"]["xba"]["statcast_expected_outcome_missing"],
            1,
        )

    def test_ops_unavailable_is_skipped_with_pa_reason(self) -> None:
        targets = load_target_registry(REGISTRY_PATH)
        row = self._prediction_row(
            target_name="ops",
            prediction_level="player_season",
            event_id=None,
            y_true=None,
            y_pred=None,
            target_available=False,
            target_source="target_ops",
            head_kind="regression",
            loss_name="huber",
            label_missing_reason="pa_manifest_unavailable",
            requires_pa_manifest=True,
        )
        metrics = evaluate_predictions([row], targets, run_id="test")
        self.assertEqual(metrics["label_availability"]["ops"]["missing"], 1)
        self.assertEqual(metrics["skipped"]["ops"]["pa_manifest_unavailable"], 1)
        self.assertNotIn("player_season", metrics["metrics"])

    def test_constant_context_baseline_emits_long_format(self) -> None:
        targets = load_target_registry(REGISTRY_PATH)
        baseline = ConstantContextBaseline(targets)
        train = [
            {
                "event_id": "e1",
                "batter_season_id": "111_2026",
                "launch_speed": 100.0,
                "launch_angle": 12.0,
                "target_hard_hit": 1.0,
                "target_barrel": 0.0,
            },
            {
                "event_id": "e2",
                "batter_season_id": "222_2026",
                "launch_speed": 80.0,
                "launch_angle": 4.0,
                "target_hard_hit": 0.0,
                "target_barrel": 0.0,
            },
        ]
        baseline.fit(train)
        rows = baseline.predict_rows(train, run_id="ctx_test", split="train")
        validate_prediction_rows(rows)
        ev_rows = [row for row in rows if row["target_name"] == "ev"]
        self.assertEqual(ev_rows[0]["y_pred"], 90.0)
        xba_rows = [row for row in rows if row["target_name"] == "xba"]
        self.assertEqual(xba_rows[0]["target_available"], False)
        self.assertEqual(xba_rows[0]["label_missing_reason"], "label_missing")
        ops_rows = [row for row in rows if row["target_name"] == "ops"]
        self.assertEqual(len(ops_rows), 2)
        self.assertTrue(all(row["prediction_level"] == "player_season" for row in ops_rows))
        self.assertTrue(all(row["event_id"] is None for row in ops_rows))

    def test_constant_context_baseline_emits_one_ops_row_per_batter_season(self) -> None:
        targets = load_target_registry(REGISTRY_PATH)
        baseline = ConstantContextBaseline(targets)
        records = [
            {
                "event_id": "e1",
                "batter_season_id": "111_2026",
                "launch_speed": 100.0,
                "launch_angle": 12.0,
                "target_hard_hit": 1.0,
                "target_barrel": 0.0,
                "target_ops": 0.850,
            },
            {
                "event_id": "e2",
                "batter_season_id": "111_2026",
                "launch_speed": 90.0,
                "launch_angle": 8.0,
                "target_hard_hit": 0.0,
                "target_barrel": 0.0,
                "target_ops": 0.850,
            },
            {
                "event_id": "e3",
                "batter_season_id": "222_2026",
                "launch_speed": 80.0,
                "launch_angle": 4.0,
                "target_hard_hit": 0.0,
                "target_barrel": 0.0,
                "target_ops": 0.700,
            },
        ]
        baseline.fit(records)
        rows = baseline.predict_rows(records, run_id="ctx_test")
        validate_prediction_rows(rows)
        ops_rows = [row for row in rows if row["target_name"] == "ops"]
        self.assertEqual(len(ops_rows), 2)
        self.assertEqual({row["batter_season_id"] for row in ops_rows}, {"111_2026", "222_2026"})
        self.assertTrue(all(row["event_id"] is None for row in ops_rows))

    def _prediction_row(
        self,
        target_name: str = "ev",
        prediction_level: str = "event",
        event_id: str | None = "e1",
        y_true: float | None = 100.0,
        y_pred: float | None = 99.0,
        target_available: bool = True,
        target_source: str = "launch_speed",
        head_kind: str = "regression",
        loss_name: str = "huber",
        label_missing_reason: str | None = None,
        requires_pa_manifest: bool = False,
    ) -> dict:
        return {
            "run_id": "test",
            "sample_id": "sample1",
            "event_id": event_id,
            "batter_season_id": "111_2026",
            "prediction_level": prediction_level,
            "target_name": target_name,
            "y_true": y_true,
            "y_pred": y_pred,
            "target_available": target_available,
            "target_source": target_source,
            "head_kind": head_kind,
            "loss_name": loss_name,
            "aggregation_scope": "context_only",
            "prior_mode": "none",
            "label_missing_reason": label_missing_reason,
            "requires_pa_manifest": requires_pa_manifest,
            "n_prior_clips": 0,
            "aggregation_method": "none",
            "same_event_ensemble": False,
            "prediction_std": None,
        }


if __name__ == "__main__":
    unittest.main()
