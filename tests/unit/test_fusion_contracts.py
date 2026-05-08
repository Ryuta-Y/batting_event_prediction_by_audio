import unittest

from sport_pipeline.evaluation import validate_prediction_rows
from sport_pipeline.models.fusion import FUSION_INPUT_AUDIT_SCHEMA, fuse_prediction_group, late_fuse_prediction_rows
from sport_pipeline.models.fusion.smoke import run_fusion_smoke
from sport_pipeline.schemas.data_manifest import validate_rows


def prediction_row(
    run_id: str,
    sample_id: str,
    event_id: str | None,
    batter_season_id: str,
    target_name: str,
    y_pred: float | None,
    y_true: float | None = 100.0,
    prediction_level: str = "event",
    target_available: bool = True,
    aggregation_scope: str = "context_only",
    prior_mode: str = "none",
    label_missing_reason: str | None = None,
    n_prior_clips: int = 0,
    same_event_ensemble: bool = False,
    prediction_std: float | None = None,
    requires_pa_manifest: bool = False,
) -> dict:
    return {
        "run_id": run_id,
        "sample_id": sample_id,
        "event_id": event_id,
        "batter_season_id": batter_season_id,
        "prediction_level": prediction_level,
        "target_name": target_name,
        "y_true": y_true if target_available else None,
        "y_pred": y_pred if target_available else None,
        "target_available": target_available,
        "target_source": "target_ops" if target_name == "ops" else "statcast_column",
        "head_kind": "regression",
        "loss_name": "huber",
        "aggregation_scope": aggregation_scope,
        "prior_mode": prior_mode,
        "label_missing_reason": label_missing_reason,
        "requires_pa_manifest": requires_pa_manifest,
        "n_prior_clips": n_prior_clips,
        "aggregation_method": aggregation_scope,
        "same_event_ensemble": same_event_ensemble,
        "prediction_std": prediction_std,
    }


class FusionContractsTest(unittest.TestCase):
    def test_late_fusion_preserves_predictions_v1(self) -> None:
        rows = [
            prediction_row("ctx", "ctx_e1_ev", "e1", "111_2026", "ev", 97.0, aggregation_scope="context_only"),
            prediction_row("seq", "seq_e1_ev", "e1", "111_2026", "ev", 101.0, aggregation_scope="current_event_only"),
            prediction_row("vid", "vid_e1_ev", "e1", "111_2026", "ev", 99.0, aggregation_scope="video_frozen_encoder"),
        ]
        result = late_fuse_prediction_rows(rows, fusion_run_id="fusion_test")
        validate_prediction_rows(result.prediction_rows)
        validate_rows(FUSION_INPUT_AUDIT_SCHEMA, result.audit_rows)
        self.assertEqual(len(result.prediction_rows), 1)
        fused = result.prediction_rows[0]
        self.assertEqual(fused["aggregation_scope"], "late_fusion_event")
        self.assertEqual(fused["target_name"], "ev")
        self.assertTrue(fused["target_available"])
        self.assertIsNotNone(fused["prediction_std"])

    def test_missing_optional_target_stays_unavailable(self) -> None:
        rows = [
            prediction_row(
                "seq",
                "seq_e1_xba",
                "e1",
                "111_2026",
                "xba",
                None,
                y_true=None,
                target_available=False,
                aggregation_scope="current_event_only",
                label_missing_reason="statcast_expected_outcome_missing",
            ),
            prediction_row(
                "video",
                "video_e1_xba",
                "e1",
                "111_2026",
                "xba",
                None,
                y_true=None,
                target_available=False,
                aggregation_scope="video_frozen_encoder",
                label_missing_reason="statcast_expected_outcome_missing",
            ),
        ]
        result = late_fuse_prediction_rows(rows, fusion_run_id="fusion_test")
        validate_prediction_rows(result.prediction_rows)
        fused = result.prediction_rows[0]
        self.assertFalse(fused["target_available"])
        self.assertIsNone(fused["y_pred"])
        self.assertIn("statcast_expected_outcome_missing", fused["label_missing_reason"])

    def test_same_event_ensemble_and_prior_are_audited_separately(self) -> None:
        rows = [
            prediction_row(
                "same_event",
                "same_e1_ev",
                "e1",
                "111_2026",
                "ev",
                100.0,
                aggregation_scope="same_event_view_crop_augmentation_ensemble",
                same_event_ensemble=True,
                prediction_std=0.7,
            ),
            prediction_row(
                "seq_prior",
                "prior_e1_ev",
                "e1",
                "111_2026",
                "ev",
                102.0,
                aggregation_scope="current_event_with_player_season_prior",
                prior_mode="past_only",
                n_prior_clips=4,
                prediction_std=1.1,
            ),
        ]
        result = late_fuse_prediction_rows(rows, fusion_run_id="fusion_test")
        scopes = {row["source_aggregation_scope"] for row in result.audit_rows}
        prior_modes = {row["source_prior_mode"] for row in result.audit_rows}
        self.assertIn("same_event_view_crop_augmentation_ensemble", scopes)
        self.assertIn("current_event_with_player_season_prior", scopes)
        self.assertIn("past_only", prior_modes)
        fused = result.prediction_rows[0]
        self.assertEqual(fused["n_prior_clips"], 4)
        self.assertIn("mixed:", fused["prior_mode"])
        self.assertEqual(fused["aggregation_scope"], "late_fusion_event")

    def test_fusion_group_rejects_mixed_events(self) -> None:
        rows = [
            prediction_row("ctx", "ctx_e1_ev", "e1", "111_2026", "ev", 97.0),
            prediction_row("ctx", "ctx_e2_ev", "e2", "111_2026", "ev", 98.0),
        ]
        with self.assertRaises(ValueError):
            fuse_prediction_group(rows, fusion_run_id="fusion_test")

    def test_late_fusion_keeps_different_events_as_separate_rows(self) -> None:
        rows = [
            prediction_row("ctx", "ctx_e1_ev", "e1", "111_2026", "ev", 97.0),
            prediction_row("seq", "seq_e1_ev", "e1", "111_2026", "ev", 101.0),
            prediction_row("ctx", "ctx_e2_ev", "e2", "111_2026", "ev", 88.0),
            prediction_row("seq", "seq_e2_ev", "e2", "111_2026", "ev", 89.0),
        ]
        result = late_fuse_prediction_rows(rows, fusion_run_id="fusion_test")
        self.assertEqual(len(result.prediction_rows), 2)
        self.assertEqual({row["event_id"] for row in result.prediction_rows}, {"e1", "e2"})

    def test_player_season_ops_only_when_upstream_available(self) -> None:
        rows = [
            prediction_row(
                "ctx_ps",
                "ctx_111_ops",
                None,
                "111_2026",
                "ops",
                0.810,
                y_true=0.820,
                prediction_level="player_season",
                aggregation_scope="player_season_aggregate",
                requires_pa_manifest=True,
            ),
            prediction_row(
                "seq_ps",
                "seq_111_ops",
                None,
                "111_2026",
                "ops",
                0.830,
                y_true=0.820,
                prediction_level="player_season",
                aggregation_scope="player_season_aggregate",
                requires_pa_manifest=True,
            ),
        ]
        result = late_fuse_prediction_rows(rows, fusion_run_id="fusion_test")
        validate_prediction_rows(result.prediction_rows)
        fused = result.prediction_rows[0]
        self.assertEqual(fused["prediction_level"], "player_season")
        self.assertEqual(fused["target_name"], "ops")
        self.assertTrue(fused["target_available"])
        self.assertTrue(fused["requires_pa_manifest"])

    def test_fusion_smoke(self) -> None:
        summary = run_fusion_smoke()
        self.assertEqual(summary["prediction_rows"], 2)
        self.assertIn("ev", summary["targets"])
        self.assertIn("xba", summary["targets"])
        self.assertFalse(summary["xba_available"])
        self.assertIn("current_event_with_player_season_prior", summary["source_scopes"])


if __name__ == "__main__":
    unittest.main()

