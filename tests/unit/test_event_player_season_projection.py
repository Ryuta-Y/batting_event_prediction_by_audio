from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from sport_pipeline.io import read_table, write_table
from sport_pipeline.models.player_season.event_projection import (
    projection_run_id,
    run_event_prediction_player_season_projection,
)


def _event(event_id: str, batter_id: int, ev: float, la: float) -> dict:
    return {
        "event_id": event_id,
        "game_date": "2024-04-01",
        "season": 2024,
        "batter_id": batter_id,
        "batter_season_id": f"{batter_id}_2024",
        "launch_speed": ev,
        "launch_angle": la,
        "target_hard_hit": float(ev >= 95.0),
        "target_barrel": float(ev >= 98.0 and 8.0 <= la <= 32.0),
        "estimated_ba_using_speedangle": 0.340 if ev >= 95.0 else 0.220,
        "estimated_woba_using_speedangle": 0.430 if ev >= 95.0 else 0.270,
    }


def _prediction(run_id: str, event: dict, target_name: str, y_pred: float) -> dict:
    y_true_by_target = {
        "ev": event["launch_speed"],
        "la": event["launch_angle"],
        "hard_hit": event["target_hard_hit"],
        "barrel": event["target_barrel"],
        "xba": event["estimated_ba_using_speedangle"],
        "xwoba": event["estimated_woba_using_speedangle"],
    }
    return {
        "run_id": run_id,
        "sample_id": f"{event['event_id']}__{target_name}",
        "event_id": event["event_id"],
        "batter_season_id": event["batter_season_id"],
        "prediction_level": "event",
        "target_name": target_name,
        "y_true": y_true_by_target[target_name],
        "y_pred": y_pred,
        "target_available": True,
        "target_source": target_name,
        "head_kind": "binary" if target_name in {"hard_hit", "barrel"} else "regression",
        "loss_name": "bce" if target_name in {"hard_hit", "barrel"} else "huber",
        "aggregation_scope": "raw_video_finetune",
        "prior_mode": "none",
        "label_missing_reason": None,
        "requires_pa_manifest": False,
        "n_prior_clips": 0,
        "aggregation_method": "unit_test",
        "same_event_ensemble": False,
        "prediction_std": None,
        "split": "train",
    }


class EventPlayerSeasonProjectionTests(unittest.TestCase):
    def test_event_method_projection_emits_ba_ops_player_season_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            events = [
                _event("e1", 111, 101.0, 18.0),
                _event("e2", 111, 92.0, 7.0),
                _event("e3", 222, 97.0, 22.0),
                _event("e4", 222, 86.0, -3.0),
            ]
            write_table(base / "manifests/bbe_events_v1.jsonl", events)
            write_table(
                base / "manifests/player_season_batting_v1.jsonl",
                [
                    {"batter_season_id": "111_2024", "target_ba": 0.300, "target_obp": 0.360, "target_slg": 0.520, "target_ops": 0.880},
                    {"batter_season_id": "222_2024", "target_ba": 0.245, "target_obp": 0.310, "target_slg": 0.410, "target_ops": 0.720},
                ],
            )
            run_id = "raw_video_test_v1"
            rows = []
            for event in events:
                rows.extend(
                    [
                        _prediction(run_id, event, "ev", float(event["launch_speed"]) - 1.0),
                        _prediction(run_id, event, "la", float(event["launch_angle"]) + 1.0),
                        _prediction(run_id, event, "hard_hit", float(event["target_hard_hit"])),
                        _prediction(run_id, event, "barrel", float(event["target_barrel"])),
                        _prediction(run_id, event, "xba", float(event["estimated_ba_using_speedangle"])),
                        _prediction(run_id, event, "xwoba", float(event["estimated_woba_using_speedangle"])),
                    ]
                )
            write_table(base / f"predictions/{run_id}/predictions_v1.jsonl", rows)

            outputs = run_event_prediction_player_season_projection(
                base,
                source_run_id=run_id,
                player_season_batting_stats=base / "manifests/player_season_batting_v1.jsonl",
                output_suffix=".jsonl",
                require_non_empty=True,
            )

            self.assertEqual(outputs["predictions"].parent.name, projection_run_id(run_id))
            projected = read_table(outputs["predictions"])
            self.assertTrue(any(row["target_name"] == "avg_ev" and row["target_available"] for row in projected))
            self.assertTrue(any(row["target_name"] == "ba" and row["target_available"] for row in projected))
            self.assertTrue(any(row["target_name"] == "ops" and row["target_available"] for row in projected))
            metrics = outputs["metrics"].read_text(encoding="utf-8")
            self.assertIn("player_season", metrics)
            self.assertIn("ba", metrics)
            self.assertIn("ops", metrics)


if __name__ == "__main__":
    unittest.main()
