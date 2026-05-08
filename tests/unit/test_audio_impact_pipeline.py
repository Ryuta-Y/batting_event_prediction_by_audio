from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest
import wave

from sport_pipeline.audio.audit import run_audio_presence_audit
from sport_pipeline.audio.impact import extract_audio_impact_features, run_audio_impact_baseline
from sport_pipeline.io import read_table, write_table
from sport_pipeline.reports.audio_baseline_compare import write_audio_baseline_comparison_report

try:
    import numpy  # noqa: F401
except ModuleNotFoundError:
    NUMPY_AVAILABLE = False
else:
    NUMPY_AVAILABLE = True


def _write_wav(path: Path, *, impulse_time: float, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    duration = 1.0
    n_samples = int(duration * sample_rate)
    values = []
    impulse_index = int(impulse_time * sample_rate)
    for index in range(n_samples):
        tone = 0.03 * math.sin(2.0 * math.pi * 440.0 * index / sample_rate)
        impact = 0.0
        distance = abs(index - impulse_index)
        if distance < 18:
            impact = 0.8 * (1.0 - distance / 18.0)
        sample = max(-1.0, min(1.0, tone + impact))
        values.append(int(sample * 32767.0))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"".join(int(value).to_bytes(2, byteorder="little", signed=True) for value in values))


def _event(event_id: str, ev: float, la: float, split: str) -> dict:
    return {
        "event_id": event_id,
        "season": 2024,
        "batter_id": 111,
        "batter_season_id": "111_2024",
        "launch_speed": ev,
        "launch_angle": la,
        "target_hard_hit": float(ev >= 95.0),
        "target_barrel": float(ev >= 98.0 and 8.0 <= la <= 32.0),
        "estimated_ba_using_speedangle": 0.340 if ev >= 95.0 else 0.220,
        "estimated_woba_using_speedangle": 0.430 if ev >= 95.0 else 0.270,
        "split": split,
    }


def _clip(event_id: str, path: Path, split: str) -> dict:
    return {
        "clip_id": f"clip_{event_id}",
        "event_id": event_id,
        "same_event_group_id": event_id,
        "batter_id": 111,
        "season": 2024,
        "batter_season_id": "111_2024",
        "clip_status": "clean_clip",
        "quality_tier": "usable_primary",
        "clip_path": str(path),
        "contact_time_sec": 0.40,
        "contact_confidence": 0.95,
        "view_confidence": 0.80,
        "split": split,
    }


def _prediction(run_id: str, event_id: str, target_name: str, y_true: float, y_pred: float) -> dict:
    return {
        "run_id": run_id,
        "sample_id": f"{event_id}_{target_name}",
        "event_id": event_id,
        "batter_season_id": "111_2024",
        "prediction_level": "event",
        "target_name": target_name,
        "y_true": y_true,
        "y_pred": y_pred,
        "target_available": True,
        "target_source": "unit",
        "head_kind": "regression",
        "loss_name": "huber",
        "aggregation_scope": run_id,
        "prior_mode": "none",
        "label_missing_reason": None,
        "requires_pa_manifest": False,
    }


@unittest.skipIf(not NUMPY_AVAILABLE, "numpy is required for audio feature extraction tests")
class AudioImpactPipelineTests(unittest.TestCase):
    def test_audio_impact_feature_extraction_from_wav(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "impact.wav"
            _write_wav(wav_path, impulse_time=0.40)
            segment, feature = extract_audio_impact_features(
                {**_clip("e1", wav_path, "train"), "_resolved_clip_path": str(wav_path)},
                preprocessing_mode="raw",
            )
            self.assertEqual(segment["audio_status"], "complete")
            self.assertEqual(feature["schema_version"], "audio_features_v1")
            self.assertGreater(feature["impact_confidence"], 0.0)
            self.assertEqual(feature["feature_dim"], len(feature["feature_values"]))

    def test_audio_impact_baseline_writes_predictions_report_and_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            wav1 = base / "audio/e1.wav"
            wav2 = base / "audio/e2.wav"
            wav3 = base / "audio/e3.wav"
            _write_wav(wav1, impulse_time=0.40)
            _write_wav(wav2, impulse_time=0.40)
            _write_wav(wav3, impulse_time=0.40)
            write_table(
                base / "manifests/bbe_events_v1.jsonl",
                [
                    _event("e1", 102.0, 18.0, "train"),
                    _event("e2", 91.0, 24.0, "train"),
                    _event("e3", 98.0, 12.0, "validation"),
                ],
            )
            write_table(
                base / "clips/full_v1/clips_v1.jsonl",
                [
                    _clip("e1", wav1, "train"),
                    _clip("e2", wav2, "train"),
                    {key: value for key, value in _clip("e3", wav3, "validation").items() if key != "split"},
                ],
            )

            outputs = run_audio_impact_baseline(
                base,
                clip_run_id="full_v1",
                prediction_run_id="audio_raw_unit_v1",
                audio_feature_id="audio_unit_v1",
                bbe_events=base / "manifests/bbe_events_v1.jsonl",
                clips_path=base / "clips/full_v1/clips_v1.jsonl",
                preprocessing_mode="raw",
                model_family="ridge",
                require_non_empty=True,
                output_suffix=".jsonl",
            )

            rows = read_table(outputs["predictions"])
            self.assertTrue(rows)
            self.assertTrue(all(row["aggregation_scope"] == "audio_raw_impact" for row in rows))
            e3_rows = [row for row in rows if row["event_id"] == "e3"]
            self.assertTrue(e3_rows)
            self.assertTrue(all(row["split"] == "validation" for row in e3_rows))
            self.assertTrue(outputs["audio_segments"].exists())
            self.assertTrue(outputs["audio_features"].exists())
            self.assertTrue(outputs["audio_report_html"].exists())
            self.assertTrue(outputs["audio_system_diagram"].exists())
            summary = json.loads(outputs["summary"].read_text(encoding="utf-8"))
            self.assertEqual(summary["sample_rows"], 3)
            self.assertEqual(summary["prediction_rows"], len(rows))

    def test_audio_impact_baseline_can_filter_empty_audio_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            wav1 = base / "audio/e1.wav"
            wav2 = base / "audio/e2.wav"
            wav3 = base / "audio/e3.wav"
            _write_wav(wav1, impulse_time=0.40)
            _write_wav(wav2, impulse_time=0.40)
            _write_wav(wav3, impulse_time=0.05)
            invalid = _clip("e3", wav3, "validation")
            invalid["contact_time_sec"] = 2.0
            write_table(
                base / "manifests/bbe_events_v1.jsonl",
                [
                    _event("e1", 102.0, 18.0, "train"),
                    _event("e2", 91.0, 24.0, "train"),
                    _event("e3", 98.0, 12.0, "validation"),
                ],
            )
            write_table(base / "clips/full_v1/clips_v1.jsonl", [_clip("e1", wav1, "train"), _clip("e2", wav2, "train"), invalid])

            outputs = run_audio_impact_baseline(
                base,
                clip_run_id="full_v1",
                prediction_run_id="audio_raw_valid_only_unit_v1",
                audio_feature_id="audio_valid_only_unit_v1",
                bbe_events=base / "manifests/bbe_events_v1.jsonl",
                clips_path=base / "clips/full_v1/clips_v1.jsonl",
                preprocessing_mode="raw",
                model_family="ridge",
                require_non_empty=True,
                valid_audio_only=True,
                output_suffix=".jsonl",
            )

            rows = read_table(outputs["predictions"])
            self.assertTrue(rows)
            self.assertFalse([row for row in rows if row["event_id"] == "e3"])
            summary = json.loads(outputs["summary"].read_text(encoding="utf-8"))
            self.assertEqual(summary["sample_rows"], 2)
            self.assertEqual(summary["invalid_audio_rows_filtered"], 1)

    def test_audio_presence_audit_writes_valid_clip_manifest_and_preview(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            wav1 = base / "audio/e1.wav"
            wav2 = base / "audio/e2.wav"
            _write_wav(wav1, impulse_time=0.40)
            _write_wav(wav2, impulse_time=0.05)
            invalid = _clip("e2", wav2, "validation")
            invalid["contact_time_sec"] = 2.0
            write_table(base / "clips/full_v1/clips_v1.jsonl", [_clip("e1", wav1, "train"), invalid])

            outputs = run_audio_presence_audit(
                base,
                clip_run_id="full_v1",
                audit_id="audio_presence_unit_v1",
                clips_path=base / "clips/full_v1/clips_v1.jsonl",
                output_suffix=".jsonl",
                preview_examples=1,
            )

            valid_rows = read_table(outputs["audio_valid_clips"])
            audit_rows = read_table(outputs["audio_presence_manifest"])
            self.assertEqual(len(valid_rows), 1)
            self.assertEqual(valid_rows[0]["event_id"], "e1")
            self.assertEqual(len(audit_rows), 2)
            self.assertTrue(outputs["audio_audit_html"].exists())
            summary = json.loads(outputs["audio_audit_summary"].read_text(encoding="utf-8"))
            self.assertEqual(summary["valid_audio_clips"], 1)
            self.assertEqual(summary["preview_examples"], 1)

    def test_audio_baseline_compare_uses_pairwise_intersection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            audio_run = "audio_unit"
            baseline_run = "baseline_unit"
            write_table(
                base / "predictions" / audio_run / "predictions_v1.jsonl",
                [
                    _prediction(audio_run, "e1", "ev", 100.0, 101.0),
                    _prediction(audio_run, "e2", "ev", 90.0, 91.0),
                    _prediction(audio_run, "e3", "ev", 95.0, 96.0),
                ],
            )
            write_table(
                base / "predictions" / baseline_run / "predictions_v1.jsonl",
                [
                    _prediction(baseline_run, "e1", "ev", 100.0, 110.0),
                    _prediction(baseline_run, "e2", "ev", 90.0, 80.0),
                    _prediction(baseline_run, "e3", "ev", 95.0, 105.0),
                    _prediction(baseline_run, "other", "ev", 99.0, 60.0),
                ],
            )

            outputs = write_audio_baseline_comparison_report(
                base,
                report_id="audio_compare_unit",
                audio_run_ids=[audio_run],
                baseline_run_ids=[baseline_run],
                output_suffix=".jsonl",
                min_intersection=3,
            )

            rows = read_table(outputs["pairwise_metrics"])
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["target_name"], "ev")
            self.assertEqual(int(rows[0]["intersection_samples"]), 3)
            self.assertEqual(rows[0]["winner"], "audio")
            self.assertTrue(outputs["audio_baseline_compare_html"].exists())


if __name__ == "__main__":
    unittest.main()
