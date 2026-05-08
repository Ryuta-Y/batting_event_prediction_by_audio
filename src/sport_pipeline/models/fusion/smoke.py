"""Local smoke checks for D3 fusion."""

from __future__ import annotations

import argparse
import json

from sport_pipeline.models.fusion.late_fusion import late_fuse_prediction_rows


def _row(
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
        "target_source": "statcast_column" if target_name != "ops" else "target_ops",
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


def run_fusion_smoke() -> dict:
    """Fuse event and optional missing target examples."""

    rows = [
        _row("ctx", "ctx_e1_ev", "e1", "111_2026", "ev", 97.0, aggregation_scope="context_only", prediction_std=2.0),
        _row("seq", "seq_e1_ev", "e1", "111_2026", "ev", 101.0, aggregation_scope="current_event_only", prediction_std=1.5),
        _row(
            "seq_prior",
            "seq_prior_e1_ev",
            "e1",
            "111_2026",
            "ev",
            102.0,
            aggregation_scope="current_event_with_player_season_prior",
            prior_mode="past_only",
            n_prior_clips=3,
            prediction_std=1.2,
        ),
        _row(
            "video",
            "video_e1_ev",
            "e1",
            "111_2026",
            "ev",
            99.0,
            aggregation_scope="video_frozen_encoder",
            prediction_std=2.5,
        ),
        _row(
            "same_event",
            "same_event_e1_ev",
            "e1",
            "111_2026",
            "ev",
            100.0,
            aggregation_scope="same_event_view_crop_augmentation_ensemble",
            same_event_ensemble=True,
            prediction_std=0.9,
        ),
        _row(
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
    result = late_fuse_prediction_rows(rows, fusion_run_id="fusion_smoke")
    return {
        "schema_version": "fusion_smoke_v1",
        "input_rows": len(rows),
        "prediction_rows": len(result.prediction_rows),
        "audit_rows": len(result.audit_rows),
        "targets": sorted({row["target_name"] for row in result.prediction_rows}),
        "ev_prediction_std": next(row["prediction_std"] for row in result.prediction_rows if row["target_name"] == "ev"),
        "xba_available": next(row["target_available"] for row in result.prediction_rows if row["target_name"] == "xba"),
        "source_scopes": sorted({row["source_aggregation_scope"] for row in result.audit_rows}),
        "source_prior_modes": sorted({row["source_prior_mode"] for row in result.audit_rows}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run D3 fusion smoke checks.")
    parser.parse_args()
    print(json.dumps(run_fusion_smoke(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

