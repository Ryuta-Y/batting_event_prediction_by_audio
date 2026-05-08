"""Run-profile helpers for Colab real-data executions."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FULL_RUN_PROFILE = PROJECT_ROOT / "configs/runs/full_colab_v1.json"
DEFAULT_REAL_RUN_PROFILE = PROJECT_ROOT / "configs/runs/mlb_2024_2026_real_colab_v1.json"
DEFAULT_ARTIFACT_NAMESPACE = {
    "structured_sequence_feature_id": "structured_sequence_v1",
    "clip_embedding_feature_id": "clip_embedding_v1",
    "player_season_embedding_feature_id": "player_season_embedding_v1",
    "sequence_dataset_id": "sequence_dataset_v1",
    "event_with_prior_dataset_id": "event_with_player_prior_v1",
    "video_lightweight_feature_id": "video_lightweight_features_v1",
    "video_embedding_feature_id": "video_embedding_v1",
    "image_embedding_feature_id": "image_embedding_v1",
    "vlm_feature_id": "vlm_mechanics_v1",
    "audio_raw_feature_id": "audio_impact_v1",
    "audio_enhanced_feature_id": "audio_enhanced_impact_v1",
    "audio_separated_feature_id": "audio_separated_impact_v1",
    "audio_embedding_feature_id": "audio_embedding_impact_v1",
}


def load_run_profile(path: str | Path | None = None) -> dict[str, Any]:
    """Load a JSON run profile.

    The real-data profile is the default because the smoke/pilot notebooks can
    still explicitly point at ``full_colab_v1.json`` when needed.
    """

    profile_path = Path(path) if path is not None else DEFAULT_REAL_RUN_PROFILE
    return json.loads(profile_path.read_text(encoding="utf-8"))


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def resolve_date_value(value: str, *, today: date | None = None) -> str:
    """Resolve date tokens such as ``today`` to ISO dates."""

    current = today or date.today()
    if value == "today":
        return current.isoformat()
    parsed = _parse_date(value)
    if parsed > current:
        return current.isoformat()
    return parsed.isoformat()


def resolve_statcast_date_range(profile: dict[str, Any], *, today: date | None = None) -> tuple[str, str]:
    """Return the configured Statcast start/end date range."""

    window = profile.get("data_window", {})
    start = resolve_date_value(str(window.get("start_date", "2024-03-20")), today=today)
    end = resolve_date_value(str(window.get("end_date", "today")), today=today)
    if _parse_date(end) < _parse_date(start):
        raise ValueError(f"run profile date window is invalid: start={start}, end={end}")
    return start, end


def run_id(profile: dict[str, Any], name: str, default: str | None = None) -> str:
    """Read a run id from ``profile['run_ids']``."""

    value = profile.get("run_ids", {}).get(name, default)
    if value is None:
        raise KeyError(f"missing run id: {name}")
    return str(value)


def artifact_namespace(profile: dict[str, Any]) -> dict[str, str]:
    """Return artifact namespace ids for shared feature/dataset locations."""

    namespace = dict(DEFAULT_ARTIFACT_NAMESPACE)
    raw_namespace = profile.get("artifact_namespace", {})
    if isinstance(raw_namespace, dict):
        for key, value in raw_namespace.items():
            if value is not None:
                namespace[str(key)] = str(value)
    return namespace


def artifact_id(profile: dict[str, Any], name: str, default: str | None = None) -> str:
    """Read an artifact namespace id with stable v1 defaults."""

    namespace = artifact_namespace(profile)
    value = namespace.get(name, default)
    if value is None:
        raise KeyError(f"missing artifact namespace id: {name}")
    return str(value)


def stage_settings(profile: dict[str, Any], stage: str) -> dict[str, Any]:
    """Return per-stage execution settings from a run profile."""

    return dict(profile.get("execution", {}).get(stage, {}))


def threshold(profile: dict[str, Any], name: str, default: int = 0) -> int:
    """Return an integer readiness threshold."""

    value = profile.get("readiness_thresholds", {}).get(name, default)
    return int(value) if value is not None else default
