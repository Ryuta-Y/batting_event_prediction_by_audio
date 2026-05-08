"""Shared Colab paths and audio-release artifact checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


REPO_DIR_DEFAULT = Path("/content/drive/MyDrive/codex/batting_codex_handoff_with_audio")
BASE_DIR_DEFAULT = Path("/content/drive/MyDrive/baseball_vision")
CACHE_DIR_DEFAULT = Path("/content/cache/baseball_vision")

AUDIO_ARTIFACT_DIRECTORIES = (
    "manifests",
    "clips",
    "raw_videos",
    "features",
    "datasets",
    "datasets/audio_feature_samples",
    "predictions",
    "reports",
    "reports/preflight",
    "reports/audio_impact",
    "reports/audio_audit",
    "reports/audio_baseline_compare",
    "reports/method_evaluation",
    "debug",
    "logs",
)

EXPECTED_ARTIFACTS = {
    "audio_inputs": (
        "manifests/bbe_events_v1.parquet",
        "clips/{full_run_id}/clips_v1.parquet",
    ),
    "audio_impact": (
        "features/{audio_raw_feature_id}/segments_v1.parquet",
        "features/{audio_raw_feature_id}/manifest.parquet",
        "predictions/{audio_raw_run_id}/predictions_v1.parquet",
        "predictions/{audio_raw_run_id}/metrics_v1.json",
        "features/{audio_enhanced_feature_id}/manifest.parquet",
        "predictions/{audio_enhanced_run_id}/predictions_v1.parquet",
        "predictions/{audio_enhanced_run_id}/metrics_v1.json",
    ),
    "audio_gpu": (
        "features/{audio_separated_feature_id}/manifest.parquet",
        "predictions/{audio_separated_run_id}/predictions_v1.parquet",
        "features/{audio_embedding_feature_id}/manifest.parquet",
        "predictions/{audio_embedding_run_id}/predictions_v1.parquet",
    ),
    "audio_fusion": (
        "predictions/{fusion_audio_run_id}/predictions_v1.parquet",
        "predictions/{fusion_audio_run_id}/metrics_v1.json",
        "predictions/{fusion_audio_run_id}/fusion_input_audit_v1.parquet",
        "reports/method_evaluation/{method_evaluation_audio_report_id}/index.html",
    ),
    "audio_audit": (
        "features/{audio_audit_feature_id}/audio_presence_manifest.parquet",
        "features/{audio_audit_feature_id}/audio_valid_clips_v1.parquet",
        "reports/audio_audit/{audio_presence_audit_id}/index.html",
        "reports/audio_baseline_compare/{audio_baseline_compare_report_id}/index.html",
    ),
}

DEFAULT_EXPECTED_ARTIFACT_IDS = {
    "full_run_id": "mlb_2024_2026_full_v2",
    "audio_raw_feature_id": "audio_impact_mlb_2024_2026_v2",
    "audio_enhanced_feature_id": "audio_enhanced_impact_mlb_2024_2026_v2",
    "audio_separated_feature_id": "audio_separated_impact_mlb_2024_2026_v2",
    "audio_embedding_feature_id": "audio_embedding_impact_mlb_2024_2026_v2",
    "audio_audit_feature_id": "audio_presence_mlb_2024_2026_v2",
    "audio_raw_run_id": "audio_raw_impact_mlb_2024_2026_v2",
    "audio_enhanced_run_id": "audio_enhanced_impact_mlb_2024_2026_v2",
    "audio_separated_run_id": "audio_separated_impact_mlb_2024_2026_v2",
    "audio_embedding_run_id": "audio_embedding_impact_mlb_2024_2026_v2",
    "fusion_audio_run_id": "fusion_audio_mlb_2024_2026_v2",
    "method_evaluation_audio_report_id": "method_evaluation_with_audio_mlb_2024_2026_v2",
    "audio_presence_audit_id": "audio_presence_mlb_2024_2026_v2",
    "audio_baseline_compare_report_id": "audio_baseline_compare_mlb_2024_2026_v2",
}


@dataclass(frozen=True)
class ColabPaths:
    repo_dir: Path = REPO_DIR_DEFAULT
    base_dir: Path = BASE_DIR_DEFAULT
    cache_dir: Path = CACHE_DIR_DEFAULT

    @classmethod
    def from_values(
        cls,
        repo_dir: str | Path | None = None,
        base_dir: str | Path | None = None,
        cache_dir: str | Path | None = None,
    ) -> "ColabPaths":
        return cls(
            repo_dir=Path(repo_dir) if repo_dir is not None else REPO_DIR_DEFAULT,
            base_dir=Path(base_dir) if base_dir is not None else BASE_DIR_DEFAULT,
            cache_dir=Path(cache_dir) if cache_dir is not None else CACHE_DIR_DEFAULT,
        )


def ensure_artifact_directories(
    base_dir: str | Path = BASE_DIR_DEFAULT,
    cache_dir: str | Path = CACHE_DIR_DEFAULT,
    directories: Iterable[str] = AUDIO_ARTIFACT_DIRECTORIES,
) -> list[Path]:
    """Create the standard Drive and cache artifact directories for audio work."""

    created: list[Path] = []
    for root in (Path(base_dir), Path(cache_dir)):
        for relative in directories:
            path = root / relative
            path.mkdir(parents=True, exist_ok=True)
            created.append(path)
    return created


def expected_artifacts_for_stage(
    stage: str,
    run_id: str | None = None,
    *,
    artifact_ids: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Return expected artifact paths for a named audio-release stage."""

    if stage not in EXPECTED_ARTIFACTS:
        raise KeyError(f"Unknown audio artifact stage: {stage}")
    format_values = dict(DEFAULT_EXPECTED_ARTIFACT_IDS)
    if artifact_ids:
        format_values.update({str(key): str(value) for key, value in artifact_ids.items()})
    if run_id is not None:
        format_values["run_id"] = run_id
    resolved = []
    for template in EXPECTED_ARTIFACTS[stage]:
        try:
            resolved.append(template.format(**format_values))
        except KeyError:
            resolved.append(template)
    return tuple(resolved)
