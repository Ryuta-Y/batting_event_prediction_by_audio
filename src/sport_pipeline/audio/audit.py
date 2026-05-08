"""Audio-presence audit and QA figures for batting clips.

This module is intentionally Colab-friendly: it uses ``ffprobe``/``ffmpeg``
when available, writes compact manifests, and keeps any recovered audio windows
as small wav files under the Drive artifact tree.
"""

from __future__ import annotations

from collections import Counter
import csv
import json
import math
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterable

from sport_pipeline.artifact_check import write_json
from sport_pipeline.audio.impact import (
    AudioImpactConfig,
    _config_from_payload,
    _contact_time_sec,
    _enhance_transient,
    _load_audio_context,
    _load_json_config,
    _resolve_clip_path,
    _select_audio_clips,
    _slice_samples,
)
from sport_pipeline.io import read_table, write_table
from sport_pipeline.reports.html import html_escape, render_kv_table, render_page, render_table, write_page


DEFAULT_MODEL_CONFIG = Path(__file__).resolve().parents[3] / "configs/models/audio/audio_impact_baseline_v1.json"


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> Path:
    row_list = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in row_list:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row_list)
    return path


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _safe_id(value: Any) -> str:
    raw = str(value or "unknown")
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw)[:96]


def _path_from_row(row: dict[str, Any], base: Path, keys: tuple[str, ...]) -> Path | None:
    for key in keys:
        value = row.get(key)
        if value in (None, ""):
            continue
        path = Path(str(value))
        if not path.is_absolute():
            path = base / path
        if path.exists():
            return path
    return None


def _ffprobe(path: Path) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return {"probe_status": "ffprobe_missing", "has_audio_stream": None}
    command = [
        ffprobe,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    proc = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        return {
            "probe_status": "ffprobe_failed",
            "has_audio_stream": None,
            "probe_error": proc.stderr.strip()[:240],
        }
    try:
        payload = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        return {"probe_status": "ffprobe_json_failed", "has_audio_stream": None, "probe_error": str(exc)}
    streams = payload.get("streams") or []
    audio_streams = [stream for stream in streams if stream.get("codec_type") == "audio"]
    first = audio_streams[0] if audio_streams else {}
    fmt = payload.get("format") or {}
    return {
        "probe_status": "ok",
        "has_audio_stream": bool(audio_streams),
        "audio_codec": first.get("codec_name"),
        "audio_sample_rate": first.get("sample_rate"),
        "audio_channels": first.get("channels"),
        "audio_duration_sec": first.get("duration") or fmt.get("duration"),
        "media_duration_sec": fmt.get("duration"),
        "media_size_bytes": fmt.get("size"),
    }


def _window_stats(row: dict[str, Any], config: AudioImpactConfig, *, preprocessing_mode: str = "raw") -> dict[str, Any]:
    try:
        context_samples, sr, contact_offset_sec, source_window_start_sec = _load_audio_context(row, config)
        start_sec = max(0.0, contact_offset_sec + config.window_start_sec)
        end_sec = max(start_sec + 0.001, contact_offset_sec + config.window_end_sec)
        impact = _slice_samples(context_samples, sr, start_sec, end_sec)
        impact = _enhance_transient(impact, sample_rate=sr, mode=preprocessing_mode)
    except Exception as exc:
        return {
            "audio_window_status": "audio_window_failed",
            "audio_window_error": str(exc)[:240],
            "audio_num_samples": 0,
            "window_rms": 0.0,
            "window_peak_abs": 0.0,
        }
    if len(impact) == 0:
        return {
            "audio_window_status": "empty_audio_window",
            "source_window_start_sec": source_window_start_sec,
            "contact_offset_sec": contact_offset_sec,
            "audio_num_samples": 0,
            "window_rms": 0.0,
            "window_peak_abs": 0.0,
        }
    import numpy as np  # type: ignore

    arr = np.asarray(impact, dtype="float32")
    rms = float(np.sqrt(np.mean(arr**2)))
    peak = float(np.max(np.abs(arr)))
    return {
        "audio_window_status": "complete",
        "source_window_start_sec": source_window_start_sec,
        "contact_offset_sec": contact_offset_sec,
        "audio_num_samples": int(len(arr)),
        "window_rms": rms,
        "window_peak_abs": peak,
        "window_duration_ms": float(len(arr) / sr * 1000.0),
    }


def _source_contact_time_sec(row: dict[str, Any]) -> float | None:
    for key in (
        "source_contact_time_sec",
        "video_contact_time_sec",
        "absolute_contact_time_sec",
        "global_contact_time_sec",
    ):
        if row.get(key) not in (None, ""):
            return max(0.0, _to_float(row.get(key)))
    for key in ("start_time_sec", "clip_start_sec", "trim_start_time_sec"):
        if row.get(key) not in (None, ""):
            return max(0.0, _to_float(row.get(key)) + _contact_time_sec(row))
    return None


def _candidate_source_path(row: dict[str, Any], base: Path) -> Path | None:
    return _path_from_row(
        row,
        base,
        (
            "raw_video_path",
            "source_video_path",
            "source_media_path",
            "video_path",
            "download_path",
            "media_path",
            "resolved_video_path",
        ),
    )


def _recover_audio_window(
    row: dict[str, Any],
    *,
    base: Path,
    config: AudioImpactConfig,
    output_dir: Path,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    source = _candidate_source_path(row, base)
    if source is None:
        return None, {"recovery_status": "no_source_video_path"}
    source_probe = _ffprobe(source)
    if source_probe.get("has_audio_stream") is not True:
        return None, {"recovery_status": "source_has_no_audio", **{f"source_{k}": v for k, v in source_probe.items()}}
    contact_sec = _source_contact_time_sec(row)
    if contact_sec is None:
        return None, {"recovery_status": "source_contact_time_missing", **{f"source_{k}": v for k, v in source_probe.items()}}
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None, {"recovery_status": "ffmpeg_missing", **{f"source_{k}": v for k, v in source_probe.items()}}
    start_sec = max(0.0, contact_sec + config.window_start_sec)
    duration_sec = max(0.001, config.window_end_sec - config.window_start_sec)
    output_dir.mkdir(parents=True, exist_ok=True)
    recovered_path = output_dir / f"{_safe_id(row.get('clip_id'))}.wav"
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_sec:.3f}",
        "-i",
        str(source),
        "-t",
        f"{duration_sec:.3f}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(config.sample_rate),
        "-f",
        "wav",
        str(recovered_path),
    ]
    proc = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0 or not recovered_path.exists():
        return None, {
            "recovery_status": "ffmpeg_recovery_failed",
            "recovery_error": proc.stderr.strip()[:240],
            **{f"source_{k}": v for k, v in source_probe.items()},
        }
    enriched = dict(row)
    enriched["audio_path"] = str(recovered_path)
    enriched["contact_time_sec"] = abs(config.window_start_sec)
    enriched["audio_recovery_status"] = "recovered_from_source_video"
    enriched["audio_recovery_source_path"] = str(source)
    return enriched, {
        "recovery_status": "recovered_from_source_video",
        "recovered_audio_path": str(recovered_path),
        **{f"source_{k}": v for k, v in source_probe.items()},
    }


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    status_counts = Counter(str(row.get("final_audio_status", "unknown")) for row in rows)
    window_counts = Counter(str(row.get("audio_window_status", "unknown")) for row in rows)
    stream_counts = Counter(str(row.get("has_audio_stream")) for row in rows)
    output = []
    for key, value in sorted(status_counts.items(), key=lambda item: (-item[1], item[0])):
        output.append({"summary_group": "final_audio_status", "key": key, "rows": value})
    for key, value in sorted(window_counts.items(), key=lambda item: (-item[1], item[0])):
        output.append({"summary_group": "audio_window_status", "key": key, "rows": value})
    for key, value in sorted(stream_counts.items(), key=lambda item: (-item[1], item[0])):
        output.append({"summary_group": "has_audio_stream", "key": key, "rows": value})
    return output


def _svg_bar(path: Path, rows: list[dict[str, Any]], *, title: str, label_key: str, value_key: str) -> Path | None:
    values = [(str(row.get(label_key)), _to_float(row.get(value_key))) for row in rows if row.get(value_key) not in (None, "")]
    if not values:
        return None
    width = 940
    left = 280
    top = 60
    bar_h = 28
    gap = 10
    height = top + len(values) * (bar_h + gap) + 35
    max_value = max(value for _label, value in values) or 1.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="34" font-family="Arial" font-size="22" font-weight="700" fill="#17202a">{html_escape(title)}</text>',
    ]
    for index, (label, value) in enumerate(values):
        y = top + index * (bar_h + gap)
        bar_w = int((value / max_value) * (width - left - 120))
        parts.append(f'<text x="24" y="{y + 19}" font-family="Arial" font-size="13" fill="#17202a">{html_escape(label[:44])}</text>')
        parts.append(f'<rect x="{left}" y="{y}" width="{max(1, bar_w)}" height="{bar_h}" rx="3" fill="#0f766e"/>')
        parts.append(f'<text x="{left + max(1, bar_w) + 8}" y="{y + 19}" font-family="Arial" font-size="13" fill="#17202a">{value:.4g}</text>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def _svg_waveform_spectrogram(path: Path, row: dict[str, Any], config: AudioImpactConfig) -> Path | None:
    try:
        context_samples, sr, contact_offset_sec, _source_window_start_sec = _load_audio_context(row, config)
        start_sec = max(0.0, contact_offset_sec + config.window_start_sec)
        end_sec = max(start_sec + 0.001, contact_offset_sec + config.window_end_sec)
        raw = _slice_samples(context_samples, sr, start_sec, end_sec)
        enhanced = _enhance_transient(raw, sample_rate=sr, mode="enhanced")
    except Exception:
        return None
    if len(raw) == 0:
        return None
    import numpy as np  # type: ignore

    raw_arr = np.asarray(raw, dtype="float32")
    enhanced_arr = np.asarray(enhanced, dtype="float32")
    width = 1080
    height = 520
    margin = 48
    plot_w = width - margin * 2

    def points(arr: Any, y_mid: float, amp: float) -> str:
        values = np.asarray(arr, dtype="float32")
        max_abs = float(np.max(np.abs(values))) if len(values) else 1.0
        max_abs = max(max_abs, 1e-6)
        step = max(1, int(math.ceil(len(values) / 480)))
        sampled = values[::step]
        pts = []
        for index, value in enumerate(sampled):
            x = margin + (index / max(1, len(sampled) - 1)) * plot_w
            y = y_mid - (float(value) / max_abs) * amp
            pts.append(f"{x:.1f},{y:.1f}")
        return " ".join(pts)

    frame = max(64, int(sr * 0.025))
    hop = max(16, int(sr * 0.010))
    frames = []
    for start in range(0, max(1, len(raw_arr) - frame + 1), hop):
        chunk = raw_arr[start : start + frame]
        if len(chunk) < frame:
            break
        spectrum = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk))))[:64]
        spectrum = np.log1p(spectrum)
        frames.append(spectrum)
    spec = np.stack(frames) if frames else np.zeros((1, 64), dtype="float32")
    spec = spec / max(float(np.max(spec)), 1e-6)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="28" y="32" font-family="Arial" font-size="21" font-weight="700" fill="#17202a">Audio QA: {html_escape(row.get("clip_id"))}</text>',
        f'<text x="28" y="56" font-family="Arial" font-size="13" fill="#536471">event_id={html_escape(row.get("event_id"))}  rms={_to_float(row.get("window_rms")):.4g}  peak={_to_float(row.get("window_peak_abs")):.4g}</text>',
        '<text x="28" y="92" font-family="Arial" font-size="13" font-weight="700" fill="#17202a">raw waveform</text>',
        f'<line x1="{margin}" y1="142" x2="{width - margin}" y2="142" stroke="#d5dde5"/>',
        f'<polyline points="{points(raw_arr, 142, 56)}" fill="none" stroke="#0f766e" stroke-width="1.4"/>',
        '<text x="28" y="220" font-family="Arial" font-size="13" font-weight="700" fill="#17202a">enhanced waveform</text>',
        f'<line x1="{margin}" y1="270" x2="{width - margin}" y2="270" stroke="#d5dde5"/>',
        f'<polyline points="{points(enhanced_arr, 270, 56)}" fill="none" stroke="#7c3aed" stroke-width="1.4"/>',
        '<text x="28" y="348" font-family="Arial" font-size="13" font-weight="700" fill="#17202a">raw log-magnitude spectrogram</text>',
    ]
    spec_left = margin
    spec_top = 365
    spec_h = 110
    cell_w = plot_w / max(1, spec.shape[0])
    cell_h = spec_h / max(1, spec.shape[1])
    for x_index in range(spec.shape[0]):
        for y_index in range(spec.shape[1]):
            value = float(spec[x_index, y_index])
            blue = int(238 - 120 * value)
            green = int(242 - 95 * value)
            red = int(255 - 245 * value)
            color = f"#{red:02x}{green:02x}{blue:02x}"
            x = spec_left + x_index * cell_w
            y = spec_top + (spec.shape[1] - y_index - 1) * cell_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w + 0.4:.1f}" height="{cell_h + 0.4:.1f}" fill="{color}"/>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def run_audio_presence_audit(
    base_dir: str | Path,
    *,
    clip_run_id: str = "mlb_2024_2026_full_v2",
    audit_id: str = "audio_presence_mlb_2024_2026_v2",
    clips_path: str | Path | None = None,
    model_config: str | Path | None = DEFAULT_MODEL_CONFIG,
    max_clips: int | None = None,
    recover_missing_from_sources: bool = False,
    output_suffix: str = ".parquet",
    preview_examples: int = 8,
) -> dict[str, Path]:
    """Audit clip audio tracks, optionally recover short wav windows from source videos."""

    payload = _load_json_config(model_config)
    config = _config_from_payload(payload)
    base = Path(base_dir)
    clips = Path(clips_path) if clips_path else base / f"clips/{clip_run_id}/clips_v1.parquet"
    clip_rows = read_table(clips) if clips.exists() else []
    selected, skipped = _select_audio_clips(
        clip_rows,
        base,
        max_clips=max_clips,
        representative_clip_per_event=config.representative_clip_per_event,
    )
    root = base / "reports" / "audio_audit" / audit_id
    feature_root = base / "features" / audit_id
    tables = root / "tables"
    figures = root / "figures"
    recovered_dir = feature_root / "recovered_audio_windows"
    audit_rows: list[dict[str, Any]] = []
    valid_clips: list[dict[str, Any]] = []
    preview_candidates: list[dict[str, Any]] = []

    for clip in selected:
        clip_path = _path_from_row(clip, base, ("audio_path", "_resolved_clip_path", "clip_path"))
        probe = _ffprobe(clip_path) if clip_path else {"probe_status": "clip_file_missing", "has_audio_stream": None}
        stats = _window_stats(clip, config)
        row = {
            "schema_version": "audio_presence_audit_v1",
            "clip_id": clip.get("clip_id"),
            "event_id": clip.get("event_id"),
            "same_event_group_id": clip.get("same_event_group_id", clip.get("event_id")),
            "batter_id": clip.get("batter_id"),
            "season": clip.get("season"),
            "batter_season_id": clip.get("batter_season_id"),
            "clip_path": str(clip_path) if clip_path else None,
            "split": clip.get("split", "unknown"),
            **probe,
            **stats,
        }
        final_clip = dict(clip)
        if stats.get("audio_window_status") == "complete":
            row["final_audio_status"] = "valid_original_audio"
            final_clip["audio_presence_audit_status"] = "valid_original_audio"
            valid_clips.append(final_clip)
            preview_candidates.append({**final_clip, **row})
        elif recover_missing_from_sources:
            recovered, recovery = _recover_audio_window(clip, base=base, config=config, output_dir=recovered_dir)
            row.update(recovery)
            if recovered is not None:
                recovered_stats = _window_stats(recovered, config)
                row.update({f"recovered_{key}": value for key, value in recovered_stats.items()})
                if recovered_stats.get("audio_window_status") == "complete":
                    row["final_audio_status"] = "valid_recovered_audio"
                    recovered["audio_presence_audit_status"] = "valid_recovered_audio"
                    valid_clips.append(recovered)
                    preview_candidates.append({**recovered, **row})
                else:
                    row["final_audio_status"] = "invalid_after_recovery"
            else:
                row["final_audio_status"] = "invalid_no_audio"
        else:
            row["final_audio_status"] = "invalid_no_audio"
        audit_rows.append(row)

    for skipped_row in skipped:
        audit_rows.append(
            {
                "schema_version": "audio_presence_audit_v1",
                "clip_id": skipped_row.get("clip_id"),
                "event_id": skipped_row.get("event_id"),
                "final_audio_status": f"skipped:{skipped_row.get('reason')}",
            }
        )

    summary = _summary_rows(audit_rows)
    outputs = {
        "audio_presence_manifest": feature_root / f"audio_presence_manifest{output_suffix}",
        "audio_valid_clips": feature_root / f"audio_valid_clips_v1{output_suffix}",
        "audio_presence_manifest_csv": tables / "audio_presence_manifest.csv",
        "audio_presence_summary": tables / "audio_presence_summary.csv",
        "audio_presence_status_chart": figures / "audio_presence_status.svg",
        "audio_audit_html": root / "index.html",
        "audio_audit_summary": root / "summary.json",
    }
    write_table(outputs["audio_presence_manifest"], audit_rows)
    write_table(outputs["audio_valid_clips"], valid_clips)
    _write_csv(outputs["audio_presence_manifest_csv"], audit_rows)
    _write_csv(outputs["audio_presence_summary"], summary)
    final_status_rows = [row for row in summary if row["summary_group"] == "final_audio_status"]
    _svg_bar(outputs["audio_presence_status_chart"], final_status_rows, title="Audio Presence Status", label_key="key", value_key="rows")

    preview_paths: list[Path] = []
    preview_sorted = sorted(preview_candidates, key=lambda row: _to_float(row.get("window_rms")), reverse=True)[: max(0, preview_examples)]
    for index, row in enumerate(preview_sorted, start=1):
        path = figures / f"audio_preview_{index:02d}_{_safe_id(row.get('clip_id'))}.svg"
        rendered = _svg_waveform_spectrogram(path, row, config)
        if rendered is not None:
            preview_paths.append(rendered)

    figure_html = ""
    if outputs["audio_presence_status_chart"].exists():
        figure_html += f'<figure><img src="{html_escape(outputs["audio_presence_status_chart"].relative_to(root))}" alt="audio status"><figcaption>Audio presence status</figcaption></figure>'
    for path in preview_paths:
        figure_html += f'<figure><img src="{html_escape(path.relative_to(root))}" alt="audio preview"><figcaption>{html_escape(path.stem)}</figcaption></figure>'

    metadata = {
        "schema_version": "audio_presence_audit_summary_v1",
        "audit_id": audit_id,
        "clip_run_id": clip_run_id,
        "input_clips": len(clip_rows),
        "selected_clips": len(selected),
        "valid_audio_clips": len(valid_clips),
        "recover_missing_from_sources": recover_missing_from_sources,
        "preview_examples": len(preview_paths),
        "audio_window": {
            "sample_rate": config.sample_rate,
            "window_start_ms": config.window_start_ms,
            "window_end_ms": config.window_end_ms,
            "context_start_ms": config.context_start_ms,
            "context_end_ms": config.context_end_ms,
        },
        "note_ja": "baseline 比較へ渡すのは audio_valid_clips のみ。empty_audio_window は音声モデル評価から除外する。",
    }
    html = render_page(
        "Audio Presence Audit",
        audit_id,
        (
            ("Run Metadata", render_kv_table(metadata)),
            ("Figures", figure_html or "<p>No figures generated.</p>"),
            ("Summary", render_table(("summary_group", "key", "rows"), summary)),
            (
                "Rows To Inspect",
                render_table(
                    (
                        "clip_id",
                        "event_id",
                        "final_audio_status",
                        "has_audio_stream",
                        "audio_window_status",
                        "window_rms",
                        "window_peak_abs",
                        "recovery_status",
                    ),
                    audit_rows[:80],
                ),
            ),
        ),
        subtitle="Checks whether batting clips really contain usable contact-window audio before audio modeling.",
    )
    write_page(outputs["audio_audit_html"], html)
    write_json(
        {
            **metadata,
            "summary": summary,
            "outputs": {key: str(path) for key, path in outputs.items()},
            "preview_figures": [str(path) for path in preview_paths],
        },
        outputs["audio_audit_summary"],
    )
    return outputs

