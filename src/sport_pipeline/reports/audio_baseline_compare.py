"""Pairwise audio-vs-baseline comparison reports."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

from sport_pipeline.artifact_check import write_json
from sport_pipeline.evaluation.evaluator import evaluate_predictions
from sport_pipeline.evaluation.target_registry import load_target_registry
from sport_pipeline.io import read_table
from sport_pipeline.reports.html import html_escape, render_kv_table, render_page, render_table, write_page


DEFAULT_TARGET_REGISTRY = Path(__file__).resolve().parents[3] / "configs/targets/target_registry_v1.yaml"


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


def _prediction_path(base: Path, run_id: str, suffix: str) -> Path:
    return base / "predictions" / run_id / f"predictions_v1{suffix}"


def _available_event_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    output = {}
    for row in rows:
        if row.get("prediction_level") != "event":
            continue
        if not row.get("target_available") or row.get("y_pred") is None or row.get("event_id") is None:
            continue
        output[(str(row["event_id"]), str(row["target_name"]))] = row
    return output


def _metric_values(payload: dict[str, Any], target_name: str) -> dict[str, Any]:
    values = ((payload.get("metrics") or {}).get("event") or {}).get(target_name) or {}
    return values if isinstance(values, dict) else {}


def _primary_metric(target_name: str, target_kind: str) -> str:
    if target_kind == "binary" or target_name in {"hard_hit", "barrel"}:
        return "brier"
    return "mae"


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _svg_delta_chart(path: Path, rows: list[dict[str, Any]]) -> Path | None:
    values = []
    for row in rows:
        delta = _to_float(row.get("primary_delta_audio_minus_baseline"))
        if delta is None:
            continue
        label = f"{row.get('audio_run_id')} vs {row.get('baseline_run_id')} / {row.get('target_name')}"
        values.append((label, delta))
    if not values:
        return None
    values = values[:24]
    width = 1180
    left = 430
    center = 780
    top = 62
    bar_h = 24
    gap = 9
    height = top + len(values) * (bar_h + gap) + 38
    max_abs = max(abs(value) for _label, value in values) or 1.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="24" y="34" font-family="Arial" font-size="22" font-weight="700" fill="#17202a">Audio minus Baseline Primary Metric Delta</text>',
        f'<line x1="{center}" y1="52" x2="{center}" y2="{height - 20}" stroke="#9aa6b2" stroke-width="1"/>',
        '<text x="24" y="54" font-family="Arial" font-size="12" fill="#536471">negative is better for MAE/Brier</text>',
    ]
    scale = 330 / max_abs
    for index, (label, value) in enumerate(values):
        y = top + index * (bar_h + gap)
        bar_w = max(1, abs(value) * scale)
        x = center - bar_w if value < 0 else center
        color = "#0f766e" if value < 0 else "#b91c1c"
        parts.append(f'<text x="24" y="{y + 17}" font-family="Arial" font-size="12" fill="#17202a">{html_escape(label[:58])}</text>')
        parts.append(f'<rect x="{x:.1f}" y="{y}" width="{bar_w:.1f}" height="{bar_h}" rx="3" fill="{color}"/>')
        text_x = x - 64 if value < 0 else x + bar_w + 8
        parts.append(f'<text x="{text_x:.1f}" y="{y + 17}" font-family="Arial" font-size="12" fill="#17202a">{value:+.4g}</text>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def write_audio_baseline_comparison_report(
    base_dir: str | Path,
    *,
    report_id: str = "audio_baseline_compare_mlb_2024_2026_v2",
    audio_run_ids: list[str] | tuple[str, ...],
    baseline_run_ids: list[str] | tuple[str, ...],
    target_registry: str | Path = DEFAULT_TARGET_REGISTRY,
    output_suffix: str = ".parquet",
    min_intersection: int = 3,
) -> dict[str, Path]:
    """Compare audio runs against named baseline runs on pairwise intersections."""

    base = Path(base_dir)
    targets = load_target_registry(target_registry)
    root = base / "reports" / "audio_baseline_compare" / report_id
    tables = root / "tables"
    figures = root / "figures"
    outputs = {
        "audio_baseline_compare_html": root / "index.html",
        "audio_baseline_compare_summary": root / "summary.json",
        "pairwise_metrics": tables / "pairwise_audio_vs_baseline_metrics.csv",
        "missing_inputs": tables / "missing_prediction_inputs.csv",
        "delta_chart": figures / "audio_vs_baseline_primary_delta.svg",
    }

    run_rows: dict[str, list[dict[str, Any]]] = {}
    missing: list[dict[str, Any]] = []
    for run_id in list(audio_run_ids) + list(baseline_run_ids):
        path = _prediction_path(base, run_id, output_suffix)
        if not path.exists():
            missing.append({"run_id": run_id, "path": str(path), "reason": "predictions_missing"})
            continue
        run_rows[run_id] = read_table(path)

    rows_by_run = {run_id: _available_event_rows(rows) for run_id, rows in run_rows.items()}
    result_rows: list[dict[str, Any]] = []
    for audio_run_id in audio_run_ids:
        if audio_run_id not in rows_by_run:
            continue
        audio_map = rows_by_run[audio_run_id]
        for baseline_run_id in baseline_run_ids:
            if baseline_run_id not in rows_by_run:
                continue
            baseline_map = rows_by_run[baseline_run_id]
            target_names = sorted({target for _event, target in audio_map}.intersection({target for _event, target in baseline_map}))
            for target_name in target_names:
                target = targets.get(target_name)
                if target is None or target.level != "event":
                    continue
                common = sorted(
                    key
                    for key in set(audio_map).intersection(baseline_map)
                    if key[1] == target_name
                )
                if len(common) < min_intersection:
                    continue
                audio_subset = [audio_map[key] for key in common]
                baseline_subset = [baseline_map[key] for key in common]
                audio_metrics = _metric_values(evaluate_predictions(audio_subset, targets, run_id=f"{audio_run_id}__pairwise"), target_name)
                baseline_metrics = _metric_values(evaluate_predictions(baseline_subset, targets, run_id=f"{baseline_run_id}__pairwise"), target_name)
                metric = _primary_metric(target_name, target.kind)
                audio_primary = _to_float(audio_metrics.get(metric))
                baseline_primary = _to_float(baseline_metrics.get(metric))
                delta = None if audio_primary is None or baseline_primary is None else audio_primary - baseline_primary
                result_rows.append(
                    {
                        "audio_run_id": audio_run_id,
                        "baseline_run_id": baseline_run_id,
                        "target_name": target_name,
                        "primary_metric": metric,
                        "intersection_samples": len(common),
                        "audio_primary": audio_primary,
                        "baseline_primary": baseline_primary,
                        "primary_delta_audio_minus_baseline": delta,
                        "winner": "audio" if delta is not None and delta < 0 else "baseline_or_tie",
                        "audio_mae": audio_metrics.get("mae"),
                        "baseline_mae": baseline_metrics.get("mae"),
                        "audio_rmse": audio_metrics.get("rmse"),
                        "baseline_rmse": baseline_metrics.get("rmse"),
                        "audio_r2": audio_metrics.get("r2"),
                        "baseline_r2": baseline_metrics.get("r2"),
                        "audio_spearman": audio_metrics.get("spearman"),
                        "baseline_spearman": baseline_metrics.get("spearman"),
                        "audio_f1": audio_metrics.get("f1"),
                        "baseline_f1": baseline_metrics.get("f1"),
                        "audio_brier": audio_metrics.get("brier"),
                        "baseline_brier": baseline_metrics.get("brier"),
                    }
                )

    result_rows = sorted(
        result_rows,
        key=lambda row: (
            str(row.get("target_name")),
            str(row.get("baseline_run_id")),
            _to_float(row.get("primary_delta_audio_minus_baseline")) or 0.0,
        ),
    )
    _write_csv(outputs["pairwise_metrics"], result_rows)
    _write_csv(outputs["missing_inputs"], missing)
    chart = _svg_delta_chart(outputs["delta_chart"], result_rows)

    metadata = {
        "schema_version": "audio_baseline_compare_summary_v1",
        "report_id": report_id,
        "audio_run_ids": list(audio_run_ids),
        "baseline_run_ids": list(baseline_run_ids),
        "pairwise_metric_rows": len(result_rows),
        "missing_inputs": len(missing),
        "min_intersection": min_intersection,
        "note_ja": "全手法 intersection ではなく、audio run と baseline run の pairwise intersection で比較する。",
    }
    figure_html = ""
    if chart is not None:
        figure_html = f'<figure><img src="{html_escape(chart.relative_to(root))}" alt="delta chart"><figcaption>Audio minus baseline primary metric. Negative is better.</figcaption></figure>'
    html = render_page(
        "Audio Vs Baseline Pairwise Comparison",
        report_id,
        (
            ("Run Metadata", render_kv_table(metadata)),
            ("Figures", figure_html or "<p>No figures generated.</p>"),
            (
                "Pairwise Metrics",
                render_table(
                    (
                        "audio_run_id",
                        "baseline_run_id",
                        "target_name",
                        "primary_metric",
                        "intersection_samples",
                        "audio_primary",
                        "baseline_primary",
                        "primary_delta_audio_minus_baseline",
                        "winner",
                    ),
                    result_rows,
                ),
            ),
            ("Missing Inputs", render_table(("run_id", "path", "reason"), missing)),
        ),
        subtitle="Baseline comparison on pairwise common events so audio coverage is not collapsed by unrelated methods.",
    )
    write_page(outputs["audio_baseline_compare_html"], html)
    write_json(
        {
            **metadata,
            "outputs": {key: str(path) for key, path in outputs.items()},
            "pairwise_metrics": result_rows,
            "missing_inputs_rows": missing,
        },
        outputs["audio_baseline_compare_summary"],
    )
    return outputs
