"""Static research outputs for audio-impact experiments."""

from __future__ import annotations

import csv
from math import sqrt
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from sport_pipeline.artifact_check import write_json
from sport_pipeline.reports.html import html_escape, render_kv_table, render_page, render_table, write_page


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


def _flatten_metrics(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for level, by_target in (payload.get("metrics") or {}).items():
        if not isinstance(by_target, dict):
            continue
        for target_name, values in by_target.items():
            if not isinstance(values, dict):
                continue
            rows.append(
                {
                    "prediction_level": level,
                    "target_name": target_name,
                    "n_available": values.get("n_available"),
                    "n_skipped": values.get("n_skipped"),
                    "mae": values.get("mae"),
                    "rmse": values.get("rmse"),
                    "r2": values.get("r2"),
                    "spearman": values.get("spearman"),
                    "f1": values.get("f1"),
                    "brier": values.get("brier"),
                }
            )
    return rows


def _status_rows(segments: list[dict[str, Any]], skipped: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for row in segments:
        key = str(row.get("audio_status", "unknown"))
        counts[key] = counts.get(key, 0) + 1
    for row in skipped:
        reason = str(row.get("reason", "skipped"))
        counts[reason] = counts.get(reason, 0) + 1
    return [{"status_or_reason": key, "rows": value} for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]


def _feature_summary(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not features:
        return []
    feature_names = list(features[0].get("feature_names") or [])
    rows = []
    for index, name in enumerate(feature_names):
        values = []
        for row in features:
            raw = row.get("feature_values") or []
            if index < len(raw) and isinstance(raw[index], (int, float)):
                values.append(float(raw[index]))
        if not values:
            continue
        sorted_values = sorted(values)
        mid = len(sorted_values) // 2
        median = sorted_values[mid] if len(sorted_values) % 2 else (sorted_values[mid - 1] + sorted_values[mid]) / 2.0
        rows.append(
            {
                "feature_name": name,
                "n": len(values),
                "mean": mean(values),
                "median": median,
                "min": min(values),
                "max": max(values),
            }
        )
    return rows


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    mx = mean(xs)
    my = mean(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs)
    dy = sum((y - my) ** 2 for y in ys)
    if dx <= 0 or dy <= 0:
        return None
    return numerator / sqrt(dx * dy)


def _feature_target_correlations(features: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    y_by_sample_target: dict[tuple[str, str], float] = {}
    for row in predictions:
        if row.get("target_available") and row.get("y_true") is not None:
            y_by_sample_target[(str(row.get("sample_id")), str(row.get("target_name")))] = float(row["y_true"])
    output: list[dict[str, Any]] = []
    if not features:
        return output
    feature_names = list(features[0].get("feature_names") or [])
    for target_name in ("ev", "la", "hard_hit", "barrel", "xba", "xwoba"):
        for index, feature_name in enumerate(feature_names):
            xs: list[float] = []
            ys: list[float] = []
            for row in features:
                y = y_by_sample_target.get((str(row.get("sample_id")), target_name))
                raw = row.get("feature_values") or []
                if y is None or index >= len(raw) or not isinstance(raw[index], (int, float)):
                    continue
                xs.append(float(raw[index]))
                ys.append(float(y))
            corr = _pearson(xs, ys)
            if corr is None:
                continue
            output.append(
                {
                    "target_name": target_name,
                    "feature_name": feature_name,
                    "pearson": corr,
                    "abs_pearson": abs(corr),
                    "n": len(xs),
                }
            )
    return sorted(output, key=lambda row: float(row["abs_pearson"]), reverse=True)


def _svg_bar_chart(path: Path, rows: list[dict[str, Any]], *, title: str, label_key: str, value_key: str, fill: str) -> Path | None:
    values = [(str(row.get(label_key)), row.get(value_key)) for row in rows if isinstance(row.get(value_key), (int, float))]
    if not values:
        return None
    values = values[:14]
    width = 1040
    left = 330
    top = 62
    bar_h = 26
    gap = 10
    height = top + len(values) * (bar_h + gap) + 34
    max_value = max(float(value) for _label, value in values) or 1.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="34" font-family="Arial" font-size="22" font-weight="700" fill="#17202a">{html_escape(title)}</text>',
    ]
    for index, (label, value) in enumerate(values):
        y = top + index * (bar_h + gap)
        bar_w = int((float(value) / max_value) * (width - left - 120))
        parts.append(f'<text x="24" y="{y + 18}" font-family="Arial" font-size="13" fill="#17202a">{html_escape(label[:48])}</text>')
        parts.append(f'<rect x="{left}" y="{y}" width="{max(bar_w, 1)}" height="{bar_h}" rx="3" fill="{fill}"/>')
        parts.append(f'<text x="{left + max(bar_w, 1) + 8}" y="{y + 18}" font-family="Arial" font-size="13" fill="#17202a">{float(value):.4g}</text>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def _svg_system_diagram(path: Path, *, preprocessing_mode: str) -> Path:
    nodes = [
        ("A", "Statcast BBE/PA manifest", 40, 70, 210, 58, "#e8f3ff"),
        ("B", "raw_videos / clips_v1", 300, 70, 190, 58, "#eaf7ef"),
        ("C", "contact-centered audio window\\n-250ms to +150ms", 540, 58, 240, 82, "#fff6df"),
        ("D", f"{preprocessing_mode} impact features", 830, 70, 180, 58, "#f7ecff"),
        ("E", "supervised heads\\nEV / LA / hard-hit / barrel / xBA / xwOBA", 230, 200, 300, 82, "#f3f4f6"),
        ("F", "predictions_v1 + metrics_v1", 590, 210, 230, 58, "#eef2ff"),
        ("G", "research report\\nHTML / CSV / SVG", 870, 210, 160, 58, "#fef2f2"),
    ]
    arrows = [("A", "E"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "F"), ("D", "G"), ("F", "G")]
    centers = {key: (x + w / 2, y + h / 2) for key, _label, x, y, w, h, _fill in nodes}
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="340" viewBox="0 0 1080 340">',
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#425466"/></marker></defs>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="34" y="34" font-family="Arial" font-size="23" font-weight="700" fill="#17202a">Audio Impact Modeling System</text>',
    ]
    for left, right in arrows:
        x1, y1 = centers[left]
        x2, y2 = centers[right]
        parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#425466" stroke-width="2" marker-end="url(#arrow)"/>')
    for _key, label, x, y, w, h, fill in nodes:
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="#b6c2cf"/>')
        for line_index, line in enumerate(label.split("\\n")):
            parts.append(
                f'<text x="{x + w / 2:.1f}" y="{y + 24 + line_index * 18:.1f}" '
                f'font-family="Arial" font-size="14" font-weight="600" text-anchor="middle" fill="#17202a">{html_escape(line)}</text>'
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts + ["</svg>"]), encoding="utf-8")
    return path


def _figure_img(path: Path | None, label: str, output_root: Path) -> str:
    if path is None:
        return ""
    rel = path.relative_to(output_root)
    return f'<figure><img src="{html_escape(rel)}" alt="{html_escape(label)}" style="max-width:100%;height:auto"><figcaption>{html_escape(label)}</figcaption></figure>'


def write_audio_research_report(
    *,
    base_dir: str | Path,
    run_id: str,
    preprocessing_mode: str,
    model_family: str,
    config: dict[str, Any],
    selected_clips: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    features: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    metrics: dict[str, Any],
    skipped: list[dict[str, Any]],
) -> dict[str, Path]:
    """Write rich static outputs for one audio-impact run."""

    base = Path(base_dir)
    output_root = base / "reports" / "audio_impact" / run_id
    tables = output_root / "tables"
    figures = output_root / "figures"
    outputs: dict[str, Path] = {
        "audio_report_html": output_root / "index.html",
        "audio_report_summary": output_root / "summary.json",
        "audio_metrics_table": tables / "metrics.csv",
        "audio_feature_summary_table": tables / "feature_summary.csv",
        "audio_correlations_table": tables / "feature_target_correlations.csv",
        "audio_status_table": tables / "status_counts.csv",
        "audio_system_diagram": figures / "system_diagram.svg",
        "audio_status_chart": figures / "status_counts.svg",
        "audio_correlation_chart": figures / "top_feature_target_correlations.svg",
    }
    metrics_rows = _flatten_metrics(metrics)
    feature_rows = _feature_summary(features)
    corr_rows = _feature_target_correlations(features, predictions)
    status_rows = _status_rows(segments, skipped)
    _write_csv(outputs["audio_metrics_table"], metrics_rows)
    _write_csv(outputs["audio_feature_summary_table"], feature_rows)
    _write_csv(outputs["audio_correlations_table"], corr_rows)
    _write_csv(outputs["audio_status_table"], status_rows)
    _svg_system_diagram(outputs["audio_system_diagram"], preprocessing_mode=preprocessing_mode)
    _svg_bar_chart(outputs["audio_status_chart"], status_rows, title="Audio Rows By Status", label_key="status_or_reason", value_key="rows", fill="#0f766e")
    corr_chart = _svg_bar_chart(
        outputs["audio_correlation_chart"],
        corr_rows,
        title="Top Absolute Feature/Target Correlations",
        label_key="feature_name",
        value_key="abs_pearson",
        fill="#7c3aed",
    )
    metadata = {
        "schema_version": "audio_impact_research_report_v1",
        "run_id": run_id,
        "preprocessing_mode": preprocessing_mode,
        "model_family": model_family,
        "selected_clips": len(selected_clips),
        "segment_rows": len(segments),
        "feature_rows": len(features),
        "prediction_rows": len(predictions),
        "skipped_rows": len(skipped),
        "config": config,
        "note_ja": "打球音の衝撃窓だけを event-level prediction に使う。実況・歓声など contact 後の長い情報は標準では入れない。",
    }
    figures_html = (
        _figure_img(outputs["audio_system_diagram"], "System diagram", output_root)
        + _figure_img(outputs["audio_status_chart"], "Audio extraction status", output_root)
        + _figure_img(corr_chart, "Top feature/target correlations", output_root)
    )
    html = render_page(
        "Audio Impact Research Report",
        run_id,
        (
            ("Run Metadata", render_kv_table(metadata)),
            ("Figures", figures_html or "<p>No figures available.</p>"),
            ("Metrics", render_table(("prediction_level", "target_name", "n_available", "mae", "rmse", "r2", "spearman", "f1", "brier"), metrics_rows)),
            ("Feature Summary", render_table(("feature_name", "n", "mean", "median", "min", "max"), feature_rows[:30])),
            ("Top Correlations", render_table(("target_name", "feature_name", "pearson", "abs_pearson", "n"), corr_rows[:30])),
            ("Status Counts", render_table(("status_or_reason", "rows"), status_rows)),
        ),
        subtitle="Contact-centered audio features, event heads, and QA outputs for Statcast batting predictions.",
    )
    write_page(outputs["audio_report_html"], html)
    write_json(
        {
            **metadata,
            "metrics": metrics_rows,
            "status_counts": status_rows,
            "top_correlations": corr_rows[:50],
            "outputs": {key: str(path) for key, path in outputs.items()},
        },
        outputs["audio_report_summary"],
    )
    return outputs
