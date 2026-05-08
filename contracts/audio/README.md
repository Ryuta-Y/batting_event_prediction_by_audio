# Audio Contracts

このディレクトリは batting clip の音声、特に contact 直前直後の打球音 impact window を研究用 artifact として扱うための契約です。

## Scope

- 入力は `clips/{full_run_id}/clips_v1.parquet` と `manifests/bbe_events_v1.parquet`。
- 標準特徴は contact 推定点に対して `-250ms` から `+150ms` の短い窓を使う。
- raw / enhanced / separated / HF embedding は独立 run として出し、同じ `predictions_v1` で比較する。
- xBA / xwOBA の欠損は欠損のまま扱い、0 埋めしない。
- OPS は event-level audio head にはしない。

## Artifacts

- `audio_segments_v1`: どの clip からどの音声窓を切ったかの QA manifest。
- `audio_features_v1`: hand-crafted impact feature または enhanced/separated feature。
- `audio_predictions_mapping_v1`: audio branch から `predictions_v1` へ流す時の target / scope mapping。
