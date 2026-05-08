# audio_segments_v1

Contact-centered audio window の切り出し単位。

## Required Columns

| column | type | nullable | description |
|---|---:|---:|---|
| schema_version | string | no | `audio_segments_v1` |
| segment_id | string | no | `{clip_id}__{mode}__impact` |
| clip_id | string | no | source clip id |
| event_id | string | no | Statcast BBE event id |
| batter_season_id | string | no | `{batter_id}_{season}` |
| clip_path | string | yes | source video/audio path used for extraction |
| preprocessing_mode | string | no | `raw`, `enhanced`, `separated`, or `embedding` |
| sample_rate | int | no | output audio sample rate |
| source_window_start_sec | number | no | source start offset used for extraction |
| contact_offset_sec | number | no | contact point within extracted context audio |
| impact_window_start_ms | number | no | usually `-250` |
| impact_window_end_ms | number | no | usually `150` |
| num_samples | int | no | extracted samples in impact window |
| audio_status | string | no | `complete`, `empty_audio_window`, or failure status |
| impact_peak_time_ms | number | yes | peak short-time energy time inside impact window |
| impact_confidence | number | yes | heuristic transient confidence, 0 to 1 |
| split | string | yes | leakage-aware split if available |

## Notes

この artifact は主に QA と leakage review 用。学習には `audio_features_v1` と `predictions_v1` を使う。
