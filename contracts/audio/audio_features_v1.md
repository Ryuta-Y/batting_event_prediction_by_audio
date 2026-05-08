# audio_features_v1

Batting impact sound の短時間特徴。

## Required Columns

| column | type | nullable | description |
|---|---:|---:|---|
| schema_version | string | no | `audio_features_v1` |
| sample_id | string | no | prediction sample id |
| segment_id | string | no | matching `audio_segments_v1.segment_id` |
| clip_id | string | no | source clip id |
| event_id | string | no | Statcast BBE event id |
| batter_season_id | string | no | batter-season id |
| preprocessing_mode | string | no | raw/enhanced/separated |
| extractor_name | string | no | e.g. `numpy_contact_impact_features` |
| extractor_version | string | no | feature extractor version |
| audio_status | string | no | extraction status |
| feature_names | list[string] | no | ordered feature names |
| feature_values | list[number] | no | ordered feature values |
| feature_dim | int | no | length of `feature_values` |
| split | string | yes | split if available |

## Standard Feature Groups

- amplitude: `rms`, `peak_abs`, `crest_factor`
- impact timing: `impact_onset_strength`, `impact_peak_time_ms`, `post_pre_energy_ratio`
- spectrum: `spectral_centroid_hz`, `spectral_bandwidth_hz`, `spectral_rolloff_85_hz`, `high_frequency_energy_ratio`, `spectral_flatness`

## Output Location

`features/{audio_feature_id}/manifest.parquet`
