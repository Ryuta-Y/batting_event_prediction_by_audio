# audio_predictions_mapping_v1

Audio branch から共通 `predictions_v1` へ流す mapping。

## Event Targets

| target_name | source column | head_kind | notes |
|---|---|---|---|
| ev | `launch_speed` | regression | Statcast EV |
| la | `launch_angle` | regression | Statcast launch angle |
| hard_hit | `target_hard_hit` | binary | EV >= 95 mph |
| barrel | `target_barrel` | binary | Statcast barrel flag |
| xba | `estimated_ba_using_speedangle` | regression | optional; preserve missing mask |
| xwoba | `estimated_woba_using_speedangle` | regression | optional; preserve missing mask |

## Aggregation Scopes

| branch | aggregation_scope | description |
|---|---|---|
| raw | `audio_raw_impact` | raw contact-window waveform features |
| enhanced | `audio_enhanced_impact` | deterministic transient emphasis |
| separated | `audio_separated_impact` | Demucs or transient-enhanced separated branch |
| embedding | `audio_embedding_impact` | HF audio transformer frozen embedding |

## Prohibited

- `ops` を event-level audio prediction として出さない。
- 異なる BBE event の prediction を event prediction として平均しない。
- xBA/xwOBA 欠損を 0 埋めしない。
