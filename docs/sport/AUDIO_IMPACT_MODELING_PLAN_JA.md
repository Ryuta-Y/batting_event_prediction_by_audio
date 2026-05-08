# 打球音 impact audio modeling 設計

作成日: 2026-05-06 JST

この文書は、既存の MLB batting clip -> Statcast prediction pipeline に、打球音を使った event-level 予測を追加するための設計です。

目的は、`raw video` / `clips` / `split` / `target registry` / `predictions_v1` を再利用しながら、音声だけの独立手法、音声分離あり手法、既存 video / sequence / VLM / context との fusion を公平に比較できるようにすることです。

## 1. 結論

最初に作るべきものは、一般的な音声分離モデルではなく、`impact transient extraction` です。

```text
contact-centered clip
  -> audio stream check
  -> contact 周辺の短い audio window
  -> onset / peak / spectrum / log-mel features
  -> audio-only predictions_v1
  -> method evaluation / fusion
```

理由:

- bat-ball contact は短い transient なので、broadcast 音声でもピークとして取りやすい可能性が高い。
- 最初から Demucs などの大きな source separation に進むと、何が効いたのか分かりにくい。
- `raw audio` と `enhanced audio` と `separated audio` を別 run に分けると、音声分離の価値を後から検証できる。

## 2. Drive で確認した既存 artifact

Google Drive の `baseball_vision` には、2026-05-06 時点で以下の構成がある。

```text
/content/drive/MyDrive/baseball_vision
  manifests/
    bbe_events_v1.parquet
    video_sources_v1.parquet
    player_season_batting_v1.parquet
    splits/
  raw_videos/
    mlb_2024_2026_full_v1/
    mlb_2024_2026_full_v2/
      download_manifest_v1.parquet
      *.mp4
  clips/
    mlb_2024_2026_full_v1/
    mlb_2024_2026_full_v2/
      clips_v1.parquet
      candidate_segments_v1.parquet
      videos/
  features/
    structured_sequence_mlb_2024_2026_v2/
    video_lightweight_features_mlb_2024_2026_v2/
    video_embedding_mlb_2024_2026_v2/
    vlm_mechanics_mlb_2024_2026_v2/
  predictions/
    context_catboost_mlb_2024_2026_v2/
    sequence_tcn_mlb_2024_2026_v2/
    video_lightweight_cv2_mlb_2024_2026_v2/
    video_frozen_encoder_mlb_2024_2026_v2/
    video_raw_finetune_mlb_2024_2026_v2/
    vlm_mechanics_mlb_2024_2026_v2/
    fusion_mlb_2024_2026_v2/
  reports/preflight/
```

したがって、音声 pipeline は動画 download をやり直さず、原則として `clips/mlb_2024_2026_full_v2/clips_v1.parquet` と `raw_videos/mlb_2024_2026_full_v2/download_manifest_v1.parquet` を読む。

## 3. 共有するデータと独立させるデータ

共有するもの:

```text
manifests/bbe_events_v1.parquet
manifests/splits/*.parquet
clips/{full_run_id}/clips_v1.parquet
raw_videos/{full_run_id}/download_manifest_v1.parquet
configs/targets/target_registry_v1.yaml
predictions_v1 / metrics_v1 contract
```

音声用に独立させるもの:

```text
contracts/audio/
src/sport_pipeline/audio/
features/audio_impact_{stem}/
features/audio_enhanced_impact_{stem}/
features/audio_separated_impact_{stem}/
predictions/audio_raw_impact_{stem}/
predictions/audio_enhanced_impact_{stem}/
predictions/audio_separated_impact_{stem}/
reports/preflight/audio_*.json
debug/{full_run_id}/audio_qa/  # 将来 optional QA
```

`stem` は v2 なら `mlb_2024_2026_v2` とする。

## 4. 入力 source 優先順位

音声抽出の入力は、次の順で選ぶ。

1. `clips/{full_run_id}/clips_v1.parquet` の `clip_path`
   - すでに contact-centered なので最速。
   - audio-only event prediction の主入力。

2. `clips/{full_run_id}/videos/`
   - `clip_path` が相対 path になっている場合の実体。

3. `raw_videos/{full_run_id}/download_manifest_v1.parquet`
   - clip に音声が無い場合、または前後 window が足りない場合の fallback。
   - v2 manifest が v1 raw video path を seed している場合でも、bytes はコピーせず path を再利用する。

4. v1 raw video
   - v2 に seed 済みの raw video はそのまま参照してよい。
   - ただし新しい artifact は v2 namespace に書く。

## 5. Artifact contract

### 5.1 audio_segments_v1

出力:

```text
features/audio_impact_{stem}/segments_v1.parquet
```

主な列:

```text
schema_version
audio_feature_id
sample_id
clip_id
event_id
same_event_group_id
batter_id
season
batter_season_id
split
source_video_path
source_kind              # clip_video | raw_video_fallback
audio_path_runtime       # cache path。Drive 永続 artifact にはしないことがある
has_audio
sample_rate
channels
source_duration_sec
contact_frame
contact_time_sec
contact_confidence
window_name              # strict | medium | full_clip
window_start_sec
window_end_sec
window_duration_sec
audio_contact_offset_sec # onset で refine した contact offset
impact_peak_time_sec
impact_peak_confidence
audio_quality_tier       # usable_primary | review_only | excluded
audio_quality_flags      # JSON/list string
reject_reason
```

推奨 window:

| window | 範囲 | 用途 |
|---|---|---|
| `strict` | contact -250ms to +150ms | 主評価。実況・歓声 leakage を抑える |
| `medium` | contact -500ms to +500ms | 感度分析 |
| `post_light` | contact -250ms to +750ms | 音の余韻を見るが、leakage 注意 |
| `full_clip` | clip 全体 | smoke / QA のみ。主張には使わない |

### 5.2 audio_features_v1

出力:

```text
features/audio_impact_{stem}/manifest.parquet
```

主な列:

```text
schema_version
audio_feature_id
sample_id
clip_id
event_id
batter_season_id
split
window_name
preprocess_mode          # raw | enhanced | separated
feature_family           # handcrafted | logmel | ast_embedding | beats_embedding
feature_vector           # JSON/list or array path
feature_dim
rms
peak_amplitude
crest_factor
onset_strength_max
onset_strength_mean
spectral_centroid_mean
spectral_bandwidth_mean
spectral_rolloff_mean
high_freq_energy_ratio
low_freq_energy_ratio
transient_snr_proxy
impact_sharpness
audio_quality_tier
audio_quality_flags
```

### 5.3 predictions_v1 mapping

音声モデルも既存の `predictions_v1` を使う。

```text
predictions/audio_raw_impact_{stem}/predictions_v1.parquet
predictions/audio_raw_impact_{stem}/metrics_v1.json
```

必須ルール:

- `prediction_level=event`
- target は `ev`, `la`, `hard_hit`, `barrel`, optional `xba`, `xwoba`
- `aggregation_scope` は下記のように分ける
  - `audio_raw_impact`
  - `audio_enhanced_impact`
  - `audio_separated_impact`
  - `audio_embedding_impact`
- `prior_mode=none`
- OPS は event-level にしない
- xBA/xwOBA 欠損は 0 にしない

## 6. Feature / model 段階

### Phase A: raw impact baseline

最初の baseline。

```text
clip audio
  -> strict window
  -> handcrafted features + log-mel summary
  -> ridge / CatBoost / small MLP
  -> predictions/audio_raw_impact_{stem}
```

期待:

- EV と hard-hit に最も効く可能性が高い。
- LA / barrel / xwOBA は単独では限界があるが、fusion では補完信号になる可能性がある。

### Phase B: enhanced impact baseline

大きな model download なしでできる軽い強調。

```text
raw audio
  -> mono 16kHz
  -> high-pass or band-pass
  -> noise gate
  -> transient emphasis
  -> same feature extractor
  -> predictions/audio_enhanced_impact_{stem}
```

これは音声分離ではなく、impact transient を見やすくする前処理。

### Phase C: separated impact baseline

音声分離の価値を検証する段階。

```text
raw audio
  -> optional source separation / denoise
  -> impact transient extraction
  -> same feature extractor
  -> predictions/audio_separated_impact_{stem}
```

候補:

- Demucs 系: 実況・観客・環境音を抑えられるか比較。
- vocal suppression: commentary overlap が強い row で効果を見る。
- spectral subtraction / noisereduce: 軽量な denoise baseline。
- event/transient extractor: 一般的な source separation より本命に近い。

音声分離は `raw audio` と同じ split / same event / same target で比較する。

### Phase D: audio embedding model

必要になったら追加。

```text
log-mel patch
  -> small CNN / AST / BEATs frozen encoder
  -> supervised head
  -> predictions/audio_embedding_impact_{stem}
```

大型 model download は Colab GPU で明示フラグを有効にした時だけ行う。

## 7. Leakage policy

音声は leakage が起きやすい。

危険:

- contact 後の歓声
- 実況の「gone」「base hit」などの結果発話
- replay 音声
- broadcast edit / music
- camera cut 後の守備音

対策:

```text
primary window = strict
strict = contact -250ms to contact +150ms
```

`medium` や `post_light` は ablation として別集計し、主結果と混ぜない。

音声品質 flag:

```text
no_audio
silent_audio
audio_too_short
audio_desync_suspected
commentary_overlap
crowd_spike
music_or_replay_suspected
clipped_audio
low_snr
impact_peak_not_found
impact_peak_far_from_contact
```

## 8. Colab notebook 設計

既存 compact notebook `30`-`35` の後ろに追加する。

### 36_cpu_audio_impact_baseline.ipynb

Runtime: CPU

目的:

- `clips_v1` と `download_manifest_v1` を読む
- audio stream を検査する
- contact 周辺 window を作る
- raw / enhanced features を作る
- audio-only baseline を学習・評価する

出力:

```text
features/audio_impact_{stem}/segments_v1.parquet
features/audio_impact_{stem}/manifest.parquet
features/audio_enhanced_impact_{stem}/manifest.parquet
predictions/audio_raw_impact_{stem}/predictions_v1.parquet
predictions/audio_raw_impact_{stem}/metrics_v1.json
predictions/audio_enhanced_impact_{stem}/predictions_v1.parquet
predictions/audio_enhanced_impact_{stem}/metrics_v1.json
reports/preflight/audio_impact_baseline_audio_raw_impact_{stem}.json
reports/preflight/audio_impact_baseline_audio_enhanced_impact_{stem}.json
reports/audio_impact/audio_raw_impact_{stem}/index.html
reports/audio_impact/audio_enhanced_impact_{stem}/index.html
```

### 37_gpu_audio_separation_and_embeddings.ipynb

Runtime: L4 推奨。small run なら CPU でも可。

目的:

- optional 音声分離 / denoise を実行
- optional AST / BEATs / small CNN embedding を作る
- separated audio-only prediction を作る

出力:

```text
features/audio_separated_impact_{stem}/manifest.parquet
features/audio_embedding_impact_{stem}/manifest.parquet
predictions/audio_separated_impact_{stem}/predictions_v1.parquet
predictions/audio_separated_impact_{stem}/metrics_v1.json
predictions/audio_embedding_impact_{stem}/predictions_v1.parquet
predictions/audio_embedding_impact_{stem}/metrics_v1.json
reports/preflight/audio_impact_baseline_audio_separated_impact_{stem}.json
reports/preflight/hf_audio_embedding_baseline_audio_embedding_impact_{stem}.json
reports/audio_impact/audio_separated_impact_{stem}/index.html
reports/audio_impact/audio_embedding_impact_{stem}/index.html
```

### 38_cpu_audio_fusion_compare.ipynb

Runtime: CPU

目的:

- audio runs を player-season projection に回す
- existing v2 fusion を上書きせず、別 run の audio-aware fusion を作る
- audio あり / なしを method evaluation で比較する

出力:

```text
predictions/audio_raw_impact_{stem}_player_season_projection/
predictions/audio_enhanced_impact_{stem}_player_season_projection/
predictions/audio_separated_impact_{stem}_player_season_projection/
predictions/fusion_audio_{stem}/
reports/method_evaluation/method_evaluation_audio_{stem}/
reports/ablation_compare/audio_ablation_{stem}/  # 将来 optional
```

## 9. Fusion 設計

既存の `fusion_mlb_2024_2026_v2` は上書きしない。

新規 fusion run:

```text
fusion_audio_mlb_2024_2026_v2
```

必要であれば将来 branch 別 fusion を追加するが、初期実装では `fusion_audio_run_id` に audio raw/enhanced/separated/embedding をまとめて接続する。

`late_fusion` の `scope_weights` に追加する候補:

```json
{
  "audio_raw_impact": 0.65,
  "audio_enhanced_impact": 0.70,
  "audio_separated_impact": 0.70,
  "audio_embedding_impact": 0.75
}
```

最初は固定 weight でよいが、既存の `learn_validation_scope_weights` を使って validation split だけで per-target weight を学習してもよい。

期待する傾向:

| target | audio-only 期待 | fusion 期待 |
|---|---|---|
| EV | 高い | 強い補完 |
| hard-hit | 高い | 強い補完 |
| LA | 低〜中 | video / sequence と組むと改善余地 |
| barrel | 中 | EV signal と LA/video signal の組み合わせが本命 |
| xBA/xwOBA | 中 | context / EV / LA と組むと改善余地 |
| OPS/OBP/SLG | event 直接では不可 | player-season projection で比較 |

## 10. 比較表

最低限この順で比較する。

```text
context_catboost
sequence_tcn
video_lightweight_cv2
video_frozen_encoder
video_raw_finetune
vlm_mechanics
audio_raw_impact
audio_enhanced_impact
audio_separated_impact
fusion_mlb_2024_2026_v2
fusion_audio_mlb_2024_2026_v2
```

重要:

- audio-only と video-only は同一 event set の intersection metric も出す。
- audio で使える clip が少ない場合、全体 metric と intersection metric を分ける。
- same-event view / crop だけは ensemble 可。異なる event の予測平均は禁止。

## 11. 実装 module 案

```text
src/sport_pipeline/audio/
  __init__.py
  impact.py
  research_report.py
  separation.py
  embeddings.py
```

Command entrypoints:

```bash
python -m sport_pipeline.audio.impact \
  --base-dir /content/drive/MyDrive/baseball_vision \
  --clip-run-id mlb_2024_2026_full_v2 \
  --audio-feature-id audio_impact_mlb_2024_2026_v2 \
  --prediction-run-id audio_raw_impact_mlb_2024_2026_v2 \
  --preprocessing-mode raw \
  --require-non-empty

python -m sport_pipeline.audio.separation \
  --base-dir /content/drive/MyDrive/baseball_vision \
  --clip-run-id mlb_2024_2026_full_v2 \
  --audio-feature-id audio_separated_impact_mlb_2024_2026_v2 \
  --prediction-run-id audio_separated_impact_mlb_2024_2026_v2 \
  --separation-backend transient_enhance \
  --require-non-empty

python -m sport_pipeline.audio.embeddings \
  --base-dir /content/drive/MyDrive/baseball_vision \
  --clip-run-id mlb_2024_2026_full_v2 \
  --audio-feature-id audio_embedding_impact_mlb_2024_2026_v2 \
  --prediction-run-id audio_embedding_impact_mlb_2024_2026_v2 \
  --hf-model-id MIT/ast-finetuned-audioset-10-10-0.4593 \
  --allow-model-download

python -m sport_pipeline.models.player_season.event_projection \
  --base-dir /content/drive/MyDrive/baseball_vision \
  --source-run-id audio_raw_impact_mlb_2024_2026_v2

python -m sport_pipeline.models.fusion.run_fusion \
  --base-dir /content/drive/MyDrive/baseball_vision \
  --fusion-run-id fusion_audio_mlb_2024_2026_v2 \
  --source-run context_catboost_mlb_2024_2026_v2 \
  --source-run sequence_tcn_mlb_2024_2026_v2 \
  --source-run video_frozen_encoder_mlb_2024_2026_v2 \
  --source-run vlm_mechanics_mlb_2024_2026_v2 \
  --source-run audio_raw_impact_mlb_2024_2026_v2 \
  --source-run audio_enhanced_impact_mlb_2024_2026_v2 \
  --source-run audio_separated_impact_mlb_2024_2026_v2 \
  --source-run audio_embedding_impact_mlb_2024_2026_v2
```

## 12. 初回 pilot

最初は 1500 clips 全部ではなく、`MAX_CLIPS=100` で確認する。

成功条件:

- `audio stream found` が十分ある
- `strict` window が作れる
- impact peak が contact 付近に来る
- `audio_raw_impact` が `predictions_v1` を出す
- EV / hard-hit で context-only とは違う signal が見える
- audio あり fusion が既存 fusion と別 run で保存される

失敗時に見るもの:

```text
reports/preflight/audio_impact_baseline_audio_raw_impact_{stem}.json
reports/preflight/audio_impact_baseline_audio_separated_impact_{stem}.json
debug/{full_run_id}/audio_qa/*.png  # 将来 optional
debug/{full_run_id}/audio_qa/*.wav  # 将来 optional
```

## 13. 残るリスク

- broadcast 音声は実況・歓声・編集音を含むため、音声が結果 leakage になり得る。
- clip に audio track が無い、または Colab の ffmpeg decode が不安定な mp4 がある。
- contact frame がずれている場合、impact window もずれる。
- 音声分離 model は打球音専用ではないため、むしろ transient を壊す可能性がある。
- audio-only が強くても、それが bat-ball impact 由来か crowd / commentary 由来かは QA が必要。

## 14. 推奨実装順

```text
1. contracts/audio/ を追加
2. src/sport_pipeline/audio/impact.py で segments_v1 / manifest / predictions_v1 / report を作る
3. notebook 36 で raw/enhanced を Colab CPU 実行する
4. method_evaluation と fusion weights に audio run を追加する
5. src/sport_pipeline/audio/separation.py と notebook 37 で separated branch を作る
6. src/sport_pipeline/audio/embeddings.py と notebook 37 で HF audio embedding branch を作る
7. notebook 38 で player-season projection / fusion / method evaluation を更新する
8. raw / enhanced / separated / embedding の same-sample intersection を見る
```

この順番なら、打球音が本当に効くかを先に確認し、その後で音声分離の価値を測れる。

## 15. 2026-05-07 追記: 音声監査と baseline 比較

38 までの結果では `empty_audio_window` が多く、全手法 same-sample intersection が n=10 まで小さくなった。研究主張としては危ないため、追加方針を次に変更する。

```text
clips_v1
  -> audio presence audit
  -> audio_valid_clips_v1
  -> audio raw/enhanced valid-only baseline
  -> audio vs context/video/fusion baseline の pairwise intersection
  -> waveform / spectrogram / correlation / delta chart を確認
```

新規 notebook:

```text
notebooks/39_cpu_audio_audit_baseline_compare.ipynb
```

主な出力:

```text
features/{audio_audit_feature_id}/audio_presence_manifest.parquet
features/{audio_audit_feature_id}/audio_valid_clips_v1.parquet
reports/audio_audit/{audio_presence_audit_id}/index.html
reports/audio_audit/{audio_presence_audit_id}/figures/audio_preview_*.svg
predictions/{audio_raw_valid_run_id}/predictions_v1.parquet
predictions/{audio_enhanced_valid_run_id}/predictions_v1.parquet
reports/audio_baseline_compare/{audio_baseline_compare_report_id}/index.html
reports/audio_baseline_compare/{audio_baseline_compare_report_id}/tables/pairwise_audio_vs_baseline_metrics.csv
```

比較は全手法 intersection ではなく、`audio run` と `baseline run` の pairwise intersection を主分析にする。baseline run は初期設定で `context_catboost_mlb_2024_2026_v2`、`video_lightweight_cv2_mlb_2024_2026_v2`、`fusion_mlb_2024_2026_v2`。

音声が無い場合:

- まず `reports/audio_audit/{audio_presence_audit_id}/tables/audio_presence_summary.csv` で `has_audio_stream` と `audio_window_status` を見る。
- `clip` には音が無いが raw/source video path に音がある場合、run profile の `execution.audio_audit.recover_missing_from_sources=true` で短い wav window を復元する。
- source video にも音が無い場合は、動画 download notebook を音声 track 優先で再実行する必要がある。
