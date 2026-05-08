# 複数 batting form clips を使う player-season modeling 設計

作成日: 2026-04-26

この文書は、入力が `1 batting clip` から `同一 batter-season の複数 batting clips` に広がった場合の設計変更をまとめます。

結論:

```text
異なる打席 clip の予測を単純平均して、1打席の EV / LA 予測に使ってはいけない。
各 BBE event の EV / LA は、その event 固有の target だからである。

複数 clip は、次のどちらかとして使う。
1. player-season mechanics prior を作る
2. player-season aggregate target を予測する

同一 event の複数 view / augmentation だけは、予測平均または重み付き平均してよい。
```

---

## 1. 何が変わったか

旧設計:

```text
1 clip = 1 BBE = 1 prediction
```

新設計:

```text
1 BBE event = 1 current swing clip = 1 event-level prediction
1 batter-season = multiple clean swing clips = 1 mechanics set
```

つまり、予測対象には2階層があります。

| 階層 | 入力 | 出力 | 例 |
|---|---|---|---|
| event-level | その打席の clip と context | その打席の EV / LA / hard-hit / barrel | `event_id=abc` の EV |
| player-season-level | 同一 batter-season の複数 clean clips | その年度の mechanics embedding / average EV / hard-hit rate | `batter_id=xxx, season=2026` |

最初の本命は event-level です。

ただし、同一 batter-season に複数 clip があるなら、そこから `player-season mechanics prior` を作り、event-level prediction に足すと強くなります。

---

## 2. 平均を取ってよい場合、だめな場合

### 2.1 だめな平均

次は基本的にだめです。

```text
同じ batter-season の複数 event clip を個別に予測
  -> y_pred を平均
  -> ある1打席の EV / LA 予測にする
```

理由:

- 各 event は pitch location, pitch type, count, timing, contact quality が違う
- EV / LA は打席ごとの target
- 別 event の予測平均は、その event の予測ではなく、選手年度の平均的傾向になる

例:

```text
clip A: 真ん中速球を強く打った EV 105
clip B: 外角低めを泳いで EV 78
clip C: 高めをこすって LA 48
```

これらを平均して、clip A の EV を予測するのは意味が違います。

### 2.2 平均してよい場合

平均してよいのは、主に次です。

#### A. 同一 event の複数 view

同じ `event_id` に対して複数 view がある場合:

```text
view 1 prediction
view 2 prediction
view 3 prediction
  -> weighted average
```

これは同じ target を見ているので平均できます。

重み:

- `view_confidence`
- `quality_tier`
- `contact_confidence`
- `pose_coverage`

#### B. 同一 event の augmentation / crop

同じ clip から作った複数 crop / time window / augmentation は、test-time augmentation として平均してよいです。

#### C. player-season aggregate target

次のような選手年度の集計値を予測する場合:

- average EV
- EV50
- hard-hit rate
- barrel rate
- average LA
- average xBA
- average xwOBA
- OPS / OBP / SLG when PA-level manifest exists
- mechanics cluster

この場合、複数 clip prediction の平均や set pooling が自然です。

### 2.3 平均より推奨する方法

event-level 予測では、複数 clip は平均ではなく `conditioning` に使います。

```text
current event clip embedding
context features
player-season mechanics embedding
  -> event-level prediction
```

---

## 3. 他研究ではどう扱うか

この問題は、直接には `複数 swing clips の set modeling` ですが、近い研究領域では次の考え方が使われています。

### 3.1 Set input の研究

Deep Sets は、入力が集合であり順序に意味がないタスクでは、permutation-invariant な関数が必要だと整理しています。

今回の `同一 batter-season の複数 clean swing clips` は、基本的に順序のない集合です。

```text
S = {swing_clip_1, swing_clip_2, ..., swing_clip_N}
```

したがって、`mean pooling` は最小 baseline として妥当ですが、最終的には Deep Sets 型の `rho(sum(phi(x)))` や、attention pooling が自然です。

### 3.2 Set Transformer

Set Transformer は、multiple instance learning や few-shot などの set-structured data に対し、attention で要素間相互作用を扱う設計です。

今回の用途:

- 同一 batter-season の swing 間の一貫性を見る
- 例外的 swing を弱く重み付けする
- clean center swing を強く重み付けする
- mechanics embedding を作る

ただし、最初から Set Transformer を本命にしません。

順番:

1. mean pooling baseline
2. quality-weighted pooling
3. attention pooling
4. Set Transformer

### 3.3 Multiple Instance Learning

動画分類では、frame / clip feature を bag として扱い、bag-level label を予測する multiple instance learning が使われます。

今回も、player-season を `bag` と見なせます。

```text
bag = one batter-season
instances = clean swing clips
bag label = player-season aggregate metric
```

ただし event-level EV / LA では、各 instance に event label があるため、MIL そのものより階層モデルに近いです。

### 3.4 Action Quality Assessment

FineDiving などの action quality assessment 研究では、単に動画全体の特徴を平均して score regression するだけではなく、action を procedure / steps に分けることで解釈性と精度を上げようとしています。

今回の batting でも、

- stance
- load
- stride
- swing initiation
- contact

を分ける phase-aware sequence が重要です。

これは `複数 swing を平均する` というより、

```text
各 swing 内で phase をそろえる
複数 swing 間では phase-aware embedding を比較・集約する
```

という扱いに近いです。

### 3.5 Biomechanics research

野球 batting / pitching の biomechanics 研究では、複数 trial を集めて、選手内・選手間・年度間の違いを見る設計がよく出てきます。

今回の実験設計もこれに合わせます。

- 1 swing を消費して終わりにしない
- 同一 batter-season に複数 swing を保持する
- player-level / season-level / event-level を分ける
- season をまたいだ変化を別分析に残す

---

## 4. 推奨アーキテクチャ

### 4.1 Baseline A: event-only

最初に必ず作る baseline です。

```text
event clip -> sequence encoder -> event embedding
context features -> context encoder
event embedding + context embedding -> prediction heads
```

これは従来の `1 clip = 1 prediction` です。

### 4.2 Baseline B: event-only + quality weighted same-event ensemble

同じ event に複数 view / crop / augmentation がある場合だけ ensemble します。

```text
same event views -> predictions
weighted average by quality
```

出力:

- `event_pred_mean`
- `event_pred_std`
- `n_views`
- `view_weights`

### 4.3 Model C: player-season mechanics prior

同一 batter-season の clean clips から mechanics embedding を作ります。

```text
S_batter_season = clean clips from same batter and season
clip_encoder(each clip) -> clip_embedding_i
set_aggregator({clip_embedding_i}) -> player_season_embedding
```

event prediction:

```text
current_event_embedding
context_embedding
player_season_embedding
quality_features
  -> prediction heads
```

この方式が v1.5 の本命です。

### 4.4 Model D: historical prior only

評価時のリークを避けるため、event 日付より前の clips だけで player-season embedding を作る variant を作ります。

```text
S_past = clips where game_date < current_event_game_date
```

これにより、実運用に近い評価ができます。

### 4.5 Model E: player-season aggregate prediction

event ではなく season aggregate を予測します。

```text
multiple clips -> player-season aggregate metrics
```

target:

- average EV
- EV50
- hard-hit rate
- barrel rate
- average LA
- average xBA
- average xwOBA
- OPS / OBP / SLG when PA-level manifest exists

これは event-level とは別 task です。

---

## 5. set aggregator の候補

### 5.1 Mean pooling

最小 baseline:

```text
m = mean(clip_embeddings)
```

良い点:

- 安定
- 実装が簡単
- leakage や weighting の検証がしやすい

弱い点:

- outlier swing に弱い
- quality を見ない
- swing 間の違いを学べない

### 5.2 Quality-weighted pooling

```text
m = sum(w_i * e_i) / sum(w_i)
```

weight:

- `quality_tier`
- `contact_confidence`
- `pose_coverage`
- `view_confidence`
- `clean_location_flag`
- `outlier_flag`

v1 の推奨です。

### 5.3 Attention pooling

```text
w_i = softmax(MLP(e_i, quality_i))
m = sum(w_i * e_i)
```

良い点:

- informative swing を学習で重くできる
- outlier を自然に下げられる

注意:

- attention が HR や派手な clip に寄る危険がある
- quality / outcome leakage の監視が必要

### 5.4 Set Transformer

複数 swing 間の関係を見る候補です。

使う段階:

- mean pooling と weighted pooling が動いた後
- clips per batter-season が十分に増えた後

---

## 6. leakage を避ける設計

複数 clip を使うと、評価リークが起きやすくなります。

### 6.1 target leakage

だめな例:

```text
validation event の target を含む season aggregate feature を使う
```

対策:

- aggregate feature は training split 内だけで作る
- temporal eval では current event より前の clips だけ使う
- target-derived aggregate と video-derived mechanics embedding を分ける

### 6.2 sibling clip leakage

同一 batter-season のほぼ同じ event / same highlight edit が train と validation に分かれると危険です。

対策:

- `event_id` だけでなく `source_video_id` と `batter_season_id` で group split
- player-season holdout split を必ず持つ
- temporal split を別で持つ

### 6.3 future leakage

ある日の event を予測するときに、その後の swing clips で作った player-season embedding を使うと future leakage です。

対策:

```text
prior_mode:
  none
  same_season_train_only
  past_only
  oracle_full_season
```

`oracle_full_season` は分析用にだけ使います。

---

## 7. artifact contract 追加

### 7.1 `features/clip_embedding_v1/manifest.parquet`

clip ごとの embedding。

列:

- `clip_id`
- `event_id`
- `batter_id`
- `batter_season_id`
- `game_date`
- `encoder_name`
- `encoder_version`
- `embedding_path`
- `embedding_dim`
- `quality_tier`
- `view_label`

### 7.2 `features/player_season_embedding_v1/manifest.parquet`

複数 clip から作る player-season mechanics embedding。

列:

- `batter_season_id`
- `batter_id`
- `season`
- `aggregator_name`
- `aggregator_version`
- `prior_mode`
- `n_clips_total`
- `n_clips_used`
- `clip_ids_used`
- `cutoff_date`
- `embedding_path`
- `embedding_dim`
- `quality_policy`

### 7.3 `datasets/event_with_player_prior_v1/manifest.parquet`

event-level prediction に player-season prior を足す dataset。

列:

- `sample_id`
- `event_id`
- `clip_id`
- `batter_id`
- `batter_season_id`
- `context_feature_path`
- `current_clip_embedding_path`
- `player_season_embedding_path`
- `prior_mode`
- `target_launch_speed`
- `target_launch_angle`
- `target_hard_hit`
- `target_barrel`
- `target_xba`
- `target_xwoba`
- `split`

### 7.4 `predictions_v1` への追加

追加推奨列:

- `prediction_level`
- `aggregation_scope`
- `prior_mode`
- `n_prior_clips`
- `aggregation_method`
- `same_event_ensemble`
- `prediction_std`

例:

```text
prediction_level = event
aggregation_scope = current_event_with_player_season_prior
prior_mode = past_only
n_prior_clips = 12
aggregation_method = quality_weighted_pooling
same_event_ensemble = false
```

---

## 8. 学習順

### Step 0: event-only baseline

1 clip で予測します。

目的:

- 既存設計との比較
- player-season prior の効果測定の基準

### Step 1: same-event ensemble

同じ event の crop / augmentation / view だけを平均します。

目的:

- view / crop の揺れに対する安定化

### Step 2: player-season embedding offline

各 batter-season について clean clips を集め、embedding を作ります。

目的:

- その年度のフォーム傾向を表現する

### Step 3: event + player-season prior

current event clip に player-season embedding を足します。

目的:

- その打者の普段の mechanics を context として使う

### Step 4: past-only prior

評価時の実運用に近づけます。

目的:

- future leakage を避ける

### Step 5: aggregate season task

event prediction とは別に season aggregate を予測します。

---

## 9. 評価指標

event-level:

- EV MAE / RMSE / R2
- LA MAE / RMSE / R2
- HardHit AUC / F1 / Brier
- Barrel AUC / F1 / Brier

player-season prior の効果:

- event-only vs event+prior
- clean cohort vs full usable
- `n_prior_clips` 別性能
- `prior_mode` 別性能
- player-season holdout での性能

aggregate-level:

- average EV MAE
- hard-hit rate MAE
- barrel rate MAE
- rank correlation

---

## 10. agent 指示で変わる点

### Agent A

追加担当:

- `batter_season_id`
- `batter_season_bbe`
- `batter_season_pa`
- `n_clean_clips_available`
- `video_availability_score`
- `prior_mode` に必要な date / split 情報

### Agent B

追加担当:

- `same_event_group_id`
- 同一 event の複数 view / crop / augmentation 管理
- quality weighting に必要な confidence

### Agent C

追加担当:

- clip embedding
- player-season set dataset
- mean / quality-weighted / attention pooling
- event+prior model

### Agent D1

追加担当:

- `predictions_v1` に aggregation metadata を追加
- event-level と player-season-level を分ける

### Agent D3

追加担当:

- context / current swing / player-season prior / video baseline の fusion
- same-event ensemble と player-season prior を混同しない

### Agent E

追加担当:

- `n_prior_clips` 別の report
- same-event ensemble vs player-season prior の比較
- player-season embedding viewer

---

## 11. agent に渡す短い共通指示

```text
入力が複数 batting clips になった場合でも、異なる event の予測を単純平均して 1 event の EV / LA 予測にしないでください。
同一 event の複数 view / augmentation は平均してよいです。
同一 batter-season の複数 clips は player-season mechanics embedding として集約し、current event prediction の prior / conditioning feature として使ってください。
評価では future leakage を避けるため prior_mode を明記し、past_only と oracle_full_season を分けてください。
```

---

## 12. 参照

- Deep Sets, NeurIPS 2017: https://papers.neurips.cc/paper/6931-deep-sets
- Set Transformer, ICML 2019: https://proceedings.mlr.press/v97/lee19d
- FineDiving, CVPR 2022: https://openaccess.thecvf.com/content/CVPR2022/html/Xu_FineDiving_A_Fine-Grained_Dataset_for_Procedure-Aware_Action_Quality_Assessment_CVPR_2022_paper.html
- FineDiving code: https://github.com/xujinglin/FineDiving
- Learning time-aware features for action quality assessment, Pattern Recognition Letters 2022: https://www.sciencedirect.com/science/article/pii/S0167865522001131
- Longitudinal changes in youth baseball batting, PubMed: https://pubmed.ncbi.nlm.nih.gov/38017563/
- Lower extremity kinematic and kinetic factors associated with bat speed, PubMed: https://pubmed.ncbi.nlm.nih.gov/37853750/
- Model heads and targets local plan: `docs/sport/MODEL_HEADS_AND_TARGETS_PLAN_JA.md`
