# model heads / target 拡張設計

作成日: 2026-04-26 JST

この文書は、既存の `EV / LA / hard-hit / barrel` に加えて、`xBA`, `xwOBA`, `OPS` 系の target / head をどう扱うかを定義します。

結論:

```text
EV と LA は主回帰 head として維持する。
hard-hit と barrel は EV / LA 由来の接触品質 head として維持する。
xBA と xwOBA は Statcast expected outcome head として追加する。
OPS は 1 BBE event の直接 head にはしない。
OPS は player-season / rolling window の aggregate target として扱う。
```

---

## 1. head を増やす理由

EV / LA / barrel は「打球そのものの物理・接触品質」を見るうえで非常に重要です。
ただし研究としては、次の疑問も見たいです。

- フォームや文脈から、実際の価値に近い expected outcome まで読めるか
- EV / LA は当たるが、xBA / xwOBA は当たらないのか
- player-season mechanics prior が、season aggregate の OPS 系指標にも効くか
- 動画特徴は、context-only baseline を超えて攻撃価値を説明するか

そのため、head は `contact physics`, `contact quality`, `expected outcome`, `aggregate offense` に分けます。

---

## 2. target 階層

| 階層 | target 例 | 予測単位 | 備考 |
|---|---|---|---|
| event physics | EV, LA | BBE event | 主 target |
| event quality | hard-hit, barrel, sweet spot | BBE event | EV / LA 由来の補助分類 |
| event expected outcome | xBA, xwOBA | BBE event | Statcast expected stat label |
| event actual value | wOBA value, hit, total bases | PA / BBE event | outcome 影響が大きいので補助扱い |
| player-season aggregate | OPS, OBP, SLG, average xwOBA, hard-hit rate | batter-season | 複数 event を集約して予測 |
| rolling window aggregate | next 30 PA OPS, rest-of-season xwOBA | batter time window | 将来予測研究用。leakage 管理が必要 |

---

## 3. event-level heads

### 3.1 primary physics heads

| head | label | type | loss | metric |
|---|---|---|---|---|
| `ev` | `launch_speed` | regression | Huber or MSE | MAE, RMSE, R2 |
| `la` | `launch_angle` | regression | Huber or MSE | MAE, RMSE, R2 |

EV / LA は全モデル比較の中心です。

### 3.2 contact quality heads

| head | label | type | loss | metric |
|---|---|---|---|---|
| `hard_hit` | `launch_speed >= 95` | binary classification | BCE | AUROC, AUPRC, F1, Brier |
| `barrel` | Statcast barrel definition / label | binary classification | BCE | AUROC, AUPRC, F1, Brier |
| `sweet_spot` | `8 <= launch_angle <= 32` | binary classification | BCE | AUROC, F1 |
| `launch_speed_angle` | Statcast speed-angle bucket | multiclass | CE | accuracy, macro F1 |

`hard_hit` と `sweet_spot` は deterministic label なので、EV / LA head と完全独立の truth ではありません。
それでも補助 head にすると、encoder が実用的な判定境界を覚えやすくなります。

### 3.3 expected outcome heads

| head | Statcast column | type | loss | metric |
|---|---|---|---|---|
| `xba` | `estimated_ba_using_speedangle` | probability regression | MSE, BCE with soft label, or Beta NLL | MAE, RMSE, calibration |
| `xwoba` | `estimated_woba_using_speedangle` | value regression | Huber or MSE | MAE, RMSE, Spearman |

扱い:

- `xBA` は 0 から 1 の確率スケール
- `xwOBA` は wOBA value scale
- どちらも Statcast 由来列がある場合だけ有効化する
- 欠損が多い場合は head を disable し、`label_availability_report` に残す

注意:

`xBA` / `xwOBA` は EV / LA と強く関係します。
したがって、動画モデルの本質的価値を見るには、次の比較が必要です。

```text
context-only xBA/xwOBA baseline
EV/LA-derived xBA/xwOBA baseline
video/sequence direct xBA/xwOBA
video/sequence + predicted EV/LA -> xBA/xwOBA calibration
```

---

## 4. OPS の扱い

OPS は `OBP + SLG` です。
これは 1 BBE event だけの自然な target ではなく、打席集合の aggregate 指標です。

したがって、v1 では次のどちらかで扱います。

### 4.1 player-season aggregate OPS

```text
input:
  same batter-season clean clips
  context aggregates
  optional player-season mechanics prior

target:
  player-season OPS
  player-season OBP
  player-season SLG
```

これは `prediction_level=player_season` です。

### 4.2 rolling future OPS

将来的には、フォームの時点から将来成績を予測する研究として扱えます。

```text
input:
  clips up to date T

target:
  OPS over next N PA
  OPS from date T to season end
```

これは面白いですが、future leakage を起こしやすいので v1 では必須にしません。

### 4.3 event-level OPS proxy

どうしても event-level に近い形で扱う場合は、OPS そのものではなく成分に分けます。

| component | event label | 備考 |
|---|---|---|
| `on_base_event` | hit, walk, HBP など | BBE-only だと walk / HBP が抜ける |
| `total_bases` | 0, 1, 2, 3, 4 | AB denominator が必要 |
| `slg_value` | total bases per AB contribution | PA-level manifest が必要 |

ただし BBE-only dataset では OPS の分母が壊れるため、正式 OPS は PA-level manifest を追加してから扱います。

---

## 5. target registry

model / evaluator は target を hardcode せず、registry で管理します。

```yaml
targets:
  ev:
    level: event
    column: launch_speed
    kind: regression
    loss: huber
    metrics: [mae, rmse, r2]
    required: true

  la:
    level: event
    column: launch_angle
    kind: regression
    loss: huber
    metrics: [mae, rmse, r2]
    required: true

  hard_hit:
    level: event
    column: target_hard_hit
    kind: binary
    loss: bce
    metrics: [auroc, auprc, f1, brier]
    required: true

  barrel:
    level: event
    column: target_barrel
    kind: binary
    loss: bce
    metrics: [auroc, auprc, f1, brier]
    required: true

  xba:
    level: event
    column: estimated_ba_using_speedangle
    kind: probability
    loss: mse
    metrics: [mae, rmse, calibration]
    required: false

  xwoba:
    level: event
    column: estimated_woba_using_speedangle
    kind: regression
    loss: huber
    metrics: [mae, rmse, spearman]
    required: false

  ops:
    level: player_season
    column: target_ops
    kind: regression
    loss: huber
    metrics: [mae, rmse, spearman]
    required: false
    requires_pa_manifest: true
```

---

## 6. predictions_v1 の拡張

`predictions_v1` は long format を維持します。
head が増えても、1 target 1 row で保存します。

必須列:

```text
run_id
sample_id
event_id
batter_season_id
prediction_level
target_name
y_true
y_pred
target_available
target_source
head_kind
loss_name
aggregation_scope
prior_mode
```

追加候補:

```text
y_pred_std
y_pred_prob
calibrated_pred
label_missing_reason
requires_pa_manifest
```

`target_name` 例:

```text
ev
la
hard_hit
barrel
sweet_spot
xba
xwoba
woba_value
on_base_event
total_bases
ops
```

---

## 7. metrics_v1 の拡張

`metrics_v1.json` は target 別・slice 別に持ちます。

```json
{
  "run_id": "example",
  "metrics": {
    "event": {
      "ev": {"mae": 0.0, "rmse": 0.0},
      "la": {"mae": 0.0, "rmse": 0.0},
      "xba": {"mae": 0.0, "rmse": 0.0},
      "xwoba": {"mae": 0.0, "rmse": 0.0}
    },
    "player_season": {
      "ops": {"mae": 0.0, "rmse": 0.0}
    }
  },
  "label_availability": {
    "xba": {"available": 0, "missing": 0},
    "xwoba": {"available": 0, "missing": 0},
    "ops": {"available": 0, "missing": 0}
  }
}
```

---

## 8. loss weighting

最初は simple にします。

```text
total_loss =
  1.0 * ev_loss
  + 1.0 * la_loss
  + 0.5 * hard_hit_loss
  + 0.5 * barrel_loss
  + 0.3 * xba_loss
  + 0.3 * xwoba_loss
```

OPS は event-level model には混ぜず、player-season aggregate model の loss として別管理します。

将来候補:

- uncertainty weighting
- GradNorm
- target group ごとの staged training
- EV / LA を先に学習し、xBA / xwOBA を後段で fine-tune

---

## 9. 実装時の注意

### 9.1 xBA / xwOBA

- Statcast CSV の `estimated_ba_using_speedangle` と `estimated_woba_using_speedangle` を使う
- 欠損を 0 扱いしない
- target availability を必ず report する
- EV / LA からの派生ラベルに近いため、EV / LA baseline との比較を必須にする

### 9.2 OPS

- BBE-only manifest だけで正式 OPS を作らない
- PA-level manifest が必要
- player-season target として `target_ops`, `target_obp`, `target_slg` を持つ
- rolling future OPS を使う場合は `target_window_start`, `target_window_end`, `min_pa` を必須にする

### 9.3 model head

- model code は target registry を読んで head を作る
- head 名を model 内で固定しすぎない
- missing target は loss から mask する
- eval は target ごとに `n_available` を必ず出す

---

## 10. agent への共通指示

```text
EV / LA / hard-hit / barrel に加えて xBA / xwOBA / OPS 系 target を扱える target registry を設計してください。
xBA は estimated_ba_using_speedangle、xwOBA は estimated_woba_using_speedangle を event-level optional head としてください。
OPS は event-level head として直接扱わず、player-season または rolling window aggregate target としてください。
BBE-only manifest で OPS を作ったふりをしないでください。
head が増えても predictions_v1 は long format を維持し、target_name, prediction_level, target_available, target_source を必ず持たせてください。
Colab 実行時には、どの target が使えたか label availability report を出してください。
```

---

## 11. 参照

- Baseball Savant CSV docs: https://baseballsavant.mlb.com/csv-docs
- MLB Glossary xBA: https://www.mlb.com/glossary/statcast/expected-batting-average
- MLB Glossary xwOBA: https://www.mlb.com/glossary/statcast/expected-woba/
- MLB Glossary OPS: https://www.mlb.com/glossary/standard-stats/on-base-plus-slugging/
- FanGraphs OPS: https://library.fangraphs.com/offense/ops/
