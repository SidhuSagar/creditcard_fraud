# FraudShield AI — Technical Details

This document explains **how the system works**, **what the algorithms are**, **how the
dataset is used**, and **exactly how the reported numbers were produced**. Every figure in
this file is read from the artifacts written by the training and evaluation scripts in
`backend/artifacts/` (`metadata.json`, `evaluation.json`, `eda.json`) — nothing here is
hand-written or estimated.

> **Educational project.** FraudShield AI is a machine-learning demo. It is **not**
> connected to any bank, card network, or payment processor, and it never blocks a real
> transaction.

---

## 1. What the system does

FraudShield AI is a full-stack application that:

1. **Trains** three fraud-detection models on the public ULB/Kaggle credit-card dataset.
2. **Serves** them through a FastAPI/JSON API.
3. **Lets a user** upload a transaction CSV from a **landing screen and one analysis page**
   and see every row classified as fraud or legitimate, with a summary and a risk score per
   transaction.

The frontend is a small React single-page app with two screens — a **Home** landing screen and
an **Analyze** screen (upload → results) — switched by state. It has no router, no database,
and no chart library — uploads are scored in memory and returned.

---

## 2. The dataset

| Property | Value |
| --- | --- |
| Source | ULB Machine Learning Group, *Credit Card Fraud Detection*, distributed on Kaggle |
| Rows | **284,807** |
| Columns | **31** (`Time`, `V1`–`V28`, `Amount`, `Class`) |
| Fraudulent rows | **492** |
| Legitimate rows | **284,315** |
| Fraud rate | **0.1727%** |
| Exact duplicate rows | **1,081** |
| `Time` range | 0 – 172,792 seconds (~48 hours) |
| `Amount` | min 0.00, max 25,691.16, mean 88.35 |
| Mean amount (fraud) | 122.21 |
| Mean amount (legitimate) | 88.29 |
| File size | 150,828,752 bytes (~144 MB) |
| SHA-256 | `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89` |

### 2.1 What the columns mean

- **`Time`** — seconds elapsed between *this* transaction and the *first* transaction in
  the file. It is a relative offset, **not** a wall-clock timestamp.
- **`V1`–`V28`** — the result of a PCA transformation applied by the dataset authors. The
  original features are confidential, so the components are anonymous and **already on a
  comparable scale** (each has mean ≈ 0 and standard deviation ≈ 1).
- **`Amount`** — the transaction amount. Raw, un-normalised, and right-skewed.
- **`Class`** — the label: `1` = fraud, `0` = legitimate.

### 2.2 How the dataset is treated (honesty rules)

These rules are enforced in code, not just documented:

- The **real** dataset is used. No synthetic rows are invented and no metrics are
  hard-coded.
- **`Class` is never a model input.** It is used only as the training target and for
  retrospective evaluation. The analyze endpoint reads it back only to score the model, and
  the model is always fed the 30 features (`Time, V1..V28, Amount`) alone.
- The **1,081 duplicate rows are kept**. Removing them would silently change the class
  balance; the pipeline is transparent about their existence instead.
- An **80/20/20-style stratified split** (60/20/20) is fixed by `random_state=42`, giving
  the exact counts shown in §2.4. Because `Class` is kept as a label, the split is
  reproducible and the test set is never used for any decision.

### 2.3 Feature order (single source of truth)

`backend/training/preprocess.py` defines:

```python
FEATURE_ORDER = ["Time", "V1", "V2", ..., "V28", "Amount"]   # 30 features
```

Every component — preprocessing, training, the model registry, the API schemas, and the
tests — uses this one list. A model can therefore never be fed columns in a different
order, which is a common and silent source of wrong predictions.

### 2.4 Splits (stratified, seed 42)

| Split | Rows | Frauds |
| --- | --- | --- |
| Train | 170,883 | 295 |
| Validation | 56,962 | 99 |
| Test (held out) | 56,962 | 98 |

The validation split is used for everything that involves a *choice* (threshold, number of
boosting rounds, recommended model). The test split is used **only once**, at the end, to
report the numbers in §5. Stratification keeps the tiny fraud rate consistent across all
three splits.

---

## 3. Preprocessing

Preprocessing is a scikit-learn `ColumnTransformer`, fitted **on the training split only**
and then applied unchanged to validation, test, and live requests:

- **`StandardScaler`** on `Time` and `Amount` (the two non-PCA columns). PCA components are
  already standardised; rescaling them again would discard the variance structure the
  authors engineered.
- **Passthrough** for `V1`–`V28`.

The transformer selects its columns **by name**, so it must receive a `pandas.DataFrame`
with the exact `FEATURE_ORDER` columns. The model registry guarantees this at inference
time. Fitting on train-only is what prevents data leakage from validation/test/live data.

### 3.1 Class imbalance

Only 0.1727% of rows are fraud, so a naive classifier can reach 99.83% accuracy by always
predicting "legitimate" — a useless model. Three different, model-appropriate strategies
are used, all restricted to the **training fold**:

| Model | Strategy | Detail |
| --- | --- | --- |
| Random Forest | SMOTE oversampling | `sampling_strategy=0.1`; 170,883 → 187,646 training rows |
| XGBoost | `scale_pos_weight` | 578.2644 (ratio of negatives to positives), no resampling |
| Isolation Forest | none | It is unsupervised and never sees the labels |

SMOTE is applied **only** to the training fold so that no synthetic minority samples leak
into validation or test.

---

## 4. The three models and their algorithms

### 4.1 Random Forest (`random_forest`) — supervised, probability

A **bagging** ensemble of 200 unpruned decision trees (`n_estimators=200`,
`max_features="sqrt"`, `random_state=42`). Each tree is grown on a bootstrap sample with a
random subset of features considered at every split; predictions are the average of the
trees' class-1 probabilities. Bagging reduces variance, and averaging many decorrelated
trees produces a well-behaved probability that is easy to threshold. Imbalance is handled
by SMOTE on the training fold.

**Score semantics:** `predict_proba(X)[:, 1]` → a probability in `[0, 1]`.

### 4.2 XGBoost (`xgboost`) — supervised, probability (**recommended**)

**Gradient-boosted** decision trees: 400 shallow trees (`max_depth=5`,
`learning_rate=0.08`, `subsample=0.8`, `colsample_bytree=0.8`, `tree_method="hist"`),
each trained to correct the residual error of the previous ones. Imbalance is handled with
`scale_pos_weight=578.2644` rather than resampling, which keeps every legitimate example
available to the model.

Two safeguards make the boosting safe on such skewed data:

- **Early stopping** (`early_stopping_rounds=40`) on the validation set, monitoring
  `aucpr` (area under the precision–recall curve). Training stopped at **iteration 381** of
  a maximum 400.
- A **recall-weighted threshold** (see §5), chosen on validation only.

**Score semantics:** `predict_proba(X)[:, 1]` → a probability in `[0, 1]`.

### 4.3 Isolation Forest (`isolation_forest`) — unsupervised, anomaly score

An **unsupervised** detector of 200 isolation trees (`n_estimators=200`,
`random_state=42`). It does not use labels at all: it isolates observations by randomly
partitioning the feature space, and points that are isolated in fewer splits are considered
anomalous. It is included to show how an unsupervised anomaly detector compares with the
supervised models.

**Score semantics:** the app uses **`-score_samples(X)`**, where *higher means more
anomalous*. This is **not a probability** and must not be read as one. The API labels it
`score_type: "anomaly_score"`, and the frontend shows a threshold marker instead of a
"probability" label.

---

## 5. Metrics and threshold selection

### 5.1 The metrics, and why accuracy is not among the headline numbers

With a 0.17% positive rate, **accuracy is misleading**: predicting "legitimate" for
everything scores > 99.8%. The app therefore leads with:

| Metric | Meaning | Why it matters here |
| --- | --- | --- |
| **Precision** | Of the transactions flagged as fraud, what fraction really are fraud? | Governs false alarms / analyst workload |
| **Recall** | Of the actual frauds, what fraction did we catch? | Governs missed fraud (usually the costliest error) |
| **F1** | Harmonic mean of precision and recall | Single balanced summary |
| **PR-AUC** | Area under the precision–recall curve | The most informative ranking metric on very imbalanced data |
| **ROC-AUC** | Area under the ROC curve | Reported for completeness; optimistic under heavy imbalance |
| **FPR** | False-positive rate (FP ÷ actual legitimate) | Operational cost proxy |
| **Confusion matrix** | TN / FP / FN / TP counts | Shows exactly where errors fall |

Accuracy *is* shown in the API payload for completeness, but it is not a headline metric
anywhere in the UI.

### 5.2 How each threshold is chosen (validation only)

Each supervised model produces a continuous score. Turning that score into a decision needs
a threshold, and the threshold is a **choice**, so it is made on the **validation set only**:

```
maximise  F2 = 5·P·R / (4·P + R)      (beta = 2 → recall weighted twice as heavily)
```

F2 deliberately favours **recall**, because in fraud detection a missed fraud (false
negative) is usually worse than a false alarm (false positive). The chosen thresholds are:

| Model | Threshold | Objective |
| --- | --- | --- |
| Random Forest | 0.315 | max validation F2 |
| XGBoost | 0.21155192 | max validation F2 |
| Isolation Forest | 0.5854316 | max validation F2 (on the anomaly score) |

### 5.3 How the recommended model is chosen

By **highest validation F2**, never by test performance:

| Model | Validation F2 (β=2) |
| --- | --- |
| **XGBoost** | **0.8128** ← recommended |
| Random Forest | 0.8016 |
| Isolation Forest | 0.2803 |

The reason string stored in metadata is: *"Highest validation F2 (0.8128); recall is
prioritised while precision and false positives are monitored."*

---

## 6. Measured results (held-out test set, 56,962 rows, 98 frauds)

These are the numbers the app displays. They were computed on the test split **after** all
choices were frozen, and independently recomputed from the saved artifacts by
`training/evaluate_models.py`.

### 6.1 XGBoost — recommended

| Metric | Validation | Test |
| --- | --- | --- |
| Precision | 0.8778 | **0.8469** |
| Recall | 0.7980 | **0.8469** |
| F1 | 0.8360 | **0.8469** |
| PR-AUC | 0.8289 | **0.8703** |
| ROC-AUC | 0.9713 | 0.9792 |
| False-positive rate | 0.0193% | 0.0264% |

Test confusion matrix: **TN 56,849 · FP 15 · FN 15 · TP 83**
(= 98 predicted frauds, 98 actual frauds; the errors balance exactly).
Training time: 7.2 s (best iteration 381).

### 6.2 Random Forest

| Metric | Validation | Test |
| --- | --- | --- |
| Precision | 0.7767 | **0.7391** |
| Recall | 0.8081 | **0.8673** |
| F1 | 0.7921 | **0.7981** |
| PR-AUC | 0.8238 | **0.8638** |
| ROC-AUC | 0.9701 | 0.9775 |
| False-positive rate | 0.0404% | 0.0528% |

Test confusion matrix: **TN 56,834 · FP 30 · FN 13 · TP 85**
(= catches 85 of 98 frauds, at the cost of 30 false alarms).
Training time: 27.33 s.

### 6.3 Isolation Forest

| Metric | Validation | Test |
| --- | --- | --- |
| Precision | 0.1008 | **0.1130** |
| Recall | 0.5051 | **0.6020** |
| F1 | 0.1681 | **0.1903** |
| PR-AUC | 0.1158 | **0.1799** |
| ROC-AUC | 0.9408 | 0.9533 |
| False-positive rate | 0.7843% | 0.8142% |

Test confusion matrix: **TN 56,401 · FP 463 · FN 39 · TP 59**.
Training time: 2.33 s.

### 6.4 Reading the comparison honestly

- **XGBoost** gives the best balance: it flags exactly as many transactions as there are
  real frauds (98 predicted, 98 actual), splitting its errors evenly (15 FP, 15 FN).
- **Random Forest** trades precision for recall — it catches more frauds (85 vs 83) but
  raises roughly twice as many false alarms (30 vs 15).
- **Isolation Forest** is markedly weaker here and is included as a *demonstration of an
  unsupervised baseline*. Its precision (~11%) means the large majority of its flagged
  transactions are legitimate. It has never seen a label, which is exactly why its numbers
  look like this. This is reported as-is; it is not a bug and it is not hidden.

Total wall-clock training time for all three models: **40.58 s**.

---

## 7. Exploratory data analysis

`training/eda.py` writes `backend/artifacts/eda.json`, which contains full per-feature
statistics and an hour-by-hour breakdown. Notable, real findings:

- The fraud rate is **0.1727%** — a ~1:578 imbalance.
- Fraudulent transactions are **more variable in amount** (mean 122.21 vs 88.29 for
  legitimate) — consistent with card-testing behaviour where attackers probe with varied
  amounts.
- The dataset spans only **~48 hours**, and the fraud counts by hour-of-day are uneven
  (e.g. hour 2: 57 frauds; hour 10: 8), showing a real temporal structure rather than a
  uniform process.
- Several `V` components separate the classes strongly (for example fraud mean `V14`
  ≈ −6.97 vs ≈ +0.012 for legitimate), which is why even shallow trees perform well.

---

## 8. System architecture

```
data/creditcard.csv
        │
        ▼
backend/training/preprocess.py ──► stratified 60/20/20 split (seed 42)
        │                                  │
        │                    ┌─────────────┴──────────────┐
        │                    ▼                            ▼
        │            fit StandardScaler              (test split kept aside)
        │            (Time, Amount) on train only
        │                    │
        ▼                    ▼
backend/training/train_models.py
   ├── Random Forest  + SMOTE(train only)   ─┐
   ├── XGBoost        + scale_pos_weight     ├─► artifacts/*.joblib
   └── Isolation Forest (unsupervised)      ─┘   + metadata.json
        │                                         + heldout_test.parquet
        ▼
backend/training/evaluate_models.py ──► evaluation.json
   (independent recomputation; asserts it matches metadata.json)

        ┌───────────────────────────────────────────────┐
        │                 FastAPI backend                │
        │   POST /api/analyze ──► analyze_service        │
        │        (parse CSV, split valid/rejected rows)  │
        │                 │                              │
        │                 ▼                              │
        │        model_registry.predict_dataframe        │
        │        (fitted preprocessor + chosen model)    │
        └───────────────────────────────────────────────┘
                        ▲  JSON /api/*
        ┌───────────────┴───────────────┐
        │   React + Vite + Tailwind SPA  │
        │  2 screens: home → analyze     │
        │  (upload → results, no router) │
        └────────────────────────────────┘
```

There is **no database**: the service is stateless. Each upload is parsed, scored, and
returned; nothing is written to disk.

### 8.1 Backend modules

| Path | Responsibility |
| --- | --- |
| `app/main.py` | App factory, CORS, routers, structured error handlers |
| `app/config.py` | All settings via env vars with safe local defaults (no secrets in source) |
| `app/routes/health.py` | `GET /api/health` |
| `app/routes/analyze.py` | `POST /api/analyze` — reads the upload (with a size cap) and delegates |
| `app/schemas/analyze.py` | Response contract for the classification endpoint |
| `app/schemas/common.py` | Health + error schemas |
| `app/services/model_registry.py` | Loads artifacts once, exposes `catalogue()`, handles score types & thresholds, scores a DataFrame in one vectorised pass |
| `app/services/analyze_service.py` | Parses the CSV, coerces features, splits valid vs. rejected rows, builds the summary and optional evaluation |
| `training/preprocess.py` | `FEATURE_ORDER`, `build_preprocessor`, `split_dataset`, `load_dataset` |
| `training/metrics.py` | `evaluate_predictions`, `select_threshold` (F2), single-class-safe AUC |
| `training/train_models.py` | End-to-end training + artifact writing |
| `training/evaluate_models.py` | Independent recomputation + verification |
| `training/eda.py` | EDA report |

### 8.2 Data flow for one upload

1. `POST /api/analyze` receives a multipart CSV (and an optional `model`).
2. `analyze_service` reads it with pandas and resolves the file's column names onto the 30
   canonical factors (`Time, V1..V28, Amount`) — matching is case-, punctuation- and
   alias-aware (see §8.4) — then coerces them to numeric.
3. Rows with a missing or non-numeric feature are set aside as **rejected** (with a reason);
   the rest are kept.
4. `model_registry.predict_dataframe` scales `Time`/`Amount` with the **fitted** pipeline and
   scores every row with the chosen model in one vectorised pass.
5. Each score is compared to that model's validation-chosen threshold → `Fraud` / `Legitimate`.
6. When the upload carried a `Class` column, `evaluate_predictions` computes precision,
   recall, F1, and the confusion matrix against it (for display only — never as an input).
7. The response contains the summary, the classified rows (**fraud first**, capped at
   `FRAUDSHIELD_MAX_RESPONSE_ROWS`), the rejected rows, and the optional evaluation.

### 8.3 Statelessness

There is nothing to persist: the models are read once at startup and the CSV is scored in
memory. Repeated uploads of the same file return the same result, which is what makes the
figures in section 6 reproducible.

### 8.4 Column-name resolution

The fitted preprocessor selects its inputs **by the canonical names** (`FEATURE_ORDER`), so
before scoring, `analyze_service.resolve_columns` maps whatever the upload calls its columns
back to those names. Matching normalises away case, spaces, punctuation, quotes and a byte
order mark, and accepts a list of common aliases:

| Canonical | Accepted (examples) |
| --- | --- |
| `Time` | `time`, `timestamp`, `unix_time`, `transaction_time`, `seconds` |
| `Amount` | `amount`, `amt`, `transaction_amount`, `transaction_dollar_amount`, `value` |
| `V1`..`V28` | `v1`..`v28`, `v01`..`v28` |
| `Class` (optional) | `class`, `is_fraud`, `label`, `target`, `fraudulent` |

Each source column is matched at most once, exact names win over aliases, and any unmatched
columns are ignored. If a canonical factor is not found at all, the request is rejected with a
message listing the missing factors **and** the columns that were found — this is how a
different dataset (for instance the synthetic `transactions.csv` with only
`credit_card, date, transaction_dollar_amount, Long, Lat`) is reported honestly instead of
being silently mis-scored.

---

## 9. Reproducing the numbers

```powershell
# from the project root, with the dataset at data/creditcard.csv
cd backend

# 1. explore
python -m training.eda

# 2. train all three models and write metadata.json + heldout_test.parquet
python -m training.train_models

# 3. independently recompute the test metrics and verify they match metadata
python -m training.evaluate_models
```

`evaluate_models.py` re-derives each model's test metrics from the saved artifacts and the
held-out split, and records the result in `evaluation.json`:

```json
"verification": {
  "matches_training_metadata": {
    "random_forest": true,
    "xgboost": true,
    "isolation_forest": true
  },
  "test_rows": 56962,
  "note": "evaluate_models.py independently recomputed the metrics above from the saved artifacts and the held-out test set."
}
```

Determinism: every estimator and the split use `random_state=42`.

### 9.1 Artifacts written

| File | Committed? | Purpose |
| --- | --- | --- |
| `preprocessor.joblib` | no (large binary) | Fitted `ColumnTransformer` |
| `random_forest.joblib` | no | Trained RF |
| `xgboost.joblib` | no | Trained XGBoost |
| `isolation_forest.joblib` | no | Trained Isolation Forest |
| `metadata.json` | **yes** | Dataset stats, splits, hyperparameters, metrics, thresholds |
| `evaluation.json` | **yes** | Independently recomputed test metrics + verification block |
| `eda.json` | **yes** | Exploratory analysis |
| `heldout_test.parquet` | no (large binary) | The exact test split, for reproduction |

The JSON files are committed on purpose: a fresh clone can show a clear **"not trained"**
state (models missing) while still documenting what the models *would* be, instead of
inventing numbers.

---

## 10. API reference

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | Service banner |
| `GET` | `/api/health` | Status, dataset/model readiness, available models, recommended model |
| `POST` | `/api/analyze` | Classify a transaction CSV (multipart: `file`, optional `model`) |

The analyze response reports `total_rows`, `processed_rows`, `rejected_rows`,
`fraud_rows`, `legitimate_rows`, the classified `results` (fraud first, capped), the
`rejected` rows with reasons, and an `evaluation` block when the file carried a `Class`
column.

Full interactive schema: `http://localhost:8010/docs`.

### 10.1 Error behaviour

- **`422`** — the CSV does not contain the model's factors (the message lists the missing
  factors and the columns that were found), or an unknown `model` was requested. The message
  names the exact factor/model.
- **`413`** — the upload exceeds `FRAUDSHIELD_MAX_UPLOAD_BYTES`.
- **`503`** — the models are not trained. Returns a clear setup message; the app never
  fabricates a prediction in this state.
- **`500`** — unexpected error; a generic message is returned and details go to the logs.

Uploads succeed **partially**: valid rows are scored, and rows with a missing or non-numeric
feature are reported individually with a reason. If a whole factor is absent from the file
(e.g. there are no V-columns at all) the file is rejected with a message listing the missing
factors and the columns that were found.

### 10.2 A single-class edge case that is handled explicitly

If the uploaded `Class` column contains only one class (all fraud or all legitimate),
precision/recall/F1 are still defined but **PR-AUC and ROC-AUC are mathematically
undefined**. Rather than silently returning `0.0`, the evaluation returns `null` for those
two metrics plus a `notes` entry explaining why.

---

## 11. Frontend

React 18 + Vite 5 + Tailwind CSS + `lucide-react`. The app has **two screens** — a **Home**
landing screen (`src/pages/Home.jsx`) and an **Analyze** screen (`src/pages/Analyze.jsx`) —
switched by state in `src/App.jsx`, with no router and no chart library, which keeps the
bundle small (~23 KB of app JavaScript). In development the Vite dev server proxies `/api` to
the FastAPI server, so there is no CORS friction; `VITE_BACKEND_URL` / `VITE_API_BASE_URL`
override the target.

The screens are deliberately plain and **black-themed**:

| Screen / area | What it shows |
| --- | --- |
| **Header** (all screens) | Clickable product name (returns home) and a live model-readiness pill (`Model ready` / `Model not trained`) |
| **Home — hero** | One-line pitch, a **Start analysis** button, and a warning if no model is trained |
| **Home — how & why** | A three-step "how it works" strip and a facts row (284,807 transactions, 3 models, 492 frauds, 0.17% fraud rate) |
| **Analyze — upload** | Drag-and-drop / browse for the CSV, a model selector, and an **Analyze** button with upload progress; a **Home** link returns to the landing screen |
| **Analyze — summary** | Transactions, Fraud, Legitimate, and the model used |
| **Analyze — accuracy** | Precision, recall, F1 and *fraud missed*, shown only when the file carries a `Class` column (labelled as a retrospective comparison) |
| **Analyze — results table** | Transactions sorted **fraud first**, with amount, verdict badge, and risk score; filters (All / Fraud / Legitimate), pagination, and a CSV download |

Nothing about the previous multi-page design remains: the removed pages, components, chart
library, and API-client surface are gone. The app is now just the two screens above, with no
routing library.

---

## 12. Honesty, limitations, and known constraints

- **It is a demo, not a product.** Models are trained on a 2013-era, two-day European
  dataset. They will not generalise to a real, current card portfolio.
- **Anonymous features.** Because `V1`–`V28` are PCA components with no published meaning,
  the system can say *that* a transaction looks anomalous but cannot explain *why* in
  business terms.
- **Isolation Forest is weak here by design.** Its ~11% precision is reported honestly; it
  exists to contrast supervised and unsupervised approaches.
- **`Time` is a relative offset**, so the models implicitly encode the dataset's 48-hour
  window. There is no real-time drift monitoring.
- **No authentication** is implemented; the API is intended for local/demo use.
- **The dataset is not bundled.** It is ~144 MB and must be downloaded separately (see
  `README.md`) to *train* the models. At run time the app does not need the server-side copy
  — you upload your own CSV on the page.
- **Model binaries are not committed**; the code that regenerates them is. Without them,
  the page shows a clear **"Model not trained"** banner and refuses to fabricate
  predictions.
- **A product-requirements PDF** was supplied with this project but could not be read by
  the build tooling used here, so the implementation follows the written brief. Any
  requirements that existed only in that PDF may not be reflected.

---

## 13. Dataset attribution

The dataset is **Credit Card Fraud Detection** by the Machine Learning Group of the
Université Libre de Bruxelles (ULB), released on Kaggle under the Open Database License
(ODbL) / Database Contents License. It originates from real, anonymised European
cardholder transactions from September 2013. Please cite the original authors when reusing
it.
