<<<<<<< HEAD
# 🛡️ FraudShield AI

**A credit-card fraud detector for the browser.** From a landing screen, upload a transaction
CSV and every row is classified as **fraud** or **legitimate** by a model trained on the real
[ULB/Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

> ⚠️ **Educational project.** FraudShield AI is **not** connected to any bank, card network,
> or payment processor, and it never blocks a real transaction. Every score is the output of
> a model trained on a public, anonymised dataset and is for demonstration only.

---

## What it does

- **Trains three models** on 284,807 real transactions (492 frauds): Random Forest,
  XGBoost, and an unsupervised Isolation Forest.
- **Two screens, no router:** a landing screen introduces the project, then one analysis
  screen uploads a transaction CSV → every transaction is labelled **Fraud** or
  **Legitimate**, with a summary, a risk score, filtering, pagination, and CSV export.
- **Any column naming:** the model reads the 30 factors it was trained on (`Time`, `V1`..`V28`,
  `Amount`), matched by name — case-insensitively and with common aliases — so a file can call
  them `time`/`timestamp`, `amount`/`amt`, `v1`/`V1`, and its label `Class`/`is_fraud`.
- **Uses the real data** end to end — no synthetic rows, no invented numbers.
- **Reports honest metrics.** When the uploaded file contains its own `Class` column, the
  page shows precision, recall, F1, and the number of frauds the model missed.

### Measured performance (held-out test set, 56,962 rows, 98 frauds)

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **XGBoost** ⭐ | 0.8469 | 0.8469 | 0.8469 | 0.8703 | 0.9792 | 15 | 15 |
| Random Forest | 0.7391 | 0.8673 | 0.7981 | 0.8638 | 0.9775 | 30 | 13 |
| Isolation Forest | 0.1130 | 0.6020 | 0.1903 | 0.1799 | 0.9533 | 463 | 39 |

**XGBoost is the default** — it has the highest **validation** F2 (0.8128). The Isolation
Forest is included as an unsupervised baseline; its low precision is shown honestly, not
hidden. Full methodology and every figure: **[`PROJECT_DETAILS.md`](PROJECT_DETAILS.md)**.

---

## Tech stack

**Backend / ML**
`Python 3.12` · `FastAPI` · `Pydantic`
`pandas` · `NumPy` · `scikit-learn` · `XGBoost` · `imbalanced-learn`

**Frontend**
`React 18` · `Vite 5` · `Tailwind CSS` · `lucide-react`
`Vitest` + `Testing Library`

---

## Project structure

```
CREDITCARD_FRAUD/
├── README.md                     ← you are here
├── PROJECT_DETAILS.md            ← algorithms, dataset, metrics, methodology
├── .gitignore
│
├── data/
│   └── creditcard.csv            ← the Kaggle dataset (not committed; ~144 MB)
│
├── backend/
│   ├── requirements.txt
│   ├── app/                      ← FastAPI application
│   │   ├── main.py               ← app factory, CORS, error handlers
│   │   ├── config.py             ← settings (env vars, safe defaults)
│   │   ├── routes/
│   │   │   ├── health.py         ← GET /api/health
│   │   │   └── analyze.py        ← POST /api/analyze (the CSV upload)
│   │   ├── schemas/              ← Pydantic request/response models
│   │   └── services/
│   │       ├── model_registry.py ← loads the trained artifacts, scores rows
│   │       └── analyze_service.py← parses the CSV and builds the response
│   ├── training/
│   │   ├── preprocess.py         ← FEATURE_ORDER, preprocessor, split, loader
│   │   ├── metrics.py            ← metrics + F2 threshold selection
│   │   ├── train_models.py       ← trains and writes all artifacts
│   │   ├── evaluate_models.py    ← independently recomputes test metrics
│   │   └── eda.py                ← exploratory analysis
│   ├── artifacts/                ← metadata.json / evaluation.json / eda.json
│   │                                are committed; *.joblib / *.parquet are not
│   └── tests/                    ← pytest
│
└── frontend/
    ├── vite.config.js            ← dev proxy + Vitest config
    ├── tailwind.config.js
    └── src/
        ├── App.jsx               ← shell: header, footer, screen switching
        ├── pages/
        │   ├── Home.jsx          ← landing screen
        │   └── Analyze.jsx       ← upload + results screen
        ├── services/api.js       ← API client
        └── lib/format.js         ← formatters
```

---

## Getting started

### Prerequisites

- **Python 3.11+** (developed on 3.12)
- **Node.js 18+** (developed on Node 22) and npm

### 1. Get the dataset

The app is built to run against the **real** dataset. Download it from Kaggle and place it
at `data/creditcard.csv`:

```powershell
# Option A — Kaggle CLI (requires ~/.kaggle/kaggle.json)
pip install kaggle
kaggle datasets download -d mlg-ulb/creditcardfraud -p data --unzip

# Option B — download manually from
#   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
#   and save the CSV as data/creditcard.csv
```

The dataset has **284,807 rows** and 31 columns (the 31st is `Class`). The trained models
must exist before you can classify anything.

### 2. Backend

```powershell
cd backend

python -m venv .venv
.venv\Scripts\Activate.ps1          # PowerShell
# source .venv/bin/activate          # macOS / Linux

pip install -r requirements.txt
```

### 3. Train the models

With the dataset in place (this takes ~40 seconds):

```powershell
# still inside backend/
python -m training.eda              # optional: write eda.json
python -m training.train_models     # trains all three models → artifacts/
python -m training.evaluate_models  # independent verification → evaluation.json
```

This writes the `*.joblib` model binaries, `metadata.json`, `evaluation.json`, and the
held-out test split into `backend/artifacts/`.

> Skip this step and the app still starts — it shows a clear **"Model not trained"**
> banner and refuses to invent a prediction.

### 4. Run the backend

```powershell
# from backend/
python -m uvicorn app.main:app --reload --port 8000
```

- API root: <http://localhost:8000>
- Interactive docs: <http://localhost:8000/docs>
- Health check: <http://localhost:8000/api/health>

> **Port conflict?** Some tools (for example a locally installed Splunk) listen on port
> 8000. This project ships a `frontend/.env` pointing at **8010** for that reason. If you
> use a different port, set it in *both* places.

### 5. Frontend

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>, upload `data/creditcard.csv`, and press **Analyze**.

The Vite dev server proxies `/api` to the backend, so no CORS setup is needed. To point it
at a non-default backend port, edit `frontend/.env` (copy `frontend/.env.example` if it is
missing):

```dotenv
# frontend/.env
VITE_BACKEND_URL=http://127.0.0.1:8010
```

For a production build:

```powershell
npm run build      # outputs frontend/dist/
npm run preview    # serves the build at http://localhost:4173
```

---

## Run it in VS Code

The repo includes a `.vscode/` folder so you can run everything without typing commands.

**One-time setup**

1. `File ▸ Open Folder…` → select `CREDITCARD_FRAUD`.
2. Open a terminal (`Ctrl` + `` ` ``) and install dependencies once:

   ```powershell
   cd backend
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt

   cd ..\frontend
   npm install
   ```

3. If VS Code did not already select your interpreter, run
   `Python: Select Interpreter` and pick `backend\.venv`.

**Run it (one click)**

- `Ctrl` + `Shift` + `P` → **Tasks: Run Task** → **`dev (backend + frontend)`**.

  This starts the FastAPI backend and the Vite dev server in two dedicated terminals.
  Open <http://localhost:5173>.

**Or press F5**

- Pick **Full Stack (backend + frontend)** from the Run and Debug panel.
- **Backend: FastAPI (debug)** attaches the Python debugger for breakpoints in `app/`.

**Run the tests**

- Task: **`backend: run tests`** or **`frontend: run tests`**.

> The backend port in `.vscode/tasks.json` / `launch.json` is **8010** because port 8000 is
> already taken on this machine. Change it there *and* in `frontend/.env` if you need to.

---

## Using the app

There are two screens:

1. **Home** — a short landing screen explaining what the app does, with a **Start analysis**
   button that opens the working screen.
2. **Analyze** — the working screen. Here you:

   - **Choose a CSV** — drag & drop a `creditcard.csv`-style file, or click to browse.
   - **Pick a model** (defaults to XGBoost).
   - **Analyze** — the file uploads (with a progress indicator) and is classified.
   - **Read the results:**

     - **Transactions / Fraud / Legitimate** counts.
     - **Model accuracy** (precision, recall, F1, fraud missed) — only when the file has its
       own `Class` column, and clearly marked as a comparison, never used as an input.
     - A table of the most suspicious transactions (**fraud first**), with a **risk score**,
       filters (**All / Fraud / Legitimate**), pagination, and a **download results** button.

The file's own `Class` column is **never** fed to the model — it is only used to score the
model retrospectively.

### Column names can be anything

The model always consumes the same 30 factors — `Time`, `V1`..`V28` and `Amount`. It decides
which uploaded column is which **by name**, and that matching is case-insensitive,
whitespace/punctuation-insensitive and alias-aware:

| Factor | Recognised names (examples) |
| --- | --- |
| Time | `Time`, `time`, `timestamp`, `unix_time`, `transaction_time` |
| Amount | `Amount`, `amount`, `amt`, `transaction_amount`, `transaction_dollar_amount` |
| V1..V28 | `V1`..`V28`, `v1`..`v28`, `v01`..`v28` |
| Label (optional) | `Class`, `is_fraud`, `label`, `target`, `fraudulent` |

Extra columns are ignored, and the label column is used only for the accuracy read-out. A file
that does **not** contain the factors at all (for example a card/date/amount statement) is
rejected with a message that lists what is missing and what was found.

---

## API

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/` | Service banner |
| `GET` | `/api/health` | Status, dataset/model readiness, available models |
| `POST` | `/api/analyze` | Upload a CSV (`file`) and classify every row (`model` optional) |

Example:

```powershell
curl -X POST http://localhost:8010/api/analyze `
  -F "file=@data/creditcard.csv" `
  -F "model=xgboost"
```

Response (abridged):

```json
{
  "filename": "creditcard.csv",
  "model": "xgboost",
  "model_display_name": "XGBoost",
  "score_type": "probability",
  "threshold": 0.21155192,
  "total_rows": 284807,
  "processed_rows": 284807,
  "rejected_rows": 0,
  "fraud_rows": 483,
  "legitimate_rows": 284324,
  "results": [
    { "row": 42758, "amount": 0.0, "prediction": 1, "label": "Fraud",
      "score": 1.0, "actual_class": 1 }
  ],
  "results_truncated": true,
  "evaluation": { "precision": 0.944099, "recall": 0.926829, "f1": 0.935385 }
}
```

Full request/response schemas are in the interactive docs at `/docs`.

---

## Tests

```powershell
# Backend
cd backend
python -m pytest

# Frontend
cd frontend
npm test
```

- **Backend:** preprocessing and feature order, metric math (including the single-class AUC
  edge case), health, the analyze endpoint (fraud vs. legitimate classification, rejection
  of malformed rows, flexible column-name resolution and missing-factor detection, unknown
  models), and the not-trained state.
- **Frontend:** formatters used by the results table.

---

## Configuration

All settings are environment variables with safe local defaults — **no secrets live in
source**.

| Variable | Default | Purpose |
| --- | --- | --- |
| `FRAUDSHIELD_DATA_PATH` | `data/creditcard.csv` | Dataset location (training only) |
| `FRAUDSHIELD_ARTIFACTS_DIR` | `backend/artifacts` | Trained artifacts location |
| `FRAUDSHIELD_CORS_ORIGINS` | localhost 5173 / 4173 / 3000 | Allowed browser origins |
| `FRAUDSHIELD_MAX_UPLOAD_BYTES` | `536870912` (512 MB) | Upload size cap |
| `FRAUDSHIELD_MAX_BATCH_ROWS` | `500000` | Max rows per file |
| `FRAUDSHIELD_MAX_RESPONSE_ROWS` | `2000` | Rows returned to the browser (summary covers all) |
| `VITE_BACKEND_URL` | `http://127.0.0.1:8000` | Frontend dev-proxy target (set in `frontend/.env`) |
| `VITE_API_BASE_URL` | *(unset)* | Point the SPA directly at a deployed backend |

---

## How the results are produced (and how they are kept honest)

- The **real** dataset and real dataset-derived metrics are used throughout — no synthetic
  rows and no hand-entered numbers.
- `Class` is **never** a model input; it is only the training target and a retrospective
  comparison label.
- Preprocessing is fitted on the **training split only**; SMOTE is applied to the
  **training fold only**.
- Thresholds and the recommended model are chosen on the **validation set** using
  recall-weighted **F2**; the **test set is used once**, at the end.
- `training/evaluate_models.py` independently recomputes the test metrics from the saved
  artifacts and records whether they match `metadata.json` (they do, for all three models).
- If precision–recall/ROC AUC are mathematically undefined (a single-class comparison), they
  are reported as `null` **with an explanation**, never a misleading `0`.
- Missing models → an explicit **"Model not trained"** state. The app never fabricates a
  prediction.

Full detail: **[`PROJECT_DETAILS.md`](PROJECT_DETAILS.md)**.

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| Backend fails to bind a port | Another service owns it (Splunk often uses 8000). Run `python -m uvicorn app.main:app --port 8010` and set `VITE_BACKEND_URL=http://127.0.0.1:8010` in `frontend/.env`. |
| Page shows "Model not trained" | Run `python -m training.train_models` from `backend/`. |
| Upload rejected / no results | The file must contain the model's factors — `Time`, `V1..V28`, `Amount` — under any column naming (see *Column names can be anything*). A card/date/amount statement has no V-features and cannot be scored. |
| Frontend shows a network error | Make sure the backend is running and the proxy target port matches `frontend/.env`. |

---

## Limitations

- Trained on a 2013-era, two-day European dataset; **not** production-ready and likely not
  generalisable to a real card portfolio.
- `V1`–`V28` are anonymous PCA components, so the system can flag an anomaly but cannot
  explain it in business terms.
- No authentication — intended for local/demo use only.
- The Isolation Forest's low precision is expected for an unsupervised model here and is
  reported as-is.

---

## Dataset attribution

**Credit Card Fraud Detection** — Machine Learning Group, Université Libre de Bruxelles
(ULB), distributed via Kaggle under the ODbL / Database Contents License. It contains
anonymised transactions by European cardholders from September 2013.

---

## License

Released for educational purposes. The bundled code is provided as-is; the dataset remains
subject to its own license terms (see attribution above).
=======
# creditcard_fraud
>>>>>>> 283ad73bf36095dbe1f8fe95ce75c761a089b260
