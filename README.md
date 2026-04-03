# F1 Overtake Prediction

Predict the probability of an on-track overtake from Formula 1 race data derived from `FastF1`. The repository includes:

- offline dataset generation pipelines
- trained model artifacts for multiple versions
- research notebooks for the current model generations
- a React + FastAPI application for single-scenario and batch scoring

The current default model is `v6`.

---

## Current project state

Two model families currently coexist in the app:

- `v2`-`v5`: battle-oriented, next-lap overtake probability models
- `v6`: broader scenario model that predicts whether an overtake occurs within the next 3 laps

Current product behavior:

- model switching happens only on the Models page
- Single, Batch, and Models page state persists during the current app session
- batch scoring is server-backed, paginated, filter-aware, and downloadable as CSV
- batch evaluation is model-aware:
  - `v1`-`v5`: standard next-lap confusion matrix
  - `v6`: standard binary confusion matrix plus horizon breakdown

---

## Architecture

| Layer | Tech | Role |
|-------|------|------|
| Frontend | React 19, TypeScript, Vite, Tailwind, Recharts | Dynamic UI, single prediction flow, batch scoring, model info |
| Backend | FastAPI, scikit-learn, XGBoost, joblib | Model registry, inference, schema generation, sensitivity, batch result storage |
| Pipeline | Python, FastF1 | Offline extraction and feature engineering for `data/v*` datasets |

The frontend is model-aware but not hardcoded to one artifact. It reads schema and metadata from the backend so new artifact versions can be exposed without rebuilding the UI structure from scratch.

---

## Quick start (Docker)

From the repository root:

```bash
docker compose up --build
```

Services:

- Frontend: [http://localhost:3000](http://localhost:3000)
- Backend API: [http://localhost:8000](http://localhost:8000)
- OpenAPI docs: [http://localhost:8000/docs](http://localhost:8000/docs)

Notes:

- model artifacts are committed under `models/artifacts/`
- Docker defaults to `DEFAULT_MODEL=v6`
- you can override the default model at startup, for example:

```bash
DEFAULT_MODEL=v5 docker compose up --build
```

---

## Local development

### Backend

```bash
cd backend
pip install -r requirements.txt
export MODEL_ARTIFACTS_DIR="$(pwd)/../models/artifacts"
export PYTHONPATH="$(pwd)/..:${PYTHONPATH}"
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Notes:

- the Vite dev server proxies `/api` to `http://127.0.0.1:8000`
- set `VITE_API_URL` only when the API is hosted on another origin

---

## Available model versions

The registry currently exposes:

- `v6` (default)
- `v5`
- `v4`
- `v3`
- `v2`

Registry source: `models/artifacts/registry.json`

High-level meaning:

- `v5` uses the widened battle definition: adjacent attacker/defender pairs with gap `< 3.0s`
- `v6` uses broader adjacent scenarios and the primary target `label = overtake_within_3`

---

## Data pipeline

Run pipeline generation from the repository root.

### Generate `v5`

```bash
python3 -m pipeline.main \
  --years 2022 2023 2024 2025 \
  --output-dir data/v5 \
  --cache cache
```

### Generate `v6`

```bash
python3 -m pipeline.main \
  --dataset-version v6 \
  --years 2022 2023 2024 2025 \
  --output-dir data/v6 \
  --cache cache
```

Dataset notes:

- `data/v5/README.md` documents the widened battle-based dataset
- `data/v6/README.md` documents the broader scenario-led dataset

---

## Model training and notebooks

The current research notebooks live in `notebooks/`.

Most relevant notebooks:

- `notebooks/model_testing_4.ipynb`
- `notebooks/model_testing_5.ipynb`
- `notebooks/model_testing_6.ipynb`
- `notebooks/pipeline_testing.ipynb`

Current notebook status:

- the active model notebooks have been trimmed to the key training and holdout analyses
- the notebooks are saved with outputs embedded
- `model_testing_6.ipynb` includes a holdout calibration section for the raw `v6` probabilities

Saved artifacts and metadata are written under `models/artifacts/` and versioned in git.

---

## Batch scoring workflow

Batch mode is designed for uploading one CSV and exploring the scored results in the app.

Current behavior:

1. upload a CSV compatible with the active model
2. run batch scoring
3. receive summary/evaluation plus the first result page
4. paginate or refetch results with filters
5. download the full scored CSV if needed

The backend stores full batch results temporarily and serves:

- paginated rows
- merged viewer filters
- CSV download by result id

This avoids the old “first preview rows only” limitation and keeps large `v6` result sets usable.

---

## API highlights

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Liveness |
| GET | `/api/models/current` | Active model metadata |
| GET | `/api/models/schema` | Dynamic feature schema for the UI |
| GET | `/api/models/versions` | Available model versions |
| GET | `/api/models/importance` | Global feature importance for the active model |
| POST | `/api/models/switch` | Switch active model |
| POST | `/api/predict/single` | Single prediction |
| POST | `/api/predict/derive` | Build derived feature row from UI inputs |
| POST | `/api/predict/batch` | Run batch scoring and create server-side result handle |
| POST | `/api/predict/batch/query` | Fetch paginated / filtered batch rows |
| GET | `/api/predict/batch/download/{result_id}` | Download full scored batch CSV |
| POST | `/api/sensitivity` | 1D sensitivity curve for a numeric feature |

Implementation note:

- if a request body already contains every feature required by `meta["features"]`, the backend can score it directly
- otherwise it engineers a row from the battle/scenario-style inputs used by the UI

---

## Repository layout

```text
backend/                 FastAPI application
frontend/                React SPA
pipeline/                Offline extraction and feature engineering
notebooks/               Model and pipeline research notebooks
models/artifacts/        registry.json + .pkl + *_meta.json
data/                    Versioned datasets such as v5 and v6
docs/                    Improvement proposals and project notes
docker-compose.yml       Local stack orchestration
README.md
```

---

## Improvement proposals

| Doc | Topic |
|-----|-------|
| [docs/IP05.md](docs/IP05.md) | Product architecture overhaul |
| [docs/IP06.md](docs/IP06.md) | Model switching, batch UX, and UI/backend behavior |
| [docs/IP07.md](docs/IP07.md) | Broad scenario-led `v6` dataset and model framing |
| [docs/IP04.md](docs/IP04.md) | Context-aware driver/team features for `v5` |
| [docs/IP03.md](docs/IP03.md) | `v4` dataset and modelling work |
| [docs/IP02.md](docs/IP02.md) | Data quality and feature engineering |
| [docs/IP01.md](docs/IP01.md) | Baseline audit |

---

## License

Academic / coursework context. Check your institution’s and FastF1’s policies before redistributing derived data or artifacts.
