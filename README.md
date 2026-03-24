# F1 Overtake Prediction

Predict the probability of an **on-track overtake** between two consecutive cars using lap-level battle features derived from **FastF1** race data. The project includes an offline data pipeline, research notebooks, serialized ML models, and a **React + FastAPI** web application (IP05).

---

## Architecture (IP05)

| Layer | Tech | Role |
|-------|------|------|
| **Frontend** | React 19, TypeScript, Vite, Tailwind, Recharts | F1-inspired UI; dynamic form from `/api/models/schema`; batch CSV upload |
| **Backend** | FastAPI, sklearn/XGBoost joblib pipelines | Model registry, `/api/predict/*`, `/api/sensitivity`, OpenAPI at `/docs` |
| **Pipeline** | Python, FastF1 | Offline generation of `data/v*/battles_*.csv` |

The frontend is **model-agnostic**: it loads feature definitions from the backend so new model versions do not require hardcoded field lists in the client.

---

## Quick start (Docker)

1. Trained artifacts live under `models/artifacts/` (at minimum `overtake_model_v5.pkl` + `overtake_model_v5_meta.json` + `registry.json`); they are versioned in this repository.
2. From the repository root:

```bash
docker compose up --build
```

- **Frontend:** http://localhost:3000 (nginx → proxies `/api` to backend)
- **Backend API:** http://localhost:8000 — interactive docs: http://localhost:8000/docs  
- Configure default model: `DEFAULT_MODEL=v4 docker compose up`

---

## Local development

### Backend

```bash
cd backend
pip install -r requirements.txt
export MODEL_ARTIFACTS_DIR="$(pwd)/../models/artifacts"
export PYTHONPATH="$(pwd)/..:${PYTHONPATH}"   # repo root for `pipeline` package
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Vite dev server proxies `/api` to `http://127.0.0.1:8000` (override with `VITE_PROXY_TARGET`).

Set `VITE_API_URL` only if the API is on another origin; leave empty when using the proxy.

---

## Data pipeline (offline)

From the **repository root** (so the `pipeline` package resolves):

```bash
export PYTHONPATH=.
python -m pipeline.main --years 2022 2023 2024 2025 --output-dir data/v5 --cache cache
```

See `data/v5/README.md` for column descriptions.

---

## Model training

Research notebooks live in `notebooks/` (e.g. `model_testing_5.ipynb` for the v5 model). Trained pipelines and JSON metadata are written under `models/artifacts/` and committed alongside the code.

---

## Project layout

```
├── backend/                 # FastAPI application
├── frontend/                # React SPA (Vite)
├── pipeline/                # FastF1 battle extraction & feature enrichment
├── notebooks/               # Jupyter experiments (model_testing_*.ipynb)
├── models/artifacts/        # registry.json + *.pkl + *_meta.json
├── data/                    # Versioned battle CSVs (v1…v5)
├── docs/                    # IP01–IP05, roadmap, images
├── docker-compose.yml
└── README.md
```

---

## API highlights

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Liveness |
| GET | `/api/models/current` | Active model metadata |
| GET | `/api/models/schema` | Feature list + types for dynamic UI |
| POST | `/api/models/switch` | `{"version": "v4"}` — switch loaded model |
| POST | `/api/predict/single` | JSON `inputs` → probability + optional local impacts |
| POST | `/api/predict/batch` | multipart CSV + `threshold` + `filter_pits` query params |
| POST | `/api/sensitivity` | 1D curve for a numeric feature |

If the request JSON contains **all** keys in `meta["features"]`, the backend uses them directly as the feature vector (useful for tests); otherwise it engineers a row from battle-oriented UI fields (same behaviour as the legacy Gradio app).

---

## Improvement proposals

| Doc | Topic |
|-----|--------|
| [docs/IP05.md](docs/IP05.md) | **Product overhaul** — FastAPI + React + Docker |
| [docs/IP04.md](docs/IP04.md) | Context-aware driver/team features (v5) |
| [docs/IP03.md](docs/IP03.md) | v4 dataset & modelling |
| [docs/IP02.md](docs/IP02.md) | Data quality & features |
| [docs/IP01.md](docs/IP01.md) | Baseline audit |

---

## License

Academic / coursework context — see your institution’s policies for redistribution of FastF1-derived data.
