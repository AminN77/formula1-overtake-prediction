# Formula One Overtake Prediction

Two independent projects live in this repository.

## `era2026/`

Active work. Overtake prediction for the Formula One regulation era that began in 2026,
built from scratch against in-era data only, with a continual retraining loop as the era
unfolds.

Currently in planning: see [`era2026/PLAN.md`](era2026/PLAN.md).

## `legacy/`

The completed 2022-2025 project (dataset and model versions v1 through v6), a FastAPI
backend, a React frontend, and the offline FastF1 pipeline that produced the datasets.

Frozen. Kept as a reference and as a source of comparison numbers. Not extended.

Tagged at `legacy-v6`. Treat `legacy/` as the repository root when running anything
inside it, for example `cd legacy && docker compose up --build`.

### Why the split

The 2026 regulations replaced DRS with an energy-based Overtake Mode, introduced active
aerodynamics, and changed the power unit to a 50/50 combustion and electric split. The
mechanism of overtaking changed, not just the lap times, so the legacy feature set and
battle definition do not carry over. `era2026/PLAN.md` covers the reasoning in full.
