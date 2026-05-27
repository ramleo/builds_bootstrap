# tests/CLAUDE.md — Testing Agent Spec

## 🧪 Testing Agent
**Trigger:** After pipeline or API is built
**Delegate when:** Writing `tests/test_pipeline.py`, running the full test suite, reporting results.
**Input to provide:** Pipeline path, label encoder path, data path, expected accuracy threshold.
**Agent must:** Write the test script (artifact integrity, single-sample predictions, full test-set evaluation, per-class accuracy, consistency check, probability check); run it; return ONLY the test summary output.
**Returns:** Pass/fail per test, overall accuracy, confirmation of 16/16 checks or list of failures.

---

## 🧪 End-to-End Test Suite (`tests/run_e2e.py`)

`tests/run_e2e.py` validates the full user journey — from `bootstrap.py` execution to a live prediction API — across **5 suites (~80 checks)**. It is stdlib-only (no pytest required) and runs everything in a temporary directory, isolated from the repo.

**Repository:** [github.com/ramleo/ml-pipeline-tests](https://github.com/ramleo/ml-pipeline-tests)

**How to run (from repo root):**
```bash
# Fast mode — skips Docker suite (~5 min total)
python3 tests/run_e2e.py --fast

# Full run including Docker build and smoke tests (~10 min)
python3 tests/run_e2e.py

# Run a single suite
python3 tests/run_e2e.py --suite 1   # bootstrap & project creation
python3 tests/run_e2e.py --suite 2   # ML pipeline artifacts
python3 tests/run_e2e.py --suite 3   # app.py, Dockerfile & frontend content
python3 tests/run_e2e.py --suite 4   # live API via uvicorn (incl. GET /)
python3 tests/run_e2e.py --suite 5   # Docker build & smoke tests (incl. GET /)

# Test against the published GitHub version instead of local bootstrap.py
python3 tests/run_e2e.py --from-github --fast
```

**Suites and checks:**

| Suite | Name | Checks | Speed |
|---|---|---|---|
| 1 | Bootstrap & Project Creation | ~11 | ~2 min |
| 2 | Pipeline Artifacts | ~13 | ~1 s |
| 3 | App, Dockerfile & Frontend Generation | ~26 | ~1 s |
| 4 | Live API via uvicorn | ~20 | ~15 s |
| 5 | Docker Build & Smoke Tests | ~10 | ~3 min |

**Suite 1 — Bootstrap & Project Creation:**
- `bootstrap.py` exists and passes `py_compile` syntax check
- `bootstrap.py` runs end-to-end without error (feeds piped inputs: project name, CSV path, platform, GitHub, launch choice 2)
- Project folder created with timestamped name (`project_TIMESTAMP/`)
- All required template files present (CLAUDE.md, README.md, auto_pipeline.py, requirements.txt, etc.)
- `.ml_config.json` contains all required keys
- `.venv/` exists with working Python binary
- Key packages importable (sklearn, pandas, fastapi, uvicorn, joblib)
- Setup scripts (`bootstrap.py`, `start.sh`, `init.py`) NOT copied into project

**Suite 2 — Pipeline Artifacts (after `auto_pipeline.py` runs):**
- `models/final_pipeline.pkl` exists and loads with `joblib`
- `models/label_encoder.pkl` exists
- `plots/eda_correlation.png` exists
- `plots/eda_target.png` exists
- `docs/auto_summary.md` exists with required section headings
- Pipeline produces a valid prediction on a sample DataFrame row
- `predict_proba` returns 2-class probabilities

**Suite 3 — App, Dockerfile & Frontend Generation (after post-pipeline menu):**
- `app.py` exists and passes `py_compile`
- `app.py` defines `/health`, `/predict`, `/predict/batch` endpoints
- `app.py` loads `final_pipeline.pkl` and `label_encoder.pkl`
- `app.py` defines an `InputData` Pydantic model with correct feature fields
- `app.py` returns `probabilities` for classification
- `app.py` imports `FileResponse` and `CORSMiddleware`
- `app.py` has a `GET /` route to serve the themed UI
- `Dockerfile` exists with multi-stage build (`FROM python:3.11-slim AS builder`)
- `Dockerfile` creates a non-root user (`appuser`)
- `Dockerfile` exposes port 8000 and starts with uvicorn CMD
- `Dockerfile` copies `index.html*` into the image
- `.dockerignore` exists and excludes `.venv/` and `data/`
- `index.html` exists and has non-zero size
- `index.html` contains no raw `TMPL_` placeholders (all substituted)
- `index.html` contains an `<html` tag and a `<form` element
- `index.html` contains themed CSS (`gradient`)

**Suite 4 — Live API via uvicorn:**
- uvicorn starts on a free port without error
- `GET /health` → 200, body `{"status": "ok"}`
- `POST /predict` with full payload → 200, contains `prediction` key
- `POST /predict` response contains `probabilities` key
- Returned prediction is a valid class label
- `POST /predict` with missing optional fields → 200 (imputed, not rejected)
- `POST /predict/batch` with 2 items → 200, 2 results returned
- `GET /docs` (Swagger UI) → 200
- `GET /` → 200 with `Content-Type: text/html`
- `GET /` body contains `<html` tag (themed prediction UI)

**Suite 5 — Docker Build & Smoke Tests:**
- `docker build` completes successfully
- `docker run` starts container on a free port
- Container responds to `GET /health` with status 200
- Container responds to `POST /predict` with a valid prediction
- `POST /predict/batch` inside Docker returns correct results
- `GET /` inside Docker → 200 serving `index.html`
- Container stops and is removed cleanly

**Exit codes:** `0` = all suites pass, `1` = one or more failures (suitable for CI use).
