#!/usr/bin/env python3
"""
bootstrap.py — ML Pipeline Template Bootstrap
Creates an ML project folder directly — no intermediate template folder.

Usage:
  python3 bootstrap.py          # interactive prompts → my-project_TIMESTAMP/

Via Docker:
  docker build -t builds_bootstrap -f Dockerfile.bootstrap .
  docker run --rm -v $(pwd):/output builds_bootstrap
"""

import os, sys, stat, shutil, subprocess, json
from pathlib import Path
from datetime import datetime, timezone

VERSION = "1.0.0"

# ── Colours ─────────────────────────────────────────────────────────
G = "\033[0;32m"; C = "\033[0;36m"; B = "\033[1m"
Y = "\033[1;33m"; R = "\033[0;31m"; X = "\033[0m"

# ── Input helpers ────────────────────────────────────────────────────
def _prompt(msg, default=""):
    suffix = f" (default: {default})" if default else ""
    val = input(f"{msg}{suffix}: ").strip()
    return val or default

def _menu(title, options, default="1"):
    print(f"\n{B}{title}{X}")
    for key, label in options:
        print(f"  {key}) {label}")
    choice = input(f"Enter choice (default: {default}): ").strip()
    return choice or default

def collect_inputs():
    print(f"\n{B}── Project Setup ──────────────────────────────────────{X}")
    project_name = _prompt("Project name", "ml-project").replace(" ", "-")

    dataset_path, dataset_filename = "", ""
    raw = _prompt("Dataset CSV path (press Enter to skip)", "")
    if raw:
        p = Path(raw).expanduser().resolve()
        if p.is_file():
            dataset_path, dataset_filename = str(p), p.name
            print(f"  {G}✔ Found: {dataset_filename}{X}")
        else:
            print(f"  {Y}⚠ File not found — copy to data/ later.{X}")

    deploy_choice = _menu("Deployment platform:", [
        ("1", "Ask me later"),
        ("2", "Render        (free tier, recommended)"),
        ("3", "Fly.io"),
        ("4", "Railway"),
        ("5", "AWS App Runner"),
        ("6", "GCP Cloud Run"),
        ("7", "Azure Container Apps"),
        ("8", "Skip (local / Docker only)"),
    ], default="1")
    platform = {
        "1": "ask_later", "2": "render",  "3": "fly.io",
        "4": "railway",   "5": "aws",     "6": "gcp",
        "7": "azure",     "8": "none",
    }.get(deploy_choice, "ask_later")

    print(f"\n{B}GitHub setup:{X}")
    gh_detected = ""
    try:
        r = subprocess.run(["gh", "api", "user", "--jq", ".login"],
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            gh_detected = r.stdout.strip()
    except Exception:
        pass

    if gh_detected:
        print(f"  {G}✔ GitHub account detected: {gh_detected}{X}")
        gh_user = _prompt(f"  GitHub username (Enter to use '{gh_detected}')", gh_detected)
    else:
        gh_user = _prompt("  GitHub username (Enter to skip)", "")

    gh_repo, gh_vis = project_name, "skip"
    if gh_user:
        gh_repo = _prompt("  GitHub repo name", project_name).replace(" ", "-")
        vis_choice = _menu("  GitHub repo visibility:", [("1", "Public"), ("2", "Private")], "1")
        gh_vis = "private" if vis_choice == "2" else "public"
    else:
        print(f"  {Y}⚠ No GitHub username — skipping GitHub setup.{X}")

    return dict(
        project_name=project_name, dataset_path=dataset_path,
        dataset_filename=dataset_filename, platform=platform,
        github_username=gh_user, github_repo=gh_repo, github_visibility=gh_vis,
    )

def write_config(cfg, dest):
    py = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    u, r = cfg["github_username"], cfg["github_repo"]
    config = {
        "project_name":        cfg["project_name"],
        "dataset_filename":    cfg["dataset_filename"] or "<not provided yet>",
        "dataset_path":        f"data/{cfg['dataset_filename']}" if cfg["dataset_filename"] else "<not provided yet>",
        "target_column":       "auto-detect",
        "task_type":           "auto-detect",
        "deployment_platform": cfg["platform"],
        "github_username":     u,
        "github_repo":         r,
        "github_visibility":   cfg["github_visibility"],
        "github_url":          f"https://github.com/{u}/{r}" if u else "",
        "python_version":      py,
        "created_at":          datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "venv_path":           ".venv",
        "template_version":    VERSION,
    }
    (dest / ".ml_config.json").write_text(json.dumps(config, indent=2))
    print(f"  {G}✔ .ml_config.json written{X}")

# ── File contents ────────────────────────────────────────────────────

FILES = {}

# ════════════════════════════════════════════════════════════════════
FILES["CLAUDE.md"] = '''# Role and Objective
You are an expert Data Scientist and Autonomous AI Agent. Your task is to dynamically discover data, build, train, and validate a reproducible end-to-end machine learning pipeline for any tabular dataset.

# Token Management & Agentic Architecture
1. **Sub-Agent Delegation**: For token-heavy tasks, delegate the task to a specialized sub-agent as defined in the Routing Guide below.
2. **Context Isolation**: Instruct sub-agents to complete their specific task in isolation and return only the final, clean code script or summary to you.
3. **Main Session Conservation**: Keep this main session clean. Do not allow large blocks of raw data, training logs, or unoptimized trial-and-error code to pollute the main context history.

# Sub-Agent Routing Guide

## When to Use a Sub-Agent
Delegate to a sub-agent whenever a task is **token-heavy, self-contained, or produces large intermediate output** (raw data, training logs, generated code). Keep the main session lean — it should only receive clean summaries and final artifacts.

**Rule of thumb:** If a task requires more than ~20 lines of output or involves trial-and-error iteration, it belongs in a sub-agent.

## Sub-Agent Roster (see local CLAUDE.md files for full specs)

| Agent | Trigger | Local spec |
|---|---|---|
| 🔬 EDA Agent | Step 2 — EDA | @src/CLAUDE.md |
| ⚙️ Data Engineering Agent | Step 3 — Preprocessing | @src/CLAUDE.md |
| 🏆 Optimization Agent | Steps 4–6 — Training | @src/CLAUDE.md |
| 🌐 FastAPI Agent | API development | @src/CLAUDE.md |
| 🐳 Docker Agent | Step 12 — Docker | @deploy/CLAUDE.md |
| 📄 Documentation Agent | Steps 8, docs | @docs/CLAUDE.md |
| 🧪 Testing Agent | After pipeline | @tests/CLAUDE.md |
| 🚀 Git & Deploy Agent | Steps 11–13 | @deploy/CLAUDE.md |
| ☁️ Cloud Deploy Agent | Step 14 — Cloud | @deploy/CLAUDE.md |

## Token Conservation Rules
1. **Never** paste raw CSV data, full training logs, or large DataFrames into the main session.
2. **Never** return a full generated script to the main session — return confirmation + printed output only.
3. Sub-agents must be given **all necessary context upfront** (file paths, parameters, prior results) so they do not need to ask back-and-forth.
4. Each sub-agent handles **one phase** only — do not chain multiple phases into a single sub-agent call.
5. If a sub-agent fails, fix the specific issue and re-run that agent only — do not re-run the entire pipeline.

# Operational Rules
1. **Immediate Execution**: Do not greet or explain. Start work immediately upon reading this file.
2. **State Tracking**: Update the task list below by checking off items as you complete each phase.
3. **Reproducibility**: Always use `random_state=42` for data splits and model initializations.

# Project Scope (loaded from .ml_config.json on startup)
- **Target CSV File**: `<read from .ml_config.json → dataset_path, or ask user>`
- **Target Variable / Label**: `<auto-detect from dataset, or ask user>`
- **ML Task Type**: `<auto-detect: classification if categorical target, regression if numeric>`
- **Deployment Platform**: `<read from .ml_config.json → deployment_platform>`
- **Tech Stack**: Python, Pandas, Scikit-Learn, Joblib, FastAPI

# ML Process Checklist
- [ ] 0.  Virtual Environment Setup (Create .venv, activate, pip install -r requirements.txt)
- [ ] 1.  Workspace Scan & Dataset Auto-Discovery
- [ ] 2.  Data Inspection & EDA (Via EDA Agent: Detect task type, save plots, report summary)
- [ ] 3.  Automated Preprocessing & Cleaning (Via Data Engineering Agent: Build robust pipelines)
- [ ] 4.  Feature Scaling & Train-Test Split (80/20 stratified split for classification, random for regression)
- [ ] 5.  Baseline Model Training & Tuning (Via Optimization Agent: Fit and tune appropriate model)
- [ ] 6.  Model Evaluation (Generate metrics: Classification Report or RMSE/R2 based on task type)
- [ ] 7.  Pipeline Export (Save the entire trained preprocessing + model pipeline as `models/final_pipeline.pkl`)
- [ ] 8.  Summary Report (Create `docs/summary.md`)
- [ ] 9.  Requirements File (Create `requirements.txt` with pinned library versions)
- [ ] 10. Workspace Reorganisation (Create subfolders; move files to reduce clutter)
- [ ] 11. Git Initialisation & GitHub Push (git init → .gitignore → commit → gh repo create → push)
- [ ] 12. Dockerfile & Containerisation (Multi-stage Dockerfile + .dockerignore; build & test locally; push to GitHub)
- [ ] 13. Cloud Deployment (Deploy to chosen platform via render.yaml / fly.toml / railway.toml / apprunner.yaml)
- [ ] 14. Generic Cloud Deployment (Optional: redeploy to AWS / GCP / Azure / Fly.io / Railway via Cloud Deploy Agent)

# Instructions for Initialization

1. **Check for `.ml_config.json`** in the project root:
   - If found: read `dataset_path`, `target_column`, `deployment_platform`, `github_username`, `github_repo`, `github_visibility` from it.
   - If not found: ask the user for the following, **ONE question at a time**:
     1. "What is your dataset CSV path?" — accept a full file path OR a filename if the file is already in `data/`. If the user presses Enter without providing a path, check the `data/` folder for any `.csv` file and use it if found; otherwise ask again.
     2. "Which column is the target variable? (or press Enter to auto-detect)"
     3. "What is your GitHub username? (press Enter to skip)"
     4. "What should the GitHub repo be named? (default: `<project_name>`)"
     5. "Deployment platform? [render / fly.io / railway / aws / gcp / azure / none]"
     Then write all answers to `.ml_config.json` before proceeding.

2. **Check for `.venv/`** virtual environment:
   - If `.venv/` is missing:
     1. Run: `python3 -m venv .venv`
     2. Run: `.venv/bin/pip install --upgrade pip -q`
     3. Run: `.venv/bin/pip install -r requirements.txt -q`
     4. For all subsequent Python commands, use `.venv/bin/python` (not `python3`).
     Mark Step 0 complete.
   - If `.venv/` exists:
     - Use `.venv/bin/python` and `.venv/bin/pip` for all commands.

3. **Auto-detect task type** from the target column:
   - If target has ≤ 20 unique values or dtype is object/bool → **Classification**
   - Otherwise → **Regression**

4. **Scan the workspace** for the CSV file; read its first 5 rows and column names.

5. **Show confirmation summary** and wait for the user to confirm before proceeding:
   ```
   Dataset   : <dataset_path>
   Target    : <target_column>
   Task      : <auto-detected type: Classification or Regression>
   Platform  : <deployment_platform>
   GitHub    : https://github.com/<github_username>/<github_repo>

   Proceed with the pipeline? [Y/n]
   ```
   Only continue after the user confirms with Y (or Enter).

6. Once confirmed, immediately launch the EDA Agent (Step 2).
'''

# ════════════════════════════════════════════════════════════════════
FILES["src/CLAUDE.md"] = '''# src/CLAUDE.md — Data Pipeline Agent Specs & Post-Pipeline Steps

## 🔬 EDA Agent
**Trigger:** Step 2 — Data Inspection & EDA
**Delegate when:** Profiling columns, plotting distributions, computing correlations, detecting outliers.
**Input to provide:** CSV file path, target variable name, task type (classification / regression).
**Agent must:** Save all plots to `plots/`; return ONLY a bullet-point text summary (no raw data, no code).
**Returns:** Dataset shape, quality issues, class balance (or target distribution), top feature insights, outlier summary, correlation highlights.

---

## ⚙️ Data Engineering Agent
**Trigger:** Step 3 — Automated Preprocessing & Cleaning
**Delegate when:** Building preprocessing pipelines, encoding categoricals, imputing missing values, writing `src/preprocess.py`.
**Input to provide:** CSV path, target column, task type, EDA summary (data types, missing counts, outlier findings).
**Agent must:** Write the complete `src/preprocess.py` and execute it; return ONLY confirmation + printed output. Do not return the full script.
**Preprocessing rules:**
- Drop non-feature columns (e.g. Id, index columns, Name, Ticket, Cabin) if detected
- Use **CoW-safe pandas assignment** — always use `df = df.assign(col=...)` or `df.loc[:, col] = ...` instead of `df[col] = ...` after any drop/copy to avoid FutureWarning with pandas 2.x+
- For any derived features (e.g. extracting Title from Name), add them BEFORE dropping the source column; use `df = df.assign(Title=df[\'Name\'].apply(extract_title))`; then drop the source column
- Impute missing values: median for numeric, most-frequent for categorical
- Encode target: LabelEncoder for classification, leave numeric for regression
- Build a `ColumnTransformer` preprocessor (do NOT apply it yet — save it unfitted as `models/preprocessor.pkl`)
- Scale features: StandardScaler for classification, RobustScaler for regression
- 80/20 stratified split (classification) or random split (regression), random_state=42 — split the **feature-engineered but un-preprocessed** DataFrames
- Save ALL of the following to `models/`:
  - `X_train_raw.pkl`, `X_test_raw.pkl` — raw (feature-engineered, unscaled) DataFrames as joblib pkl
  - `y_train.npy`, `y_test.npy` — encoded label arrays
  - `label_encoder.pkl` — fitted LabelEncoder (classification only)
  - `preprocessor.pkl` — fitted ColumnTransformer (fit on X_train_raw only)
- Also save `X_train.npy`, `X_test.npy` (preprocessed arrays) for reference/testing
**Returns:** Confirmation script ran, split shapes, class/target distribution, paths of all saved artifacts including X_train_raw.pkl and preprocessor.pkl.

---

## 🏆 Optimization Agent
**Trigger:** Steps 4–6 — Feature Scaling, Model Training & Evaluation
**Delegate when:** Running GridSearchCV, fitting multiple candidate models, evaluating on test set.
**Input to provide:** Paths to `models/X_train_raw.pkl`, `models/X_test_raw.pkl`, `models/y_train.npy`, `models/y_test.npy`, `models/preprocessor.pkl`, task type.

**Candidate models and hyperparameter grids:**
- Classification:
  - `LogisticRegression(solver=\'saga\', random_state=42)` — grid: `C=[0.1, 1, 10]`
  - `RandomForestClassifier(random_state=42)` — grid: `n_estimators=[100, 200], max_depth=[None, 5, 10]`
  - `SVC(probability=True, random_state=42)` — grid: `C=[1, 10], kernel=[\'rbf\', \'linear\']`
  - `GradientBoostingClassifier(random_state=42)` — grid: `n_estimators=[100, 200], max_depth=[3, 5], learning_rate=[0.05, 0.1]`
- Regression:
  - `Ridge()` — grid: `alpha=[0.1, 1.0, 10.0]`
  - `RandomForestRegressor(random_state=42)` — grid: `n_estimators=[100, 200], max_depth=[None, 5, 10]`
  - `GradientBoostingRegressor(random_state=42)` — grid: `n_estimators=[100, 200], max_depth=[3, 5], learning_rate=[0.05, 0.1]`
  - `SVR()` — grid: `C=[1, 10], kernel=[\'rbf\', \'linear\']`

**Pipeline architecture — IMPORTANT:**
- Load `models/preprocessor.pkl` (already fitted ColumnTransformer)
- For each candidate, build: `Pipeline([(\'preprocessor\', preprocessor), (\'model\', candidate)])`
- Run `GridSearchCV` on this full pipeline using `X_train_raw` DataFrame (NOT the .npy files)
- Select best model across all candidates by CV score
- Refit the winning pipeline on full `X_train_raw` / `y_train`
- Evaluate on `X_test_raw` / `y_test`
- Save as `models/final_pipeline.pkl`

**Agent must:** Run full hyperparameter search, select best model, build and save final pipeline, evaluate; return ONLY the results table and metrics.
**Returns:** Best model name, optimal hyperparameters, CV score, test metrics (accuracy + classification report, or RMSE + R²), confirmation `final_pipeline.pkl` saved.

---

## 🌐 FastAPI Agent
**Trigger:** API development task
**Delegate when:** Writing or expanding the FastAPI `app.py` (new endpoints, input validation, response schemas, batch prediction).
**Input to provide:** Model path (`models/final_pipeline.pkl`), preprocessing script path (`src/preprocess.py`), task type, list of endpoints needed.

**BEFORE writing app.py — inspect the pipeline:**
Run this inspection snippet to discover exactly what the pipeline expects:
```python
import joblib, pandas as pd
pipeline = joblib.load(\'models/final_pipeline.pkl\')
print("Steps:", list(pipeline.named_steps.keys()))
pre = pipeline.named_steps[\'preprocessor\']
print("Numeric features:", pre.transformers_[0][2])
print("Categorical features:", pre.transformers_[1][2])
```
Also read `src/preprocess.py` to identify any feature engineering done **before** the sklearn pipeline (e.g. Title extraction from Name, date parsing, ratio columns). These steps must be replicated in `app.py`.

**app.py requirements:**
- Accept ALL original dataset columns the user would naturally provide (e.g. `Name`, `PassengerId`) in the Pydantic input model — but only forward the pipeline\'s expected feature columns to `pipeline.predict()`
- Replicate all pre-pipeline feature engineering (e.g. extract `Title` from `Name`) inside `app.py` before calling predict
- Make columns that the pipeline can impute (Age, Fare, etc.) `Optional` with `None` default
- Use `predict_proba` for probability output (SVC must be trained with `probability=True`)
- Endpoints: `GET /` (redirect to `/docs`), `GET /health`, `POST /predict`, `POST /predict/batch`
- For `GET /`: `from fastapi.responses import RedirectResponse` and return `RedirectResponse(url="/docs")` with `include_in_schema=False`
- Write the **complete, final `app.py` in ONE pass** — do not write a partial version and patch it later

**Agent must:** Inspect the pipeline first, write the complete `app.py`, start the server, smoke-test all endpoints via curl, return ONLY confirmation + curl responses.
**Returns:** Confirmation all endpoints respond correctly, sample curl outputs (`/health`, `/predict`, `/predict/batch`), any errors encountered.

---

## Step 8 — Create `docs/summary.md`
After completing Steps 1–7, delegate to Documentation Agent. The summary must include:
1. Dataset Overview (shape, quality, class/target balance, feature descriptions)
2. Exploratory Data Analysis (key insights, outliers, correlations, plot index)
3. Preprocessing Pipeline (steps applied, split shapes, distribution)
4. Model Selection & Hyperparameter Tuning (all candidates, CV scores, best hyperparameters)
5. Model Evaluation (test metrics, classification report or RMSE/R² table)
6. Final Pipeline Architecture (text flow diagram)
7. Artifacts (table of all generated files with descriptions)
8. Reproducibility (Python snippet to reload and run the final pipeline)

---

## Step 9 — Create `requirements.txt`
Detect exact installed versions by running:
```python
import pandas, numpy, sklearn, joblib, matplotlib, seaborn, fastapi, uvicorn
```
Create `requirements.txt` at project root with pinned versions. Include a header comment: project name, generation date, Python version.

---

## Step 10 — Workspace Reorganisation
Reorganise into the standard folder structure. Perform all moves, then update file paths in scripts.

**Target structure:**
```
project-root/
├── data/               ← CSV datasets
├── models/             ← .pkl artifacts & .npy splits
├── plots/              ← EDA charts (.png)
├── src/preprocess.py   ← preprocessing script
├── tests/test_pipeline.py
├── docs/               ← all markdown documentation
├── app.py              ← stays at root (Render: uvicorn app:app)
├── Dockerfile          ← stays at root
├── render.yaml / fly.toml / railway.toml  ← stays at root
├── requirements.txt    ← stays at root
└── CLAUDE.md           ← stays at root
```

Steps: create `src/`, `tests/`, `docs/` → move scripts → move `*.md` files (except `CLAUDE.md`) → remove `__pycache__/` → commit.
'''

# ════════════════════════════════════════════════════════════════════
FILES["deploy/CLAUDE.md"] = '''# deploy/CLAUDE.md — Deployment Agent Specs & Steps 11–12

For Steps 13–14 (Render & generic cloud) see @deploy/cloud.md.

---

## 🐳 Docker Agent
**Trigger:** Step 12 — Dockerfile & Containerisation
**Delegate when:** Writing the Dockerfile, building the image, running the container, smoke-testing endpoints inside Docker.
**Input to provide:** Project root path, `requirements.txt` path, `app.py` location, `models/` path, desired base image.
**Agent must:** Write `Dockerfile` and `.dockerignore`, build the image, run the container, hit `/health` and `/predict`, stop and remove container; return ONLY confirmation + test outputs.
**Returns:** Build success confirmation, image size, smoke-test results, any warnings or errors.

---

## 🚀 Git & Deploy Agent
**Trigger:** Steps 11–13 — Git, GitHub, Render deployment
**Delegate when:** Running multi-step git workflows (init → commit → push) or setting up deployment configs.
**Input to provide:** Project root path, GitHub username, repo name, visibility (public/private), Render service name.
**Agent must:** Execute all git commands, create the GitHub repo, push the code, create `render.yaml`, push it, print manual Render dashboard instructions for the user; return ONLY the GitHub repo URL, confirmation, and the printed instructions.
**Returns:** GitHub repo URL, commit hash, push confirmation, manual Render deploy steps printed for user, any errors.
**NEVER use:** `render` CLI, `render login`, `render deploy`, Render API keys, or any automated Render deployment — Render requires a browser and must be completed manually by the user.

---

## ☁️ Cloud Deploy Agent
**Trigger:** Step 14 — Generic Cloud Deployment
**Delegate when:** Provisioning cloud infrastructure and deploying the containerised API to any cloud platform (Render, AWS, GCP, Azure, Fly.io, Railway, etc.).
**Input to provide:** Target platform name, Docker image name, GitHub repo URL, project name, required env vars, desired region, instance/tier preference (free/standard).
**Agent must:** Generate the platform-specific config file(s), create all required cloud resources (container registry push if needed, service/task definition, load balancer, secrets), deploy the service, run smoke tests against the live URL; return ONLY the live URL, config file paths, and smoke-test outputs.
**Returns:** Live service URL, config files created, resource names provisioned, smoke-test results (`/health` + `/predict`), any warnings or quota notes.

---

## Step 11 — Initialise Git & Push to GitHub
Perform this step after the workspace is organised (Step 10).

### 11a — Check & Install GitHub CLI
```bash
gh --version        # check if installed
brew install gh     # install if missing (macOS)
gh auth status      # check login
gh auth login       # login if not authenticated (opens browser OAuth)
```

### 11b — Create `.gitignore`
Create `.gitignore` at the project root. Include:
- Python: `__pycache__/`, `*.pyc`, `.venv/`, `venv/`
- macOS: `.DS_Store`
- IDE: `.vscode/`, `.idea/`
- Secrets: `.env`, `*.env.*`
- Logs: `*.log`

### 11c — Initialise Git & First Commit
```bash
git init
git add .
git commit -m "Initial commit: end-to-end ML pipeline with FastAPI"
```

### 11d — Create GitHub Repo & Push
Read `github_username`, `github_repo`, and `github_visibility` from `.ml_config.json`, then run:
```bash
gh repo create <github_repo> \\
  --<github_visibility> \\
  --description "<brief description>" \\
  --source=. \\
  --remote=origin \\
  --push
```
Confirm the repo is live at `https://github.com/<github_username>/<github_repo>`.

> **Note:** Never hardcode the username or repo name. Always read both from `.ml_config.json`. If `.ml_config.json` is missing or the fields are empty, ask the user: "What is your GitHub username?" and "What should the repo be named?" before running any `gh` command.

---

## Step 12 — Create Dockerfile & Push to GitHub
Perform this step after the GitHub repo exists (Step 11).

### 12a — Create `Dockerfile`
Use a **multi-stage build** with `python:3.11-slim` as the base image:
- **Stage 1 (builder):** Copy `requirements.txt`, run `pip install --prefix=/install`
- **Stage 2 (runtime):** Copy installed packages from builder; copy `app.py` and `models/`; create a non-root `appuser`; expose port `8000`; set CMD with JSON array form

### 12b — Create `.dockerignore`
Exclude from the Docker build context:
- `.git/`, `__pycache__/`, `.venv/`
- `data/`, `plots/` (not needed at runtime)
- `tests/`, `docs/`, `*.md`
- `.env`, `.DS_Store`, `.claude/`

### 12c — Build & Test Locally
```bash
# Build
docker build -t <image-name>:latest .

# Run
docker run -d -p 8000:8000 --name <container-name> <image-name>:latest

# Smoke test
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d \'<use feature values from your dataset>\'

# Stop & remove
docker stop <container-name> && docker rm <container-name>
```

### 12d — Create `docker_guide.md`
Document the following in `docs/docker_guide.md`:
- Build command
- Run command
- All test-it-live curl examples (health, single predict, batch predict, Swagger UI)
- Post-deploy test commands (replace localhost with live URL)
- Useful Docker commands reference table
- Image details table

### 12e — Push to GitHub
```bash
git add Dockerfile .dockerignore docs/docker_guide.md
git commit -m "Add Dockerfile, .dockerignore, and Docker guide"
git push origin main
```
'''

# ════════════════════════════════════════════════════════════════════
FILES["deploy/cloud.md"] = '''# deploy/cloud.md — Cloud Deployment Index

Imported by @deploy/CLAUDE.md.

- Step 13 — Render Deployment: @deploy/cloud-render.md
- Step 14 — Generic Cloud Platforms: @deploy/cloud-platforms.md
'''

# ════════════════════════════════════════════════════════════════════
FILES["deploy/cloud-render.md"] = '''# deploy/cloud-render.md — Step 13: Render Deployment

Imported by @deploy/cloud.md.

---

## Step 13 — Deploy on Render

> ⚠️ **AGENT RULE — READ FIRST:**
> - **DO NOT** use `render` CLI, `render login`, `render deploy`, or any Render CLI commands.
> - **DO NOT** ask for a Render API key or attempt API-based deployment.
> - **DO NOT** try to automate the Render dashboard — it requires a browser.
> - Your job is: (1) create `render.yaml`, (2) push it to GitHub, (3) print the manual steps for the user.
> - Stop after step 13e. The user will complete the browser steps themselves.

Perform this step after the Dockerfile is pushed to GitHub (Step 12).

### 13a — Create `render.yaml`
Create `render.yaml` at the project root:
```yaml
services:
  - type: web
    name: <project-name>
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn app:app --host 0.0.0.0 --port $PORT
    healthCheckPath: /health
    envVars:
      - key: PYTHON_VERSION
        value: "3.11.0"
```

### 13b — Push render.yaml to GitHub
```bash
git add render.yaml
git commit -m "Add render.yaml for Render deployment"
git push origin main
```

### 13c — Print manual deploy instructions for the user
Print this message to the user **exactly** (replace `<username>` and `<project-name>` from `.ml_config.json`):

```
✅ render.yaml is ready and pushed to GitHub.

To go live on Render (free, ~2 minutes):
  1. Visit https://render.com → sign in with GitHub
  2. Click New + → Web Service
  3. Connect repo: <username>/<project-name>
  4. Render auto-detects render.yaml → click Create Web Service
  5. Your API will be live at: https://<project-name>.onrender.com

Once live, test it:
  curl https://<project-name>.onrender.com/health
```

### 13d — Create `deployment_guide.md`
Document the following in `docs/deployment_guide.md`:
- Prerequisites (files needed before deploying)
- 5-step Render deploy walkthrough with exact settings
- All API endpoint descriptions
- Test-it-live curl commands (health, predict, batch)
- Run locally instructions
- Input field reference table
- Free tier cold-start note

### 13e — Push deployment guide to GitHub
```bash
git add docs/deployment_guide.md
git commit -m "Add deployment guide"
git push origin main
```

**Stop here.** The user will open their browser to complete the Render dashboard steps.
'''

# ════════════════════════════════════════════════════════════════════
FILES["deploy/cloud-platforms.md"] = '''# deploy/cloud-platforms.md — Step 14: Generic Cloud Deployment

Imported by @deploy/cloud.md.

---

## Step 14 — Deploy to Any Cloud Platform (via Cloud Deploy Agent)
Perform this step after the Dockerfile and GitHub repo exist (Steps 11–12).

### 14a — Platform Selection

| Platform | Best For | Free Tier | Config File |
|---|---|---|---|
| **Render** | Simplest deploy from GitHub | ✅ Yes | `render.yaml` |
| **Fly.io** | Global edge, fast cold starts | ✅ Yes | `fly.toml` |
| **Railway** | One-click GitHub deploy | ✅ Yes | `railway.toml` |
| **AWS ECS (Fargate)** | Production, auto-scaling | ❌ Paid | `task-definition.json` |
| **AWS App Runner** | Easiest managed AWS container | ✅ Free tier | `apprunner.yaml` |
| **GCP Cloud Run** | Serverless containers, pay-per-use | ✅ Free tier | `cloudrun.yaml` |
| **Azure Container Apps** | Serverless containers on Azure | ✅ Free tier | `containerapp.yaml` |

### 14b — Prerequisites Checklist
```
✅ app.py              — FastAPI app at project root
✅ Dockerfile          — multi-stage build at project root
✅ requirements.txt    — pinned library versions
✅ models/             — final_pipeline.pkl + label_encoder.pkl
✅ .dockerignore       — excludes data/, plots/, tests/, docs/
✅ GitHub repo         — code pushed and up to date
✅ Docker image built  — verified locally with smoke tests
```

### 14c — Platform-Specific Deploy Commands

#### 🚁 Fly.io
```bash
brew install flyctl && fly auth login
fly launch --name <project-name> --region lax --no-deploy
# Edit fly.toml: set internal_port = 8000
fly deploy
curl https://<project-name>.fly.dev/health
```

#### 🚂 Railway
```bash
npm install -g @railway/cli && railway login
railway init && railway up
railway variables set PORT=8000
railway open
```

#### 🟠 AWS App Runner
```bash
brew install awscli && aws configure
aws ecr create-repository --repository-name <project-name>
aws ecr get-login-password | docker login --username AWS \\
  --password-stdin <account-id>.dkr.ecr.<region>.amazonaws.com
docker tag <image-name>:latest \\
  <account-id>.dkr.ecr.<region>.amazonaws.com/<project-name>:latest
docker push <account-id>.dkr.ecr.<region>.amazonaws.com/<project-name>:latest
aws apprunner create-service --cli-input-json file://apprunner.yaml
```

#### 🔵 GCP Cloud Run
```bash
brew install google-cloud-sdk && gcloud auth login
gcloud config set project <project-id>
gcloud services enable run.googleapis.com containerregistry.googleapis.com
gcloud builds submit --tag gcr.io/<project-id>/<project-name>:latest
gcloud run deploy <project-name> \\
  --image gcr.io/<project-id>/<project-name>:latest \\
  --platform managed --region us-central1 \\
  --allow-unauthenticated --port 8000 --set-env-vars PORT=8000
```

#### 🟦 Azure Container Apps
```bash
brew install azure-cli && az login
az group create --name <project-name>-rg --location eastus
az containerapp env create --name <project-name>-env \\
  --resource-group <project-name>-rg --location eastus
az acr create --resource-group <project-name>-rg \\
  --name <project-name>acr --sku Basic
az acr login --name <project-name>acr
docker tag <image-name>:latest <project-name>acr.azurecr.io/<project-name>:latest
docker push <project-name>acr.azurecr.io/<project-name>:latest
az containerapp create --name <project-name> \\
  --resource-group <project-name>-rg \\
  --environment <project-name>-env \\
  --image <project-name>acr.azurecr.io/<project-name>:latest \\
  --target-port 8000 --ingress external --env-vars PORT=8000
```

### 14d — Universal Smoke Tests
```bash
curl https://<LIVE_URL>/health
curl -X POST https://<LIVE_URL>/predict \\
  -H "Content-Type: application/json" \\
  -d \'<SAMPLE_PAYLOAD>\'
curl -X POST https://<LIVE_URL>/predict/batch \\
  -H "Content-Type: application/json" \\
  -d \'[<SAMPLE_PAYLOAD>, <SAMPLE_PAYLOAD_2>]\'
open https://<LIVE_URL>/docs
```

### 14e — Push to GitHub
```bash
git add docs/cloud_deployment_guide.md
git add .
git commit -m "Add cloud deployment config and guide for <platform>"
git push origin main
```
'''

# ════════════════════════════════════════════════════════════════════
FILES["docs/CLAUDE.md"] = '''# docs/CLAUDE.md — Documentation Agent Spec

## 📄 Documentation Agent
**Trigger:** Steps 8, 12d, 13d — Markdown documentation files
**Delegate when:** Writing `docs/summary.md`, `docs/testing_guide.md`, `docs/test_results.md`, `docs/deployment_guide.md`, `docs/docker_guide.md`.
**Input to provide:** The specific content to document (model results, test output, deployment steps, Docker commands).
**Agent must:** Write the complete `.md` file with proper sections, tables, and code blocks; return ONLY confirmation that the file was created and a one-line description of each section.
**Returns:** File path created, section headings list, confirmation.

---

## Post-Pipeline Steps 8 & 9
Full instructions for creating `summary.md` and `requirements.txt` are in **@src/CLAUDE.md**.
'''

# ════════════════════════════════════════════════════════════════════
FILES["tests/CLAUDE.md"] = '''# tests/CLAUDE.md — Testing Agent Spec

## 🧪 Testing Agent
**Trigger:** After pipeline or API is built
**Delegate when:** Writing `tests/test_pipeline.py`, running the full test suite, reporting results.
**Input to provide:** Pipeline path, label encoder path, data path, expected accuracy threshold.
**Agent must:** Write the test script (artifact integrity, single-sample predictions, full test-set evaluation, per-class accuracy, consistency check, probability check); run it; return ONLY the test summary output.
**Returns:** Pass/fail per test, overall accuracy, confirmation of 16/16 checks or list of failures.

---

## 🧪 Template Test Suite (Automated)

Tests every option in `docs/how_to_run.md` to catch regressions before users encounter them.

**How to run:**
```bash
cd tests/template_tests
./run_tests.sh --fast        # 34 checks, ~30 seconds (no pip install)
./run_tests.sh               # all checks including end-to-end (~10 min)
./run_tests.sh --suite 03    # just bootstrap.py tests
```

**Suites:**
| # | Name | Speed |
|---|---|---|
| 01 | Prerequisites & Template Integrity | ~3s |
| 02 | Template File Content Validation | ~2s |
| 03 | bootstrap.py Behaviour | ~15s |
| 04 | start.sh Shell Mode (end-to-end) | ~3–4 min |
| 05 | init.py Python CLI Mode (end-to-end) | ~3–4 min |
| 06 | Project Structure Deep Validation | ~15s |
'''

# ════════════════════════════════════════════════════════════════════
FILES["docs/claude_structure.md"] = '''# CLAUDE.md Split Structure

The root `CLAUDE.md` was split into a global file and local sub-directory files so that:
- No single file exceeds 150 lines
- The root file holds only what is **always** needed
- Local files are loaded only when that phase is active

---

## Final File Structure

| File | Lines | Contains |
|---|---|---|
| `CLAUDE.md` (root) | **65** | Role, condensed agent roster, rules, checklist, initialization |
| `src/CLAUDE.md` | **95** | EDA, Data Engineering, Optimization, FastAPI agents + Steps 3–10 |
| `tests/CLAUDE.md` | **8** | Testing Agent spec |
| `docs/CLAUDE.md` | **13** | Documentation Agent spec |
| `deploy/CLAUDE.md` | **120** | Docker, Git & Deploy, Cloud agents + Steps 11–12 |
| `deploy/cloud-render.md` | **58** | Step 13 — Render deployment |
| `deploy/cloud-platforms.md` | **148** | Step 14 — AWS, GCP, Azure, Fly.io, Railway |
| `deploy/cloud.md` | **6** | Index — `@`-imports cloud-render.md + cloud-platforms.md |

All files are ≤ 150 lines. ✅ No content was dropped — only reorganised.

---

## How It Works

- The root `CLAUDE.md` is **always loaded** (65 lines — very lean)
- Local files are only read when Claude navigates to that subdirectory or a sub-agent is triggered for that phase
- The `@`-import pointers in the roster table (`@src/CLAUDE.md`, `@deploy/CLAUDE.md`, etc.) tell Claude exactly where to look when a specific step is needed
- Token usage stays low because full step instructions only enter context when that agent/phase is actually active

---

## Agent → File Mapping

| Agent | Triggered By | Local File |
|---|---|---|
| 🔬 EDA Agent | Step 2 — EDA | `src/CLAUDE.md` |
| ⚙️ Data Engineering Agent | Step 3 — Preprocessing | `src/CLAUDE.md` |
| 🏆 Optimization Agent | Steps 4–6 — Training | `src/CLAUDE.md` |
| 🌐 FastAPI Agent | API development | `src/CLAUDE.md` |
| 🐳 Docker Agent | Step 12 — Docker | `deploy/CLAUDE.md` |
| 📄 Documentation Agent | Steps 8, docs | `docs/CLAUDE.md` |
| 🧪 Testing Agent | After pipeline | `tests/CLAUDE.md` |
| 🚀 Git & Deploy Agent | Steps 11–13 | `deploy/CLAUDE.md` |
| ☁️ Cloud Deploy Agent | Step 14 — Cloud | `deploy/CLAUDE.md` → `deploy/cloud.md` |
'''

# ════════════════════════════════════════════════════════════════════
FILES[".gitignore"] = '''# Python
__pycache__/
*.py[cod]
*.pyo
*.pyd
*.egg-info/
dist/
build/
.eggs/
*.egg

# Virtual environments
.venv/
venv/
env/
ENV/

# Jupyter
.ipynb_checkpoints/

# macOS
.DS_Store
.AppleDouble

# IDE
.vscode/
.idea/
*.swp
*.swo

# Secrets
.env
*.env.*

# Logs
*.log
logs/

# ML artifacts — exclude raw split files but KEEP final pipeline for deployment
# final_pipeline.pkl and label_encoder.pkl must be committed so Render can load them
data/*.csv
models/X_train_raw.pkl
models/X_test_raw.pkl
models/preprocessor.pkl
models/*.npy
plots/*.png

# ML config (user-specific, auto-generated by start.sh / init.py)
.ml_config.json

# Template output folders (new projects created as siblings)
# *_[0-9]*/    ← uncomment to also ignore timestamped project siblings
'''

# ════════════════════════════════════════════════════════════════════
FILES[".ml_config.json.example"] = '''{
  "_comment": "Copy this to .ml_config.json and fill in your values. Auto-generated by start.sh or init.py.",
  "project_name": "my-ml-project",
  "dataset_filename": "my_data.csv",
  "dataset_path": "data/my_data.csv",
  "target_column": "auto-detect",
  "task_type": "auto-detect",
  "deployment_platform": "render",
  "github_username": "your-github-username",
  "github_repo": "my-ml-project",
  "github_visibility": "public",
  "github_url": "https://github.com/your-github-username/my-ml-project",
  "python_version": "3.11",
  "created_at": "2026-01-01T00:00:00Z",
  "venv_path": ".venv",
  "template_version": "1.0.0"
}
'''

# ════════════════════════════════════════════════════════════════════
FILES["requirements.txt"] = '''# ML Pipeline Template — auto-generated by the pipeline
# Generated: <date>
# Python <version> | random_state=42

# Core data manipulation
pandas==2.2.2
numpy==1.26.4
tabulate==0.9.0

# Machine learning
scikit-learn==1.8.0
joblib==1.5.3

# Visualisation
matplotlib==3.10.9
seaborn==0.13.2

# API
fastapi==0.136.3
uvicorn==0.41.0
'''

# ════════════════════════════════════════════════════════════════════
FILES["start.sh"] = r'''#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
#  ML Pipeline Template — Interactive Setup Script
#  Usage: ./start.sh
# ──────────────────────────────────────────────────────────────────
set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

echo ""
echo -e "${CYAN}${BOLD}╔══════════════════════════════════════════════════╗${RESET}"
echo -e "${CYAN}${BOLD}║        🤖  ML Pipeline Template  v1.0.0          ║${RESET}"
echo -e "${CYAN}${BOLD}║   End-to-End Machine Learning Automation         ║${RESET}"
echo -e "${CYAN}${BOLD}╚══════════════════════════════════════════════════╝${RESET}"
echo ""

echo -e "${BOLD}Checking prerequisites...${RESET}"
set +e

if ! command -v brew &>/dev/null; then
    echo -e "${YELLOW}⚠  Homebrew not found — installing...${RESET}"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    [[ -f "/opt/homebrew/bin/brew" ]] && eval "$(/opt/homebrew/bin/brew shellenv)"
else
    echo -e "  ${GREEN}✔ Homebrew${RESET}"
fi

if ! command -v npm &>/dev/null; then
    echo -e "${YELLOW}⚠  Node.js not found — installing via Homebrew...${RESET}"
    brew install node
else
    echo -e "  ${GREEN}✔ Node.js $(node --version)${RESET}"
fi

if ! command -v claude &>/dev/null; then
    echo -e "${YELLOW}⚠  Claude Code CLI not found — installing...${RESET}"
    if npm install -g @anthropic-ai/claude-code; then
        echo -e "  ${GREEN}✔ Claude Code CLI installed${RESET}"
    else
        echo -e "  ${RED}✗ Auto-install failed. Run: npm install -g @anthropic-ai/claude-code${RESET}"
    fi
else
    echo -e "  ${GREEN}✔ Claude Code CLI$(claude --version 2>/dev/null | head -1 | sed 's/^/ /')${RESET}"
fi

set -e
echo ""

echo -e "${BOLD}How would you like to run this template?${RESET}"
echo "  1) Shell script  — guided prompts here in the terminal"
echo "  2) Python CLI    — richer prompts via init.py"
echo "  3) Claude Code   — AI-driven, fully automated (recommended)"
echo ""
read -rp "Enter choice [1/2/3] (default: 3): " ENTRY_MODE
ENTRY_MODE="${ENTRY_MODE:-3}"

if [ "$ENTRY_MODE" = "2" ]; then
    echo -e "${GREEN}▶ Launching Python CLI...${RESET}"
    exec python3 "$(dirname "$0")/init.py"
fi

echo ""
echo -e "${BOLD}── Project Setup ────────────────────────────────────${RESET}"

read -rp "Project name (default: ml-project): " PROJECT_NAME
PROJECT_NAME="${PROJECT_NAME:-ml-project}"
PROJECT_NAME="${PROJECT_NAME// /-}"

echo ""
read -rp "Dataset CSV path (press Enter to copy manually later): " DATASET_PATH
DATASET_FILENAME=""
if [ -n "$DATASET_PATH" ]; then
    if [ -f "$DATASET_PATH" ]; then
        DATASET_FILENAME=$(basename "$DATASET_PATH")
        echo -e "  ${GREEN}✔ Found: $DATASET_FILENAME${RESET}"
    else
        echo -e "  ${YELLOW}⚠ File not found — copy to data/ later.${RESET}"
        DATASET_PATH=""
    fi
fi

echo ""
echo -e "${BOLD}Deployment platform:${RESET}"
echo "  1) Ask me later  2) Render  3) Fly.io  4) Railway"
echo "  5) AWS App Runner  6) GCP Cloud Run  7) Azure  8) Skip"
read -rp "Enter choice [1-8] (default: 1): " DEPLOY_CHOICE
DEPLOY_CHOICE="${DEPLOY_CHOICE:-1}"

case "$DEPLOY_CHOICE" in
  2) PLATFORM="render" ;; 3) PLATFORM="fly.io" ;; 4) PLATFORM="railway" ;;
  5) PLATFORM="aws" ;;    6) PLATFORM="gcp" ;;   7) PLATFORM="azure" ;;
  8) PLATFORM="none" ;;   *) PLATFORM="ask_later" ;;
esac

echo ""
echo -e "${BOLD}GitHub setup:${RESET}"
GH_DETECTED=$(gh api user --jq '.login' 2>/dev/null || echo "")
if [ -n "$GH_DETECTED" ]; then
    echo -e "  ${GREEN}✔ GitHub account detected: ${GH_DETECTED}${RESET}"
    read -rp "  GitHub username (Enter to use '${GH_DETECTED}'): " GH_USER
    GH_USER="${GH_USER:-$GH_DETECTED}"
else
    read -rp "  GitHub username (Enter to skip): " GH_USER
fi

if [ -n "$GH_USER" ]; then
    read -rp "  GitHub repo name (default: ${PROJECT_NAME}): " GH_REPO
    GH_REPO="${GH_REPO:-$PROJECT_NAME}"
    GH_REPO="${GH_REPO// /-}"
    read -rp "  Visibility [1=Public / 2=Private] (default: 1): " GH_CHOICE
    case "${GH_CHOICE:-1}" in 2) GH_VIS="private" ;; *) GH_VIS="public" ;; esac
else
    GH_REPO=""; GH_VIS="skip"
    echo -e "  ${YELLOW}⚠ No GitHub username — skipping.${RESET}"
fi

TEMPLATE_DIR="$(cd "$(dirname "$0")" && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROJECT_DIR="$(dirname "$TEMPLATE_DIR")/${PROJECT_NAME}_${TIMESTAMP}"
STAGING_DIR="$(dirname "$PROJECT_DIR")/.ml_staging_${PROJECT_NAME}_${TIMESTAMP}"
STAGING_DIR_SET=true

cleanup_staging() {
    if [ "${STAGING_DIR_SET:-false}" = "true" ] && [ -d "$STAGING_DIR" ]; then
        rm -rf "$STAGING_DIR" 2>/dev/null
    fi
}
trap cleanup_staging EXIT

mkdir -p "$STAGING_DIR"

echo ""
echo -e "${GREEN}▶ Preparing project files...${RESET}"
rsync -a \
    --exclude='.git/' --exclude='data/*.csv' --exclude='models/*.pkl' \
    --exclude='models/*.npy' --exclude='plots/*.png' --exclude='.venv/' \
    --exclude='__pycache__/' --exclude='.DS_Store' --exclude='.ml_config.json' \
    --exclude='bootstrap.py' --exclude='Dockerfile.bootstrap' \
    --exclude='start.sh' --exclude='init.py' \
    "$TEMPLATE_DIR/" "$STAGING_DIR/"
echo -e "  ${GREEN}✔ Template files ready${RESET}"

if [ -n "$DATASET_PATH" ] && [ -f "$DATASET_PATH" ]; then
    cp "$DATASET_PATH" "$STAGING_DIR/data/"
    echo -e "  ${GREEN}✔ Dataset staged: $DATASET_FILENAME${RESET}"
fi

DATASET_FILENAME_SAFE="${DATASET_FILENAME:-<not provided yet>}"
PY_VER=$(python3 --version 2>&1 | awk '{print $2}')
CREATED_AT=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

cat > "$STAGING_DIR/.ml_config.json" << CONFIGEOF
{
  "project_name": "${PROJECT_NAME}",
  "dataset_filename": "${DATASET_FILENAME_SAFE}",
  "dataset_path": "data/${DATASET_FILENAME_SAFE}",
  "target_column": "auto-detect",
  "task_type": "auto-detect",
  "deployment_platform": "${PLATFORM}",
  "github_username": "${GH_USER}",
  "github_repo": "${GH_REPO:-$PROJECT_NAME}",
  "github_visibility": "${GH_VIS}",
  "github_url": "https://github.com/${GH_USER}/${GH_REPO:-$PROJECT_NAME}",
  "python_version": "${PY_VER}",
  "created_at": "${CREATED_AT}",
  "venv_path": ".venv",
  "template_version": "1.0.0"
}
CONFIGEOF
echo -e "  ${GREEN}✔ Config ready${RESET}"

echo -e "${GREEN}▶ Creating project at: $PROJECT_DIR${RESET}"
mv "$STAGING_DIR" "$PROJECT_DIR"
STAGING_DIR_SET=false

echo -e "${GREEN}▶ Creating Python virtual environment (.venv)...${RESET}"
rm -rf "$PROJECT_DIR/.venv"
python3 -m venv "$PROJECT_DIR/.venv"
echo -e "  ${GREEN}✔ Virtual environment created${RESET}"

echo -e "${GREEN}▶ Installing dependencies (this may take a minute)...${RESET}"
"$PROJECT_DIR/.venv/bin/pip" install --upgrade pip -q
"$PROJECT_DIR/.venv/bin/pip" install -r "$PROJECT_DIR/requirements.txt" -q
echo -e "  ${GREEN}✔ Dependencies installed${RESET}"

echo ""
echo -e "${CYAN}${BOLD}╔══════════════════════════════════════════════════╗${RESET}"
echo -e "${CYAN}${BOLD}║  ✅  Project ready! Launching Claude Code...     ║${RESET}"
echo -e "${CYAN}${BOLD}╚══════════════════════════════════════════════════╝${RESET}"
echo ""

cd "$PROJECT_DIR"
source ".venv/bin/activate"
if command -v claude &>/dev/null; then
    claude .
else
    echo -e "${YELLOW}Claude Code CLI not found. Install: npm install -g @anthropic-ai/claude-code${RESET}"
    echo -e "Then run: ${BOLD}cd $PROJECT_DIR && source .venv/bin/activate && claude .${RESET}"
fi
'''

# ════════════════════════════════════════════════════════════════════
FILES["init.py"] = r'''#!/usr/bin/env python3
"""
init.py — ML Pipeline Template: Python CLI Setup
Usage: python3 init.py
Requires: Python 3.9+ stdlib only (runs before venv is active)
"""

import os, sys, json, shutil, subprocess
from pathlib import Path
from datetime import datetime, timezone

G = "\033[0;32m"; Y = "\033[1;33m"; C = "\033[0;36m"
B = "\033[1m";    R = "\033[0;31m"; X = "\033[0m"

PLATFORMS = {
    "1": "ask_later", "2": "render",  "3": "fly.io",
    "4": "railway",   "5": "aws",     "6": "gcp",
    "7": "azure",     "8": "none",
}

PLATFORM_LABELS = {
    "ask_later": "Ask me later", "render": "Render (free tier)",
    "fly.io": "Fly.io", "railway": "Railway", "aws": "AWS App Runner",
    "gcp": "GCP Cloud Run", "azure": "Azure Container Apps",
    "none": "Skip (local / Docker only)",
}

def prompt(msg, default=""):
    suffix = f" (default: {default})" if default else ""
    val = input(f"{msg}{suffix}: ").strip()
    return val or default

def menu(title, options, default="1"):
    print(f"\n{B}{title}{X}")
    for key, label in options:
        print(f"  {key}) {label}")
    choice = input(f"Enter choice (default: {default}): ").strip()
    return choice or default

def banner():
    print(f"\n{C}{B}╔══════════════════════════════════════════════════╗\n║        🤖  ML Pipeline Template  v1.0.0          ║\n║   End-to-End Machine Learning Automation         ║\n╚══════════════════════════════════════════════════╝{X}\n")

def collect_inputs():
    print(f"\n{B}── Project Setup ──────────────────────────────────────{X}")
    project_name = prompt("Project name", "ml-project").replace(" ", "-")
    dataset_path = ""
    dataset_filename = ""
    raw = prompt("Dataset CSV path (press Enter to provide manually later)", "")
    if raw:
        p = Path(raw).expanduser().resolve()
        if p.is_file():
            dataset_path = str(p); dataset_filename = p.name
            print(f"  {G}✔ Found: {dataset_filename}{X}")
        else:
            print(f"  {Y}⚠ File not found — copy to data/ later.{X}")
    deploy_choice = menu("Deployment platform:", [
        ("1","Ask me later"),("2","Render (free tier, recommended)"),
        ("3","Fly.io"),("4","Railway"),("5","AWS App Runner"),
        ("6","GCP Cloud Run"),("7","Azure Container Apps"),("8","Skip"),
    ], default="1")
    platform = PLATFORMS.get(deploy_choice, "ask_later")
    print(f"\n{B}GitHub setup:{X}")
    gh_detected = ""
    try:
        result = subprocess.run(["gh","api","user","--jq",".login"],capture_output=True,text=True,timeout=5)
        if result.returncode == 0: gh_detected = result.stdout.strip()
    except Exception: pass
    if gh_detected:
        print(f"  {G}✔ GitHub account detected: {gh_detected}{X}")
        gh_user = prompt(f"  GitHub username (Enter to use \'{gh_detected}\')", gh_detected)
    else:
        gh_user = prompt("  GitHub username (Enter to skip)", "")
    gh_repo = ""; gh_vis = "skip"
    if gh_user:
        gh_repo = prompt("  GitHub repo name", project_name).replace(" ", "-")
        gh_vis = "private" if menu("  Visibility:",[("1","Public"),("2","Private")],default="1") == "2" else "public"
    else:
        print(f"  {Y}⚠ No GitHub username — skipping.{X}")
    return {"project_name": project_name, "dataset_path": dataset_path,
            "dataset_filename": dataset_filename, "platform": platform,
            "github_username": gh_user, "github_repo": gh_repo or project_name,
            "github_visibility": gh_vis}

def _make_staging_dir(project_dir):
    staging = project_dir.parent / f".ml_staging_{project_dir.name}"
    staging.mkdir(parents=True, exist_ok=True)
    return staging

def create_project(cfg):
    template_dir = Path(__file__).parent.resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _pname = cfg["project_name"]
    project_dir = template_dir.parent / f"{_pname}_{timestamp}"
    staging_dir = _make_staging_dir(project_dir)
    return project_dir, staging_dir

EXCLUDE = {".git","__pycache__",".venv",".DS_Store",".ml_config.json",
           "bootstrap.py","Dockerfile.bootstrap","start.sh","init.py"}
EXCLUDE_EXTS = {".csv",".pkl",".npy",".png",".pyc"}

def _ignore(src, names):
    ignored = set()
    for name in names:
        full = Path(src) / name
        if name in EXCLUDE or full.suffix in EXCLUDE_EXTS:
            ignored.add(name)
    return ignored

def copy_template(template_dir, staging_dir):
    print(f"{G}▶ Preparing template files...{X}")
    shutil.copytree(str(template_dir), str(staging_dir), ignore=_ignore, dirs_exist_ok=True)
    print(f"  {G}✔ Template files ready{X}")

def copy_dataset(cfg, staging_dir):
    if cfg["dataset_path"] and Path(cfg["dataset_path"]).is_file():
        dest = staging_dir / "data"; dest.mkdir(exist_ok=True)
        shutil.copy2(cfg["dataset_path"], dest / cfg["dataset_filename"])
        _dfn = cfg["dataset_filename"]
        print(f"  {G}✔ Dataset copied: {_dfn}{X}")

def write_config(cfg, staging_dir):
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    gh_user = cfg.get("github_username",""); gh_repo = cfg.get("github_repo", cfg["project_name"])
    config = {"project_name": cfg["project_name"],
              "dataset_filename": cfg["dataset_filename"] or "<not provided yet>",
              "dataset_path": ("data/" + cfg["dataset_filename"]) if cfg["dataset_filename"] else "<not provided yet>",
              "target_column": "auto-detect", "task_type": "auto-detect",
              "deployment_platform": cfg["platform"], "github_username": gh_user,
              "github_repo": gh_repo, "github_visibility": cfg["github_visibility"],
              "github_url": f"https://github.com/{gh_user}/{gh_repo}" if gh_user else "",
              "python_version": py_ver,
              "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "venv_path": ".venv", "template_version": "1.0.0"}
    (staging_dir / ".ml_config.json").write_text(json.dumps(config, indent=2))
    print(f"  {G}✔ .ml_config.json ready{X}")

def move_to_final(staging_dir, project_dir):
    print(f"\n{G}▶ Creating project at: {project_dir}{X}")
    shutil.move(str(staging_dir), str(project_dir))

def show_summary(cfg, project_dir):
    fn = cfg["dataset_filename"] or "<not provided yet>"
    plat = PLATFORM_LABELS.get(cfg["platform"], cfg["platform"])
    gh_user = cfg.get("github_username",""); gh_repo = cfg.get("github_repo", cfg["project_name"])
    gh_line = f"\n{C}{B}║{X}  🐙  GitHub : github.com/{gh_user}/{gh_repo}" if gh_user else ""
    print(f"\n{C}{B}╔══════════════════════════════════════════════════╗\n║  ✅  Project ready!                              ║\n╠══════════════════════════════════════════════════╣{X}\n{C}{B}║{X}  📁  {project_dir}\n{C}{B}║{X}  🐍  Venv   : .venv/\n{C}{B}║{X}  📊  Data   : {fn}\n{C}{B}║{X}  🚀  Deploy : {plat}{gh_line}\n{C}{B}╠══════════════════════════════════════════════════╣{X}\n{C}{B}║{X}  ✅  Launching Claude Code...\n{C}{B}╚══════════════════════════════════════════════════╝{X}\n")

def maybe_open_claude(project_dir):
    print(f"{G}▶ Launching Claude Code in your new project...{X}")
    os.chdir(project_dir)
    if shutil.which("claude"): subprocess.run(["claude","."])
    else: print(f"{Y}Claude Code CLI not found. Install: npm install -g @anthropic-ai/claude-code{X}\nThen run: {B}cd {project_dir} && source .venv/bin/activate && claude .{X}")

if __name__ == "__main__":
    banner()
    choice = menu("How would you like to run this template?",[
        ("1","Shell script — guided prompts here in the terminal"),
        ("2","Python CLI   — richer prompts via init.py  ← you are here"),
        ("3","Claude Code  — AI-driven, fully automated (recommended)"),
    ], default="3")
    if choice == "1":
        script = Path(__file__).parent / "start.sh"
        os.execv("/bin/bash",["/bin/bash",str(script)])
    cfg = collect_inputs()
    project_dir, staging_dir = create_project(cfg)
    copy_template(Path(__file__).parent.resolve(), staging_dir)
    copy_dataset(cfg, staging_dir)
    write_config(cfg, staging_dir)
    move_to_final(staging_dir, project_dir)
    print(f"{G}▶ Creating Python virtual environment (.venv)...{X}")
    shutil.rmtree(str(project_dir/".venv"), ignore_errors=True)
    subprocess.run([sys.executable,"-m","venv",str(project_dir/".venv")],check=True)
    print(f"  {G}✔ Virtual environment created{X}")
    print(f"{G}▶ Installing dependencies (this may take a minute)...{X}")
    pip = str(project_dir/".venv"/"bin"/"pip")
    subprocess.run([pip,"install","--upgrade","pip","-q"],check=True)
    req = project_dir/"requirements.txt"
    if req.exists(): subprocess.run([pip,"install","-r",str(req),"-q"],check=True)
    print(f"  {G}✔ Dependencies installed{X}")
    show_summary(cfg, project_dir)
    maybe_open_claude(project_dir)
'''

# ════════════════════════════════════════════════════════════════════
FILES["README.md"] = '''# 🤖 ML Pipeline Template

> An autonomous, end-to-end machine learning template powered by Claude Code.
> Bring your CSV — the AI builds the pipeline, API, Docker image, and deploys it.

📖 **Full usage guide:** [docs/how_to_run.md](docs/how_to_run.md)

---

## What This Template Does

- 🔍 **Auto-detects** task type (classification vs regression) from your data
- 🧹 **Preprocesses** data: missing values, encoding, scaling
- 🏆 **Trains & tunes** models with GridSearchCV (multiple candidates)
- 📊 **Evaluates** with classification report / RMSE + R²
- 🌐 **Wraps** the model in a FastAPI REST API (`/predict`, `/predict/batch`)
- 🐳 **Containerises** with a multi-stage Docker image
- 🚀 **Deploys** to your chosen cloud platform
- 📄 **Documents** everything in `docs/`

---

## Step 1 — Get the Template

### 🔥 Bootstrap (no git, no clone required)
```bash
curl -O https://raw.githubusercontent.com/ramleo/builds_bootstrap/main/bootstrap.py
python3 bootstrap.py
cd builds_bootstrap
```

### 📦 Git Clone
```bash
git clone https://github.com/ramleo/builds_bootstrap
cd builds_bootstrap
```

---

## Step 2 — Run It

```bash
./start.sh
```

Choose **3** (Claude Code, recommended) or press Enter.

> 📖 Full details: [docs/how_to_run.md](docs/how_to_run.md)

---

## Supported Deployment Platforms

| Platform | Free Tier | Config File |
|---|---|---|
| Render | ✅ | `render.yaml` |
| Fly.io | ✅ | `fly.toml` |
| Railway | ✅ | `railway.toml` |
| AWS App Runner | ✅ | `apprunner.yaml` |
| GCP Cloud Run | ✅ | — |
| Azure Container Apps | ✅ | — |

---

## License

MIT — free to use, modify, and distribute.
'''

# ════════════════════════════════════════════════════════════════════
FILES[".claude/settings.local.json"] = '''{
  "permissions": {
    "allow": [
      "Bash(.venv/bin/pip install *)",
      "Bash(.venv/bin/pip install --upgrade *)",
      "Bash(.venv/bin/pip show *)",
      "Bash(.venv/bin/pip list *)",
      "Bash(.venv/bin/python *)",
      "Bash(.venv/bin/uvicorn *)",
      "Bash(python3 *)",
      "Bash(pip3 install *)",
      "Bash(pip install *)",
      "Bash(pkill -f uvicorn*)",
      "Bash(curl *)",
      "Bash(git init)",
      "Bash(git add *)",
      "Bash(git commit *)",
      "Bash(git push *)",
      "Bash(git remote *)",
      "Bash(git status)",
      "Bash(git log *)",
      "Bash(git branch *)",
      "Bash(gh repo create *)",
      "Bash(gh auth status)",
      "Bash(gh api *)",
      "Bash(docker build *)",
      "Bash(docker run *)",
      "Bash(docker stop *)",
      "Bash(docker rm *)",
      "Bash(docker images *)",
      "Bash(docker ps *)",
      "Bash(mkdir -p *)",
      "Bash(mv *)",
      "Bash(cp *)",
      "Bash(rm -rf __pycache__)",
      "Bash(find . -name __pycache__ *)",
      "Bash(find . -name *.pyc *)",
      "Bash(chmod +x *)"
    ]
  }
}
'''

# ════════════════════════════════════════════════════════════════════
FILES["docs/how_to_run.md"] = '''# How to Run the ML Pipeline Template

## Step 1 — Check Python
```bash
python3 --version
```

## Step 2 — Download the bootstrap script
```bash
curl -O https://raw.githubusercontent.com/ramleo/builds_bootstrap/main/bootstrap.py
```

## Step 3 — Run the bootstrap
```bash
python3 bootstrap.py
```

## Step 4 — Enter the folder
```bash
cd builds_bootstrap
```

## Step 5 — Start the wizard
```bash
./start.sh
```

Choose **3** (Claude Code, default) and answer the prompts.

## Troubleshooting

| Problem | Fix |
|---|---|
| `python3: command not found` | Install Python 3.9+ from python.org |
| `claude: command not found` | Run `npm install -g @anthropic-ai/claude-code` |
| `Permission denied: ./start.sh` | Run `chmod +x start.sh` first |
| Dataset not found | Copy your `.csv` into `data/` and tell Claude the filename |
'''

# ════════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════
FILES["auto_pipeline.py"] = r'''
#!/usr/bin/env python3
"""
auto_pipeline.py — Automated ML Pipeline (no Claude/AI required)
Usage: python3 auto_pipeline.py
       .venv/bin/python auto_pipeline.py
"""

import json
import os
import shutil as _shutil
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ── ANSI colour codes ────────────────────────────────────────────────
G = "\033[0;32m"   # green
C = "\033[0;36m"   # cyan
B = "\033[1m"      # bold
Y = "\033[1;33m"   # yellow
R = "\033[0;31m"   # red
M = "\033[0;35m"   # magenta
X = "\033[0m"      # reset

# Save ANSI codes under stable names — X is later overwritten by the feature
# matrix (sklearn convention: X = df.drop(...)), so any code that runs AFTER
# that point must use these aliases instead of the bare single-letter names.
_G = G; _C = C; _B = B; _Y = Y; _R = R; _X = X


# ════════════════════════════════════════════════════════════════════
# 0.  Paths & Config
# ════════════════════════════════════════════════════════════════════

ROOT = Path(__file__).parent.resolve()
os.chdir(ROOT)   # ensure all relative paths resolve from the project root

CONFIG_PATH = ROOT / ".ml_config.json"
DATA_DIR    = ROOT / "data"
MODELS_DIR  = ROOT / "models"
PLOTS_DIR   = ROOT / "plots"
DOCS_DIR    = ROOT / "docs"

for d in (DATA_DIR, MODELS_DIR, PLOTS_DIR, DOCS_DIR):
    d.mkdir(parents=True, exist_ok=True)


def _print_header(text: str) -> None:
    width = 60
    print(f"\n{_C}{_B}{'═' * width}{_X}")
    print(f"{_C}{_B}  {text}{_X}")
    print(f"{_C}{_B}{'═' * width}{_X}")


def _ok(msg: str) -> None:
    print(f"  {_G}✔  {msg}{_X}")


def _warn(msg: str) -> None:
    print(f"  {_Y}⚠  {msg}{_X}")


def _err(msg: str) -> None:
    print(f"  {_R}✗  {msg}{_X}")


def _info(msg: str) -> None:
    print(f"  {_C}→  {msg}{_X}")


# ════════════════════════════════════════════════════════════════════
# 1.  Load Config
# ════════════════════════════════════════════════════════════════════

_print_header("Step 1 — Loading Configuration")

if not CONFIG_PATH.exists():
    _err(f".ml_config.json not found at {CONFIG_PATH}")
    _info("Creating a minimal config — edit it and re-run.")
    # Ask for the minimum needed
    dataset_input = input(f"  {B}Dataset CSV path (or filename inside data/): {X}").strip()
    target_input = input(f"  {B}Target column name (press Enter to auto-detect): {X}").strip()
    cfg = {
        "dataset_path": dataset_input,
        "target_column": target_input or None,
        "deployment_platform": "none",
        "github_username": "",
        "github_repo": "",
        "github_visibility": "public",
    }
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    _ok(f".ml_config.json written → {CONFIG_PATH}")
else:
    with open(CONFIG_PATH) as fh:
        cfg = json.load(fh)
    _ok(f"Config loaded from {CONFIG_PATH}")

# Resolve dataset path
dataset_path_raw = cfg.get("dataset_path", "")
target_col_cfg = cfg.get("target_column") or None
platform = cfg.get("deployment_platform", "none")

# Try to find the CSV
csv_path = None
if dataset_path_raw:
    p = Path(dataset_path_raw).expanduser()
    if p.is_absolute() and p.is_file():
        csv_path = p
    else:
        # Try relative to ROOT, then data/
        for candidate in (ROOT / p, ROOT / "data" / p.name, ROOT / "data" / p):
            if Path(candidate).is_file():
                csv_path = Path(candidate)
                break

if csv_path is None:
    # Scan data/ for any CSV
    csvs = list(DATA_DIR.glob("*.csv"))
    if csvs:
        csv_path = csvs[0]
        _warn(f"dataset_path not found; using auto-discovered: {csv_path.name}")

if csv_path is None:
    _err("No CSV dataset found.")
    _info(f"Copy your dataset into: {DATA_DIR}/")
    _info("Then re-run: .venv/bin/python auto_pipeline.py")
    sys.exit(1)

_ok(f"Dataset: {csv_path}")

# ════════════════════════════════════════════════════════════════════
# 2.  Import heavy deps (after path check so errors are clear)
# ════════════════════════════════════════════════════════════════════

try:
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import joblib
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import (
        accuracy_score, classification_report,
        mean_squared_error, r2_score,
    )
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import (
        LabelEncoder, OneHotEncoder, RobustScaler, StandardScaler,
    )
except ImportError as exc:
    _err(f"Missing dependency: {exc}")
    _info("Run: .venv/bin/pip install -r requirements.txt")
    sys.exit(1)

_ok("All dependencies imported")

# ════════════════════════════════════════════════════════════════════
# 3.  Load Data
# ════════════════════════════════════════════════════════════════════

_print_header("Step 2 — EDA & Data Inspection")

try:
    df = pd.read_csv(csv_path)
except Exception as exc:
    _err(f"Could not read CSV: {exc}")
    sys.exit(1)

_ok(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ── Resolve target column ────────────────────────────────────────────
if target_col_cfg and target_col_cfg not in ("auto-detect", "") and target_col_cfg in df.columns:
    target_col = target_col_cfg
    _ok(f"Target column (from config): {_B}{target_col}{_X}")
else:
    # Ask the user — never silently guess on a real dataset
    print(f"\n{_B}Available columns:{_X}")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2}) {col}")
    print()
    while True:
        try:
            raw = input(f"Which column is the target variable? "
                        f"(name or number, default: {df.columns[-1]}): ").strip()
        except EOFError:
            raw = ""
        if raw == "":
            target_col = df.columns[-1]
            _warn(f"No input — using last column as target: {_B}{target_col}{_X}")
            break
        # Accept column number
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(df.columns):
                target_col = df.columns[idx]
                break
            else:
                _err(f"Number out of range (1–{len(df.columns)}), try again.")
                continue
        # Accept column name (case-insensitive)
        matches = [c for c in df.columns if c.lower() == raw.lower()]
        if matches:
            target_col = matches[0]
            break
        _err(f"'{raw}' not found in columns, try again.")
    # Persist the chosen column back to .ml_config.json for future runs
    try:
        cfg["target_column"] = target_col
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    except Exception:
        pass
    _ok(f"Target column set to: {_B}{target_col}{_X}")

# ── Auto-detect task type ────────────────────────────────────────────
n_unique = df[target_col].nunique()
if df[target_col].dtype in (object, bool, "bool") or n_unique <= 20:
    task_type = "classification"
else:
    task_type = "regression"
_ok(f"Task type: {_B}{task_type}{_X}  (unique target values: {n_unique})")

# ── Basic EDA printout ───────────────────────────────────────────────
print(f"\n{_B}Column dtypes:{_X}")
print(df.dtypes.to_string())

missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(1)
missing_df = pd.DataFrame({"missing": missing, "pct": missing_pct})
missing_with = missing_df[missing_df["missing"] > 0]
if not missing_with.empty:
    print(f"\n{_B}Missing values:{_X}")
    print(missing_with.to_string())
else:
    _ok("No missing values")

if task_type == "classification":
    print(f"\n{_B}Class balance ({target_col}):{_X}")
    vc = df[target_col].value_counts()
    print(vc.to_string())
else:
    print(f"\n{_B}Target distribution ({target_col}):{_X}")
    print(df[target_col].describe().to_string())

# ── EDA plots ────────────────────────────────────────────────────────
_info("Saving EDA plots...")

# Correlation heatmap (numeric cols only)
try:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 2:
        fig, ax = plt.subplots(figsize=(max(6, len(numeric_cols)), max(5, len(numeric_cols) - 1)))
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
                    linewidths=0.5, square=True)
        ax.set_title("Feature Correlation Heatmap", fontsize=14, pad=12)
        plt.tight_layout()
        heatmap_path = PLOTS_DIR / "eda_correlation.png"
        fig.savefig(heatmap_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        _ok(f"Saved: {heatmap_path}")
    else:
        _warn("Not enough numeric columns for a correlation heatmap")
        heatmap_path = None
except Exception as exc:
    _warn(f"Correlation heatmap failed: {exc}")
    heatmap_path = None

# Target distribution plot
try:
    fig, ax = plt.subplots(figsize=(8, 5))
    if task_type == "classification":
        vc = df[target_col].value_counts()
        ax.bar(vc.index.astype(str), vc.values, color="#4C8CBF", edgecolor="white")
        ax.set_xlabel(target_col)
        ax.set_ylabel("Count")
        ax.set_title(f"Target Distribution — {target_col}", fontsize=13)
    else:
        ax.hist(df[target_col].dropna(), bins=40, color="#4C8CBF", edgecolor="white")
        ax.set_xlabel(target_col)
        ax.set_ylabel("Frequency")
        ax.set_title(f"Target Distribution — {target_col}", fontsize=13)
    plt.tight_layout()
    target_plot_path = PLOTS_DIR / "eda_target.png"
    fig.savefig(target_plot_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    _ok(f"Saved: {target_plot_path}")
except Exception as exc:
    _warn(f"Target distribution plot failed: {exc}")
    target_plot_path = None

# ════════════════════════════════════════════════════════════════════
# 4.  Preprocessing
# ════════════════════════════════════════════════════════════════════

_print_header("Step 3 — Preprocessing")

# Separate features / target
X = df.drop(columns=[target_col]).copy()
y = df[target_col].copy()

# Drop columns with >50% missing
drop_thresh = 0.5
high_missing = [c for c in X.columns if X[c].isnull().mean() > drop_thresh]
if high_missing:
    _warn(f"Dropping columns with >50% missing: {high_missing}")
    X = X.drop(columns=high_missing)

# Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

_ok(f"Numeric features  ({len(numeric_features)}): {numeric_features}")
_ok(f"Categorical features ({len(categorical_features)}): {categorical_features}")

# ── Encode target ────────────────────────────────────────────────────
label_encoder = None
if task_type == "classification":
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y.astype(str))
    _ok(f"Target classes: {list(label_encoder.classes_)}")
else:
    y_enc = y.values.astype(float)

# ── Train / test split ───────────────────────────────────────────────
split_kwargs = {"test_size": 0.2, "random_state": 42}
if task_type == "classification":
    split_kwargs["stratify"] = y_enc

X_train, X_test, y_train, y_test = train_test_split(X, y_enc, **split_kwargs)
_ok(f"Train: {X_train.shape}  Test: {X_test.shape}")

# ── Build ColumnTransformer ──────────────────────────────────────────
scaler = StandardScaler() if task_type == "classification" else RobustScaler()

transformers = []
if numeric_features:
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", scaler),
    ])
    transformers.append(("num", numeric_transformer, numeric_features))

if categorical_features:
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    transformers.append(("cat", categorical_transformer, categorical_features))

if not transformers:
    _err("No numeric or categorical features found after preprocessing.")
    sys.exit(1)

ct = ColumnTransformer(transformers=transformers, remainder="drop")
_ok("ColumnTransformer built")

# ════════════════════════════════════════════════════════════════════
# 5.  Model Training & Hyperparameter Search
# ════════════════════════════════════════════════════════════════════

_print_header("Step 4 — Model Training & Hyperparameter Search")

if task_type == "classification":
    candidates = [
        (
            "LogisticRegression",
            LogisticRegression(solver="saga", random_state=42, max_iter=1000),
            {"model__C": [0.1, 1, 10]},
        ),
        (
            "RandomForest",
            RandomForestClassifier(random_state=42),
            {"model__n_estimators": [100, 200], "model__max_depth": [None, 5]},
        ),
        (
            "GradientBoosting",
            GradientBoostingClassifier(random_state=42),
            {"model__n_estimators": [100, 200], "model__learning_rate": [0.05, 0.1]},
        ),
    ]
    scoring = "accuracy"
else:
    candidates = [
        (
            "Ridge",
            Ridge(),
            {"model__alpha": [0.1, 1.0, 10.0]},
        ),
        (
            "RandomForest",
            RandomForestRegressor(random_state=42),
            {"model__n_estimators": [100, 200], "model__max_depth": [None, 5]},
        ),
        (
            "GradientBoosting",
            GradientBoostingRegressor(random_state=42),
            {"model__n_estimators": [100, 200], "model__learning_rate": [0.05, 0.1]},
        ),
    ]
    scoring = "r2"

results = []

for name, estimator, param_grid in candidates:
    try:
        _info(f"Training {_B}{name}{_X} with GridSearchCV(cv=3)...")
        pipe = Pipeline([("preprocessor", ct), ("model", estimator)])
        gs = GridSearchCV(pipe, param_grid, cv=3, n_jobs=-1, scoring=scoring, refit=True)
        gs.fit(X_train, y_train)
        best_score = gs.best_score_
        best_params = {k.replace("model__", ""): v for k, v in gs.best_params_.items()}
        results.append({
            "name": name,
            "gs": gs,
            "cv_score": best_score,
            "best_params": best_params,
            "best_estimator": gs.best_estimator_,
        })
        _ok(f"{name}: CV {scoring} = {best_score:.4f}  params={best_params}")
    except Exception as exc:
        _err(f"{name} failed, skipping: {exc}")

if not results:
    _err("All models failed. Check your dataset.")
    sys.exit(1)

# ── Select best ──────────────────────────────────────────────────────
best_result = max(results, key=lambda r: r["cv_score"])
best_name = best_result["name"]
best_params = best_result["best_params"]
best_cv = best_result["cv_score"]

_print_header("Step 5 — Best Model & Final Evaluation")
print(f"\n  {_G}{_B}Best model: {best_name}{_X}")
print(f"  CV {scoring}: {best_cv:.4f}")
print(f"  Hyperparams: {best_params}")

# ── Refit best pipeline on full train set ────────────────────────────
_info("Refitting best pipeline on full training set...")
final_pipe = Pipeline([
    ("preprocessor", ColumnTransformer(transformers=transformers, remainder="drop")),
    ("model", best_result["best_estimator"].named_steps["model"]),
])
final_pipe.fit(X_train, y_train)
_ok("Final pipeline fitted")

# ── Evaluate on test set ─────────────────────────────────────────────
y_pred = final_pipe.predict(X_test)

metrics = {}

if task_type == "classification":
    acc = accuracy_score(y_test, y_pred)
    # Build label list restricted to classes that appear in y_test or y_pred,
    # so target_names length always matches the number of labels reported.
    if label_encoder is not None:
        present_labels = sorted(set(y_test) | set(y_pred))
        target_names_filtered = [str(label_encoder.classes_[i]) for i in present_labels]
    else:
        present_labels = None
        target_names_filtered = None
    report = classification_report(
        y_test, y_pred,
        labels=present_labels,
        target_names=target_names_filtered,
    )
    metrics["accuracy"] = acc
    metrics["classification_report"] = report
    print(f"\n  {_B}Test Accuracy: {_G}{acc:.4f}{_X}")
    print(f"\n{_B}Classification Report:{_X}")
    print(report)
else:
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))
    metrics["rmse"] = rmse
    metrics["r2"] = r2
    print(f"\n  {_B}Test RMSE : {_G}{rmse:.4f}{_X}")
    print(f"  {_B}Test R²   : {_G}{r2:.4f}{_X}")

# ════════════════════════════════════════════════════════════════════
# 6.  Save Artifacts
# ════════════════════════════════════════════════════════════════════

_print_header("Step 6 — Saving Artifacts")

pipeline_path = MODELS_DIR / "final_pipeline.pkl"
joblib.dump(final_pipe, pipeline_path)
_ok(f"Final pipeline → {pipeline_path}")

le_path = None
if label_encoder is not None:
    le_path = MODELS_DIR / "label_encoder.pkl"
    joblib.dump(label_encoder, le_path)
    _ok(f"Label encoder → {le_path}")

# ════════════════════════════════════════════════════════════════════
# 7.  Write docs/auto_summary.md
# ════════════════════════════════════════════════════════════════════

_print_header("Step 7 — Writing Summary Report")

from datetime import datetime


def _df_to_md(df: "pd.DataFrame") -> str:
    """Format a DataFrame as a Markdown table without requiring tabulate."""
    idx_name = df.index.name or "column"
    cols = [idx_name] + list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep    = "| " + " | ".join("---" for _ in cols) + " |"
    rows   = [
        "| " + " | ".join([str(df.index[i])] + [str(v) for v in df.iloc[i]]) + " |"
        for i in range(len(df))
    ]
    return "\n".join([header, sep] + rows)

now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

if task_type == "classification":
    metrics_section = f"""
| Metric | Value |
|---|---|
| Test Accuracy | {metrics['accuracy']:.4f} |
| CV Score (accuracy) | {best_cv:.4f} |

**Classification Report:**
```
{metrics['classification_report']}
```
""".strip()
else:
    metrics_section = f"""
| Metric | Value |
|---|---|
| Test RMSE | {metrics['rmse']:.4f} |
| Test R² | {metrics['r2']:.4f} |
| CV Score (r2) | {best_cv:.4f} |
""".strip()

candidates_table_rows = "\n".join(
    f"| {r['name']} | {r['cv_score']:.4f} | {r['best_params']} |"
    for r in sorted(results, key=lambda r: r["cv_score"], reverse=True)
)

artifact_rows = f"| models/final_pipeline.pkl | Trained sklearn Pipeline (preprocessor + model) |\n"
if le_path:
    artifact_rows += f"| models/label_encoder.pkl | Fitted LabelEncoder for target classes |\n"
if heatmap_path:
    artifact_rows += f"| plots/eda_correlation.png | Feature correlation heatmap |\n"
if target_plot_path:
    artifact_rows += f"| plots/eda_target.png | Target variable distribution |\n"

summary_md = f"""# ML Pipeline Summary

_Generated by auto_pipeline.py on {now_str}_

---

## 1. Dataset Overview

| Property | Value |
|---|---|
| File | {csv_path.name} |
| Rows | {df.shape[0]:,} |
| Columns | {df.shape[1]} |
| Target Column | `{target_col}` |
| Task Type | **{task_type.title()}** |
| Unique Target Values | {n_unique} |
| Deployment Platform | {platform} |

### Missing Values
{"No missing values detected." if missing_with.empty else _df_to_md(missing_with)}

---

## 2. Features Used

**Numeric ({len(numeric_features)}):** {", ".join(f"`{c}`" for c in numeric_features) or "_none_"}

**Categorical ({len(categorical_features)}):** {", ".join(f"`{c}`" for c in categorical_features) or "_none_"}

**Dropped (>50% missing):** {", ".join(f"`{c}`" for c in high_missing) or "_none_"}

---

## 3. Preprocessing Pipeline

| Step | Detail |
|---|---|
| Missing (numeric) | SimpleImputer(strategy="median") |
| Missing (categorical) | SimpleImputer(strategy="most_frequent") |
| Scaling | {"StandardScaler" if task_type == "classification" else "RobustScaler"} |
| Encoding | OneHotEncoder(handle_unknown="ignore") |
| Train/Test Split | 80/20, {"stratified, " if task_type == "classification" else ""}random_state=42 |
| Train size | {X_train.shape[0]:,} rows |
| Test size | {X_test.shape[0]:,} rows |

---

## 4. Model Selection & Hyperparameter Tuning

GridSearchCV(cv=3, n_jobs=-1, scoring="{scoring}")

| Model | CV Score | Best Params |
|---|---|---|
{candidates_table_rows}

**Winner:** `{best_name}` with CV {scoring} = **{best_cv:.4f}**

---

## 5. Model Evaluation

{metrics_section}

---

## 6. Final Pipeline Architecture

```
CSV Input
    └── ColumnTransformer
            ├── numeric  → SimpleImputer(median) → {"StandardScaler" if task_type == "classification" else "RobustScaler"}
            └── categorical → SimpleImputer(most_frequent) → OneHotEncoder
                └── {best_name}({", ".join(f"{k}={v}" for k, v in best_params.items())})
                        └── Prediction
```

---

## 7. Artifacts

| File | Description |
|---|---|
{artifact_rows}

---

## 8. Reproducibility

```python
import joblib, pandas as pd

pipeline = joblib.load("models/final_pipeline.pkl")
df = pd.read_csv("data/{csv_path.name}")
X = df.drop(columns=["{target_col}"])
predictions = pipeline.predict(X)
print(predictions)
```
"""

summary_path = DOCS_DIR / "auto_summary.md"
summary_path.write_text(summary_md, encoding="utf-8")
_ok(f"Summary report → {summary_path}")

# ════════════════════════════════════════════════════════════════════
# 8.  Final Terminal Summary
# ════════════════════════════════════════════════════════════════════

print(f"""
{_C}{_B}╔══════════════════════════════════════════════════════╗
║  ✅  Pipeline Complete!                              ║
╠══════════════════════════════════════════════════════╣{_X}
{_C}{_B}║{_X}  Dataset    : {csv_path.name} ({df.shape[0]:,} rows × {df.shape[1]} cols)
{_C}{_B}║{_X}  Task       : {task_type.title()}
{_C}{_B}║{_X}  Best Model : {best_name}
{_C}{_B}║{_X}  CV Score   : {best_cv:.4f}  ({scoring})""")

if task_type == "classification":
    print(f"{_C}{_B}║{_X}  Accuracy   : {metrics['accuracy']:.4f}")
else:
    print(f"{_C}{_B}║{_X}  RMSE       : {metrics['rmse']:.4f}")
    print(f"{_C}{_B}║{_X}  R²         : {metrics['r2']:.4f}")

print(f"""{_C}{_B}╠══════════════════════════════════════════════════════╣{_X}
{_C}{_B}║{_X}  {_G}models/final_pipeline.pkl{_X}   ← ready to use
{_C}{_B}║{_X}  {_G}docs/auto_summary.md{_X}        ← full report
{_C}{_B}║{_X}  {_G}plots/eda_correlation.png{_X}   ← correlation heatmap
{_C}{_B}║{_X}  {_G}plots/eda_target.png{_X}        ← target distribution
{_C}{_B}╚══════════════════════════════════════════════════════╝{_X}
""")

# ════════════════════════════════════════════════════════════════════
# 9.  Post-Pipeline: App · Docker · GitHub · Render
# ════════════════════════════════════════════════════════════════════

def _get_features(pipeline):
    """Extract numeric and categorical feature names from the fitted pipeline."""
    pre = pipeline.named_steps.get("preprocessor")
    if pre is None:
        return [], []
    num_feats, cat_feats = [], []
    for tname, _, cols in pre.transformers_:
        if tname == "num":
            num_feats = list(cols)
        elif tname == "cat":
            cat_feats = list(cols)
    return num_feats, cat_feats



def _fetch_unsplash_photo(search_query, fallback_url):
    """
    Return a photo URL for the given search query.

    Calls the Unsplash Search API when UNSPLASH_ACCESS_KEY is set in the
    environment (Unsplash free tier: 50 req / hr, takes ~2 min to register).
    Always picks the top result so the same domain always gets the same photo
    (deterministic).  Falls back silently to fallback_url on any error or when
    the key is absent — zero friction for users who haven't set the key.
    """
    import urllib.request, urllib.parse as _urlparse
    key = os.environ.get("UNSPLASH_ACCESS_KEY", "").strip()
    if not key:
        return fallback_url
    try:
        q = _urlparse.quote_plus(search_query)
        req = urllib.request.Request(
            f"https://api.unsplash.com/search/photos?query={q}&per_page=1&orientation=landscape",
            headers={"Authorization": f"Client-ID {key}", "Accept-Version": "v1"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        results = data.get("results", [])
        if results:
            raw = results[0]["urls"]["raw"]
            return raw + "&w=1400&h=560&fit=crop&auto=format&q=85"
    except Exception:
        pass
    return fallback_url


def _detect_domain(dataset_filename, column_names, project_name=""):
    """
    Dynamically detect dataset domain from filename, column names, and project name.
    Returns a theme dict with colors, gradient, header_pattern, name, and description.

    header_pattern is a CSS background value: a semi-transparent colour overlay
    (theme-tinted, for text readability) layered over a specific, curated
    Unsplash photo chosen to match the domain.  Falls back to a pure-CSS mesh
    gradient for completely unrecognised datasets.

    Scores each known domain by counting keyword matches in the combined text.
    Falls back to a smart generic theme when nothing matches.
    """
    def _norm(s):
        return str(s).lower().replace("_", " ").replace("-", " ")

    search = " ".join(
        [_norm(dataset_filename), _norm(project_name)]
        + [_norm(c) for c in column_names]
    )

    # Curated Unsplash photo IDs (verified 200 OK).
    # Each header_pattern = colour-tinted overlay gradient  +  photo URL.
    # Overlay opacity ~0.65 keeps the photo visible while ensuring white
    # text stays legible at any screen size.
    _UP = "https://images.unsplash.com"
    _Q  = "?w=1400&h=560&fit=crop&auto=format&q=85"

    domains = [
        # ── Botanical ────────────────────────────────────────────────────────
        (
            ["iris", "sepal", "petal", "setosa", "versicolor", "virginica",
             "flower", "species", "botanical", "petal length", "petal width"],
            {"name": "Botanical",
             "primary": "#2e7d32", "accent": "#9c27b0",
             "btn": "#7b1fa2", "btn_hover": "#4a1942", "body_bg": "#f3e5f5",
             "gradient": "linear-gradient(135deg,#2e7d32 0%,#4a148c 100%)",
             "header_pattern": (
                 f"linear-gradient(135deg,rgba(46,125,50,.65) 0%,rgba(74,20,140,.70) 100%),"
                 f"url('{_UP}/photo-1462275646964-a0e3386b89fa{_Q}') center/cover no-repeat"
             ),
                          "search_q": "botanical wildflowers flowers nature",
"desc": "AI-powered botanical species classification"},
        ),
        # ── Health ───────────────────────────────────────────────────────────
        (
            ["glucose", "insulin", "bmi", "blood", "diabetes", "cancer", "heart",
             "cholesterol", "medical", "health", "patient", "clinical", "disease",
             "pregnancies", "hemoglobin", "thyroid", "tumor", "pulse", "pressure"],
            {"name": "Health",
             "primary": "#0d6e6e", "accent": "#00897b",
             "btn": "#00897b", "btn_hover": "#00695c", "body_bg": "#f0faf9",
             "gradient": "linear-gradient(135deg,#0d6e6e 0%,#004d40 100%)",
             "header_pattern": (
                 f"linear-gradient(135deg,rgba(13,110,110,.62) 0%,rgba(0,77,64,.68) 100%),"
                 f"url('{_UP}/photo-1576091160399-112ba8d25d1d{_Q}') center/cover no-repeat"
             ),
                          "search_q": "hospital medical healthcare clinical",
"desc": "AI-powered health risk assessment"},
        ),
        # ── Aviation ─────────────────────────────────────────────────────────
        (
            ["flight", "airline", "delay", "airport", "departure",
             "arrival", "route", "travel", "boarding"],
            {"name": "Aviation",
             "primary": "#1565c0", "accent": "#1976d2",
             "btn": "#1976d2", "btn_hover": "#1565c0", "body_bg": "#e8f4fc",
             "gradient": "linear-gradient(135deg,#1565c0 0%,#0d47a1 100%)",
             "header_pattern": (
                 f"linear-gradient(135deg,rgba(13,71,161,.60) 0%,rgba(21,101,192,.55) 100%),"
                 f"url('{_UP}/photo-1436491865332-7a61a109cc05{_Q}') center/cover no-repeat"
             ),
                          "search_q": "airplane flight sky clouds",
"desc": "AI-powered flight prediction"},
        ),
        # ── Finance ──────────────────────────────────────────────────────────
        (
            ["loan", "credit", "fraud", "income", "bank", "salary", "payment",
             "default", "financial", "mortgage", "debt", "interest", "stock",
             "revenue", "profit", "market"],
            {"name": "Finance",
             "primary": "#1a237e", "accent": "#ffa000",
             "btn": "#ffa000", "btn_hover": "#f57f17", "body_bg": "#eef0fc",
             "gradient": "linear-gradient(135deg,#1a237e 0%,#283593 100%)",
             "header_pattern": (
                 f"linear-gradient(135deg,rgba(26,35,126,.68) 0%,rgba(40,53,147,.62) 100%),"
                 f"url('{_UP}/photo-1611974789855-9c2a0a7236a3{_Q}') center/cover no-repeat"
             ),
                          "search_q": "stock market finance trading charts",
"desc": "AI-powered financial prediction"},
        ),
        # ── Shipping ─────────────────────────────────────────────────────────
        (
            ["ship", "cargo", "freight", "delivery", "container", "port",
             "logistics", "shipping", "vessel", "warehouse", "supplier"],
            {"name": "Shipping",
             "primary": "#1b4f72", "accent": "#2e86c1",
             "btn": "#2e86c1", "btn_hover": "#1b4f72", "body_bg": "#d6eaf8",
             "gradient": "linear-gradient(135deg,#1b4f72 0%,#154360 100%)",
             "header_pattern": (
                 f"linear-gradient(135deg,rgba(27,79,114,.65) 0%,rgba(21,67,96,.70) 100%),"
                 f"url('{_UP}/photo-1494412651409-8963ce7935a7{_Q}') center/cover no-repeat"
             ),
                          "search_q": "cargo ship port container logistics",
"desc": "AI-powered logistics prediction"},
        ),
        # ── Real Estate ──────────────────────────────────────────────────────
        (
            ["house", "sqft", "bedroom", "bathroom", "property", "rent",
             "real estate", "floor", "garage", "neighborhood", "zip",
             "price", "lot", "dwelling"],
            {"name": "Real Estate",
             "primary": "#5d4037", "accent": "#e53935",
             "btn": "#e53935", "btn_hover": "#c62828", "body_bg": "#fbe9e7",
             "gradient": "linear-gradient(135deg,#5d4037 0%,#3e2723 100%)",
             "header_pattern": (
                 f"linear-gradient(135deg,rgba(93,64,55,.62) 0%,rgba(62,39,35,.68) 100%),"
                 f"url('{_UP}/photo-1568605114967-8130f3a36994{_Q}') center/cover no-repeat"
             ),
                          "search_q": "house property real estate architecture",
"desc": "AI-powered property prediction"},
        ),
        # ── HR Analytics ─────────────────────────────────────────────────────
        (
            ["employee", "attrition", "department", "hire", "churn",
             "satisfaction", "performance", "job", "tenure", "workforce",
             "promotion", "manager"],
            {"name": "HR Analytics",
             "primary": "#4a148c", "accent": "#8e24aa",
             "btn": "#8e24aa", "btn_hover": "#6a1b9a", "body_bg": "#f5edf9",
             "gradient": "linear-gradient(135deg,#4a148c 0%,#311b92 100%)",
             "header_pattern": (
                 f"linear-gradient(135deg,rgba(74,20,140,.60) 0%,rgba(49,27,146,.65) 100%),"
                 f"url('{_UP}/photo-1522202176988-66273c2fd55f{_Q}') center/cover no-repeat"
             ),
                          "search_q": "team office workplace people collaboration",
"desc": "AI-powered people analytics"},
        ),
        # ── Quality / Wine ────────────────────────────────────────────────────
        (
            ["wine", "quality", "alcohol", "acidity", "sugar", "flavor",
             "volatile", "sulphates", "density", "residual"],
            {"name": "Quality",
             "primary": "#880e4f", "accent": "#ad1457",
             "btn": "#ad1457", "btn_hover": "#880e4f", "body_bg": "#fce4ec",
             "gradient": "linear-gradient(135deg,#880e4f 0%,#4a148c 100%)",
             "header_pattern": (
                 f"linear-gradient(135deg,rgba(136,14,79,.65) 0%,rgba(74,20,140,.68) 100%),"
                 f"url('{_UP}/photo-1510812431401-41d2bd2722f3{_Q}') center/cover no-repeat"
             ),
                          "search_q": "wine glasses winery vineyard",
"desc": "AI-powered quality assessment"},
        ),
        # ── Maritime ─────────────────────────────────────────────────────────
        (
            ["titanic", "survived", "pclass", "embarked", "lifeboat", "survival"],
            {"name": "Maritime",
             "primary": "#0d47a1", "accent": "#1565c0",
             "btn": "#1565c0", "btn_hover": "#0d47a1", "body_bg": "#e3f2fd",
             "gradient": "linear-gradient(135deg,#0d47a1 0%,#01579b 100%)",
             "header_pattern": (
                 f"linear-gradient(135deg,rgba(13,71,161,.58) 0%,rgba(1,87,155,.65) 100%),"
                 f"url('{_UP}/photo-1507525428034-b723cf961d3e{_Q}') center/cover no-repeat"
             ),
                          "search_q": "ocean sea waves beach",
"desc": "AI-powered survival prediction"},
        ),
        # ── Retail ───────────────────────────────────────────────────────────
        (
            ["customer", "churn", "product", "purchase", "review", "rating",
             "sales", "retail", "cart", "discount", "conversion", "order"],
            {"name": "Retail",
             "primary": "#e65100", "accent": "#ef6c00",
             "btn": "#ef6c00", "btn_hover": "#e65100", "body_bg": "#fff3e0",
             "gradient": "linear-gradient(135deg,#e65100 0%,#bf360c 100%)",
             "header_pattern": (
                 f"linear-gradient(135deg,rgba(230,81,0,.62) 0%,rgba(191,54,12,.68) 100%),"
                 f"url('{_UP}/photo-1441986300917-64674bd600d8{_Q}') center/cover no-repeat"
             ),
                          "search_q": "shopping mall retail store",
"desc": "AI-powered customer prediction"},
        ),
        # ── Energy ───────────────────────────────────────────────────────────
        (
            ["energy", "power", "electricity", "solar", "wind", "temperature",
             "weather", "co2", "emission", "renewable", "consumption"],
            {"name": "Energy",
             "primary": "#e65100", "accent": "#f57f17",
             "btn": "#f57f17", "btn_hover": "#e65100", "body_bg": "#fff8e1",
             "gradient": "linear-gradient(135deg,#f9a825 0%,#e65100 100%)",
             "header_pattern": (
                 f"linear-gradient(135deg,rgba(249,168,37,.60) 0%,rgba(230,81,0,.68) 100%),"
                 f"url('{_UP}/photo-1509391366360-2e959784a276{_Q}') center/cover no-repeat"
             ),
                          "search_q": "solar panels renewable energy",
"desc": "AI-powered energy prediction"},
        ),
        # ── Automotive ───────────────────────────────────────────────────────
        (
            ["car", "vehicle", "mileage", "engine", "horsepower", "mpg",
             "cylinders", "transmission", "fuel", "auto"],
            {"name": "Automotive",
             "primary": "#37474f", "accent": "#546e7a",
             "btn": "#546e7a", "btn_hover": "#37474f", "body_bg": "#eceff1",
             "gradient": "linear-gradient(135deg,#37474f 0%,#263238 100%)",
             "header_pattern": (
                 f"linear-gradient(135deg,rgba(55,71,79,.62) 0%,rgba(38,50,56,.68) 100%),"
                 f"url('{_UP}/photo-1503376780353-7e6692767b70{_Q}') center/cover no-repeat"
             ),
                          "search_q": "car vehicle highway road",
"desc": "AI-powered vehicle prediction"},
        ),
        # ── Agriculture ──────────────────────────────────────────────────────
        (
            ["crop", "soil", "rainfall", "humidity", "nitrogen", "phosphorus",
             "potassium", "agriculture", "farm", "harvest", "yield"],
            {"name": "Agriculture",
             "primary": "#33691e", "accent": "#558b2f",
             "btn": "#558b2f", "btn_hover": "#33691e", "body_bg": "#f1f8e9",
             "gradient": "linear-gradient(135deg,#33691e 0%,#1b5e20 100%)",
             "header_pattern": (
                 f"linear-gradient(135deg,rgba(51,105,30,.60) 0%,rgba(27,94,32,.65) 100%),"
                 f"url('{_UP}/photo-1500382017468-9049fed747ef{_Q}') center/cover no-repeat"
             ),
                          "search_q": "wheat field farm harvest golden",
"desc": "AI-powered crop prediction"},
        ),
        # ── Cybersecurity ────────────────────────────────────────────────────
        (
            ["attack", "threat", "malware", "intrusion", "network", "packet",
             "protocol", "firewall", "vulnerability", "exploit", "cyber"],
            {"name": "Cybersecurity",
             "primary": "#1a1a2e", "accent": "#00e676",
             "btn": "#00c853", "btn_hover": "#1a1a2e", "body_bg": "#e8f5e9",
             "gradient": "linear-gradient(135deg,#1a1a2e 0%,#0d0d1a 100%)",
             "header_pattern": (
                 f"linear-gradient(135deg,rgba(26,26,46,.72) 0%,rgba(13,13,26,.78) 100%),"
                 f"url('{_UP}/photo-1550751827-4bd374c3f58b{_Q}') center/cover no-repeat"
             ),
                          "search_q": "cybersecurity digital network security code",
"desc": "AI-powered security threat detection"},
        ),
        # ── Education ────────────────────────────────────────────────────────
        (
            ["student", "grade", "score", "exam", "pass", "fail", "gpa",
             "attendance", "education", "school", "university", "course"],
            {"name": "Education",
             "primary": "#1565c0", "accent": "#42a5f5",
             "btn": "#1976d2", "btn_hover": "#0d47a1", "body_bg": "#e3f2fd",
             "gradient": "linear-gradient(135deg,#1565c0 0%,#0d47a1 100%)",
             "header_pattern": (
                 f"linear-gradient(135deg,rgba(21,101,192,.62) 0%,rgba(13,71,161,.68) 100%),"
                 f"url('{_UP}/photo-1562774053-701939374585{_Q}') center/cover no-repeat"
             ),
                          "search_q": "university campus students library",
"desc": "AI-powered academic performance prediction"},
        ),
        # ── Sports ───────────────────────────────────────────────────────────
        (
            ["game", "score", "player", "team", "win", "loss", "sport",
             "match", "goal", "point", "season", "league", "tournament"],
            {"name": "Sports",
             "primary": "#1b5e20", "accent": "#43a047",
             "btn": "#388e3c", "btn_hover": "#1b5e20", "body_bg": "#e8f5e9",
             "gradient": "linear-gradient(135deg,#1b5e20 0%,#004d00 100%)",
             "header_pattern": (
                 f"linear-gradient(135deg,rgba(27,94,32,.60) 0%,rgba(0,77,0,.65) 100%),"
                 f"url('{_UP}/photo-1461896836934-ffe607ba8211{_Q}') center/cover no-repeat"
             ),
                          "search_q": "stadium sports crowd arena",
"desc": "AI-powered sports performance prediction"},
        ),
        # ── Insurance ────────────────────────────────────────────────────────
        (
            ["insurance", "premium", "claim", "policy", "coverage", "deductible",
             "insured", "beneficiary", "underwrite", "copay", "annuity",
             "reinsurance", "indemnity", "actuary", "liability", "peril"],
            {"name": "Insurance",
             "primary": "#1a3c5e", "accent": "#2979ff",
             "btn": "#1565c0", "btn_hover": "#0d47a1", "body_bg": "#e8f0fe",
             "gradient": "linear-gradient(135deg,#1a3c5e 0%,#0d2137 100%)",
             "header_pattern": (
                 f"linear-gradient(135deg,rgba(26,60,94,.65) 0%,rgba(13,33,55,.70) 100%),"
                 f"url('{_UP}/photo-1560472354-b33ff0c44a43{_Q}') center/cover no-repeat"
             ),
                          "search_q": "protection safety shield security",
"desc": "AI-powered insurance risk prediction"},
        ),
        # ── Supply Chain ─────────────────────────────────────────────────────
        (
            ["supply", "inventory", "demand", "procurement", "vendor", "sku",
             "stock", "replenishment", "lead time", "warehouse", "distribution"],
            {"name": "Supply Chain",
             "primary": "#4e342e", "accent": "#ff7043",
             "btn": "#f4511e", "btn_hover": "#bf360c", "body_bg": "#fbe9e7",
             "gradient": "linear-gradient(135deg,#4e342e 0%,#3e2723 100%)",
             "header_pattern": (
                 f"linear-gradient(135deg,rgba(78,52,46,.65) 0%,rgba(62,39,35,.70) 100%),"
                 f"url('{_UP}/photo-1586528116311-ad8dd3c8310d{_Q}') center/cover no-repeat"
             ),
                          "search_q": "warehouse logistics inventory shelves",
"desc": "AI-powered supply chain prediction"},
        ),
        # ── NLP ──────────────────────────────────────────────────────────────
        (
            ["text", "sentiment", "review", "nlp", "tweet", "comment", "opinion",
             "positive", "negative", "corpus", "token", "language", "document"],
            {"name": "NLP",
             "primary": "#4527a0", "accent": "#7c4dff",
             "btn": "#651fff", "btn_hover": "#4527a0", "body_bg": "#ede7f6",
             "gradient": "linear-gradient(135deg,#4527a0 0%,#311b92 100%)",
             "header_pattern": (
                 f"linear-gradient(135deg,rgba(69,39,160,.68) 0%,rgba(49,27,146,.72) 100%),"
                 f"url('{_UP}/photo-1526374965328-7f61d4dc18c5{_Q}') center/cover no-repeat"
             ),
                          "search_q": "text code programming digital abstract",
"desc": "AI-powered text analysis"},
        ),
    ]

    best_theme, best_score = None, 0
    for keywords, theme in domains:
        score = sum(1 for kw in keywords if kw in search)
        if score > best_score:
            best_score = score
            best_theme = theme

    if best_theme is None or best_score == 0:
        # Generic fallback: derive a unique hue from the dataset's column names
        # so even novel datasets get a consistent, distinct visual identity.
        stop_words = {"id", "no", "num", "the", "and", "for", "with", "from",
                      "this", "that", "have", "data", "value", "label", "class",
                      "target", "output", "result", "flag", "code", "type"}
        raw_words = []
        for token in ([dataset_filename, project_name] + list(column_names)):
            for w in _norm(token).split():
                if len(w) > 3 and w not in stop_words:
                    raw_words.append(w)
        seed_str = " ".join(raw_words[:6]) or dataset_filename
        hue  = abs(hash(seed_str)) % 360
        hue2 = (hue + 30) % 360

        best_theme = {
            "name": "ML",
            "primary": "#1e3a5f", "accent": "#1a73e8",
            "btn": "#1a73e8", "btn_hover": "#1558d6", "body_bg": "#eef2ff",
            "gradient": "linear-gradient(135deg,#1e3a5f 0%,#0d2137 100%)",
            "header_pattern": (
                f"radial-gradient(ellipse at 80% 20%,hsla({hue},70%,55%,.42) 0%,transparent 50%),"
                f"radial-gradient(ellipse at 20% 80%,hsla({hue2},65%,40%,.38) 0%,transparent 45%),"
                "linear-gradient(135deg,#1e3a5f 0%,#0d2137 100%)"
            ),
            "desc": "AI-powered prediction",
        }

    return best_theme





def _generate_frontend(root, cfg, task_type, num_feats, cat_feats, label_encoder=None):
    """
    Generate index.html — a themed, responsive prediction UI.
    Theme is auto-detected from column names + dataset filename + project name.
    Uses placeholder substitution (not f-strings) so CSS braces need no escaping.
    """
    _print_header("Generating frontend (index.html)")

    dataset_filename = cfg.get("dataset_filename", "") or cfg.get("dataset_path", "")
    project_name     = cfg.get("project_name", "ML Project")

    classes = []
    if task_type == "classification" and label_encoder is not None:
        try:
            classes = [str(c) for c in label_encoder.classes_]
        except Exception:
            classes = []

    theme    = _detect_domain(dataset_filename, num_feats + cat_feats, project_name)
    title    = project_name.replace("-", " ").replace("_", " ").title()
    icon     = theme.get("icon", "")
    desc     = theme["desc"]
    primary  = theme["primary"]
    accent   = theme["accent"]
    btn      = theme["btn"]
    btn_hover = theme["btn_hover"]
    body_bg  = theme["body_bg"]
    gradient = theme["gradient"]

    # ── Pure-CSS header background ────────────────────────────────────────
    # Each theme carries its own `header_pattern` — layered mesh gradients
    # (radial colour spots + linear base) that look rich and unique with
    # zero external HTTP requests.  Falls back to the plain gradient if missing.
    header_bg = theme.get("header_pattern", gradient)

    # Optional: replace header photo dynamically via Unsplash Search API.
    # Set UNSPLASH_ACCESS_KEY env var once; pipeline auto-uses it every run.
    # Falls back silently to the hardcoded curated photo when key is absent.
    _sq = theme.get("search_q", "")
    if _sq and "images.unsplash.com" in header_bg:
        import re as _re
        _m = _re.search(r"url\('(https://images\.unsplash\.com/[^']+)'\)", header_bg)
        if _m:
            _dyn = _fetch_unsplash_photo(_sq, _m.group(1))
            if _dyn != _m.group(1):
                header_bg = header_bg.replace(_m.group(1), _dyn)
                _ok("Unsplash: dynamic photo fetched via API")

    _ok(f"Theme  : {theme['name']}  {icon}")
    _ok(f"Colors : primary={primary}  accent={accent}  body={body_bg}")

    is_class_js = "true" if task_type == "classification" else "false"
    classes_js  = str(classes).replace("'", '"')          # valid JS array literal

    # ── Build form fields (dark-glass inputs for Tailwind template) ───────────
    import re as _re

    def _fmt_label(name):
        """Split camelCase and snake_case into a readable title label."""
        s = _re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', name)
        s = _re.sub(r'([a-z\d])([A-Z])', r'\1 \2', s)
        return s.replace('_', ' ').replace('-', ' ').title()

    _ID_COLS = {'id', 'index', 'serial', 'rowid', 'row', 'no', 'num',
                'uuid', 'guid', 'pk', 'key', 'rid', 'sid'}

    def _is_id_col(name):
        n = _re.sub(r'[^a-z]', '', name.lower())
        return n in _ID_COLS or (n.endswith('id') and len(n) <= 5)

    fields_html = ""
    for feat in num_feats:
        if _is_id_col(feat):
            continue
        lbl = _fmt_label(feat)
        fields_html += (
            '\n          <div>'
            '\n            <label class="inp-lbl">' + lbl + '</label>'
            '\n            <input type="number" name="' + feat + '" class="inp"'
            ' placeholder="e.g. 0" step="any" min="0">'
            '\n          </div>'
        )
    for feat in cat_feats:
        if _is_id_col(feat):
            continue
        lbl = _fmt_label(feat)
        fields_html += (
            '\n          <div>'
            '\n            <label class="inp-lbl">' + lbl + '</label>'
            '\n            <input type="text" name="' + feat + '" class="inp"'
            ' placeholder="Enter value">'
            '\n          </div>'
        )

    # ── Derive mesh-background CSS from theme colours ─────────────────────────
    def _hex_rgb(h):
        h = h.lstrip("#")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    pr, pg, pb = _hex_rgb(primary)
    ar, ag, ab = _hex_rgb(accent)
    mesh_css = (
        f"radial-gradient(ellipse at 15% 55%,rgba({pr},{pg},{pb},.42) 0%,transparent 48%),"
        f"radial-gradient(ellipse at 85% 15%,rgba({ar},{ag},{ab},.36) 0%,transparent 45%),"
        f"radial-gradient(ellipse at 55% 90%,rgba({pr},{pg},{pb},.20) 0%,transparent 40%),"
        f"radial-gradient(ellipse at 90% 65%,rgba({ar},{ag},{ab},.15) 0%,transparent 38%),"
        f"linear-gradient(145deg,#060c06 0%,#0d0820 50%,#070f0f 100%)"
    )
    badge_bg     = f"rgba({pr},{pg},{pb},.18)"
    badge_border = f"rgba({pr},{pg},{pb},.40)"
    badge_text   = accent

    # ── About-strip values ────────────────────────────────────────────────────
    domain_name   = theme["name"]
    algo_str      = cfg.get("best_model", "—")
    _acc          = cfg.get("test_accuracy")
    accuracy_str  = (f"{_acc:.1%}" if _acc is not None else "—")
    acc_color     = "#4ade80" if (_acc or 0) >= 0.9 else ("#facc15" if (_acc or 0) >= 0.7 else "#f87171")
    algo_str      = cfg.get("best_model", "—")
    _acc          = cfg.get("test_accuracy")
    accuracy_str  = (f"{_acc:.1%}" if _acc is not None else "—")
    acc_color     = "#4ade80" if (_acc or 0) >= 0.9 else ("#facc15" if (_acc or 0) >= 0.7 else "#f87171")
    task_label    = "Classification" if task_type == "classification" else "Regression"
    feat_count    = str(len(num_feats) + len(cat_feats))
    classes_count = str(len(classes)) if task_type == "classification" else "&mdash;"

    # ── HTML template (TMPL_ placeholders, not f-strings → CSS {} safe) ─────────
    html = (
        '<!DOCTYPE html>\n'
        '<html lang="en">\n'
        '<head>\n'
        '  <meta charset="UTF-8">\n'
        '  <meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        '  <title>TMPL_TITLE</title>\n'
        '  <script src="https://cdn.tailwindcss.com"></script>\n'
        '  <link rel="preconnect" href="https://fonts.googleapis.com">\n'
        '  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">\n'
        '  <style>\n'
        '    * { font-family: \'Inter\', sans-serif; }\n'
        '    body { min-height: 100vh; background: #080f1a; overflow-x: hidden; }\n'
        '\n'
        '    .bg-mesh {\n'
        '      position: fixed; inset: 0; z-index: 0;\n'
        '      background: TMPL_MESH_CSS;\n'
        '      animation: meshAnim 14s ease-in-out infinite alternate;\n'
        '    }\n'
        '    @keyframes meshAnim {\n'
        '      0%   { filter: hue-rotate(0deg)   brightness(1);    }\n'
        '      50%  { filter: hue-rotate(18deg)  brightness(1.06); }\n'
        '      100% { filter: hue-rotate(-12deg) brightness(0.94); }\n'
        '    }\n'
        '\n'
        '    .hero {\n'
        '      background: TMPL_HEADER_BG;\n'
        '      background-size: cover;\n'
        '      background-position: center;\n'
        '    }\n'
        '\n'
        '    .glass {\n'
        '      background: rgba(255,255,255,.055);\n'
        '      backdrop-filter: blur(28px);\n'
        '      -webkit-backdrop-filter: blur(28px);\n'
        '      border: 1px solid rgba(255,255,255,.11);\n'
        '      box-shadow: 0 8px 48px rgba(0,0,0,.45), inset 0 1px 0 rgba(255,255,255,.09);\n'
        '    }\n'
        '\n'
        '    .inp {\n'
        '      width: 100%; padding: 11px 14px;\n'
        '      background: rgba(255,255,255,.07);\n'
        '      border: 1px solid rgba(255,255,255,.13);\n'
        '      border-radius: 10px; color: #fff;\n'
        '      font-size: .94rem; transition: all .2s; outline: none;\n'
        '    }\n'
        '    .inp::placeholder { color: rgba(255,255,255,.28); }\n'
        '    .inp:focus {\n'
        '      background: rgba(255,255,255,.11);\n'
        '      border-color: TMPL_ACCENT;\n'
        '      box-shadow: 0 0 0 3px TMPL_ACCENT_ALPHA;\n'
        '    }\n'
        '    .inp-lbl {\n'
        '      display: block; font-size: .68rem; font-weight: 700;\n'
        '      text-transform: uppercase; letter-spacing: .09em;\n'
        '      color: rgba(255,255,255,.45); margin-bottom: 6px;\n'
        '    }\n'
        '\n'
        '    .btn-predict {\n'
        '      width: 100%; padding: 15px;\n'
        '      background: linear-gradient(135deg, TMPL_BTN 0%, TMPL_ACCENT 100%);\n'
        '      border: none; border-radius: 12px; color: #fff;\n'
        '      font-size: 1rem; font-weight: 700; letter-spacing: .03em;\n'
        '      cursor: pointer; transition: all .22s; position: relative; overflow: hidden;\n'
        '    }\n'
        '    .btn-predict::after {\n'
        '      content: \'\'; position: absolute; inset: 0;\n'
        '      background: linear-gradient(135deg, rgba(255,255,255,.13), transparent);\n'
        '      pointer-events: none;\n'
        '    }\n'
        '    .btn-predict:hover:not(:disabled) {\n'
        '      transform: translateY(-2px);\n'
        '      box-shadow: 0 10px 28px rgba(0,0,0,.4);\n'
        '      background: linear-gradient(135deg, TMPL_BTN_HOVER 0%, TMPL_ACCENT 100%);\n'
        '    }\n'
        '    .btn-predict:active  { transform: translateY(0); }\n'
        '    .btn-predict:disabled { opacity: .55; cursor: not-allowed; }\n'
        '\n'
        '    .conf-track {\n'
        '      height: 7px; background: rgba(255,255,255,.09);\n'
        '      border-radius: 99px; overflow: hidden;\n'
        '    }\n'
        '    .conf-fill {\n'
        '      height: 100%; border-radius: 99px;\n'
        '      background: linear-gradient(90deg, TMPL_BTN, TMPL_ACCENT);\n'
        '      transition: width .9s cubic-bezier(.4,0,.2,1);\n'
        '    }\n'
        '\n'
        '    .mini-stat {\n'
        '      background: rgba(255,255,255,.045);\n'
        '      border: 1px solid rgba(255,255,255,.08);\n'
        '      border-radius: 10px; padding: 12px 10px; text-align: center;\n'
        '    }\n'
        '\n'
        '    .badge {\n'
        '      display: inline-flex; align-items: center; gap: 6px;\n'
        '      padding: 5px 14px;\n'
        '      background: TMPL_BADGE_BG;\n'
        '      border: 1px solid TMPL_BADGE_BORDER;\n'
        '      border-radius: 99px;\n'
        '      font-size: .72rem; font-weight: 700;\n'
        '      color: TMPL_BADGE_TEXT; text-transform: uppercase; letter-spacing: .07em;\n'
        '    }\n'
        '\n'
        '    .srv-dot {\n'
        '      width: 8px; height: 8px; border-radius: 50%;\n'
        '      background: #d1d5db; flex-shrink: 0; transition: background .3s;\n'
        '    }\n'
        '    .srv-dot.online   { background: #22c55e; box-shadow: 0 0 8px rgba(34,197,94,.6); animation: blink 2.2s infinite; }\n'
        '    .srv-dot.offline  { background: #ef4444; }\n'
        '    .srv-dot.checking { background: #f59e0b; animation: blink 1s infinite; }\n'
        '    @keyframes blink { 0%,100% { opacity:1; } 50% { opacity:.4; } }\n'
        '\n'
        '    @keyframes fadeUp {\n'
        '      from { opacity:0; transform:translateY(14px); }\n'
        '      to   { opacity:1; transform:translateY(0);    }\n'
        '    }\n'
        '    .fade-up { animation: fadeUp .45s ease forwards; }\n'
        '\n'
        '    .spinner {\n'
        '      display: inline-block; width: 16px; height: 16px;\n'
        '      border: 2px solid rgba(255,255,255,.3);\n'
        '      border-top-color: #fff; border-radius: 50%;\n'
        '      animation: spin .7s linear infinite;\n'
        '      vertical-align: middle; margin-right: 6px;\n'
        '    }\n'
        '    @keyframes spin { to { transform: rotate(360deg); } }\n'
        '\n'
        '    .err-box {\n'
        '      background: rgba(220,38,38,.15); border: 1px solid rgba(220,38,38,.35);\n'
        '      border-radius: 10px; padding: 12px 14px;\n'
        '      color: #fca5a5; font-size: .875rem; white-space: pre-wrap;\n'
        '    }\n'
        '\n'
        '    ::-webkit-scrollbar { width: 5px; }\n'
        '    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,.18); border-radius: 3px; }\n'
        '\n'
        '    @media (max-width: 768px) {\n'
        '      #mainGrid { grid-template-columns: 1fr !important; }\n'
        '      .hero-h1  { font-size: 2rem !important; }\n'
        '    }\n'
        '  </style>\n'
        '</head>\n'
        '<body>\n'
        '\n'
        '  <div class="bg-mesh"></div>\n'
        '\n'
        '  <div style="position:relative;z-index:10;min-height:100vh">\n'
        '\n'
        '    <!-- Hero -->\n'
        '    <div class="hero">\n'
        '      <div style="max-width:1024px;margin:0 auto;padding:64px 24px;text-align:center;color:#fff">\n'
        '        <div style="display:inline-flex;align-items:center;gap:8px;padding:6px 16px;border-radius:99px;background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.18);backdrop-filter:blur(8px);font-size:.85rem;font-weight:600;margin-bottom:20px">\n'
        '          <span style="width:8px;height:8px;border-radius:50%;background:#4ade80;display:inline-block"></span>\n'
        '          TMPL_DOMAIN_NAME\n'
        '        </div>\n'
        '        <h1 class="hero-h1" style="font-size:2.8rem;font-weight:900;letter-spacing:-0.5px;margin-bottom:14px;text-shadow:0 2px 24px rgba(0,0,0,.55)">\n'
        '          TMPL_TITLE\n'
        '        </h1>\n'
        '        <p style="color:rgba(255,255,255,.75);font-size:1.05rem;max-width:480px;margin:0 auto;text-shadow:0 1px 8px rgba(0,0,0,.4)">\n'
        '          TMPL_DESC\n'
        '        </p>\n'
        '      </div>\n'
        '    </div>\n'
        '\n'
        '    <!-- Main content -->\n'
        '    <div style="max-width:1024px;margin:-32px auto 0;padding:0 16px 64px">\n'
        '\n'
        '      <!-- Server status -->\n'
        '      <div class="glass" style="display:inline-flex;align-items:center;gap:10px;padding:10px 18px;border-radius:14px;margin-bottom:20px">\n'
        '        <span class="srv-dot checking" id="srvDot"></span>\n'
        '        <span style="color:rgba(255,255,255,.65);font-size:.83rem;font-weight:600" id="srvTxt">Checking server…</span>\n'
        '      </div>\n'
        '\n'
        '      <!-- Two-column grid -->\n'
        '      <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px" id="mainGrid">\n'
        '\n'
        '        <!-- LEFT: Input Form -->\n'
        '        <div class="glass" style="border-radius:20px;padding:28px">\n'
        '          <div style="display:flex;align-items:center;gap:12px;margin-bottom:24px">\n'
        '            <div style="width:40px;height:40px;border-radius:12px;background:linear-gradient(135deg,TMPL_BTN,TMPL_ACCENT);display:flex;align-items:center;justify-content:center;font-size:1.2rem;flex-shrink:0;box-shadow:0 4px 12px rgba(0,0,0,.3)">⚗</div>\n'
        '            <div>\n'
        '              <div style="color:#fff;font-weight:700;font-size:1.05rem">Feature Inputs</div>\n'
        '              <div style="color:rgba(255,255,255,.38);font-size:.82rem">Enter your feature values below</div>\n'
        '            </div>\n'
        '          </div>\n'
        '\n'
        '          <form id="pForm">\n'
        '            <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:14px;margin-bottom:20px">\n'
        '              TMPL_FIELDS\n'
        '            </div>\n'
        '            <button type="submit" class="btn-predict" id="pBtn">\n'
        '              <span id="btnTxt">Predict</span>\n'
        '            </button>\n'
        '          </form>\n'
        '\n'
        '          <div style="margin-top:24px;padding-top:20px;border-top:1px solid rgba(255,255,255,.08)">\n'
        '            <div style="color:rgba(255,255,255,.25);font-size:.65rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;margin-bottom:10px">About this model</div>\n'
        '            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px">\n'
        '              <div class="mini-stat">\n'
        '                <div style="color:rgba(255,255,255,.35);font-size:.7rem;margin-bottom:4px">Algorithm</div>\n'
        '                <div style="color:#fff;font-weight:600;font-size:.82rem">TMPL_ALGO</div>\n'
        '              </div>\n'
        '              <div class="mini-stat">\n'
        '                <div style="color:rgba(255,255,255,.35);font-size:.7rem;margin-bottom:4px">Accuracy</div>\n'
        '                <div style="color:TMPL_ACC_COLOR;font-weight:700;font-size:.82rem">TMPL_ACCURACY</div>\n'
        '              </div>\n'
        '              <div class="mini-stat">\n'
        '                <div style="color:rgba(255,255,255,.35);font-size:.7rem;margin-bottom:4px">Classes</div>\n'
        '                <div style="color:#fff;font-weight:600;font-size:.82rem">TMPL_CLASSES_COUNT</div>\n'
        '              </div>\n'
        '            </div>\n'
        '          </div>\n'
        '        </div>\n'
        '\n'
        '        <!-- RIGHT: Results Panel -->\n'
        '        <div class="glass" style="border-radius:20px;padding:28px;display:flex;flex-direction:column">\n'
        '          <div style="display:flex;align-items:center;gap:12px;margin-bottom:24px">\n'
        '            <div style="width:40px;height:40px;border-radius:12px;background:linear-gradient(135deg,TMPL_ACCENT,TMPL_BTN);display:flex;align-items:center;justify-content:center;font-size:1.2rem;flex-shrink:0;box-shadow:0 4px 12px rgba(0,0,0,.3)">◎</div>\n'
        '            <div>\n'
        '              <div style="color:#fff;font-weight:700;font-size:1.05rem">Prediction Result</div>\n'
        '              <div style="color:rgba(255,255,255,.38);font-size:.82rem">Output from your trained model</div>\n'
        '            </div>\n'
        '          </div>\n'
        '\n'
        '          <!-- Empty state -->\n'
        '          <div id="emptyState" style="flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;padding:40px 0">\n'
        '            <div style="width:56px;height:56px;border-radius:16px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.09);display:flex;align-items:center;justify-content:center;font-size:1.7rem;margin-bottom:14px">🤖</div>\n'
        '            <div style="color:rgba(255,255,255,.38);font-weight:500">No prediction yet</div>\n'
        '            <div style="color:rgba(255,255,255,.22);font-size:.85rem;margin-top:6px">Fill in the form and hit Predict</div>\n'
        '          </div>\n'
        '\n'
        '          <!-- Result state -->\n'
        '          <div id="resultState" style="display:none;flex:1;flex-direction:column">\n'
        '\n'
        '            <div style="text-align:center;margin-bottom:24px">\n'
        '              <div class="badge" style="margin-bottom:12px">\n'
        '                <span>●</span> <span id="resLabel">Result</span>\n'
        '              </div>\n'
        '              <div id="resVal" style="font-size:2.6rem;font-weight:900;color:#fff;margin-bottom:6px;letter-spacing:-0.5px;line-height:1.1">&#8212;</div>\n'
        '              <div id="resConf" style="color:rgba(255,255,255,.45);font-size:.9rem"></div>\n'
        '            </div>\n'
        '\n'
        '            <div id="probSec" style="display:none;margin-bottom:20px">\n'
        '              <div style="color:rgba(255,255,255,.28);font-size:.65rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;margin-bottom:12px">Class Probabilities</div>\n'
        '              <div id="probBars"></div>\n'
        '            </div>\n'
        '\n'
        '            <div style="background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.07);border-radius:12px;padding:14px;margin-top:auto">\n'
        '              <div style="color:rgba(255,255,255,.28);font-size:.65rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;margin-bottom:10px">Input Summary</div>\n'
        '              <div id="inputSummary" style="display:grid;grid-template-columns:1fr 1fr;gap:6px;font-size:.84rem"></div>\n'
        '            </div>\n'
        '\n'
        '          </div>\n'
        '\n'
        '          <!-- Error state -->\n'
        '          <div id="errState" style="display:none">\n'
        '            <div class="err-box" id="errMsg"></div>\n'
        '          </div>\n'
        '\n'
        '        </div>\n'
        '      </div>\n'
        '\n'
        '      <!-- Footer -->\n'
        '      <div style="text-align:center;margin-top:40px">\n'
        '        <p style="color:rgba(255,255,255,.18);font-size:.82rem">\n'
        '          Powered by\n'
        '          <span style="color:rgba(255,255,255,.35);font-weight:600">FastAPI</span> &middot;\n'
        '          <span style="color:rgba(255,255,255,.35);font-weight:600">scikit-learn</span> &middot;\n'
        '          Built with <span style="color:rgba(255,255,255,.35);font-weight:600">&#10022; AI</span>\n'
        '        </p>\n'
        '      </div>\n'
        '\n'
        '    </div>\n'
        '  </div>\n'
        '\n'
        '<script>\n'
        '  var IS_CLASS = TMPL_IS_CLASS;\n'
        '  var CLASSES  = TMPL_CLASSES;\n'
        '\n'
        '  var pForm       = document.getElementById(\'pForm\');\n'
        '  var pBtn        = document.getElementById(\'pBtn\');\n'
        '  var btnTxt      = document.getElementById(\'btnTxt\');\n'
        '  var srvDot      = document.getElementById(\'srvDot\');\n'
        '  var srvTxt      = document.getElementById(\'srvTxt\');\n'
        '  var emptyState  = document.getElementById(\'emptyState\');\n'
        '  var resultState = document.getElementById(\'resultState\');\n'
        '  var errState    = document.getElementById(\'errState\');\n'
        '  var resVal      = document.getElementById(\'resVal\');\n'
        '  var resConf     = document.getElementById(\'resConf\');\n'
        '  var resLabel    = document.getElementById(\'resLabel\');\n'
        '  var probSec     = document.getElementById(\'probSec\');\n'
        '  var probBars    = document.getElementById(\'probBars\');\n'
        '  var errMsg      = document.getElementById(\'errMsg\');\n'
        '  var inputSum    = document.getElementById(\'inputSummary\');\n'
        '\n'
        '  // ── Server health check with cold-start retry ──────────────────────────────\n'
        '  var _healthTimer = null;\n'
        '  var _serverReady = false;\n'
        '\n'
        '  function setServerState(state, msg) {\n'
        '    srvDot.className = \'srv-dot \' + state;\n'
        '    srvTxt.textContent = msg;\n'
        '    _serverReady = (state === \'online\');\n'
        '    pBtn.disabled = !_serverReady;\n'
        '    pBtn.title = _serverReady ? \'\' : \'Waiting for server…\';\n'
        '  }\n'
        '\n'
        '  function checkServer() {\n'
        '    setServerState(\'checking\', \'Checking server…\');\n'
        '    fetch(\'/health\')\n'
        '      .then(function(r) {\n'
        '        if (r.ok) {\n'
        '          return r.json().then(function(d) {\n'
        '            if (d && d.status === \'ok\') {\n'
        '              setServerState(\'online\', \'Server online\');\n'
        '              if (_healthTimer) { clearInterval(_healthTimer); _healthTimer = null; }\n'
        '            } else {\n'
        '              setServerState(\'offline\', \'Server warming up… retrying\');\n'
        '            }\n'
        '          });\n'
        '        } else {\n'
        '          setServerState(\'offline\', \'Server error (HTTP \' + r.status + \') — retrying\');\n'
        '        }\n'
        '      })\n'
        '      .catch(function() {\n'
        '        setServerState(\'offline\', \'Server offline — run: uvicorn app:app --reload\');\n'
        '        if (_healthTimer) { clearInterval(_healthTimer); _healthTimer = null; }\n'
        '      });\n'
        '  }\n'
        '  checkServer();\n'
        '  _healthTimer = setInterval(checkServer, 5000);\n'
        '\n'
        '  // ── Form submit ────────────────────────────────────────────────────────────\n'
        '  pForm.addEventListener(\'submit\', function(e) {\n'
        '    e.preventDefault();\n'
        '    if (!_serverReady) {\n'
        '      showError(\'Server is not ready yet.\\nPlease wait until the status shows Server online.\');\n'
        '      return;\n'
        '    }\n'
        '\n'
        '    var payload = {};\n'
        '    var summary = [];\n'
        '    pForm.querySelectorAll(\'input\').forEach(function(el) {\n'
        '      var v = el.value.trim();\n'
        '      if (v !== \'\') {\n'
        '        payload[el.name] = (el.type === \'number\' && !isNaN(v)) ? parseFloat(v) : v;\n'
        '        var lbl = el.previousElementSibling ? el.previousElementSibling.textContent.trim() : el.name;\n'
        '        summary.push([lbl, v]);\n'
        '      }\n'
        '    });\n'
        '\n'
        '    pBtn.disabled = true;\n'
        '    btnTxt.innerHTML = \'<span class="spinner"></span>Predicting…\';\n'
        '    emptyState.style.display  = \'none\';\n'
        '    resultState.style.display = \'none\';\n'
        '    errState.style.display    = \'none\';\n'
        '\n'
        '    fetch(\'/predict\', {\n'
        '      method: \'POST\',\n'
        '      headers: {\'Content-Type\': \'application/json\'},\n'
        '      body: JSON.stringify(payload)\n'
        '    })\n'
        '    .then(function(r) {\n'
        '      if (!r.ok) return r.text().then(function(t) { throw new Error(\'API \' + r.status + \': \' + t); });\n'
        '      return r.json();\n'
        '    })\n'
        '    .then(function(data) {\n'
        '      showResult(data, summary);\n'
        '    })\n'
        '    .catch(function(err) {\n'
        '      var msg = err.message || String(err);\n'
        '      if (msg.indexOf(\'Failed to fetch\') !== -1 || msg.indexOf(\'NetworkError\') !== -1\n'
        '          || msg.indexOf(\'ERR_CONNECTION_REFUSED\') !== -1 || err instanceof TypeError) {\n'
        '        msg = \'⚠ Cannot reach the API server.\\nThe server may have gone to sleep \u2014 please wait a moment.\';\n'
        '        setServerState(\'offline\', \'Server offline \u2014 retrying\u2026\');\n'
        '        if (!_healthTimer) _healthTimer = setInterval(checkServer, 5000);\n'
        '      }\n'
        '      showError(msg);\n'
        '    })\n'
        '    .finally(function() {\n'
        '      pBtn.disabled = !_serverReady;\n'
        '      btnTxt.textContent = \'Predict Again\';\n'
        '    });\n'
        '  });\n'
        '\n'
        '  function showResult(data, summary) {\n'
        '    resVal.textContent = IS_CLASS ? _fmt(String(data.prediction)) : data.prediction;\n'
        '\n'
        '    if (IS_CLASS && data.probabilities) {\n'
        '      var probs  = Array.isArray(data.probabilities) ? data.probabilities : Object.values(data.probabilities);\n'
        '      var labels = CLASSES.length ? CLASSES : probs.map(function(_,i){ return \'Class \' + i; });\n'
        '      var maxP   = Math.max.apply(null, probs);\n'
        '\n'
        '      resLabel.textContent = \'Classified\';\n'
        '      resConf.textContent  = \'Confidence: \' + (maxP * 100).toFixed(1) + \'%\';\n'
        '\n'
        '      probBars.innerHTML = \'\';\n'
        '      var pairs = labels.map(function(l,i){ return {l:l, p:probs[i]||0}; })\n'
        '                        .sort(function(a,b){ return b.p - a.p; });\n'
        '      function _fmt(s){return s.replace(/[-_]/g,\' \').replace(/\\b\\w/g,function(c){return c.toUpperCase();});}\n'
        '      pairs.forEach(function(item) {\n'
        '        var pct   = (item.p * 100).toFixed(1);\n'
        '        var isTop = item.p === maxP;\n'
        '        var d = document.createElement(\'div\');\n'
        '        d.style.marginBottom = \'12px\';\n'
        '        d.innerHTML =\n'
        '          \'<div style="display:flex;justify-content:space-between;margin-bottom:5px">\' +\n'
        '            \'<span style="font-size:.85rem;font-weight:\' + (isTop ? \'600\' : \'400\') +\n'
        '            \';color:rgba(255,255,255,\' + (isTop ? \'.9\' : \'.45\') + \')">\' + _fmt(item.l) + \'</span>\' +\n'
        '            \'<span style="font-size:.85rem;font-weight:700;color:rgba(255,255,255,\' + (isTop ? \'.95\' : \'.35\') + \')">\' + pct + \'%</span>\' +\n'
        '          \'</div>\' +\n'
        '          \'<div class="conf-track"><div class="conf-fill" data-p="\' + pct + \'" style="width:0"></div></div>\';\n'
        '        probBars.appendChild(d);\n'
        '      });\n'
        '      probSec.style.display = \'block\';\n'
        '      setTimeout(function() {\n'
        '        document.querySelectorAll(\'.conf-fill\').forEach(function(b) {\n'
        '          b.style.width = b.dataset.p + \'%\';\n'
        '        });\n'
        '      }, 60);\n'
        '    } else {\n'
        '      resLabel.textContent = \'Predicted Value\';\n'
        '      resConf.textContent  = \'\';\n'
        '      probSec.style.display = \'none\';\n'
        '    }\n'
        '\n'
        '    inputSum.innerHTML = \'\';\n'
        '    summary.forEach(function(pair) {\n'
        '      var d = document.createElement(\'div\');\n'
        '      d.style.cssText = \'display:flex;justify-content:space-between\';\n'
        '      d.innerHTML =\n'
        '        \'<span style="color:rgba(255,255,255,.35)">\' + pair[0] + \'</span>\' +\n'
        '        \'<span style="color:#fff;font-weight:500">\' + pair[1] + \'</span>\';\n'
        '      inputSum.appendChild(d);\n'
        '    });\n'
        '\n'
        '    resultState.style.display = \'flex\';\n'
        '    resultState.classList.remove(\'fade-up\');\n'
        '    void resultState.offsetWidth;\n'
        '    resultState.classList.add(\'fade-up\');\n'
        '    errState.style.display = \'none\';\n'
        '  }\n'
        '\n'
        '  function showError(msg) {\n'
        '    errMsg.textContent = msg;\n'
        '    errState.style.display    = \'block\';\n'
        '    resultState.style.display = \'none\';\n'
        '    emptyState.style.display  = \'none\';\n'
        '  }\n'
        '</script>\n'
        '</body>\n'
        '</html>\n'
    )
    # Substitute all placeholders — longer tokens first to avoid partial matches.
    accent_alpha = accent + "33"
    replacements = [
        ("TMPL_MESH_CSS",      mesh_css),
        ("TMPL_HEADER_BG",     header_bg),
        ("TMPL_BADGE_BORDER",  badge_border),
        ("TMPL_BADGE_TEXT",    badge_text),
        ("TMPL_BADGE_BG",      badge_bg),
        ("TMPL_BTN_HOVER",     btn_hover),
        ("TMPL_BTN",           btn),
        ("TMPL_ACCENT_ALPHA",  accent_alpha),
        ("TMPL_ACCENT",        accent),
        ("TMPL_IS_CLASS",      is_class_js),
        ("TMPL_CLASSES_COUNT", classes_count),
        ("TMPL_CLASSES",       classes_js),
        ("TMPL_ACCURACY",      accuracy_str),
        ("TMPL_ACC_COLOR",     acc_color),
        ("TMPL_ALGO",          algo_str),
        ("TMPL_FIELDS",        fields_html),
        ("TMPL_DOMAIN_NAME",   domain_name),
        ("TMPL_DESC",          desc),
        ("TMPL_TITLE",         title),
    ]
    for placeholder, value in replacements:
        html = html.replace(placeholder, value)

    out_path = root / "index.html"
    out_path.write_text(html, encoding="utf-8")
    _ok("index.html → " + str(out_path))
    _info("Theme detected: " + theme["name"] + " " + icon)
    _info("Open at: http://localhost:8000/")
    return out_path


def _generate_app(root, task_type, num_feats, cat_feats):
    """Write a working FastAPI app.py based on the fitted pipeline."""
    _print_header("Step 8 — Generating FastAPI app.py")
    lines = [
        '#!/usr/bin/env python3',
        '"""Auto-generated FastAPI prediction API."""',
        'import os',
        'from fastapi import FastAPI',
        'from fastapi.responses import FileResponse',
        'from fastapi.middleware.cors import CORSMiddleware',
        'from pydantic import BaseModel',
        'from typing import Optional, List',
        'import joblib, pandas as pd',
        '',
        'app = FastAPI(title="ML Prediction API")',
        'app.add_middleware(',
        '    CORSMiddleware,',
        '    allow_origins=["*"],',
        '    allow_methods=["*"],',
        '    allow_headers=["*"],',
        ')',
        'pipeline = joblib.load("models/final_pipeline.pkl")',
    ]
    if task_type == "classification":
        lines.append('label_encoder = joblib.load("models/label_encoder.pkl")')
    lines += ['', 'class InputData(BaseModel):']
    for feat in num_feats:
        lines.append('    ' + feat + ': Optional[float] = None')
    for feat in cat_feats:
        lines.append('    ' + feat + ': Optional[str] = None')
    if not num_feats and not cat_feats:
        lines.append('    pass')
    lines += [
        '',
        '@app.get("/")',
        'def index():',
        '    """Serve the prediction UI if index.html exists."""',
        '    if os.path.exists("index.html"):',
        '        return FileResponse("index.html")',
        '    return {"message": "ML Prediction API", "docs": "/docs"}',
        '',
        '@app.get("/health")',
        'def health():',
        '    return {"status": "ok", "model": "loaded"}',
        '',
        '@app.post("/predict")',
        'def predict(data: InputData):',
        '    df = pd.DataFrame([data.dict()])',
        '    pred = pipeline.predict(df)[0]',
    ]
    if task_type == "classification":
        lines += [
            '    label = label_encoder.inverse_transform([pred])[0]',
            '    proba = pipeline.predict_proba(df)[0].tolist()',
            '    return {"prediction": str(label), "probabilities": proba}',
        ]
    else:
        lines.append('    return {"prediction": float(pred)}')
    lines += [
        '',
        '@app.post("/predict/batch")',
        'def predict_batch(data: List[InputData]):',
        '    df = pd.DataFrame([d.dict() for d in data])',
        '    preds = pipeline.predict(df)',
    ]
    if task_type == "classification":
        lines += [
            '    labels = label_encoder.inverse_transform(preds).tolist()',
            '    return {"predictions": [str(l) for l in labels]}',
        ]
    else:
        lines.append('    return {"predictions": preds.tolist()}')
    lines += [
        '',
        'if __name__ == "__main__":',
        '    import uvicorn',
        '    uvicorn.run(app, host="0.0.0.0", port=8000)',
    ]
    app_path = root / "app.py"
    app_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _ok("app.py → " + str(app_path))
    _info('Run locally: .venv/bin/uvicorn app:app --reload')


def _generate_docker(root):
    """Write Dockerfile and .dockerignore."""
    _print_header("Step 9 — Generating Dockerfile")
    dockerfile_lines = [
        "# Multi-stage Docker build — auto-generated by auto_pipeline.py",
        "FROM python:3.11-slim AS builder",
        "WORKDIR /app",
        "COPY requirements.txt .",
        "RUN pip install --prefix=/install --no-cache-dir -r requirements.txt",
        "",
        "FROM python:3.11-slim",
        "WORKDIR /app",
        "COPY --from=builder /install /usr/local",
        "COPY app.py .",
        "COPY index.html* .",
        "COPY models/ models/",
        "RUN useradd -m appuser && chown -R appuser /app",
        "USER appuser",
        "EXPOSE 8000",
        'CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]',
    ]
    dockerignore_lines = [
        ".git/", "__pycache__/", ".venv/",
        "data/", "plots/", "tests/", "docs/",
        "*.md", ".env", ".DS_Store", ".claude/",
        "*.py[cod]", "*.log",
    ]
    (root / "Dockerfile").write_text("\n".join(dockerfile_lines) + "\n", encoding="utf-8")
    _ok("Dockerfile generated")
    (root / ".dockerignore").write_text("\n".join(dockerignore_lines) + "\n", encoding="utf-8")
    _ok(".dockerignore generated")


def _push_github(root, cfg):
    """Git init, commit, create GitHub repo, push."""
    _print_header("Step 10 — Pushing to GitHub")
    gh_user = cfg.get("github_username", "")
    gh_repo = cfg.get("github_repo", "")
    gh_vis  = cfg.get("github_visibility", "public")

    if not gh_user or not gh_repo:
        _warn("No GitHub username/repo found in .ml_config.json — skipping.")
        _info("Add github_username + github_repo to .ml_config.json and re-run.")
        return False

    if not _shutil.which("gh"):
        _err("GitHub CLI not found.  Install: brew install gh")
        _err("Then authenticate: gh auth login")
        return False

    r = subprocess.run(["gh", "auth", "status"], capture_output=True, text=True)
    if r.returncode != 0:
        _err("Not logged into GitHub CLI.  Run: gh auth login")
        return False

    for cmd in [["git", "init"], ["git", "add", "."],
                ["git", "commit", "-m", "Initial commit: auto ML pipeline with FastAPI"]]:
        r = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True)
        if r.returncode != 0 and "nothing to commit" not in (r.stdout + r.stderr):
            _warn("git: " + r.stderr.strip())

    r = subprocess.run(
        ["gh", "repo", "create", gh_repo,
         "--" + gh_vis, "--description", "Auto-generated ML pipeline",
         "--source=.", "--remote=origin", "--push"],
        cwd=str(root), capture_output=True, text=True,
    )
    if r.returncode != 0:
        if "already exists" in r.stderr:
            _warn("Repo already exists — pushing to existing remote...")
            subprocess.run(["git", "push", "-u", "origin", "main"],
                           cwd=str(root), capture_output=True)
        else:
            _err("GitHub push failed: " + r.stderr.strip())
            return False

    _ok("Pushed → https://github.com/" + gh_user + "/" + gh_repo)
    return True


def _deploy_render(root, cfg):
    """Write render.yaml, commit, push, and print next steps."""
    _print_header("Step 11 — Setting up Render Deployment")
    proj     = cfg.get("project_name", "ml-project")
    gh_user  = cfg.get("github_username", "")
    gh_repo  = cfg.get("github_repo", "")

    # Render normalises the service name to lowercase and replaces
    # underscores with hyphens when forming the subdomain URL.
    render_name = proj.lower().replace("_", "-")

    render_lines = [
        "services:",
        "  - type: web",
        "    name: " + render_name,
        "    runtime: python",
        "    buildCommand: pip install -r requirements.txt",
        "    startCommand: uvicorn app:app --host 0.0.0.0 --port $PORT",
        "    envVars:",
        '      - key: PYTHON_VERSION',
        '        value: "3.11.0"',
    ]
    (root / "render.yaml").write_text("\n".join(render_lines) + "\n", encoding="utf-8")
    _ok("render.yaml created  (service name: " + render_name + ")")

    subprocess.run(["git", "add", "render.yaml"], cwd=str(root), capture_output=True)
    subprocess.run(["git", "commit", "-m", "Add render.yaml for Render deployment"],
                   cwd=str(root), capture_output=True)
    subprocess.run(["git", "push", "origin", "main"], cwd=str(root), capture_output=True)
    _ok("render.yaml pushed to GitHub")

    live = "https://" + render_name + ".onrender.com"
    print("\n" + _B + "  Go live on Render (free, ~2 minutes):" + _X)
    print("    1. Visit https://render.com → sign in with GitHub")
    print("    2. Click New + → Web Service")
    print("    3. Connect repo: " + gh_user + "/" + gh_repo)
    print("    4. Render detects render.yaml → click Create Web Service")
    print("    5. Live at: " + _G + live + _X)
    print(_Y + "    Note: URL is always lowercase with hyphens — Render ignores case/underscores." + _X)
    print("\n  Test once deployed:")
    print("    curl " + live + "/health")
    _ok("Render setup complete — finish in the Render dashboard")


def _post_pipeline_menu(root, cfg, task_type, pipeline, label_encoder=None):
    """Show post-pipeline options and run the chosen ones."""
    num_feats, cat_feats = _get_features(pipeline)
    sep = _C + _B + "─" * 56 + _X
    print("\n" + sep)
    print(_B + "  What would you like to do next?" + _X)
    print(sep)
    print("  1) Generate FastAPI app + Dockerfile")
    print("  2) Push to GitHub")
    print("  3) Deploy to Render  (requires GitHub push first)")
    print("  4) All of the above  ← recommended")
    print("  5) Done — I'll handle it myself")
    print(sep)
    try:
        choice = input("  Enter choice (default: 4): ").strip() or "4"
    except EOFError:
        choice = "4"   # non-interactive / piped — use default

    do_app    = choice in ("1", "4")
    do_github = choice in ("2", "4")
    do_render = choice in ("3", "4")

    if do_app:
        _generate_app(root, task_type, num_feats, cat_feats)
        _generate_frontend(root, cfg, task_type, num_feats, cat_feats, label_encoder)
        _generate_docker(root)

    if do_github:
        pushed = _push_github(root, cfg)
        if not pushed and do_render:
            _warn("Skipping Render setup — GitHub push failed or was skipped.")
            do_render = False

    if do_render:
        _deploy_render(root, cfg)

    if choice == "5":
        print("\n" + _Y + "  Manual mode — your project is at: " + str(root) + _X)
        print("  Generate API : run this script again and choose option 1")
        print("  Push GitHub  : gh repo create <name> --public --source=. --remote=origin --push")
        print("  Deploy Render: connect your GitHub repo at render.com")


# ── Run the post-pipeline menu ────────────────────────────────────────
_post_pipeline_menu(ROOT, cfg, task_type, final_pipe, label_encoder)'''
# .gitkeep for empty directories
FILES["data/.gitkeep"]   = ""
FILES["models/.gitkeep"] = ""
FILES["plots/.gitkeep"]  = ""

# ── Prerequisites Check ──────────────────────────────────────────────
def check_prereqs():
    print(f"\n{C}{B}Checking prerequisites...{X}")

    def _run(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None

    if shutil.which("brew"):
        print(f"  {G}✔ Homebrew{X}")
    else:
        print(f"  {Y}⚠  Homebrew not found — installing...{X}")
        os.system('/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"')
        if Path("/opt/homebrew/bin/brew").exists():
            os.environ["PATH"] = "/opt/homebrew/bin:" + os.environ.get("PATH", "")

    if shutil.which("npm"):
        r = _run(["node", "--version"])
        print(f"  {G}✔ Node.js {r.stdout.strip() if r else ''}{X}")
    else:
        print(f"  {Y}⚠  Node.js not found — installing via Homebrew...{X}")
        os.system("brew install node")

    if shutil.which("claude"):
        r = _run(["claude", "--version"])
        ver = r.stdout.strip().split("\n")[0] if r else "installed"
        print(f"  {G}✔ Claude Code CLI {ver}{X}")
    else:
        print(f"  {Y}⚠  Claude Code CLI not found — installing...{X}")
        result = os.system("npm install -g @anthropic-ai/claude-code")
        if result == 0:
            print(f"  {G}✔ Claude Code CLI installed{X}")
        else:
            print(f"  {R}✗ Auto-install failed. Run manually:{X}")
            print(f"    npm install -g @anthropic-ai/claude-code")

# ── Main ─────────────────────────────────────────────────────────────
print(f"""
{C}{B}╔══════════════════════════════════════════════════╗
║        🤖  ML Pipeline Template  v{VERSION}          ║
║   End-to-End Machine Learning Automation         ║
╚══════════════════════════════════════════════════╝{X}
""")

check_prereqs()
cfg = collect_inputs()

# ── Create project dir (staging → atomic move, APFS-safe) ────────────
timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
project_dir = Path(".").resolve() / f"{cfg['project_name']}_{timestamp}"
staging_dir = project_dir.parent / f".ml_staging_{project_dir.name}"

if project_dir.exists():
    print(f"\n{R}Error:{X} '{project_dir.name}' already exists. Choose a different name.")
    sys.exit(1)

staging_dir.mkdir(parents=True, exist_ok=True)

# Write template files — skip setup scripts (they belong in the template source, not projects)
SKIP_IN_PROJECT = {"start.sh", "init.py"}
print(f"\n{G}▶ Preparing project files...{X}")
for rel_path, content in FILES.items():
    if rel_path in SKIP_IN_PROJECT:
        continue
    full = staging_dir / rel_path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content, encoding="utf-8")
print(f"  {G}✔ Template files ready{X}")

# Copy dataset if provided
if cfg["dataset_path"] and Path(cfg["dataset_path"]).is_file():
    (staging_dir / "data").mkdir(exist_ok=True)
    shutil.copy2(cfg["dataset_path"], staging_dir / "data" / cfg["dataset_filename"])
    print(f"  {G}✔ Dataset copied: {cfg['dataset_filename']}{X}")

write_config(cfg, staging_dir)

# .venv + deps
print(f"\n{G}▶ Creating Python virtual environment (.venv)...{X}")
subprocess.run([sys.executable, "-m", "venv", str(staging_dir / ".venv")], check=True)
print(f"  {G}✔ Virtual environment created{X}")

print(f"{G}▶ Installing dependencies (this may take a minute)...{X}")
_pip = str(staging_dir / ".venv" / "bin" / "pip")
subprocess.run([_pip, "install", "--upgrade", "pip", "-q"], check=True)
_req = staging_dir / "requirements.txt"
if _req.exists():
    subprocess.run([_pip, "install", "-r", str(_req), "-q"], check=True)
print(f"  {G}✔ Dependencies installed{X}")

# Atomic move → VS Code sees project appear once, fully complete
print(f"\n{G}▶ Creating project at: {project_dir}{X}")
shutil.move(str(staging_dir), str(project_dir))

# Fix .venv shebangs — rewrite every script in .venv/bin/ that has a
# shebang pointing to the old staging path, replacing it with the real path.
_venv_bin = project_dir / ".venv" / "bin"
_venv_python = str(_venv_bin / "python")
_bad_prefix  = str(staging_dir)   # staging_dir was moved, so this path is gone
for _script in _venv_bin.iterdir():
    try:
        if not _script.is_file() or _script.is_symlink():
            continue
        raw = _script.read_bytes()
        if not raw.startswith(b"#!"):
            continue
        first_nl = raw.index(b"\n")
        shebang = raw[:first_nl].decode("utf-8", errors="replace")
        if _bad_prefix in shebang:
            new_shebang = ("#!" + _venv_python).encode("utf-8")
            _script.write_bytes(new_shebang + raw[first_nl:])
    except Exception:
        pass

# Summary
plat_labels = {
    "ask_later": "Ask me later",   "render":   "Render (free tier)",
    "fly.io":    "Fly.io",         "railway":  "Railway",
    "aws":       "AWS App Runner", "gcp":      "GCP Cloud Run",
    "azure":     "Azure Container Apps", "none": "Local / Docker only",
}
fn   = cfg["dataset_filename"] or "<not provided yet>"
plat = plat_labels.get(cfg["platform"], cfg["platform"])
u, r = cfg["github_username"], cfg["github_repo"]
gh_line = f"\n{C}{B}║{X}  🐙  GitHub : github.com/{u}/{r}" if u else ""
print(f"""
{C}{B}╔══════════════════════════════════════════════════╗
║  ✅  Project ready!                              ║
╠══════════════════════════════════════════════════╣{X}
{C}{B}║{X}  📁  {project_dir}
{C}{B}║{X}  🐍  Venv   : .venv/
{C}{B}║{X}  📊  Data   : {fn}
{C}{B}║{X}  🚀  Deploy : {plat}{gh_line}
{C}{B}╠══════════════════════════════════════════════════╣{X}
{C}{B}║{X}  ✅  Launching Claude Code...
{C}{B}╚══════════════════════════════════════════════════╝{X}
""")

# Launch menu
print(f"\n{B}How would you like to run the pipeline?{X}")
print(f"  1) {G}Claude Code{X}   — AI-driven, fully automated (recommended)")
print(f"  2) {C}Auto Pipeline{X} — no Claude subscription needed (pure sklearn)")
print(f"  3) {Y}Manual{X}        — I'll run it myself later")
launch_choice = input("Enter choice (default: 1): ").strip() or "1"

os.chdir(project_dir)

if launch_choice == "2":
    print(f"\n{G}▶ Running automated pipeline (no Claude required)...{X}")
    python = str(project_dir / ".venv" / "bin" / "python")
    # os.execv replaces this process — child gets stdin cleanly (no buffering issue)
    os.execv(python, [python, "auto_pipeline.py"])
elif launch_choice == "3":
    print(f"\n{Y}▶ Manual mode — project is ready at:{X}")
    print(f"   cd {project_dir}")
    print(f"   source .venv/bin/activate")
    print(f"   python3 auto_pipeline.py   # auto pipeline")
    print(f"   claude .                   # AI-driven pipeline")
else:
    print(f"\n{G}▶ Launching Claude Code in your new project...{X}")
    if shutil.which("claude"):
        subprocess.run(["claude", "."])
    else:
        print(f"{Y}Claude Code CLI not found. Install: npm install -g @anthropic-ai/claude-code{X}")
        print(f"Then run: {B}cd {project_dir} && source .venv/bin/activate && claude .{X}")
