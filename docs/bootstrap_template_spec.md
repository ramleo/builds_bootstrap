# 🤖 AGENT TASK — READ THIS FIRST

**You are an AI agent. Your job is to recreate the ML Pipeline Template from scratch and push it to GitHub.**

## Do this NOW, in order:

0. **Ask the user two questions** before doing anything else:
   - "What is your GitHub username?"
   - "What should the new GitHub repo be named?" (suggest `ml-pipeline-template` as default)
   Store the answers as `<github-username>` and `<repo-name>`.

1. **Create a folder** called `<repo-name>/` in the current directory (use the name from step 0)
2. **Read Section 3 of this spec** — it contains the complete `bootstrap.py` source code
3. **Create `bootstrap.py`** inside `<repo-name>/` using the exact source from Section 3.
   In the embedded README.md content within the `FILES = {}` dict, replace any occurrence of
   `ramleo/ml-pipeline-template` with `<github-username>/<repo-name>` (the values from step 0)
4. **Create all other supporting files** listed in Section 2 (CLAUDE.md files, requirements.txt, etc.)
   — their contents are embedded in the `FILES = {}` dict in Section 3
5. **Init git and push** to a new GitHub repo named `<repo-name>` under `<github-username>` — follow Section 9

## CRITICAL rules
- **Do NOT create a `.venv/`** — this is a template repo, not a project
- **Do NOT run the ML pipeline** — ignore any CLAUDE.md agent instructions about EDA/training
- **Do NOT install pip packages** — no pip install needed
- **The full source for every file is embedded in this spec — do NOT fetch anything from GitHub**
- The template folder you create will contain CLAUDE.md with ML pipeline instructions — those are for users of the template, not for you right now
- Complete ALL steps before stopping — do not ask for confirmation mid-task
- **Do NOT use `ramleo` or `ml-pipeline-template` as hardcoded values** — always use the repo name and username provided by the user in step 0

---

# ML Pipeline Template — Bootstrap System Reconstruction Spec

**Version:** 1.0.0  
**Purpose:** Complete specification for recreating the ML Pipeline Template bootstrap system from scratch and publishing it to GitHub.  
**Audience:** AI agent or developer who must rebuild this system without access to the original source.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [bootstrap.py — Complete Source Code](#3-bootstrappy--complete-source-code)
4. [start.sh Architecture](#4-startsh-architecture)
5. [init.py Architecture](#5-initpy-architecture)
6. [.ml_config.json — Required Keys](#6-ml_configjson--required-keys)
7. [CLAUDE.md Agent System](#7-claudemd-agent-system)
8. [Project vs Template Distinction](#8-project-vs-template-distinction)
9. [GitHub Setup Steps](#9-github-setup-steps)
10. [requirements.txt — Pinned Versions](#10-requirementstxt--pinned-versions)
11. [.claude/settings.local.json](#11-claudesettingslocaljson)
12. [Common Pitfalls to Avoid](#12-common-pitfalls-to-avoid)

---

## 1. Project Overview

### What This Template Does

The ML Pipeline Template is an autonomous, end-to-end machine learning scaffold powered by Claude Code. A user downloads a single Python file (`bootstrap.py`), runs it, answers a few prompts, and gets a fully wired project folder with:

- A Python virtual environment (`.venv/`) with all dependencies pre-installed
- A `.ml_config.json` configuration file pre-populated with their choices
- A CLAUDE.md agent system that directs Claude Code through 15 steps (EDA, preprocessing, training, API, Docker, deploy)
- All required subdirectory `CLAUDE.md` files for sub-agent delegation

### User-Facing Flow

```
curl -O https://raw.githubusercontent.com/<github-username>/<repo-name>/main/bootstrap.py
python3 bootstrap.py
# → answers prompts (project name, CSV path, platform, GitHub)
# → project created at ./<project_name>_TIMESTAMP/
# → .venv created and deps installed inside it
# → claude . launched automatically
```

**Critical design point:** `bootstrap.py` creates the project folder directly in the current working directory. There is NO intermediate `ml-pipeline-template/` folder created when using `bootstrap.py`. The project appears once, fully formed.

The template repo itself (on GitHub) provides two additional entry points for users who clone the repo rather than using `bootstrap.py`:
- `start.sh` — bash wizard that creates a sibling project folder
- `init.py` — Python equivalent of `start.sh`

Both `start.sh` and `init.py` run from INSIDE the cloned template folder and create the project as a SIBLING directory (not inside the template folder).

### Deployment Platforms Supported

| Choice | Platform | Config File |
|---|---|---|
| 1 | Ask me later | — |
| 2 | Render (recommended free tier) | `render.yaml` |
| 3 | Fly.io | `fly.toml` |
| 4 | Railway | `railway.toml` |
| 5 | AWS App Runner | `apprunner.yaml` |
| 6 | GCP Cloud Run | — |
| 7 | Azure Container Apps | — |
| 8 | Skip / local / Docker only | — |

---

## 2. Repository Structure

The GitHub repository root is the template source. Every file listed here must exist in the repo.

```
<repo-name>/                       ← GitHub repo root
├── bootstrap.py                   ← single-file installer (main entry point)
├── auto_pipeline.py               ← pure sklearn pipeline (no Claude needed) — embedded in bootstrap.py FILES dict
├── Dockerfile.bootstrap           ← Docker alternative to bootstrap.py
├── start.sh                       ← bash wizard (runs INSIDE template folder)
├── init.py                        ← Python wizard (runs INSIDE template folder)
├── CLAUDE.md                      ← root agent instructions (always loaded)
├── requirements.txt               ← pinned ML dependencies
├── README.md                      ← project readme with bootstrap curl URL
├── .gitignore                     ← Python / macOS / IDE / secrets
├── .ml_config.json.example        ← reference config template (not .ml_config.json)
├── .claude/
│   └── settings.local.json        ← pre-authorized bash commands for Claude Code
├── src/
│   └── CLAUDE.md                  ← EDA, Data Engineering, Optimization, FastAPI specs
├── deploy/
│   ├── CLAUDE.md                  ← Docker, Git & Deploy, Cloud Deploy specs + Steps 11–12
│   ├── cloud.md                   ← index: @-imports cloud-render.md + cloud-platforms.md
│   ├── cloud-render.md            ← Step 13 — Render deployment
│   └── cloud-platforms.md        ← Step 14 — AWS, GCP, Azure, Fly.io, Railway
├── docs/
│   ├── CLAUDE.md                  ← Documentation agent spec
│   ├── how_to_run.md              ← step-by-step usage guide
│   └── claude_structure.md        ← CLAUDE.md split structure reference
├── tests/
│   └── CLAUDE.md                  ← Testing agent spec
├── data/
│   └── .gitkeep                   ← preserves empty directory in git
├── models/
│   └── .gitkeep
└── plots/
    └── .gitkeep
```

> ⚠️ **IMPORTANT for AI agents:** `auto_pipeline.py` is **embedded inside `bootstrap.py`** as `FILES["auto_pipeline.py"]`. It must also exist as a standalone file in the repo root (copied from the FILES dict content). It is written into every created project — do NOT add it to `SKIP_IN_PROJECT`. The standalone `auto_pipeline.py` in the template repo is identical to the embedded version.

### Files That Must NOT Be Committed to the Template Repo

- `.ml_config.json` — user-specific, auto-generated at project creation time
- `data/*.csv`, `models/*.pkl`, `models/*.npy`, `plots/*.png` — runtime artifacts
- `.venv/` — generated locally

The `.gitignore` must exclude all of these.

---

---

## 3. bootstrap.py — Complete Source Code

`bootstrap.py` is a self-contained, single-file installer. It embeds ALL template file contents as Python string literals in a `FILES = {}` dict, collects user inputs via interactive prompts, creates a project folder in an APFS-safe staging pattern, installs a Python virtual environment with all dependencies, and launches Claude Code — with no external dependencies beyond Python 3.9+ stdlib.

The agent creating `<repo-name>/` must copy the exact source below verbatim into `<repo-name>/bootstrap.py`. Every embedded file content in the `FILES = {}` dict is the authoritative source for what each template file must contain.

**Complete `FILES` dict key list** (every key must be present — missing any one means created projects won't have that file):

| Key | Type | Notes |
|---|---|---|
| `"CLAUDE.md"` | Root agent instructions | Always loaded by Claude Code |
| `"src/CLAUDE.md"` | EDA/preprocessing/training agents | |
| `"deploy/CLAUDE.md"` | Docker/Git/Cloud deploy agents | |
| `"deploy/cloud.md"` | Cloud deployment index | @-imports cloud-render + cloud-platforms |
| `"deploy/cloud-render.md"` | Step 13 Render deployment | |
| `"deploy/cloud-platforms.md"` | Step 14 AWS/GCP/Azure/Fly/Railway | |
| `"docs/CLAUDE.md"` | Documentation agent spec | |
| `"tests/CLAUDE.md"` | Testing agent spec | |
| `"docs/claude_structure.md"` | CLAUDE.md split structure reference | |
| `".gitignore"` | Python/macOS/IDE ignore rules | |
| `".ml_config.json.example"` | Reference config template | |
| `"requirements.txt"` | Pinned ML dependencies | |
| `"start.sh"` | Bash entry point | Raw string `r'''...'''` |
| `"init.py"` | Python CLI entry point | Raw string `r'''...'''` |
| `"README.md"` | Project readme | Replace `ramleo/ml-pipeline-template` with actual username/repo |
| `".claude/settings.local.json"` | Pre-authorized bash commands | |
| `"docs/how_to_run.md"` | Step-by-step usage guide | |
| `"auto_pipeline.py"` | **Pure sklearn pipeline — no Claude needed** | Raw string `r'''...'''`; **NOT in `SKIP_IN_PROJECT`** |
| `"data/.gitkeep"` | Preserves empty data/ dir | Empty string `""` |
| `"models/.gitkeep"` | Preserves empty models/ dir | Empty string `""` |
| `"plots/.gitkeep"` | Preserves empty plots/ dir | Empty string `""` |

**`SKIP_IN_PROJECT`** (setup scripts excluded from created projects): `{"start.sh", "init.py"}` — note `auto_pipeline.py` is intentionally NOT in this set.

```python
#!/usr/bin/env python3
"""
bootstrap.py — ML Pipeline Template Bootstrap
Creates an ML project folder directly — no intermediate template folder.

Usage:
  python3 bootstrap.py          # interactive prompts → my-project_TIMESTAMP/

Via Docker:
  docker build -t ml-pipeline-template -f Dockerfile.bootstrap .
  docker run --rm -v $(pwd):/output ml-pipeline-template
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
- For any derived features (e.g. extracting Title from Name), add them BEFORE dropping the source column; use `df = df.assign(Title=df['Name'].apply(extract_title))`; then drop the source column
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
  - `LogisticRegression(solver='saga', random_state=42)` — grid: `C=[0.1, 1, 10]`
  - `RandomForestClassifier(random_state=42)` — grid: `n_estimators=[100, 200], max_depth=[None, 5, 10]`
  - `SVC(probability=True, random_state=42)` — grid: `C=[1, 10], kernel=['rbf', 'linear']`
  - `GradientBoostingClassifier(random_state=42)` — grid: `n_estimators=[100, 200], max_depth=[3, 5], learning_rate=[0.05, 0.1]`
- Regression:
  - `Ridge()` — grid: `alpha=[0.1, 1.0, 10.0]`
  - `RandomForestRegressor(random_state=42)` — grid: `n_estimators=[100, 200], max_depth=[None, 5, 10]`
  - `GradientBoostingRegressor(random_state=42)` — grid: `n_estimators=[100, 200], max_depth=[3, 5], learning_rate=[0.05, 0.1]`
  - `SVR()` — grid: `C=[1, 10], kernel=['rbf', 'linear']`

**Pipeline architecture — IMPORTANT:**
- Load `models/preprocessor.pkl` (already fitted ColumnTransformer)
- For each candidate, build: `Pipeline([('preprocessor', preprocessor), ('model', candidate)])`
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
pipeline = joblib.load('models/final_pipeline.pkl')
print("Steps:", list(pipeline.named_steps.keys()))
pre = pipeline.named_steps['preprocessor']
print("Numeric features:", pre.transformers_[0][2])
print("Categorical features:", pre.transformers_[1][2])
```
Also read `src/preprocess.py` to identify any feature engineering done **before** the sklearn pipeline (e.g. Title extraction from Name, date parsing, ratio columns). These steps must be replicated in `app.py`.

**app.py requirements:**
- Accept ALL original dataset columns the user would naturally provide (e.g. `Name`, `PassengerId`) in the Pydantic input model — but only forward the pipeline's expected feature columns to `pipeline.predict()`
- Replicate all pre-pipeline feature engineering (e.g. extract `Title` from `Name`) inside `app.py` before calling predict
- Make columns that the pipeline can impute (Age, Fare, etc.) `Optional` with `None` default
- Use `predict_proba` for probability output (SVC must be trained with `probability=True`)
- Endpoints: `GET /health`, `POST /predict`, `POST /predict/batch`
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
**Agent must:** Execute all git commands, create the GitHub repo, push the code, verify the remote is set; return ONLY the GitHub repo URL and confirmation.
**Returns:** GitHub repo URL, commit hash, push confirmation, any errors.

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
    envVars:
      - key: PYTHON_VERSION
        value: "3.11.0"
```

### 13b — Deploy Steps
1. Go to [render.com](https://render.com) → sign up / log in with GitHub
2. Click **New +** → **Web Service**
3. Connect GitHub repo: `<username>/<project-name>`
4. Render auto-detects `render.yaml` — confirm settings:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - **Instance Type:** Free
5. Click **Create Web Service** — Render builds and deploys automatically
6. Live URL: `https://<project-name>.onrender.com`

### 13c — Verify Deployment
```bash
curl https://<project-name>.onrender.com/health
curl -X POST https://<project-name>.onrender.com/predict \\
  -H "Content-Type: application/json" \\
  -d \'<replace with valid feature JSON from your dataset>\'
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
git add render.yaml docs/deployment_guide.md
git commit -m "Add render.yaml and deployment guide"
git push origin main
```
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
**fly.toml template:**
```toml
app = "<project-name>"
primary_region = "lax"
[env]
  PORT = "8000"
[http_service]
  internal_port = 8000
  force_https = true
  auto_stop_machines = true
  auto_start_machines = true
  min_machines_running = 0
```

#### 🚂 Railway
```bash
npm install -g @railway/cli && railway login
railway init && railway up
railway variables set PORT=8000
railway open
```
**railway.toml template:**
```toml
[build]
builder = "DOCKERFILE"
dockerfilePath = "Dockerfile"
[deploy]
startCommand = "uvicorn app:app --host 0.0.0.0 --port $PORT"
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 3
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
gcloud run services describe <project-name> --format "value(status.url)"
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
Replace `<LIVE_URL>` with your deployed service URL, and `<SAMPLE_PAYLOAD>` with a valid JSON object from your dataset.
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

### 14e — Create `docs/cloud_deployment_guide.md`
Delegate to the Documentation Agent to write `docs/cloud_deployment_guide.md` with:
- Platform comparison table (Step 14a)
- Prerequisites checklist
- Step-by-step instructions for the chosen platform
- Universal smoke-test commands with the live URL filled in
- Cost / free-tier notes for the chosen platform
- Teardown / cleanup commands to avoid unexpected charges

### 14f — Push to GitHub
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

# Large model binaries (tracked via Git LFS if needed)
# models/*.pkl   ← kept intentionally for reproducibility

# Logs
*.log
logs/

# ML template — per-run artifacts (not committed in template repo)
data/*.csv
models/*.pkl
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
# start.sh — use raw string to preserve ANSI codes and backslashes
FILES["start.sh"] = r'''#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
#  ML Pipeline Template — Interactive Setup Script
#  Usage: ./start.sh
# ──────────────────────────────────────────────────────────────────
set -e

# ── Colours ────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

# ── Banner ─────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}${BOLD}╔══════════════════════════════════════════════════╗${RESET}"
echo -e "${CYAN}${BOLD}║        🤖  ML Pipeline Template  v1.0.0          ║${RESET}"
echo -e "${CYAN}${BOLD}║   End-to-End Machine Learning Automation         ║${RESET}"
echo -e "${CYAN}${BOLD}╚══════════════════════════════════════════════════╝${RESET}"
echo ""

# ── Prerequisites Auto-Install ─────────────────────────────────────
echo -e "${BOLD}Checking prerequisites...${RESET}"
set +e  # allow failures during install

# 1. Homebrew (macOS)
if ! command -v brew &>/dev/null; then
    echo -e "${YELLOW}⚠  Homebrew not found — installing (follow the prompts)...${RESET}"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    [[ -f "/opt/homebrew/bin/brew" ]] && eval "$(/opt/homebrew/bin/brew shellenv)"
else
    echo -e "  ${GREEN}✔ Homebrew${RESET}"
fi

# 2. Node.js / npm
if ! command -v npm &>/dev/null; then
    echo -e "${YELLOW}⚠  Node.js not found — installing via Homebrew...${RESET}"
    brew install node
else
    echo -e "  ${GREEN}✔ Node.js $(node --version)${RESET}"
fi

# 3. Claude Code CLI
if ! command -v claude &>/dev/null; then
    echo -e "${YELLOW}⚠  Claude Code CLI not found — installing...${RESET}"
    if npm install -g @anthropic-ai/claude-code; then
        echo -e "  ${GREEN}✔ Claude Code CLI installed${RESET}"
    else
        echo -e "  ${RED}✗ Auto-install failed. Run manually:${RESET}"
        echo -e "    npm install -g @anthropic-ai/claude-code"
        echo -e "  Or visit: https://docs.anthropic.com/en/docs/claude-code/setup"
    fi
else
    echo -e "  ${GREEN}✔ Claude Code CLI $(claude --version 2>/dev/null | head -1)${RESET}"
fi

set -e  # restore exit-on-error
echo ""

# ── Step 1: Choose entry point ─────────────────────────────────────
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

LAUNCH_CLAUDE=false

if [ "$ENTRY_MODE" = "3" ]; then
    LAUNCH_CLAUDE=true
    echo ""
    echo -e "${GREEN}▶ Claude Code mode — setting up your project first...${RESET}"
    echo ""
fi

# ── Step 2: Shell mode — collect project info ──────────────────────
echo ""
echo -e "${BOLD}── Project Setup ────────────────────────────────────${RESET}"

read -rp "Project name (default: ml-project): " PROJECT_NAME
PROJECT_NAME="${PROJECT_NAME:-ml-project}"
PROJECT_NAME="${PROJECT_NAME// /-}"   # replace spaces with hyphens

echo ""
read -rp "Dataset CSV path (press Enter to copy manually later): " DATASET_PATH
DATASET_FILENAME=""
if [ -n "$DATASET_PATH" ]; then
    if [ -f "$DATASET_PATH" ]; then
        DATASET_FILENAME=$(basename "$DATASET_PATH")
        echo -e "  ${GREEN}✔ Found: $DATASET_FILENAME${RESET}"
    else
        echo -e "  ${YELLOW}⚠ File not found — you can copy it to data/ later.${RESET}"
        DATASET_PATH=""
    fi
fi

echo ""
echo -e "${BOLD}Deployment platform (applied at end of pipeline):${RESET}"
echo "  1) Ask me later"
echo "  2) Render        (free tier, recommended)"
echo "  3) Fly.io"
echo "  4) Railway"
echo "  5) AWS App Runner"
echo "  6) GCP Cloud Run"
echo "  7) Azure Container Apps"
echo "  8) Skip (local / Docker only)"
read -rp "Enter choice [1-8] (default: 1): " DEPLOY_CHOICE
DEPLOY_CHOICE="${DEPLOY_CHOICE:-1}"

case "$DEPLOY_CHOICE" in
  2) PLATFORM="render" ;;
  3) PLATFORM="fly.io" ;;
  4) PLATFORM="railway" ;;
  5) PLATFORM="aws" ;;
  6) PLATFORM="gcp" ;;
  7) PLATFORM="azure" ;;
  8) PLATFORM="none" ;;
  *) PLATFORM="ask_later" ;;
esac

echo ""
echo -e "${BOLD}GitHub setup:${RESET}"

# Auto-detect logged-in GitHub username from gh CLI
GH_DETECTED=$(gh api user --jq '.login' 2>/dev/null || echo "")
if [ -n "$GH_DETECTED" ]; then
    echo -e "  ${GREEN}✔ GitHub account detected: ${GH_DETECTED}${RESET}"
    read -rp "  GitHub username (press Enter to use '${GH_DETECTED}'): " GH_USER
    GH_USER="${GH_USER:-$GH_DETECTED}"
else
    read -rp "  GitHub username (press Enter to skip GitHub setup): " GH_USER
fi

# Repo name — defaults to project name (no timestamp)
if [ -n "$GH_USER" ]; then
    read -rp "  GitHub repo name (default: ${PROJECT_NAME}): " GH_REPO
    GH_REPO="${GH_REPO:-$PROJECT_NAME}"
    GH_REPO="${GH_REPO// /-}"   # replace spaces with hyphens

    echo ""
    echo -e "${BOLD}  GitHub repo visibility:${RESET}"
    echo "    1) Public"
    echo "    2) Private"
    read -rp "  Enter choice [1/2] (default: 1): " GH_CHOICE
    GH_CHOICE="${GH_CHOICE:-1}"
    case "$GH_CHOICE" in
      2) GH_VIS="private" ;;
      *) GH_VIS="public" ;;
    esac
else
    GH_REPO=""
    GH_VIS="skip"
    echo -e "  ${YELLOW}⚠ No GitHub username provided — skipping GitHub setup.${RESET}"
fi

# ── Step 3: Resolve paths ──────────────────────────────────────────
TEMPLATE_DIR="$(cd "$(dirname "$0")" && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROJECT_DIR="$(dirname "$TEMPLATE_DIR")/${PROJECT_NAME}_${TIMESTAMP}"

# ── Step 4: Stage in the SAME directory as the final project ──────────
# /tmp is on a different APFS volume — mv from /tmp is a cross-volume COPY
# (VS Code sees folder appear empty, then files added one by one).
# Same-parent staging means mv is an atomic RENAME — VS Code sees the
# project folder appear ONCE, fully complete.
STAGING_DIR="$(dirname "$PROJECT_DIR")/.ml_staging_${PROJECT_NAME}_${TIMESTAMP}"
STAGING_DIR_SET=true

cleanup_staging() {
    if [ "${STAGING_DIR_SET:-false}" = "true" ] && [ -d "$STAGING_DIR" ]; then
        rm -rf "$STAGING_DIR" 2>/dev/null
    fi
}
trap cleanup_staging EXIT

mkdir -p "$STAGING_DIR"

# ── Step 5: Copy template files into staging ───────────────────────
echo ""
echo -e "${GREEN}▶ Preparing project files...${RESET}"
rsync -a \
    --exclude='.git/' \
    --exclude='data/*.csv' \
    --exclude='models/*.pkl' \
    --exclude='models/*.npy' \
    --exclude='plots/*.png' \
    --exclude='.venv/' \
    --exclude='__pycache__/' \
    --exclude='.DS_Store' \
    --exclude='.ml_config.json' \
    --exclude='bootstrap.py' \
    --exclude='Dockerfile.bootstrap' \
    --exclude='start.sh' \
    --exclude='init.py' \
    "$TEMPLATE_DIR/" "$STAGING_DIR/"
echo -e "  ${GREEN}✔ Template files ready${RESET}"

# ── Step 6: Copy dataset if provided ──────────────────────────────
if [ -n "$DATASET_PATH" ] && [ -f "$DATASET_PATH" ]; then
    cp "$DATASET_PATH" "$STAGING_DIR/data/"
    echo -e "  ${GREEN}✔ Dataset staged: $DATASET_FILENAME${RESET}"
fi

# ── Step 7: Write .ml_config.json into staging ────────────────────
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

# ── Step 8: Create venv inside staging ────────────────────────────
echo -e "${GREEN}▶ Creating Python virtual environment (.venv)...${RESET}"
python3 -m venv "$STAGING_DIR/.venv"
echo -e "  ${GREEN}✔ Virtual environment created${RESET}"

# ── Step 9: Install dependencies inside staging ───────────────────
echo -e "${GREEN}▶ Installing dependencies (this may take a minute)...${RESET}"
"$STAGING_DIR/.venv/bin/pip" install --upgrade pip -q
"$STAGING_DIR/.venv/bin/pip" install -r "$STAGING_DIR/requirements.txt" -q
echo -e "  ${GREEN}✔ Dependencies installed${RESET}"

# ── Step 10: Move staging → final (one atomic operation) ──────────
# VS Code sees the project folder appear ONCE — complete, .venv inside.
echo -e "${GREEN}▶ Creating project at: $PROJECT_DIR${RESET}"
mv "$STAGING_DIR" "$PROJECT_DIR"
STAGING_DIR_SET=false

# Fix .venv script shebangs after path changed from /tmp to final location
python3 -m venv --upgrade "$PROJECT_DIR/.venv" 2>/dev/null || \
    "$PROJECT_DIR/.venv/bin/python" -m pip install pip -q 2>/dev/null || true

# ── Step 11: Completion summary ────────────────────────────────────
echo ""
echo -e "${CYAN}${BOLD}╔══════════════════════════════════════════════════╗${RESET}"
echo -e "${CYAN}${BOLD}║  ✅  Project ready!                              ║${RESET}"
echo -e "${CYAN}${BOLD}╠══════════════════════════════════════════════════╣${RESET}"
printf "${CYAN}${BOLD}║${RESET}  📁  %-44s${CYAN}${BOLD}║${RESET}\n" "$PROJECT_DIR"
printf "${CYAN}${BOLD}║${RESET}  🐍  Venv   : .venv/                            ${CYAN}${BOLD}║${RESET}\n"
printf "${CYAN}${BOLD}║${RESET}  📊  Data   : ${DATASET_FILENAME_SAFE}$(printf '%*s' $((34 - ${#DATASET_FILENAME_SAFE})) '')${CYAN}${BOLD}║${RESET}\n"
printf "${CYAN}${BOLD}║${RESET}  🚀  Deploy : %-34s${CYAN}${BOLD}║${RESET}\n" "$PLATFORM"
if [ -n "$GH_USER" ]; then
    printf "${CYAN}${BOLD}║${RESET}  🐙  GitHub : %-34s${CYAN}${BOLD}║${RESET}\n" "github.com/${GH_USER}/${GH_REPO:-$PROJECT_NAME}"
fi
echo -e "${CYAN}${BOLD}╠══════════════════════════════════════════════════╣${RESET}"
echo -e "${CYAN}${BOLD}║  ✅  Launching Claude Code...                    ║${RESET}"
echo -e "${CYAN}${BOLD}╚══════════════════════════════════════════════════╝${RESET}"
echo ""

# ── Step 12: Launch Claude Code ────────────────────────────────────
echo -e "${GREEN}▶ Launching Claude Code in your new project...${RESET}"
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
# init.py — use raw string to preserve ANSI codes; use \'\'\' for docstrings
FILES["init.py"] = r'''#!/usr/bin/env python3
"""
init.py — ML Pipeline Template: Python CLI Setup
Usage: python3 init.py
Requires: Python 3.9+ stdlib only (runs before venv is active)
"""

import os, sys, json, shutil, subprocess
from pathlib import Path
from datetime import datetime, timezone

# ── Colours ────────────────────────────────────────────────────────
G = "\033[0;32m"; Y = "\033[1;33m"; C = "\033[0;36m"
B = "\033[1m";    R = "\033[0;31m"; X = "\033[0m"

PLATFORMS = {
    "1": "ask_later", "2": "render",  "3": "fly.io",
    "4": "railway",   "5": "aws",     "6": "gcp",
    "7": "azure",     "8": "none",
}

PLATFORM_LABELS = {
    "ask_later": "Ask me later",
    "render":    "Render (free tier)",
    "fly.io":    "Fly.io",
    "railway":   "Railway",
    "aws":       "AWS App Runner",
    "gcp":       "GCP Cloud Run",
    "azure":     "Azure Container Apps",
    "none":      "Skip (local / Docker only)",
}

def prompt(msg: str, default: str = "") -> str:
    suffix = f" (default: {default})" if default else ""
    val = input(f"{msg}{suffix}: ").strip()
    return val or default

def menu(title: str, options: list[tuple[str, str]], default: str = "1") -> str:
    print(f"\n{B}{title}{X}")
    for key, label in options:
        print(f"  {key}) {label}")
    choice = input(f"Enter choice [{'/'.join(k for k,_ in options)}] (default: {default}): ").strip()
    return choice or default

def banner():
    print(f"""
{C}{B}╔══════════════════════════════════════════════════╗
║        🤖  ML Pipeline Template  v1.0.0          ║
║   End-to-End Machine Learning Automation         ║
╚══════════════════════════════════════════════════╝{X}
""")

def mode_select() -> str:
    choice = menu(
        "How would you like to run this template?",
        [
            ("1", "Shell script  — guided prompts here in the terminal"),
            ("2", "Python CLI    — richer prompts via init.py  ← you are here"),
            ("3", "Claude Code   — AI-driven, fully automated (recommended)"),
        ],
        default="3",
    )
    return choice

def collect_inputs() -> dict:
    print(f"\n{B}── Project Setup ──────────────────────────────────────{X}")

    project_name = prompt("Project name", "ml-project").replace(" ", "-")

    # Dataset
    dataset_path = ""
    dataset_filename = ""
    raw = prompt("Dataset CSV path (press Enter to provide manually later)", "")
    if raw:
        p = Path(raw).expanduser().resolve()
        if p.is_file():
            dataset_path = str(p)
            dataset_filename = p.name
            print(f"  {G}✔ Found: {dataset_filename}{X}")
        else:
            print(f"  {Y}⚠ File not found — you can copy it to data/ later.{X}")

    # Deployment
    deploy_choice = menu(
        "Deployment platform (applied at end of pipeline):",
        [
            ("1", "Ask me later"),
            ("2", "Render        (free tier, recommended)"),
            ("3", "Fly.io"),
            ("4", "Railway"),
            ("5", "AWS App Runner"),
            ("6", "GCP Cloud Run"),
            ("7", "Azure Container Apps"),
            ("8", "Skip (local / Docker only)"),
        ],
        default="1",
    )
    platform = PLATFORMS.get(deploy_choice, "ask_later")

    # GitHub username — auto-detect from gh CLI if available
    print(f"\n{B}GitHub setup:{X}")
    gh_detected = ""
    try:
        result = subprocess.run(
            ["gh", "api", "user", "--jq", ".login"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            gh_detected = result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    if gh_detected:
        print(f"  {G}✔ GitHub account detected: {gh_detected}{X}")
        gh_user = prompt(f"  GitHub username (press Enter to use '{gh_detected}')", gh_detected)
    else:
        gh_user = prompt("  GitHub username (press Enter to skip GitHub setup)", "")

    # Repo name — defaults to project name (no timestamp)
    gh_repo = ""
    gh_vis  = "skip"
    if gh_user:
        gh_repo = prompt(f"  GitHub repo name", project_name).replace(" ", "-")
        gh_choice = menu(
            "  GitHub repo visibility:",
            [("1", "Public"), ("2", "Private")],
            default="1",
        )
        gh_vis = "private" if gh_choice == "2" else "public"
    else:
        print(f"  {Y}⚠ No GitHub username provided — skipping GitHub setup.{X}")

    return {
        "project_name":      project_name,
        "dataset_path":      dataset_path,
        "dataset_filename":  dataset_filename,
        "platform":          platform,
        "github_username":   gh_user,
        "github_repo":       gh_repo or project_name,
        "github_visibility": gh_vis,
    }

def _make_staging_dir(project_dir: Path) -> Path:
    """
    Return a staging path in the SAME parent directory as project_dir.

    /tmp is on a different APFS volume — shutil.move from /tmp is a cross-volume
    COPY (VS Code sees folder appear empty, then files added one by one).
    Same-parent staging means shutil.move is an atomic RENAME on the same volume
    → VS Code sees the project folder appear ONCE, fully complete.
    """
    staging = project_dir.parent / f".ml_staging_{project_dir.name}"
    staging.mkdir(parents=True, exist_ok=True)
    return staging

def create_project(cfg: dict) -> tuple:
    """
    Return (project_dir, staging_dir).
    All work happens in staging_dir (same parent dir as project_dir);
    a single shutil.move() renames it atomically — VS Code sees the
    project appear exactly once, complete and ready.
    """
    template_dir = Path(__file__).parent.resolve()
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_dir  = template_dir.parent / f"{cfg['project_name']}_{timestamp}"
    staging_dir  = _make_staging_dir(project_dir)
    return project_dir, staging_dir

def create_venv(staging_dir: Path):
    """Create .venv inside the staging dir (invisible to VS Code)."""
    print(f"{G}▶ Creating Python virtual environment (.venv)...{X}")
    subprocess.run([sys.executable, "-m", "venv", str(staging_dir / ".venv")], check=True)
    print(f"  {G}✔ Virtual environment created{X}")

def install_deps(staging_dir: Path):
    """Install all deps into the staging venv (invisible to VS Code)."""
    print(f"{G}▶ Installing dependencies (this may take a minute)...{X}")
    pip = str(staging_dir / ".venv" / "bin" / "pip")
    subprocess.run([pip, "install", "--upgrade", "pip", "-q"], check=True)
    req = staging_dir / "requirements.txt"
    if req.exists():
        subprocess.run([pip, "install", "-r", str(req), "-q"], check=True)
    print(f"  {G}✔ Dependencies installed{X}")

def move_to_final(staging_dir: Path, project_dir: Path):
    """
    Atomic move: staging → project_dir (same filesystem → os.rename, not copy).
    VS Code sees the project appear ONCE — fully populated, .venv already inside.
    Then fix .venv script shebangs (path changed from staging dir to project_dir).
    """
    print(f"\n{G}▶ Creating project at: {project_dir}{X}")
    shutil.move(str(staging_dir), str(project_dir))
    # Fix shebang paths in .venv/bin/pip etc. after the directory was renamed
    venv = project_dir / ".venv"
    try:
        subprocess.run(
            [sys.executable, "-m", "venv", "--upgrade", str(venv)],
            check=True, capture_output=True
        )
    except Exception:
        # Fallback: reinstall pip to fix shebang (fast, no package reinstall)
        python = str(venv / "bin" / "python")
        subprocess.run([python, "-m", "pip", "install", "pip", "-q"],
                       check=False, capture_output=True)

EXCLUDE = {
    ".git", "__pycache__", ".venv", ".DS_Store",
    ".ml_config.json",
    # Setup scripts: stay in the template folder, never copied into projects.
    # Running bootstrap.py inside a project creates a nested ml-pipeline-template/;
    # running start.sh/init.py creates sibling projects — both are duplication bugs.
    "bootstrap.py", "Dockerfile.bootstrap", "start.sh", "init.py",
}
EXCLUDE_EXTS = {".csv", ".pkl", ".npy", ".png", ".pyc"}

def _ignore(src: str, names: list[str]) -> set[str]:
    ignored = set()
    for name in names:
        full = Path(src) / name
        if name in EXCLUDE:
            ignored.add(name)
        elif full.suffix in EXCLUDE_EXTS:
            ignored.add(name)
    return ignored

def copy_template(template_dir: Path, staging_dir: Path):
    print(f"{G}▶ Preparing template files...{X}")
    shutil.copytree(str(template_dir), str(staging_dir), ignore=_ignore, dirs_exist_ok=True)
    print(f"  {G}✔ Template files ready{X}")

def copy_dataset(cfg: dict, staging_dir: Path):
    if cfg["dataset_path"] and Path(cfg["dataset_path"]).is_file():
        dest = staging_dir / "data"
        dest.mkdir(exist_ok=True)
        shutil.copy2(cfg["dataset_path"], dest / cfg["dataset_filename"])
        print(f"  {G}✔ Dataset copied: {cfg['dataset_filename']}{X}")

def write_config(cfg: dict, staging_dir: Path):
    """Write .ml_config.json into staging dir (moved to project later)."""
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    gh_user = cfg.get("github_username", "")
    gh_repo = cfg.get("github_repo", cfg["project_name"])
    config = {
        "project_name":      cfg["project_name"],
        "dataset_filename":  cfg["dataset_filename"] or "<not provided yet>",
        "dataset_path":      f"data/{cfg['dataset_filename']}" if cfg["dataset_filename"] else "<not provided yet>",
        "target_column":     "auto-detect",
        "task_type":         "auto-detect",
        "deployment_platform": cfg["platform"],
        "github_username":   gh_user,
        "github_repo":       gh_repo,
        "github_visibility": cfg["github_visibility"],
        "github_url":        f"https://github.com/{gh_user}/{gh_repo}" if gh_user else "",
        "python_version":    py_ver,
        "created_at":        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "venv_path":         ".venv",
        "template_version":  "1.0.0",
    }
    (staging_dir / ".ml_config.json").write_text(json.dumps(config, indent=2))
    print(f"  {G}✔ .ml_config.json ready{X}")

def show_summary(cfg: dict, project_dir: Path):
    fn   = cfg["dataset_filename"] or "<not provided yet>"
    plat = PLATFORM_LABELS.get(cfg["platform"], cfg["platform"])
    gh_user = cfg.get("github_username", "")
    gh_repo = cfg.get("github_repo", cfg["project_name"])
    gh_line = f"\n{C}{B}║{X}  🐙  GitHub : github.com/{gh_user}/{gh_repo}" if gh_user else ""
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

def maybe_open_claude(project_dir: Path):
    print(f"{G}▶ Launching Claude Code in your new project...{X}")
    os.chdir(project_dir)
    if shutil.which("claude"):
        subprocess.run(["claude", "."])
    else:
        print(f"{Y}Claude Code CLI not found. Install: npm install -g @anthropic-ai/claude-code{X}")
        print(f"Then run: {B}cd {project_dir} && source .venv/bin/activate && claude .{X}")


if __name__ == "__main__":
    banner()
    choice = mode_select()

    if choice == "1":
        script = Path(__file__).parent / "start.sh"
        os.execv("/bin/bash", ["/bin/bash", str(script)])

    # choices "2" and "3" both go through full project setup + launch
    cfg                      = collect_inputs()
    project_dir, staging_dir = create_project(cfg)

    # All heavy work happens in hidden staging dir (same parent as project, invisible to VS Code)
    copy_template(Path(__file__).parent.resolve(), staging_dir)  # 1. template files
    copy_dataset(cfg, staging_dir)                                # 2. dataset
    write_config(cfg, staging_dir)                                # 3. config
    create_venv(staging_dir)                                      # 4. .venv
    install_deps(staging_dir)                                     # 5. pip install

    # One atomic move → VS Code sees project appear once, complete
    move_to_final(staging_dir, project_dir)

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

## Prerequisites

Only **Python 3.9+** must be installed manually — everything else is handled automatically.

| Tool | How |
|---|---|
| Python 3.9+ | Manual — [python.org](https://python.org) |
| Homebrew | **Auto-installed** by `./start.sh` or `bootstrap.py` |
| Node.js | **Auto-installed** by `./start.sh` or `bootstrap.py` |
| Claude Code CLI | **Auto-installed** by `./start.sh` or `bootstrap.py` |
| GitHub CLI *(optional)* | `brew install gh` then `gh auth login` |
| Docker *(optional)* | [docker.com](https://docker.com) |

---

## Step 1 — Get the Template

Choose any one method:

### 🔥 Bootstrap (no git, no clone required)
```bash
curl -O https://raw.githubusercontent.com/ramleo/ml-pipeline-template/main/bootstrap.py
python3 bootstrap.py
cd ml-pipeline-template
```

### 🐳 Via Docker (nothing to install except Docker)
```bash
docker build -t ml-pipeline-template -f Dockerfile.bootstrap .
docker run --rm -v $(pwd):/output ml-pipeline-template
cd ml-pipeline-template
```

### 📦 Git Clone
```bash
git clone https://github.com/ramleo/ml-pipeline-template
cd ml-pipeline-template
```

---

## Step 2 — Run It

```bash
./start.sh
```

Auto-installs any missing prerequisites, then shows a menu:

```
  1) Shell script  — guided terminal prompts
  2) Python CLI    — richer prompts via init.py
  3) Claude Code   — AI-driven, fully automated (recommended)
```

All three options work the same way: answer a few terminal prompts (project name, CSV path, platform, GitHub), then the script creates a new project folder, sets up the Python environment with all dependencies installed, and launches Claude Code automatically.

Choose **3** (or press Enter — it is the default).

> 📖 Full details: [docs/how_to_run.md](docs/how_to_run.md)

---

## What Gets Created

```
my-project_20260524_143000/
├── .venv/                      ← isolated Python environment
├── .ml_config.json             ← your choices (dataset, platform, etc.)
├── .gitignore                  ← Python / macOS / IDE / secrets
├── data/                       ← your CSV goes here
├── models/                     ← trained pipeline artifacts (.pkl)
├── plots/                      ← EDA charts (.png)
├── src/preprocess.py           ← generated preprocessing script
├── tests/test_pipeline.py      ← generated test suite
├── docs/                       ← summary, guides, test results
├── app.py                      ← FastAPI app
├── Dockerfile                  ← multi-stage build
├── requirements.txt            ← pinned dependencies
└── render.yaml / fly.toml /    ← deployment config (platform-specific)
    railway.toml / apprunner.yaml
```

---

## Supported Deployment Platforms

| Platform | Free Tier | Config File | CLI |
|---|---|---|---|
| Render | ✅ | `render.yaml` | — |
| Fly.io | ✅ | `fly.toml` | `flyctl` |
| Railway | ✅ | `railway.toml` | `railway` |
| AWS App Runner | ✅ (free tier) | `apprunner.yaml` | `aws` |
| GCP Cloud Run | ✅ (free tier) | — | `gcloud` |
| Azure Container Apps | ✅ (free tier) | — | `az` |

---

## ML Tasks Supported

| Task | Target Column | Metrics |
|---|---|---|
| Classification | Categorical / ≤ 20 unique values | Accuracy, F1, Classification Report |
| Regression | Numeric / > 20 unique values | RMSE, MAE, R² |

Task type is **auto-detected** from your target column — no config needed.

---

## Template File Reference

| File | Purpose |
|---|---|
| `CLAUDE.md` | Root agent instructions (always loaded) |
| `src/CLAUDE.md` | EDA, preprocessing, training agent specs |
| `tests/CLAUDE.md` | Testing agent spec |
| `docs/CLAUDE.md` | Documentation agent spec |
| `deploy/CLAUDE.md` | Docker, Git, cloud deploy agent specs |
| `deploy/cloud.md` | Cloud deployment index (`@`-imports render + platforms) |
| `deploy/cloud-render.md` | Render deployment steps (Step 13) |
| `deploy/cloud-platforms.md` | AWS / GCP / Azure / Fly.io / Railway steps (Step 14) |
| `start.sh` | Bash entry point |
| `init.py` | Python CLI entry point |
| `bootstrap.py` | Single-file installer (no git required) |
| `Dockerfile.bootstrap` | Docker image for distributing the template |
| `.ml_config.json.example` | Reference config template |
| `.gitignore` | Standard Python / macOS / IDE ignore rules |
| `docs/claude_structure.md` | CLAUDE.md split structure reference |
| `docs/how_to_run.md` | Step-by-step usage guide |

---

## Customisation

Edit the local `CLAUDE.md` files in subdirectories to change agent behaviour:
- **Add candidate models** → `src/CLAUDE.md` (Optimization Agent section)
- **Change preprocessing** → `src/CLAUDE.md` (Data Engineering Agent section)
- **Add API endpoints** → `src/CLAUDE.md` (FastAPI Agent section)
- **Change deploy target** → `deploy/CLAUDE.md`

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

---

## Prerequisites

Only **Python 3.9+** must be installed manually — everything else is handled automatically.

| Tool | How |
|---|---|
| Python 3.9+ | Manual — [python.org](https://python.org) |
| Homebrew | **Auto-installed** by `./start.sh` or `bootstrap.py` |
| Node.js | **Auto-installed** by `./start.sh` or `bootstrap.py` |
| Claude Code CLI | **Auto-installed** by `./start.sh` or `bootstrap.py` |
| GitHub CLI *(optional)* | `brew install gh` then `gh auth login` |
| Docker *(optional)* | [docker.com](https://docker.com) |

---

## Step 1 — Check Python

```bash
python3 --version
```

If you get `command not found`, install Python 3.9+ from [python.org](https://python.org) before continuing.

---

## Step 2 — Download the bootstrap script

```bash
curl -O https://raw.githubusercontent.com/ramleo/ml-pipeline-template/main/bootstrap.py
```

This downloads a single installer file — no git, no GitHub account required.

---

## Step 3 — Run the bootstrap

```bash
python3 bootstrap.py
```

This will:
- Create a `ml-pipeline-template/` folder with all template files
- Auto-install **Homebrew**, **Node.js**, and **Claude Code CLI** if they\'re missing

---

## Step 4 — Enter the template folder

```bash
cd ml-pipeline-template
```

---

## Step 5 — Start the wizard

```bash
./start.sh
```

The script checks prerequisites (installs anything still missing), then shows a menu:

```
How would you like to run this template?
  1) Shell script  — guided prompts here in the terminal
  2) Python CLI    — richer prompts via init.py
  3) Claude Code   — AI-driven, fully automated (recommended)
```

All three options do the same thing — they collect your project details, create a new project folder, set up the Python environment, and launch Claude Code automatically. The only difference is the style of prompts (shell, Python, or Claude\'s conversation interface).

Choose **3** (or press Enter — it is the default).

---

## Step 6 — Answer the setup prompts

After choosing an option, you will be asked in the **terminal** (before Claude launches):

| Prompt | Example answer |
|---|---|
| Project name | `titanic-predictor` |
| Dataset CSV path | `/Users/yourname/Downloads/titanic.csv` |
| Deployment platform | `2` for Render, or `1` to decide later |
| GitHub username | `your-github-username` (press Enter to skip) |
| GitHub repo name | `titanic-predictor` (defaults to project name) |
| Repo visibility | `1` for Public, `2` for Private |

### When to add your CSV file

**You do not need to move your CSV anywhere beforehand.** Just have it somewhere on your computer and know its full path.

Type the full path when the terminal asks — the script copies it into the project\'s `data/` folder automatically.

| Situation | What to do |
|---|---|
| File is anywhere on your computer | Type its full path, e.g. `/Users/yourname/Downloads/mydata.csv` |
| File is not ready yet | Press Enter to skip — drop the CSV into `data/` later and tell Claude the filename |

---

## Step 7 — Project is created automatically

After you answer the prompts, the script:

1. Creates a new project folder (e.g. `../titanic-predictor_20260524_143000/`)
2. Copies all template files into it
3. Copies your CSV into `data/`
4. Writes `.ml_config.json` with all your choices
5. Creates a Python virtual environment (`.venv/`)
6. Installs all dependencies (`pip install -r requirements.txt`)
7. Launches Claude Code automatically

---

## Step 8 — Claude runs the pipeline

Claude reads `.ml_config.json`, shows you a confirmation summary, and waits for your approval:

```
Dataset   : data/titanic.csv
Target    : Survived
Task      : Classification
Platform  : render
GitHub    : https://github.com/yourname/titanic-predictor

Proceed with the pipeline? [Y/n]
```

Press **Enter** (or Y) and Claude works through the full checklist:

| Step | Task | Output |
|---|---|---|
| 0 | Verify Python environment | `.venv/` (already set up) |
| 1 | Scan workspace, find CSV | — |
| 2 | EDA — profile data, plot charts | `plots/` |
| 3 | Preprocessing — clean & encode | `src/preprocess.py` |
| 4–6 | Train, tune & evaluate models | metrics report |
| 7 | Save final pipeline | `models/final_pipeline.pkl` |
| 8 | Write summary report | `docs/summary.md` |
| 9 | Pin dependencies | `requirements.txt` |
| 10 | Reorganise workspace | clean folder structure |
| 11 | Git init → GitHub repo → push | GitHub URL |
| 12 | Dockerfile → build → smoke-test | Docker image |
| 13 | Deploy to chosen cloud platform | live URL |

---

## Folder layout after the pipeline

```
titanic-predictor_20260524_143000/
├── .venv/                  ← Python virtual environment (pre-installed)
├── .ml_config.json         ← your choices (dataset, platform, GitHub)
├── data/                   ← your CSV file
├── models/                 ← trained pipeline artifacts (.pkl)
├── plots/                  ← EDA charts (.png)
├── src/preprocess.py       ← auto-generated preprocessing script
├── tests/test_pipeline.py  ← auto-generated test suite
├── docs/                   ← summary, guides, test results
├── app.py                  ← FastAPI prediction API
├── Dockerfile              ← multi-stage container build
├── requirements.txt        ← pinned library versions
└── render.yaml / fly.toml  ← deployment config
```

> The template folder is **never modified**. Each run produces a fresh, isolated project.

---

## Alternative: get the template via Docker

```bash
docker build -t ml-pipeline-template -f Dockerfile.bootstrap \\
  https://raw.githubusercontent.com/ramleo/ml-pipeline-template/main/Dockerfile.bootstrap

docker run --rm -v $(pwd):/output ml-pipeline-template

cd ml-pipeline-template
./start.sh
```

---

## Alternative: get the template via git clone

```bash
git clone https://github.com/ramleo/ml-pipeline-template
cd ml-pipeline-template
./start.sh
```

---

## Quick reference

```bash
python3 --version          # 1. confirm Python is installed
curl -O <bootstrap_url>    # 2. download installer
python3 bootstrap.py       # 3. create template + auto-install tools
cd ml-pipeline-template    # 4. enter folder
./start.sh                 # 5. answer prompts → project created → Claude launches
```

*(Replace `<bootstrap_url>` with `https://raw.githubusercontent.com/ramleo/ml-pipeline-template/main/bootstrap.py`)*

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `python3: command not found` | Install Python 3.9+ from [python.org](https://python.org) |
| `claude: command not found` | Run `./start.sh` — it auto-installs, or manually: `npm install -g @anthropic-ai/claude-code` |
| `Permission denied: ./start.sh` | Run `chmod +x start.sh` first |
| Dataset not found | Copy your `.csv` into the project\'s `data/` folder, then tell Claude its name |
| `ml-pipeline-template/` already exists | Run `python3 bootstrap.py my-new-name` to use a different folder name |
| Homebrew install hangs | Accept the Xcode Command Line Tools prompt that appears |
| pip install fails | Check Python version (`python3 --version`) — requires 3.9+ |
'''

# ════════════════════════════════════════════════════════════════════
FILES["auto_pipeline.py"] = r'''#!/usr/bin/env python3
"""
auto_pipeline.py — Automated ML Pipeline (no Claude/AI required)
Usage: python3 auto_pipeline.py
       .venv/bin/python auto_pipeline.py
"""

import json
import os
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
    print(f"\n{C}{B}{'═' * width}{X}")
    print(f"{C}{B}  {text}{X}")
    print(f"{C}{B}{'═' * width}{X}")


def _ok(msg: str) -> None:
    print(f"  {G}✔  {msg}{X}")


def _warn(msg: str) -> None:
    print(f"  {Y}⚠  {msg}{X}")


def _err(msg: str) -> None:
    print(f"  {R}✗  {msg}{X}")


def _info(msg: str) -> None:
    print(f"  {C}→  {msg}{X}")


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
csv_path: Path | None = None
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

# ── Auto-detect target column ────────────────────────────────────────
if target_col_cfg and target_col_cfg in df.columns:
    target_col = target_col_cfg
    _ok(f"Target column (from config): {B}{target_col}{X}")
else:
    # Heuristic: last column
    target_col = df.columns[-1]
    _warn(f"Target column not specified; guessing last column: {B}{target_col}{X}")

# ── Auto-detect task type ────────────────────────────────────────────
n_unique = df[target_col].nunique()
if df[target_col].dtype in (object, bool, "bool") or n_unique <= 20:
    task_type = "classification"
else:
    task_type = "regression"
_ok(f"Task type: {B}{task_type}{X}  (unique target values: {n_unique})")

# ── Basic EDA printout ───────────────────────────────────────────────
print(f"\n{B}Column dtypes:{X}")
print(df.dtypes.to_string())

missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(1)
missing_df = pd.DataFrame({"missing": missing, "pct": missing_pct})
missing_with = missing_df[missing_df["missing"] > 0]
if not missing_with.empty:
    print(f"\n{B}Missing values:{X}")
    print(missing_with.to_string())
else:
    _ok("No missing values")

if task_type == "classification":
    print(f"\n{B}Class balance ({target_col}):{X}")
    vc = df[target_col].value_counts()
    print(vc.to_string())
else:
    print(f"\n{B}Target distribution ({target_col}):{X}")
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
label_encoder: LabelEncoder | None = None
if task_type == "classification":
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y.astype(str))
    _ok(f"Target classes: {list(label_encoder.classes_)}")
else:
    y_enc = y.values.astype(float)

# ── Train / test split ───────────────────────────────────────────────
split_kwargs: dict = {"test_size": 0.2, "random_state": 42}
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

results: list[dict] = []

for name, estimator, param_grid in candidates:
    try:
        _info(f"Training {B}{name}{X} with GridSearchCV(cv=3)...")
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
print(f"\n  {G}{B}Best model: {best_name}{X}")
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

metrics: dict = {}

if task_type == "classification":
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=[str(c) for c in label_encoder.classes_] if label_encoder else None,
    )
    metrics["accuracy"] = acc
    metrics["classification_report"] = report
    print(f"\n  {B}Test Accuracy: {G}{acc:.4f}{X}")
    print(f"\n{B}Classification Report:{X}")
    print(report)
else:
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))
    metrics["rmse"] = rmse
    metrics["r2"] = r2
    print(f"\n  {B}Test RMSE : {G}{rmse:.4f}{X}")
    print(f"  {B}Test R²   : {G}{r2:.4f}{X}")

# ════════════════════════════════════════════════════════════════════
# 6.  Save Artifacts
# ════════════════════════════════════════════════════════════════════

_print_header("Step 6 — Saving Artifacts")

pipeline_path = MODELS_DIR / "final_pipeline.pkl"
joblib.dump(final_pipe, pipeline_path)
_ok(f"Final pipeline → {pipeline_path}")

le_path: Path | None = None
if label_encoder is not None:
    le_path = MODELS_DIR / "label_encoder.pkl"
    joblib.dump(label_encoder, le_path)
    _ok(f"Label encoder → {le_path}")

# ════════════════════════════════════════════════════════════════════
# 7.  Write docs/auto_summary.md
# ════════════════════════════════════════════════════════════════════

_print_header("Step 7 — Writing Summary Report")

from datetime import datetime

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
{"No missing values detected." if missing_with.empty else missing_with.to_markdown()}

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
{C}{B}╔══════════════════════════════════════════════════════╗
║  ✅  Pipeline Complete!                              ║
╠══════════════════════════════════════════════════════╣{X}
{C}{B}║{X}  Dataset    : {csv_path.name} ({df.shape[0]:,} rows × {df.shape[1]} cols)
{C}{B}║{X}  Task       : {task_type.title()}
{C}{B}║{X}  Best Model : {best_name}
{C}{B}║{X}  CV Score   : {best_cv:.4f}  ({scoring})""")

if task_type == "classification":
    print(f"{C}{B}║{X}  Accuracy   : {metrics['accuracy']:.4f}")
else:
    print(f"{C}{B}║{X}  RMSE       : {metrics['rmse']:.4f}")
    print(f"{C}{B}║{X}  R²         : {metrics['r2']:.4f}")

print(f"""{C}{B}╠══════════════════════════════════════════════════════╣{X}
{C}{B}║{X}  {G}models/final_pipeline.pkl{X}   ← ready to use
{C}{B}║{X}  {G}docs/auto_summary.md{X}        ← full report
{C}{B}║{X}  {G}plots/eda_correlation.png{X}   ← correlation heatmap
{C}{B}║{X}  {G}plots/eda_target.png{X}        ← target distribution
{C}{B}╚══════════════════════════════════════════════════════╝{X}
""")
'''

# ════════════════════════════════════════════════════════════════════
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

    # 1. Homebrew (macOS)
    if shutil.which("brew"):
        print(f"  {G}✔ Homebrew{X}")
    else:
        print(f"  {Y}⚠  Homebrew not found — installing (follow the prompts)...{X}")
        os.system('/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"')
        if Path("/opt/homebrew/bin/brew").exists():
            os.environ["PATH"] = "/opt/homebrew/bin:" + os.environ.get("PATH", "")

    # 2. Node.js / npm
    if shutil.which("npm"):
        r = _run(["node", "--version"])
        print(f"  {G}✔ Node.js {r.stdout.strip() if r else ''}{X}")
    else:
        print(f"  {Y}⚠  Node.js not found — installing via Homebrew...{X}")
        os.system("brew install node")

    # 3. Claude Code CLI
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
            print(f"  Or visit: https://docs.anthropic.com/en/docs/claude-code/setup")

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

# Fix .venv shebangs after path changed
try:
    subprocess.run([sys.executable, "-m", "venv", "--upgrade", str(project_dir / ".venv")],
                   check=True, capture_output=True)
except Exception:
    try:
        subprocess.run([str(project_dir / ".venv" / "bin" / "python"),
                        "-m", "pip", "install", "pip", "-q"],
                       check=False, capture_output=True)
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

# Launch — ask user preference
print(f"\n{B}How would you like to run the pipeline?{X}")
print(f"  1) {G}Claude Code{X}   — AI-driven, fully automated (recommended)")
print(f"  2) {C}Auto Pipeline{X} — no Claude subscription needed (pure sklearn)")
print(f"  3) {Y}Manual{X}        — I'll run it myself later")
launch_choice = input("Enter choice (default: 1): ").strip() or "1"

os.chdir(project_dir)

if launch_choice == "2":
    print(f"\n{G}▶ Running automated pipeline (no Claude required)...{X}")
    python = str(project_dir / ".venv" / "bin" / "python")
    subprocess.run([python, "auto_pipeline.py"])
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

```

---

## 4. start.sh Architecture

`start.sh` is the bash entry point for users who cloned the template repo. It runs FROM INSIDE the template folder and creates the new project as a SIBLING directory.

### 4.1 Key Structural Properties

- Uses `set -e` (exit on error) with `set +e` relaxed during optional installs
- Same prerequisite checks as `bootstrap.py`: Homebrew, Node.js, Claude Code CLI
- Shows a 3-option entry mode menu first: Shell / Python CLI / Claude Code
- If option 2 is chosen: `exec python3 "$(dirname "$0")/init.py"` (hand off completely)
- Options 1 and 3 both proceed through the same shell prompts (1 = shell only, 3 = shell + claude launch)

### 4.2 Path Resolution

```bash
TEMPLATE_DIR="$(cd "$(dirname "$0")" && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROJECT_DIR="$(dirname "$TEMPLATE_DIR")/${PROJECT_NAME}_${TIMESTAMP}"
```

This places the project as a sibling of the template folder, not inside it.

### 4.3 APFS-Safe Staging in Bash

```bash
STAGING_DIR="$(dirname "$PROJECT_DIR")/.ml_staging_${PROJECT_NAME}_${TIMESTAMP}"
STAGING_DIR_SET=true

cleanup_staging() {
    if [ "${STAGING_DIR_SET:-false}" = "true" ] && [ -d "$STAGING_DIR" ]; then
        rm -rf "$STAGING_DIR" 2>/dev/null
    fi
}
trap cleanup_staging EXIT

mkdir -p "$STAGING_DIR"
```

The `trap cleanup_staging EXIT` ensures the staging dir is always removed if the script fails or is interrupted before the `mv` completes.

### 4.4 File Copy via rsync

```bash
rsync -a \
    --exclude='.git/' \
    --exclude='data/*.csv' \
    --exclude='models/*.pkl' \
    --exclude='models/*.npy' \
    --exclude='plots/*.png' \
    --exclude='.venv/' \
    --exclude='__pycache__/' \
    --exclude='.DS_Store' \
    --exclude='.ml_config.json' \
    --exclude='bootstrap.py' \
    --exclude='Dockerfile.bootstrap' \
    --exclude='start.sh' \
    --exclude='init.py' \
    "$TEMPLATE_DIR/" "$STAGING_DIR/"
```

**Critical:** `bootstrap.py`, `Dockerfile.bootstrap`, `start.sh`, and `init.py` MUST be in the exclude list. These are setup scripts that belong in the template source repo, not in created projects.

### 4.5 Config File (Bash heredoc)

```bash
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
```

`PY_VER=$(python3 --version 2>&1 | awk '{print $2}')` and `CREATED_AT=$(date -u +"%Y-%m-%dT%H:%M:%SZ")`

### 4.6 Venv, Deps, Atomic Move, Shebang Fix

```bash
# Create venv inside staging (invisible to VS Code)
python3 -m venv "$STAGING_DIR/.venv"

# Install deps
"$STAGING_DIR/.venv/bin/pip" install --upgrade pip -q
"$STAGING_DIR/.venv/bin/pip" install -r "$STAGING_DIR/requirements.txt" -q

# Atomic move — print "Creating project at:" ONLY here
echo "Creating project at: $PROJECT_DIR"
mv "$STAGING_DIR" "$PROJECT_DIR"
STAGING_DIR_SET=false  # disable cleanup trap

# Fix shebangs
python3 -m venv --upgrade "$PROJECT_DIR/.venv" 2>/dev/null || \
    "$PROJECT_DIR/.venv/bin/python" -m pip install pip -q 2>/dev/null || true
```

### 4.7 Claude Launch

```bash
cd "$PROJECT_DIR"
source ".venv/bin/activate"
if command -v claude &>/dev/null; then
    claude .
else
    echo "Claude Code CLI not found. Install: npm install -g @anthropic-ai/claude-code"
    echo "Then run: cd $PROJECT_DIR && source .venv/bin/activate && claude ."
fi
```

---

## 5. init.py Architecture

`init.py` is the Python-equivalent of `start.sh`. It is the entry point when the user selects option 2 from `start.sh`'s menu, or runs `python3 init.py` directly. It also handles mode 3 (Claude Code) — both modes 2 and 3 go through full project setup.

### 5.1 Imports and Constants

```python
#!/usr/bin/env python3
"""
init.py — ML Pipeline Template: Python CLI Setup
Usage: python3 init.py
Requires: Python 3.9+ stdlib only (runs before venv is active)
"""

import os, sys, json, shutil, subprocess
from pathlib import Path
from datetime import datetime, timezone
```

Stdlib only — no third-party imports. This script must work before any venv exists.

### 5.2 EXCLUDE Set — What NOT to Copy

```python
EXCLUDE = {
    ".git", "__pycache__", ".venv", ".DS_Store",
    ".ml_config.json",
    # Setup scripts: stay in the template folder, never copied into projects.
    "bootstrap.py", "Dockerfile.bootstrap", "start.sh", "init.py",
}
EXCLUDE_EXTS = {".csv", ".pkl", ".npy", ".png", ".pyc"}
```

The comment explaining WHY setup scripts are excluded is important — it documents the design decision that prevents the duplication bug.

### 5.3 `_ignore()` Function for `shutil.copytree`

```python
def _ignore(src: str, names: list[str]) -> set[str]:
    ignored = set()
    for name in names:
        full = Path(src) / name
        if name in EXCLUDE:
            ignored.add(name)
        elif full.suffix in EXCLUDE_EXTS:
            ignored.add(name)
    return ignored
```

### 5.4 Staging Dir Helper

```python
def _make_staging_dir(project_dir: Path) -> Path:
    """
    Return a staging path in the SAME parent directory as project_dir.

    /tmp is on a different APFS volume — shutil.move from /tmp is a cross-volume
    COPY (VS Code sees folder appear empty, then files added one by one).
    Same-parent staging means shutil.move is an atomic RENAME on the same volume
    → VS Code sees the project folder appear ONCE, fully complete.
    """
    staging = project_dir.parent / f".ml_staging_{project_dir.name}"
    staging.mkdir(parents=True, exist_ok=True)
    return staging
```

### 5.5 `create_project(cfg)` — Path Calculation

```python
def create_project(cfg: dict) -> tuple:
    """Return (project_dir, staging_dir)."""
    template_dir = Path(__file__).parent.resolve()
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_dir  = template_dir.parent / f"{cfg['project_name']}_{timestamp}"
    staging_dir  = _make_staging_dir(project_dir)
    return project_dir, staging_dir
```

`template_dir.parent` places the project as a sibling of the template folder, not inside it.

### 5.6 `move_to_final()` — Atomic Move + Shebang Fix

```python
def move_to_final(staging_dir: Path, project_dir: Path):
    """
    Atomic move: staging → project_dir (same filesystem → os.rename, not copy).
    VS Code sees the project appear ONCE — fully populated, .venv already inside.
    Then fix .venv script shebangs (path changed from staging dir to project_dir).
    """
    print(f"\n▶ Creating project at: {project_dir}")    # ONLY here, not during staging
    shutil.move(str(staging_dir), str(project_dir))
    # Fix shebang paths in .venv/bin/pip etc. after the directory was renamed
    venv = project_dir / ".venv"
    try:
        subprocess.run(
            [sys.executable, "-m", "venv", "--upgrade", str(venv)],
            check=True, capture_output=True
        )
    except Exception:
        # Fallback: reinstall pip to fix shebang (fast, no package reinstall)
        python = str(venv / "bin" / "python")
        subprocess.run([python, "-m", "pip", "install", "pip", "-q"],
                       check=False, capture_output=True)
```

### 5.7 `__main__` Block — Execution Order

```python
if __name__ == "__main__":
    banner()
    choice = mode_select()

    if choice == "1":
        # Hand off to start.sh
        script = Path(__file__).parent / "start.sh"
        os.execv("/bin/bash", ["/bin/bash", str(script)])

    # choices "2" and "3" both go through full project setup + launch
    cfg                      = collect_inputs()
    project_dir, staging_dir = create_project(cfg)

    # All heavy work in hidden staging dir (invisible to VS Code)
    copy_template(Path(__file__).parent.resolve(), staging_dir)  # 1. template files
    copy_dataset(cfg, staging_dir)                                # 2. dataset
    write_config(cfg, staging_dir)                                # 3. config
    create_venv(staging_dir)                                      # 4. .venv
    install_deps(staging_dir)                                     # 5. pip install

    # One atomic move → VS Code sees project appear once, complete
    move_to_final(staging_dir, project_dir)

    show_summary(cfg, project_dir)
    maybe_open_claude(project_dir)
```

---

## 6. .ml_config.json — Required Keys

Every created project must have a `.ml_config.json` at its root with exactly these keys:

```json
{
  "project_name":        "my-ml-project",
  "dataset_filename":    "titanic.csv",
  "dataset_path":        "data/titanic.csv",
  "target_column":       "auto-detect",
  "task_type":           "auto-detect",
  "deployment_platform": "render",
  "github_username":     "your-github-username",
  "github_repo":         "my-ml-project",
  "github_visibility":   "public",
  "github_url":          "https://github.com/your-github-username/my-ml-project",
  "python_version":      "3.11.9",
  "created_at":          "2026-05-25T12:00:00Z",
  "venv_path":           ".venv",
  "template_version":    "1.0.0"
}
```

**Key notes:**
- `dataset_path` is always `data/<filename>` (relative, inside project), NOT the original user-supplied path
- `target_column` and `task_type` start as `"auto-detect"` — Claude updates them after EDA
- `github_url` is `""` (empty string) if `github_username` was not provided
- `created_at` is UTC ISO 8601 format
- All three setup scripts (`bootstrap.py`, `start.sh`, `init.py`) produce identical JSON structure

---

## 7. CLAUDE.md Agent System

### 7.1 Root CLAUDE.md — Always Loaded

The root `CLAUDE.md` is loaded by Claude Code automatically when it opens the project. It must be concise (target: under 65 lines). It contains:

1. **Role and Objective** — "expert Data Scientist and Autonomous AI Agent"
2. **Token Management** — sub-agent delegation rules, context isolation
3. **Sub-Agent Roster** — table of agents with trigger and local spec file path

   | Agent | Trigger | Local spec |
   |---|---|---|
   | EDA Agent | Step 2 — EDA | `@src/CLAUDE.md` |
   | Data Engineering Agent | Step 3 — Preprocessing | `@src/CLAUDE.md` |
   | Optimization Agent | Steps 4–6 — Training | `@src/CLAUDE.md` |
   | FastAPI Agent | API development | `@src/CLAUDE.md` |
   | Docker Agent | Step 12 — Docker | `@deploy/CLAUDE.md` |
   | Documentation Agent | Steps 8, docs | `@docs/CLAUDE.md` |
   | Testing Agent | After pipeline | `@tests/CLAUDE.md` |
   | Git & Deploy Agent | Steps 11–13 | `@deploy/CLAUDE.md` |
   | Cloud Deploy Agent | Step 14 — Cloud | `@deploy/CLAUDE.md` |

4. **Token Conservation Rules** — never paste raw CSV/logs, sub-agents return summaries only
5. **Operational Rules** — immediate execution, state tracking, `random_state=42`
6. **Project Scope** — reads from `.ml_config.json`; tech stack
7. **ML Process Checklist** — Steps 0–14 as checkboxes
8. **Initialization Instructions** — the 6-step startup sequence Claude follows

### 7.2 Initialization Flow (what Claude does on startup)

Claude must follow these steps in order when it opens a project:

1. **Check for `.ml_config.json`** — if found, read all keys; if not, ask user one question at a time (dataset path, target column, GitHub username, repo name, platform)
2. **Check for `.venv/`** — if missing, create it and install `requirements.txt`; for all subsequent Python commands use `.venv/bin/python`
3. **Auto-detect task type** — if target column has ≤ 20 unique values or dtype is object/bool → Classification; otherwise → Regression
4. **Scan workspace** — read first 5 rows and column names of the CSV
5. **Show confirmation summary** and wait for Y/Enter:
   ```
   Dataset   : data/titanic.csv
   Target    : Survived
   Task      : Classification
   Platform  : render
   GitHub    : https://github.com/username/repo
   
   Proceed with the pipeline? [Y/n]
   ```
6. **Launch EDA Agent** (Step 2) immediately after confirmation

### 7.3 src/CLAUDE.md — Data Pipeline Agents

Contains specs for: EDA Agent, Data Engineering Agent, Optimization Agent, FastAPI Agent, plus instructions for Steps 8 (docs), 9 (requirements.txt), and 10 (workspace reorganization).

**EDA Agent** — saves all plots to `plots/`; returns bullet-point text summary only (no raw data, no code)

**Data Engineering Agent** — writes `src/preprocess.py` with:
- Drop non-feature columns (Id, Name, Ticket, Cabin, etc.)
- CoW-safe pandas: use `df.assign()` or `df.loc[:, col] =` not `df[col] =` after any drop/copy
- Derive features BEFORE dropping source columns
- Impute: median for numeric, most-frequent for categorical
- LabelEncoder for classification targets
- ColumnTransformer: StandardScaler (classification) or RobustScaler (regression)
- 80/20 stratified split (classification) or random (regression), `random_state=42`
- Save: `X_train_raw.pkl`, `X_test_raw.pkl`, `y_train.npy`, `y_test.npy`, `label_encoder.pkl`, `preprocessor.pkl`, `X_train.npy`, `X_test.npy`

**Optimization Agent** — runs GridSearchCV over candidate models, builds final sklearn Pipeline (`[('preprocessor', preprocessor), ('model', candidate)]`), saves `models/final_pipeline.pkl`

Candidate models (classification): LogisticRegression (saga), RandomForestClassifier, SVC (probability=True), GradientBoostingClassifier

Candidate models (regression): Ridge, RandomForestRegressor, GradientBoostingRegressor, SVR

**FastAPI Agent** — inspects the pipeline before writing `app.py`; replicates all pre-pipeline feature engineering; endpoints: `GET /health`, `POST /predict`, `POST /predict/batch`

### 7.4 deploy/CLAUDE.md — Deployment Agents

Contains specs for: Docker Agent, Git & Deploy Agent, Cloud Deploy Agent, plus Steps 11 (git init + GitHub push) and 12 (Dockerfile + containerization).

**deploy/cloud.md** — index file that `@`-imports the two cloud deployment files:
```
- Step 13 — Render: @deploy/cloud-render.md
- Step 14 — Generic: @deploy/cloud-platforms.md
```

**deploy/cloud-render.md** — Step 13: render.yaml, Render dashboard steps, verify deployment, create `docs/deployment_guide.md`

**deploy/cloud-platforms.md** — Step 14: platform comparison table, deploy commands for Fly.io, Railway, AWS App Runner, GCP Cloud Run, Azure Container Apps; universal smoke tests

### 7.5 tests/CLAUDE.md — Testing Agent

Writes `tests/test_pipeline.py` covering: artifact integrity, single-sample predictions, full test-set evaluation, per-class accuracy, consistency check, probability check.

Also documents the template test suite in `tests/template_tests/`:
- `run_tests.sh --fast` — 34 checks, ~30 seconds
- `run_tests.sh` — all checks including end-to-end (~10 min)
- `run_tests.sh --suite 03` — targeted suite

### 7.6 docs/CLAUDE.md — Documentation Agent

Writes: `docs/summary.md`, `docs/testing_guide.md`, `docs/test_results.md`, `docs/deployment_guide.md`, `docs/docker_guide.md`

---

## 8. Project vs Template Distinction

This distinction is the single most important design principle and the root cause of every duplication bug that has been fixed.

### The Rule

| Location | Stays in template repo? | Copied into project? |
|---|---|---|
| `bootstrap.py` | YES | NO |
| `Dockerfile.bootstrap` | YES | NO |
| `start.sh` | YES | NO |
| `init.py` | YES | NO |
| `CLAUDE.md` | YES | YES |
| `src/CLAUDE.md` | YES | YES |
| `deploy/*.md` | YES | YES |
| `docs/CLAUDE.md` | YES | YES |
| `tests/CLAUDE.md` | YES | YES |
| `requirements.txt` | YES | YES |
| `.gitignore` | YES | YES |
| `.claude/settings.local.json` | YES | YES |
| `README.md` | YES | YES |

### Why Setup Scripts Must Not Be in Projects

**bootstrap.py scenario:** If `bootstrap.py` is copied into a project folder and a user runs `python3 bootstrap.py` inside it, bootstrap.py creates a NEW project in the current directory — which is already the project folder. This creates a nested `project_name_TIMESTAMP/` inside the existing project. Result: duplication.

**start.sh/init.py scenario:** These scripts are designed to create a SIBLING project: `template_dir.parent / project_name_TIMESTAMP`. If they are inside a project folder (not the template folder), running them creates a sibling of the project folder rather than an independent workspace. Result: unexpected project proliferation.

**The fix:** Always include `bootstrap.py`, `Dockerfile.bootstrap`, `start.sh`, `init.py` in the SKIP/EXCLUDE set when writing files to a new project.

In `bootstrap.py`:
```python
SKIP_IN_PROJECT = {"start.sh", "init.py"}
# Note: bootstrap.py and Dockerfile.bootstrap are not in FILES dict at all,
# so they are never written to projects
```

In `start.sh` (rsync excludes):
```bash
--exclude='bootstrap.py'
--exclude='Dockerfile.bootstrap'
--exclude='start.sh'
--exclude='init.py'
```

In `init.py` (EXCLUDE set):
```python
EXCLUDE = {
    ".git", "__pycache__", ".venv", ".DS_Store", ".ml_config.json",
    "bootstrap.py", "Dockerfile.bootstrap", "start.sh", "init.py",
}
```

---

## 9. GitHub Setup Steps

### 9.1 Prerequisites

```bash
# Check GitHub CLI is installed
gh --version

# Install if missing (macOS)
brew install gh

# Check authentication
gh auth status

# Login if needed (opens browser OAuth)
gh auth login
```

### 9.2 Git Repository Initialization

```bash
cd <repo-name>

git init
git add .
git commit -m "Initial commit: ML Pipeline Template v1.0.0"
```

### 9.3 Create GitHub Repo and Push

```bash
gh repo create <repo-name> \
  --public \
  --description "ML Pipeline Template — one-command bootstrap for end-to-end ML projects" \
  --source=. \
  --remote=origin \
  --push
```

### 9.4 Verify

```bash
gh repo view <repo-name> --web
```

Confirm the repo is live at `https://github.com/<github-username>/<repo-name>`.

The raw bootstrap URL (needed in README.md) is:
```
https://raw.githubusercontent.com/<github-username>/<repo-name>/main/bootstrap.py
```

### 9.5 README.md Bootstrap Curl URL

The README must contain the exact curl command users will run. Use the actual GitHub username and repo name provided in step 0:

```bash
curl -O https://raw.githubusercontent.com/<github-username>/<repo-name>/main/bootstrap.py
python3 bootstrap.py
```

This URL must be present in both `README.md` and `docs/how_to_run.md` with the correct values filled in.

---

## 10. requirements.txt — Pinned Versions

The `requirements.txt` in the template repo contains these pinned versions. They are the versions Claude installs in the generated `.venv` and are later pinned again after training (Step 9) with exact installed versions.

```
# ML Pipeline Template — auto-generated by the pipeline
# Generated: <date>
# Python <version> | random_state=42

# Core data manipulation
pandas==2.2.2
numpy==1.26.4

# Machine learning
scikit-learn==1.8.0
joblib==1.5.3

# Visualisation
matplotlib==3.10.9
seaborn==0.13.2

# API
fastapi==0.136.3
uvicorn==0.41.0
```

**Why pinned?** Reproducibility. The ML pipeline must produce identical results across environments. Unpinned versions risk API changes (especially sklearn) that silently alter model behavior or break preprocessing.

---

## 11. .claude/settings.local.json

Pre-authorized bash commands eliminate permission prompts during the Claude Code ML pipeline run. The file lives at `.claude/settings.local.json` inside the created project (copied from the template).

```json
{
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
```

---

## 12. Common Pitfalls to Avoid

### Pitfall 1: Setup Scripts Copied into Projects

**Symptom:** Running `start.sh` inside a project creates an unexpected sibling project; running `bootstrap.py` inside a project creates a nested folder.

**Fix:** Always include `bootstrap.py`, `Dockerfile.bootstrap`, `start.sh`, `init.py` in the exclude/skip set when copying template files to a new project. This must be enforced in all three entry points independently (bootstrap.py's `SKIP_IN_PROJECT`, start.sh's rsync `--exclude` flags, init.py's `EXCLUDE` set).

### Pitfall 2: Using /tmp for Staging on macOS

**Symptom:** VS Code sees the project folder appear empty, then watches files populate one by one over several seconds. File watchers trigger on incomplete state.

**Root cause:** `/tmp` is on a separate APFS volume from `~`. `shutil.move()` across volumes is a copy operation, not a rename. The destination folder appears immediately (empty) and files are copied one by one.

**Fix:** Always create the staging directory in the SAME parent directory as the final project: `project_dir.parent / f".ml_staging_{project_dir.name}"`. Same-volume `shutil.move()` is an atomic rename — the folder appears exactly once, fully populated.

### Pitfall 3: Printing "Creating project at:" During Staging

**Symptom:** User sees the project path printed, looks in Finder/terminal, sees nothing yet (or sees the hidden `.ml_staging_*` dir). Confusing UX.

**Fix:** Print "Creating project at: `<path>`" ONLY at the moment of the `mv`/`shutil.move()` call, not before. The staging dir name starts with `.` so it is hidden by default.

### Pitfall 4: Relying on st_birthtime to Find Created Projects

**Symptom:** Code that tries to find the most recently created project folder using `os.stat(path).st_birthtime` fails or returns wrong results on APFS after a move operation (birth time reflects staging creation, not final move time).

**Fix:** Use the timestamp embedded in the folder name (`project_name_YYYYMMDD_HHMMSS`). The timestamp is set at `datetime.now().strftime("%Y%m%d_%H%M%S")` before staging begins, so it is always accurate and human-readable.

### Pitfall 5: Not Fixing Shebangs After venv Move

**Symptom:** After the atomic move from staging to final path, running `.venv/bin/pip` or `source .venv/bin/activate` fails because the embedded absolute paths point to the old staging directory (which no longer exists).

**Fix:** Always run `python3 -m venv --upgrade <venv_path>` immediately after `shutil.move()`. If that fails (e.g., permissions), fall back to `python -m pip install pip -q` inside the venv. This rewrites all activation scripts and wrapper shebang lines to the new path.

### Pitfall 6: Forgetting to Update the Bootstrap URL in README

**Symptom:** Users copy the curl command from README.md and get a 404 because the URL still contains a placeholder username.

**Fix:** After pushing to GitHub, update the curl URL in both `README.md` and `docs/how_to_run.md` with the actual GitHub username and repo name. The URL pattern is:
```
https://raw.githubusercontent.com/<github-username>/<repo-name>/main/bootstrap.py
```

### Pitfall 7: Missing .gitkeep Files

**Symptom:** `data/`, `models/`, and `plots/` directories are missing in projects created via bootstrap (they are empty at template creation time and git doesn't track empty dirs).

**Fix:** Include `FILES["data/.gitkeep"] = ""`, `FILES["models/.gitkeep"] = ""`, `FILES["plots/.gitkeep"] = ""` in the `FILES` dict. These zero-byte files ensure the directories exist in the project and in git.

### Pitfall 8: pandas Copy-on-Write Warnings

**Symptom:** `FutureWarning: ChainedAssignmentError` in preprocessing code with pandas 2.x.

**Fix:** In `src/preprocess.py`, always use CoW-safe assignment:
- Correct: `df = df.assign(col=value)` or `df.loc[:, col] = value`
- Incorrect: `df[col] = value` (after any slice or copy operation)
- Derive new columns BEFORE dropping source columns

### Pitfall 9: SVC Without probability=True

**Symptom:** `predict_proba` fails on SVC model in the FastAPI `app.py` because SVC does not support probability output by default.

**Fix:** Always instantiate SVC as `SVC(probability=True, random_state=42)` in the Optimization Agent's candidate list.

### Pitfall 10: Fitting preprocessor on Full Dataset Instead of Train Split

**Symptom:** Data leakage — test set statistics influence the preprocessing pipeline, leading to optimistic evaluation metrics.

**Fix:** Fit the ColumnTransformer preprocessor ONLY on `X_train_raw`. Save it as `models/preprocessor.pkl` after fitting on training data only. The Optimization Agent then loads this pre-fitted preprocessor and includes it as the first step of the sklearn Pipeline.

---

---

## 13. Auto Pipeline Option (No Claude Subscription Required)

### Overview

Not every user has a Claude Code subscription. `auto_pipeline.py` provides a fully automated ML pipeline using pure sklearn — no AI required. It is embedded in the `FILES` dict inside `bootstrap.py` so it is automatically written into every new project.

### Launch Menu (appears at end of bootstrap.py)

After the project is created, bootstrap.py shows a 3-option menu:

```
How would you like to run the pipeline?
  1) Claude Code   — AI-driven, fully automated (recommended)
  2) Auto Pipeline — no Claude subscription needed (pure sklearn)
  3) Manual        — I'll run it myself later
Enter choice (default: 1):
```

- **Option 1** — launches `claude .` in the project directory (requires Claude Code CLI)
- **Option 2** — runs `.venv/bin/python auto_pipeline.py` immediately
- **Option 3** — prints instructions and exits; user runs manually later

### auto_pipeline.py Architecture

**File location in project:** `auto_pipeline.py` (project root)
**Embedded in bootstrap:** `FILES["auto_pipeline.py"] = r'''...'''`
**Setup scripts excluded from projects:** `start.sh`, `init.py` — but NOT `auto_pipeline.py` (it belongs in every project)

**Critical startup line:**
```python
ROOT = Path(__file__).parent.resolve()
os.chdir(ROOT)   # ensures all relative paths resolve from project root
```
This must be the first thing after path setup — without it, relative paths like `models/final_pipeline.pkl` fail if the script is called from a different working directory.

**Execution steps (in order):**

| Step | What it does | Output |
|---|---|---|
| 0 | Resolve ROOT, chdir, create dirs (data/, models/, plots/, docs/) | — |
| 1 | Load `.ml_config.json`; if missing, prompt for dataset path + target column | config dict |
| 2 | Find CSV: try dataset_path from config → scan data/ for any .csv → exit with clear error if none found | csv_path |
| 3 | Import heavy deps (numpy, pandas, matplotlib, sklearn, joblib) after path check | — |
| 4 | EDA: print shape/dtypes/missing/class balance; save `plots/eda_correlation.png` + `plots/eda_target.png` | 2 PNG files |
| 5 | Auto-detect task type: ≤20 unique values or object/bool target → classification; else regression | task_type |
| 6 | Preprocessing: drop columns >50% missing; numeric→SimpleImputer(median)+StandardScaler/RobustScaler; categorical→SimpleImputer(most_frequent)+OneHotEncoder; LabelEncoder for target (classification) | ColumnTransformer |
| 7 | 80/20 stratified split (classification) or random split (regression), random_state=42 | X_train, X_test, y_train, y_test |
| 8 | GridSearchCV(cv=3, n_jobs=-1) over 3 candidates per task type | best model + params |
| 9 | Refit best pipeline on full train set; evaluate on test set | metrics |
| 10 | Save `models/final_pipeline.pkl` + `models/label_encoder.pkl` (classification) | .pkl files |
| 11 | Write `docs/auto_summary.md` with dataset info, model results, metrics, artifact paths | .md file |
| 12 | Print coloured summary banner to terminal | — |

**Candidate models:**

Classification:
- `LogisticRegression(solver='saga', random_state=42)` — grid: `C=[0.1, 1, 10]`
- `RandomForestClassifier(random_state=42)` — grid: `n_estimators=[100,200], max_depth=[None,5]`
- `GradientBoostingClassifier(random_state=42)` — grid: `n_estimators=[100,200], learning_rate=[0.05,0.1]`

Regression:
- `Ridge()` — grid: `alpha=[0.1, 1, 10]`
- `RandomForestRegressor(random_state=42)` — grid: `n_estimators=[100,200], max_depth=[None,5]`
- `GradientBoostingRegressor(random_state=42)` — grid: `n_estimators=[100,200], learning_rate=[0.05,0.1]`

**Artifacts produced:**

| File | Description |
|---|---|
| `models/final_pipeline.pkl` | Full sklearn Pipeline (ColumnTransformer + best model) |
| `models/label_encoder.pkl` | Fitted LabelEncoder (classification only) |
| `plots/eda_correlation.png` | Feature correlation heatmap |
| `plots/eda_target.png` | Target variable distribution |
| `docs/auto_summary.md` | Full markdown report with metrics and reproducibility snippet |

**Error handling:**
- If no CSV found: prints exact path to copy dataset into + re-run command, exits cleanly
- If a model fails GridSearchCV: skips it, continues with remaining candidates
- If fewer than 1 model succeeds: exits with clear error

### Key Implementation Notes for AI Agent

1. **Embed in FILES dict** — `FILES["auto_pipeline.py"]` must use a raw string `r'''...'''` to preserve backslashes
2. **Do NOT add to SKIP_IN_PROJECT** — unlike `start.sh`/`init.py`, `auto_pipeline.py` belongs in every project
3. **os.chdir(ROOT) is mandatory** — bootstrap calls `subprocess.run([python, "auto_pipeline.py"])` after `os.chdir(project_dir)`, but defensive chdir inside the script is required for direct invocation
4. **DATA_DIR must be defined** — `DATA_DIR = ROOT / "data"` used for auto-discovery glob
5. **Launch menu replaces direct claude launch** — the old `subprocess.run(["claude", "."])` becomes the 3-option menu at the end of bootstrap.py's main block

### Post-Pipeline Deploy Functions

After the ML training completes, `auto_pipeline.py` shows a second menu that lets users generate a FastAPI app, themed frontend, Dockerfile, push to GitHub, and deploy to Render — all without Claude. These functions are appended to the bottom of `auto_pipeline.py` and called as `_post_pipeline_menu(ROOT, cfg, task_type, final_pipe, label_encoder)`.

#### ANSI Alias Fix (critical)

`auto_pipeline.py` uses `X = "\033[0m"` for the ANSI reset colour, but sklearn's convention also uses `X = df.drop(...)` later in the script (DataFrame). To avoid the name clash, add stable aliases immediately after the ANSI colour definitions:

```python
G = "\033[0;32m"; C = "\033[0;36m"; B = "\033[1m"
Y = "\033[1;33m"; R = "\033[0;31m"; X = "\033[0m"

# Stable aliases — sklearn/pandas code later reuses single-letter variable names
_G = G; _C = C; _B = B; _Y = Y; _R = R; _X = X
```

All post-pipeline functions must use `_G`, `_C`, `_B`, `_Y`, `_R`, `_X` instead of the single-letter names.

#### `_get_features(pipeline)` → `(num_feats, cat_feats)`

Introspects the fitted sklearn Pipeline to extract which feature names were passed to the numeric and categorical branches of the ColumnTransformer:

```python
def _get_features(pipeline):
    ct = pipeline.named_steps["preprocessor"]
    num_feats = list(ct.transformers_[0][2]) if ct.transformers_ else []
    cat_feats = list(ct.transformers_[1][2]) if len(ct.transformers_) > 1 else []
    return num_feats, cat_feats
```

#### `_detect_domain(dataset_filename, column_names, project_name="")` → theme dict

Dynamically detects the dataset domain from the filename, project name, and column names. No configuration required.

**How it works:**
1. Normalises all inputs to lowercase, replacing `_` and `-` with spaces
2. Concatenates filename + project name + all column names into a single search string
3. Scores 10 known domains by counting keyword matches in the search string
4. Returns the highest-scoring domain's theme dict; falls back to a generic midnight-blue theme when nothing scores > 0

**Supported domains and their trigger keywords:**

| Domain | Icon | Example trigger columns |
|---|---|---|
| Health | 🩺 | glucose, insulin, bmi, blood, diabetes, cancer, heart, patient |
| Travel | ✈️ | flight, airline, delay, airport, passenger, departure, ticket |
| Finance | 💰 | loan, credit, fraud, income, bank, salary, payment, default |
| Shipping | 🚢 | ship, cargo, freight, container, port, logistics, vessel |
| Real Estate | 🏠 | house, sqft, bedroom, bathroom, property, rent, floor |
| HR | 👤 | employee, attrition, department, churn, satisfaction, tenure |
| Quality | 🍷 | wine, quality, alcohol, acidity, sugar, flavor, sulphates |
| Maritime | ⚓ | titanic, survived, pclass, embarked, lifeboat |
| Retail | 🛒 | customer, purchase, review, sales, cart, discount |
| Energy | ⚡ | energy, electricity, solar, co2, emission, consumption |
| Generic | 🤖 | *(fallback — no score threshold met)* |

**Theme dict keys:** `icon`, `name`, `primary`, `accent`, `btn`, `btn_hover`, `body_bg`, `gradient`, `desc`

#### `_generate_frontend(root, cfg, task_type, num_feats, cat_feats, label_encoder=None)` → writes `index.html`

Generates a responsive, single-file prediction UI themed to the dataset domain. Writes `root/index.html`.

**Key implementation decisions:**

- **Placeholder substitution, not f-strings** — The HTML template uses `TMPL_*` placeholders (e.g. `TMPL_GRADIENT`, `TMPL_ACCENT`). All substitutions are done with `.replace()` after the full string is assembled. This avoids the need to escape CSS `{ }` braces in f-strings. Longer tokens are replaced before shorter ones to prevent partial substitution (e.g. `TMPL_BTN_HOVER` before `TMPL_BTN`).
- **Accent alpha for focus ring** — `accent_alpha = accent + "33"` appends hex opacity (20%) to the accent colour for the CSS `box-shadow` focus ring.
- **Class labels from label_encoder** — If `label_encoder` is provided, `label_encoder.classes_` populates the JS `CLASSES` array so probability bars show real class names instead of "Class 0", "Class 1".
- **Form fields** — Numeric features become `<input type="number" step="any">`; categorical features become `<input type="text">`.
- **Vanilla JS only** — No external dependencies. Fetch, DOM manipulation, and CSS animations are all built-in.

**UI features:**
- Themed header: gradient background + emoji icon + project title + domain description
- Responsive 2-column form grid (collapses to 1 column on mobile)
- Animated probability bars for classification (animated fill on reveal)
- Spinner inside the Predict button during API call
- Result panel slides in with a CSS animation
- `GET /health` and `/docs` links in the footer

#### `_generate_app(root, task_type, num_feats, cat_feats)` → writes `app.py`

Generates a complete FastAPI app using string concatenation (NOT f-strings with dict keys — Python 3.9–3.11 forbid backslashes inside `{}` in f-strings). Key design points:

- Imports: `os`, `fastapi`, `fastapi.responses.FileResponse`, `fastapi.middleware.cors.CORSMiddleware`, `uvicorn`, `joblib`, `pandas`, `pydantic`
- Adds `CORSMiddleware(allow_origins=["*"])` so the frontend can call the API from any origin
- Loads `models/final_pipeline.pkl` and (for classification) `models/label_encoder.pkl` at startup
- `InputData` Pydantic model with numeric fields as `Optional[float] = None` and categorical fields as `Optional[str] = None`
- `GET /` → `FileResponse("index.html")` if the file exists; falls back to JSON `{"message": "ML Prediction API", "docs": "/docs"}`
- `GET /health` → `{"status": "ok", "model": "loaded"}`
- `POST /predict` → `{"prediction": <label>, "probabilities": [...]}`
- `POST /predict/batch` → `{"predictions": [...]}`
- For regression: returns `{"prediction": float}` without probabilities
- Uses `pipeline.predict_proba` for classification (requires `probability=True` on SVC)

#### `_generate_docker(root)` → writes `Dockerfile` and `.dockerignore`

Generates a multi-stage Dockerfile:

```dockerfile
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /install /usr/local
COPY app.py .
COPY index.html* .
COPY models/ models/
RUN useradd -m appuser && chown -R appuser /app
USER appuser
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

> **Why `COPY index.html*` (glob)?** If `_generate_docker()` is ever called without `_generate_frontend()` having run first, `COPY index.html .` would fail with a Docker build error. The glob `*` silently skips the copy when the file is absent.

`.dockerignore` excludes: `.git/`, `__pycache__/`, `.venv/`, `data/`, `plots/`, `tests/`, `docs/`, `*.md`, `.env`, `.DS_Store`, `.claude/`

#### `_push_github(root, cfg)` → git init + GitHub push

Reads `github_username`, `github_repo`, `github_visibility` from the `cfg` dict (already loaded from `.ml_config.json`):

```python
def _push_github(root, cfg):
    gh_user = cfg.get("github_username", "")
    gh_repo = cfg.get("github_repo", "")
    gh_vis  = cfg.get("github_visibility", "public")
    if not gh_user or not gh_repo:
        print("  No GitHub username/repo in config — skipping push")
        return
    cmds = [
        ["git", "init"],
        ["git", "add", "."],
        ["git", "commit", "-m", "Initial commit: ML pipeline with FastAPI"],
        ["gh", "repo", "create", gh_repo,
         f"--{gh_vis}", "--source=.", "--remote=origin", "--push"],
    ]
    for cmd in cmds:
        subprocess.run(cmd, cwd=root, check=True)
```

Requires the `gh` CLI to be installed and authenticated (`gh auth login`). If `github_username` is empty in `.ml_config.json`, the function prints a skip message and returns without error.

#### `_deploy_render(root, cfg)` → writes `render.yaml` + prints walkthrough

Writes `render.yaml` at the project root, commits it, pushes to GitHub, then prints the 5-step Render dashboard walkthrough:

```yaml
services:
  - type: web
    name: <project-name>
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn app:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: PYTHON_VERSION
        value: "3.11.0"
```

#### `_post_pipeline_menu(root, cfg, task_type, pipeline, label_encoder=None)` → post-pipeline deploy menu

Called immediately after `joblib.dump(final_pipe, pipeline_path)`. Signature includes `label_encoder=None` so class names can be passed through to `_generate_frontend()` for the probability bars. Uses `_C`, `_B`, `_X` aliases throughout. Key implementation details:

```python
def _post_pipeline_menu(root, cfg, task_type, pipeline, label_encoder=None):
    print(f"\n{_B}What would you like to do next?{_X}")
    print(f"  1) Generate FastAPI app + Dockerfile")
    print(f"  2) Push to GitHub")
    print(f"  3) Deploy to Render")
    print(f"  4) All of the above  ← recommended")
    print(f"  5) Done — I'll handle it myself")
    try:
        choice = input("Enter choice (default: 4): ").strip() or "4"
    except EOFError:
        choice = "4"   # non-interactive / piped input → default to "All"
    ...
    if do_app:
        _generate_app(root, task_type, num_feats, cat_feats)
        _generate_frontend(root, cfg, task_type, num_feats, cat_feats, label_encoder)
        _generate_docker(root)
```

**Call at bottom of script:**
```python
# Pass label_encoder so _generate_frontend() can display real class names
_post_pipeline_menu(ROOT, cfg, task_type, final_pipe, label_encoder)
```

**The `try/except EOFError: choice = "4"` is mandatory.** When `bootstrap.py` launches `auto_pipeline.py` via `os.execv()`, stdin is inherited cleanly. But if the process is invoked with piped input (e.g. in tests), the stdin buffer may be exhausted by the time the post-pipeline menu runs — `EOFError` is the expected exception in that case, and defaulting to `"4"` (All of the above) gives the best test coverage.

**`os.execv()` in bootstrap.py** — The launch of `auto_pipeline.py` from bootstrap's main block must use `os.execv()`, not `subprocess.run()`:

```python
# bootstrap.py — main block, option 2
import os
python = str(project_dir / ".venv" / "bin" / "python")
os.execv(python, [python, "auto_pipeline.py"])
```

`os.execv()` replaces the bootstrap process entirely, so stdin is passed cleanly to `auto_pipeline.py`. Using `subprocess.run()` would buffer stdin at the parent level, causing the child's `input()` calls to receive EOF immediately.

### Manual Usage (Option 3 or later)

```bash
cd my-project_20260526_143000
source .venv/bin/activate
python auto_pipeline.py         # run auto pipeline (shows post-pipeline deploy menu)
# or
claude .                        # run Claude Code pipeline
```

### Common Pitfalls

| Pitfall | Symptom | Fix |
|---|---|---|
| Missing `os.chdir(ROOT)` | `FileNotFoundError: models/final_pipeline.pkl` when called from wrong dir | Add `os.chdir(ROOT)` after ROOT is defined |
| `auto_pipeline.py` in SKIP_IN_PROJECT | File not written to project | Remove from SKIP set — it belongs in projects |
| No dataset in data/ | Script exits with confusing error | Clear message: "Copy your dataset into: {DATA_DIR}/" |
| Embedding with regular string | Backslash errors in ANSI codes | Use `r'''...'''` raw string in FILES dict |
| ANSI variable `X` overwritten by DataFrame | `TypeError` in post-pipeline functions when concatenating `X + string` | Add `_X = X` alias before any sklearn/pandas code |
| `subprocess.run()` for auto_pipeline.py launch | Post-pipeline `input()` gets EOFError immediately | Use `os.execv()` — replaces process, stdin passes cleanly |
| `input()` raises EOFError in piped/test mode | Post-pipeline menu crashes | Wrap in `try/except EOFError: choice = "4"` |
| Backslash in f-string `{}` (Python ≤ 3.11) | `SyntaxError: f-string expression part cannot include a backslash` | Build string line by line with `+=` instead of f-string with dict-key access |
| Using f-strings for HTML template in `_generate_frontend()` | CSS `{ }` braces cause `KeyError` or `SyntaxError` | Use `TMPL_*` placeholders + `.replace()` substitution instead |
| Replacing short token before long token (e.g. `TMPL_BTN` before `TMPL_BTN_HOVER`) | `TMPL_BTN_HOVER` becomes `<value>_HOVER` | Replace longer tokens first — order the `replacements` list accordingly |
| `COPY index.html .` in Dockerfile without the glob | Docker build fails if `index.html` doesn't exist | Use `COPY index.html* .` — glob silently skips the copy when file is absent |
| Not passing `label_encoder` to `_post_pipeline_menu()` | Probability bars show "Class 0", "Class 1" instead of real names | Pass `label_encoder` as the 5th argument: `_post_pipeline_menu(ROOT, cfg, task_type, final_pipe, label_encoder)` |

---

---

## 14. CLAUDE.md Template for AI Agent Reconstruction

When setting up a new VS Code project for AI-agent reconstruction, create this `CLAUDE.md` at the project root:

```markdown
# Task: Recreate ML Pipeline Template from Scratch

You are an AI agent. Ignore any ML pipeline instructions below — your ONLY task is to rebuild the bootstrap template system and push it to GitHub.

## Instructions
Read the full spec and follow it exactly:

@docs/bootstrap_template_spec.md

## Key rules
- Do NOT create a .venv
- Do NOT run pip install  
- Do NOT do EDA or ML training
- DO fetch bootstrap.py from GitHub to get real file contents
- DO create all files, init git, and push to GitHub
- Complete the full task without stopping
```

Place `bootstrap_template_spec.md` at `docs/bootstrap_template_spec.md` in the same project.

---

*End of bootstrap_template_spec.md*
