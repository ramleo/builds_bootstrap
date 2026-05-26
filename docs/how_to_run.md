# How to Run the ML Pipeline Template

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
curl -O https://raw.githubusercontent.com/ramleo/builds_bootstrap/main/bootstrap.py
```

This downloads a single installer file — no git, no GitHub account required.

---

## Step 3 — Run the bootstrap

```bash
python3 bootstrap.py
```

This will:
- Auto-install **Homebrew**, **Node.js**, and **Claude Code CLI** if they're missing
- Ask you a few setup prompts (project name, CSV path, platform, GitHub)
- Create a timestamped project folder (e.g. `my-project_20260526_143000/`)
- Set up a Python virtual environment (`.venv/`) with all dependencies installed

---

## Step 4 — Answer the setup prompts

| Prompt | Example answer |
|---|---|
| Project name | `titanic-predictor` |
| Dataset CSV path | `/Users/yourname/Downloads/titanic.csv` |
| Deployment platform | `2` for Render, or `1` to decide later |
| GitHub username | `your-github-username` (press Enter to skip) |
| GitHub repo name | `titanic-predictor` (defaults to project name) |
| Repo visibility | `1` for Public, `2` for Private |

**You do not need to move your CSV beforehand.** Type its full path — the script copies it into `data/` automatically. Press Enter to skip and add it later.

---

## Step 5 — Choose how to run the pipeline

After the project is created, a launch menu appears:

```
How would you like to run the pipeline?
  1) Claude Code   — AI-driven, fully automated (recommended)
  2) Auto Pipeline — no Claude subscription needed (pure sklearn)
  3) Manual        — I'll run it myself later
```

### Option 1 — Claude Code (recommended)

Launches `claude .` in your new project. Claude reads `.ml_config.json`, shows a confirmation summary, and runs the full 15-step pipeline automatically:

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

Requires a Claude Code subscription. Claude shows a confirmation prompt before starting — press **Enter** (or Y) to proceed.

### Option 2 — Auto Pipeline (no Claude subscription needed)

Runs `auto_pipeline.py` — a pure-sklearn pipeline with zero AI dependency:

1. Loads your CSV and auto-detects the task type (classification or regression)
2. Profiles the data and saves EDA plots to `plots/`
3. Builds a preprocessing pipeline (imputation, encoding, scaling)
4. Runs GridSearchCV over 3 candidate models
5. Evaluates the best model and saves `models/final_pipeline.pkl`
6. Writes `docs/auto_summary.md` with full metrics and a reproducibility snippet

After training completes, a **second menu** appears with deploy options:

```
What would you like to do next?
  1) Generate FastAPI app + Dockerfile
  2) Push to GitHub
  3) Deploy to Render
  4) All of the above  ← recommended
  5) Done — I'll handle it myself
```

| Option | What it does |
|---|---|
| 1 | Writes `app.py` (FastAPI with `/health`, `/predict`, `/predict/batch`) and a multi-stage `Dockerfile` |
| 2 | `git init` → commit → `gh repo create` → push to GitHub |
| 3 | Writes `render.yaml` and prints the 5-step Render dashboard walkthrough |
| 4 | All three steps in sequence — app, Docker, GitHub, then Render |
| 5 | Exits; run any step manually later |

> **Prerequisite for option 2/4:** `brew install gh && gh auth login` (GitHub CLI must be authenticated)

### Option 3 — Manual

Exits and prints the commands to run manually:

```bash
cd my-project_20260526_143000
source .venv/bin/activate
python auto_pipeline.py   # run the auto pipeline
# or
claude .                  # run the Claude Code pipeline
```

---

## Folder layout after the pipeline

```
my-project_20260526_143000/
├── .venv/                  ← Python virtual environment (pre-installed)
├── .ml_config.json         ← your choices (dataset, platform, GitHub)
├── data/                   ← your CSV file
├── models/                 ← trained pipeline artifacts (.pkl)
├── plots/                  ← EDA charts (.png)
├── src/preprocess.py       ← auto-generated preprocessing script (Claude option)
├── tests/test_pipeline.py  ← auto-generated test suite (Claude option)
├── docs/                   ← summary, guides, test results
├── app.py                  ← FastAPI prediction API
├── Dockerfile              ← multi-stage container build
├── requirements.txt        ← pinned library versions
└── render.yaml / fly.toml  ← deployment config (platform-specific)
```

---

## Alternatives to bootstrap.py

### Via git clone

```bash
git clone https://github.com/ramleo/builds_bootstrap
cd builds_bootstrap
./start.sh
```

`start.sh` shows the same 3-option setup menu and creates the project as a sibling folder of `builds_bootstrap/`.

### Via Docker (nothing to install except Docker)

```bash
docker build -t builds_bootstrap -f Dockerfile.bootstrap \
  https://raw.githubusercontent.com/ramleo/builds_bootstrap/main/Dockerfile.bootstrap

docker run --rm -v $(pwd):/output builds_bootstrap

cd builds_bootstrap
./start.sh
```

---

## Running the End-to-End Test Suite

`tests/run_e2e.py` validates the full flow from bootstrap to live API across 58 checks.

```bash
# Fast mode — skips Docker suite (~5 min)
python3 tests/run_e2e.py --fast

# Full run including Docker build (~10 min)
python3 tests/run_e2e.py

# Single suite only
python3 tests/run_e2e.py --suite 1   # bootstrap & project creation
python3 tests/run_e2e.py --suite 2   # pipeline artifacts
python3 tests/run_e2e.py --suite 3   # app.py & Dockerfile content
python3 tests/run_e2e.py --suite 4   # live API (uvicorn)
python3 tests/run_e2e.py --suite 5   # Docker smoke tests

# Test the published GitHub version
python3 tests/run_e2e.py --from-github --fast
```

The test suite also lives in its own repo: [github.com/ramleo/ml-pipeline-tests](https://github.com/ramleo/ml-pipeline-tests)

---

## Quick reference

```bash
python3 --version                        # confirm Python is installed
curl -O <bootstrap_url>                  # download installer
python3 bootstrap.py                     # answer prompts → project created
#  → choose 2 (Auto Pipeline)
#  → choose 4 (All of the above) after training
```

*(Replace `<bootstrap_url>` with `https://raw.githubusercontent.com/ramleo/builds_bootstrap/main/bootstrap.py`)*

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `python3: command not found` | Install Python 3.9+ from [python.org](https://python.org) |
| `claude: command not found` | Run `./start.sh` — it auto-installs, or manually: `npm install -g @anthropic-ai/claude-code` |
| `Permission denied: ./start.sh` | Run `chmod +x start.sh` first |
| Dataset not found | Copy your `.csv` into the project's `data/` folder, then re-run `python auto_pipeline.py` |
| GitHub push fails | Run `gh auth login` first, then choose option 2 from the deploy menu |
| `builds_bootstrap/` already exists | Rename the existing folder before re-running `bootstrap.py` |
| Homebrew install hangs | Accept the Xcode Command Line Tools prompt that appears |
| pip install fails | Check Python version (`python3 --version`) — requires 3.9+ |
| uvicorn: bad interpreter | Run `python3 -m venv --upgrade .venv` inside the project folder |
