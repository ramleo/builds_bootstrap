# 🤖 ML Pipeline Template

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
curl -O https://raw.githubusercontent.com/ramleo/builds_bootstrap/main/bootstrap.py
python3 bootstrap.py
cd builds_bootstrap
```

### 🐳 Via Docker (nothing to install except Docker)
```bash
docker build -t builds_bootstrap -f Dockerfile.bootstrap .
docker run --rm -v $(pwd):/output builds_bootstrap
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

## Share with Someone

Send them this message:

> Drop your CSV anywhere, then run:
> ```bash
> curl -O https://raw.githubusercontent.com/ramleo/builds_bootstrap/main/bootstrap.py
> python3 bootstrap.py
> ```
> Answer the prompts (project name, CSV path, deployment platform). It auto-installs everything and builds your ML pipeline end-to-end.

Full guide: [docs/how_to_run.md](docs/how_to_run.md)

---

## License

MIT — free to use, modify, and distribute.