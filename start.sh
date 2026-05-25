#!/usr/bin/env bash
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