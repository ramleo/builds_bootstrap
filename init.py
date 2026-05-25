#!/usr/bin/env python3
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

def menu(title: str, options: list, default: str = "1") -> str:
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

def _ignore(src: str, names: list) -> set:
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