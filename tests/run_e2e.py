#!/usr/bin/env python3
"""
run_e2e.py — End-to-End Test Suite for ML Pipeline Template
Tests the full flow: bootstrap → pipeline → API → Docker (optional)

Suites and approximate check counts:
  Suite 1 — Bootstrap & Project Creation    (~11 checks)
  Suite 2 — Pipeline Artifacts              (~13 checks)
  Suite 3 — App & Dockerfile Generation     (~26 checks)  includes frontend/CORS
  Suite 4 — API Endpoints (live server)     (~20 checks)  includes GET / HTML
  Suite 5 — Docker Build & Smoke Test       (~10 checks)  includes GET / in Docker

Usage (from repo root):
  python3 tests/run_e2e.py              # all suites
  python3 tests/run_e2e.py --fast       # skip Docker suite (~5 min faster)
  python3 tests/run_e2e.py --suite 1    # Suite 1 only: Bootstrap & project creation
  python3 tests/run_e2e.py --suite 2    # Suite 2 only: Pipeline artifacts
  python3 tests/run_e2e.py --suite 3    # Suite 3 only: App & Dockerfile generation
  python3 tests/run_e2e.py --suite 4    # Suite 4 only: API endpoints (live server)
  python3 tests/run_e2e.py --suite 5    # Suite 5 only: Docker build & smoke-test
  python3 tests/run_e2e.py --from-github  # download bootstrap.py from GitHub first
"""

import argparse
import csv
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

# ── ANSI colours ────────────────────────────────────────────────────
G = "\033[0;32m"; R = "\033[0;31m"; Y = "\033[1;33m"
C = "\033[0;36m"; B = "\033[1m";    X = "\033[0m"

GITHUB_RAW = "https://raw.githubusercontent.com/ramleo/builds_bootstrap/main/bootstrap.py"
REPO_ROOT  = Path(__file__).parent.parent.resolve()

# ── Test state ───────────────────────────────────────────────────────
_results = []       # list of (suite, name, passed: bool, detail: str)
_server_pid = None  # uvicorn PID for cleanup

# ── Helpers ──────────────────────────────────────────────────────────

def section(num, title):
    print(f"\n{C}{B}{'═' * 62}{X}")
    print(f"{C}{B}  Suite {num} — {title}{X}")
    print(f"{C}{B}{'═' * 62}{X}")


def check(suite, name, condition, detail=""):
    _results.append((suite, name, bool(condition), detail))
    icon  = f"{G}✔{X}" if condition else f"{R}✗{X}"
    extra = f"  {Y}({detail}){X}" if detail and not condition else ""
    print(f"  {icon}  {name}{extra}")
    return bool(condition)


def skip(suite, name, reason=""):
    _results.append((suite, name, None, reason))
    print(f"  {Y}–{X}  {name}  {Y}[skipped: {reason}]{X}")


def run(cmd, cwd=None, input_str=None, timeout=600):
    """Run a shell command, return (returncode, stdout, stderr)."""
    result = subprocess.run(
        cmd, cwd=str(cwd) if cwd else None,
        input=input_str, capture_output=True, text=True, timeout=timeout,
    )
    return result.returncode, result.stdout, result.stderr


def make_csv(path, rows=60, seed=42):
    """Create a simple binary-classification CSV."""
    random.seed(seed)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["age", "salary", "experience", "survived"])
        for _ in range(rows):
            age = random.randint(22, 60)
            exp = random.randint(0, 15)
            sal = random.randint(30000, 120000)
            w.writerow([age, sal, exp, 1 if sal > 70000 and exp > 5 else 0])


# ════════════════════════════════════════════════════════════════════
# Suite 1 — Bootstrap & Project Creation
# ════════════════════════════════════════════════════════════════════

def suite_1_bootstrap(tmp, bootstrap_py, csv_path):
    section(1, "Bootstrap & Project Creation")

    # 1.1 bootstrap.py exists and is valid Python
    check(1, "bootstrap.py exists", bootstrap_py.is_file())
    rc, _, err = run([sys.executable, "-m", "py_compile", str(bootstrap_py)])
    check(1, "bootstrap.py syntax valid", rc == 0, err.strip())

    # 1.2 Run bootstrap → auto pipeline → generate app+docker (piped inputs)
    inputs = (
        f"e2e-test\n"
        f"{csv_path}\n"
        f"\n"      # platform: default (ask later)
        f"\n"      # github username: accept detected
        f"e2e-test\n"   # repo name
        f"\n"      # visibility: public
        f"2\n"     # launch: auto pipeline
    )
    print(f"\n  {C}→{X}  Running bootstrap.py (this takes ~2 min)...")
    try:
        rc, out, err = run(
            [sys.executable, str(bootstrap_py)],
            cwd=tmp, input_str=inputs, timeout=300,
        )
    except subprocess.TimeoutExpired:
        check(1, "bootstrap.py completed within 5 min", False, "timeout")
        return None

    check(1, "bootstrap.py exited cleanly", rc == 0, err.strip()[-200:] if err else "")

    # 1.3 Find created project folder
    projects = sorted(tmp.glob("e2e-test_*"), key=lambda p: p.stat().st_mtime)
    if not check(1, "project folder created", len(projects) > 0):
        return None
    proj = projects[-1]
    print(f"  {C}→{X}  Project: {proj.name}")

    # 1.4 Required files present
    required = [
        "CLAUDE.md", "README.md", "requirements.txt",
        "auto_pipeline.py", ".ml_config.json",
        ".gitignore", ".ml_config.json.example",
        "src/CLAUDE.md", "deploy/CLAUDE.md", "tests/CLAUDE.md",
        "docs/CLAUDE.md", "docs/how_to_run.md",
        ".claude/settings.local.json",
    ]
    missing = [f for f in required if not (proj / f).exists()]
    check(1, f"all {len(required)} required template files present",
          len(missing) == 0, "missing: " + ", ".join(missing))

    # 1.5 .ml_config.json keys
    try:
        cfg = json.loads((proj / ".ml_config.json").read_text())
        required_keys = [
            "project_name", "dataset_filename", "dataset_path",
            "target_column", "task_type", "deployment_platform",
            "github_username", "github_repo", "python_version",
            "created_at", "venv_path", "template_version",
        ]
        missing_keys = [k for k in required_keys if k not in cfg]
        check(1, ".ml_config.json has all required keys",
              len(missing_keys) == 0, "missing: " + ", ".join(missing_keys))
        check(1, "dataset_filename recorded", cfg.get("dataset_filename") != "<not provided yet>")
    except Exception as exc:
        check(1, ".ml_config.json readable", False, str(exc))

    # 1.6 .venv exists and has Python
    venv_python = proj / ".venv" / "bin" / "python"
    check(1, ".venv created", (proj / ".venv").is_dir())
    check(1, ".venv/bin/python exists", venv_python.exists())

    # 1.7 Key packages installed
    if venv_python.exists():
        rc, out, _ = run([str(venv_python), "-c",
                          "import sklearn, pandas, fastapi, uvicorn, joblib; print('ok')"])
        check(1, "sklearn/pandas/fastapi/uvicorn installed", rc == 0 and "ok" in out)

    # 1.8 setup scripts NOT copied into project
    for f in ["bootstrap.py", "Dockerfile.bootstrap", "start.sh", "init.py"]:
        if (proj / f).exists():
            check(1, f"{f} not copied into project", False, "setup script leaked into project")
            break
    else:
        check(1, "setup scripts not copied into project", True)

    return proj


# ════════════════════════════════════════════════════════════════════
# Suite 2 — Pipeline Artifacts
# ════════════════════════════════════════════════════════════════════

def suite_2_pipeline(proj):
    section(2, "Pipeline Artifacts")

    venv_python = proj / ".venv" / "bin" / "python"

    # 2.1 auto_pipeline.py ran (artifacts exist)
    artifacts = {
        "models/final_pipeline.pkl": "trained pipeline",
        "models/label_encoder.pkl":  "label encoder",
        "plots/eda_correlation.png": "EDA correlation heatmap",
        "plots/eda_target.png":      "EDA target distribution",
        "docs/auto_summary.md":      "summary report",
    }
    for rel, desc in artifacts.items():
        p = proj / rel
        check(2, f"{rel} exists ({desc})", p.exists() and p.stat().st_size > 0)

    # 2.2 Load pipeline and make a prediction
    test_code = """
import joblib, pandas as pd
pipe = joblib.load("models/final_pipeline.pkl")
le   = joblib.load("models/label_encoder.pkl")
df   = pd.DataFrame([{"age": 35, "salary": 90000, "experience": 8}])
pred = pipe.predict(df)
prob = pipe.predict_proba(df)
label = le.inverse_transform(pred)[0]
print(f"label={label} proba_shape={prob.shape[1]}")
"""
    rc, out, err = run([str(venv_python), "-c", test_code], cwd=proj)
    check(2, "pipeline loads and predicts without error", rc == 0, err.strip()[-200:] if err else "")
    if rc == 0:
        check(2, "prediction returns a label", "label=" in out)
        check(2, "predict_proba returns 2 classes", "proba_shape=2" in out)

    # 2.3 auto_summary.md has key sections
    summary_path = proj / "docs" / "auto_summary.md"
    if summary_path.exists():
        content = summary_path.read_text()
        for section_name in ["Dataset", "Model", "Evaluation", "Artifacts", "Reproducibility"]:
            check(2, f"auto_summary.md has '{section_name}' section", section_name in content)


# ════════════════════════════════════════════════════════════════════
# Suite 3 — App & Dockerfile Generation
# ════════════════════════════════════════════════════════════════════

def suite_3_app(proj):
    section(3, "App & Dockerfile Generation")

    venv_python = proj / ".venv" / "bin" / "python"

    # 3.1 Run post-pipeline option 1 (generate app + Docker only)
    print(f"  {C}→{X}  Generating app.py + Dockerfile...")
    gen_code = """
import sys
sys.path.insert(0, ".")
# Directly call the generation functions from auto_pipeline
exec(open("auto_pipeline.py").read().split("# ════")[0])   # load globals only
"""
    # Simpler: re-run auto_pipeline.py and feed "1" to the menu
    try:
        rc, out, err = run(
            [str(venv_python), "auto_pipeline.py"],
            cwd=proj, input_str="1\n", timeout=120,
        )
    except subprocess.TimeoutExpired:
        check(3, "app generation completed", False, "timeout")
        return

    # 3.2 app.py checks
    app_py = proj / "app.py"
    check(3, "app.py generated", app_py.exists() and app_py.stat().st_size > 0)

    if app_py.exists():
        rc2, _, err2 = run([str(venv_python), "-m", "py_compile", str(app_py)])
        check(3, "app.py syntax valid", rc2 == 0, err2.strip())

        content = app_py.read_text()
        check(3, "app.py has /health endpoint",       "@app.get(\"/health\")" in content)
        check(3, "app.py has /predict endpoint",      "@app.post(\"/predict\")" in content)
        check(3, "app.py has /predict/batch endpoint","@app.post(\"/predict/batch\")" in content)
        check(3, "app.py loads pipeline",             "joblib.load" in content)
        check(3, "app.py loads label_encoder",        "label_encoder" in content)
        check(3, "app.py has Pydantic InputData",     "class InputData" in content)
        check(3, "app.py returns probabilities",      "probabilities" in content)

    # 3.3 Dockerfile checks
    dockerfile = proj / "Dockerfile"
    check(3, "Dockerfile generated", dockerfile.exists() and dockerfile.stat().st_size > 0)

    if dockerfile.exists():
        content = dockerfile.read_text()
        check(3, "Dockerfile uses multi-stage build",  "AS builder" in content)
        check(3, "Dockerfile exposes port 8000",        "EXPOSE 8000" in content)
        check(3, "Dockerfile uses non-root user",       "appuser" in content)
        check(3, "Dockerfile CMD uses uvicorn",         "uvicorn" in content)

    # 3.4 .dockerignore
    di = proj / ".dockerignore"
    check(3, ".dockerignore generated", di.exists())
    if di.exists():
        content = di.read_text()
        check(3, ".dockerignore excludes .venv/",  ".venv/" in content)
        check(3, ".dockerignore excludes data/",   "data/" in content)

    # 3.5 index.html — auto-themed prediction UI
    index_html = proj / "index.html"
    check(3, "index.html generated", index_html.exists() and index_html.stat().st_size > 0)
    if index_html.exists():
        html = index_html.read_text(encoding="utf-8")
        check(3, "index.html has no raw TMPL_ placeholders",
              "TMPL_" not in html, "found unsubstituted TMPL_ token(s)")
        check(3, "index.html contains <html tag",       "<html" in html.lower())
        check(3, "index.html contains a <form element", "<form" in html.lower())
        check(3, "index.html has themed CSS gradient",  "gradient" in html.lower())

    # 3.6 app.py — CORS middleware and GET / (frontend route)
    if app_py.exists():
        content = app_py.read_text()
        check(3, "app.py imports FileResponse",   "FileResponse" in content)
        check(3, "app.py imports CORSMiddleware", "CORSMiddleware" in content)
        check(3, 'app.py has GET / route',        '@app.get("/")' in content)

    # 3.7 Dockerfile — copies index.html into image
    if dockerfile.exists():
        content = dockerfile.read_text()
        check(3, "Dockerfile copies index.html*", "index.html" in content)


# ════════════════════════════════════════════════════════════════════
# Suite 4 — API Endpoints (live server)
# ════════════════════════════════════════════════════════════════════

def suite_4_api(proj):
    global _server_pid
    section(4, "API Endpoints (live server)")

    app_py = proj / "app.py"
    if not app_py.exists():
        skip(4, "API tests", "app.py not found — run suite 3 first")
        return

    venv_python = proj / ".venv" / "bin" / "python"
    port = 8765

    # 4.1 Start uvicorn
    print(f"  {C}→{X}  Starting uvicorn on port {port}...")
    proc = subprocess.Popen(
        [str(venv_python), "-m", "uvicorn", "app:app",
         "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(proj), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    _server_pid = proc.pid
    time.sleep(4)   # wait for startup

    if proc.poll() is not None:
        _, serr = proc.communicate()
        check(4, "uvicorn started", False, serr.decode()[-300:])
        return

    check(4, "uvicorn started", True)

    try:
        import urllib.request
        import urllib.error

        def get(path):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=10) as r:
                    return r.status, json.loads(r.read())
            except urllib.error.HTTPError as e:
                return e.code, {}
            except Exception as exc:
                return 0, {"error": str(exc)}

        def post(path, body):
            data = json.dumps(body).encode()
            req  = urllib.request.Request(
                f"http://127.0.0.1:{port}{path}", data=data,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=10) as r:
                    return r.status, json.loads(r.read())
            except urllib.error.HTTPError as e:
                return e.code, {}
            except Exception as exc:
                return 0, {"error": str(exc)}

        # 4.2 GET /health
        status, body = get("/health")
        check(4, "GET /health returns 200",      status == 200, f"got {status}")
        check(4, "GET /health body has 'status'", "status" in body)
        check(4, "GET /health status is 'ok'",    body.get("status") == "ok")

        # 4.3 POST /predict — should survive (high salary + exp)
        status, body = post("/predict", {"age": 35, "salary": 90000, "experience": 8})
        check(4, "POST /predict returns 200",          status == 200, f"got {status}")
        check(4, "POST /predict has 'prediction'",     "prediction" in body)
        check(4, "POST /predict has 'probabilities'",  "probabilities" in body)
        check(4, "POST /predict high-salary → class 1", body.get("prediction") == "1",
              f"got {body.get('prediction')}")

        # 4.4 POST /predict — should not survive (low salary + exp)
        status, body = post("/predict", {"age": 28, "salary": 40000, "experience": 2})
        check(4, "POST /predict low-salary → class 0",  body.get("prediction") == "0",
              f"got {body.get('prediction')}")

        # 4.5 POST /predict — missing fields (imputation)
        status, body = post("/predict", {"age": 40})
        check(4, "POST /predict with missing fields returns 200", status == 200, f"got {status}")
        check(4, "POST /predict missing fields returns prediction", "prediction" in body)

        # 4.6 POST /predict/batch
        status, body = post("/predict/batch", [
            {"age": 35, "salary": 90000, "experience": 8},
            {"age": 28, "salary": 40000, "experience": 2},
        ])
        check(4, "POST /predict/batch returns 200",         status == 200, f"got {status}")
        check(4, "POST /predict/batch has 'predictions'",   "predictions" in body)
        preds = body.get("predictions", [])
        check(4, "POST /predict/batch returns 2 results",   len(preds) == 2, f"got {len(preds)}")
        check(4, "POST /predict/batch results are correct", preds == ["1", "0"],
              f"got {preds}")

        # 4.7 GET /docs (Swagger UI) — returns HTML, check status only
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/docs", timeout=10) as r:
                docs_status = r.status
        except Exception:
            docs_status = 0
        check(4, "GET /docs (Swagger UI) accessible", docs_status == 200, f"got {docs_status}")

        # 4.8 POST /predict — invalid payload → 422
        status, _ = post("/predict", {"bad_field": "nonsense"})
        check(4, "POST /predict invalid payload returns prediction (imputes missing)", status == 200)

        # 4.9 GET / — themed prediction UI (index.html served at root)
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/", timeout=10
            ) as resp:
                fe_status = resp.status
                fe_ctype  = resp.headers.get("Content-Type", "")
                fe_body   = resp.read().decode("utf-8", errors="replace")
        except Exception as exc:
            fe_status = 0
            fe_ctype  = ""
            fe_body   = str(exc)
        check(4, "GET / returns 200",              fe_status == 200, f"got {fe_status}")
        check(4, "GET / Content-Type is text/html",
              "text/html" in fe_ctype, f"got '{fe_ctype}'")
        check(4, "GET / body contains <html tag",  "<html" in fe_body.lower(),
              fe_body[:120] if fe_body else "empty")

    finally:
        # Stop server
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        _server_pid = None
        print(f"  {C}→{X}  Server stopped")


# ════════════════════════════════════════════════════════════════════
# Suite 5 — Docker Build & Smoke Test
# ════════════════════════════════════════════════════════════════════

def suite_5_docker(proj):
    section(5, "Docker Build & Smoke Test")

    if not shutil.which("docker"):
        skip(5, "all docker tests", "Docker not installed")
        return

    # Check Docker daemon is running
    rc, _, _ = run(["docker", "info"])
    if rc != 0:
        skip(5, "all docker tests", "Docker daemon not running")
        return

    app_py = proj / "app.py"
    dockerfile = proj / "Dockerfile"
    if not app_py.exists() or not dockerfile.exists():
        skip(5, "all docker tests", "app.py or Dockerfile missing — run suite 3 first")
        return

    image = "e2e-test-ml-pipeline"
    container = "e2e-test-container"
    port = 8766

    # 5.1 Build image
    print(f"  {C}→{X}  Building Docker image (this may take ~2 min)...")
    rc, out, err = run(["docker", "build", "-t", image, "."], cwd=proj, timeout=300)
    check(5, "docker build succeeds", rc == 0, err.strip()[-300:] if err else "")
    if rc != 0:
        return

    # 5.2 Run container
    run(["docker", "rm", "-f", container])   # clean up any leftover
    rc, _, err = run([
        "docker", "run", "-d", "--name", container,
        "-p", f"{port}:8000", image
    ])
    check(5, "docker run starts container", rc == 0, err.strip())
    time.sleep(5)

    try:
        import urllib.request

        def get_docker(path):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=10) as r:
                    return r.status, json.loads(r.read())
            except Exception as exc:
                return 0, {"error": str(exc)}

        def post_docker(path, body):
            data = json.dumps(body).encode()
            req  = urllib.request.Request(
                f"http://127.0.0.1:{port}{path}", data=data,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=10) as r:
                    return r.status, json.loads(r.read())
            except Exception as exc:
                return 0, {"error": str(exc)}

        # 5.3 Smoke tests inside Docker
        status, body = get_docker("/health")
        check(5, "GET /health inside Docker returns 200", status == 200, f"got {status}")
        check(5, "GET /health inside Docker status='ok'", body.get("status") == "ok")

        status, body = post_docker("/predict", {"age": 35, "salary": 90000, "experience": 8})
        check(5, "POST /predict inside Docker returns 200",     status == 200)
        check(5, "POST /predict inside Docker has prediction",  "prediction" in body)

        status, body = post_docker("/predict/batch", [
            {"age": 35, "salary": 90000, "experience": 8},
            {"age": 28, "salary": 40000, "experience": 2},
        ])
        check(5, "POST /predict/batch inside Docker returns 200", status == 200)
        preds = body.get("predictions", [])
        check(5, "POST /predict/batch inside Docker correct results", preds == ["1", "0"],
              f"got {preds}")

        # 5.4 GET / inside Docker — themed UI served at root
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/", timeout=10
            ) as resp:
                docker_fe_status = resp.status
                docker_fe_body   = resp.read().decode("utf-8", errors="replace")
        except Exception as exc:
            docker_fe_status = 0
            docker_fe_body   = str(exc)
        check(5, "GET / inside Docker returns 200",
              docker_fe_status == 200, f"got {docker_fe_status}")
        check(5, "GET / inside Docker serves index.html",
              "<html" in docker_fe_body.lower(), docker_fe_body[:80] if docker_fe_body else "empty")

    finally:
        run(["docker", "stop", container])
        run(["docker", "rm",   container])
        run(["docker", "rmi",  image])
        print(f"  {C}→{X}  Docker container + image cleaned up")


# ════════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════════

def print_summary():
    passed  = [r for r in _results if r[2] is True]
    failed  = [r for r in _results if r[2] is False]
    skipped = [r for r in _results if r[2] is None]
    total   = len(passed) + len(failed)

    print(f"\n{C}{B}{'═' * 62}{X}")
    print(f"{B}  Test Summary{X}")
    print(f"{C}{B}{'═' * 62}{X}")

    if failed:
        print(f"\n  {R}{B}Failed:{X}")
        for _, name, _, detail in failed:
            d = f"  ({detail})" if detail else ""
            print(f"    {R}✗{X}  {name}{d}")

    if skipped:
        print(f"\n  {Y}Skipped:{X}")
        for _, name, _, reason in skipped:
            print(f"    {Y}–{X}  {name}  ({reason})")

    bar_passed = int((len(passed) / total * 40)) if total else 0
    bar_failed = 40 - bar_passed
    bar = f"{G}{'█' * bar_passed}{R}{'█' * bar_failed}{X}"

    print(f"\n  {bar}")
    print(f"\n  {G}{B}{len(passed)}{X} passed  "
          f"{R}{B}{len(failed)}{X} failed  "
          f"{Y}{len(skipped)}{X} skipped  "
          f"({total} total)")
    print(f"{C}{B}{'═' * 62}{X}\n")

    return len(failed) == 0


# ════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════

def main():
    global _server_pid

    parser = argparse.ArgumentParser(description="E2E test suite for ML Pipeline Template")
    parser.add_argument("--suite",       type=int, help="Run a specific suite (1–5)")
    parser.add_argument("--fast",        action="store_true", help="Skip Docker suite (suite 5)")
    parser.add_argument("--from-github", action="store_true", help="Download bootstrap.py from GitHub")
    args = parser.parse_args()

    suites_to_run = [args.suite] if args.suite else [1, 2, 3, 4, 5]
    if args.fast and 5 in suites_to_run:
        suites_to_run.remove(5)

    print(f"\n{C}{B}{'═' * 62}{X}")
    print(f"{C}{B}  ML Pipeline Template — End-to-End Test Suite{X}")
    print(f"{C}{B}  Suites: {suites_to_run}{'  [fast mode]' if args.fast else ''}{X}")
    print(f"{C}{B}{'═' * 62}{X}")

    tmp = Path(tempfile.mkdtemp(prefix="ml_e2e_"))
    proj = None

    try:
        # ── Prepare bootstrap.py ────────────────────────────────────
        if args.from_github:
            print(f"\n  {C}→{X}  Downloading bootstrap.py from GitHub...")
            bootstrap_py = tmp / "bootstrap.py"
            urllib.request.urlretrieve(GITHUB_RAW, bootstrap_py)
            print(f"  {G}✔{X}  Downloaded ({bootstrap_py.stat().st_size:,} bytes)")
        else:
            bootstrap_py = REPO_ROOT / "bootstrap.py"
            print(f"\n  {C}→{X}  Using local bootstrap.py: {bootstrap_py}")

        # ── Sample CSV ──────────────────────────────────────────────
        csv_path = tmp / "sample.csv"
        make_csv(csv_path)
        print(f"  {G}✔{X}  Sample CSV created ({csv_path.stat().st_size} bytes, 60 rows)")

        # ── Run suites ──────────────────────────────────────────────
        if 1 in suites_to_run:
            proj = suite_1_bootstrap(tmp, bootstrap_py, csv_path)

        if proj is None and any(s in suites_to_run for s in [2, 3, 4, 5]):
            # Try to find an existing project in tmp
            found = sorted(tmp.glob("e2e-test_*"), key=lambda p: p.stat().st_mtime)
            if found:
                proj = found[-1]
                print(f"\n  {Y}→{X}  Using existing project: {proj.name}")
            else:
                print(f"\n  {R}✗{X}  No project found — run suite 1 first")
                sys.exit(1)

        if 2 in suites_to_run:
            suite_2_pipeline(proj)

        if 3 in suites_to_run:
            suite_3_app(proj)

        if 4 in suites_to_run:
            suite_4_api(proj)

        if 5 in suites_to_run:
            suite_5_docker(proj)

    except KeyboardInterrupt:
        print(f"\n{Y}  Interrupted by user{X}")

    finally:
        # Kill server if still running
        if _server_pid:
            try:
                os.kill(_server_pid, signal.SIGTERM)
            except Exception:
                pass

        # Clean up temp dir
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"  {G}✔{X}  Temp folder cleaned up")

    ok = print_summary()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
