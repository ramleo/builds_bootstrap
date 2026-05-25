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

metrics = {}

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


def _generate_app(root, task_type, num_feats, cat_feats):
    """Write a working FastAPI app.py based on the fitted pipeline."""
    _print_header("Step 8 — Generating FastAPI app.py")
    lines = [
        '#!/usr/bin/env python3',
        '"""Auto-generated FastAPI prediction API."""',
        'from fastapi import FastAPI',
        'from pydantic import BaseModel',
        'from typing import Optional, List',
        'import joblib, pandas as pd',
        '',
        'app = FastAPI(title="ML Prediction API")',
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

    render_lines = [
        "services:",
        "  - type: web",
        "    name: " + proj,
        "    runtime: python",
        "    buildCommand: pip install -r requirements.txt",
        "    startCommand: uvicorn app:app --host 0.0.0.0 --port $PORT",
        "    envVars:",
        '      - key: PYTHON_VERSION',
        '        value: "3.11.0"',
    ]
    (root / "render.yaml").write_text("\n".join(render_lines) + "\n", encoding="utf-8")
    _ok("render.yaml created")

    subprocess.run(["git", "add", "render.yaml"], cwd=str(root), capture_output=True)
    subprocess.run(["git", "commit", "-m", "Add render.yaml for Render deployment"],
                   cwd=str(root), capture_output=True)
    subprocess.run(["git", "push", "origin", "main"], cwd=str(root), capture_output=True)
    _ok("render.yaml pushed to GitHub")

    live = "https://" + proj + ".onrender.com"
    print("\n" + _B + "  Go live on Render (free, ~2 minutes):" + _X)
    print("    1. Visit https://render.com → sign in with GitHub")
    print("    2. Click New + → Web Service")
    print("    3. Connect repo: " + gh_user + "/" + gh_repo)
    print("    4. Render detects render.yaml → click Create Web Service")
    print("    5. Live at: " + _G + live + _X)
    print("\n  Test once deployed:")
    print("    curl " + live + "/health")
    _ok("Render setup complete — finish in the Render dashboard")


def _post_pipeline_menu(root, cfg, task_type, pipeline):
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
_post_pipeline_menu(ROOT, cfg, task_type, final_pipe)