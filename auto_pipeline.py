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


def _detect_domain(dataset_filename, column_names, project_name=""):
    """
    Dynamically detect dataset domain from filename, column names, and project name.
    Returns a theme dict with colors, gradient, icon, description, img_overlay, and
    img_keywords.  img_keywords drive a live Unsplash image URL so every dataset
    gets a contextually relevant, unique background photo — not a hardcoded image.
    For unknown datasets, keywords are derived directly from the column names so
    even completely novel data gets a sensible visual theme.
    Scores each known domain by counting keyword matches in the combined text.
    Falls back to a smart generic theme when nothing matches.
    """
    def _norm(s):
        return str(s).lower().replace("_", " ").replace("-", " ")

    search = " ".join(
        [_norm(dataset_filename), _norm(project_name)]
        + [_norm(c) for c in column_names]
    )

    # img_overlay : semi-transparent gradient layered ON TOP of the photo so
    #               text stays readable no matter what the image looks like.
    # img_keywords: free-text query sent to Unsplash — vivid, on-theme terms
    #               chosen to consistently return strong, relevant photos.
    domains = [
        (
            ["iris", "sepal", "petal", "setosa", "versicolor", "virginica",
             "flower", "species", "botanical", "petal length", "petal width"],
            {"icon": "🌸", "name": "Botanical",
             "primary": "#4a1942", "accent": "#9c27b0",
             "btn": "#7b1fa2", "btn_hover": "#4a1942", "body_bg": "#f3e5f5",
             "gradient": "linear-gradient(135deg, #2e7d32 0%, #1b5e20 100%)",
             "img_overlay": "linear-gradient(135deg,rgba(46,125,50,0.78) 0%,rgba(27,94,32,0.78) 100%)",
             "img_keywords": "iris flowers purple nature garden botanical",
             "desc": "AI-powered botanical species classification"},
        ),
        (
            ["glucose", "insulin", "bmi", "blood", "diabetes", "cancer", "heart",
             "cholesterol", "medical", "health", "patient", "clinical", "disease",
             "pregnancies", "hemoglobin", "thyroid", "tumor", "pulse", "pressure"],
            {"icon": "🩺", "name": "Health",
             "primary": "#0d6e6e", "accent": "#00897b",
             "btn": "#00897b", "btn_hover": "#00695c", "body_bg": "#f0faf9",
             "gradient": "linear-gradient(135deg, #0d6e6e 0%, #004d40 100%)",
             "img_overlay": "linear-gradient(135deg,rgba(13,110,110,0.80) 0%,rgba(0,77,64,0.80) 100%)",
             "img_keywords": "hospital medical doctor healthcare stethoscope",
             "desc": "AI-powered health risk assessment"},
        ),
        (
            ["flight", "airline", "delay", "airport", "departure",
             "arrival", "route", "travel", "boarding"],
            {"icon": "✈️", "name": "Aviation",
             "primary": "#1565c0", "accent": "#1976d2",
             "btn": "#1976d2", "btn_hover": "#1565c0", "body_bg": "#e8f4fc",
             "gradient": "linear-gradient(135deg, #1565c0 0%, #0d47a1 100%)",
             "img_overlay": "linear-gradient(135deg,rgba(21,101,192,0.78) 0%,rgba(13,71,161,0.78) 100%)",
             "img_keywords": "airplane flying sky airport clouds",
             "desc": "AI-powered flight prediction"},
        ),
        (
            ["loan", "credit", "fraud", "income", "bank", "salary", "payment",
             "default", "financial", "mortgage", "debt", "interest", "stock",
             "revenue", "profit", "market"],
            {"icon": "💰", "name": "Finance",
             "primary": "#1a237e", "accent": "#ffa000",
             "btn": "#ffa000", "btn_hover": "#f57f17", "body_bg": "#eef0fc",
             "gradient": "linear-gradient(135deg, #1a237e 0%, #283593 100%)",
             "img_overlay": "linear-gradient(135deg,rgba(26,35,126,0.80) 0%,rgba(40,53,147,0.80) 100%)",
             "img_keywords": "finance stock market trading currency banking",
             "desc": "AI-powered financial prediction"},
        ),
        (
            ["ship", "cargo", "freight", "delivery", "container", "port",
             "logistics", "shipping", "vessel", "warehouse", "supplier"],
            {"icon": "🚢", "name": "Shipping",
             "primary": "#1b4f72", "accent": "#2e86c1",
             "btn": "#2e86c1", "btn_hover": "#1b4f72", "body_bg": "#d6eaf8",
             "gradient": "linear-gradient(135deg, #1b4f72 0%, #154360 100%)",
             "img_overlay": "linear-gradient(135deg,rgba(27,79,114,0.80) 0%,rgba(21,67,96,0.80) 100%)",
             "img_keywords": "cargo ship port container freight ocean",
             "desc": "AI-powered logistics prediction"},
        ),
        (
            ["house", "sqft", "bedroom", "bathroom", "property", "rent",
             "real estate", "floor", "garage", "neighborhood", "zip",
             "price", "lot", "dwelling"],
            {"icon": "🏠", "name": "Real Estate",
             "primary": "#5d4037", "accent": "#e53935",
             "btn": "#e53935", "btn_hover": "#c62828", "body_bg": "#fbe9e7",
             "gradient": "linear-gradient(135deg, #5d4037 0%, #3e2723 100%)",
             "img_overlay": "linear-gradient(135deg,rgba(93,64,55,0.80) 0%,rgba(62,39,35,0.80) 100%)",
             "img_keywords": "house suburb real estate property architecture",
             "desc": "AI-powered property prediction"},
        ),
        (
            ["employee", "attrition", "department", "hire", "churn",
             "satisfaction", "performance", "job", "tenure", "workforce",
             "promotion", "manager"],
            {"icon": "👤", "name": "HR Analytics",
             "primary": "#4a148c", "accent": "#8e24aa",
             "btn": "#8e24aa", "btn_hover": "#6a1b9a", "body_bg": "#f5edf9",
             "gradient": "linear-gradient(135deg, #4a148c 0%, #311b92 100%)",
             "img_overlay": "linear-gradient(135deg,rgba(74,20,140,0.80) 0%,rgba(49,27,146,0.80) 100%)",
             "img_keywords": "office team corporate workplace people collaboration",
             "desc": "AI-powered people analytics"},
        ),
        (
            ["wine", "quality", "alcohol", "acidity", "sugar", "flavor",
             "volatile", "sulphates", "density", "residual"],
            {"icon": "🍷", "name": "Quality",
             "primary": "#880e4f", "accent": "#ad1457",
             "btn": "#ad1457", "btn_hover": "#880e4f", "body_bg": "#fce4ec",
             "gradient": "linear-gradient(135deg, #880e4f 0%, #4a148c 100%)",
             "img_overlay": "linear-gradient(135deg,rgba(136,14,79,0.80) 0%,rgba(74,20,140,0.80) 100%)",
             "img_keywords": "wine vineyard grapes winery red wine bottle",
             "desc": "AI-powered quality assessment"},
        ),
        (
            ["titanic", "survived", "pclass", "embarked", "lifeboat", "survival"],
            {"icon": "⚓", "name": "Maritime",
             "primary": "#0d47a1", "accent": "#1565c0",
             "btn": "#1565c0", "btn_hover": "#0d47a1", "body_bg": "#e3f2fd",
             "gradient": "linear-gradient(135deg, #0d47a1 0%, #01579b 100%)",
             "img_overlay": "linear-gradient(135deg,rgba(13,71,161,0.78) 0%,rgba(1,87,155,0.78) 100%)",
             "img_keywords": "ocean ship sea storm waves dramatic",
             "desc": "AI-powered survival prediction"},
        ),
        (
            ["customer", "churn", "product", "purchase", "review", "rating",
             "sales", "retail", "cart", "discount", "conversion", "order"],
            {"icon": "🛒", "name": "Retail",
             "primary": "#e65100", "accent": "#ef6c00",
             "btn": "#ef6c00", "btn_hover": "#e65100", "body_bg": "#fff3e0",
             "gradient": "linear-gradient(135deg, #e65100 0%, #bf360c 100%)",
             "img_overlay": "linear-gradient(135deg,rgba(230,81,0,0.80) 0%,rgba(191,54,12,0.80) 100%)",
             "img_keywords": "shopping retail store ecommerce market",
             "desc": "AI-powered customer prediction"},
        ),
        (
            ["energy", "power", "electricity", "solar", "wind", "temperature",
             "weather", "co2", "emission", "renewable", "consumption"],
            {"icon": "⚡", "name": "Energy",
             "primary": "#e65100", "accent": "#f57f17",
             "btn": "#f57f17", "btn_hover": "#e65100", "body_bg": "#fff8e1",
             "gradient": "linear-gradient(135deg, #f9a825 0%, #e65100 100%)",
             "img_overlay": "linear-gradient(135deg,rgba(249,168,37,0.78) 0%,rgba(230,81,0,0.78) 100%)",
             "img_keywords": "solar panels wind turbine renewable energy power",
             "desc": "AI-powered energy prediction"},
        ),
        (
            ["car", "vehicle", "mileage", "engine", "horsepower", "mpg",
             "cylinders", "transmission", "fuel", "auto"],
            {"icon": "🚗", "name": "Automotive",
             "primary": "#37474f", "accent": "#546e7a",
             "btn": "#546e7a", "btn_hover": "#37474f", "body_bg": "#eceff1",
             "gradient": "linear-gradient(135deg, #37474f 0%, #263238 100%)",
             "img_overlay": "linear-gradient(135deg,rgba(55,71,79,0.80) 0%,rgba(38,50,56,0.80) 100%)",
             "img_keywords": "car automobile vehicle road speed",
             "desc": "AI-powered vehicle prediction"},
        ),
        (
            ["crop", "soil", "rainfall", "humidity", "nitrogen", "phosphorus",
             "potassium", "agriculture", "farm", "harvest", "yield"],
            {"icon": "🌾", "name": "Agriculture",
             "primary": "#33691e", "accent": "#558b2f",
             "btn": "#558b2f", "btn_hover": "#33691e", "body_bg": "#f1f8e9",
             "gradient": "linear-gradient(135deg, #33691e 0%, #1b5e20 100%)",
             "img_overlay": "linear-gradient(135deg,rgba(51,105,30,0.80) 0%,rgba(27,94,32,0.80) 100%)",
             "img_keywords": "farm field crops agriculture harvest wheat",
             "desc": "AI-powered crop prediction"},
        ),
        (
            ["attack", "threat", "malware", "intrusion", "network", "packet",
             "protocol", "firewall", "vulnerability", "exploit", "cyber"],
            {"icon": "🔒", "name": "Cybersecurity",
             "primary": "#1a1a2e", "accent": "#00e676",
             "btn": "#00c853", "btn_hover": "#1a1a2e", "body_bg": "#e8f5e9",
             "gradient": "linear-gradient(135deg, #1a1a2e 0%, #0d0d1a 100%)",
             "img_overlay": "linear-gradient(135deg,rgba(26,26,46,0.82) 0%,rgba(13,13,26,0.82) 100%)",
             "img_keywords": "cybersecurity network digital hacker binary code",
             "desc": "AI-powered security threat detection"},
        ),
        (
            ["student", "grade", "score", "exam", "pass", "fail", "gpa",
             "attendance", "education", "school", "university", "course"],
            {"icon": "🎓", "name": "Education",
             "primary": "#1565c0", "accent": "#42a5f5",
             "btn": "#1976d2", "btn_hover": "#0d47a1", "body_bg": "#e3f2fd",
             "gradient": "linear-gradient(135deg, #1565c0 0%, #0d47a1 100%)",
             "img_overlay": "linear-gradient(135deg,rgba(21,101,192,0.80) 0%,rgba(13,71,161,0.80) 100%)",
             "img_keywords": "university education students graduation campus",
             "desc": "AI-powered academic performance prediction"},
        ),
        (
            ["game", "score", "player", "team", "win", "loss", "sport",
             "match", "goal", "point", "season", "league", "tournament"],
            {"icon": "🏆", "name": "Sports",
             "primary": "#1b5e20", "accent": "#43a047",
             "btn": "#388e3c", "btn_hover": "#1b5e20", "body_bg": "#e8f5e9",
             "gradient": "linear-gradient(135deg, #1b5e20 0%, #004d00 100%)",
             "img_overlay": "linear-gradient(135deg,rgba(27,94,32,0.80) 0%,rgba(0,77,0,0.80) 100%)",
             "img_keywords": "sports stadium competition victory trophy athletics",
             "desc": "AI-powered sports performance prediction"},
        ),
    ]

    best_theme, best_score = None, 0
    for keywords, theme in domains:
        score = sum(1 for kw in keywords if kw in search)
        if score > best_score:
            best_score = score
            best_theme = theme

    if best_theme is None or best_score == 0:
        # ── Fully generic fallback: derive image keywords from the actual column
        # names and filename so even completely novel datasets get a relevant image.
        stop_words = {"id", "no", "num", "the", "and", "for", "with", "from",
                      "this", "that", "have", "data", "value", "label", "class",
                      "target", "output", "result", "flag", "code", "type"}
        raw_words = []
        for token in ([dataset_filename, project_name] + list(column_names)):
            for w in _norm(token).split():
                if len(w) > 3 and w not in stop_words:
                    raw_words.append(w)
        seen, img_kws = set(), []
        for w in raw_words:
            if w not in seen:
                seen.add(w)
                img_kws.append(w)
            if len(img_kws) == 4:
                break
        derived_query = " ".join(img_kws) if img_kws else "data science technology abstract"

        best_theme = {
            "icon": "🤖", "name": "ML",
            "primary": "#1e3a5f", "accent": "#1a73e8",
            "btn": "#1a73e8", "btn_hover": "#1558d6", "body_bg": "#eef2ff",
            "gradient": "linear-gradient(135deg, #1e3a5f 0%, #0d2137 100%)",
            "img_overlay": "linear-gradient(135deg,rgba(30,58,95,0.80) 0%,rgba(13,33,55,0.80) 100%)",
            "img_keywords": derived_query,
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
    icon     = theme["icon"]
    desc     = theme["desc"]
    primary  = theme["primary"]
    accent   = theme["accent"]
    btn      = theme["btn"]
    btn_hover = theme["btn_hover"]
    body_bg  = theme["body_bg"]
    gradient = theme["gradient"]

    # ── Build dynamic background image URL ────────────────────────────────
    # Unsplash's source endpoint returns a contextually relevant photo for any
    # free-text query — no API key required.  The semi-transparent img_overlay
    # gradient sits on top so text stays readable no matter what photo loads.
    try:
        import urllib.parse as _urlparse
        _img_kw      = theme.get("img_keywords", "")
        _img_overlay = theme.get("img_overlay", gradient)
        if _img_kw:
            _encoded  = _urlparse.quote(_img_kw)
            _img_url  = f"https://source.unsplash.com/featured/1400x560?{_encoded}"
            header_bg = f"{_img_overlay}, url('{_img_url}') center / cover no-repeat"
        else:
            header_bg = gradient
    except Exception:
        header_bg = gradient

    _ok(f"Theme  : {theme['name']}  {icon}  |  image query: \"{theme.get('img_keywords','')}\"")
    _ok(f"Colors : primary={primary}  accent={accent}  body={body_bg}")

    is_class_js = "true" if task_type == "classification" else "false"
    classes_js  = str(classes).replace("'", '"')          # valid JS array literal

    # ── Build form fields ─────────────────────────────────────────────
    fields_html = ""
    for feat in num_feats:
        lbl = feat.replace("_", " ").replace("-", " ").title()
        fields_html += (
            '\n        <div class="field">'
            '\n          <label>' + lbl + '</label>'
            '\n          <input type="number" name="' + feat + '" placeholder="e.g. 0" step="any">'
            '\n        </div>'
        )
    for feat in cat_feats:
        lbl = feat.replace("_", " ").replace("-", " ").title()
        fields_html += (
            '\n        <div class="field">'
            '\n          <label>' + lbl + '</label>'
            '\n          <input type="text" name="' + feat + '" placeholder="Enter value">'
            '\n        </div>'
        )

    # ── HTML template — uses TMPL_ placeholders, not f-string {} ─────
    # This avoids the need to escape CSS { } braces in f-strings.
    html = (
        '<!DOCTYPE html>\n'
        '<html lang="en">\n'
        '<head>\n'
        '  <meta charset="UTF-8">\n'
        '  <meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        '  <title>TMPL_TITLE</title>\n'
        '  <style>\n'
        '    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }\n'
        '    :root {\n'
        '      --primary:  TMPL_PRIMARY;\n'
        '      --accent:   TMPL_ACCENT;\n'
        '      --btn:      TMPL_BTN;\n'
        '      --btn-hov:  TMPL_BTN_HOVER;\n'
        '      --body-bg:  TMPL_BODY_BG;\n'
        '    }\n'
        '    body {\n'
        '      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;\n'
        '      background: var(--body-bg); min-height: 100vh; color: #1a1a2e;\n'
        '    }\n'
        '    /* Header */\n'
        '    .hdr {\n'
        '      background: TMPL_HEADER_BG;\n'
        '      background-size: cover;\n'
        '      background-position: center;\n'
        '      color: white;\n'
        '      padding: 52px 24px 44px; text-align: center;\n'
        '      position: relative;\n'
        '    }\n'
        '    .hdr-icon { font-size: 58px; display: block; margin-bottom: 14px; }\n'
        '    .hdr h1 { font-size: 2.1rem; font-weight: 800; letter-spacing: -0.5px; text-shadow: 0 2px 8px rgba(0,0,0,0.35); }\n'
        '    .hdr p  { opacity: 0.90; margin-top: 8px; font-size: 1.05rem; text-shadow: 0 1px 4px rgba(0,0,0,0.30); }\n'
        '    /* Container */\n'
        '    .ctr { max-width: 880px; margin: 0 auto; padding: 36px 20px 64px; }\n'
        '    /* Card */\n'
        '    .card {\n'
        '      background: white; border-radius: 20px;\n'
        '      box-shadow: 0 8px 36px rgba(0,0,0,0.10); padding: 36px;\n'
        '    }\n'
        '    .card-ttl {\n'
        '      font-size: 1.1rem; font-weight: 700; color: var(--primary);\n'
        '      margin-bottom: 24px; border-bottom: 2px solid var(--body-bg);\n'
        '      padding-bottom: 14px;\n'
        '    }\n'
        '    /* Form grid */\n'
        '    .fgrid {\n'
        '      display: grid;\n'
        '      grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));\n'
        '      gap: 18px; margin-bottom: 28px;\n'
        '    }\n'
        '    .field label {\n'
        '      display: block; font-size: 0.73rem; font-weight: 700;\n'
        '      text-transform: uppercase; letter-spacing: 0.06em;\n'
        '      color: #888; margin-bottom: 6px;\n'
        '    }\n'
        '    .field input {\n'
        '      width: 100%; padding: 11px 14px;\n'
        '      border: 2px solid #e8eaed; border-radius: 10px;\n'
        '      font-size: 0.97rem; color: #1a1a2e; outline: none;\n'
        '      background: #fafafa;\n'
        '      transition: border-color 0.18s, box-shadow 0.18s;\n'
        '    }\n'
        '    .field input:focus {\n'
        '      border-color: var(--accent);\n'
        '      box-shadow: 0 0 0 3px TMPL_ACCENT_ALPHA;\n'
        '      background: white;\n'
        '    }\n'
        '    .field input::placeholder { color: #c0c0c0; }\n'
        '    /* Button */\n'
        '    .btn-p {\n'
        '      width: 100%; padding: 15px;\n'
        '      background: var(--btn); color: white; border: none;\n'
        '      border-radius: 12px; font-size: 1.1rem; font-weight: 700;\n'
        '      cursor: pointer; letter-spacing: 0.02em;\n'
        '      box-shadow: 0 4px 14px rgba(0,0,0,0.15);\n'
        '      transition: background 0.2s, transform 0.12s, box-shadow 0.2s;\n'
        '    }\n'
        '    .btn-p:hover  { background: var(--btn-hov); box-shadow: 0 6px 18px rgba(0,0,0,0.2); }\n'
        '    .btn-p:active { transform: scale(0.99); }\n'
        '    .btn-p:disabled { opacity: 0.6; cursor: not-allowed; box-shadow: none; }\n'
        '    /* Result */\n'
        '    .result {\n'
        '      margin-top: 28px; padding: 24px 28px;\n'
        '      background: var(--body-bg); border-radius: 14px;\n'
        '      border-left: 5px solid var(--accent);\n'
        '      display: none; animation: slideIn 0.32s ease;\n'
        '    }\n'
        '    .result.show { display: block; }\n'
        '    @keyframes slideIn {\n'
        '      from { opacity: 0; transform: translateY(-10px); }\n'
        '      to   { opacity: 1; transform: translateY(0); }\n'
        '    }\n'
        '    .res-row { display: flex; align-items: baseline; gap: 12px; margin-bottom: 6px; }\n'
        '    .res-lbl { font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #aaa; }\n'
        '    .res-val { font-size: 2rem; font-weight: 800; color: var(--primary); }\n'
        '    /* Probability bars */\n'
        '    .prob-sec { margin-top: 18px; }\n'
        '    .prob-ttl { font-size: 0.73rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #bbb; margin-bottom: 12px; }\n'
        '    .prob-item { margin-bottom: 10px; }\n'
        '    .prob-hdr  { display: flex; justify-content: space-between; font-size: 0.85rem; color: #555; margin-bottom: 5px; font-weight: 500; }\n'
        '    .prob-bg   { height: 10px; background: #e5e7eb; border-radius: 5px; overflow: hidden; }\n'
        '    .prob-fill {\n'
        '      height: 10px; border-radius: 5px; background: var(--accent);\n'
        '      transition: width 0.6s cubic-bezier(.4,0,.2,1); width: 0;\n'
        '    }\n'
        '    .err { color: #dc2626; font-size: 0.9rem; font-weight: 500; margin-top: 8px; }\n'
        '    /* Spinner */\n'
        '    .spin {\n'
        '      display: inline-block; width: 17px; height: 17px;\n'
        '      border: 3px solid rgba(255,255,255,0.35);\n'
        '      border-top-color: white; border-radius: 50%;\n'
        '      animation: rot 0.75s linear infinite;\n'
        '      vertical-align: middle; margin-right: 8px;\n'
        '    }\n'
        '    @keyframes rot { to { transform: rotate(360deg); } }\n'
        '    /* Footer */\n'
        '    .ftr { text-align: center; padding: 28px; color: #ccc; font-size: 0.82rem; }\n'
        '    .ftr a { color: var(--accent); text-decoration: none; font-weight: 500; }\n'
        '    .ftr a:hover { text-decoration: underline; }\n'
        '    @media (max-width: 520px) {\n'
        '      .hdr h1 { font-size: 1.5rem; }\n'
        '      .card   { padding: 22px 16px; }\n'
        '      .res-val { font-size: 1.5rem; }\n'
        '    }\n'
        '  </style>\n'
        '</head>\n'
        '<body>\n'
        '\n'
        '<div class="hdr">\n'
        '  <span class="hdr-icon">TMPL_ICON</span>\n'
        '  <h1>TMPL_TITLE</h1>\n'
        '  <p>TMPL_DESC</p>\n'
        '</div>\n'
        '\n'
        '<div class="ctr">\n'
        '  <div class="card">\n'
        '    <div class="card-ttl">Enter values and click Predict</div>\n'
        '    <form id="pForm">\n'
        '      <div class="fgrid">TMPL_FIELDS\n'
        '      </div>\n'
        '      <button type="submit" class="btn-p" id="pBtn">&#x1F50D;&nbsp; Predict</button>\n'
        '    </form>\n'
        '    <div class="result" id="result">\n'
        '      <div class="res-row">\n'
        '        <span class="res-lbl">Prediction</span>\n'
        '        <span class="res-val" id="resVal">—</span>\n'
        '      </div>\n'
        '      <div class="prob-sec" id="probSec" style="display:none">\n'
        '        <div class="prob-ttl">Confidence</div>\n'
        '        <div id="probBars"></div>\n'
        '      </div>\n'
        '      <div class="err" id="errMsg"></div>\n'
        '    </div>\n'
        '  </div>\n'
        '</div>\n'
        '\n'
        '<div class="ftr">\n'
        '  Powered by <a href="/docs" target="_blank">FastAPI ML API</a>'
        ' &nbsp;&middot;&nbsp; <a href="/health" target="_blank">Health check</a>\n'
        '</div>\n'
        '\n'
        '<script>\n'
        '  var IS_CLASS = TMPL_IS_CLASS;\n'
        '  var CLASSES  = TMPL_CLASSES;\n'
        '\n'
        '  var form    = document.getElementById("pForm");\n'
        '  var result  = document.getElementById("result");\n'
        '  var resVal  = document.getElementById("resVal");\n'
        '  var probSec = document.getElementById("probSec");\n'
        '  var probBars= document.getElementById("probBars");\n'
        '  var errMsg  = document.getElementById("errMsg");\n'
        '  var pBtn    = document.getElementById("pBtn");\n'
        '\n'
        '  form.addEventListener("submit", function(e) {\n'
        '    e.preventDefault();\n'
        '    var payload = {};\n'
        '    form.querySelectorAll("input").forEach(function(el) {\n'
        '      var v = el.value.trim();\n'
        '      if (v !== "") payload[el.name] = isNaN(v) ? v : parseFloat(v);\n'
        '    });\n'
        '\n'
        '    pBtn.disabled = true;\n'
        '    pBtn.innerHTML = \'<span class="spin"></span>Predicting&#x2026;\';\n'
        '    result.classList.remove("show");\n'
        '    errMsg.textContent = "";\n'
        '    probSec.style.display = "none";\n'
        '    probBars.innerHTML = "";\n'
        '\n'
        '    fetch("/predict", {\n'
        '      method: "POST",\n'
        '      headers: {"Content-Type": "application/json"},\n'
        '      body: JSON.stringify(payload)\n'
        '    })\n'
        '    .then(function(r) {\n'
        '      if (!r.ok) return r.text().then(function(t) { throw new Error("API " + r.status + ": " + t); });\n'
        '      return r.json();\n'
        '    })\n'
        '    .then(function(data) {\n'
        '      resVal.textContent = data.prediction;\n'
        '      if (IS_CLASS && data.probabilities) {\n'
        '        var probs  = Array.isArray(data.probabilities) ? data.probabilities : Object.values(data.probabilities);\n'
        '        var labels = CLASSES.length ? CLASSES : probs.map(function(_,i){ return "Class " + i; });\n'
        '        labels.forEach(function(cls, i) {\n'
        '          var pct = ((probs[i] || 0) * 100).toFixed(1);\n'
        '          var d = document.createElement("div");\n'
        '          d.className = "prob-item";\n'
        '          d.innerHTML = \'<div class="prob-hdr"><span>\' + cls + \'</span><span>\' + pct + \'%</span></div>\'\n'
        '                      + \'<div class="prob-bg"><div class="prob-fill" data-p="\' + pct + \'"></div></div>\';\n'
        '          probBars.appendChild(d);\n'
        '        });\n'
        '        probSec.style.display = "block";\n'
        '        setTimeout(function() {\n'
        '          document.querySelectorAll(".prob-fill").forEach(function(b) {\n'
        '            b.style.width = b.dataset.p + "%";\n'
        '          });\n'
        '        }, 60);\n'
        '      }\n'
        '      result.classList.add("show");\n'
        '    })\n'
        '    .catch(function(err) {\n'
        '      errMsg.textContent = err.message;\n'
        '      result.classList.add("show");\n'
        '    })\n'
        '    .finally(function() {\n'
        '      pBtn.disabled = false;\n'
        '      pBtn.innerHTML = "&#x1F50D;&nbsp; Predict";\n'
        '    });\n'
        '  });\n'
        '</script>\n'
        '</body>\n'
        '</html>\n'
    )

    # Substitute all placeholders — order matters: longer tokens first to avoid
    # partial replacements (e.g. TMPL_BTN_HOVER before TMPL_BTN).
    accent_alpha = accent + "33"          # hex + 20% alpha for focus ring
    replacements = [
        ("TMPL_HEADER_BG",   header_bg),
        ("TMPL_BTN_HOVER",   btn_hover),
        ("TMPL_BTN",         btn),
        ("TMPL_BODY_BG",     body_bg),
        ("TMPL_PRIMARY",     primary),
        ("TMPL_ACCENT_ALPHA", accent_alpha),
        ("TMPL_ACCENT",      accent),
        ("TMPL_IS_CLASS",    is_class_js),
        ("TMPL_CLASSES",     classes_js),
        ("TMPL_FIELDS",      fields_html),
        ("TMPL_ICON",        icon),
        ("TMPL_DESC",        desc),
        ("TMPL_TITLE",       title),
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
_post_pipeline_menu(ROOT, cfg, task_type, final_pipe, label_encoder)