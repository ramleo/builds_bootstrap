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

# Optional fast-tree boosters — skipped gracefully if not installed
_HAS_LGBM = False
_HAS_XGB  = False
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    _HAS_LGBM = True
    _ok("LightGBM available")
except ImportError:
    _warn("LightGBM not installed — skipping (pip install lightgbm)")
try:
    from xgboost import XGBClassifier, XGBRegressor
    _HAS_XGB = True
    _ok("XGBoost available")
except ImportError:
    _warn("XGBoost not installed — skipping (pip install xgboost)")

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
    if _HAS_LGBM:
        candidates.append((
            "LightGBM",
            LGBMClassifier(random_state=42, verbose=-1),
            {"model__n_estimators": [100, 300], "model__learning_rate": [0.05, 0.1],
             "model__num_leaves": [31, 63]},
        ))
    if _HAS_XGB:
        candidates.append((
            "XGBoost",
            XGBClassifier(random_state=42, eval_metric="logloss", verbosity=0),
            {"model__n_estimators": [100, 300], "model__learning_rate": [0.05, 0.1],
             "model__max_depth": [4, 6]},
        ))
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
    if _HAS_LGBM:
        candidates.append((
            "LightGBM",
            LGBMRegressor(random_state=42, verbose=-1),
            {"model__n_estimators": [100, 300], "model__learning_rate": [0.05, 0.1],
             "model__num_leaves": [31, 63]},
        ))
    if _HAS_XGB:
        candidates.append((
            "XGBoost",
            XGBRegressor(random_state=42, verbosity=0),
            {"model__n_estimators": [100, 300], "model__learning_rate": [0.05, 0.1],
             "model__max_depth": [4, 6]},
        ))
    scoring = "r2"

results = []

# ── Optional MLflow experiment tracking ──────────────────────────────────────
_mlflow_ok = False
try:
    import mlflow, mlflow.sklearn
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment(project_name)
    _mlflow_ok = True
    _ok("MLflow tracking enabled  (mlruns/ folder)")
except ImportError:
    _warn("MLflow not installed — skipping tracking (pip install mlflow)")

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
        if _mlflow_ok:
            with mlflow.start_run(run_name=name):
                mlflow.log_param("model", name)
                mlflow.log_params({f"param_{k}": v for k, v in best_params.items()})
                mlflow.log_metric(f"cv_{scoring}", best_score)
                mlflow.sklearn.log_model(gs.best_estimator_, "model")
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

# ── Feature importance (for SHAP-style explanation in the UI) ────────────────
try:
    import json as _json
    _model_obj = final_pipe.named_steps["model"]
    try:
        _feat_names = list(final_pipe.named_steps["preprocessor"].get_feature_names_out())
    except Exception:
        _feat_names = num_feats + cat_feats
    _importances = None
    if hasattr(_model_obj, "feature_importances_"):
        _importances = _model_obj.feature_importances_.tolist()
    elif hasattr(_model_obj, "coef_"):
        import numpy as _np2
        _c = _model_obj.coef_
        _importances = _np2.abs(_c[0] if _c.ndim > 1 else _c).tolist()
    if _importances and _feat_names:
        _fi_max = max(abs(v) for v in _importances) or 1
        _fi_norm = sorted(
            [{"feature": n, "importance": round(abs(v) / _fi_max, 4)}
             for n, v in zip(_feat_names, _importances)],
            key=lambda x: x["importance"], reverse=True
        )[:15]
        _fi_path = MODELS_DIR / "feature_importance.json"
        _fi_path.write_text(_json.dumps(_fi_norm, indent=2))
        _ok(f"Feature importance → {_fi_path}  ({len(_fi_norm)} features)")
except Exception as _fie:
    _warn(f"Feature importance skipped: {_fie}")

# ════════════════════════════════════════════════════════════════════
# 6c. Save regression metrics (R², MAE, RMSE, target stats)
# ════════════════════════════════════════════════════════════════════
if task_type == 'regression':
    try:
        from sklearn.metrics import r2_score as _r2s, mean_absolute_error as _mae_fn, mean_squared_error as _mse_fn
        import numpy as _np2
        _p = final_pipe.predict(X_test)
        _metrics_out = {
            'task': 'regression',
            'r2':   round(float(_r2s(y_test, _p)), 4),
            'mae':  round(float(_mae_fn(y_test, _p)), 2),
            'rmse': round(float(_mse_fn(y_test, _p)**0.5), 2),
            'target_mean': round(float(_np2.mean(y)), 2),
            'target_std':  round(float(_np2.std(y)), 2),
            'target_min':  round(float(_np2.min(y)), 2),
            'target_max':  round(float(_np2.max(y)), 2),
        }
        import json as _js
        _mp = MODELS_DIR / 'metrics.json'
        _mp.write_text(_js.dumps(_metrics_out, indent=2))
        _ok(f'Metrics → {_mp}  R²={_metrics_out["r2"]}  MAE={_metrics_out["mae"]}  RMSE={_metrics_out["rmse"]}')
    except Exception as _me:
        _warn(f'Metrics save skipped: {_me}')

# Feature ranges for slider UI
try:
    import json as _js2
    _rng_rows = {}
    for _c in num_feats:
        if _c in X.columns:
            _mn, _mx = float(X[_c].min()), float(X[_c].max())
            _pad = (_mx - _mn) * 0.05
            _mag = 10 ** max(0, int(len(str(int(_mx))) - 2))
            _rng_rows[_c] = {'min': round(max(0,_mn),2), 'max': round(_mx+_pad,2), 'step': max(1,_mag//100)}
    (_rp := MODELS_DIR / 'feature_ranges.json').write_text(_js2.dumps(_rng_rows, indent=2))
    _ok(f'Feature ranges → {_rp}')
except Exception as _re:
    _warn(f'Feature ranges skipped: {_re}')

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
            ' placeholder="e.g. 0" step="any" min="0" required>'
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
            ' placeholder="Enter value" required>'
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
        '    .inp:-webkit-autofill,\n'
        '    .inp:-webkit-autofill:hover,\n'
        '    .inp:-webkit-autofill:focus {\n'
        '      -webkit-text-fill-color: #fff;\n'
        '      -webkit-box-shadow: 0 0 0px 1000px rgba(255,255,255,.07) inset;\n'
        '      transition: background-color 5000s ease-in-out 0s;\n'
        '    }\n'
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
        '    .slider { -webkit-appearance:none; appearance:none; width:100%; height:4px; margin-top:6px;\n'
        '      background:rgba(255,255,255,.13); border-radius:99px; outline:none; cursor:pointer; display:block; }\n'
        '    .slider::-webkit-slider-thumb { -webkit-appearance:none; width:14px; height:14px;\n'
        '      border-radius:50%; background:TMPL_BTN; cursor:pointer; box-shadow:0 0 6px TMPL_BTN66; }\n'
        '    .slider::-moz-range-thumb { width:14px; height:14px; border-radius:50%;\n'
        '      background:TMPL_BTN; cursor:pointer; border:none; }\n'
        '    .hist-table { width:100%; border-collapse:collapse; font-size:.8rem; }\n'
        '    .hist-table th { padding:8px 12px; text-align:left; color:rgba(255,255,255,.3); font-size:.68rem;\n'
        '      font-weight:700; text-transform:uppercase; letter-spacing:.06em; border-bottom:1px solid rgba(255,255,255,.08); white-space:nowrap; }\n'
        '    .hist-table td { padding:9px 12px; color:rgba(255,255,255,.65); border-bottom:1px solid rgba(255,255,255,.04); white-space:nowrap; }\n'
        '    .hist-table tr:last-child td { border-bottom:none; }\n'
        '    .hist-table tr:hover td { background:rgba(255,255,255,.03); }\n'
        '    .hist-table .pred-col { color:#fff; font-weight:700; }\n'
        '    .hist-table .run-col  { color:rgba(255,255,255,.25); }\n'
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
        '          <div style="margin-top:16px;padding-top:16px;border-top:1px solid rgba(255,255,255,.08)">\n'
        '            <div style="color:rgba(255,255,255,.25);font-size:.65rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;margin-bottom:8px">Batch Predict via CSV</div>\n'
        '            <label style="display:flex;align-items:center;gap:10px;padding:10px 14px;background:rgba(255,255,255,.04);border:1px dashed rgba(255,255,255,.2);border-radius:10px;cursor:pointer">\n'
        '              <span style="font-size:1.1rem">&#128196;</span>\n'
        '              <span id="uploadTxt" style="color:rgba(255,255,255,.5);font-size:.83rem">Drop CSV or click to upload</span>\n'
        '              <input type="file" accept=".csv" id="csvFile" style="display:none">\n'
        '            </label>\n'
        '            <div id="batchResult" style="display:none;margin-top:8px;max-height:140px;overflow-y:auto;background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.07);border-radius:8px;padding:10px;font-size:.78rem;color:rgba(255,255,255,.65)"></div>\n'
        '          </div>\n'
        '\n'
        '          <div style="margin-top:24px;padding-top:20px;border-top:1px solid rgba(255,255,255,.08)">\n'
        '            <div style="color:rgba(255,255,255,.25);font-size:.65rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;margin-bottom:10px">About this model</div>\n'
        '            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px">\n'
        '              <div class="mini-stat">\n'
        '                <div style="color:rgba(255,255,255,.35);font-size:.7rem;margin-bottom:4px">Algorithm</div>\n'
        '                <div style="color:#fff;font-weight:600;font-size:.82rem">TMPL_ALGO</div>\n'
        '              </div>\n'
        '              <div class="mini-stat">\n'
        '                <div style="color:rgba(255,255,255,.35);font-size:.7rem;margin-bottom:4px" id="metricLabel">TMPL_METRIC_LABEL</div>\n'
        '                <div style="color:TMPL_ACC_COLOR;font-weight:700;font-size:.82rem" id="metricValue">TMPL_ACCURACY</div>\n'
        '              </div>\n'
        '              <div class="mini-stat">\n'
        '                <div style="color:rgba(255,255,255,.35);font-size:.7rem;margin-bottom:4px" id="metricLabel2">TMPL_METRIC_LABEL2</div>\n'
        '                <div style="color:#fff;font-weight:600;font-size:.82rem" id="metricValue2">TMPL_METRIC_VAL2</div>\n'
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
        '            <div id="benchmarkBadge" style="display:none;font-size:.78rem;font-weight:600;padding:4px 14px;border-radius:99px;margin-bottom:12px"></div>\n'
        '            <div id="ciSec" style="display:none;margin-bottom:18px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.07);border-radius:12px;padding:14px">\n'
        '              <div style="color:rgba(255,255,255,.28);font-size:.65rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;margin-bottom:10px">Confidence Interval <span style="font-weight:400;opacity:.6">(±1σ &middot; ~68%)</span></div>\n'
        '              <div style="display:flex;justify-content:space-between;font-size:.8rem;color:rgba(255,255,255,.5);margin-bottom:6px"><span id="ciLower">—</span><span style="color:rgba(255,255,255,.3)">range</span><span id="ciUpper">—</span></div>\n'
        '              <div style="height:8px;background:rgba(255,255,255,.06);border-radius:99px;position:relative">\n'
        '                <div style="position:absolute;height:100%;width:100%;border-radius:99px;background:linear-gradient(90deg,TMPL_BTN59,TMPL_BTN99)"></div>\n'
        '                <div id="ciDot" style="position:absolute;width:12px;height:12px;border-radius:50%;background:TMPL_BTN;top:50%;transform:translate(-50%,-50%);box-shadow:0 0 8px TMPL_BTN66"></div>\n'
        '              </div>\n'
        '            </div>\n'
        '            <div id="probSec" style="display:none;margin-bottom:20px">\n'
        '              <div style="color:rgba(255,255,255,.28);font-size:.65rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;margin-bottom:12px">Class Probabilities</div>\n'
        '              <div id="probBars"></div>\n'
        '            </div>\n'
        '\n'
        '            <div id="fiSec" style="display:none;flex-direction:column;flex:1;margin-bottom:16px">\n'
        '              <div style="color:rgba(255,255,255,.28);font-size:.65rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;margin-bottom:10px">Key Factors</div>\n'
        '              <div id="fiBars"></div>\n'
        '            </div>\n'
        '\n'
        '            <div style="background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.07);border-radius:12px;padding:14px;margin-top:12px">\n'
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
        '      <!-- Comparison table -->\n'
        '      <div id="histSection" style="display:none;margin-top:24px">\n'
        '        <div class="glass" style="border-radius:16px;padding:20px 24px">\n'
        '          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px">\n'
        '            <div><span style="color:#fff;font-weight:700;font-size:.95rem">Prediction History</span>\n'
        '            <span id="histCount" style="color:rgba(255,255,255,.3);font-size:.8rem;margin-left:8px"></span></div>\n'
        '            <button onclick="clearHistory()" style="padding:5px 14px;background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.12);border-radius:8px;color:rgba(255,255,255,.55);font-size:.78rem;cursor:pointer">Clear</button>\n'
        '          </div>\n'
        '          <div style="overflow-x:auto"><table class="hist-table"><thead id="histHead"></thead><tbody id="histBody"></tbody></table></div>\n'
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
        '  var fiSec       = document.getElementById(\'fiSec\');\n'
        '  var fiBars      = document.getElementById(\'fiBars\');\n'
        '  var csvFile     = document.getElementById(\'csvFile\');\n'
        '  var batchResult = document.getElementById(\'batchResult\');\n'
        '  var uploadTxt   = document.getElementById(\'uploadTxt\');\n'
        '\n'
        '  // Populate About strip from /metrics\n'
        '  fetch(\'/metrics\').then(function(r){return r.json();}).then(function(m){\n'
        '    if(!m||!m.task) return;\n'
        '    var ml=document.getElementById(\'metricLabel\'),mv=document.getElementById(\'metricValue\');\n'
        '    var ml2=document.getElementById(\'metricLabel2\'),mv2=document.getElementById(\'metricValue2\');\n'
        '    if(m.task===\'regression\'){\n'
        '      if(m.r2!==undefined&&m.r2>=0.7){ml.textContent=\'R²\';mv.textContent=m.r2.toFixed(3);mv.style.color=m.r2>=0.9?\'#4ade80\':\'#facc15\';}\n'
        '      else if(m.mae!==undefined){ml.textContent=\'MAE\';mv.textContent=\'±\'+Math.round(m.mae);mv.style.color=\'#facc15\';}\n'
        '      if(ml2&&m.rmse!==undefined){ml2.textContent=\'Est. Error\';mv2.textContent=\'±\'+Math.round(m.rmse);}\n'
        '    }\n'
        '  }).catch(function(){});\n'
        '  // ── CSV upload → /predict/upload ───────────────────────────────────────────\n'
        '  if (csvFile) {\n'
        '    csvFile.addEventListener(\'change\', function() {\n'
        '      var f = csvFile.files[0]; if (!f) return;\n'
        '      uploadTxt.textContent = f.name;\n'
        '      var fd = new FormData(); fd.append(\'file\', f);\n'
        '      batchResult.style.display = \'block\';\n'
        '      batchResult.textContent = \'Uploading…\';\n'
        '      fetch(\'/predict/upload\', {method:\'POST\', body:fd})\n'
        '        .then(function(r){ if(!r.ok) throw new Error(\'HTTP \'+r.status); return r.json(); })\n'
        '        .then(function(d){\n'
        '          var preview = d.predictions.slice(0,20).join(\', \') + (d.count>20 ? \'…\' : \'\');\n'
        '          batchResult.innerHTML = \'<b>\'+d.count+\' predictions:</b><br>\'+preview;\n'
        '        })\n'
        '        .catch(function(e){ batchResult.textContent = \'Upload failed: \'+e.message; });\n'
        '    });\n'
        '  }\n'
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
        '    if (Object.keys(payload).length === 0) {\n'
        '      showError(\'Please fill in at least one field before predicting.\');\n'
        '      return;\n'
        '    }\n'
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
        '        msg = \'⚠ Cannot reach the API server.\\nThe server may have gone to sleep — please wait a moment.\';\n'
        '        setServerState(\'offline\', \'Server offline — retrying…\');\n'
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
        '    function _fmt(s){return s.replace(/[-_]/g,\' \').replace(/\\b\\w/g,function(c){return c.toUpperCase();});}\n'
        '    resVal.textContent = IS_CLASS ? _fmt(String(data.prediction)) : Number(data.prediction).toFixed(2);\n'
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
        '      resLabel.textContent = \'Estimated Value\';\n'
        '      probSec.style.display = \'none\';\n'
        '      if (data.ci_lower !== undefined && data.ci_upper !== undefined) {\n'
        '        var lo = data.ci_lower, hi = data.ci_upper, pred = data.prediction;\n'
        '        document.getElementById(\'ciLower\').textContent = lo < 0 ? \'0\' : lo.toFixed(2);\n'
        '        document.getElementById(\'ciUpper\').textContent = hi.toFixed(2);\n'
        '        resConf.textContent = \'Range: \' + (lo < 0 ? \'0\' : lo.toFixed(0)) + \' – \' + hi.toFixed(0) + \'  (±1σ)\';\n'
        '        var leftPct = hi > lo ? ((pred-lo)/(hi-lo)*0.8) : 0.5;\n'
        '        document.getElementById(\'ciDot\').style.left = Math.max(5,Math.min(95,leftPct*100)).toFixed(1)+\'%\';\n'
        '        document.getElementById(\'ciSec\').style.display = \'block\';\n'
        '      } else { resConf.textContent = \'\'; }\n'
        '      var m = data.metrics || {};\n'
        '      if (m.target_mean !== undefined) {\n'
        '        var sig = (data.prediction - m.target_mean) / (m.target_std || 1);\n'
        '        var bb = document.getElementById(\'benchmarkBadge\');\n'
        '        if (bb) {\n'
        '          var bl=sig<-0.5?\'Below Average\':sig>0.5?\'Above Average\':\'Near Average\';\n'
        '          var bc=sig<-0.5?\'#4ade80\':sig>0.5?\'#f87171\':\'#facc15\';\n'
        '          bb.textContent=bl; bb.style.color=bc; bb.style.background=\'rgba(255,255,255,.06)\';\n'
        '          bb.style.border=\'1px solid \'+bc+\'44\'; bb.style.display=\'inline-block\';\n'
        '        }\n'
        '      }\n'
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
        '    if (data.feature_importance && data.feature_importance.length) {\n'
        '      var fiMax = data.feature_importance[0].importance || 1;\n'
        '      fiBars.innerHTML = data.feature_importance.map(function(fi){\n'
        '        var pct = Math.round(fi.importance / fiMax * 100);\n'
        '        var opa = (0.35 + (fi.importance / fiMax) * 0.65).toFixed(2);\n'
        '        var lbl = fi.feature.replace(/x\\d+_/g,\'\').replace(/[_\\-]/g,\' \').trim();\n'
        '        return \'<div style="margin-bottom:7px"><div style="display:flex;justify-content:space-between;font-size:.76rem;margin-bottom:3px"><span style="color:rgba(255,255,255,.6)">\'+lbl+\'</span><span style="color:rgba(255,255,255,.4)">\'+pct+\'%</span></div>\'+\n'
        '               \'<div style="height:5px;background:rgba(255,255,255,.08);border-radius:99px"><div style="height:5px;border-radius:99px;background:linear-gradient(90deg,TMPL_BTN,TMPL_ACCENT);width:\'+pct+\'%"></div></div></div>\';\n'
        '      }).join(\'\');\n'
        '      fiSec.style.display = \'flex\';\n'
        '      fiSec.style.flexDirection = \'column\';\n'
        '      fiSec.style.flex = \'1\';\n'
        '    } else { fiSec.style.display = \'none\'; }\n'
        '\n'
        '    resultState.style.display = \'flex\';\n'
        '    resultState.classList.remove(\'fade-up\');\n'
        '    void resultState.offsetWidth;\n'
        '    resultState.classList.add(\'fade-up\');\n'
        '    errState.style.display = \'none\';\n'
        '    addToHistory(data);\n'
        '  }\n'
        '\n'
        '  function showError(msg) {\n'
        '    errMsg.textContent = msg;\n'
        '    errState.style.display    = \'block\';\n'
        '    resultState.style.display = \'none\';\n'
        '    emptyState.style.display  = \'none\';\n'
        '  }\n'
        '\n'
        '  // ── Sliders ────────────────────────────────────────────────\n'
        '  var _debounceTimer = null;\n'
        '  function debouncedPredict(ms) {\n'
        '    clearTimeout(_debounceTimer);\n'
        '    _debounceTimer = setTimeout(function() {\n'
        '      if (_serverReady) pForm.dispatchEvent(new Event(\'submit\',{bubbles:true,cancelable:true}));\n'
        '    }, ms || 400);\n'
        '  }\n'
        '  function initSliders(ranges) {\n'
        '    document.querySelectorAll(\'.inp[type="number"]\').forEach(function(inp) {\n'
        '      var fname = inp.name, key = fname.replace(/\\s+/g,\'_\');\n'
        '      var r = (ranges && ranges[fname]) || {min:0, max:100, step:1};\n'
        '      var lbl = inp.parentElement.querySelector(\'.inp-lbl\');\n'
        '      if (lbl && !lbl.querySelector(\'.sv\')) {\n'
        '        var sv = document.createElement(\'span\'); sv.className=\'sv\'; sv.id=\'sv_\'+key;\n'
        '        sv.style.cssText=\'float:right;color:TMPL_BTN;font-size:.73rem;font-weight:600\';\n'
        '        lbl.appendChild(sv);\n'
        '      }\n'
        '      if (!document.getElementById(\'sl_\'+key)) {\n'
        '        var sl = document.createElement(\'input\'); sl.type=\'range\'; sl.className=\'slider\';\n'
        '        sl.id=\'sl_\'+key; sl.min=r.min; sl.max=r.max; sl.step=r.step;\n'
        '        inp.parentElement.appendChild(sl);\n'
        '        var sv2 = document.getElementById(\'sv_\'+key);\n'
        '        inp.addEventListener(\'input\', function() {\n'
        '          var v=parseFloat(this.value);\n'
        '          if (!isNaN(v)&&v>=0) { if(v>parseFloat(sl.max))sl.max=Math.ceil(v*2); sl.value=v; if(sv2)sv2.textContent=v.toLocaleString(); }\n'
        '          else { if(sv2)sv2.textContent=\'\'; } debouncedPredict(420);\n'
        '        });\n'
        '        sl.addEventListener(\'input\', function() {\n'
        '          inp.value=this.value; if(sv2)sv2.textContent=Number(this.value).toLocaleString(); debouncedPredict(300);\n'
        '        });\n'
        '      }\n'
        '    });\n'
        '  }\n'
        '  fetch(\'/ranges\').then(function(r){return r.json();}).then(function(rng){initSliders(rng);}).catch(function(){initSliders(null);});\n'
        '\n'
        '  // ── Prediction history ─────────────────────────────────────\n'
        '  var _history=[], _histKeyFields=TMPL_HIST_COLS;\n'
        '  function addToHistory(data) {\n'
        '    var e={inputs:{},prediction:data.prediction,ci_lower:data.ci_lower,ci_upper:data.ci_upper,metrics:data.metrics};\n'
        '    document.querySelectorAll(\'#pForm .inp\').forEach(function(inp){if(inp.value!==\'\')e.inputs[inp.name]=inp.value;});\n'
        '    _history.push(e); renderHistory();\n'
        '  }\n'
        '  function renderHistory() {\n'
        '    if(_history.length<2) return;\n'
        '    var hs=document.getElementById(\'histSection\'); hs.style.display=\'block\';\n'
        '    document.getElementById(\'histCount\').textContent=\'(\'+_history.length+\' runs)\';\n'
        '    var cols=[\'#\'].concat(_histKeyFields).concat(IS_CLASS?[\'Prediction\',\'Confidence\']:[\'Prediction\',\'Range\',\'vs Avg\']);\n'
        '    document.getElementById(\'histHead\').innerHTML=\'<tr>\'+cols.map(function(c){return\'<th>\'+c+\'</th>\';}).join(\'\')+ \'</tr>\';\n'
        '    document.getElementById(\'histBody\').innerHTML=_history.map(function(e,i){\n'
        '      var prev=i>0?_history[i-1]:null;\n'
        '      var cells=_histKeyFields.map(function(col){\n'
        '        var v=e.inputs[col]!==undefined?e.inputs[col]:\'—\';\n'
        '        var changed=prev&&prev.inputs[col]!==e.inputs[col];\n'
        '        return \'<td style="\'+( changed?\'color:#fff;font-weight:600\':\'\')+\'">\'+v+\'</td>\';\n'
        '      });\n'
        '      var predCell=IS_CLASS?\'<td class="pred-col">\'+String(e.prediction)+\'</td>\'+\n'
        '        \'<td>\'+( typeof e.probabilities!==\'undefined\'?\'—\':\'—\')+\'</td>\' :\n'
        '        \'<td class="pred-col">\'+Number(e.prediction).toFixed(2)+\'</td>\'+\n'
        '        \'<td>\'+( e.ci_lower!==undefined?(Math.max(0,e.ci_lower).toFixed(0)+\' – \'+e.ci_upper.toFixed(0)):\'—\')+\'</td>\'+\n'
        '        (function(){var m=e.metrics||{};if(m.target_mean===undefined)return\'<td>—</td>\';\n'
        '          var s=(e.prediction-m.target_mean)/(m.target_std||1);\n'
        '          var l=s<-0.5?\'↓ Below\':s>0.5?\'↑ Above\':\'→ Avg\';\n'
        '          var c=s<-0.5?\'#4ade80\':s>0.5?\'#f87171\':\'#facc15\';\n'
        '          return\'<td style="color:\'+c+\';font-weight:600">\'+l+\'</td>\';})();\n'
        '      return \'<tr><td class="run-col">\'+( i+1)+\'</td>\'+cells.join(\'\')+ predCell+\'</tr>\';\n'
        '    }).join(\'\');\n'
        '  }\n'
        '  function clearHistory(){ _history=[]; document.getElementById(\'histSection\').style.display=\'none\'; }\n'
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
        ("TMPL_METRIC_LABEL",  "R²" if task_type == "classification" else "R² / MAE"),
        ("TMPL_METRIC_LABEL2", "Classes" if task_type == "classification" else "Est. Error"),
        ("TMPL_METRIC_VAL2",   classes_count if task_type == "classification" else "—"),
        ("TMPL_BTN59",         btn + "59"),
        ("TMPL_BTN99",         btn + "99"),
        ("TMPL_BTN66",         btn + "66"),
        ("TMPL_ALGO",          algo_str),
        ("TMPL_FIELDS",        fields_html),
        ("TMPL_HIST_COLS",     str([f for f in (num_feats + cat_feats)[:4] if not any(f.lower().replace("_","").startswith(x) for x in ["id","pass","serial"])])),
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
        'from fastapi import FastAPI, UploadFile, File, HTTPException',
        'from fastapi.responses import FileResponse',
        'from fastapi.middleware.cors import CORSMiddleware',
        'from pydantic import BaseModel',
        'from typing import Optional, List',
        'import joblib, pandas as pd, io, json, os',
        '',
        'app = FastAPI(title="ML Prediction API")',
        'app.add_middleware(',
        '    CORSMiddleware,',
        '    allow_origins=["*"],',
        '    allow_methods=["*"],',
        '    allow_headers=["*"],',
        ')',
        'pipeline = joblib.load("models/final_pipeline.pkl")',
        '_fi_file = "models/feature_importance.json"',
        '_feature_importance = json.load(open(_fi_file)) if os.path.exists(_fi_file) else []',
        '_metrics  = json.load(open("models/metrics.json")) if os.path.exists("models/metrics.json") else {}',
        '_rmse     = _metrics.get("rmse")',
        '_ranges   = json.load(open("models/feature_ranges.json")) if os.path.exists("models/feature_ranges.json") else {}',
    ]
    if task_type == "classification":
        lines.append('label_encoder = joblib.load("models/label_encoder.pkl")')
    import re as _re_ap
    _id_set = {'id','index','serial','rowid','row','uuid','guid','pk','key','rid','sid'}
    def _is_id(n): s=_re_ap.sub(r'[^a-z]','',n.lower()); return s in _id_set or (s.endswith('id') and len(s)<=6)
    _id_col_names = [f for f in num_feats + cat_feats if _is_id(f)]
    _non_id_num   = [f for f in num_feats if not _is_id(f)]
    _non_id_cat   = [f for f in cat_feats if not _is_id(f)]
    lines += ['', 'class InputData(BaseModel):']
    for feat in _non_id_num:
        lines.append('    ' + feat + ': Optional[float] = None')
    for feat in _non_id_cat:
        lines.append('    ' + feat + ': Optional[str] = None')
    if not _non_id_num and not _non_id_cat:
        lines.append('    pass')
    if _id_col_names:
        lines.append('')
        lines.append('_ID_COLS = ' + repr(_id_col_names) + '  # injected as NaN at predict time')
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
        '    for col in _ID_COLS: df[col] = float("nan")',
        '    pred = pipeline.predict(df)[0]',
    ]
    if task_type == "classification":
        lines += [
            '    label = label_encoder.inverse_transform([pred])[0]',
            '    proba = pipeline.predict_proba(df)[0].tolist()',
            '    return {"prediction": str(label), "probabilities": proba,',
            '            "feature_importance": _feature_importance}',
        ]
    else:
        lines += [
            '    result = {"prediction": float(pred), "feature_importance": _feature_importance}',
            '    if _rmse:',
            '        result["ci_lower"] = round(float(pred) - _rmse, 2)',
            '        result["ci_upper"] = round(float(pred) + _rmse, 2)',
            '    result["metrics"] = _metrics',
            '    return result',
        ]
    lines += [
        '',
        '@app.post("/predict/batch")',
        'def predict_batch(data: List[InputData]):',
        '    df = pd.DataFrame([d.dict() for d in data])',
        '    for col in _ID_COLS: df[col] = float("nan")',
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
        '@app.post("/predict/upload")',
        'async def predict_upload(file: UploadFile = File(...)):',
        '    """Upload a CSV — returns predictions for every row."""',
        '    if not file.filename.lower().endswith(".csv"):',
        '        raise HTTPException(400, "Only CSV files are accepted")',
        '    contents = await file.read()',
        '    df_up = pd.read_csv(io.BytesIO(contents))',
        '    preds_up = pipeline.predict(df_up)',
    ]
    if task_type == "classification":
        lines += [
            '    try:',
            '        labels_up = label_encoder.inverse_transform(preds_up).tolist()',
            '    except Exception:',
            '        labels_up = [str(p) for p in preds_up]',
            '    return {"count": len(labels_up), "predictions": labels_up}',
        ]
    else:
        lines += [
            '    return {"count": len(preds_up), "predictions": preds_up.tolist()}',
        ]
    lines += [
        '',
        '@app.get("/metrics")',
        'def metrics_endpoint():',
        '    return _metrics',
        '',
        '@app.get("/ranges")',
        'def ranges_endpoint():',
        '    return _ranges',
        '',
        '@app.get("/importance")',
        'def importance():',
        '    return {"feature_importance": _feature_importance}',
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