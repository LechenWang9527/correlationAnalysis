# -*- coding: utf-8 -*-
"""
Step 3 Part 1: Multi-Model Benchmark, Historical Validation, and Weighted Voting.

Implements 5 base models (Ridge, ExtraTrees, CatBoost, LightGBM, XGBoost) and
weighted voting ensembles (Equal, SLSQP, Caruana greedy selection) across two
candidate pools (Trees_Only vs All_5_Models) under two anti-leakage protocols:
1. 5-fold work-order-grouped GroupKFold / Nested OOF on Aug-Oct Dev set (n=43,451).
2. Strict out-of-time blind evaluation on Nov Test set (n=9,694).
"""

import json
import logging
import os
import pickle
import sys
import time
import warnings
from typing import Any, Dict, List, Tuple

# Import lightgbm before catboost/xgboost to avoid Windows OpenMP DLL conflicts.
import lightgbm as lgb
import catboost as cb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    TargetEncoder,
)

RANDOM_STATE = 42
DATA_PATH = os.path.join("data", "processed", "clean_dataset.csv")
SCHEMA_PATH = os.path.join("data", "processed", "feature_schema.json")
LOG_PATH = os.path.join("logs", "step3_train.log")
REPORT_BENCHMARK_PATH = os.path.join("reports", "step3_model_benchmark_report.csv")
REPORT_WEIGHTS_PATH = os.path.join("reports", "step3_voting_weights.csv")
FIG_PATH = os.path.join("reports", "step3_model_benchmark_summary.png")
BUNDLE_PATH = os.path.join("models", "step3_models_bundle.pkl")

WORK_ORDER_COLS = [
    "RecipeID",
    "WorkCenterNameCoupled",
    "ActualStartMonth",
    "ActualStartDayOfWeek",
    "ActualStartHour",
    "PlannedQuantity",
    "QuantityDone",
    "ActualVsIEGap",
]

BASE_MODELS = ["Ridge", "ExtraTrees", "CatBoost", "LightGBM", "XGBoost"]
TREE_MODELS = ["CatBoost", "LightGBM", "XGBoost"]


def setup_logging(log_path: str) -> logging.Logger:
    """Configure root logger with both console and file handlers."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("step3_benchmark")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression R2, RMSE, and MAE."""
    r2 = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {"R2": r2, "RMSE": rmse, "MAE": mae}


def build_categorical_frame(
    df: pd.DataFrame,
    feature_cols: List[str],
    cat_cols: List[str],
    train_categories: Dict[str, List[str]],
) -> pd.DataFrame:
    """Apply fixed train-fold category vocabularies; unseen levels become NaN."""
    out = df[feature_cols].copy()
    for col in cat_cols:
        out[col] = pd.Categorical(
            out[col].astype(str),
            categories=train_categories[col],
        )
    return out


def probe_gpu_capabilities(
    dev_df: pd.DataFrame,
    feature_cols: List[str],
    cat_cols: List[str],
    target_col: str,
    logger: logging.Logger,
) -> Dict[str, str]:
    """Probe GPU availability and compatibility for CatBoost, LightGBM, and XGBoost."""
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "None"
    logger.info("[INFO] CUDA available: %s (Device: %s)", cuda_available, gpu_name)

    sample = dev_df.iloc[:300].copy()
    y_sample = sample[target_col].to_numpy(dtype=np.float64)
    train_cats = {col: sorted(sample[col].unique().tolist()) for col in cat_cols}

    devices = {
        "Ridge": "CPU",
        "ExtraTrees": "CPU",
        "CatBoost": "CPU",
        "LightGBM": "CPU",
        "XGBoost": "CPU",
    }

    # 1. CatBoost GPU probe
    if cuda_available:
        try:
            cb_probe = cb.CatBoostRegressor(
                iterations=20,
                learning_rate=0.05,
                depth=6,
                l2_leaf_reg=5.0,
                task_type="GPU",
                cat_features=cat_cols,
                random_seed=RANDOM_STATE,
                verbose=0,
            )
            cb_probe.fit(sample[feature_cols], y_sample)
            devices["CatBoost"] = "GPU"
            logger.info("[PASS] CatBoost GPU probe succeeded (task_type='GPU').")
        except Exception as exc:
            devices["CatBoost"] = "CPU"
            logger.warning("[WARN] CatBoost GPU probe failed (%s); falling back to CPU.", exc)
    else:
        logger.warning("[WARN] CUDA unavailable; CatBoost set to CPU.")

    # 2. XGBoost CUDA probe
    if cuda_available:
        try:
            X_xgb = build_categorical_frame(sample, feature_cols, cat_cols, train_cats)
            xgb_probe = xgb.XGBRegressor(
                n_estimators=20,
                max_depth=6,
                tree_method="hist",
                device="cuda",
                enable_categorical=True,
                random_state=RANDOM_STATE,
            )
            xgb_probe.fit(X_xgb, y_sample)
            devices["XGBoost"] = "GPU"
            logger.info("[PASS] XGBoost CUDA probe succeeded (device='cuda', tree_method='hist').")
        except Exception as exc:
            devices["XGBoost"] = "CPU"
            logger.warning("[WARN] XGBoost CUDA probe failed (%s); falling back to CPU.", exc)
    else:
        logger.warning("[WARN] CUDA unavailable; XGBoost set to CPU.")

    # 3. LightGBM GPU/CUDA probe
    X_lgb = build_categorical_frame(sample, feature_cols, cat_cols, train_cats)
    lgb_gpu_ok = False
    if cuda_available:
        for dev_flag in ("gpu", "cuda"):
            try:
                lgb_probe = lgb.LGBMRegressor(
                    n_estimators=20,
                    device=dev_flag,
                    verbose=-1,
                    random_state=RANDOM_STATE,
                )
                lgb_probe.fit(X_lgb, y_sample)
                devices["LightGBM"] = f"GPU ({dev_flag})"
                lgb_gpu_ok = True
                logger.info("[PASS] LightGBM GPU probe succeeded (device='%s').", dev_flag)
                break
            except Exception as exc:
                logger.warning(
                    "[WARN] LightGBM device='%s' unavailable (%s).",
                    dev_flag,
                    str(exc).splitlines()[0],
                )
    if not lgb_gpu_ok:
        devices["LightGBM"] = "CPU"
        logger.warning("[WARN] LightGBM falling back to CPU (n_jobs=-1).")

    return devices


def make_ridge_pipeline(
    num_cols: List[str],
    low_cat_cols: List[str],
    high_cat_cols: List[str],
) -> Pipeline:
    """Create fold-isolated Ridge pipeline with OHE + TargetEncoder + StandardScaler."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("low_cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), low_cat_cols),
            (
                "high_cat",
                Pipeline([
                    (
                        "te",
                        TargetEncoder(
                            target_type="continuous",
                            smooth="auto",
                            cv=5,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                    ("scaler", StandardScaler()),
                ]),
                high_cat_cols,
            ),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", Ridge(alpha=10.0, random_state=RANDOM_STATE)),
    ])


def make_extratrees_pipeline(
    num_cols: List[str],
    low_cat_cols: List[str],
    high_cat_cols: List[str],
) -> Pipeline:
    """Create fold-isolated ExtraTrees pipeline with OrdinalEncoder + TargetEncoder."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            (
                "low_cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                low_cat_cols,
            ),
            (
                "high_cat",
                TargetEncoder(
                    target_type="continuous",
                    smooth="auto",
                    cv=5,
                    random_state=RANDOM_STATE,
                ),
                high_cat_cols,
            ),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        (
            "regressor",
            ExtraTreesRegressor(
                n_estimators=300,
                max_features=0.7,
                min_samples_leaf=3,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
        ),
    ])


def solve_slsqp_weights(pred_matrix: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Solve non-negative weights summing to 1 minimizing MSE via SLSQP."""
    n_models = pred_matrix.shape[1]
    init_w = np.full(n_models, 1.0 / n_models, dtype=np.float64)

    def objective(w: np.ndarray) -> float:
        diff = y_true - pred_matrix @ w
        return float(np.mean(diff * diff))

    def gradient(w: np.ndarray) -> np.ndarray:
        diff = pred_matrix @ w - y_true
        return (2.0 / len(y_true)) * (pred_matrix.T @ diff)

    bounds = [(0.0, 1.0)] * n_models
    constraints = {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}
    res = minimize(
        objective,
        init_w,
        jac=gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-10, "maxiter": 500},
    )
    weights = np.clip(res.x, 0.0, 1.0)
    weights /= np.sum(weights)
    return weights


def solve_caruana_weights(
    pred_matrix: np.ndarray,
    y_true: np.ndarray,
    ensemble_size: int = 100,
) -> np.ndarray:
    """Caruana (2004) greedy ensemble selection with replacement."""
    n_samples, n_models = pred_matrix.shape
    counts = np.zeros(n_models, dtype=np.float64)
    running_sum = np.zeros(n_samples, dtype=np.float64)

    for step in range(1, ensemble_size + 1):
        best_idx = 0
        best_mse = float("inf")
        for m in range(n_models):
            cand_pred = (running_sum + pred_matrix[:, m]) / float(step)
            mse = float(np.mean((y_true - cand_pred) ** 2))
            if mse < best_mse:
                best_mse = mse
                best_idx = m
        running_sum += pred_matrix[:, best_idx]
        counts[best_idx] += 1.0

    return counts / float(ensemble_size)


def solve_equal_weights(n_models: int) -> np.ndarray:
    """Return uniform weights summing to 1."""
    return np.full(n_models, 1.0 / n_models, dtype=np.float64)


def run_voting_evaluation(
    oof_preds: Dict[str, np.ndarray],
    nov_preds: Dict[str, np.ndarray],
    y_dev: np.ndarray,
    y_nov: np.ndarray,
    fold_ids: np.ndarray,
    unseen_recipe_mask_nov: np.ndarray,
    logger: logging.Logger,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, float]], Dict[str, np.ndarray], Dict[str, np.ndarray], List[Dict[str, Any]]]:
    """Evaluate Equal, SLSQP, and Caruana voting on Trees_Only and All_5_Models via Nested OOF."""
    pools = {
        "Trees_Only": TREE_MODELS,
        "All_5_Models": BASE_MODELS,
    }
    methods = ["Equal", "SLSQP", "Caruana"]

    # Base model residual correlation matrix on Dev OOF
    residuals = {m: y_dev - oof_preds[m] for m in BASE_MODELS}
    resid_df = pd.DataFrame(residuals)
    corr_matrix = resid_df.corr()

    voting_rows: List[Dict[str, Any]] = []
    final_weights_map: Dict[str, Dict[str, float]] = {}
    nested_oof_preds: Dict[str, np.ndarray] = {}
    voting_nov_preds: Dict[str, np.ndarray] = {}
    weight_records: List[Dict[str, Any]] = []

    for pool_name, model_list in pools.items():
        P_dev = np.column_stack([oof_preds[m] for m in model_list])
        P_nov = np.column_stack([nov_preds[m] for m in model_list])

        for method in methods:
            scheme_name = f"Weighted_Voting_{pool_name}_{method}"
            nested_pred = np.zeros_like(y_dev, dtype=np.float64)
            fold_weights = []

            for fold_idx in range(5):
                inner_mask = fold_ids != fold_idx
                outer_mask = fold_ids == fold_idx

                if method == "Equal":
                    w_fold = solve_equal_weights(len(model_list))
                elif method == "SLSQP":
                    w_fold = solve_slsqp_weights(P_dev[inner_mask], y_dev[inner_mask])
                else:
                    w_fold = solve_caruana_weights(P_dev[inner_mask], y_dev[inner_mask], ensemble_size=100)

                fold_weights.append(w_fold)
                nested_pred[outer_mask] = P_dev[outer_mask] @ w_fold

            fold_weights_arr = np.vstack(fold_weights)
            if method == "Equal":
                w_final = solve_equal_weights(len(model_list))
            elif method == "SLSQP":
                w_final = solve_slsqp_weights(P_dev, y_dev)
            else:
                w_final = solve_caruana_weights(P_dev, y_dev, ensemble_size=100)

            assert np.all(w_final >= -1e-12), f"Negative weights in {scheme_name}"
            assert np.isclose(np.sum(w_final), 1.0, atol=1e-6), f"Weights do not sum to 1 in {scheme_name}"

            nov_pred = P_nov @ w_final
            nested_oof_preds[scheme_name] = nested_pred
            voting_nov_preds[scheme_name] = nov_pred

            # Per-fold Nested OOF R2 stats
            fold_r2s = [
                float(r2_score(y_dev[fold_ids == k], nested_pred[fold_ids == k]))
                for k in range(5)
            ]
            dev_m = compute_metrics(y_dev, nested_pred)
            nov_m = compute_metrics(y_nov, nov_pred)
            cold_m = compute_metrics(
                y_nov[unseen_recipe_mask_nov],
                nov_pred[unseen_recipe_mask_nov],
            )

            full_weight_dict = {m: 0.0 for m in BASE_MODELS}
            fold_mean_dict = {m: 0.0 for m in BASE_MODELS}
            fold_std_dict = {m: 0.0 for m in BASE_MODELS}
            for idx, m in enumerate(model_list):
                full_weight_dict[m] = float(w_final[idx])
                fold_mean_dict[m] = float(np.mean(fold_weights_arr[:, idx]))
                fold_std_dict[m] = float(np.std(fold_weights_arr[:, idx]))

            final_weights_map[scheme_name] = full_weight_dict

            for m in BASE_MODELS:
                weight_records.append({
                    "Scheme": scheme_name,
                    "Pool": pool_name,
                    "Method": method,
                    "Model": m,
                    "Final_Weight": round(full_weight_dict[m], 6),
                    "Nested_OOF_Weight_Mean": round(fold_mean_dict[m], 6),
                    "Nested_OOF_Weight_Std": round(fold_std_dict[m], 6),
                    "Corr_Ridge": round(float(corr_matrix.loc[m, "Ridge"]), 6),
                    "Corr_ExtraTrees": round(float(corr_matrix.loc[m, "ExtraTrees"]), 6),
                    "Corr_CatBoost": round(float(corr_matrix.loc[m, "CatBoost"]), 6),
                    "Corr_LightGBM": round(float(corr_matrix.loc[m, "LightGBM"]), 6),
                    "Corr_XGBoost": round(float(corr_matrix.loc[m, "XGBoost"]), 6),
                })

            voting_rows.append({
                "Model": scheme_name,
                "Category": f"Voting_{pool_name}",
                "Device": "CPU_Ensemble",
                "Train_Time_Sec": 0.0,
                "Dev_OOF_R2": dev_m["R2"],
                "Dev_OOF_RMSE": dev_m["RMSE"],
                "Dev_OOF_MAE": dev_m["MAE"],
                "Dev_Fold_R2_Mean": float(np.mean(fold_r2s)),
                "Dev_Fold_R2_Std": float(np.std(fold_r2s)),
                "Nov_Blind_R2": nov_m["R2"],
                "Nov_Blind_RMSE": nov_m["RMSE"],
                "Nov_Blind_MAE": nov_m["MAE"],
                "ColdStart_Unseen_Recipe_R2": cold_m["R2"],
                "ColdStart_Unseen_Recipe_RMSE": cold_m["RMSE"],
            })

            w_str = ", ".join(f"{m}={full_weight_dict[m]:.4f}" for m in model_list)
            logger.info(
                "[INFO] %-34s | Dev Nested R2=%.5f RMSE=%.4f | Nov R2=%.5f RMSE=%.4f | Weights: %s",
                scheme_name,
                dev_m["R2"],
                dev_m["RMSE"],
                nov_m["R2"],
                nov_m["RMSE"],
                w_str,
            )

    return voting_rows, final_weights_map, nested_oof_preds, voting_nov_preds, weight_records


def create_summary_figure(
    report_df: pd.DataFrame,
    corr_matrix: pd.DataFrame,
    best_trees_scheme: str,
    best_all_scheme: str,
    final_weights_map: Dict[str, Dict[str, float]],
    fig_path: str,
) -> None:
    """Create a clean dual-panel benchmark summary chart without emojis."""
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    selected_models = BASE_MODELS + [best_trees_scheme, best_all_scheme]
    sub_df = report_df.set_index("Model").loc[selected_models].reset_index()

    display_labels = []
    for m in sub_df["Model"]:
        if m == best_trees_scheme:
            display_labels.append(f"Voting (Trees)\n[{m.split('_')[-1]}]")
        elif m == best_all_scheme:
            display_labels.append(f"Voting (All 5)\n[{m.split('_')[-1]}]")
        else:
            display_labels.append(m)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.8), gridspec_kw={"width_ratios": [1.25, 1.0]})

    # Panel A: Dev OOF R2 vs Nov Blind Test R2
    ax1 = axes[0]
    y_pos = np.arange(len(sub_df))
    bar_h = 0.36

    bars_dev = ax1.barh(
        y_pos - bar_h / 2,
        sub_df["Dev_OOF_R2"],
        height=bar_h,
        color="#2B5C8F",
        label="Dev 5-Fold OOF / Nested OOF (Aug-Oct, n=43,451)",
    )
    bars_nov = ax1.barh(
        y_pos + bar_h / 2,
        sub_df["Nov_Blind_R2"],
        height=bar_h,
        color="#D97724",
        label="November historical holdout (n=9,694)",
    )

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(display_labels, fontsize=10, fontweight="bold")
    ax1.invert_yaxis()
    ax1.set_xlabel("Explained Variance (R2)", fontsize=11, fontweight="bold")
    ax1.set_title(
        "Panel A: Model R² by Evaluation Split",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )

    min_r2 = max(0.0, float(min(sub_df["Dev_OOF_R2"].min(), sub_df["Nov_Blind_R2"].min())) - 0.08)
    max_r2 = min(1.02, float(max(sub_df["Dev_OOF_R2"].max(), sub_df["Nov_Blind_R2"].max())) + 0.10)
    ax1.set_xlim(min_r2, max_r2)
    ax1.grid(axis="x", linestyle="--", alpha=0.4)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    for idx, (b_dev, b_nov) in enumerate(zip(bars_dev, bars_nov)):
        dev_r2 = sub_df.loc[idx, "Dev_OOF_R2"]
        dev_rmse = sub_df.loc[idx, "Dev_OOF_RMSE"]
        nov_r2 = sub_df.loc[idx, "Nov_Blind_R2"]
        nov_rmse = sub_df.loc[idx, "Nov_Blind_RMSE"]
        ax1.text(
            b_dev.get_width() + 0.004,
            b_dev.get_y() + b_dev.get_height() / 2,
            f"{dev_r2:.4f} (RMSE {dev_rmse:.2f})",
            va="center",
            fontsize=8.5,
            color="#1F3E5E",
        )
        ax1.text(
            b_nov.get_width() + 0.004,
            b_nov.get_y() + b_nov.get_height() / 2,
            f"{nov_r2:.4f} (RMSE {nov_rmse:.2f})",
            va="center",
            fontsize=8.5,
            color="#9C4D0E",
        )

    ax1.legend(loc="upper right", frameon=True, fontsize=9)

    # Panel B: Residual Correlation Heatmap + Weight Annotation Box
    ax2 = axes[1]
    corr_vals = corr_matrix.loc[BASE_MODELS, BASE_MODELS].to_numpy()
    im = ax2.imshow(corr_vals, cmap="YlGnBu", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Dev OOF Residual Pearson r", fontsize=10)

    ax2.set_xticks(np.arange(len(BASE_MODELS)))
    ax2.set_yticks(np.arange(len(BASE_MODELS)))
    ax2.set_xticklabels(BASE_MODELS, rotation=30, ha="right", fontsize=10, fontweight="bold")
    ax2.set_yticklabels(BASE_MODELS, fontsize=10, fontweight="bold")
    ax2.set_title(
        "Panel B: Base-Model OOF Residual Correlation Matrix",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )

    for i in range(len(BASE_MODELS)):
        for j in range(len(BASE_MODELS)):
            val = corr_vals[i, j]
            txt_color = "white" if val > 0.60 else "black"
            ax2.text(j, i, f"{val:.3f}", ha="center", va="center", color=txt_color, fontsize=9.5)

    w_trees = final_weights_map[best_trees_scheme]
    w_all = final_weights_map[best_all_scheme]
    trees_txt = ", ".join(f"{m}: {w_trees[m]:.3f}" for m in TREE_MODELS)
    all_txt = ", ".join(f"{m}: {w_all[m]:.3f}" for m in BASE_MODELS if w_all[m] > 0.0005)
    summary_text = (
        f"Best Trees-Only ({best_trees_scheme.split('_')[-1]}): {trees_txt}\n"
        f"Best All-Models ({best_all_scheme.split('_')[-1]}): {all_txt}"
    )
    ax2.text(
        0.5,
        -0.22,
        summary_text,
        transform=ax2.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#F4F6F9", edgecolor="#B0BEC5"),
    )

    fig.suptitle(
        "Historical Model Benchmark and Weighted Voting",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.94])
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)


def predict_from_bundle_model(
    bundle: Dict[str, Any],
    model_name: str,
    df_input: pd.DataFrame,
) -> np.ndarray:
    """Predict target values using a persisted 5-fold base model or voting scheme."""
    feature_cols = bundle["feature_schema"]["feature_cols"]
    cat_cols = bundle["feature_schema"]["categorical_features"]
    X_raw = df_input[feature_cols].copy()
    for col in cat_cols:
        X_raw[col] = X_raw[col].astype(str)

    if model_name in BASE_MODELS:
        fold_entries = bundle["models"][model_name]
        preds = np.zeros(len(df_input), dtype=np.float64)
        for entry in fold_entries:
            if model_name in ("LightGBM", "XGBoost"):
                X_cat = build_categorical_frame(
                    X_raw,
                    feature_cols,
                    cat_cols,
                    entry["train_categories"],
                )
                preds += entry["estimator"].predict(X_cat)
            else:
                preds += entry["estimator"].predict(X_raw)
        return preds / float(len(fold_entries))

    weights = bundle["voting_weights"][model_name]
    preds = np.zeros(len(df_input), dtype=np.float64)
    for base_m, w in weights.items():
        if w > 0.0:
            preds += w * predict_from_bundle_model(bundle, base_m, df_input)
    return preds


def main() -> None:
    overall_start = time.time()
    logger = setup_logging(LOG_PATH)
    logger.info("[INFO] Starting Step 3 Part 1: Model Benchmark & Weighted Voting Pipeline.")

    warnings.filterwarnings("ignore", category=UserWarning)

    os.makedirs("reports", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # 1. Load data & schema
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)

    num_cols: List[str] = schema["numeric_features"]
    cat_cols: List[str] = schema["categorical_features"]
    target_col: str = schema["target_col"]
    feature_cols = num_cols + cat_cols

    df = pd.read_csv(DATA_PATH, low_memory=False)
    for col in cat_cols:
        df[col] = df[col].astype(str)

    assert len(df) == 53145, f"Expected 53145 rows, got {len(df)}"
    logger.info("[PASS] Dataset loaded: %d rows x %d columns.", df.shape[0], df.shape[1])

    # 2. Construct work_order_id and split Dev (Aug-Oct) vs Nov Blind Test
    df["work_order_id"] = pd.util.hash_pandas_object(df[WORK_ORDER_COLS], index=False).astype(np.uint64)

    dev_mask = df["ActualStartMonth"].isin([8, 9, 10])
    nov_mask = df["ActualStartMonth"] == 11
    dev_df = df.loc[dev_mask].reset_index(drop=True)
    nov_df = df.loc[nov_mask].reset_index(drop=True)

    assert len(dev_df) + len(nov_df) == 53145, "Dev + Nov row count mismatch."
    assert len(dev_df) == 43451 and len(nov_df) == 9694, "Unexpected Dev/Nov split sizes."
    dev_wo_set = set(dev_df["work_order_id"].tolist())
    nov_wo_set = set(nov_df["work_order_id"].tolist())
    assert dev_wo_set.isdisjoint(nov_wo_set), "Work order overlap detected between Dev and Nov!"

    logger.info(
        "[PASS] Split verified: Dev=%d rows (%d work orders), Nov=%d rows (%d work orders), Overlap=0.",
        len(dev_df),
        len(dev_wo_set),
        len(nov_df),
        len(nov_wo_set),
    )

    # Identify ultra-high cardinality (>250) strictly on Dev set
    high_cat_cols = [col for col in cat_cols if dev_df[col].nunique() > 250]
    low_cat_cols = [col for col in cat_cols if dev_df[col].nunique() <= 250]
    logger.info(
        "[INFO] Categorical partition on Dev: %d high-cardinality (>250), %d low-cardinality (<=250).",
        len(high_cat_cols),
        len(low_cat_cols),
    )

    # Cold-start unseen RecipeID mask in Nov
    dev_recipes = set(dev_df["RecipeID"].unique().tolist())
    unseen_recipe_mask_nov = (~nov_df["RecipeID"].isin(dev_recipes)).to_numpy()
    logger.info(
        "[INFO] Nov cold-start unseen RecipeID rows: %d / %d (%.2f%%).",
        int(unseen_recipe_mask_nov.sum()),
        len(nov_df),
        100.0 * unseen_recipe_mask_nov.mean(),
    )

    # 3. GPU probes
    devices = probe_gpu_capabilities(dev_df, feature_cols, cat_cols, target_col, logger)

    # 4. 5-Fold GroupKFold setup
    gkf = GroupKFold(n_splits=5)
    dev_groups = dev_df["work_order_id"].to_numpy()
    fold_splits = list(gkf.split(dev_df[feature_cols], dev_df[target_col], groups=dev_groups))
    fold_ids = np.zeros(len(dev_df), dtype=np.int32)

    for fold_idx, (tr_idx, va_idx) in enumerate(fold_splits):
        tr_wo = set(dev_groups[tr_idx].tolist())
        va_wo = set(dev_groups[va_idx].tolist())
        assert tr_wo.isdisjoint(va_wo), f"Work order leakage in Fold {fold_idx}!"
        fold_ids[va_idx] = fold_idx

    logger.info("[PASS] 5-fold GroupKFold work-order isolation verified across all 5 folds.")

    y_dev = dev_df[target_col].to_numpy(dtype=np.float64)
    y_nov = nov_df[target_col].to_numpy(dtype=np.float64)
    X_dev_raw = dev_df[feature_cols].copy()
    X_nov_raw = nov_df[feature_cols].copy()

    # Containers for models, OOF predictions, Nov predictions, and metrics
    models_bundle: Dict[str, List[Dict[str, Any]]] = {m: [] for m in BASE_MODELS}
    oof_preds: Dict[str, np.ndarray] = {m: np.zeros(len(dev_df), dtype=np.float64) for m in BASE_MODELS}
    nov_preds: Dict[str, np.ndarray] = {m: np.zeros(len(nov_df), dtype=np.float64) for m in BASE_MODELS}
    fold_r2_records: Dict[str, List[float]] = {m: [] for m in BASE_MODELS}
    train_times: Dict[str, float] = {m: 0.0 for m in BASE_MODELS}

    # 5. Train 5 Base Models across 5 Folds
    for fold_idx, (tr_idx, va_idx) in enumerate(fold_splits):
        logger.info("[INFO] --- Fold %d / 5 (Train=%d, Val=%d) ---", fold_idx + 1, len(tr_idx), len(va_idx))
        X_tr_raw = X_dev_raw.iloc[tr_idx].reset_index(drop=True)
        y_tr = y_dev[tr_idx]
        X_va_raw = X_dev_raw.iloc[va_idx].reset_index(drop=True)
        y_va = y_dev[va_idx]

        # Build fold-isolated category dictionary strictly from X_tr_raw
        train_cats = {col: sorted(X_tr_raw[col].unique().tolist()) for col in cat_cols}
        X_tr_cat = build_categorical_frame(X_tr_raw, feature_cols, cat_cols, train_cats)
        X_va_cat = build_categorical_frame(X_va_raw, feature_cols, cat_cols, train_cats)
        X_nov_cat = build_categorical_frame(X_nov_raw, feature_cols, cat_cols, train_cats)

        # Model 1: Ridge
        t0 = time.time()
        ridge_pipe = make_ridge_pipeline(num_cols, low_cat_cols, high_cat_cols)
        ridge_pipe.fit(X_tr_raw, y_tr)
        p_va = ridge_pipe.predict(X_va_raw)
        p_nov = ridge_pipe.predict(X_nov_raw)
        train_times["Ridge"] += time.time() - t0
        oof_preds["Ridge"][va_idx] = p_va
        nov_preds["Ridge"] += p_nov / 5.0
        fold_r2_records["Ridge"].append(float(r2_score(y_va, p_va)))
        models_bundle["Ridge"].append({"estimator": ridge_pipe})

        # Model 2: ExtraTrees
        t0 = time.time()
        et_pipe = make_extratrees_pipeline(num_cols, low_cat_cols, high_cat_cols)
        et_pipe.fit(X_tr_raw, y_tr)
        p_va = et_pipe.predict(X_va_raw)
        p_nov = et_pipe.predict(X_nov_raw)
        train_times["ExtraTrees"] += time.time() - t0
        oof_preds["ExtraTrees"][va_idx] = p_va
        nov_preds["ExtraTrees"] += p_nov / 5.0
        fold_r2_records["ExtraTrees"].append(float(r2_score(y_va, p_va)))
        models_bundle["ExtraTrees"].append({"estimator": et_pipe})

        # Model 3: CatBoost
        t0 = time.time()
        cb_task = "GPU" if devices["CatBoost"] == "GPU" else "CPU"
        cb_params = dict(
            iterations=2500,
            learning_rate=0.05,
            depth=7,
            l2_leaf_reg=5.0,
            loss_function="RMSE",
            eval_metric="RMSE",
            early_stopping_rounds=100,
            cat_features=cat_cols,
            task_type=cb_task,
            random_seed=RANDOM_STATE + fold_idx,
            verbose=0,
        )
        cb_model = cb.CatBoostRegressor(**cb_params)
        try:
            cb_model.fit(X_tr_raw, y_tr, eval_set=(X_va_raw, y_va))
        except Exception as exc:
            if cb_task == "GPU":
                logger.warning("[WARN] CatBoost GPU fit failed on Fold %d (%s); retrying CPU.", fold_idx + 1, exc)
                cb_params["task_type"] = "CPU"
                devices["CatBoost"] = "CPU (Fallback)"
                cb_model = cb.CatBoostRegressor(**cb_params)
                cb_model.fit(X_tr_raw, y_tr, eval_set=(X_va_raw, y_va))
            else:
                raise
        p_va = cb_model.predict(X_va_raw)
        p_nov = cb_model.predict(X_nov_raw)
        train_times["CatBoost"] += time.time() - t0
        oof_preds["CatBoost"][va_idx] = p_va
        nov_preds["CatBoost"] += p_nov / 5.0
        fold_r2_records["CatBoost"].append(float(r2_score(y_va, p_va)))
        models_bundle["CatBoost"].append({"estimator": cb_model})

        # Model 4: LightGBM
        t0 = time.time()
        lgb_params = dict(
            n_estimators=2500,
            learning_rate=0.04,
            num_leaves=63,
            min_child_samples=30,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            cat_smooth=20.0,
            min_data_per_group=50,
            n_jobs=-1,
            random_state=RANDOM_STATE + fold_idx,
            verbose=-1,
        )
        if devices["LightGBM"].startswith("GPU"):
            lgb_params["device"] = "gpu" if "gpu" in devices["LightGBM"] else "cuda"
        lgb_model = lgb.LGBMRegressor(**lgb_params)
        lgb_model.fit(
            X_tr_cat,
            y_tr,
            eval_X=X_va_cat,
            eval_y=y_va,
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        p_va = lgb_model.predict(X_va_cat)
        p_nov = lgb_model.predict(X_nov_cat)
        train_times["LightGBM"] += time.time() - t0
        oof_preds["LightGBM"][va_idx] = p_va
        nov_preds["LightGBM"] += p_nov / 5.0
        fold_r2_records["LightGBM"].append(float(r2_score(y_va, p_va)))
        models_bundle["LightGBM"].append({"estimator": lgb_model, "train_categories": train_cats})

        # Model 5: XGBoost
        t0 = time.time()
        xgb_dev = "cuda" if devices["XGBoost"] == "GPU" else "cpu"
        xgb_model = xgb.XGBRegressor(
            n_estimators=2500,
            learning_rate=0.04,
            max_depth=7,
            min_child_weight=10,
            subsample=0.8,
            colsample_bytree=0.8,
            max_cat_to_onehot=16,
            reg_lambda=5.0,
            tree_method="hist",
            device=xgb_dev,
            enable_categorical=True,
            early_stopping_rounds=100,
            eval_metric="rmse",
            random_state=RANDOM_STATE + fold_idx,
            n_jobs=-1,
        )
        xgb_model.fit(X_tr_cat, y_tr, eval_set=[(X_va_cat, y_va)], verbose=False)
        xgb_model.set_params(device="cpu")
        p_va = xgb_model.predict(X_va_cat)
        p_nov = xgb_model.predict(X_nov_cat)
        train_times["XGBoost"] += time.time() - t0
        oof_preds["XGBoost"][va_idx] = p_va
        nov_preds["XGBoost"] += p_nov / 5.0
        fold_r2_records["XGBoost"].append(float(r2_score(y_va, p_va)))
        models_bundle["XGBoost"].append({"estimator": xgb_model, "train_categories": train_cats})

        logger.info(
            "[INFO] Fold %d Val R2 | Ridge=%.4f | ExtraTrees=%.4f | CatBoost=%.4f | LightGBM=%.4f | XGBoost=%.4f",
            fold_idx + 1,
            fold_r2_records["Ridge"][-1],
            fold_r2_records["ExtraTrees"][-1],
            fold_r2_records["CatBoost"][-1],
            fold_r2_records["LightGBM"][-1],
            fold_r2_records["XGBoost"][-1],
        )

    # 6. Base Model Benchmark Summary
    benchmark_rows: List[Dict[str, Any]] = []
    for m in BASE_MODELS:
        dev_m = compute_metrics(y_dev, oof_preds[m])
        nov_m = compute_metrics(y_nov, nov_preds[m])
        cold_m = compute_metrics(y_nov[unseen_recipe_mask_nov], nov_preds[m][unseen_recipe_mask_nov])
        benchmark_rows.append({
            "Model": m,
            "Category": "Base_Model",
            "Device": devices[m],
            "Train_Time_Sec": round(train_times[m], 2),
            "Dev_OOF_R2": dev_m["R2"],
            "Dev_OOF_RMSE": dev_m["RMSE"],
            "Dev_OOF_MAE": dev_m["MAE"],
            "Dev_Fold_R2_Mean": float(np.mean(fold_r2_records[m])),
            "Dev_Fold_R2_Std": float(np.std(fold_r2_records[m])),
            "Nov_Blind_R2": nov_m["R2"],
            "Nov_Blind_RMSE": nov_m["RMSE"],
            "Nov_Blind_MAE": nov_m["MAE"],
            "ColdStart_Unseen_Recipe_R2": cold_m["R2"],
            "ColdStart_Unseen_Recipe_RMSE": cold_m["RMSE"],
        })
        logger.info(
            "[PASS] Base %-10s | Dev OOF R2=%.5f RMSE=%.4f MAE=%.4f | Nov Blind R2=%.5f RMSE=%.4f MAE=%.4f | Time=%.1fs",
            m,
            dev_m["R2"],
            dev_m["RMSE"],
            dev_m["MAE"],
            nov_m["R2"],
            nov_m["RMSE"],
            nov_m["MAE"],
            train_times[m],
        )

    # 7. Weighted Voting Evaluation (Nested OOF + Nov Blind)
    voting_rows, final_weights_map, nested_oof_preds, voting_nov_preds, weight_records = run_voting_evaluation(
        oof_preds=oof_preds,
        nov_preds=nov_preds,
        y_dev=y_dev,
        y_nov=y_nov,
        fold_ids=fold_ids,
        unseen_recipe_mask_nov=unseen_recipe_mask_nov,
        logger=logger,
    )

    benchmark_rows.extend(voting_rows)
    report_df = pd.DataFrame(benchmark_rows)
    weights_df = pd.DataFrame(weight_records)

    # Identify best voting schemes by Dev Nested OOF R2
    trees_rows = report_df[report_df["Category"] == "Voting_Trees_Only"]
    all_rows = report_df[report_df["Category"] == "Voting_All_5_Models"]
    best_trees_scheme = str(trees_rows.sort_values("Dev_OOF_R2", ascending=False).iloc[0]["Model"])
    best_all_scheme = str(all_rows.sort_values("Dev_OOF_R2", ascending=False).iloc[0]["Model"])
    best_overall_voting = str(
        report_df[report_df["Category"].str.startswith("Voting_")]
        .sort_values("Dev_OOF_R2", ascending=False)
        .iloc[0]["Model"]
    )
    report_df["Is_Best_Trees_Voting"] = report_df["Model"] == best_trees_scheme
    report_df["Is_Best_All_Voting"] = report_df["Model"] == best_all_scheme
    report_df["Is_Overall_Best_Voting"] = report_df["Model"] == best_overall_voting

    assert not report_df.isna().any().any(), "NaN values found in benchmark report!"
    assert not weights_df.isna().any().any(), "NaN values found in voting weights report!"

    report_df.to_csv(REPORT_BENCHMARK_PATH, index=False, encoding="utf-8-sig")
    weights_df.to_csv(REPORT_WEIGHTS_PATH, index=False, encoding="utf-8-sig")
    logger.info("[PASS] Reports saved: %s, %s", REPORT_BENCHMARK_PATH, REPORT_WEIGHTS_PATH)

    # 8. Visualization
    corr_matrix = pd.DataFrame({m: y_dev - oof_preds[m] for m in BASE_MODELS}).corr()
    create_summary_figure(
        report_df=report_df,
        corr_matrix=corr_matrix,
        best_trees_scheme=best_trees_scheme,
        best_all_scheme=best_all_scheme,
        final_weights_map=final_weights_map,
        fig_path=FIG_PATH,
    )
    logger.info("[PASS] Summary figure saved: %s", FIG_PATH)

    # 9. Persist Bundle
    bundle = {
        "models": models_bundle,
        "feature_schema": {
            "feature_cols": feature_cols,
            "numeric_features": num_cols,
            "categorical_features": cat_cols,
            "low_cardinality_features": low_cat_cols,
            "high_cardinality_features": high_cat_cols,
            "target_col": target_col,
            "work_order_cols": WORK_ORDER_COLS,
        },
        "fold_assignments": fold_ids,
        "oof_predictions": {**oof_preds, **nested_oof_preds},
        "nov_predictions": {**nov_preds, **voting_nov_preds},
        "voting_weights": final_weights_map,
        "best_schemes": {
            "best_trees_voting": best_trees_scheme,
            "best_all_voting": best_all_scheme,
            "best_overall_voting": best_overall_voting,
        },
        "devices": devices,
        "versions": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "lightgbm": lgb.__version__,
            "catboost": cb.__version__,
            "xgboost": xgb.__version__,
        },
    }

    with open(BUNDLE_PATH, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("[PASS] Model bundle persisted: %s", BUNDLE_PATH)

    # 10. Reload & Inference Self-Check
    with open(BUNDLE_PATH, "rb") as f:
        reloaded = pickle.load(f)

    verify_slice = nov_df.iloc[:200].copy()
    for m_check in BASE_MODELS + [best_trees_scheme, best_overall_voting]:
        pred_reloaded = predict_from_bundle_model(reloaded, m_check, verify_slice)
        pred_stored = reloaded["nov_predictions"][m_check][:200]
        max_diff = float(np.max(np.abs(pred_reloaded - pred_stored)))
        assert np.allclose(pred_reloaded, pred_stored, atol=1e-5), (
            f"Bundle reload prediction mismatch for {m_check}: max_diff={max_diff}"
        )
    logger.info("[PASS] Bundle reload and inference verification passed (max_diff <= 1e-5).")

    total_sec = time.time() - overall_start
    logger.info("[PASS] Step 3 Part 1 completed successfully in %.2f seconds.", total_sec)


if __name__ == "__main__":
    main()
