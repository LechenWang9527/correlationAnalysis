# -*- coding: utf-8 -*-
"""
Step 3 Part 2: Multi-Dimensional Feature Attribution and Causal Net-Effect Quantification.

Implements a 6-stage interpretability and causal inference pipeline on top of the
Step 3 Part 1 CatBoost ensemble (models/step3_models_bundle.pkl):
1. Stage 1: CatBoost native TreeSHAP + Physical Cluster Group Shapley on Nov blind test.
2. Stage 2: Cross-month Permutation Importance on Nov blind test (9,694 rows).
3. Stage 3: Top-10 Lightweight Grouped SAGE (MSE loss reduction via Monte Carlo).
4. Stage 4: Targeted LOCO Retraining (Pure Physical, Pure Categorical, Top-8 Single).
5. Stage 5: 1D Accumulated Local Effects (ALE) for Top-6 continuous numeric parameters.
6. Stage 6: 5-Fold Work-Order-Isolated Partially Linear DML + Placebo Refutation.
"""

import json
import logging
import os
import pickle
import sys
import time
import warnings
from typing import Any, Dict, List, Tuple

import lightgbm as lgb
import catboost as cb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

RANDOM_STATE = 42
DATA_PATH = os.path.join("data", "processed", "clean_dataset.csv")
SCHEMA_PATH = os.path.join("data", "processed", "feature_schema.json")
BUNDLE_PATH = os.path.join("models", "step3_models_bundle.pkl")
LOG_PATH = os.path.join("logs", "step3_part2.log")

OUT_SHAP_CSV = os.path.join("reports", "step3_part2_shap_ranking.csv")
OUT_SHAP_PNG = os.path.join("reports", "step3_part2_shap_summary.png")
OUT_PERM_CSV = os.path.join("reports", "step3_part2_permutation_nov_blind.csv")
OUT_SAGE_CSV = os.path.join("reports", "step3_part2_lightweight_sage_top10.csv")
OUT_LOCO_CSV = os.path.join("reports", "step3_part2_loco_ablation.csv")
OUT_ALE_CSV = os.path.join("reports", "step3_part2_ale_summary.csv")
OUT_ALE_PNG = os.path.join("reports", "step3_part2_ale_curves.png")
OUT_DML_CSV = os.path.join("reports", "step3_part2_dml_causal_effects.csv")
OUT_DASH_PNG = os.path.join("reports", "step3_part2_final_synthesis_dashboard.png")

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

NON_CONTROLLABLE_FOR_DML = {
    "ActualStartMonth",
    "ActualStartDayOfWeek",
    "ActualStartHour",
    "PlannedStartMonth",
    "PlannedStartDayOfWeek",
    "PlannedStartHour",
    "ShiftNumber",
    "PlannedQuantity",
    "QuantityDone",
    "IEStandardCycleTimePer100Tire",
    "CycleTimeTolerancePer100Tire",
    "article_cluster_link_status",
    "oeindicator",
    "Has_2nd_Carcass_Ply",
    "Has_Bead_Reinforcement",
    "PTC_Min_Was_Missing",
    "1_flange_cushion_-_compound",
}

NON_CONTINUOUS_FOR_ALE = {
    "ActualStartMonth",
    "PlannedStartMonth",
    "ActualStartDayOfWeek",
    "PlannedStartDayOfWeek",
    "ShiftNumber",
    "oeindicator",
    "Has_2nd_Carcass_Ply",
    "Has_Bead_Reinforcement",
    "PTC_Min_Was_Missing",
    "1_flange_cushion_-_compound",
    "article_cluster_link_status",
    "ply_lock",
}


def setup_logging(log_path: str) -> logging.Logger:
    """Configure logger with both file and stdout handlers."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("step3_part2")
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


def configure_matplotlib() -> None:
    """Configure clean matplotlib defaults with Chinese font support."""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"


def assign_feature_cluster(feat: str) -> str:
    """Map each of the 75 features into a mutually exclusive physical/business cluster."""
    if feat in {
        "PlannedQuantity",
        "QuantityDone",
        "IEStandardCycleTimePer100Tire",
        "CycleTimeTolerancePer100Tire",
        "ShiftNumber",
        "ActualStartHour",
        "ActualStartDayOfWeek",
        "ActualStartMonth",
        "PlannedStartHour",
        "PlannedStartDayOfWeek",
        "PlannedStartMonth",
    }:
        return "Order_and_Schedule"

    if feat in {
        "OperatorUserID",
        "OperatorOverlapSeconds",
        "OperatorCoverageRatio",
        "WorkCenterNameCoupled",
        "Comment",
        "DestinationInfo",
    }:
        return "Operator_and_WorkCenter"

    if feat in {
        "RecipeID",
        "marketing_line",
        "tire_size",
        "pog_segment",
        "pog_segment_description",
        "article_development_cluster_name",
        "article_cluster_link_status",
        "oeindicator",
        "speed_symbol",
        "tire_section_width",
        "tire_aspect_ratio",
    }:
        return "Recipe_and_Product_Line"

    if feat in {
        "rim_diameter",
        "1_stage_drum_width",
        "2_stage_drum_diameter",
        "bladder_code",
        "ply_lock",
    }:
        return "Drum_and_Bladder_Geometry"

    if feat.startswith(("1_tread", "1_belt", "2_belt")):
        return "Tread_and_Belt"

    if feat.startswith("1_wound_cap_ply") or feat == "PTC_Min_Was_Missing":
        return "Wound_Cap_Ply"

    if "carcass_ply" in feat.lower() or feat.startswith(("1_sidewall", "1_flange_cushion")):
        return "Carcass_and_Sidewall"

    if feat.startswith(("1_liner", "1_apex", "bead_")) or "bead_reinforcement" in feat.lower():
        return "Liner_Apex_Bead"

    raise ValueError(f"Unmapped feature in cluster assignment: {feat}")


def predict_catboost_ensemble(
    cb_models: List[cb.CatBoostRegressor],
    df_features: pd.DataFrame,
) -> np.ndarray:
    """Average predictions across the 5-fold CatBoost models."""
    preds = np.zeros(len(df_features), dtype=np.float64)
    for model in cb_models:
        preds += model.predict(df_features)
    return preds / float(len(cb_models))


def run_stage1_treeshap(
    cb_models: List[cb.CatBoostRegressor],
    nov_df: pd.DataFrame,
    feature_cols: List[str],
    cat_cols: List[str],
    target_col: str,
    stored_nov_pred: np.ndarray,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, float]:
    """Stage 1: Compute 5-fold averaged CatBoost TreeSHAP and Group Shapley on Nov blind test."""
    t0 = time.time()
    logger.info("[INFO] === Stage 1: CatBoost TreeSHAP + Group Shapley (Nov n=%d) ===", len(nov_df))

    X_nov = nov_df[feature_cols].copy()
    y_nov = nov_df[target_col].to_numpy(dtype=np.float64)
    pool_nov = cb.Pool(X_nov, cat_features=cat_cols)

    shap_sum = np.zeros((len(nov_df), len(feature_cols)), dtype=np.float64)
    base_sum = 0.0
    for fold_idx, model in enumerate(cb_models):
        sv = model.get_feature_importance(pool_nov, type="ShapValues")
        shap_sum += sv[:, :-1]
        base_sum += float(np.mean(sv[:, -1]))
        logger.info("[INFO] Stage 1 fold %d/5 TreeSHAP computed.", fold_idx + 1)

    shap_values = shap_sum / float(len(cb_models))
    base_value = base_sum / float(len(cb_models))

    # Verify SHAP additivity against stored CatBoost Nov predictions
    reconstructed = shap_values.sum(axis=1) + base_value
    max_diff = float(np.max(np.abs(reconstructed - stored_nov_pred)))
    assert max_diff < 1e-3, f"SHAP additivity check failed: max_diff={max_diff}"
    logger.info("[PASS] TreeSHAP additivity verified (max_diff=%.6f sec).", max_diff)

    cat_set = set(cat_cols)
    rows: List[Dict[str, Any]] = []
    for idx, feat in enumerate(feature_cols):
        sv_col = shap_values[:, idx]
        mean_abs = float(np.mean(np.abs(sv_col)))
        std_shap = float(np.std(sv_col))
        cluster = assign_feature_cluster(feat)

        if feat in cat_set:
            feat_type = "Categorical"
            spearman_r = np.nan
            direction = "Categorical"
        else:
            feat_type = "Numeric"
            vals = X_nov[feat].to_numpy(dtype=np.float64)
            if np.nanstd(vals) < 1e-12 or np.std(sv_col) < 1e-12:
                spearman_r = 0.0
                direction = "Nonlinear / Constant"
            else:
                r_val, _ = stats.spearmanr(vals, sv_col, nan_policy="omit")
                spearman_r = float(r_val) if np.isfinite(r_val) else 0.0
                if abs(spearman_r) < 0.10:
                    direction = "Nonlinear / Bidirectional"
                elif spearman_r >= 0.10:
                    direction = "Positive (Increases Gap)"
                else:
                    direction = "Negative (Reduces Gap)"

        rows.append({
            "Feature": feat,
            "Cluster": cluster,
            "Feature_Type": feat_type,
            "Mean_Abs_SHAP_Sec": mean_abs,
            "SHAP_Std_Sec": std_shap,
            "Spearman_r": spearman_r,
            "Direction_Label": direction,
        })

    shap_df = pd.DataFrame(rows).sort_values("Mean_Abs_SHAP_Sec", ascending=False).reset_index(drop=True)
    shap_df["SHAP_Rank"] = np.arange(1, len(shap_df) + 1)

    # Group Shapley aggregation
    feat_to_idx = {f: i for i, f in enumerate(feature_cols)}
    cluster_names = sorted(shap_df["Cluster"].unique().tolist())
    group_rows: List[Dict[str, Any]] = []
    for cl in cluster_names:
        members = shap_df.loc[shap_df["Cluster"] == cl, "Feature"].tolist()
        member_indices = [feat_to_idx[m] for m in members]
        group_sv = shap_values[:, member_indices].sum(axis=1)
        group_mean_abs = float(np.mean(np.abs(group_sv)))
        sum_member_abs = float(shap_df.loc[shap_df["Cluster"] == cl, "Mean_Abs_SHAP_Sec"].sum())
        group_rows.append({
            "Cluster": cl,
            "Member_Count": len(members),
            "Group_Mean_Abs_SHAP_Sec": group_mean_abs,
            "Sum_Member_Mean_Abs_SHAP_Sec": sum_member_abs,
            "Collinear_Cancellation_Ratio": float(group_mean_abs / max(sum_member_abs, 1e-12)),
        })

    group_df = pd.DataFrame(group_rows).sort_values("Group_Mean_Abs_SHAP_Sec", ascending=False).reset_index(drop=True)
    total_group_shap = float(group_df["Group_Mean_Abs_SHAP_Sec"].sum())
    group_df["Group_Share_Pct"] = 100.0 * group_df["Group_Mean_Abs_SHAP_Sec"] / total_group_shap

    shap_df = shap_df.merge(
        group_df[["Cluster", "Group_Mean_Abs_SHAP_Sec", "Sum_Member_Mean_Abs_SHAP_Sec", "Group_Share_Pct"]],
        on="Cluster",
        how="left",
    )
    shap_df.to_csv(OUT_SHAP_CSV, index=False, encoding="utf-8-sig")
    logger.info("[PASS] Stage 1 CSV saved: %s (Elapsed=%.2fs)", OUT_SHAP_CSV, time.time() - t0)

    for _, r in group_df.iterrows():
        logger.info(
            "[INFO] Group Shapley | %-26s | Members=%2d | Group_Mean(|SHAP|)=%7.2fs (%.1f%%) | Sum_Members=%7.2fs",
            r["Cluster"],
            int(r["Member_Count"]),
            r["Group_Mean_Abs_SHAP_Sec"],
            r["Group_Share_Pct"],
            r["Sum_Member_Mean_Abs_SHAP_Sec"],
        )

    # Visualization: 2x2 Summary Figure
    _plot_stage1_summary(shap_df, group_df, shap_values, base_value, X_nov, y_nov, reconstructed, feature_cols)
    logger.info("[PASS] Stage 1 summary figure saved: %s", OUT_SHAP_PNG)

    return shap_df, group_df, shap_values, base_value


def _plot_stage1_summary(
    shap_df: pd.DataFrame,
    group_df: pd.DataFrame,
    shap_values: np.ndarray,
    base_value: float,
    X_nov: pd.DataFrame,
    y_nov: np.ndarray,
    y_pred: np.ndarray,
    feature_cols: List[str],
) -> None:
    """Create 2x2 SHAP summary plot: Top 20 bar, directional strip, Group Shapley, Waterfall."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    feat_to_idx = {f: i for i, f in enumerate(feature_cols)}

    # Panel A: Top 20 Mean(|SHAP|) Bar Chart
    ax1 = axes[0, 0]
    top20 = shap_df.head(20).copy()
    colors_a = ["#2B5C8F" if t == "Numeric" else "#D97724" for t in top20["Feature_Type"]]
    y_pos = np.arange(len(top20))
    ax1.barh(y_pos, top20["Mean_Abs_SHAP_Sec"], color=colors_a, alpha=0.92)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(top20["Feature"], fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel("Mean(|SHAP|) [Seconds]", fontsize=10, fontweight="bold")
    ax1.set_title("Panel A: Top 20 Features by Global TreeSHAP (Nov Blind)", fontsize=11.5, fontweight="bold")
    ax1.grid(axis="x", linestyle="--", alpha=0.35)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    for idx, val in enumerate(top20["Mean_Abs_SHAP_Sec"]):
        ax1.text(val + 1.5, idx, f"{val:.1f}s", va="center", fontsize=8.2)
    from matplotlib.patches import Patch
    ax1.legend(
        handles=[Patch(color="#2B5C8F", label="Numeric"), Patch(color="#D97724", label="Categorical")],
        loc="lower right",
        fontsize=9,
    )

    # Panel B: Directional Beeswarm-style Strip Plot for Top 12 Numeric Features
    ax2 = axes[0, 1]
    top12_num = shap_df[shap_df["Feature_Type"] == "Numeric"].head(12)["Feature"].tolist()
    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = rng.choice(len(X_nov), size=min(1500, len(X_nov)), replace=False)
    for row_i, feat in enumerate(top12_num):
        col_idx = feat_to_idx[feat]
        sv_s = shap_values[sample_idx, col_idx]
        x_s = X_nov[feat].iloc[sample_idx].to_numpy(dtype=np.float64)
        q_low, q_high = np.nanpercentile(x_s, [5, 95])
        denom = max(q_high - q_low, 1e-9)
        norm_val = np.clip((x_s - q_low) / denom, 0.0, 1.0)
        jitter = rng.uniform(-0.26, 0.26, size=len(sv_s))
        sc = ax2.scatter(
            sv_s,
            np.full_like(sv_s, row_i, dtype=np.float64) + jitter,
            c=norm_val,
            cmap="coolwarm",
            s=8,
            alpha=0.55,
            edgecolors="none",
        )
    ax2.axvline(0.0, color="#333333", linestyle="--", linewidth=1.0)
    ax2.set_yticks(np.arange(len(top12_num)))
    ax2.set_yticklabels(top12_num, fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel("SHAP Value [Seconds] (Impact on ActualVsIEGap)", fontsize=10, fontweight="bold")
    ax2.set_title("Panel B: Top 12 Numeric Feature Directionality (Blue=Low, Red=High)", fontsize=11.5, fontweight="bold")
    ax2.grid(axis="x", linestyle="--", alpha=0.35)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    cbar = fig.colorbar(sc, ax=ax2, fraction=0.035, pad=0.02)
    cbar.set_label("Normalized Feature Value", fontsize=8.5)

    # Panel C: Group Shapley Physical Clusters
    ax3 = axes[1, 0]
    y_cl = np.arange(len(group_df))
    bar_h = 0.36
    ax3.barh(
        y_cl - bar_h / 2,
        group_df["Group_Mean_Abs_SHAP_Sec"],
        height=bar_h,
        color="#1B7837",
        label="Group Mean(|Sum SHAP|) [Net Cluster]",
    )
    ax3.barh(
        y_cl + bar_h / 2,
        group_df["Sum_Member_Mean_Abs_SHAP_Sec"],
        height=bar_h,
        color="#A6DBA0",
        label="Sum of Member Mean(|SHAP|)",
    )
    ax3.set_yticks(y_cl)
    ax3.set_yticklabels(group_df["Cluster"], fontsize=9.5, fontweight="bold")
    ax3.invert_yaxis()
    ax3.set_xlabel("Cluster Attribution [Seconds]", fontsize=10, fontweight="bold")
    ax3.set_title("Panel C: Physical Cluster Group Shapley (Collinear Consolidation)", fontsize=11.5, fontweight="bold")
    ax3.grid(axis="x", linestyle="--", alpha=0.35)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    for idx, r in group_df.iterrows():
        ax3.text(
            r["Group_Mean_Abs_SHAP_Sec"] + 4.0,
            idx - bar_h / 2,
            f"{r['Group_Mean_Abs_SHAP_Sec']:.1f}s ({r['Group_Share_Pct']:.1f}%)",
            va="center",
            fontsize=8.3,
            color="#115522",
        )
    ax3.legend(loc="lower right", fontsize=8.8)

    # Panel D: Single High-Deviation Sample Waterfall
    ax4 = axes[1, 1]
    target_sample_idx = int(np.argmax(np.abs(y_pred - base_value)))
    sv_sample = shap_values[target_sample_idx]
    order_abs = np.argsort(np.abs(sv_sample))[::-1]
    top_k = 10
    top_indices = order_abs[:top_k]
    other_val = float(sv_sample[order_abs[top_k:]].sum())

    wf_labels = [f"{feature_cols[i]}={X_nov.iloc[target_sample_idx][feature_cols[i]]}" for i in top_indices]
    wf_labels = [lbl[:34] + "..." if len(lbl) > 37 else lbl for lbl in wf_labels]
    wf_labels.append(f"Other {len(feature_cols) - top_k} Features")
    wf_vals = [float(sv_sample[i]) for i in top_indices] + [other_val]

    running = base_value
    starts = []
    for v in wf_vals:
        starts.append(running if v >= 0 else running + v)
        running += v

    wf_y = np.arange(len(wf_vals))
    wf_colors = ["#C62828" if v >= 0 else "#1565C0" for v in wf_vals]
    ax4.barh(wf_y, np.abs(wf_vals), left=starts, color=wf_colors, height=0.62, alpha=0.9)
    ax4.axvline(base_value, color="#666666", linestyle=":", linewidth=1.2, label=f"Base={base_value:.1f}s")
    ax4.axvline(y_pred[target_sample_idx], color="#111111", linestyle="-", linewidth=1.3, label=f"Pred={y_pred[target_sample_idx]:.1f}s")
    ax4.set_yticks(wf_y)
    ax4.set_yticklabels(wf_labels, fontsize=8.5)
    ax4.invert_yaxis()
    ax4.set_xlabel("ActualVsIEGap [Seconds]", fontsize=10, fontweight="bold")
    ax4.set_title(
        f"Panel D: High-Deviation Sample Waterfall (Nov #{target_sample_idx}, Actual={y_nov[target_sample_idx]:.1f}s)",
        fontsize=11.5,
        fontweight="bold",
    )
    ax4.grid(axis="x", linestyle="--", alpha=0.35)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)
    ax4.legend(loc="lower right", fontsize=8.8)

    fig.suptitle("Step 3 Part 2 Stage 1: CatBoost TreeSHAP & Group Shapley Attribution (Nov Blind Test)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(OUT_SHAP_PNG, dpi=200)
    plt.close(fig)


def run_stage2_permutation(
    cb_models: List[cb.CatBoostRegressor],
    nov_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    shap_df: pd.DataFrame,
    logger: logging.Logger,
    n_repeats: int = 5,
) -> pd.DataFrame:
    """Stage 2: Out-of-time Nov blind test Permutation Importance across all 75 features."""
    t0 = time.time()
    logger.info("[INFO] === Stage 2: Nov Blind Permutation Importance (75 features x R=%d) ===", n_repeats)

    X_nov = nov_df[feature_cols].copy()
    y_nov = nov_df[target_col].to_numpy(dtype=np.float64)

    base_pred = predict_catboost_ensemble(cb_models, X_nov)
    base_r2 = float(r2_score(y_nov, base_pred))
    base_rmse = float(np.sqrt(mean_squared_error(y_nov, base_pred)))
    logger.info("[INFO] Stage 2 baseline Nov R2=%.5f, RMSE=%.4f sec.", base_r2, base_rmse)

    rows: List[Dict[str, Any]] = []
    for f_idx, feat in enumerate(feature_cols):
        orig_col = X_nov[feat].copy()
        delta_r2_list = []
        delta_rmse_list = []

        for r in range(n_repeats):
            rng = np.random.default_rng(RANDOM_STATE * 100 + r)
            perm_vals = orig_col.to_numpy().copy()
            rng.shuffle(perm_vals)
            X_nov[feat] = perm_vals
            y_perm = predict_catboost_ensemble(cb_models, X_nov)
            delta_r2_list.append(base_r2 - float(r2_score(y_nov, y_perm)))
            delta_rmse_list.append(float(np.sqrt(mean_squared_error(y_nov, y_perm))) - base_rmse)

        X_nov[feat] = orig_col
        rows.append({
            "Feature": feat,
            "Base_Nov_R2": base_r2,
            "Base_Nov_RMSE": base_rmse,
            "Delta_R2_Mean": float(np.mean(delta_r2_list)),
            "Delta_R2_Std": float(np.std(delta_r2_list)),
            "Delta_RMSE_Mean": float(np.mean(delta_rmse_list)),
            "Delta_RMSE_Std": float(np.std(delta_rmse_list)),
        })
        if (f_idx + 1) % 25 == 0 or (f_idx + 1) == len(feature_cols):
            logger.info("[INFO] Stage 2 progress: %d / %d features evaluated.", f_idx + 1, len(feature_cols))

    perm_df = pd.DataFrame(rows).sort_values("Delta_R2_Mean", ascending=False).reset_index(drop=True)
    perm_df["Perm_Rank"] = np.arange(1, len(perm_df) + 1)

    perm_df = perm_df.merge(
        shap_df[["Feature", "Cluster", "Feature_Type", "Mean_Abs_SHAP_Sec", "SHAP_Rank"]],
        on="Feature",
        how="left",
    )
    perm_df["Rank_Diff_Perm_Minus_SHAP"] = perm_df["Perm_Rank"] - perm_df["SHAP_Rank"]
    perm_df["Consensus_Rank_Score"] = 0.5 * (perm_df["SHAP_Rank"] + perm_df["Perm_Rank"])
    perm_df["Stability_Status"] = np.where(
        perm_df["Delta_R2_Mean"] <= 0.0,
        "DECAY_OR_NOISE",
        np.where(np.abs(perm_df["Rank_Diff_Perm_Minus_SHAP"]) <= 5, "ROBUST_CONSENSUS", "SHIFTED_RANK"),
    )

    perm_df.to_csv(OUT_PERM_CSV, index=False, encoding="utf-8-sig")
    logger.info("[PASS] Stage 2 CSV saved: %s (Elapsed=%.2fs)", OUT_PERM_CSV, time.time() - t0)
    return perm_df


def run_stage3_lightweight_sage(
    cb_models: List[cb.CatBoostRegressor],
    nov_df: pd.DataFrame,
    feature_cols: List[str],
    perm_df: pd.DataFrame,
    target_col: str,
    logger: logging.Logger,
    n_eval: int = 1000,
    n_bg: int = 16,
    n_perms: int = 12,
) -> pd.DataFrame:
    """Stage 3: Lightweight Grouped SAGE (Covert et al., NeurIPS 2020) for Top 10 features."""
    t0 = time.time()
    top10_df = perm_df.sort_values(["Consensus_Rank_Score", "SHAP_Rank"]).head(10).reset_index(drop=True)
    top10_feats = top10_df["Feature"].tolist()
    bg_group_feats = [f for f in feature_cols if f not in set(top10_feats)]
    players = top10_feats + ["BG_GROUP_65_Features"]
    n_players = len(players)

    logger.info(
        "[INFO] === Stage 3: Lightweight Grouped SAGE (n_eval=%d, n_bg=%d, n_perms=%d) ===",
        n_eval,
        n_bg,
        n_perms,
    )
    logger.info("[INFO] Stage 3 Top 10 targets: %s", ", ".join(top10_feats))

    rng = np.random.default_rng(RANDOM_STATE)
    perm_indices = rng.permutation(len(nov_df))
    eval_idx = perm_indices[:n_eval]
    bg_idx = perm_indices[n_eval : n_eval + n_bg]

    X_eval = nov_df.iloc[eval_idx][feature_cols].reset_index(drop=True)
    y_eval = nov_df.iloc[eval_idx][target_col].to_numpy(dtype=np.float64)
    X_bg = nov_df.iloc[bg_idx][feature_cols].reset_index(drop=True)

    # Pre-build repeated background template (n_eval * n_bg rows)
    X_bg_tiled = pd.concat([X_bg] * n_eval, ignore_index=True)
    # Row mapping: block i has rows [i*n_bg : (i+1)*n_bg]
    eval_repeat_idx = np.repeat(np.arange(n_eval), n_bg)
    X_eval_tiled = X_eval.iloc[eval_repeat_idx].reset_index(drop=True)

    def eval_coalition_mse(active_players: List[int]) -> np.ndarray:
        """Compute per-sample squared error using marginalized prediction over n_bg draws."""
        X_mix = X_bg_tiled.copy()
        for p_idx in active_players:
            if p_idx < 10:
                col = top10_feats[p_idx]
                X_mix[col] = X_eval_tiled[col].to_numpy()
            else:
                for col in bg_group_feats:
                    X_mix[col] = X_eval_tiled[col].to_numpy()
        preds_flat = predict_catboost_ensemble(cb_models, X_mix)
        preds_mean = preds_flat.reshape(n_eval, n_bg).mean(axis=1)
        return (y_eval - preds_mean) ** 2

    # Empty coalition loss
    loss_empty = eval_coalition_mse([])
    # Store per-sample marginal loss reductions: shape (n_perms, n_players, n_eval)
    marginals = np.zeros((n_perms, n_players, n_eval), dtype=np.float64)

    for p_iter in range(n_perms):
        order = rng.permutation(n_players)
        active: List[int] = []
        prev_loss = loss_empty
        for p_idx in order:
            active.append(int(p_idx))
            curr_loss = eval_coalition_mse(active)
            marginals[p_iter, p_idx, :] = prev_loss - curr_loss
            prev_loss = curr_loss
        logger.info("[INFO] Stage 3 permutation %d/%d completed.", p_iter + 1, n_perms)

    # Average across permutations first -> (n_players, n_eval), then mean & SE across n_eval
    per_sample_sage = marginals.mean(axis=0)
    sage_vals = per_sample_sage.mean(axis=1)
    sage_se = per_sample_sage.std(axis=1, ddof=1) / np.sqrt(float(n_eval))

    total_mse_reduction = float(np.mean(loss_empty) - np.mean(eval_coalition_mse(list(range(n_players)))))
    sage_sum = float(np.sum(sage_vals))

    rows: List[Dict[str, Any]] = []
    for idx, name in enumerate(players):
        rows.append({
            "Player": name,
            "Is_Background_Group": idx == 10,
            "SAGE_MSE_Reduction": float(sage_vals[idx]),
            "SAGE_Std_Error": float(sage_se[idx]),
            "SAGE_Share_Pct": 100.0 * float(sage_vals[idx]) / max(sage_sum, 1e-9),
            "Total_Explained_MSE_Drop": total_mse_reduction,
        })

    sage_df = pd.DataFrame(rows)
    top10_part = sage_df[~sage_df["Is_Background_Group"]].sort_values("SAGE_MSE_Reduction", ascending=False)
    bg_part = sage_df[sage_df["Is_Background_Group"]]
    sage_df = pd.concat([top10_part, bg_part], ignore_index=True)
    sage_df["SAGE_Rank"] = np.arange(1, len(sage_df) + 1)

    sage_df.to_csv(OUT_SAGE_CSV, index=False, encoding="utf-8-sig")
    logger.info(
        "[PASS] Stage 3 CSV saved: %s | Sum(SAGE)=%.2f vs Total_MSE_Drop=%.2f (Elapsed=%.2fs)",
        OUT_SAGE_CSV,
        sage_sum,
        total_mse_reduction,
        time.time() - t0,
    )
    return sage_df


def run_stage4_loco_ablation(
    dev_df: pd.DataFrame,
    nov_df: pd.DataFrame,
    fold_ids: np.ndarray,
    num_cols: List[str],
    cat_cols: List[str],
    feature_cols: List[str],
    target_col: str,
    shap_df: pd.DataFrame,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Stage 4: Targeted LOCO retraining on Dev Fold 1 and Nov blind test evaluation."""
    t0 = time.time()
    logger.info("[INFO] === Stage 4: Targeted LOCO Retraining (CatBoost GPU) ===")

    tr_mask = fold_ids != 0
    va_mask = fold_ids == 0
    X_tr = dev_df.loc[tr_mask, feature_cols].reset_index(drop=True)
    y_tr = dev_df.loc[tr_mask, target_col].to_numpy(dtype=np.float64)
    X_va = dev_df.loc[va_mask, feature_cols].reset_index(drop=True)
    y_va = dev_df.loc[va_mask, target_col].to_numpy(dtype=np.float64)
    X_nov = nov_df[feature_cols].reset_index(drop=True)
    y_nov = nov_df[target_col].to_numpy(dtype=np.float64)

    top8_feats = shap_df.head(8)["Feature"].tolist()
    experiments: List[Tuple[str, str, List[str]]] = [
        ("Baseline_All_75", "Baseline", feature_cols),
        ("Group_LOCO_Pure_Physical_49", "Group_LOCO", num_cols),
        ("Group_LOCO_Pure_Categorical_26", "Group_LOCO", cat_cols),
    ]
    for rank_i, feat in enumerate(top8_feats, start=1):
        experiments.append((
            f"Single_LOCO_Drop_{rank_i}_{feat}",
            "Single_LOCO",
            [f for f in feature_cols if f != feat],
        ))

    cat_set = set(cat_cols)
    use_gpu = True
    rows: List[Dict[str, Any]] = []
    base_nov_r2 = None
    base_nov_rmse = None

    for exp_name, exp_type, cols_keep in experiments:
        t_exp = time.time()
        exp_cats = [c for c in cols_keep if c in cat_set]
        params = dict(
            iterations=1500,
            learning_rate=0.05,
            depth=7,
            l2_leaf_reg=5.0,
            loss_function="RMSE",
            eval_metric="RMSE",
            early_stopping_rounds=100,
            cat_features=exp_cats,
            task_type="GPU" if use_gpu else "CPU",
            random_seed=RANDOM_STATE,
            verbose=0,
        )
        model = cb.CatBoostRegressor(**params)
        try:
            model.fit(X_tr[cols_keep], y_tr, eval_set=(X_va[cols_keep], y_va))
        except Exception as exc:
            if use_gpu:
                logger.warning("[WARN] CatBoost GPU failed in %s (%s); falling back to CPU.", exp_name, exc)
                use_gpu = False
                params["task_type"] = "CPU"
                model = cb.CatBoostRegressor(**params)
                model.fit(X_tr[cols_keep], y_tr, eval_set=(X_va[cols_keep], y_va))
            else:
                raise

        pred_nov = model.predict(X_nov[cols_keep])
        nov_r2 = float(r2_score(y_nov, pred_nov))
        nov_rmse = float(np.sqrt(mean_squared_error(y_nov, pred_nov)))
        elapsed = time.time() - t_exp

        if base_nov_r2 is None:
            base_nov_r2 = nov_r2
            base_nov_rmse = nov_rmse

        delta_r2 = base_nov_r2 - nov_r2
        delta_rmse = nov_rmse - base_nov_rmse
        rows.append({
            "Experiment": exp_name,
            "Experiment_Type": exp_type,
            "Feature_Count": len(cols_keep),
            "Nov_Blind_R2": nov_r2,
            "Nov_Blind_RMSE": nov_rmse,
            "Delta_R2_LOCO": delta_r2,
            "Delta_RMSE_LOCO": delta_rmse,
            "Best_Iteration": int(model.get_best_iteration()),
            "Train_Time_Sec": round(elapsed, 2),
        })
        logger.info(
            "[INFO] %-42s | Nov R2=%.5f (dR2=%.5f) | RMSE=%.3fs | Iter=%d | %.1fs",
            exp_name,
            nov_r2,
            delta_r2,
            nov_rmse,
            int(model.get_best_iteration()),
            elapsed,
        )

    loco_df = pd.DataFrame(rows)
    loco_df.to_csv(OUT_LOCO_CSV, index=False, encoding="utf-8-sig")
    logger.info("[PASS] Stage 4 CSV saved: %s (Total Elapsed=%.2fs)", OUT_LOCO_CSV, time.time() - t0)
    return loco_df


def compute_1d_ale(
    cb_models: List[cb.CatBoostRegressor],
    X_df: pd.DataFrame,
    feature_name: str,
    n_bins: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute centered 1D Accumulated Local Effects (Apley & Zhu, 2020) over quantile bins."""
    x_vals = X_df[feature_name].to_numpy(dtype=np.float64)
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(x_vals, quantiles))
    if len(edges) < 3:
        raise ValueError(f"Insufficient distinct quantile edges for ALE on {feature_name}")

    n_intervals = len(edges) - 1
    local_effects = np.zeros(n_intervals, dtype=np.float64)
    bin_counts = np.zeros(n_intervals, dtype=np.int64)

    for k in range(n_intervals):
        low_e, high_e = edges[k], edges[k + 1]
        if k == 0:
            mask = (x_vals >= low_e) & (x_vals <= high_e)
        else:
            mask = (x_vals > low_e) & (x_vals <= high_e)
        cnt = int(mask.sum())
        bin_counts[k] = cnt
        if cnt == 0:
            continue

        X_low = X_df.loc[mask].copy()
        X_high = X_low.copy()
        X_low[feature_name] = low_e
        X_high[feature_name] = high_e

        pred_low = predict_catboost_ensemble(cb_models, X_low)
        pred_high = predict_catboost_ensemble(cb_models, X_high)
        local_effects[k] = float(np.mean(pred_high - pred_low))

    cum_effects = np.concatenate(([0.0], np.cumsum(local_effects)))
    midpoints = 0.5 * (edges[:-1] + edges[1:])
    ale_raw = 0.5 * (cum_effects[:-1] + cum_effects[1:])
    total_cnt = max(int(bin_counts.sum()), 1)
    ale_centered = ale_raw - float(np.sum(ale_raw * bin_counts) / total_cnt)
    return midpoints, ale_centered, bin_counts


def run_stage5_ale(
    cb_models: List[cb.CatBoostRegressor],
    nov_df: pd.DataFrame,
    feature_cols: List[str],
    shap_df: pd.DataFrame,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Stage 5: Compute 1D ALE curves for Top 6 continuous numeric parameters."""
    t0 = time.time()
    logger.info("[INFO] === Stage 5: 1D ALE Curves for Top 6 Continuous Numeric Parameters ===")

    X_nov = nov_df[feature_cols].copy()
    num_ranked = shap_df[shap_df["Feature_Type"] == "Numeric"]["Feature"].tolist()
    ale_candidates = [
        f for f in num_ranked
        if f not in NON_CONTINUOUS_FOR_ALE and X_nov[f].nunique() >= 10
    ][:6]
    logger.info("[INFO] Stage 5 selected Top 6 ALE features: %s", ", ".join(ale_candidates))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5))
    axes_flat = axes.flatten()
    summary_rows: List[Dict[str, Any]] = []

    for idx, feat in enumerate(ale_candidates):
        midpoints, ale_vals, counts = compute_1d_ale(cb_models, X_nov, feat, n_bins=20)
        best_i = int(np.argmin(ale_vals))
        worst_i = int(np.argmax(ale_vals))
        ale_span = float(ale_vals[worst_i] - ale_vals[best_i])

        summary_rows.append({
            "Feature": feat,
            "SHAP_Rank": int(shap_df.loc[shap_df["Feature"] == feat, "SHAP_Rank"].iloc[0]),
            "ALE_Bins": len(midpoints),
            "Min_ALE_Feature_Value": float(midpoints[best_i]),
            "Min_ALE_Sec": float(ale_vals[best_i]),
            "Max_ALE_Feature_Value": float(midpoints[worst_i]),
            "Max_ALE_Sec": float(ale_vals[worst_i]),
            "ALE_Peak_to_Trough_Sec": ale_span,
        })

        ax = axes_flat[idx]
        ax.plot(midpoints, ale_vals, color="#2B5C8F", linewidth=2.2, marker="o", markersize=4.5)
        ax.axhline(0.0, color="#555555", linestyle="--", linewidth=1.0)
        ax.scatter([midpoints[best_i]], [ale_vals[best_i]], color="#1B7837", s=70, zorder=5, label=f"Minimum ALE: {midpoints[best_i]:.1f} ({ale_vals[best_i]:+.1f}s)")
        ax.scatter([midpoints[worst_i]], [ale_vals[worst_i]], color="#C62828", s=70, zorder=5, label=f"Maximum ALE: {midpoints[worst_i]:.1f} ({ale_vals[worst_i]:+.1f}s)")
        ax.set_title(f"{feat} (Span={ale_span:.1f}s)", fontsize=11, fontweight="bold")
        ax.set_xlabel(feat, fontsize=9.5)
        ax.set_ylabel("Centered ALE [Seconds]", fontsize=9.5)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="best", fontsize=8.3)

    fig.suptitle("Step 3 Part 2 Stage 5: 1D Accumulated Local Effects on the November Historical Holdout", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_ALE_PNG, dpi=200)
    plt.close(fig)

    ale_df = pd.DataFrame(summary_rows)
    ale_df.to_csv(OUT_ALE_CSV, index=False, encoding="utf-8-sig")
    logger.info("[PASS] Stage 5 outputs saved: %s, %s (Elapsed=%.2fs)", OUT_ALE_CSV, OUT_ALE_PNG, time.time() - t0)
    return ale_df


def _dml_crossfit_ols(
    df_dev: pd.DataFrame,
    t_vals: np.ndarray,
    y_vals: np.ndarray,
    w_cols: List[str],
    cat_cols: List[str],
    fold_splits: List[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[float, float, float, float, float, float]:
    """Run 5-fold work-order-isolated Partially Linear DML with LightGBM nuisance models."""
    n_samples = len(df_dev)
    y_hat_oof = np.zeros(n_samples, dtype=np.float64)
    t_hat_oof = np.zeros(n_samples, dtype=np.float64)

    w_cats = [c for c in w_cols if c in set(cat_cols)]
    X_w_raw = df_dev[w_cols].copy()

    lgb_params = dict(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=30,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        cat_smooth=20.0,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=-1,
    )

    for tr_idx, va_idx in fold_splits:
        X_tr = X_w_raw.iloc[tr_idx].copy()
        X_va = X_w_raw.iloc[va_idx].copy()
        for col in w_cats:
            cats = sorted(X_tr[col].unique().tolist())
            X_tr[col] = pd.Categorical(X_tr[col], categories=cats)
            X_va[col] = pd.Categorical(X_va[col], categories=cats)

        m_y = lgb.LGBMRegressor(**lgb_params)
        m_y.fit(X_tr, y_vals[tr_idx])
        y_hat_oof[va_idx] = m_y.predict(X_va)

        m_t = lgb.LGBMRegressor(**lgb_params)
        m_t.fit(X_tr, t_vals[tr_idx])
        t_hat_oof[va_idx] = m_t.predict(X_va)

    y_res = y_vals - y_hat_oof
    t_res = t_vals - t_hat_oof

    denom = float(np.sum(t_res ** 2))
    theta = float(np.sum(t_res * y_res) / denom)
    eps = y_res - theta * t_res

    # HC1 heteroskedasticity-robust standard error
    hc1_scale = float(n_samples) / float(n_samples - 1)
    var_theta = hc1_scale * float(np.sum((t_res * eps) ** 2)) / (denom ** 2)
    se = float(np.sqrt(var_theta))
    t_stat = theta / max(se, 1e-12)
    p_val = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=n_samples - 1)))
    ci_low = theta - 1.96 * se
    ci_high = theta + 1.96 * se

    nuisance_y_r2 = float(r2_score(y_vals, y_hat_oof))
    nuisance_t_r2 = float(r2_score(t_vals, t_hat_oof))
    return theta, se, ci_low, ci_high, p_val, nuisance_y_r2, nuisance_t_r2


def run_stage6_dml(
    dev_df: pd.DataFrame,
    feature_cols: List[str],
    cat_cols: List[str],
    target_col: str,
    shap_df: pd.DataFrame,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Stage 6: Partially Linear DML for Top 3 controllable physical parameters + Placebo Refutation."""
    t0 = time.time()
    logger.info("[INFO] === Stage 6: 5-Fold Work-Order-Isolated DML + Placebo Refutation ===")

    num_ranked = shap_df[shap_df["Feature_Type"] == "Numeric"]["Feature"].tolist()
    dml_targets = [
        f for f in num_ranked
        if f not in NON_CONTROLLABLE_FOR_DML and dev_df[f].nunique() >= 15
    ][:3]
    logger.info("[INFO] Stage 6 selected Top 3 controllable continuous treatments: %s", ", ".join(dml_targets))

    gkf = GroupKFold(n_splits=5)
    dev_groups = dev_df["work_order_id"].to_numpy()
    fold_splits = list(gkf.split(dev_df[feature_cols], dev_df[target_col], groups=dev_groups))
    y_dev = dev_df[target_col].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(RANDOM_STATE)

    rows: List[Dict[str, Any]] = []
    for treat in dml_targets:
        t_vals = dev_df[treat].to_numpy(dtype=np.float64)
        w_cols = [f for f in feature_cols if f != treat]

        # Naive OLS slope (unadjusted)
        t_centered = t_vals - np.mean(t_vals)
        y_centered = y_dev - np.mean(y_dev)
        naive_slope = float(np.sum(t_centered * y_centered) / np.sum(t_centered ** 2))

        # Partially Linear DML
        theta, se, ci_low, ci_high, p_val, n_y_r2, n_t_r2 = _dml_crossfit_ols(
            dev_df, t_vals, y_dev, w_cols, cat_cols, fold_splits
        )

        # Placebo Refutation (random permutation of T)
        t_placebo = rng.permutation(t_vals)
        p_theta, p_se, _, _, p_pval, _, _ = _dml_crossfit_ols(
            dev_df, t_placebo, y_dev, w_cols, cat_cols, fold_splits
        )
        placebo_status = "PASS" if p_pval > 0.05 else "WARN"
        sign_flipped = (naive_slope * theta) < 0.0

        rows.append({
            "Treatment_Feature": treat,
            "Naive_OLS_Slope_Sec_Per_Unit": naive_slope,
            "DML_Theta_Sec_Per_Unit": theta,
            "DML_Std_Error": se,
            "CI_95_Low": ci_low,
            "CI_95_High": ci_high,
            "DML_P_Value": p_val,
            "Nuisance_Y_OOF_R2": n_y_r2,
            "Nuisance_T_OOF_R2": n_t_r2,
            "Simpson_Sign_Flip": sign_flipped,
            "Placebo_Theta": p_theta,
            "Placebo_Std_Error": p_se,
            "Placebo_P_Value": p_pval,
            "Placebo_Check": placebo_status,
        })
        logger.info(
            "[%s] DML %-35s | Naive=%.4f | DML Theta=%.4f (95%% CI [%.4f, %.4f], p=%.2e) | Placebo Theta=%.4f (p=%.3f)",
            placebo_status,
            treat,
            naive_slope,
            theta,
            ci_low,
            ci_high,
            p_val,
            p_theta,
            p_pval,
        )

    dml_df = pd.DataFrame(rows)
    dml_df.to_csv(OUT_DML_CSV, index=False, encoding="utf-8-sig")
    logger.info("[PASS] Stage 6 CSV saved: %s (Elapsed=%.2fs)", OUT_DML_CSV, time.time() - t0)
    return dml_df


def create_synthesis_dashboard(
    perm_df: pd.DataFrame,
    sage_df: pd.DataFrame,
    loco_df: pd.DataFrame,
    dml_df: pd.DataFrame,
) -> None:
    """Create 2x2 executive synthesis dashboard integrating Stages 1-6."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12.5))

    # Panel A: SHAP Rank vs Nov Permutation Rank (Top 25)
    ax1 = axes[0, 0]
    sub_p = perm_df[perm_df["SHAP_Rank"] <= 25].copy()
    colors_p = [
        "#1B7837" if s == "ROBUST_CONSENSUS" else ("#C62828" if s == "DECAY_OR_NOISE" else "#D97724")
        for s in sub_p["Stability_Status"]
    ]
    ax1.scatter(sub_p["SHAP_Rank"], sub_p["Perm_Rank"], c=colors_p, s=65, alpha=0.88, edgecolors="#222222", linewidths=0.5)
    ax1.plot([1, 25], [1, 25], linestyle="--", color="#666666", linewidth=1.2)
    for _, r in sub_p.head(10).iterrows():
        ax1.annotate(r["Feature"][:22], (r["SHAP_Rank"], r["Perm_Rank"]), xytext=(4, 3), textcoords="offset points", fontsize=7.8)
    ax1.set_xlabel("Stage 1 TreeSHAP Rank (Lower = More Important)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Stage 2 Nov Permutation Rank", fontsize=10, fontweight="bold")
    ax1.set_title("Panel A: TreeSHAP and November Holdout Permutation Ranks", fontsize=11.5, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel B: Stage 3 Top 10 Grouped SAGE Loss Reduction
    ax2 = axes[0, 1]
    sage_top10 = sage_df[~sage_df["Is_Background_Group"]].copy()
    y_s = np.arange(len(sage_top10))
    ax2.barh(
        y_s,
        sage_top10["SAGE_MSE_Reduction"],
        xerr=1.96 * sage_top10["SAGE_Std_Error"],
        color="#2B5C8F",
        alpha=0.9,
        capsize=3,
    )
    ax2.set_yticks(y_s)
    ax2.set_yticklabels(sage_top10["Player"], fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel("SAGE MSE Loss Reduction (with 95% CI)", fontsize=10, fontweight="bold")
    ax2.set_title("Panel B: Top 10 Lightweight Grouped SAGE (Nov Subsample)", fontsize=11.5, fontweight="bold")
    ax2.grid(axis="x", linestyle="--", alpha=0.35)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Panel C: Stage 4 LOCO Ablation Delta R2
    ax3 = axes[1, 0]
    loco_plot = loco_df[loco_df["Experiment_Type"] != "Baseline"].copy()
    y_l = np.arange(len(loco_plot))
    l_colors = ["#8E24AA" if t == "Group_LOCO" else "#D97724" for t in loco_plot["Experiment_Type"]]
    ax3.barh(y_l, loco_plot["Delta_R2_LOCO"], color=l_colors, alpha=0.9)
    ax3.set_yticks(y_l)
    ax3.set_yticklabels(loco_plot["Experiment"].str.replace("Single_LOCO_Drop_", "Drop ", regex=False), fontsize=8.8)
    ax3.invert_yaxis()
    ax3.set_xlabel("November Historical Holdout R² Drop", fontsize=10, fontweight="bold")
    ax3.set_title("Panel C: LOCO Refit Comparison", fontsize=11.5, fontweight="bold")
    ax3.grid(axis="x", linestyle="--", alpha=0.35)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    for idx, val in enumerate(loco_plot["Delta_R2_LOCO"]):
        ax3.text(max(val, 0.0) + 0.002, idx, f"{val:+.4f}", va="center", fontsize=8.2)

    # Panel D: Stage 6 DML Causal Effects vs Naive OLS vs Placebo
    ax4 = axes[1, 1]
    y_d = np.arange(len(dml_df))
    ax4.axvline(0.0, color="#333333", linestyle="--", linewidth=1.1)
    ax4.errorbar(
        dml_df["DML_Theta_Sec_Per_Unit"],
        y_d - 0.14,
        xerr=1.96 * dml_df["DML_Std_Error"],
        fmt="o",
        color="#1B7837",
        markersize=7,
        capsize=4,
        label="Historical DML Estimate (95% CI)",
    )
    ax4.scatter(dml_df["Naive_OLS_Slope_Sec_Per_Unit"], y_d + 0.02, color="#D97724", marker="s", s=55, label="Naive OLS (Confounded)")
    ax4.errorbar(
        dml_df["Placebo_Theta"],
        y_d + 0.18,
        xerr=1.96 * dml_df["Placebo_Std_Error"],
        fmt="^",
        color="#757575",
        markersize=6.5,
        capsize=3,
        label="Placebo Refutation (95% CI)",
    )
    ax4.set_yticks(y_d)
    ax4.set_yticklabels(dml_df["Treatment_Feature"], fontsize=9.5, fontweight="bold")
    ax4.invert_yaxis()
    ax4.set_xlabel("Effect Slope [Seconds per Unit Change]", fontsize=10, fontweight="bold")
    ax4.set_title("Panel D: Historical DML Estimates and Placebo Check", fontsize=11.5, fontweight="bold")
    ax4.grid(axis="x", linestyle="--", alpha=0.35)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)
    ax4.legend(loc="best", fontsize=8.5)

    fig.suptitle("Step 3 Part 2 Synthesis Dashboard: Attribution, LOCO, and Historical DML Estimates", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(OUT_DASH_PNG, dpi=200)
    plt.close(fig)


def main() -> None:
    overall_start = time.time()
    logger = setup_logging(LOG_PATH)
    configure_matplotlib()
    warnings.filterwarnings("ignore", category=UserWarning)
    os.makedirs("reports", exist_ok=True)

    logger.info("[INFO] Starting Step 3 Part 2: Feature Attribution & Causal Inference Pipeline.")

    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)
    with open(BUNDLE_PATH, "rb") as f:
        bundle = pickle.load(f)

    num_cols: List[str] = schema["numeric_features"]
    cat_cols: List[str] = schema["categorical_features"]
    target_col: str = schema["target_col"]
    feature_cols: List[str] = bundle["feature_schema"]["feature_cols"]
    fold_ids: np.ndarray = bundle["fold_assignments"]
    cb_models: List[cb.CatBoostRegressor] = [e["estimator"] for e in bundle["models"]["CatBoost"]]
    stored_nov_pred: np.ndarray = bundle["nov_predictions"]["CatBoost"]

    df = pd.read_csv(DATA_PATH, low_memory=False)
    for col in cat_cols:
        df[col] = df[col].astype(str)
    df["work_order_id"] = pd.util.hash_pandas_object(df[WORK_ORDER_COLS], index=False).astype(np.uint64)

    dev_df = df.loc[df["ActualStartMonth"].isin([8, 9, 10])].reset_index(drop=True)
    nov_df = df.loc[df["ActualStartMonth"] == 11].reset_index(drop=True)
    assert len(dev_df) == 43451 and len(nov_df) == 9694, "Unexpected Dev/Nov split sizes."
    logger.info("[PASS] Loaded bundle and dataset: Dev=%d, Nov=%d, Features=%d.", len(dev_df), len(nov_df), len(feature_cols))

    # Stage 1
    shap_df, _, _, _ = run_stage1_treeshap(
        cb_models, nov_df, feature_cols, cat_cols, target_col, stored_nov_pred, logger
    )

    # Stage 2
    perm_df = run_stage2_permutation(
        cb_models, nov_df, feature_cols, target_col, shap_df, logger, n_repeats=5
    )

    # Stage 3
    sage_df = run_stage3_lightweight_sage(
        cb_models, nov_df, feature_cols, perm_df, target_col, logger, n_eval=1000, n_bg=16, n_perms=12
    )

    # Stage 4
    loco_df = run_stage4_loco_ablation(
        dev_df, nov_df, fold_ids, num_cols, cat_cols, feature_cols, target_col, shap_df, logger
    )

    # Stage 5
    _ = run_stage5_ale(cb_models, nov_df, feature_cols, shap_df, logger)

    # Stage 6
    dml_df = run_stage6_dml(dev_df, feature_cols, cat_cols, target_col, shap_df, logger)

    # Final Synthesis Dashboard
    create_synthesis_dashboard(perm_df, sage_df, loco_df, dml_df)
    logger.info("[PASS] Final synthesis dashboard saved: %s", OUT_DASH_PNG)

    logger.info("[PASS] Step 3 Part 2 completed successfully in %.2f seconds.", time.time() - overall_start)


if __name__ == "__main__":
    main()
