# -*- coding: utf-8 -*-
"""
====================================================================
第二步（试纸 B）：素颜树团战探针 + Y-Scrambling 检验 + 辛普森悖论分家排查
(Step 2B: Strip B - Team Probe, Y-Scrambling & Simpson's Paradox Audit)
====================================================================
【核心任务】
试纸 A 看完了每个特征的“单兵作战能力”，现在进入试纸 B，重点排查三件事：
1. 【团战合作探针（5,000 行 5 折交叉验证，绝对零泄露）】：
   - 对比三个选手：Dummy 基线（全猜平均值）、Ridge 线性对照、HistGBDT 素颜浅层树。
   - 对 26 个无序文本特征实行“因材施教”：
     * 4 个超高基数 ID/几何代号列（类别数 > 250）：Pipeline 内部使用 TargetEncoder
       （5 折平滑目标编码，杜绝过拟合与维度爆炸）；
     * 22 个低/中基数无序文本列（<= 250）：OrdinalEncoder + HistGBDT 原生类别切分
       （或者给 Ridge 用 OneHotEncoder），绝不把编码序号当成大小关系！
2. 【防靠运气蒙对（19 次 Y-Scrambling 洗牌 + 影子卧底）】：
   - 把目标列 ActualVsIEGap 随机洗牌 19 次，每次重跑完整 5 折 HistGBDT 管道，
     用 Phipson & Smyth 公式计算置换检验 p = (C + 1) / (M + 1)，看真实分数是否
     碾压纯随机洗牌。
   - 注入 1 列标准正态噪声 __shadow_random_noise__，在 80/20 留出验证集上跑
     Permutation Importance，检验核心特征是否把“卧底”甩开十万八千里。
3. 【全量 53,145 行辛普森悖论分家排查】：
   - 分别按主力机台（WorkCenterNameCoupled，样本量 >= 200）和主力轮胎规格
     （tire_size，样本量 >= 200）切片，检查 49 个数值特征在各个“小圈子”内的
     Pearson r 极值、跨度 r_span 与正负撕裂情况，揪出被全表均值掩盖的真相！

【导出产物】
- reports/step2b_strip_b_simpson_report.csv（UTF-8 BOM，0 NaN，按最大撕裂跨度降序）
- reports/step2b_strip_b_summary.png（双拼总结图：左图团战+洗牌检验，右图 Top10 哑铃图）
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, TargetEncoder


# ====================================================================
# 第一部分：零泄露建模管道构造器（Dummy / Ridge / HistGBDT）
# ====================================================================

def build_ridge_pipeline(
    high_cat_cols: list,
    low_cat_cols: list,
    num_cols: list,
    random_state: int = 42,
) -> Pipeline:
    """
    构造防泄露的 Ridge 线性对照管道：
    - 超高基数文本列（>250 类）：TargetEncoder（在每折训练集内部做 cv=5 平滑编码）
    - 低/中基数文本列（<=250 类）：OneHotEncoder(handle_unknown='ignore')
    - 数值列（含影子列）：StandardScaler 标准化（均在折内 fit，绝不提前看验证集！）
    - 模型：Ridge(alpha=10.0) 正则化线性回归
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "high_cat",
                TargetEncoder(
                    target_type="continuous",
                    smooth="auto",
                    cv=5,
                    random_state=random_state,
                ),
                high_cat_cols,
            ),
            (
                "low_cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                low_cat_cols,
            ),
            (
                "num",
                StandardScaler(),
                num_cols,
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", Ridge(alpha=10.0)),
        ]
    )


def build_hgb_pipeline(
    high_cat_cols: list,
    low_cat_cols: list,
    num_cols: list,
    random_state: int = 42,
) -> Pipeline:
    """
    构造防泄露的素颜浅层树（HistGradientBoostingRegressor）管道：
    - 超高基数文本列（>250 类）：TargetEncoder（避免超出 HistGBDT 最大 255 类限制）
    - 低/中基数文本列（<=250 类）：OrdinalEncoder（未知类别映射为 np.nan）
      随后直接把这几列的索引交给 HistGBDT 的 categorical_features 参数，
      让树模型执行原生无序集合切分（Fisher 精确最优分裂），绝不把编码数字当大小！
    - 数值列（含影子列）：直接 passthrough
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "high_cat",
                TargetEncoder(
                    target_type="continuous",
                    smooth="auto",
                    cv=5,
                    random_state=random_state,
                ),
                high_cat_cols,
            ),
            (
                "low_cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=np.nan,
                ),
                low_cat_cols,
            ),
            (
                "num",
                "passthrough",
                num_cols,
            ),
        ],
        remainder="drop",
    )

    # ColumnTransformer 按顺序拼接：[high_cat, low_cat, num]
    # 因此 low_cat 在输出矩阵中的列索引紧跟在 high_cat 之后
    n_high = len(high_cat_cols)
    n_low = len(low_cat_cols)
    low_cat_indices = list(range(n_high, n_high + n_low))

    hgb_model = HistGradientBoostingRegressor(
        max_depth=5,
        max_iter=80,
        min_samples_leaf=25,
        categorical_features=low_cat_indices,
        random_state=random_state,
    )
    return Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", hgb_model),
        ]
    )


# ====================================================================
# 第二部分：全量辛普森悖论分家排查工具函数
# ====================================================================

def safe_pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """
    安全计算一维数组 x 与 y 的 Pearson 相关系数 r：
    若任一数组标准差接近 0（常数列，比如某个机台只生产同一款轮胎），
    则返回 np.nan 以便上层安全跳过，绝不抛出 RuntimeWarning 或产生脏数值。
    """
    if len(x) < 3:
        return np.nan
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denom = np.sqrt(np.sum(x_centered * x_centered) * np.sum(y_centered * y_centered))
    if denom <= 1e-12:
        return np.nan
    r = float(np.sum(x_centered * y_centered) / denom)
    return float(np.clip(r, -1.0, 1.0))


def compute_slice_correlations(
    df: pd.DataFrame,
    group_col: str,
    numeric_cols: list,
    target_col: str,
    min_samples: int = 200,
) -> tuple:
    """
    在完整数据集上按 group_col 分组（仅保留样本量 >= min_samples 的主力组），
    批量向量化计算 49 个数值特征在每个主力组内与 target_col 的 Pearson r。

    返回：
    - stats_dict: {feature_name: {"r_max": ..., "group_max": ..., "r_min": ..., "group_min": ..., "r_span": ..., "n_valid_groups": ...}}
    - n_qualifying_groups: 满足样本量 >= min_samples 的组数
    - covered_rows: 这些主力组总共覆盖的样本行数
    """
    group_counts = df[group_col].value_counts()
    qualifying_groups = group_counts[group_counts >= min_samples].index.tolist()
    n_qualifying = len(qualifying_groups)
    covered_rows = int(group_counts[qualifying_groups].sum())

    # 为每个特征收集其在各个有效组内的 (group_name, r_value)
    feat_group_rs = {col: [] for col in numeric_cols}

    # 按组切片并向量化计算 49 列同时与 y 的 Pearson r，速度极快且零 Warning
    df_sub = df[df[group_col].isin(qualifying_groups)]
    for grp_name, grp_df in df_sub.groupby(group_col, sort=False):
        y_g = grp_df[target_col].to_numpy(dtype=np.float64)
        y_c = y_g - np.mean(y_g)
        y_ss = float(np.sum(y_c * y_c))
        if y_ss <= 1e-12:
            # 该组目标列自身为常数时，整体跳过
            continue

        X_g = grp_df[numeric_cols].to_numpy(dtype=np.float64)
        X_c = X_g - np.mean(X_g, axis=0, keepdims=True)
        X_ss = np.sum(X_c * X_c, axis=0)

        # 仅对标准差非零（非组内常数）的特征计算 r
        valid_mask = X_ss > 1e-12
        if not np.any(valid_mask):
            continue

        cov_xy = np.dot(y_c, X_c[:, valid_mask])
        denom = np.sqrt(y_ss * X_ss[valid_mask])
        r_vals = np.clip(cov_xy / denom, -1.0, 1.0)

        valid_indices = np.where(valid_mask)[0]
        for idx, r_val in zip(valid_indices, r_vals):
            feat_name = numeric_cols[idx]
            feat_group_rs[feat_name].append((str(grp_name), float(r_val)))

    # 汇总每个特征的最强正/负相关组与跨度
    stats_dict = {}
    for col in numeric_cols:
        pairs = feat_group_rs[col]
        if len(pairs) == 0:
            stats_dict[col] = {
                "r_max": 0.0,
                "group_max": "无有效变动组",
                "r_min": 0.0,
                "group_min": "无有效变动组",
                "r_span": 0.0,
                "n_valid_groups": 0,
            }
        else:
            max_pair = max(pairs, key=lambda item: item[1])
            min_pair = min(pairs, key=lambda item: item[1])
            stats_dict[col] = {
                "r_max": float(max_pair[1]),
                "group_max": max_pair[0],
                "r_min": float(min_pair[1]),
                "group_min": min_pair[0],
                "r_span": float(max_pair[1] - min_pair[1]),
                "n_valid_groups": len(pairs),
            }

    return stats_dict, n_qualifying, covered_rows


def check_simpson_triggers(r_global: float, r_max: float, r_min: float, n_valid: int) -> tuple:
    """
    判断单一分组维度下是否触发辛普森悖论警报：
    1. 正负双向撕裂：r_max > 0.20 且 r_min < -0.20（不同圈子关系完全相反！）
    2. 全局失明掩盖：|r_global| < 0.05 且 max(|r_max|, |r_min|) >= 0.30（全表假装没关系，局部强相关！）
    """
    if n_valid < 2:
        return False, False
    is_split = (r_max > 0.20) and (r_min < -0.20)
    is_masked = (abs(r_global) < 0.05) and (max(abs(r_max), abs(r_min)) >= 0.30)
    return is_split, is_masked


# ====================================================================
# 第三部分：双拼可视化图表（团战+洗牌检验 & 辛普森哑铃对比图）
# ====================================================================

def truncate_label(text: str, max_len: int = 16) -> str:
    """过长的机台或规格代号截断，防止图上文字溢出重叠。"""
    s = str(text)
    return s if len(s) <= max_len else s[: max_len - 2] + ".."


def plot_strip_b_summary(
    cv_summary: dict,
    scrambled_r2_pct: np.ndarray,
    perm_p_value: float,
    holdout_info: dict,
    simpson_df: pd.DataFrame,
    output_png_path: str,
) -> None:
    """
    绘制试纸 B 双拼可视化大图：
    - 左图：5,000 行 5 折交叉验证团战对比（Dummy / Ridge / HistGBDT）+ 19 次 Y-Scrambling 噪声分布
    - 右图：全量 53,145 行 Top 10 最撕裂辛普森特征哑铃图（灰点=全表相关，红/蓝点=最强正/负切片相关）
    """
    # 配置 Windows 下中文字体并确保负号正常显示（图内避免使用 Emoji 防方框乱码）
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(21, 9.5), dpi=180, gridspec_kw={"width_ratios": [1.0, 1.35]})
    fig.patch.set_facecolor("#FFFFFF")

    # ----------------------------------------------------------------
    # 左图：模型团战对比 + 19 次 Y-Scrambling 随机洗牌对比
    # ----------------------------------------------------------------
    ax1 = axes[0]
    model_labels = [
        "Dummy 基线\n(均值猜测)",
        "Ridge 线性\n(防泄露对照)",
        "HistGBDT 素颜树\n(真实目标)",
        "HistGBDT 洗牌\n(19次Y打乱)",
    ]
    x_pos = np.arange(len(model_labels))

    real_r2_pcts = [
        cv_summary["Dummy"]["r2_mean"] * 100.0,
        cv_summary["Ridge"]["r2_mean"] * 100.0,
        cv_summary["HistGBDT"]["r2_mean"] * 100.0,
    ]
    real_r2_stds = [
        cv_summary["Dummy"]["r2_std"] * 100.0,
        cv_summary["Ridge"]["r2_std"] * 100.0,
        cv_summary["HistGBDT"]["r2_std"] * 100.0,
    ]
    bar_colors = ["#ADB5BD", "#457B9D", "#E63946"]

    bars = ax1.bar(
        x_pos[:3],
        real_r2_pcts,
        yerr=real_r2_stds,
        capsize=5,
        width=0.52,
        color=bar_colors,
        edgecolor="#FFFFFF",
        linewidth=1.0,
        alpha=0.92,
        zorder=3,
    )

    # 在前三个条形上方/下方标注精确 R² 与 MAE
    model_keys = ["Dummy", "Ridge", "HistGBDT"]
    for idx, (bar, key) in enumerate(zip(bars, model_keys)):
        r2_val = cv_summary[key]["r2_mean"] * 100.0
        mae_val = cv_summary[key]["mae_mean"]
        y_err = real_r2_stds[idx]
        text_y = max(r2_val, 0.0) + y_err + 1.8
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            text_y,
            f"R² = {r2_val:.2f}%\nMAE = {mae_val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9.2,
            fontweight="bold" if key == "HistGBDT" else "normal",
            color="#9B1C1C" if key == "HistGBDT" else "#1F2937",
        )

    # 第 4 个位置绘制 19 次 Y-Scrambling 盒图 + 抖动散点
    bp = ax1.boxplot(
        [scrambled_r2_pct],
        positions=[x_pos[3]],
        widths=0.38,
        patch_artist=True,
        showfliers=False,
        zorder=3,
        boxprops=dict(facecolor="#FDE68A", edgecolor="#B45309", linewidth=1.3),
        medianprops=dict(color="#92400E", linewidth=2.0),
        whiskerprops=dict(color="#B45309", linewidth=1.2),
        capprops=dict(color="#B45309", linewidth=1.2),
    )
    rng_jitter = np.random.RandomState(42)
    jitter_x = x_pos[3] + rng_jitter.uniform(-0.11, 0.11, size=len(scrambled_r2_pct))
    ax1.scatter(
        jitter_x,
        scrambled_r2_pct,
        color="#D97706",
        edgecolor="#FFFFFF",
        s=36,
        alpha=0.88,
        zorder=4,
        label="19次打乱单次5折R²",
    )

    ax1.axhline(0.0, color="#374151", linestyle="--", linewidth=1.1, alpha=0.7, zorder=2)

    # 标注洗牌均值与 Phipson & Smyth p 值
    scram_med = float(np.median(scrambled_r2_pct))
    scram_max = float(np.max(scrambled_r2_pct))
    ax1.text(
        x_pos[3],
        min(scrambled_r2_pct) - 2.4,
        f"中位数 = {scram_med:.2f}%\n最大值 = {scram_max:.2f}%\np = {perm_p_value:.3f} (M=19)",
        ha="center",
        va="top",
        fontsize=8.8,
        fontweight="bold",
        color="#92400E",
    )

    # 左下角/右上角放置“影子卧底与留出集质检”总结卡片
    shadow_rank = holdout_info["shadow_rank"]
    shadow_imp = holdout_info["shadow_imp"]
    top1_name = holdout_info["top1_name"]
    top1_imp = holdout_info["top1_imp"]
    n_beat_shadow = holdout_info["n_beat_shadow"]
    val_r2_pct = holdout_info["val_r2"] * 100.0
    card_text = (
        f"[80/20 留出集与影子卧底质检]\n"
        f"• 留出集验证 R² = {val_r2_pct:.2f}% (MAE = {holdout_info['val_mae']:.2f})\n"
        f"• 影子噪声排名 #{shadow_rank}/76 (重要度 = {shadow_imp:.5f})\n"
        f"• 榜首特征 {top1_name} 重要度 = {top1_imp:.4f}\n"
        f"• 共 {n_beat_shadow}/75 个真特征重要度高于影子噪声"
    )
    ax1.text(
        0.03,
        0.96,
        card_text,
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=8.8,
        color="#1E3A8A",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#EFF6FF", edgecolor="#93C5FD", alpha=0.95),
    )

    y_min_ax1 = min(float(np.min(scrambled_r2_pct)) - 13.5, -18.0)
    y_max_ax1 = max(real_r2_pcts) + max(real_r2_stds) + 14.0
    ax1.set_ylim(y_min_ax1, y_max_ax1)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(model_labels, fontsize=9.8, fontweight="bold", color="#111827")
    ax1.set_ylabel("5 折交叉验证解释率 R² [%]", fontsize=10.8)
    ax1.set_title("团战模型大比拼 vs 19 次 Y-Scrambling 真实性检验", fontsize=12.8, fontweight="bold", pad=12)
    ax1.grid(axis="y", linestyle=":", alpha=0.45)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # ----------------------------------------------------------------
    # 右图：Top 10 最撕裂辛普森悖论特征哑铃对比图
    # ----------------------------------------------------------------
    ax2 = axes[1]
    top10 = simpson_df.head(10).iloc[::-1].reset_index(drop=True)  # 倒序使第 1 名在最上方
    y_idx = np.arange(len(top10))

    # 阴影标出 [-0.05, +0.05] 全局失明微弱区
    ax2.axvspan(-0.05, 0.05, color="#F3F4F6", alpha=0.9, zorder=1, label="全局微弱区 (|r|<0.05)")
    ax2.axvline(0.0, color="#6B7280", linestyle="--", linewidth=1.2, zorder=2)

    y_labels = []
    for i, row in top10.iterrows():
        r_glob = float(row["全局_r"])
        r_min = float(row["主导切片_r_min"])
        r_max = float(row["主导切片_r_max"])
        g_min = truncate_label(row["主导切片_最负组"], 14)
        g_max = truncate_label(row["主导切片_最正组"], 14)
        dim_short = "按机台" if "机台" in str(row["最大撕裂维度"]) else "按规格"
        span_val = float(row["最大撕裂跨度"])

        # 连接线（红蓝哑铃杆）
        ax2.plot([r_min, r_max], [i, i], color="#9CA3AF", linewidth=2.6, alpha=0.85, zorder=3)
        # 最负组（蓝点）、最正组（红点）、全表全局相关（灰菱形）
        ax2.scatter(r_min, i, color="#2563EB", s=78, edgecolor="#FFFFFF", linewidth=0.9, zorder=5)
        ax2.scatter(r_max, i, color="#DC2626", s=78, edgecolor="#FFFFFF", linewidth=0.9, zorder=5)
        ax2.scatter(r_glob, i, color="#4B5563", marker="D", s=48, edgecolor="#FFFFFF", linewidth=0.8, zorder=6)

        # 在蓝点左侧标出最负值及组名，红点右侧标出最正值及组名
        ax2.text(
            r_min - 0.025,
            i,
            f"{r_min:+.2f} ({g_min})",
            ha="right",
            va="center",
            fontsize=8.0,
            color="#1D4ED8",
            fontweight="bold",
        )
        ax2.text(
            r_max + 0.025,
            i,
            f"{r_max:+.2f} ({g_max})",
            ha="left",
            va="center",
            fontsize=8.0,
            color="#B91C1C",
            fontweight="bold",
        )

        # 左侧 Y 轴标签展示：特征名 + 切片维度 + 跨度
        alert_mark = "[警报] " if row["是否触发警报"] == "是" else ""
        y_labels.append(f"{alert_mark}{row['特征名称']}\n({dim_short} 跨度={span_val:.2f}, 全局r={r_glob:+.2f})")

    ax2.set_yticks(y_idx)
    ax2.set_yticklabels(y_labels, fontsize=8.4, color="#111827")
    ax2.set_xlim(-1.18, 1.18)
    ax2.set_xlabel("与目标 ActualVsIEGap 的 Pearson 相关系数 r", fontsize=10.8)
    ax2.set_title("全量 53,145 行辛普森悖论排查：Top 10 最撕裂特征哑铃图", fontsize=12.8, fontweight="bold", pad=12)
    ax2.grid(axis="x", linestyle=":", alpha=0.45)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    legend_items = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#2563EB", markersize=8.5, label="最强负相关切片组"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="#4B5563", markersize=7.5, label="全表全局 r_global"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#DC2626", markersize=8.5, label="最强正相关切片组"),
        Patch(facecolor="#F3F4F6", edgecolor="#D1D5DB", label="全局微弱区 (|r|<0.05)"),
    ]
    ax2.legend(handles=legend_items, loc="lower right", fontsize=8.5, frameon=True, facecolor="#FFFFFF", framealpha=0.95)

    n_alert = int((simpson_df["是否触发警报"] == "是").sum())
    fig.suptitle(
        f"试纸 B 体检总览：素颜树 5 折 R²={cv_summary['HistGBDT']['r2_mean']*100:.2f}% (碾压线性 {cv_summary['Ridge']['r2_mean']*100:.2f}% 与 19次洗牌 p={perm_p_value:.3f}) | 捕获 {n_alert} 个辛普森悖论警报特征",
        fontsize=13.8,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_png_path, dpi=180)
    plt.close(fig)


# ====================================================================
# 第四部分：主流程入口
# ====================================================================

def run_strip_b(
    data_path: str = "data/processed/clean_dataset.csv",
    schema_path: str = "data/processed/feature_schema.json",
    report_csv_path: str = "reports/step2b_strip_b_simpson_report.csv",
    summary_png_path: str = "reports/step2b_strip_b_summary.png",
    sample_size: int = 5000,
    n_scrambles: int = 19,
    min_group_samples: int = 200,
    random_state: int = 42,
) -> dict:
    """
    执行完整试纸 B 流程：
    1. 5,000 行 5 折零泄露模型团战对比（Dummy / Ridge / HistGBDT）
    2. 19 次 Y-Scrambling 置换检验 + 80/20 留出集影子卧底 Permutation Importance 排查
    3. 全量 53,145 行机台 & 规格辛普森悖论切片排查并导出 CSV 与 PNG
    """
    t_start = time.perf_counter()
    os.makedirs(os.path.dirname(report_csv_path), exist_ok=True)

    print("=" * 74)
    print("🧪 启动【试纸 B】素颜树团战探针 + Y-Scrambling + 辛普森悖论排查 (Step 2B)")
    print("=" * 74)

    # 1. 读取数据与元数据（强制把 26 个无序文本列按字符串读入，严禁 pandas 自动转数字）
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    target_col = schema["target_col"]
    numeric_cols = list(schema["numeric_features"])
    categorical_cols = list(schema["categorical_features"])

    dtype_map = {col: str for col in categorical_cols}
    df = pd.read_csv(data_path, dtype=dtype_map, keep_default_na=False)
    print(f"📂 已加载洁净数据集：{len(df):,} 行 × {df.shape[1]} 列 | 目标列：{target_col}")

    # 区分超高基数文本列（>250 类）与低/中基数文本列（<=250 类）
    high_cat_cols = [c for c in categorical_cols if df[c].nunique() > 250]
    low_cat_cols = [c for c in categorical_cols if c not in high_cat_cols]
    print(f"🏷️ 文本特征分流：{len(high_cat_cols)} 个超高基数(>250)走 TargetEncoder | {len(low_cat_cols)} 个低/中基数走原生类别切分")
    print(f"   超高基数列：{high_cat_cols}")

    # ----------------------------------------------------------------
    # 模块 1：5,000 行抽样 + 影子卧底注入 + 5 折零泄露团战对比
    # ----------------------------------------------------------------
    t_m1 = time.perf_counter()
    df_sample = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
    rng = np.random.RandomState(random_state)
    shadow_col = "__shadow_random_noise__"
    df_sample[shadow_col] = rng.normal(loc=0.0, scale=1.0, size=len(df_sample))

    num_cols_with_shadow = numeric_cols + [shadow_col]
    feature_cols = high_cat_cols + low_cat_cols + num_cols_with_shadow
    X_sample = df_sample[feature_cols]
    y_sample = df_sample[target_col].to_numpy(dtype=np.float64)

    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    scoring = {"r2": "r2", "neg_mae": "neg_mean_absolute_error"}

    models = {
        "Dummy": DummyRegressor(strategy="mean"),
        "Ridge": build_ridge_pipeline(high_cat_cols, low_cat_cols, num_cols_with_shadow, random_state=random_state),
        "HistGBDT": build_hgb_pipeline(high_cat_cols, low_cat_cols, num_cols_with_shadow, random_state=random_state),
    }

    cv_summary = {}
    print("\n" + "-" * 74)
    print(f"⚔️ 【模块 1】5,000 行 × 5 折零泄露团战比拼（共 {len(feature_cols)} 特征含 1 个影子噪声）")
    for name, est in models.items():
        t_model = time.perf_counter()
        res = cross_validate(est, X_sample, y_sample, cv=cv, scoring=scoring, n_jobs=5)
        r2_arr = res["test_r2"]
        mae_arr = -res["test_neg_mae"]
        cv_summary[name] = {
            "r2_mean": float(np.mean(r2_arr)),
            "r2_std": float(np.std(r2_arr)),
            "mae_mean": float(np.mean(mae_arr)),
            "mae_std": float(np.std(mae_arr)),
            "elapsed": time.perf_counter() - t_model,
        }
        print(
            f"   • {name:10s} | 5折 R² = {cv_summary[name]['r2_mean']*100:6.2f}% ± {cv_summary[name]['r2_std']*100:4.2f}% "
            f"| MAE = {cv_summary[name]['mae_mean']:6.2f} ± {cv_summary[name]['mae_std']:4.2f} "
            f"| 耗时 {cv_summary[name]['elapsed']:.2f}s"
        )
    print(f"⏱️ 模块 1 完成，累计耗时 {time.perf_counter() - t_m1:.2f}s")

    # ----------------------------------------------------------------
    # 模块 2A：19 次 Y-Scrambling 随机洗牌置换检验
    # ----------------------------------------------------------------
    t_m2 = time.perf_counter()
    print("\n" + "-" * 74)
    print(f"🎲 【模块 2A】执行 M = {n_scrambles} 次 Y-Scrambling 真实性检验（每次重跑完整 5 折 HistGBDT）...")
    real_hgb_r2 = cv_summary["HistGBDT"]["r2_mean"]
    scrambled_r2_list = []
    rng_scram = np.random.RandomState(random_state)

    for m in range(1, n_scrambles + 1):
        y_perm = rng_scram.permutation(y_sample)
        hgb_perm = build_hgb_pipeline(high_cat_cols, low_cat_cols, num_cols_with_shadow, random_state=random_state)
        res_perm = cross_validate(hgb_perm, X_sample, y_perm, cv=cv, scoring="r2", n_jobs=5)
        r2_perm_mean = float(np.mean(res_perm["test_score"]))
        scrambled_r2_list.append(r2_perm_mean)

    scrambled_r2_arr = np.array(scrambled_r2_list, dtype=np.float64)
    # Phipson & Smyth (2010) 精确置换 p 值公式：p = (C + 1) / (M + 1)
    c_exceed = int(np.sum(scrambled_r2_arr >= real_hgb_r2))
    perm_p_value = float((c_exceed + 1) / (n_scrambles + 1))

    print(
        f"   • 19 次洗牌 5折 R² 分布：最小={np.min(scrambled_r2_arr)*100:.2f}% | "
        f"中位={np.median(scrambled_r2_arr)*100:.2f}% | 最大={np.max(scrambled_r2_arr)*100:.2f}%"
    )
    print(f"   • 真实 HistGBDT R² = {real_hgb_r2*100:.2f}% | 超越真实分数的洗牌次数 C = {c_exceed}/{n_scrambles}")
    print(f"   • Phipson & Smyth 置换检验 p = ({c_exceed}+1)/({n_scrambles}+1) = {perm_p_value:.4f}")
    print(f"⏱️ 模块 2A（19 次洗牌）耗时 {time.perf_counter() - t_m2:.2f}s")

    # ----------------------------------------------------------------
    # 模块 2B：80/20 留出验证集 + Permutation Importance + 影子卧底核验
    # ----------------------------------------------------------------
    t_m2b = time.perf_counter()
    print("\n" + "-" * 74)
    print("🕵️ 【模块 2B】80/20 留出验证集特征排位与影子卧底 (__shadow_random_noise__) 检验...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_sample, y_sample, test_size=0.20, random_state=random_state
    )
    hgb_holdout = build_hgb_pipeline(high_cat_cols, low_cat_cols, num_cols_with_shadow, random_state=random_state)
    hgb_holdout.fit(X_tr, y_tr)
    y_val_pred = hgb_holdout.predict(X_val)
    val_r2 = float(r2_score(y_val, y_val_pred))
    val_mae = float(mean_absolute_error(y_val, y_val_pred))

    pi_res = permutation_importance(
        hgb_holdout,
        X_val,
        y_val,
        n_repeats=5,
        random_state=random_state,
        scoring="r2",
        n_jobs=5,
    )
    pi_df = pd.DataFrame({
        "feature_name": feature_cols,
        "importance_mean": pi_res.importances_mean,
        "importance_std": pi_res.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    pi_df["rank"] = np.arange(1, len(pi_df) + 1)

    shadow_row_pi = pi_df[pi_df["feature_name"] == shadow_col].iloc[0]
    shadow_rank = int(shadow_row_pi["rank"])
    shadow_imp = float(shadow_row_pi["importance_mean"])
    real_pi_df = pi_df[pi_df["feature_name"] != shadow_col]
    n_beat_shadow = int((real_pi_df["importance_mean"] > shadow_imp).sum())

    print(f"   • 留出验证集 (n=1,000) 表现：R² = {val_r2*100:.2f}% | MAE = {val_mae:.2f}")
    print(f"   • 影子卧底排名：#{shadow_rank}/{len(pi_df)} | 重要度 = {shadow_imp:+.6f} (±{shadow_row_pi['importance_std']:.6f})")
    print(f"   • 共有 {n_beat_shadow}/75 个真实特征排在影子卧底之上")
    print("   🏆 Top 10 团战贡献核心特征（Permutation Importance on 留出集）：")
    for _, row in real_pi_df.head(10).iterrows():
        print(
            f"      #{int(row['rank']):2d} | {row['feature_name']:52s} "
            f"| ΔR² = {row['importance_mean']:+.5f} (±{row['importance_std']:.5f})"
        )
    print(f"⏱️ 模块 2B 耗时 {time.perf_counter() - t_m2b:.2f}s")

    holdout_info = {
        "val_r2": val_r2,
        "val_mae": val_mae,
        "shadow_rank": shadow_rank,
        "shadow_imp": shadow_imp,
        "top1_name": str(real_pi_df.iloc[0]["feature_name"]),
        "top1_imp": float(real_pi_df.iloc[0]["importance_mean"]),
        "n_beat_shadow": n_beat_shadow,
    }

    # ----------------------------------------------------------------
    # 模块 3：全量 53,145 行辛普森悖论分家排查（按机台 & 按规格）
    # ----------------------------------------------------------------
    t_m3 = time.perf_counter()
    print("\n" + "-" * 74)
    print(f"🔍 【模块 3】全量 {len(df):,} 行辛普森悖论分家排查（仅保留样本量 >= {min_group_samples} 的主力组）...")

    y_full = df[target_col].to_numpy(dtype=np.float64)
    global_r_map = {}
    for col in numeric_cols:
        r_g = safe_pearson_r(df[col].to_numpy(dtype=np.float64), y_full)
        global_r_map[col] = 0.0 if np.isnan(r_g) else r_g

    wc_stats, n_wc_groups, wc_covered = compute_slice_correlations(
        df=df,
        group_col="WorkCenterNameCoupled",
        numeric_cols=numeric_cols,
        target_col=target_col,
        min_samples=min_group_samples,
    )
    ts_stats, n_ts_groups, ts_covered = compute_slice_correlations(
        df=df,
        group_col="tire_size",
        numeric_cols=numeric_cols,
        target_col=target_col,
        min_samples=min_group_samples,
    )
    print(f"   • 主力机台分组 (WorkCenterNameCoupled >= {min_group_samples})：{n_wc_groups} 个机台，覆盖 {wc_covered:,} 行")
    print(f"   • 主力规格分组 (tire_size >= {min_group_samples})：{n_ts_groups} 个规格，覆盖 {ts_covered:,} 行")

    simpson_rows = []
    for col in numeric_cols:
        r_glob = global_r_map[col]
        w = wc_stats[col]
        t = ts_stats[col]

        wc_split, wc_masked = check_simpson_triggers(r_glob, w["r_max"], w["r_min"], w["n_valid_groups"])
        ts_split, ts_masked = check_simpson_triggers(r_glob, t["r_max"], t["r_min"], t["n_valid_groups"])

        is_alert = wc_split or wc_masked or ts_split or ts_masked

        # 汇总清晰的警报原因标签（零 NaN）
        reasons = []
        if wc_split or ts_split:
            dims = []
            if wc_split:
                dims.append("机台")
            if ts_split:
                dims.append("规格")
            reasons.append(f"正负双向撕裂({'+'.join(dims)})")
        if wc_masked or ts_masked:
            dims = []
            if wc_masked:
                dims.append("机台")
            if ts_masked:
                dims.append("规格")
            reasons.append(f"全局失明掩盖({'+'.join(dims)})")
        reason_str = " & ".join(reasons) if reasons else "正常"

        # 选取跨度更大的切片维度作为“主导撕裂维度”（用于排序与哑铃图展示）
        if w["r_span"] >= t["r_span"]:
            dom_dim = "机台 (WorkCenter)"
            dom_r_max, dom_g_max = w["r_max"], w["group_max"]
            dom_r_min, dom_g_min = w["r_min"], w["group_min"]
            max_span = w["r_span"]
        else:
            dom_dim = "规格 (tire_size)"
            dom_r_max, dom_g_max = t["r_max"], t["group_max"]
            dom_r_min, dom_g_min = t["r_min"], t["group_min"]
            max_span = t["r_span"]

        simpson_rows.append({
            "特征名称": col,
            "全局_r": round(r_glob, 6),
            "最大撕裂跨度": round(max_span, 6),
            "最大撕裂维度": dom_dim,
            "是否触发警报": "是" if is_alert else "否",
            "警报类型": reason_str,
            "主导切片_r_max": round(dom_r_max, 6),
            "主导切片_最正组": dom_g_max,
            "主导切片_r_min": round(dom_r_min, 6),
            "主导切片_最负组": dom_g_min,
            "机台切片_r_max": round(w["r_max"], 6),
            "机台切片_最正机台": w["group_max"],
            "机台切片_r_min": round(w["r_min"], 6),
            "机台切片_最负机台": w["group_min"],
            "机台切片_r_span": round(w["r_span"], 6),
            "机台有效组数": int(w["n_valid_groups"]),
            "规格切片_r_max": round(t["r_max"], 6),
            "规格切片_最正规格": t["group_max"],
            "规格切片_r_min": round(t["r_min"], 6),
            "规格切片_最负规格": t["group_min"],
            "规格切片_r_span": round(t["r_span"], 6),
            "规格有效组数": int(t["n_valid_groups"]),
        })

    simpson_df = pd.DataFrame(simpson_rows).sort_values("最大撕裂跨度", ascending=False).reset_index(drop=True)
    simpson_df.insert(0, "撕裂排名", np.arange(1, len(simpson_df) + 1))

    # 强制确认零 NaN 后导出 UTF-8 BOM CSV
    assert simpson_df.isna().sum().sum() == 0, "辛普森报告中存在意外 NaN！"
    simpson_df.to_csv(report_csv_path, index=False, encoding="utf-8-sig")

    alert_df = simpson_df[simpson_df["是否触发警报"] == "是"]
    print(f"   • 共捕获 {len(alert_df)}/49 个触发【辛普森悖论警报】的特征！")
    print("   🚨 Top 10 最撕裂特征榜单：")
    for _, r in simpson_df.head(10).iterrows():
        print(
            f"      #{int(r['撕裂排名']):2d} | {r['特征名称']:38s} | 全局r={r['全局_r']:+.3f} "
            f"| 跨度={r['最大撕裂跨度']:.3f} [{r['最大撕裂维度']}: {r['主导切片_r_min']:+.3f}({r['主导切片_最负组']}) ~ "
            f"{r['主导切片_r_max']:+.3f}({r['主导切片_最正组']})] | {r['警报类型']}"
        )
    print(f"⏱️ 模块 3 耗时 {time.perf_counter() - t_m3:.2f}s")

    # ----------------------------------------------------------------
    # 导出双拼可视化大图
    # ----------------------------------------------------------------
    plot_strip_b_summary(
        cv_summary=cv_summary,
        scrambled_r2_pct=scrambled_r2_arr * 100.0,
        perm_p_value=perm_p_value,
        holdout_info=holdout_info,
        simpson_df=simpson_df,
        output_png_path=summary_png_path,
    )

    total_time = time.perf_counter() - t_start
    print("\n" + "=" * 74)
    print(f"✅ 试纸 B 全部完成！总耗时：{total_time:.2f} 秒")
    print(f"📄 辛普森报表：{report_csv_path}（{len(simpson_df)} 行 × {simpson_df.shape[1]} 列，0 NaN）")
    print(f"🖼️ 综合汇总图：{summary_png_path}")
    print("=" * 74)

    return {
        "cv_summary": cv_summary,
        "scrambled_r2": scrambled_r2_arr.tolist(),
        "perm_p_value": perm_p_value,
        "holdout_info": holdout_info,
        "pi_df": pi_df,
        "simpson_df": simpson_df,
        "total_time": total_time,
    }


if __name__ == "__main__":
    run_strip_b()
