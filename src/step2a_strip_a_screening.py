# -*- coding: utf-8 -*-
"""
====================================================================
第二步（试纸 A）：单兵尖子扫雷与双轨公平体检 (Step 2A: Strip A Screening)
====================================================================
【核心任务】
在进入复杂机器学习建模（试纸 B 及后续建模）之前，先用最快、最透明、零编码偏误的方式，
给清洗后的 75 个真实特征（+ 1 个随机噪声卧底特征）做一次“单兵能力公平大体检”。

【双轨公平法则（严禁乱编数字）】
1. 轨道 1（49 个物理数值特征 + 1 个随机影子哨兵）：走“数字对数字”尺子
   - Pearson r & 线性 R²（抓直线关系）
   - Spearman rho（抓单调递增/递减关系，抗极端值）
   - 15 分箱非线性 R²_binned + 单因素 F 检验（把数值切成最多 15 段，专抓 U 型/折线/台阶！）
   - 抽样 1D 距离相关系数 dCor（n=2000，抓均值、方差及任意形状依赖）

2. 轨道 2（26 个无序文本特征）：走“文本对数字免编码”尺子（绝对不做 1,2,3 编码！）
   - 相关比 Eta²（组间方差解释率，等价于 One-Hot 回归 R²）
   - 无偏 Omega²（扣除类别数 k 的自由度水分，与数值轨道 R² 公平同台比较！）
   - 单因素 ANOVA F 检验 p 值
   - Kruskal-Wallis 秩检验 η²_H 与 p 值（抗异常生产批次扰动）
   - 抽样 0-1 离散度量 dCor（n=2000，仅判断“是否同一类别”，100% 满足标签置换不变性）

3. 影子哨兵 + BH 多重检验校正（防靠运气蒙对）：
   - 注入 1 列标准正态随机噪声 __shadow_random_noise__ 作为“纯混子基准线”。
   - 对全部 76 个条目统一执行 Benjamini-Hochberg (BH) FDR 校正，得到 q_value_bh。

4. 统一尺度与红黄绿灯评级：
   - 统一代表性有效解释率：数值列取 max(R²_linear, R²_binned)；文本列取无偏 ω²。
   - 输出体检明细表 reports/step2a_strip_a_report.csv（UTF-8 BOM，Excel 直接打开不乱码）
   - 输出直观双拼对比图 reports/step2a_strip_a_summary.png
"""

import os
import json
import time
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


# ====================================================================
# 第一部分：基础工具函数（距离相关 dCor 与单因素分组方差分解）
# ====================================================================

def compute_double_centered_matrix(dist_mat: np.ndarray) -> np.ndarray:
    """
    对 n x n 的距离矩阵执行“双重中心化”（Double Centering）。
    大白话解释：把每一行均值、每一列均值减掉，再加上总均值，
    让距离矩阵处于“零中心”状态，方便后续像算协方差一样算距离相关性。
    """
    row_mean = dist_mat.mean(axis=1, keepdims=True)
    col_mean = dist_mat.mean(axis=0, keepdims=True)
    grand_mean = dist_mat.mean()
    return dist_mat - row_mean - col_mean + grand_mean


def compute_dcor_from_centered(A: np.ndarray, B: np.ndarray, dvar_y_sq: float) -> float:
    """
    根据特征的双重中心化矩阵 A 和预计算好的目标矩阵 B，计算精确样本距离相关系数 dCor。
    公式：dCor(X, Y) = sqrt( dCov²(X, Y) / sqrt(dVar²(X) * dVar²(Y)) )
    返回值范围严格落在 [0.0, 1.0]。
    """
    dvar_x_sq = float(np.mean(A * A))
    if dvar_x_sq <= 1e-15 or dvar_y_sq <= 1e-15:
        return 0.0

    dcov_sq = float(np.mean(A * B))
    if dcov_sq <= 0.0:
        return 0.0

    denom = np.sqrt(dvar_x_sq * dvar_y_sq)
    if denom <= 1e-15:
        return 0.0

    dcor_sq = np.clip(dcov_sq / denom, 0.0, 1.0)
    return float(np.sqrt(dcor_sq))


def compute_1d_numeric_dcor(x_sample: np.ndarray, B_y: np.ndarray, dvar_y_sq: float) -> float:
    """
    计算数值特征在抽样样本上的精确 1D 距离相关系数 dCor（向量化加速版）。
    """
    if np.std(x_sample) <= 1e-15:
        return 0.0
    dist_x = np.abs(x_sample[:, None] - x_sample[None, :])
    A = compute_double_centered_matrix(dist_x)
    return compute_dcor_from_centered(A, B_y, dvar_y_sq)


def compute_categorical_01_dcor(cat_sample: np.ndarray, B_y: np.ndarray, dvar_y_sq: float) -> float:
    """
    计算无序文本特征在抽样样本上的 0-1 离散距离相关系数 dCor。
    距离定义：d_X(u, v) = 1 (若 u != v)，否则 0 (若 u == v)。
    注意：这里只判断两个样本的字符串“是不是同一个名字”，完全不涉及大小或顺序，
    哪怕把所有代号随机换个名字，算出来的结果也 100% 一模一样（标签置换不变性）。
    """
    # 为了加速矩阵比对，先临时映射为内部相等性编号（仅用于 != 判断，绝不参与大小运算）
    codes, _ = pd.factorize(cat_sample, sort=False)
    if len(np.unique(codes)) <= 1:
        return 0.0
    dist_x = (codes[:, None] != codes[None, :]).astype(np.float64)
    A = compute_double_centered_matrix(dist_x)
    return compute_dcor_from_centered(A, B_y, dvar_y_sq)


def safe_clip_p(p_val: float) -> float:
    """
    安全处理 p 值：遇到 NaN 回退为 1.0，其余裁剪至 [1e-300, 1.0] 防止后续校正报错。
    """
    if p_val is None or np.isnan(p_val):
        return 1.0
    return float(np.clip(p_val, 1e-300, 1.0))


# ====================================================================
# 第二部分：双轨体检核心计算函数
# ====================================================================

def evaluate_numeric_feature(
    x: np.ndarray,
    y: np.ndarray,
    y_mean: float,
    ss_total: float,
    x_sample: np.ndarray,
    B_y: np.ndarray,
    dvar_y_sq: float,
    n_bins: int = 15,
) -> dict:
    """
    【轨道 1】对单个数值特征（或随机影子哨兵）执行 4 大核心指标评估：
    1. Pearson r & 线性 R²_linear & p_linear
    2. Spearman rho & p_spearman
    3. 15 等频分箱非线性 R²_binned & F 检验 p_binned
    4. 抽样 1D dCor
    """
    n = len(y)

    # 极端保护：若特征退化为常数，直接安全返回 0 信号
    if np.std(x) <= 1e-15 or ss_total <= 1e-15:
        return {
            "pearson_r": 0.0,
            "r2_linear": 0.0,
            "p_linear": 1.0,
            "spearman_rho": 0.0,
            "p_spearman": 1.0,
            "r2_binned": 0.0,
            "p_binned": 1.0,
            "n_bins_actual": 1,
            "dcor": 0.0,
        }

    # 1. Pearson 线性相关
    r_val, p_lin = stats.pearsonr(x, y)
    r_val = 0.0 if np.isnan(r_val) else float(r_val)
    p_lin = safe_clip_p(p_lin)
    r2_lin = float(r_val ** 2)

    # 2. Spearman 单调秩相关
    rho_val, p_sp = stats.spearmanr(x, y)
    rho_val = 0.0 if np.isnan(rho_val) else float(rho_val)
    p_sp = safe_clip_p(p_sp)

    # 3. 15 等频分箱非线性 R²_binned（专抓 U 型、抛物线、分段阶梯关系）
    try:
        bins = pd.qcut(x, q=n_bins, duplicates="drop")
        # 提取每个分箱的样本数和目标均值
        grp_df = pd.DataFrame({"bin": bins, "y": y}).groupby("bin", observed=True)["y"].agg(["count", "mean"])
        k_bins = len(grp_df)
    except Exception:
        k_bins = 1
        grp_df = None

    if k_bins <= 1 or grp_df is None:
        r2_binned = 0.0
        p_binned = 1.0
        k_bins = 1
    else:
        counts = grp_df["count"].to_numpy(dtype=np.float64)
        means = grp_df["mean"].to_numpy(dtype=np.float64)
        ss_between = float(np.sum(counts * (means - y_mean) ** 2))
        ss_within = max(0.0, ss_total - ss_between)
        r2_binned = float(np.clip(ss_between / ss_total, 0.0, 1.0))

        df1 = k_bins - 1
        df2 = n - k_bins
        if df1 > 0 and df2 > 0:
            if ss_within <= 1e-15:
                p_binned = 1e-300 if ss_between > 1e-15 else 1.0
            else:
                f_stat = (ss_between / df1) / (ss_within / df2)
                p_binned = safe_clip_p(stats.f.sf(f_stat, df1, df2))
        else:
            p_binned = 1.0

    # 4. 抽样 1D 距离相关 dCor
    dcor_val = compute_1d_numeric_dcor(x_sample, B_y, dvar_y_sq)

    return {
        "pearson_r": r_val,
        "r2_linear": r2_lin,
        "p_linear": p_lin,
        "spearman_rho": rho_val,
        "p_spearman": p_sp,
        "r2_binned": r2_binned,
        "p_binned": p_binned,
        "n_bins_actual": int(k_bins),
        "dcor": dcor_val,
    }


def evaluate_categorical_feature(
    cat_series: pd.Series,
    y: np.ndarray,
    y_mean: float,
    ss_total: float,
    cat_sample: np.ndarray,
    B_y: np.ndarray,
    dvar_y_sq: float,
) -> dict:
    """
    【轨道 2】对单个无序文本特征执行免编码分组评估：
    1. 相关比 Eta²（等价于 One-Hot 解释率，含自由度水分）
    2. 无偏校正 Omega²（挤掉高基数水分后，与数值轨道 R² 公平同台比较）
    3. 单因素 ANOVA F 检验 p_anova
    4. Kruskal-Wallis 秩检验 η²_H 与 p_kw（抗极端批）
    5. 抽样 0-1 离散度量 dCor
    """
    n = len(y)
    # 按原始字符串分组，严禁任何数字编码！
    temp_df = pd.DataFrame({"cat": cat_series.values, "y": y})
    grouped = temp_df.groupby("cat", sort=False)["y"]

    # 提取各有效非空分组数组，用于 scipy.stats.f_oneway 与 kruskal
    groups = [g.to_numpy(dtype=np.float64) for _, g in grouped if len(g) > 0]
    k = len(groups)

    if k <= 1 or n <= k or ss_total <= 1e-15:
        return {
            "k_categories": int(k),
            "eta2": 0.0,
            "omega2": 0.0,
            "p_anova": 1.0,
            "kw_eta2_h": 0.0,
            "p_kw": 1.0,
            "dcor": 0.0,
        }

    # 计算组间平方和 SS_between 与组内均方 MS_within
    counts = np.array([len(g) for g in groups], dtype=np.float64)
    means = np.array([np.mean(g) for g in groups], dtype=np.float64)
    ss_between = float(np.sum(counts * (means - y_mean) ** 2))
    ss_within = max(0.0, ss_total - ss_between)
    ms_within = ss_within / (n - k)

    # 相关比 Eta² 与无偏 Omega²
    eta2 = float(np.clip(ss_between / ss_total, 0.0, 1.0))
    omega2_num = ss_between - (k - 1) * ms_within
    omega2_den = ss_total + ms_within
    omega2 = float(np.clip(max(0.0, omega2_num / omega2_den) if omega2_den > 1e-15 else 0.0, 0.0, 1.0))

    # 单因素 ANOVA F 检验
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, p_anova_raw = stats.f_oneway(*groups)
    p_anova = safe_clip_p(p_anova_raw)

    # Kruskal-Wallis 非参数秩检验及效应量 η²_H = max(0, (H - k + 1) / (N - k))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        h_stat, p_kw_raw = stats.kruskal(*groups)
    h_stat = 0.0 if np.isnan(h_stat) else float(h_stat)
    kw_eta2_h = float(np.clip(max(0.0, (h_stat - k + 1.0) / (n - k)), 0.0, 1.0))
    p_kw = safe_clip_p(p_kw_raw)

    # 抽样 0-1 离散距离相关 dCor
    dcor_val = compute_categorical_01_dcor(cat_sample, B_y, dvar_y_sq)

    return {
        "k_categories": int(k),
        "eta2": eta2,
        "omega2": omega2,
        "p_anova": p_anova,
        "kw_eta2_h": kw_eta2_h,
        "p_kw": p_kw,
        "dcor": dcor_val,
    }


# ====================================================================
# 第三部分：红黄绿灯评级与可视化出图
# ====================================================================

def assign_traffic_light(
    row: dict,
    shadow_dcor: float,
) -> tuple:
    """
    严格按照批准的《试纸 A 实施计划》执行四级红黄绿灯评定与预警标注：
    - 🟢 强信号尖子生：q < 0.05 且 (有效解释率 >= 0.10 或 dCor >= 0.25)
    - 🟡 明确有效 / 隐藏非线性：q < 0.05 且 (0.02 <= 有效解释率 < 0.10 或 dCor >= 0.12)
    - ⚪ 微弱辅助信号：q < 0.05 且 0.003 <= 有效解释率 < 0.02
    - 🔴 无显著信号 / 建议弃用：q >= 0.05，或 (有效解释率 < 0.003 且 dCor 未显著超越基准)
    """
    q_val = row["q_value_bh"]
    eff = row["effective_r2"]
    dcor = row["dcor"]
    track = row["track"]
    feat = row["feature_name"]

    if q_val >= 0.05:
        rating = "🔴 无显著信号 / 建议弃用"
    elif eff >= 0.10 or dcor >= 0.25:
        rating = "🟢 强信号尖子生"
    elif eff >= 0.02 or dcor >= 0.12:
        rating = "🟡 明确有效 / 隐藏非线性"
    elif eff >= 0.003:
        rating = "⚪ 微弱辅助信号"
    else:
        # 有效解释率 < 0.003 且 dCor < 0.12：仅当 dCor 远超影子基准且接近门限时才给微弱信号，否则淘汰
        if dcor > max(0.08, shadow_dcor * 2.5):
            rating = "⚪ 微弱辅助信号"
        else:
            rating = "🔴 无显著信号 / 建议弃用"

    # 生成清晰直白的备注提示
    notes = []
    if feat == "__shadow_random_noise__":
        notes.append("[随机数卧底基准]")

    if track == "numeric" and feat != "__shadow_random_noise__":
        r2_lin = row["r2_linear_num"]
        if r2_lin < 0.01 and rating in ("🟢 强信号尖子生", "🟡 明确有效 / 隐藏非线性"):
            notes.append("[线性误杀预警]")

    if track == "categorical":
        inflation = row["eta2_num"] - row["omega2_num"]
        if inflation >= 0.02:
            notes.append(f"[高基数水分已挤出 -{inflation * 100:.1f}%]")

    note_str = " ".join(notes) if notes else "正常"
    return rating, note_str


def plot_strip_a_summary(
    report_df: pd.DataFrame,
    numeric_df: pd.DataFrame,
    shadow_eff: float,
    output_png_path: str,
) -> None:
    """
    绘制高颜值、大白话中文标注的试纸 A 双拼体检图：
    - 左图：Top 15 特征有效解释率横向对比（蓝=无序文本 ω²，橙=数值特征 max(R²)）+ 影子随机线
    - 右图：49 个数值特征 传统线性 R² vs 15分箱非线性 R²_binned，高亮[线性误杀预警]黑马
    """
    # 解决 Windows 下 Matplotlib 中文及负号显示问题
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(19, 8.8), dpi=180)
    fig.patch.set_facecolor("#FFFFFF")

    # ------------------------------------------------------------
    # 左图：Top 15 特征横向条形图（剔除影子列本身，影子列用竖虚线标基准）
    # ------------------------------------------------------------
    ax1 = axes[0]
    real_df = report_df[report_df["特征名称"] != "__shadow_random_noise__"].copy()
    top15 = real_df.head(15).iloc[::-1]  # 倒序使第 1 名在最上方

    color_map = {
        "无序文本特征": "#2A7B9B",
        "数值特征": "#E07A5F",
    }
    colors = [color_map.get(t, "#888888") for t in top15["特征类型"]]

    y_pos = np.arange(len(top15))
    bars = ax1.barh(
        y_pos,
        top15["有效解释率"].values * 100.0,
        color=colors,
        height=0.68,
        edgecolor="white",
        linewidth=0.8,
        alpha=0.92,
    )

    # 影子基准虚线
    shadow_pct = shadow_eff * 100.0
    ax1.axvline(
        x=shadow_pct,
        color="#D62828",
        linestyle="--",
        linewidth=1.8,
        label=f"影子随机噪声基准 ({shadow_pct:.3f}%)",
    )

    # 条形末端标注具体数值与预警标签
    for bar, (_, row) in zip(bars, top15.iterrows()):
        val_pct = row["有效解释率"] * 100.0
        tag = " ★线性误杀黑马" if "[线性误杀预警]" in str(row["备注提示"]) else ""
        ax1.text(
            val_pct + 0.45,
            bar.get_y() + bar.get_height() / 2.0,
            f"{val_pct:.2f}%{tag}",
            va="center",
            ha="left",
            fontsize=8.8,
            fontweight="bold" if tag else "normal",
            color="#B22222" if tag else "#222222",
        )

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(top15["特征名称"].values, fontsize=9.5, color="#222222")
    ax1.set_xlabel("有效方差解释率 (数值取 max(R²_linear, R²_binned)，文本取无偏 ω²) [%]", fontsize=10.5)
    ax1.set_title("Top 15 单兵尖子生排行榜（双轨公平同台）", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlim(0, max(top15["有效解释率"].max() * 100.0 * 1.22, 10.0))
    ax1.grid(axis="x", linestyle=":", alpha=0.4)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # 自定义图例
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#2A7B9B", label="无序文本特征 (免编码无偏 ω²)"),
        Patch(facecolor="#E07A5F", label="数值特征 (max 线性/分箱 R²)"),
        plt.Line2D([0], [0], color="#D62828", linestyle="--", linewidth=1.8, label=f"影子噪声基准 ({shadow_pct:.3f}%)"),
    ]
    ax1.legend(handles=legend_handles, loc="lower right", fontsize=9, frameon=True, facecolor="#F9FAFB")

    # ------------------------------------------------------------
    # 右图：49 个数值特征 线性 R² vs 15分箱非线性 R²_binned 对比散点图
    # ------------------------------------------------------------
    ax2 = axes[1]
    x_lin = numeric_df["r2_linear_num"].values * 100.0
    y_bin = numeric_df["r2_binned_num"].values * 100.0
    is_ambush = numeric_df["is_linear_ambush"].values

    max_lim = max(float(np.max(x_lin)), float(np.max(y_bin)), 5.0) * 1.12

    # 画 y = x 对角线（在上方说明分箱捕获到了线性没抓到的非线性信号）
    ax2.plot([0, max_lim], [0, max_lim], linestyle="--", color="#6C757D", linewidth=1.3, label="y = x (纯线性参照线)")
    # 画 x = 1% 线性误杀警戒线
    ax2.axvline(x=1.0, linestyle=":", color="#D62828", linewidth=1.4, label="线性 R² = 1% 误杀警戒线")

    # 普通数值点
    ax2.scatter(
        x_lin[~is_ambush],
        y_bin[~is_ambush],
        c="#457B9D",
        s=55,
        alpha=0.75,
        edgecolors="white",
        linewidth=0.7,
        label="常规数值特征",
        zorder=3,
    )

    # 线性误杀黑马点
    if np.any(is_ambush):
        ax2.scatter(
            x_lin[is_ambush],
            y_bin[is_ambush],
            c="#E63946",
            marker="*",
            s=220,
            edgecolors="#330000",
            linewidth=0.8,
            label="线性误杀黑马 (线性<1% 但非线性达标)",
            zorder=5,
        )

    # 标注关键点位名称：仅标注最具代表性的 Top 5 线性误杀黑马 + Top 3 数值尖子，避免密集文字堆叠重叠
    top_num_names = list(
        numeric_df.sort_values("effective_r2", ascending=False).head(3)["feature_name"]
    )
    top_ambush_names = list(
        numeric_df[is_ambush].sort_values("r2_binned_num", ascending=False).head(5)["feature_name"]
    )

    # 分别以错落有致的坐标偏移标注，确保零文字遮挡
    for feat_name in top_num_names:
        r = numeric_df[numeric_df["feature_name"] == feat_name].iloc[0]
        px = r["r2_linear_num"] * 100.0
        py = r["r2_binned_num"] * 100.0
        dx, dy = (-75, -14) if px > 12 else (10, 4)
        ax2.annotate(
            f"{feat_name} ({py:.1f}%)",
            xy=(px, py),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.8,
            color="#1D3557",
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#6C757D", lw=0.8),
        )

    ambush_offsets = [
        (28, 12),
        (36, 10),
        (42, 0),
        (36, -12),
        (28, -24),
    ]
    for idx, feat_name in enumerate(top_ambush_names):
        r = numeric_df[numeric_df["feature_name"] == feat_name].iloc[0]
        px = r["r2_linear_num"] * 100.0
        py = r["r2_binned_num"] * 100.0
        dx, dy = ambush_offsets[idx % len(ambush_offsets)]
        ax2.annotate(
            f"{feat_name} ({px:.2f}% -> {py:.2f}%)",
            xy=(px, py),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.3,
            fontweight="bold",
            color="#B22222",
            arrowprops=dict(arrowstyle="->", color="#E63946", lw=0.8),
        )

    # 在空白区域加一个小卡片提示总黑马数
    ambush_cnt = int(np.sum(is_ambush))
    ax2.text(
        0.97,
        0.23,
        f"雷达共救回 {ambush_cnt} 个被线性 R² 误杀的特征\n(线性 < 1%，但分箱非线性或 dCor 显著达标)",
        transform=ax2.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.8,
        color="#9B1C1C",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#FFF5F5", edgecolor="#F5C2C7", alpha=0.92),
    )

    ax2.set_xlabel("传统线性解释率 R²_linear [%]", fontsize=10.5)
    ax2.set_ylabel("15 分箱非线性解释率 R²_binned [%]", fontsize=10.5)
    ax2.set_title("49 个数值特征：传统线性 vs 分箱非线性抓黑马雷达", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlim(-max_lim * 0.03, max_lim)
    ax2.set_ylim(-max_lim * 0.03, max_lim)
    ax2.grid(True, linestyle=":", alpha=0.4)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(loc="lower right", fontsize=9, frameon=True, facecolor="#F9FAFB")

    # 总体大标题（汇总红黄绿灯分布，使用纯中文括号标识避免 Windows 字体缺 Emoji 方框）
    counts = real_df["体检评级"].value_counts().to_dict()
    g_cnt = counts.get("🟢 强信号尖子生", 0)
    y_cnt = counts.get("🟡 明确有效 / 隐藏非线性", 0)
    w_cnt = counts.get("⚪ 微弱辅助信号", 0)
    r_cnt = counts.get("🔴 无显著信号 / 建议弃用", 0)
    fig.suptitle(
        f"试纸 A 单兵双轨体检总览（75 个真特征：[绿灯]强信号 {g_cnt} | [黄灯]有效/非线性 {y_cnt} | [灰灯]微弱 {w_cnt} | [红灯]无显著 {r_cnt}）",
        fontsize=14.5,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_png_path, dpi=180)
    plt.close(fig)


# ====================================================================
# 第四部分：主流程入口
# ====================================================================

def run_strip_a_screening(
    data_path: str = "data/processed/clean_dataset.csv",
    schema_path: str = "data/processed/feature_schema.json",
    report_csv_path: str = "reports/step2a_strip_a_report.csv",
    summary_png_path: str = "reports/step2a_strip_a_summary.png",
    dcor_sample_size: int = 2000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    执行完整的试纸 A 体检流程并导出 CSV 报告与 PNG 可视化图表。
    """
    t0 = time.perf_counter()
    os.makedirs(os.path.dirname(report_csv_path), exist_ok=True)

    print("=" * 72)
    print("🧪 启动【试纸 A】单兵尖子扫雷与双轨公平体检 (Step 2A)")
    print("=" * 72)

    # 1. 读取特征元数据（严格区分数值列与无序文本列）
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    target_col = schema["target_col"]
    numeric_cols = list(schema["numeric_features"])
    categorical_cols = list(schema["categorical_features"])

    # 关键防坑：读取 CSV 时强制将 26 个无序文本列按字符串读入，严禁 pandas 自动转成数字！
    dtype_map = {col: str for col in categorical_cols}
    df = pd.read_csv(data_path, dtype=dtype_map, keep_default_na=False)
    n_rows = len(df)
    print(f"📂 已加载洁净数据集：{n_rows:,} 行 × {df.shape[1]} 列 | 目标列：{target_col}")
    print(f"   - 轨道 1（数值特征）：{len(numeric_cols)} 个 + 1 个随机影子哨兵")
    print(f"   - 轨道 2（无序文本特征，免编码原字符分组）：{len(categorical_cols)} 个")

    # 2. 提取目标列与全局方差准备
    y = df[target_col].to_numpy(dtype=np.float64)
    y_mean = float(np.mean(y))
    ss_total = float(np.sum((y - y_mean) ** 2))

    # 3. 注入影子哨兵列（固定随机种子 42 生成标准正态白噪声）
    rng = np.random.RandomState(random_state)
    shadow_col_name = "__shadow_random_noise__"
    shadow_values = rng.normal(loc=0.0, scale=1.0, size=n_rows)

    # 4. 构造用于 dCor 的固定抽样索引（n=2000，确保全流程可复现且秒级跑完）
    sample_n = min(dcor_sample_size, n_rows)
    sample_idx = rng.choice(n_rows, size=sample_n, replace=False)
    y_sample = y[sample_idx]
    dist_y = np.abs(y_sample[:, None] - y_sample[None, :])
    B_y = compute_double_centered_matrix(dist_y)
    dvar_y_sq = float(np.mean(B_y * B_y))
    print(f"🎯 dCor 抽样矩阵预计算完成（n={sample_n}，固定种子={random_state}）...")

    records = []

    # 5. 运行轨道 1：49 个数值特征 + 1 个影子哨兵
    t_track1 = time.perf_counter()
    track1_features = [(col, df[col].to_numpy(dtype=np.float64), "数值特征") for col in numeric_cols]
    track1_features.append((shadow_col_name, shadow_values, "影子随机哨兵"))

    for feat_name, x_arr, feat_label in track1_features:
        res = evaluate_numeric_feature(
            x=x_arr,
            y=y,
            y_mean=y_mean,
            ss_total=ss_total,
            x_sample=x_arr[sample_idx],
            B_y=B_y,
            dvar_y_sq=dvar_y_sq,
            n_bins=15,
        )
        p_raw = min(res["p_linear"], res["p_binned"])
        effective_r2 = max(res["r2_linear"], res["r2_binned"])

        records.append({
            "feature_name": feat_name,
            "track": "numeric",
            "feat_label": feat_label,
            "effective_r2": effective_r2,
            "p_raw": safe_clip_p(p_raw),
            "dcor": res["dcor"],
            # 数值轨道专属数值
            "pearson_r_num": res["pearson_r"],
            "r2_linear_num": res["r2_linear"],
            "p_linear_num": res["p_linear"],
            "spearman_rho_num": res["spearman_rho"],
            "p_spearman_num": res["p_spearman"],
            "r2_binned_num": res["r2_binned"],
            "p_binned_num": res["p_binned"],
            "n_bins_actual": res["n_bins_actual"],
            # 文本轨道占位
            "k_categories": np.nan,
            "eta2_num": np.nan,
            "omega2_num": np.nan,
            "p_anova_num": np.nan,
            "kw_eta2_h_num": np.nan,
            "p_kw_num": np.nan,
        })
    print(f"⏱️ 轨道 1（50 项数值/影子评估）完成，耗时 {time.perf_counter() - t_track1:.2f} 秒")

    # 6. 运行轨道 2：26 个无序文本特征（绝对免编码！）
    t_track2 = time.perf_counter()
    for feat_name in categorical_cols:
        cat_series = df[feat_name]
        cat_sample = cat_series.iloc[sample_idx].to_numpy()
        res = evaluate_categorical_feature(
            cat_series=cat_series,
            y=y,
            y_mean=y_mean,
            ss_total=ss_total,
            cat_sample=cat_sample,
            B_y=B_y,
            dvar_y_sq=dvar_y_sq,
        )
        p_raw = min(res["p_anova"], res["p_kw"])
        effective_r2 = res["omega2"]  # 统一以无偏 ω² 与数值 R² 同台对比

        records.append({
            "feature_name": feat_name,
            "track": "categorical",
            "feat_label": "无序文本特征",
            "effective_r2": effective_r2,
            "p_raw": safe_clip_p(p_raw),
            "dcor": res["dcor"],
            # 数值轨道占位
            "pearson_r_num": np.nan,
            "r2_linear_num": np.nan,
            "p_linear_num": np.nan,
            "spearman_rho_num": np.nan,
            "p_spearman_num": np.nan,
            "r2_binned_num": np.nan,
            "p_binned_num": np.nan,
            "n_bins_actual": np.nan,
            # 文本轨道专属数值
            "k_categories": res["k_categories"],
            "eta2_num": res["eta2"],
            "omega2_num": res["omega2"],
            "p_anova_num": res["p_anova"],
            "kw_eta2_h_num": res["kw_eta2_h"],
            "p_kw_num": res["p_kw"],
        })
    print(f"⏱️ 轨道 2（26 项无序文本免编码评估）完成，耗时 {time.perf_counter() - t_track2:.2f} 秒")

    # 7. 全局 BH 多重检验校正（76 项一起校正）
    p_raw_array = np.array([r["p_raw"] for r in records], dtype=np.float64)
    q_values = stats.false_discovery_control(p_raw_array, method="bh")
    for r, q_val in zip(records, q_values):
        r["q_value_bh"] = float(q_val)

    # 找到影子哨兵指标，作为红黄绿灯及画图的参考基准
    shadow_rec = next(r for r in records if r["feature_name"] == shadow_col_name)
    shadow_eff = shadow_rec["effective_r2"]
    shadow_dcor = shadow_rec["dcor"]

    # 8. 评定红黄绿灯与备注预警
    for r in records:
        rating, note_str = assign_traffic_light(r, shadow_dcor=shadow_dcor)
        r["rating"] = rating
        r["note"] = note_str
        r["is_linear_ambush"] = ("[线性误杀预警]" in note_str)

    # 9. 按有效解释率降序排序，并组装 0 NaN 的导出报表（跨轨道不适用项以 "-" 填充）
    records_sorted = sorted(records, key=lambda d: d["effective_r2"], reverse=True)

    def fmt_num(val, precision: int = 6) -> str:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "-"
        return f"{val:.{precision}f}"

    def fmt_p(val) -> str:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "-"
        return f"{val:.3e}"

    def fmt_int(val) -> str:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "-"
        return str(int(val))

    export_rows = []
    for rank_idx, r in enumerate(records_sorted, start=1):
        export_rows.append({
            "排名": rank_idx,
            "特征名称": r["feature_name"],
            "特征类型": r["feat_label"],
            "有效解释率": round(r["effective_r2"], 6),
            "体检评级": r["rating"],
            "主检测_p值": r["p_raw"],
            "BH校正_q值": r["q_value_bh"],
            "距离相关_dCor": round(r["dcor"], 6),
            "线性_Pearson_r": fmt_num(r["pearson_r_num"], 6),
            "线性_R2": fmt_num(r["r2_linear_num"], 6),
            "线性_p值": fmt_p(r["p_linear_num"]),
            "单调_Spearman_rho": fmt_num(r["spearman_rho_num"], 6),
            "单调_p值": fmt_p(r["p_spearman_num"]),
            "分箱非线性_R2_binned": fmt_num(r["r2_binned_num"], 6),
            "分箱非线性_p值": fmt_p(r["p_binned_num"]),
            "分箱或类别数_k": fmt_int(r["n_bins_actual"] if r["track"] == "numeric" else r["k_categories"]),
            "文本_Eta2": fmt_num(r["eta2_num"], 6),
            "文本_无偏Omega2": fmt_num(r["omega2_num"], 6),
            "文本_ANOVA_p值": fmt_p(r["p_anova_num"]),
            "文本_KW秩效应量_eta2_H": fmt_num(r["kw_eta2_h_num"], 6),
            "文本_KW_p值": fmt_p(r["p_kw_num"]),
            "备注提示": r["note"],
        })

    report_df = pd.DataFrame(export_rows)

    # 导出 UTF-8 BOM CSV（Excel 直接双击打开不乱码）
    report_df.to_csv(report_csv_path, index=False, encoding="utf-8-sig")

    # 10. 绘制双拼总结图
    numeric_only_df = pd.DataFrame([
        r for r in records_sorted
        if r["track"] == "numeric" and r["feature_name"] != shadow_col_name
    ])
    plot_strip_a_summary(
        report_df=report_df,
        numeric_df=numeric_only_df,
        shadow_eff=shadow_eff,
        output_png_path=summary_png_path,
    )

    total_time = time.perf_counter() - t0

    # 11. 控制台汇总打印与试纸 A 综合 Pass/Caution/Fail 结论
    real_only = report_df[report_df["特征名称"] != shadow_col_name]
    rating_counts = real_only["体检评级"].value_counts()
    top3 = real_only.head(3)
    top1_eff = float(top3.iloc[0]["有效解释率"])
    top1_q = float(top3.iloc[0]["BH校正_q值"])

    if top1_eff >= 0.10 and top1_q < 0.05:
        verdict = "PASS（存在显著强信号尖子生，数据富含可解释物理与工艺规律，建议继续试纸 B 团战探针）"
    elif top1_eff >= 0.02 and top1_q < 0.05:
        verdict = "CAUTION（信号中等偏弱，存在一定规律但需警惕多特征组合交互或分群效应）"
    else:
        verdict = "FAIL（单变量均未见明显解释力，需排查目标定义或纯随机波动）"

    print("\n" + "=" * 72)
    print(f"✅ 试纸 A 体检全部完成！总耗时：{total_time:.2f} 秒")
    print(f"📄 报表导出：{report_csv_path}（共 {len(report_df)} 行 × {report_df.shape[1]} 列）")
    print(f"🖼️ 图表导出：{summary_png_path}")
    print("-" * 72)
    print("📊 75 个真特征红黄绿灯分布：")
    for label in ["🟢 强信号尖子生", "🟡 明确有效 / 隐藏非线性", "⚪ 微弱辅助信号", "🔴 无显著信号 / 建议弃用"]:
        print(f"   {label:20s}: {int(rating_counts.get(label, 0)):2d} 个")

    shadow_row = report_df[report_df["特征名称"] == shadow_col_name].iloc[0]
    print("-" * 72)
    print("🕵️ 影子哨兵卧底表现（基准对照）：")
    print(
        f"   排名 #{int(shadow_row['排名'])} | 有效解释率={shadow_row['有效解释率']:.6f} "
        f"| dCor={shadow_row['距离相关_dCor']:.6f} | q={shadow_row['BH校正_q值']:.4f} "
        f"| 评级={shadow_row['体检评级']}"
    )

    print("-" * 72)
    print("🏆 Top 15 单兵尖子生榜单：")
    for _, row in real_only.head(15).iterrows():
        print(
            f"   #{int(row['排名']):2d} | {row['特征名称']:52s} | {row['特征类型']:6s} "
            f"| 有效解释率={row['有效解释率']*100:6.2f}% | dCor={row['距离相关_dCor']:.3f} "
            f"| {row['体检评级']} {row['备注提示']}"
        )

    ambush_df = real_only[real_only["备注提示"].str.contains("线性误杀预警", na=False)]
    print("-" * 72)
    print(f"⚡ 抓获【线性误杀预警】黑马特征（{len(ambush_df)} 个）：")
    if len(ambush_df) == 0:
        print("   （本次未发现线性<1%但非线性突围的数值特征）")
    else:
        for _, row in ambush_df.iterrows():
            print(
                f"   #{int(row['排名']):2d} | {row['特征名称']:36s} | 线性R²={float(row['线性_R2'])*100:5.2f}% "
                f"-> 分箱R²={float(row['分箱非线性_R2_binned'])*100:5.2f}% | dCor={row['距离相关_dCor']:.3f}"
            )

    print("-" * 72)
    print(f"🏁 试纸 A 综合裁决：{verdict}")
    print("=" * 72)

    return report_df


if __name__ == "__main__":
    run_strip_a_screening()
