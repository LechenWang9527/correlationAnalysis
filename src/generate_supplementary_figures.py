"""Render three 300 dpi supplementary figures from the audited snapshots."""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.ticker import MultipleLocator


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PHASE2 = REPORTS / "step3_phase2"
TARGET = "ActualVsIEGap"
SIZE = "tire_size"
APEX = "1_apex_-_width"
BEAD = "bead_-_bead_inner_diameter"
MIN_GROUP_ROWS = 200
DPI = 300

INK = "#272725"
MUTED = "#56544E"
GREY = "#8A8277"
GREY_LIGHT = "#F3F0EB"
GRID = "#DED9D1"
TEAL = "#126261"
TEAL_LIGHT = "#E7F2EF"
AMBER = "#9B5A17"
AMBER_LIGHT = "#FCF1DF"


@dataclass(frozen=True)
class Phase2:
    audit: dict
    freeze: dict
    seal: dict
    cohort: pd.DataFrame
    outer_sizes: pd.Series
    alerts: int
    legacy: pd.Series


@dataclass(frozen=True)
class SizeSlice:
    size: str
    rows: pd.DataFrame
    r: float


def load_phase2(clean_count: int) -> Phase2:
    """Read the audit, benchmark and fold artifacts used by the schematics."""
    audit = json.loads((PHASE2 / "provenance_summary.json").read_text(encoding="utf-8"))
    freeze = json.loads((PHASE2 / "freeze_manifest.json").read_text(encoding="utf-8"))
    seal = json.loads((PHASE2 / "replay_seal.json").read_text(encoding="utf-8"))
    cohort = pd.read_csv(PHASE2 / "cohort_audit.csv").pivot_table(
        index="PlannedStartMonth", columns="status", values="proxies",
        aggfunc="sum", fill_value=0,
    )
    cohort["candidates"] = cohort.sum(axis=1)
    cohort["excluded"] = cohort["candidates"] - cohort["ELIGIBLE_PROXY"]

    oof = pd.read_csv(
        PHASE2 / "development_oof.csv", usecols=["proxy_id", "family", "outer_fold"]
    )
    replay = pd.read_csv(PHASE2 / "november_predictions.csv", usecols=["family"])
    inner = pd.read_csv(PHASE2 / "inner_membership.csv", dtype={"outer": str})
    nested = inner.loc[inner["outer"].ne("final")].merge(
        oof, on="proxy_id", validate="many_to_one"
    )
    outer_sizes = oof.groupby("outer_fold").size()
    inner_sizes = nested.groupby("outer")["inner_fold"].nunique()
    metrics = pd.read_csv(PHASE2 / "development_metrics.csv")
    scored = metrics.loc[
        metrics["scope"].eq("development_oof") & metrics["model"].eq("A_CatBoost"), "n"
    ].item()
    if (
        clean_count != audit["clean_rows"]
        or cohort["candidates"].sum() != audit["proxy_orders"]
        or cohort.loc[:10, "ELIGIBLE_PROXY"].sum() != len(oof)
        or len(oof) != scored
        or len(oof) != freeze["n_development_eligible"]
        or cohort.loc[11, "ELIGIBLE_PROXY"] != len(replay)
        or len(replay) != seal["eligible_replay"]
    ):
        raise ValueError("Clean data, cohort audit and scoring counts disagree")
    if (
        freeze["frozen_utc"] >= seal["scored_utc"]
        or len(outer_sizes) != freeze["outer_folds"]
        or set(inner_sizes.index) != set(outer_sizes.index.astype(str))
        or not inner_sizes.eq(freeze["inner_folds"]).all()
        or nested["outer"].eq(nested["outer_fold"].astype(str)).any()
        or oof.groupby("family")["outer_fold"].nunique().max() != 1
        or nested.groupby(["outer", "family"])["inner_fold"].nunique().max() != 1
        or len(set(oof["family"]) & set(replay["family"])) != seal["eligible_family_overlap"]
    ):
        raise ValueError("Nested fold membership disagrees with the freeze or replay seal")

    strip_b = pd.read_csv(REPORTS / "step2b_strip_b_simpson_report.csv")
    legacy = pd.read_csv(REPORTS / "step3_model_benchmark_report.csv").set_index("Model")
    return Phase2(
        audit, freeze, seal, cohort, outer_sizes,
        int(strip_b["是否触发警报"].eq("是").sum()), legacy.loc["CatBoost"],
    )


def correlation(rows: pd.DataFrame, feature: str) -> float:
    return float(rows[feature].corr(rows[TARGET]))


def size_slices(clean: pd.DataFrame, feature: str) -> list[SizeSlice]:
    """Apply the Step 2 minimum size and nonconstant-variable rules."""
    groups = []
    for size, rows in clean.groupby(SIZE, sort=False):
        if len(rows) < MIN_GROUP_ROWS:
            continue
        pairs = rows[[feature, TARGET]].dropna()
        centered = pairs - pairs.mean()
        if len(pairs) >= MIN_GROUP_ROWS and centered.pow(2).sum().gt(1e-12).all():
            groups.append(SizeSlice(str(size), pairs, correlation(pairs, feature)))
    return sorted(groups, key=lambda group: group.r)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": GREY,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.45,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 5.5,
            "ytick.labelsize": 5.5,
            "xtick.major.width": 0.45,
            "ytick.major.width": 0.45,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "legend.frameon": False,
            "mathtext.fontset": "dejavusans",
        }
    )


def canvas(height_mm: int):
    return plt.figure(figsize=(183 / 25.4, height_mm / 25.4), facecolor="white")


def schematic(fig, bounds=(0, 0, 1, 1)):
    ax = fig.add_axes(bounds)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_axis_off()
    return ax


def heading(ax, x, y, title, letter=None):
    if letter:
        ax.text(x, y, letter, fontsize=8, weight="bold", va="top")
        x += 0.043
    ax.text(x, y, title, fontsize=7, weight="bold", va="top")


def box(ax, x, y, w, h, title, detail, fill=GREY_LIGHT, edge=GRID):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge, lw=0.55))
    ax.text(x + 0.055 * w, y + h * 0.74, title, fontsize=7, weight="bold", va="center")
    ax.text(
        x + 0.055 * w, y + h * 0.33, detail, fontsize=5.6, va="center",
        linespacing=1.35,
    )


def arrow(ax, start, end, color=GREY):
    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=6, lw=0.65,
            shrinkA=0, shrinkB=0, color=color,
        )
    )


def figure_pipeline(data: Phase2):
    fig = canvas(100)
    ax = schematic(fig)
    heading(ax, 0.04, 0.95, "Row-level data and historical model development")
    stages = [
        ("Raw export", f"{data.audit['raw_rows']:,} source rows\nManufacturing snapshot"),
        (
            "Step 1  Cleaning",
            f"{data.audit['clean_rows']:,} clean rows\nStandardized schema and lineage",
        ),
        (
            "Step 2  Screening",
            "Test A   Univariate screening\n"
            f"Test B   {data.alerts} Simpson alerts",
        ),
        (
            "Step 3  Part 1",
            "Historical benchmark and voting\n"
            f"CatBoost R² {data.legacy['Dev_OOF_R2']:.3f} CV / "
            f"{data.legacy['Nov_Blind_R2']:.3f} Nov",
        ),
    ]
    x_positions = (0.04, 0.275, 0.51, 0.745)
    for x, (title, detail) in zip(x_positions, stages):
        box(ax, x, 0.71, 0.215, 0.17, title, detail)
    for left, right in zip(x_positions, x_positions[1:]):
        arrow(ax, (left + 0.216, 0.795), (right - 0.002, 0.795))

    flags = data.audit["row_flags"]
    matches = flags["PASS"]
    denominator = matches + flags["IDENTITY_RESIDUAL"]
    arrow(ax, (0.8525, 0.708), (0.8525, 0.660), AMBER)
    ax.add_patch(Rectangle((0.04, 0.49), 0.92, 0.165, fc=AMBER_LIGHT, ec=AMBER, lw=0.65))
    heading(ax, 0.055, 0.624, "Leakage audit")
    ax.text(
        0.055, 0.559, f"{matches:,} / {denominator:,} valid-denominator rows match",
        fontsize=6, va="center",
    )
    ax.text(0.055, 0.520, f"{matches / denominator:.3%} identity match", weight="bold", va="center")
    ax.text(0.445, 0.581, r"$Y = O/(C\,Q) - (IE + TOL)/100$", fontsize=7, va="center")
    ax.text(
        0.445, 0.524, "Outcome-defining inputs were available to the historical models",
        fontsize=5.8, va="center",
    )

    arrow(ax, (0.156, 0.488), (0.156, 0.428), TEAL)
    heading(ax, 0.04, 0.405, "Phase 2  Proxy-order modeling and attribution")
    a_count = len(data.freeze["features"]["A"])
    b_count = len(data.freeze["features"]["B"])
    box(
        ax, 0.04, 0.13, 0.232, 0.23, "Proxy-order aggregation",
        f"{data.audit['proxy_orders']:,} candidate orders\n"
        f"{data.audit['proxy_families']:,} proxy families\n"
        "Planned keys define orders",
        fill=TEAL_LIGHT, edge=TEAL,
    )
    box(
        ax, 0.294, 0.13, 0.314, 0.23, "A/B input whitelists",
        f"A   {a_count} fields, physical + planning\n"
        f"B   {b_count} physical fields\n"
        "Source Y retained as label\n"
        "Outcome-defining fields held for audit",
        fill=TEAL_LIGHT, edge=TEAL,
    )
    box(
        ax, 0.63, 0.13, 0.33, 0.23, "Attribution and causal assessment",
        "TreeSHAP · conditional permutation\n"
        "Module LOCO · support-aware 1D ALE\n"
        "Causal DML   [BLOCKED]",
        fill=TEAL_LIGHT, edge=TEAL,
    )
    for left, right in ((0.272, 0.294), (0.608, 0.63)):
        arrow(ax, (left + 0.001, 0.245), (right - 0.002, 0.245), TEAL)
    ax.text(
        0.04, 0.058,
        "Validation sequence   Develop on Aug–Oct proxy families, freeze, score November, audit.",
        fontsize=5.6, color=MUTED,
    )
    return fig


def scatter(ax, rows, feature, color, alpha, marker="o", size=3):
    ax.scatter(rows[feature], rows[TARGET], color=color, alpha=alpha, marker=marker,
               s=size, linewidths=0, rasterized=True)


def fit_line(ax, rows, feature, color, linestyle="-", width=1.1, zorder=4):
    x = rows[feature].to_numpy()
    y = rows[TARGET].to_numpy()
    limits = np.array([x.min(), x.max()])
    slope = np.dot(x - x.mean(), y - y.mean()) / np.sum((x - x.mean()) ** 2)
    ax.plot(limits, y.mean() + slope * (limits - x.mean()), color=color,
            linestyle=linestyle, lw=width, zorder=zorder)


def line_key(color, label, linestyle="-", marker=None):
    return Line2D(
        [], [], color=color, linestyle=linestyle, linewidth=1, marker=marker,
        markersize=3, label=label,
    )


def figure_simpson(clean: pd.DataFrame, apex: list[SizeSlice], bead: list[SizeSlice]):
    fig = canvas(137)
    top_a = fig.add_axes((0.103, 0.565, 0.386, 0.275))
    top_b = fig.add_axes((0.576, 0.565, 0.386, 0.275), sharey=top_a)
    labels = schematic(fig)

    a_full = clean[[APEX, TARGET]].dropna()
    b_full = clean[[BEAD, TARGET]].dropna()
    b_pool = pd.concat([group.rows for group in bead])
    heading(labels, 0.035, 0.962, "Apex width masks opposite trends", "a")
    heading(labels, 0.535, 0.962, "Bead inner diameter reverses sign", "b")
    labels.text(
        0.078, 0.917, f"Full table  r = {correlation(a_full, APEX):.3f}  (n = {len(a_full):,})"
    )
    labels.text(
        0.578, 0.917, f"Full table  r = {correlation(b_full, BEAD):.3f}  (n = {len(b_full):,})"
    )

    scatter(top_a, a_full, APEX, GREY, 0.09, size=2.4)
    fit_line(top_a, a_full, APEX, GREY, "--")
    a_keys = []
    for group, color, marker in ((apex[-1], TEAL, "o"), (apex[0], AMBER, "^")):
        scatter(top_a, group.rows, APEX, color, 0.32, marker)
        fit_line(top_a, group.rows, APEX, color)
        label = f"{group.size}  r = {group.r:+.3f}  (n = {len(group.rows):,})"
        a_keys.append(line_key(color, label, marker=marker))
    top_a.legend(handles=a_keys, loc="lower left", bbox_to_anchor=(0, 1.015),
                 fontsize=5.4, borderaxespad=0, handlelength=1.5, labelspacing=0.3)

    scatter(top_b, b_pool, BEAD, TEAL, 0.13, size=2.6)
    for group in bead:
        fit_line(top_b, group.rows, BEAD, TEAL, width=0.85)
    fit_line(top_b, b_pool, BEAD, AMBER, "--", width=1.25, zorder=5)
    top_b.legend(
        handles=[
            line_key(AMBER, f"Pooled valid sizes  r = {correlation(b_pool, BEAD):.3f}"
                     f"  (n = {len(b_pool):,})", "--"),
            line_key(TEAL, f"Within-size fits  ({len(bead)} sizes)"),
        ],
        loc="lower left", bbox_to_anchor=(0, 1.015), fontsize=5.4,
        borderaxespad=0, handlelength=1.5, labelspacing=0.3,
    )
    top_a.set_ylabel("Actual–IE cycle-time gap (s per tire)")
    top_a.set_xlabel("Apex width (source units)")
    top_b.set_xlabel("Bead inner diameter (source units)")
    top_b.tick_params(labelleft=False)
    for ax in (top_a, top_b):
        ax.set_ylim(-10, 51)
        ax.yaxis.set_major_locator(MultipleLocator(10))
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=GRID, linewidth=0.4, alpha=0.7)

    labels.text(0.103, 0.474, f"Within-size correlations  ({len(apex)} valid sizes)",
                fontsize=6.5, weight="bold")
    labels.text(
        0.576, 0.474,
        f"All {len(bead)} valid sizes have positive r ({bead[0].r:.3f}–{bead[-1].r:.3f})",
        fontsize=6.5, weight="bold",
    )
    ranks = fig.add_axes((0.103, 0.154, 0.386, 0.278))
    ranks.scatter([group.r for group in apex], range(1, len(apex) + 1), c=GREY,
                  s=8, edgecolors="none")
    ranks.scatter([apex[0].r], [1], c=AMBER, s=18, marker="^", edgecolors="none")
    ranks.scatter([apex[-1].r], [len(apex)], c=TEAL, s=14, edgecolors="none")
    ranks.set_xlim(-0.74, 0.74)
    ranks.set_ylim(-2, len(apex) + 3)
    ranks.set_yticks([1, 20, 40, len(apex)])
    ranks.set_ylabel("Tire size rank")
    ranks.axvline(0, color=GREY, lw=0.6, linestyle="--", zorder=0)

    dots = fig.add_axes((0.66, 0.154, 0.302, 0.278))
    dots.scatter([group.r for group in bead], range(len(bead)), c=TEAL,
                 s=10, edgecolors="none")
    dots.set_yticks(range(len(bead)), [group.size for group in bead], fontsize=5.1)
    dots.tick_params(axis="y", length=0)
    dots.spines["left"].set_visible(False)
    dots.set_xlim(0, 0.72)
    dots.set_ylim(-0.6, len(bead) - 0.4)
    dots.axvline(0, color=GREY, lw=0.6, linestyle="--", zorder=0)
    for ax in (ranks, dots):
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.set_xlabel("Pearson r")

    within = sum(
        len(group.rows) * group.rows[BEAD].cov(group.rows[TARGET], ddof=0)
        for group in bead
    ) / len(b_pool)
    total = b_pool[BEAD].cov(b_pool[TARGET], ddof=0)
    labels.text(
        0.103, 0.067,
        "Valid sizes have at least 200 rows and both variables vary. "
        "Lines are ordinary least squares.",
        fontsize=5.5, color=MUTED,
    )
    labels.text(
        0.103, 0.033,
        f"Bead covariance on valid rows   Within {within:+.2f}   "
        f"Between {total - within:+.2f}   Total {total:+.2f}  (source units × s per tire)",
        fontsize=5.5, color=MUTED,
    )
    return fig


def cohort_panel(ax, data: Phase2):
    heading(ax, 0.0, 1.0, "Cohort attrition", "a")
    for width, y, title, detail, fill, edge in (
        (1.0, 0.76, f"{data.audit['raw_rows']:,} raw rows", "Source manufacturing export",
         GREY_LIGHT, GRID),
        (0.91, 0.56, f"{data.audit['clean_rows']:,} clean rows",
         f"{data.audit['raw_multi_matches']:,} excess source mappings resolved", GREY_LIGHT, GRID),
        (0.79, 0.36, f"{data.audit['proxy_orders']:,} candidate proxy orders",
         f"{data.cohort['excluded'].sum():,} excluded at cohort screening", TEAL_LIGHT, TEAL),
    ):
        box(ax, (1 - width) / 2, y, width, 0.145, title, detail, fill, edge)
    arrow(ax, (0.5, 0.752), (0.5, 0.708))
    arrow(ax, (0.5, 0.552), (0.5, 0.508))

    x = (0.005, 0.44, 0.72, 0.985)
    for field, pos in zip(("Plan month", "Candidate", "Excluded", "Scored"), x):
        ax.text(pos, 0.295, field, weight="bold", fontsize=5.6,
                ha="left" if pos == x[0] else "right", va="center")
    ax.plot([0, 1], [0.265, 0.265], color=GRID, lw=0.65)
    for i, (month, row) in enumerate(data.cohort.iterrows()):
        y = 0.231 - i * 0.051
        values = (f"{month:02d}", row["candidates"], row["excluded"], row["ELIGIBLE_PROXY"])
        for pos, value in zip(x, values):
            text = f"{value:,}" if pos != x[0] else value
            ax.text(pos, y, text, ha="left" if pos == x[0] else "right", va="center")
    ax.text(0.0, 0.047,
            f"Development   {data.outer_sizes.sum():,}    "
            f"Replay   {data.seal['eligible_replay']:,}",
            fontsize=5.6, weight="bold", va="center")
    ax.text(0.0, 0.007,
            f"Excluded   {data.cohort['AMBIGUOUS_LABEL'].sum():,} label conflicts, "
            f"{data.cohort['CROSS_BOUNDARY'].sum():,} cross-boundary family",
            fontsize=5.4, color=MUTED)


def validation_panel(ax, data: Phase2):
    fold_design = f"{data.freeze['outer_folds']} × {data.freeze['inner_folds']}"
    heading(ax, 0, 1, f"{fold_design} nested validation and time holdout", "b")
    ax.text(
        0.0, 0.907, f"Aug–Oct development   {data.outer_sizes.sum():,} scored proxy orders",
        weight="bold", fontsize=6.4,
    )
    ax.text(0.0, 0.852, "Run", fontsize=5.6, weight="bold")
    ax.text(0.125, 0.852, "Outer proxy-family folds", fontsize=5.6, weight="bold")
    ax.text(0.555, 0.852, "Inner folds", fontsize=5.6, weight="bold")
    ax.text(1.0, 0.852, "Held-out n", fontsize=5.6, weight="bold", ha="right")
    for i, (fold, count) in enumerate(data.outer_sizes.items()):
        y = 0.788 - i * 0.067
        ax.text(0.0, y, f"{fold + 1}", va="center")
        for j in range(data.freeze["outer_folds"]):
            color = TEAL if i == j else TEAL_LIGHT
            ax.add_patch(Rectangle((0.125 + 0.064 * j, y - 0.022), 0.05, 0.043,
                                   fc=color, ec=TEAL, lw=0.45))
        arrow(ax, (0.45, y), (0.534, y), TEAL)
        for j in range(data.freeze["inner_folds"]):
            ax.add_patch(Rectangle((0.555 + 0.064 * j, y - 0.022), 0.05, 0.043,
                                   fc="white", ec=TEAL, lw=0.55))
        ax.text(1.0, y, f"{count:,}", ha="right", va="center")

    for x, color, label in ((0.0, TEAL, "Held out"), (0.197, TEAL_LIGHT, "Train"),
                            (0.357, "white", "MAE tuning within outer training families")):
        ax.add_patch(Rectangle((x, 0.445), 0.027, 0.025, fc=color, ec=TEAL, lw=0.45))
        ax.text(x + 0.04, 0.457, label, fontsize=5.3, va="center")

    box(ax, 0, 0.272, 1, 0.129, "Final refit and development freeze",
        "Refit on Aug–Oct, then seal model, folds and hashes", TEAL_LIGHT, TEAL)
    ax.plot([0, 1], [0.22, 0.22], color=AMBER, linestyle="--", lw=0.7)
    ax.text(0.025, 0.22, "Freeze boundary", fontsize=5.5, va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
    arrow(ax, (0.5, 0.268), (0.5, 0.173), AMBER)
    box(ax, 0, 0.036, 1, 0.133,
        f"November replay   {data.seal['eligible_replay']:,} scored proxy orders",
        "Separate scoring process  ·  "
        f"{data.seal['eligible_family_overlap']} shared proxy families",
        AMBER_LIGHT, AMBER)
    ax.text(0.0, 0.0, "November labels were used in the historical workflow.",
            fontsize=5.4, color=MUTED)


def figure_cohort(data: Phase2):
    fig = canvas(118)
    cohort_panel(schematic(fig, (0.04, 0.065, 0.385, 0.89)), data)
    validation_panel(schematic(fig, (0.475, 0.065, 0.485, 0.89)), data)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PHASE2 / "figures")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    configure_style()
    clean = pd.read_csv(
        ROOT / "data/processed/clean_dataset.csv", usecols=[APEX, BEAD, SIZE, TARGET]
    )
    phase2 = load_phase2(len(clean))
    apex = size_slices(clean, APEX)
    bead = size_slices(clean, BEAD)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, figure in (
        ("fig1_1_data_pipeline_architecture.png", figure_pipeline(phase2)),
        ("fig2_2_simpson_paradox_reversal.png", figure_simpson(clean, apex, bead)),
        ("fig4_3_cohort_attrition_and_freeze_boundary.png", figure_cohort(phase2)),
    ):
        figure.savefig(args.output_dir / filename, dpi=DPI)
        plt.close(figure)
    logging.info("Generated 3 supplementary figures at %d dpi", DPI)


if __name__ == "__main__":
    main()
