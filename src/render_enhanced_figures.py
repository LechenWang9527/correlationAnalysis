"""Render the four Phase 2 evidence panels from sealed predictions and summaries."""

import matplotlib
import numpy as np

from step3_no_leakage_attribution import ALE_COLUMNS, cluster_weights
from step3_no_leakage_data import MODULES, RECIPE, TARGET

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator


MODELS = ('Schedule_CatBoost', 'A_CatBoost', 'B_CatBoost', 'A_Ridge', 'B_Ridge')
MODEL_LABELS = ('Schedule\nonly', 'A\nCat.', 'B\nCat.', 'A\nRidge', 'B\nRidge')
COHORTS = (
    ('development_oof', 'Aug–Oct OOF', '#4d7099'),
    ('november_replay', 'Nov replay', '#d7a779'),
)
MODULE_STYLE = {
    'geometry_bead': ('Geometry and bead', '#738eaf'),
    'liner': ('Liner', '#448d8e'),
    'carcass_reinforcement': ('Carcass and reinforcement', '#907da4'),
    'sidewall': ('Sidewall', '#b4875a'),
    'belt_cap': ('Belt and cap ply', '#668769'),
    'tread': ('Tread', '#b97476'),
}
FEATURE_MODULE = {feature: module for module, features in MODULES.items() for feature in features}
ALE_FEATURES = ('1_liner_-_thickness', '1_apex_-_width')
FOLD_COLORS = ('#456d9d', '#448d8e', '#ac7745', '#8b70a5', '#6b8057')
FOLD_MARKERS = ('o', 's', '^', 'D', 'v')
INK = '#202b35'
MUTED = '#445361'
GRID = '#e5e9ed'
LOW_SUPPORT = '#dbe1e6'
STYLE = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 8,
    'text.color': INK,
    'axes.labelcolor': INK,
    'axes.edgecolor': '#697681',
    'axes.linewidth': 0.65,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.color': INK,
    'ytick.color': INK,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'legend.frameon': False,
}


def heading(fig, title, subtitle):
    """Place the panel title and its statistical context above the axes."""
    fig.text(0.025, 0.975, title, va='top', fontsize=14, fontweight='bold')
    fig.text(0.025, 0.91, subtitle, va='top', fontsize=8.6, color=MUTED)


def performance_intervals(frame):
    """Calculate percentile intervals by resampling recipes within one cohort."""
    y = frame[TARGET].to_numpy()
    errors = frame[list(MODELS)].to_numpy() - y[:, None]
    levels, inverse, weights = cluster_weights(frame[RECIPE])
    moments = np.column_stack((np.abs(errors), errors ** 2, y, y ** 2))
    totals = np.zeros((len(levels), moments.shape[1]))
    np.add.at(totals, inverse, moments)
    draws = (weights @ totals) / (weights @ np.bincount(inverse))[:, None]
    mae, mse = np.split(draws[:, :-2], 2, axis=1)
    variance = draws[:, -1] - draws[:, -2] ** 2
    r2 = 1 - mse / np.where(variance > 0, variance, np.nan)[:, None]
    return {metric: np.nanpercentile(values, [2.5, 97.5], axis=0)
            for metric, values in (('mae', mae), ('rmse', np.sqrt(mse)), ('r2', r2))}


def model_performance(scores, oof, nov):
    cohorts = [(label, color, scores.loc[scores['scope'].eq(scope)].set_index('model').loc[list(MODELS)],
                performance_intervals(frame))
               for (scope, label, color), frame in zip(COHORTS, (oof, nov))]
    fig, axes = plt.subplots(1, 3, figsize=(7.9, 4.05))
    fig.subplots_adjust(left=0.069, right=0.977, bottom=0.155, top=0.75, wspace=0.29)
    heading(fig, 'a  Model performance', 'Five models with 95% recipe-cluster bootstrap intervals')

    positions = np.arange(len(MODELS))
    offsets = (-0.19, 0.19)
    for ax, (metric, label) in zip(axes, (('mae', 'MAE (s/tire)'), ('rmse', 'RMSE (s/tire)'),
                                          ('r2', 'Coefficient of determination ($R^2$)'))):
        ax.set_title(label, loc='left', fontsize=9, fontweight='bold', pad=6)
        ax.set_axisbelow(True)
        ax.grid(axis='y', linewidth=0.6, color=GRID)
        for offset, (_, color, scores_by_model, intervals) in zip(offsets, cohorts):
            x = positions + offset
            low, high = intervals[metric]
            ax.bar(x, scores_by_model[metric], width=0.34, color=color)
            ax.vlines(x, low, high, color=INK, linewidth=0.85)
            for bound in (low, high):
                ax.hlines(bound, x - 0.09, x + 0.09, color=INK, linewidth=0.75)
        ax.set_xticks(positions, MODEL_LABELS, fontsize=6.8)
        ax.tick_params(axis='x', length=0, pad=5)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        low_ci = min(interval[metric][0].min() for _, _, _, interval in cohorts)
        high_ci = max(interval[metric][1].max() for _, _, _, interval in cohorts)
        lower = min(0, low_ci * 1.10)
        upper = max(0, high_ci)
        ax.set_ylim(lower, upper + (upper - lower) * (0.24 if metric == 'r2' else 0.10))

    _, _, development, intervals = cohorts[0]
    gain = development.loc['A_CatBoost', 'r2'] - development.loc['Schedule_CatBoost', 'r2']
    x = positions[:2] + offsets[0]
    ymin, ymax = axes[-1].get_ylim()
    span = ymax - ymin
    bracket = max(intervals['r2'][1, :2]) + span * 0.06
    axes[-1].plot(np.repeat(x, 2), [intervals['r2'][1, 0] + 0.008, bracket,
                  bracket, intervals['r2'][1, 1] + 0.008], color='#396551', linewidth=0.9)
    axes[-1].text(x.mean(), bracket + 0.01, f'{gain * 100:+.1f} pp $R^2$', ha='center',
                  va='bottom', color='#396551', fontsize=9, fontweight='bold')

    keys = [Patch(facecolor=color, label=f'{label} ($n$ = {int(table["n"].iloc[0]):,})')
            for label, color, table, _ in cohorts]
    fig.legend(handles=keys, loc='upper left', bbox_to_anchor=(0.018, 0.875), ncols=2)
    fig.text(0.025, 0.035, 'Cat. and schedule-only models use CatBoost. '
             'A: physical + scheduling. B: physical.',
             fontsize=7.7, color=MUTED)
    return fig


def treeshap_physical(shap):
    fig, ax = plt.subplots(figsize=(7.9, 4.05))
    fig.subplots_adjust(left=0.405, right=0.967, bottom=0.198, top=0.845)
    top = shap.nsmallest(20, 'rank')
    positions = np.arange(len(top))
    labels = top['feature'].str.replace('_-_', ' / ', regex=False).str.replace('_', ' ', regex=False)
    colors = [MODULE_STYLE[FEATURE_MODULE[name]][1] for name in top['feature']]
    ax.set_axisbelow(True)
    ax.grid(axis='x', linewidth=0.6, color=GRID)
    for pos, color in zip(positions, colors):
        ax.axhspan(pos - 0.47, pos + 0.47, color=color, alpha=0.11, linewidth=0)
    ax.barh(positions, top['mean_abs'], height=0.60, color=colors)
    ax.hlines(positions, top['ci_low'], top['ci_high'], color=INK, linewidth=0.8)
    for pos, row in zip(positions, top.itertuples(index=False)):
        ax.annotate(f'{row.mean_abs:.3f}', (row.ci_high, pos), xytext=(4, 0),
                    textcoords='offset points', va='center', fontsize=7.3)
    ax.set_yticks(positions, [f'{rank:02d}  {label}' for rank, label in zip(top['rank'], labels)],
                  fontsize=7.4)
    ax.tick_params(axis='y', length=0)
    ax.set_ylim(len(top) - 0.5, -0.5)
    ax.set_xlim(0, top['ci_high'].max() * 1.20)
    ax.set_xlabel('Mean absolute TreeSHAP (s/tire)')
    heading(fig, 'b  Top 20 physical features',
            f'Group B CatBoost OOF ($n$ = {int(top["n"].iloc[0]):,}) with 95% recipe-cluster intervals')
    keys = [Patch(facecolor=color, label=label) for label, color in MODULE_STYLE.values()]
    fig.legend(handles=keys, loc='lower center', ncols=6, fontsize=7.5,
               bbox_to_anchor=(0.5, 0.027), columnspacing=1.5)
    return fig


def module_attribution(permutation, loco):
    loco = loco.loc[loco['scope'].eq('development_oof') & loco['after'].str.startswith('LOCO_')]
    loco = loco.assign(module=loco['after'].str.removeprefix('LOCO_')).set_index('module')
    permutation = permutation.set_index('module')
    modules = permutation.loc[list(MODULES), 'delta_mae'].sort_values(ascending=False).index
    permutation, loco = permutation.loc[modules], loco.loc[modules]
    colors = [MODULE_STYLE[module][1] for module in modules]
    positions = np.arange(len(modules))
    low = min(0, permutation['mae_lo'].min(), loco['mae_lo'].min())
    high = max(0, permutation['mae_hi'].max(), loco['mae_hi'].max())
    pad = (high - low) * 0.07

    fig = plt.figure(figsize=(7.9, 3.53))
    grid = fig.add_gridspec(1, 5, width_ratios=(1.12, 2.32, 1.56, 2.32, 0.52),
                           left=0.026, right=0.974, bottom=0.209, top=0.775, wspace=0.08)
    values_left, left, names, right, values_right = grid.subplots(sharey=True)
    heading(fig, 'c  Physical-module attribution',
            'Paired change in MAE (s/tire) with 95% recipe-cluster bootstrap intervals')
    left.set_title('Conditional permutation', fontsize=9, fontweight='bold')
    right.set_title('LOCO refit', fontsize=9, fontweight='bold')
    for ax, data in ((left, permutation), (right, loco)):
        ax.set_axisbelow(True)
        ax.grid(axis='x', color=GRID, linewidth=0.6)
        ax.barh(positions, data['delta_mae'], color=colors, height=0.55)
        ax.hlines(positions, data['mae_lo'], data['mae_hi'], color=INK, linewidth=0.9)
        ax.scatter(data['delta_mae'], positions, color=INK, s=7, zorder=3)
        ax.axvline(0, color=MUTED, linewidth=0.65)
        ax.set_xlim(low - pad, high + pad)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
    left.invert_xaxis()
    left.set_ylim(len(modules) - 0.5, -0.5)
    for ax in (values_left, names, values_right):
        ax.set_axis_off()
    for pos, module in zip(positions, modules):
        row = permutation.loc[module]
        values_left.text(0, pos, f'{row.delta_mae:+.3f}  |  {row.coverage:.1%}',
                         va='center', fontsize=8, transform=values_left.get_yaxis_transform())
        names.text(0.5, pos, MODULE_STYLE[module][0], ha='center', va='center', fontsize=7.5,
                   transform=names.get_yaxis_transform())
        values_right.text(1, pos, f'{loco.loc[module, "delta_mae"]:+.3f}', ha='right', va='center',
                          fontsize=8, transform=values_right.get_yaxis_transform())
    values_left.set_title('Effect | Coverage', loc='left', fontsize=7.5, color=MUTED)
    values_right.set_title('Effect', loc='right', fontsize=7.5, color=MUTED)
    fig.text(0.025, 0.075, 'Permutation estimates use supported subsets. LOCO uses the full OOF cohort.',
             fontsize=7.7, color=MUTED)
    fig.text(0.025, 0.032, 'Positive change indicates higher error after replacement or refitting.',
             fontsize=7.7, color=MUTED)
    return fig


def ale_supported_ranges(ale):
    fig, axes = plt.subplots(2, 2, figsize=(7.9, 3.53), sharex='col', sharey='row',
                             gridspec_kw={'height_ratios': [1.35, 1]})
    fig.subplots_adjust(left=0.091, right=0.978, bottom=0.19, top=0.70, hspace=0.18, wspace=0.18)
    heading(fig, 'd  ALE responses and support',
            'Group B CatBoost OOF, responses shown over qualifying bins')
    for column, feature in enumerate(ALE_FEATURES):
        response, support = axes[:, column]
        bins = ale.loc[ale['feature'].eq(feature)].dropna(subset=['low', 'high'])
        supported = bins.loc[bins['status'].eq('SUPPORTED')]
        response.set_title(f'{ALE_COLUMNS[feature]} ({len(supported)}/{len(bins)} bins qualify)',
                           fontsize=9, fontweight='bold', loc='left')
        response.axhline(0, color='#82909b', linewidth=0.65)
        response.grid(axis='y', color=GRID, linewidth=0.6)
        response.yaxis.set_major_locator(MaxNLocator(nbins=4))
        for (fold, _), segment in supported.groupby(['fold', 'segment']):
            segment = segment.sort_values('bin')
            response.plot(np.r_[segment['low'].iloc[0], segment['high']],
                          np.r_[segment['ale_low'].iloc[0], segment['ale_high']],
                          color=FOLD_COLORS[fold], marker=FOLD_MARKERS[fold],
                          markersize=3, linewidth=1)

        for row in bins.itertuples(index=False):
            color = FOLD_COLORS[row.fold] if row.status == 'SUPPORTED' else LOW_SUPPORT
            support.barh(row.fold, row.high - row.low, left=row.low, height=0.72,
                         color=color, edgecolor='white', linewidth=0.45)
        support.set_yticks(range(len(FOLD_COLORS)), [f'Fold {i + 1}' for i in range(len(FOLD_COLORS))])
        support.tick_params(axis='y', length=0)
        support.spines['left'].set_visible(False)
        support.set_ylim(len(FOLD_COLORS) - 0.5, -0.5)
        support.set_xlabel('Recorded specification')
        support.xaxis.set_major_locator(MaxNLocator(nbins=5))

    axes[0, 0].set_ylabel('Centered ALE (s/tire)')
    keys = [Line2D([], [], color=color, marker=marker, markersize=3, label=f'Fold {i + 1}')
            for i, (color, marker) in enumerate(zip(FOLD_COLORS, FOLD_MARKERS))]
    keys.append(Patch(facecolor=LOW_SUPPORT, label='Low-support bin'))
    fig.legend(handles=keys, loc='upper left', bbox_to_anchor=(0.018, 0.867),
               ncols=len(keys), columnspacing=1.4)
    fig.text(0.025, 0.034, 'Qualified bins have at least 50 supported proxies and 5 recipes. '
             'Each contiguous ALE segment is centered separately.', fontsize=7.7, color=MUTED)
    return fig


def chart_files(output, scores, shap, permutation, loco, ale, oof, nov):
    """Write the four publication panels at 300 DPI to the report's figures folder."""
    folder = output / 'figures'
    folder.mkdir(exist_ok=True)
    panels = (
        ('model_performance', model_performance, (scores, oof, nov)),
        ('treeshap_physical', treeshap_physical, (shap,)),
        ('module_attribution', module_attribution, (permutation, loco)),
        ('ale_supported_ranges', ale_supported_ranges, (ale,)),
    )
    with plt.rc_context(STYLE):
        for name, draw, data in panels:
            fig = draw(*data)
            fig.savefig(folder / f'{name}.png', dpi=300, facecolor='white')
            plt.close(fig)
