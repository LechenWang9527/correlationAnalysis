"""Verify Phase 2 artifacts and render the industrial report."""

import argparse
import ast
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from render_enhanced_figures import MODELS, chart_files
from step3_no_leakage import DEFAULT_OUTPUT, PRIMARY
from step3_no_leakage_attribution import ALE_COLUMNS, MIN_COVERAGE, cluster_mean_intervals, paired_intervals
from step3_no_leakage_data import A, B, MODULES, MONTH, RECIPE, ROOT, TARGET, digest
from step3_no_leakage_models import LOCO, metrics


MODULE_CN = {'geometry_bead': '几何与胎圈', 'liner': '内衬', 'carcass_reinforcement': '胎体与加强',
             'sidewall': '胎侧', 'belt_cap': '带束与冠带', 'tread': '胎面'}
ACTION = {
    'geometry_bead': '按轮辋直径与胎圈结构核对换型工装、装配间隙及机台节拍分布。',
    'liner': '用现行 BOM 核对内衬宽厚及断面，并在同规格、同能力设备上复核工序时间。',
    'carcass_reinforcement': '按胎体层数和胎圈加强状态检查上料、反包与换型环节的等待记录。',
    'sidewall': '核对胎侧 BOM 设定和型材重量的计量口径，再对齐供料与贴合周期。',
    'belt_cap': '核对带束宽度、角度和冠带绕制规格，对齐贴合与绕制的独立设备事件。',
    'tread': '按胎面组件规格核对供料、贴合与换型时间，并复核称重与 BOM 版本。',
}
STATUS = {'ELIGIBLE_PROXY': '可回放代理单', 'AMBIGUOUS_LABEL': '标签有冲突',
          'AMBIGUOUS_OUTCOME': '结算定义项有冲突', 'AMBIGUOUS_FEATURE': '物理特征有冲突',
          'AMBIGUOUS_EXECUTION': '实际时点有冲突', 'CROSS_BOUNDARY': '代理族跨边界',
          'NO_VALID_FORMULA_ROW': '无公式通过的同伴记录'}


def table(headers, rows):
    def cell(value):
        return str(value).replace('|', '\\|')

    lines = [headers, ['---'] * len(headers), *rows]
    return '\n'.join('| ' + ' | '.join(map(cell, row)) + ' |' for row in lines)


def number(value, digits=3):
    return f'{value:.{digits}f}' if pd.notna(value) else 'NA'


def verify(output):
    freeze_path = output / 'freeze_manifest.json'
    freeze = json.loads(freeze_path.read_text(encoding='utf-8'))
    seal = json.loads((output / 'replay_seal.json').read_text(encoding='utf-8'))
    audit = json.loads((output / 'provenance_summary.json').read_text(encoding='utf-8'))
    if digest(freeze_path) != seal['freeze_sha256'] or freeze['frozen_utc'] > seal['scored_utc']:
        raise ValueError('Replay freeze chronology or fingerprint failed')
    for file, checksum in (freeze['development_artifacts_sha256'] | seal['artifacts_sha256']).items():
        if digest(output / file) != checksum:
            raise ValueError(f'Artifact verification failed: {file}')
    for file, checksum in freeze['input_sha256'].items():
        if digest(ROOT / file) != checksum:
            raise ValueError(f'Input snapshot mismatch: {file}')
    for file, checksum in freeze['code_sha256'].items():
        current = ROOT / file
        if digest(current) == checksum:
            continue
        archived = ROOT / 'scratch/phase2_frozen_source' / current.name
        if current.name != 'step3_no_leakage_data.py' or not archived.exists() or digest(archived) != checksum:
            raise ValueError(f'Frozen source mismatch: {file}')
        for_audit = lambda node: getattr(node, 'name', None) != 'source_audit'
        before = ast.parse(archived.read_text(encoding='utf-8'))
        after = ast.parse(current.read_text(encoding='utf-8'))
        before.body = list(filter(for_audit, before.body))
        after.body = list(filter(for_audit, after.body))
        if ast.dump(before) != ast.dump(after):
            raise ValueError('Code changes after freeze reached beyond raw source audit')
    oof = pd.read_csv(output / 'development_oof.csv', dtype={RECIPE: str, 'family': str})
    nov = pd.read_csv(output / 'november_predictions.csv', dtype={RECIPE: str, 'family': str})
    identity = pd.read_csv(output / 'proxy_identity.csv', dtype={RECIPE: str, 'family': str})
    inner = pd.read_csv(output / 'inner_membership.csv', dtype={'outer': str})
    if len(oof) != freeze['n_development_eligible'] or len(nov) != seal['eligible_replay']:
        raise ValueError('Coverage mismatch')
    if not oof['proxy_id'].is_unique or not nov['proxy_id'].is_unique or set(oof['family']) & set(nov['family']):
        raise ValueError('Duplicate scored proxy or development/replay family crossover')
    if oof.groupby('family')['outer_fold'].nunique().max() != 1:
        raise ValueError('Development family appears in multiple outer folds')
    joined = inner.merge(oof[['proxy_id', 'family', 'outer_fold']], on='proxy_id', validate='many_to_one')
    if len(joined) != len(inner) or joined.groupby(['outer', 'family'])['inner_fold'].nunique().max() != 1:
        raise ValueError('Inner fold membership is not isolated')
    for fold, members in joined.groupby('outer'):
        expected = oof if fold == 'final' else oof.loc[oof['outer_fold'].ne(int(fold))]
        if set(members['proxy_id']) != set(expected['proxy_id']) or not members['proxy_id'].is_unique:
            raise ValueError(f'Invalid nested fit membership: {fold}')
    for group, columns in {'A': A, 'B': B}.items():
        header = pd.read_csv(output / f'X_{group}_proxy.csv', nrows=0).columns.tolist()
        if header != ['proxy_id', *columns] or len(identity) != audit['proxy_orders']:
            raise ValueError('Physical or pre-execution allowlist mismatch')
    if audit['raw_rows'] != audit['clean_rows'] + audit['raw_multi_matches']:
        raise ValueError('Source provenance conservation failed')
    if (audit['row_flags'].get('NEGATIVE_CLIPPED'), audit['row_flags'].get('SOURCE_ZERO'),
            audit['row_flags'].get('IDENTITY_RESIDUAL'), audit['historical_label_conflicts']) != (94, 2, 17, 8):
        raise ValueError('Unexpected results for the frozen historical snapshot')
    if identity['source_rows'].sum() != audit['clean_rows']:
        raise ValueError('Clean record count did not survive proxy aggregation')
    return freeze, seal, audit, oof, nov, identity


def shap_statistics(output, group, columns, reference):
    local = pd.read_csv(output / f'treeshap_{group}_oof.csv', dtype={'recipe': str})
    keys = local[['proxy_id', 'prediction']].merge(reference[['proxy_id', f'{group}_CatBoost']],
                                                  on='proxy_id', validate='one_to_one')
    if len(keys) != len(reference) or not np.allclose(keys['prediction'], keys[f'{group}_CatBoost']):
        raise ValueError('SHAP evaluation cohorts or predictions differ from OOF')
    residual = local[columns].sum(axis=1) + local['baseline'] - local['prediction']
    if not np.allclose(local['prediction'] + residual, local['prediction'], atol=1e-6, rtol=1e-8):
        raise ValueError('Local SHAP additivity failed')
    absolute = local[columns].abs()
    intervals = cluster_mean_intervals(absolute.to_numpy(), local['recipe'])
    folds = absolute.groupby(local['outer_fold']).mean()
    ranks = folds.rank(axis=1, ascending=False, method='min')
    summary = pd.DataFrame({'mean_abs': absolute.mean(), 'p50_abs': absolute.quantile(.5),
                            'p95_abs': absolute.quantile(.95), 'ci_low': intervals[0], 'ci_high': intervals[1],
                            'fold_mean_min': folds.min(), 'fold_mean_max': folds.max(),
                            'fold_rank_min': ranks.min(), 'fold_rank_max': ranks.max(),
                            'n': len(local), 'n_recipe_clusters': local['recipe'].nunique()})
    summary = summary.sort_values('mean_abs', ascending=False).rename_axis('feature').reset_index()
    summary.insert(0, 'rank', np.arange(1, len(summary) + 1))
    module_columns = MODULES if group == 'B' else MODULES | {'schedule': [c for c in A if c not in B]}
    grouped = pd.DataFrame({name: local[group_columns].sum(axis=1).abs()
                            for name, group_columns in module_columns.items()})
    module_ci = cluster_mean_intervals(grouped.to_numpy(), local['recipe'])
    modules = pd.DataFrame({'module': grouped.columns, 'mean_abs_signed_sum': grouped.mean().to_numpy(),
                            'ci_low': module_ci[0], 'ci_high': module_ci[1]})
    features = pd.read_csv(output / f'X_{group}_proxy.csv').set_index('proxy_id').loc[local['proxy_id']]
    baseline = []
    for fold, frame in local.groupby('outer_fold'):
        constant = features.loc[frame['proxy_id']].nunique().loc[lambda x: x.le(1)].index.tolist()
        baseline.append({'fold': fold, 'baseline_mean': frame['baseline'].mean(),
                         'additivity_max_abs': residual.loc[frame.index].abs().max(),
                         'constant_validation_features': ', '.join(constant)})
    pd.DataFrame(baseline).to_csv(output / f'treeshap_{group}_diagnostics.csv', index=False)
    summary.to_csv(output / f'treeshap_{group}_ranking.csv', index=False)
    modules.to_csv(output / f'treeshap_{group}_modules.csv', index=False)
    return summary, float(residual.abs().max())


def paired_predictions(data, before, after):
    y = data[TARGET].to_numpy()
    e1, e2 = data[before].to_numpy() - y, data[after].to_numpy() - y
    m1, m2 = metrics(y, data[before]), metrics(y, data[after])
    return {'n': len(data), 'n_recipe_clusters': data[RECIPE].nunique(),
            **{f'delta_{key}': m2[key] - m1[key] for key in ['mae', 'rmse', 'r2']},
            **paired_intervals(y, np.abs(e1), np.abs(e2), e1 ** 2, e2 ** 2, data[RECIPE])}


def loco_statistics(output, oof, nov):
    comparisons = [('B_CatBoost', name) for name in LOCO] + [('B_CatBoost', 'A_CatBoost'),
                   ('Schedule_CatBoost', 'A_CatBoost'), ('A_CatBoost', 'A_no_month_CatBoost'),
                   ('A_Ridge', 'A_CatBoost'), ('B_Ridge', 'B_CatBoost')]
    rows = []
    for before, after in comparisons:
        for scope, data in [('development_oof', oof), *[(f'fold_{f + 1}', subset)
                                                        for f, subset in oof.groupby('outer_fold')]]:
            rows.append({'scope': scope, 'before': before, 'after': after, **paired_predictions(data, before, after)})
        if after in PRIMARY:
            rows.append({'scope': 'november_replay', 'before': before, 'after': after,
                         **paired_predictions(nov, before, after)})
    result = pd.DataFrame(rows)
    result.to_csv(output / 'loco_and_paired_comparisons.csv', index=False)
    return result


def permutation_statistics(output):
    support = pd.read_csv(output / 'permutation_support.csv')
    pairs = pd.read_csv(output / 'permutation_paired.csv', dtype={RECIPE: str})
    repeats = pd.read_csv(output / 'permutation_repetitions.csv')
    rows = []
    for module, info in support.groupby('module'):
        sample = pairs.loc[pairs['module'].eq(module)]
        n, count = len(sample), int(info['n_target'].sum())
        if n != int(info['n_supported'].sum()) or not sample['proxy_id'].is_unique:
            raise ValueError(f'Inconsistent conditional permutation coverage: {module}')
        row = {'module': module, 'n_supported': n, 'n_target': count, 'coverage': n / count,
               'min_fold_coverage': info['coverage'].min(), 'n_recipe_clusters': sample[RECIPE].nunique(),
               'status': 'SUPPORTED' if n / count >= MIN_COVERAGE else 'SUPPORTED_SUBGROUP_ONLY'}
        if n:
            y = sample['y'].to_numpy()
            variance = y.var()
            before, after = sample['mse_before'].mean(), sample['mse_after'].mean()
            random_draws = repeats.loc[repeats['module'].eq(module)].copy()
            supported_folds = info.loc[info['n_supported'].gt(0), 'fold']
            counts = random_draws.groupby('fold')['repeat'].nunique()
            if not counts.reindex(supported_folds).eq(50).all():
                raise ValueError(f'Incomplete permutation repetitions: {module}')
            random_draws['weighted_mae'] = random_draws['n'] * random_draws['delta_mae']
            draw_means = random_draws.groupby('repeat')['weighted_mae'].sum() / n
            row.update(delta_mae=(sample['mae_after'] - sample['mae_before']).mean(),
                       delta_rmse=np.sqrt(after) - np.sqrt(before), delta_r2=(before - after) / variance,
                       monte_carlo_se_mae=draw_means.std() / np.sqrt(len(draw_means)),
                       **paired_intervals(y, sample['mae_before'], sample['mae_after'],
                                          sample['mse_before'], sample['mse_after'], sample[RECIPE]))
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(output / 'permutation_module_summary.csv', index=False)
    return result


def results_text(output, report, freeze, seal, audit, scores, shap, shap_a, shap_b, permutation, loco, ale, oof, nov):
    cohort = pd.read_csv(output / 'cohort_audit.csv')
    coverage = []
    for month, items in cohort.groupby(MONTH):
        good = items['status'].eq('ELIGIBLE_PROXY')
        total, eligible = items['proxies'].sum(), items.loc[good, 'proxies'].sum()
        exclusions = '，'.join(f'{STATUS[s]} {n:,}' for s, n in zip(items.loc[~good, 'status'],
                                                              items.loc[~good, 'proxies'])) or '0'
        coverage.append([month, f'{items["rows"].sum():,}', f'{total:,}', f'{eligible:,}',
                         f'{eligible / total:.2%}', exclusions])
    main = scores.loc[scores['scope'].isin(['development_oof', 'november_replay'])
                      & scores['model'].isin([*PRIMARY, 'Median', 'Mean'])]
    metric_table = []
    for name in main['model'].unique():
        pair = main.loc[main['model'].eq(name)].set_index('scope')
        metric_table.append([name, *(number(pair.loc[scope, statistic])
                              for scope in ('development_oof', 'november_replay')
                              for statistic in ('mae', 'rmse', 'r2', 'bias'))])
    folds = []
    for fold, items in oof.groupby('outer_fold'):
        counts = items[MONTH].value_counts()
        folds.append([fold + 1, len(items), items['family'].nunique(), *(counts.get(m, 0) for m in (8, 9, 10))])
    top = [[r['rank'], r['feature'], number(r['mean_abs']),
            f'{number(r["ci_low"])} 至 {number(r["ci_high"])}',
            f'{int(r["fold_rank_min"])} 至 {int(r["fold_rank_max"])}']
           for _, r in shap.head(10).iterrows()]
    p_rows = []
    for _, r in permutation.sort_values('delta_mae', ascending=False).iterrows():
        p_rows.append([MODULE_CN[r['module']], f'{r["n_supported"]:,}', f'{r["coverage"]:.1%}',
                       number(r['delta_mae']), f'{number(r["mae_lo"])} 至 {number(r["mae_hi"])}',
                       number(r['delta_rmse']), number(r['delta_r2']), number(r['monte_carlo_se_mae'], 4),
                       r['status']])
    l_rows, l_main = [], loco.loc[loco['scope'].eq('development_oof')].copy()
    l_main['module'] = l_main['after'].str.removeprefix('LOCO_')
    l_ranked = l_main.loc[l_main['after'].str.startswith('LOCO_')].sort_values('delta_mae', ascending=False)
    for _, r in l_ranked.iterrows():
        changes = loco.loc[loco['scope'].str.startswith('fold_') & loco['after'].eq(r['after']), 'delta_mae']
        l_rows.append([MODULE_CN[r['module']], number(r['delta_mae']),
                       f'{number(r["mae_lo"])} 至 {number(r["mae_hi"])}',
                       number(r['delta_rmse']), number(r['delta_r2']),
                       f'{number(changes.min())} 至 {number(changes.max())}'])
    contrasts = []
    for _, r in loco.loc[~loco['after'].str.startswith('LOCO_') & ~loco['scope'].str.startswith('fold_')].iterrows():
        contrasts.append([r['scope'], r['before'], r['after'], number(r['delta_mae']),
                          f'{number(r["mae_lo"])} 至 {number(r["mae_hi"])}',
                          number(r['delta_rmse']), number(r['delta_r2'])])
    a_rows = []
    for column in ALE_COLUMNS:
        selected = ale.loc[ale['feature'].eq(column)]
        good = selected.loc[selected['status'].eq('SUPPORTED')]
        segments = [f'折 {fold + 1} [{number(segment["low"].min(), 2)}, {number(segment["high"].max(), 2)}]'
                    for (fold, _), segment in good.groupby(['fold', 'segment'])]
        a_rows.append([column, good['fold'].nunique(), len(good), int(good['n_supported'].sum()),
                       '，'.join(segments) or 'NA'])

    unseen = []
    for value, cohort_nov in nov.groupby('unseen_recipe'):
        if len(cohort_nov) >= 30:
            unseen.append(['未见配方' if value else '已见配方', len(cohort_nov),
                           *(number(metrics(cohort_nov[TARGET], cohort_nov[name])['mae'])
                             for name in ('A_CatBoost', 'B_CatBoost'))])
    suspects = scores.loc[scores['scope'].isin(['development_anomaly_0', 'development_anomaly_1'])
                         & scores['model'].isin(['A_CatBoost', 'B_CatBoost'])]
    sensitivity = [[r['scope'], int(r['n']), r['model'], number(r['mae']), number(r['rmse']), number(r['r2'])]
                   for _, r in suspects.iterrows()]
    diagnostics = pd.read_csv(output / 'development_sensitivity_metrics.csv')
    sensitivity_names = {'recipe_cold_start': '配方整簇冷启动', 'exclude_anomaly_refit': '排除异常组后重训'}
    diagnostics = diagnostics.loc[diagnostics['scope'].isin(sensitivity_names)]
    diagnostic_rows = [[sensitivity_names[r['scope']], int(r['n']), r['model'], number(r['mae']),
                        number(r['rmse']), number(r['r2'])] for _, r in diagnostics.iterrows()]
    a_vs_b = loco.loc[loco['scope'].eq('november_replay') & loco['after'].eq('A_CatBoost')
                      & loco['before'].eq('B_CatBoost'), 'delta_mae'].iloc[0]
    relative = Path(os.path.relpath(output, report.parent)).as_posix()

    sections = [f'''# 无泄露工单级重构与真实物理归因报告

## 1. 结论与适用范围

[PASS] 按审定白名单完成 CSV 快照级特征净化、代理工单聚合、开发期嵌套交叉验证、11 月冻结回放和四项预测归因。全量保留 **{audit['proxy_orders']:,}** 个候选组、**{len(oof):,}** 个开发期可评分代理单和 **{len(nov):,}** 个 11 月可评分代理单。标签为源表 `ActualVsIEGap`，误差单位为秒/胎。每个可评分代理单权重相同。

11 月回放中，A 组 CatBoost 的 MAE 比 B 组低 **{-a_vs_b:.3f}** 秒/胎。条件置换覆盖最高 **{permutation['coverage'].max():.1%}**，六个模块均只达到子群发布条件。内衬厚度的 ALE 在五折有支撑，响应方向和幅度随折变化。

[BLOCKED] MES 原生执行 ID、父单谱系、实际排产发布时间、PLM/BOM 历史版本、闭单可见时间和独立设备循环事件均未包含在两份 CSV。生产工单唯一性、真实工单族隔离、t0 时点一致性、开发期滚动验证、独立评分及生产发布依赖这些材料。以下结果反映该快照上的代理工单回放和预测归因。旧流程使用过 11 月标签，历史首轮盲测资格无法恢复。本次依次执行开发冻结和独立评分进程，进程由同一分析方运行。

## 2. 身份、血缘与异常裁定

- 代理主键为配方、现有工作中心、计划月/星期/小时和计划批量，按类型化元组计算 SHA-256。工作中心的计划属性尚待确认，仅参与对账和相容条件。实际起点只参与代理族连边、跨边界隔离和审计。代理族是共享同一配方、工作中心、计划批量和实际月/星期/小时的计划键传递闭包，共 {audit['proxy_families']:,} 族。它只能隔离可观察的重复或修订候选。
- 原始导出 {audit['raw_rows']:,} 行全部映射到 {audit['clean_rows']:,} 条清洗记录，余下 {audit['raw_multi_matches']:,} 条是多条源行映射到同一清洗记录的血缘差额。按导出顺序定位原始行和清洗行。原值、源文件校验值、来源序号和转换后的状态全部可追溯。
- **{audit['historical_groups']:,}** 为含实际起点、完成量的历史分组数，专用于旧口径对账。该口径有 8 组、{audit['historical_conflict_rows']} 行的 Y 冲突。候选计划键口径实际隔离 {audit['proxy_label_conflicts']} 个 Y 冲突组，对该键下的重复记录要求结算四项一致。全体清洗行 Q 合计 {audit['raw_quantity_sum_after_cleaning']:,}，历史分组各计一次 Q 为 {audit['historical_quantity_once']:,}，供现场核对重复量。
- 96 条零覆盖记录溯源为 94 条 `NEGATIVE_CLIPPED` 和 2 条 `SOURCE_ZERO`。17 条 `IDENTITY_RESIDUAL` 的绝对残差至少为 0.0001 秒/胎，两类合计涉及 {audit['anomaly_historical_groups']} 个历史分组。公式对账按 `O/(C×Q) − (IE+TOL)/100 − Y` 计算。异常行保留原始标签及残差，运行秒数代理留空。该组有通过公式的同伴且结算一致时，允许进入快照回放。待 MES 按执行与闭单版本裁定。
- 计划月与实际月不同的源记录有 {audit['actual_vs_planned_month_rows']} 行。代理族或该组计划与实际落在开发/回放不同侧的候选共 {audit['proxy_cross_boundary']} 个，评分集合间的代理族交集为 {seal['eligible_family_overlap']}。下表每组只记一个主要隔离原因，完整冲突标志见审计表。覆盖率分母为快照中该计划月所有候选键。

{table(['计划月', '清洗记录', '候选组', '可评分', '覆盖率', '隔离原因'], coverage)}

## 3. 字段与验证协议

- B 含 42 个登记的物理设计与工艺字段，A 含 B 和 4 项计划字段。O、C、Q、IE、TOL、Y、实际时间与人员等事后字段全部隔离。配方、机台、工装、PTC 和胶囊代码留在模型输入之外。试验使用现有清洗快照，继承上游全表删列与去重口径。入模列由方案固定，折内词表和数值转换由模型训练分区估计。
- 数字形态的胶料代号、反包结构、胎圈和冠带构造按名义属性处理。胎圈 5x6 与 6x5 的原结构单列保留在原始血缘。胎侧 BOM 数值按快照规格保留，单位待工程确认。二层帘布与加强件的结构性缺件保留同步编码。
- 开发期使用按计划月平衡的 5 个代理族外折，每个外折内用 3 个代理族折按 MAE 选参。CatBoost 从两套预设迭代数和深度中选择，固定迭代并关闭早停。Ridge 折内进行中位数插补、数值缩放、名义独热编码和正则调参。基准为折训练中位数与均值。A、B、排产单独、A 去计划月和 LOCO 共享外折与评价标签。
- 开发冻结（UTC）`{freeze['frozen_utc']}`。11 月批量评分封存（UTC）`{seal['scored_utc']}`。冻结清单记录源、代码、模型、阈值、折成员与产物校验值。评分封存后，原始血缘函数修正了清洗表分类字段的读取类型，冻结源码副本保存在 `scratch/phase2_frozen_source`。验收核对源码差异限于该审计函数。11 月计划月份为训练未见类别，另检查去月敏感性。历史发布时间和闭单时间缺失，日期块与结算滞后无法复原。

{table(['外折', '评价代理单', '代理族', '8 月', '9 月', '10 月'], folds)}

## 4. 模型表现

误差按外折预测合并及 11 月冻结预测分别计算。偏差为预测减观测的均值。下表前四列指标对应 8–10 月，后四列对应 11 月。

{table(['模型', '开发 MAE', '开发 RMSE', '开发 R²', '开发偏差', '11 月 MAE', '11 月 RMSE', '11 月 R²', '11 月偏差'], metric_table)}

![开发期与 11 月模型指标]({relative}/figures/model_performance.png)

11 月有 {seal['unseen_recipe']} 个未见配方代理单、{seal['unseen_center']} 个未见工作中心代理单、{seal['unseen_physical_category']} 个包含未见物理类别的代理单。分层指标仅在 n ≥ 30 时发布。

{table(['配方层', '代理单', 'A CatBoost MAE', 'B CatBoost MAE'], unseen)}

公式异常先按固定外折预测分层，`0` 表示全部行通过，`1` 表示组含异常操作工行。

{table(['开发分层', '代理单', '模型', 'MAE', 'RMSE', 'R²'], sensitivity)}

开发期补充验证使用相同模型搜索预算。冷启动将完整 RecipeID 分到同一外折与内折。异常重训沿用主分析的外折评价集，从各训练分区去除含异常操作工行的组，并重新调参。

{table(['开发补充验证', '代理单', '模型', 'MAE', 'RMSE', 'R²'], diagnostic_rows)}

## 5. 物理预测归因

所有归因取开发期外折。95% 区间采用 1,000 次 RecipeID 簇重采样，描述固定外折模型下评价样本波动。配方只作可观察簇代理，生产日、产线状态和真实生产簇引起的相关性尚未量化。签核前应以 MES 生产簇重新估计。

### 5.1 TreeSHAP

CatBoost 原生树路径解释使用已训练模型的内部背景。每个代理单仅由其外折 CatBoost 解释，统一用秒/胎。加性最大误差为 A **{shap_a:.3g}**、B **{shap_b:.3g}** 秒/胎，均通过阈值检查。B 的平均绝对贡献排名如下。模块 SHAP 单列发布，模块内先求有符号和再取绝对值。基线和验证折常量见诊断表。

{table(['排名', '物理字段', '平均绝对 SHAP', '配方簇区间', '折内排名范围'], top)}

![B 组物理 TreeSHAP]({relative}/figures/treeshap_physical.png)

### 5.2 模块条件置换

从外折训练区抽取不同代理单的完整模块，要求工作中心和轮辋直径相同。模块外物理项使用训练折缩放的混合型 Gower 距离，半径 0.14，每条至少 20 个值有变化的训练供体，按距离和候选供体数抽样。已实施的硬约束为二层胎体和胎圈加强状态一致性。几何与胎圈的置换结果以轮辋直径固定为条件。车间设备资格、工程公差和全部 BOM 联合约束待现场确认。每个支持代理单做 50 次扰动，对同一代理单配对计损失，RMSE 由期望平方损失转换。覆盖低于 80% 时仅发布支撑子群。全部模块均保留分母。

{table(['模块', '支撑代理单', '覆盖', 'ΔMAE', '配方簇区间', 'ΔRMSE', 'ΔR²', 'MAE 抽样标准误', '状态'], p_rows)}

### 5.3 模块 LOCO

对 B CatBoost 逐个剔除物理模块，保持与全模型相同的 5 个评价折、训练成员和超参搜索预算，各次重新拟合。Δ 为消融减去 B 全模型。误差增量为负时，移除该模块后预测误差下降。配方簇区间涵盖零时需关注折间稳定性。

{table(['模块', 'ΔMAE', '配方簇区间', 'ΔRMSE', 'ΔR²', '各折 ΔMAE 范围'], l_rows)}

![条件置换与 LOCO]({relative}/figures/module_attribution.png)

下表差值为后模型减前模型。

{table(['队列', '前模型', '后模型', 'ΔMAE', '配方簇区间', 'ΔRMSE', 'ΔR²'], contrasts)}

### 5.4 有支撑的 1D-ALE

每折从训练 p5–p95 取最近观测分位点，最多 10 个箱，训练稀疏箱合并。每箱的两个端点均需存在同工作中心、同轮辋直径且其余物理特征在距离门槛内的训练供体。每箱至少 50 个可行验证代理单和 5 个配方簇代理。只有满足门槛的连续箱在各自折内累计、独立居中。曲线断点保留，横轴为源表规格值。低支撑箱及训练端点供体量见 `ale_bins.csv`。模型响应受跨配方关联与历史采集口径影响。

{table(['变量', '通过折数', '通过箱数', '支撑代理单计数', '各折连续支撑区间'], a_rows)}

![分折 ALE 支撑区间]({relative}/figures/ale_supported_ranges.png)

## 6. 现场核验建议

''']
    review = set(l_ranked.loc[l_ranked['mae_lo'].gt(0), 'module'])
    review.update(permutation.loc[permutation['mae_lo'].gt(0), 'module'])
    priorities = [m for m in l_ranked['module'] if m in review][:3]
    if not priorities:
        priorities = [next(module for module, columns in MODULES.items() if shap.iloc[0]['feature'] in columns)]
    sections.extend(f'- {ACTION[module]}' for module in priorities)
    sections.append(f'''- MES 优先裁定 8 个历史标签冲突组、8 条跨计划/实际月份记录和 113 条异常行，补齐执行 ID、父单映射、历史排产与 BOM 版本及闭单时间。用机台独立循环事件核验节拍，在同配方、同设备且经工程批准的可行范围内评估调整。
- SAGE 按本阶段范围暂缓。物理 DML 保持 [BLOCKED]，待独立节拍、受控工艺变更、共同支持、可比工单族和混杂识别证据到位。

## 7. 复算与交付索引

使用 Python {freeze['python'].split()[0]} 和仓库 `scratch/step3_phase2_requirements.txt` 的锁定依赖。按以下顺序执行。向 `--output` 指定一个新的目录可重跑开发，受封存保护的回放目录只能评分一次。默认目录为 `reports/step3_phase2`。

```text
python -m unittest discover -s src -p test_step3_no_leakage.py
python src/step3_no_leakage.py develop --output reports/step3_phase2_reproduction
python src/step3_no_leakage.py score --output reports/step3_phase2_reproduction
python src/step3_no_leakage.py audit --output reports/step3_phase2_reproduction
python src/diagnostics_step3_phase2.py --output reports/step3_phase2_reproduction
python src/render_step3_phase2.py --output reports/step3_phase2_reproduction --report reports/step3_phase2_reproduction.md
```

| 核验内容 | 文件 |
| --- | --- |
| 字段白名单及代理身份 | [A 组特征]({relative}/X_A_proxy.csv)、[B 组特征]({relative}/X_B_proxy.csv)、[身份与折成员]({relative}/proxy_identity.csv) |
| 源行、异常与结算 | [原始行血缘]({relative}/raw_lineage.csv)、[清洗行结果审计]({relative}/clean_lineage_outcome_audit.csv)、[历史分组对账]({relative}/historical_groups_audit.csv)、[月度隔离]({relative}/cohort_audit.csv) |
| 验证与冻结 | [开发指标]({relative}/development_metrics.csv)、[回放指标]({relative}/november_metrics.csv)、[外折预测]({relative}/development_oof.csv)、[回放预测]({relative}/november_predictions.csv)、[拟合记录]({relative}/nested_fit_log.csv)、[冻结清单]({relative}/freeze_manifest.json)、[评分封存]({relative}/replay_seal.json) |
| 开发补充验证 | [敏感性指标]({relative}/development_sensitivity_metrics.csv)、[敏感性外折预测]({relative}/development_sensitivity_oof.csv) |
| TreeSHAP | [A 组排名]({relative}/treeshap_A_ranking.csv)、[B 组排名]({relative}/treeshap_B_ranking.csv)、[B 组模块归因]({relative}/treeshap_B_modules.csv) |
| 模块重要性与响应 | [置换汇总]({relative}/permutation_module_summary.csv)、[供体及覆盖]({relative}/permutation_support.csv)、[50 次重复]({relative}/permutation_repetitions.csv)、[LOCO 和配对比较]({relative}/loco_and_paired_comparisons.csv)、[ALE 逐箱]({relative}/ale_bins.csv) |
''')
    return '\n'.join(sections)


def render(output, report):
    freeze, seal, audit, oof, nov, _ = verify(output)
    _, error_a = shap_statistics(output, 'A', A, oof)
    shap, error_b = shap_statistics(output, 'B', B, oof)
    permutation = permutation_statistics(output)
    loco = loco_statistics(output, oof, nov)
    ale = pd.read_csv(output / 'ale_bins.csv')
    qualified = ale.loc[ale['status'].eq('SUPPORTED')]
    if qualified.empty or not (qualified['n_supported'].ge(50) & qualified['n_recipes'].ge(5)
                                & qualified['min_low_donors'].ge(1) & qualified['min_high_donors'].ge(1)).all():
        raise ValueError('ALE support integrity failed')
    scores = pd.concat([pd.read_csv(output / name) for name in ['development_metrics.csv', 'november_metrics.csv']])
    for frame, scope in [(oof, 'development_oof'), (nov, 'november_replay')]:
        for name in MODELS:
            expected = scores.loc[scores['scope'].eq(scope) & scores['model'].eq(name)].iloc[0]
            observed = metrics(frame[TARGET], frame[name])
            if not np.allclose(list(observed.values()), expected[list(observed)].astype(float)):
                raise ValueError(f'Metric reconciliation failed: {scope}, {name}')
    chart_files(output, scores, shap, permutation, loco, ale, oof, nov)
    report.write_text(results_text(output.resolve(), report.resolve(), freeze, seal, audit, scores, shap, error_a, error_b,
                                   permutation, loco, ale, oof, nov), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--report', type=Path, default=ROOT / 'reports/step3_no_leakage_attribution_report.md')
    args = parser.parse_args()
    render(args.output, args.report)


if __name__ == '__main__':
    main()
