"""Develop, freeze, score, and trace the Phase 2 snapshot analysis."""

import argparse
import json
import logging
import pickle
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from step3_no_leakage_attribution import (
    ALE_COLUMNS, BOOTSTRAPS, MAX_DISTANCE, MIN_COVERAGE, MIN_DONORS, REPEATS,
    ale, conditional_permutation, legal_carcass,
)
from step3_no_leakage_data import (
    A, ACTUAL, B, CENTER, MODULES, MONTH, NOMINAL, OUTCOME, RECIPE, ROOT, SEED, SNAPSHOTS,
    TARGET, assign_folds, build_proxy, digest, label_proxies, read_inputs, read_outcomes, source_audit,
)
from step3_no_leakage_models import (
    CAT_OPTIONS, MODELS, PRIMARY, RIDGE_OPTIONS, baseline, performance, predict, save_bundle,
    tree_shap, tune_fit,
)


DEFAULT_OUTPUT = ROOT / 'reports/step3_phase2'
DEVELOPMENT = [8, 9, 10]
REPLAY = [11]
BASELINES = ['Median', 'Mean']
FEATURES = {'A': A, 'B': B}
LOGGER = logging.getLogger(__name__)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def model_code():
    return sorted(Path(__file__).resolve().parent.glob('step3_no_leakage*.py'))


def validate_eligible(frame):
    if not frame.index.is_unique or not legal_carcass(frame).all():
        raise ValueError('Invalid proxy uniqueness or carcass structure')
    if not np.isfinite(frame[[*OUTCOME, *[c for c in A if c not in NOMINAL]]]).all().all():
        raise ValueError('Non-finite eligible outcome or predictor')


def apply_models(names, train, valid, inner, audits, fold, predictions):
    fitted = {}
    for name in names:
        engine, columns = MODELS[name]
        fitted[name] = tune_fit(name, engine, columns, train, inner, audits, fold)
        if valid is not None:
            predictions[name] = predict(fitted[name], valid, columns)
    return fitted


def develop(output):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Choose an empty run directory: {output}')
    output.mkdir(parents=True, exist_ok=True)
    inputs = read_inputs()
    orders, source_ids = build_proxy(inputs)
    development = orders.loc[orders[MONTH].isin(DEVELOPMENT)]
    all_folds = assign_folds(development, 5, SEED)
    results = label_proxies(development, read_outcomes(source_ids, DEVELOPMENT))
    eligible = results.loc[results['status'].eq('ELIGIBLE_PROXY')]
    validate_eligible(eligible)
    for group, columns in FEATURES.items():
        orders[columns].to_csv(output / f'X_{group}_proxy.csv')
    meta = orders[[RECIPE, CENTER, MONTH, 'family', 'cross_boundary', 'feature_conflict',
                   'actual_time_conflict', 'source_rows']].copy()
    meta['outer_fold'] = all_folds
    meta.to_csv(output / 'proxy_identity.csv')
    results[[*OUTCOME, 'formula_pass_rows', 'anomaly_rows', 'label_conflict',
             'outcome_conflict', 'status']].to_csv(output / 'development_labels_audit.csv')
    LOGGER.info('Snapshot contains %s keys and %s eligible development proxies', len(orders), len(eligible))
    audits, membership, statistics, oof = [], [], [], []
    shap = {'A': [], 'B': []}
    permutation, permutation_losses, permutation_repeats, ale_bins = [], [], [], []
    eligible_folds = all_folds.loc[eligible.index]

    for fold in range(5):
        train = eligible.loc[eligible_folds.ne(fold)]
        valid = eligible.loc[eligible_folds.eq(fold)]
        if set(train['family']) & set(valid['family']):
            raise ValueError('Outer family crossover')
        LOGGER.info('Outer fold %s started: %s training, %s evaluation', fold, len(train), len(valid))
        predicted = valid[[TARGET, RECIPE, CENTER, MONTH, 'family', 'anomaly_rows']].copy()
        predicted['outer_fold'] = fold
        baseline(train, predicted)
        inner = assign_folds(train, 3, SEED + fold)
        membership.append(inner.rename('inner_fold').reset_index().assign(outer=fold))
        fitted = apply_models(MODELS, train, valid, inner, audits, fold, predicted)
        for group, columns in FEATURES.items():
            name = f'{group}_CatBoost'
            local, discrepancy = tree_shap(fitted[name], valid, columns, predicted[name].to_numpy())
            local = local.assign(outer_fold=fold, additivity_max_abs=discrepancy,
                                 recipe=valid[RECIPE], prediction=predicted[name])
            shap[group].append(local)
        conditional, losses, repeats = conditional_permutation(
            fitted['B_CatBoost'], train, valid, fold, predicted['B_CatBoost'].to_numpy(),
        )
        permutation.extend(conditional)
        permutation_losses.extend(losses)
        permutation_repeats.extend(repeats)
        ale_bins.extend(ale(fitted['B_CatBoost'], train, valid, fold))
        statistics.extend(performance(predicted, [*MODELS, *BASELINES], f'fold_{fold + 1}'))
        oof.append(predicted)
        LOGGER.info('Outer fold %s completed', fold)

    predictions = pd.concat(oof).sort_index()
    if not predictions.index.equals(eligible.index):
        raise ValueError('Missing or repeated OOF predictions')
    predictions.to_csv(output / 'development_oof.csv')
    for group, values in shap.items():
        pd.concat(values).sort_index().to_csv(output / f'treeshap_{group}_oof.csv')
    pd.DataFrame(permutation).to_csv(output / 'permutation_support.csv', index=False)
    pd.concat(permutation_losses, ignore_index=True).to_csv(output / 'permutation_paired.csv', index=False)
    pd.DataFrame(permutation_repeats).to_csv(output / 'permutation_repetitions.csv', index=False)
    pd.DataFrame(ale_bins).to_csv(output / 'ale_bins.csv', index=False)
    statistics.extend(performance(predictions, [*MODELS, *BASELINES], 'development_oof'))
    for flag, subgroup in predictions.groupby(predictions['anomaly_rows'].gt(0)):
        statistics.extend(performance(subgroup, list(PRIMARY), f'development_anomaly_{int(flag)}'))
    pd.DataFrame(statistics).to_csv(output / 'development_metrics.csv', index=False)

    LOGGER.info('Refitting full development models for freeze')
    inner = assign_folds(eligible, 3, SEED + 100)
    membership.append(inner.rename('inner_fold').reset_index().assign(outer='final'))
    fitted = apply_models(PRIMARY, eligible, None, inner, audits, 'final', None)
    pd.concat(membership).to_csv(output / 'inner_membership.csv', index=False)
    pd.DataFrame(audits).to_csv(output / 'nested_fit_log.csv', index=False)
    bundle = {'models': fitted, 'mean': eligible[TARGET].mean(), 'median': eligible[TARGET].median(),
              'recipe': set(eligible[RECIPE]), 'center': set(eligible[CENTER]),
              'family': set(eligible['family']),
              'column_cardinality': {c: sorted(eligible[c].unique().tolist()) for c in A if c in NOMINAL}}
    save_bundle(output / 'frozen_models.pkl', bundle)
    manifest = {
        'frozen_utc': utc_now(), 'seed': SEED, 'python': sys.version,
        'packages': {name: version(name) for name in ['numpy', 'pandas', 'scipy', 'scikit-learn',
                                                      'catboost', 'matplotlib']},
        'input_sha256': {p.relative_to(ROOT).as_posix(): digest(p) for p in SNAPSHOTS},
        'code_sha256': {p.relative_to(ROOT).as_posix(): digest(p) for p in model_code()},
        'plan_sha256': digest(ROOT / 'scratch/codex_no_leakage_plan.md'),
        'development_artifacts_sha256': {p.name: digest(p) for p in output.iterdir() if p.is_file()},
        'features': FEATURES, 'nominal': sorted(NOMINAL), 'physical_modules': MODULES,
        'catboost_options': CAT_OPTIONS, 'ridge_options': RIDGE_OPTIONS,
        'outer_folds': 5, 'inner_folds': 3, 'thresholds': {'identity_residual': 1e-4,
            'shap_atol': 1e-6, 'shap_rtol': 1e-8, 'conditional_donor_work_orders': MIN_DONORS,
            'conditional_min_coverage': MIN_COVERAGE, 'conditional_repetitions': REPEATS,
            'gower_radius': MAX_DISTANCE, 'bootstrap_repetitions': BOOTSTRAPS,
            'ale_features': ALE_COLUMNS, 'ale_min_orders': 50, 'ale_min_recipe_proxies': 5},
        'n_development_eligible': len(eligible), 'n_all_proxy_ids': len(orders),
    }
    write_json(output / 'freeze_manifest.json', manifest)
    LOGGER.info('Development frozen')


def score(output):
    seal = output / 'replay_seal.json'
    if seal.exists():
        raise FileExistsError('The frozen replay has already been scored')
    manifest_path = output / 'freeze_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    for file, checksum in manifest['code_sha256'].items():
        if digest(ROOT / file) != checksum:
            raise ValueError(f'Code changed after freeze: {file}')
    for file, checksum in manifest['development_artifacts_sha256'].items():
        if digest(output / file) != checksum:
            raise ValueError(f'Artifact changed after freeze: {file}')
    inputs = read_inputs()
    orders, source_ids = build_proxy(inputs)
    with (output / 'frozen_models.pkl').open('rb') as handle:
        bundle = pickle.load(handle)
    holdout = orders.loc[orders[MONTH].isin(REPLAY)]
    source = read_outcomes(source_ids, REPLAY)
    cohort = label_proxies(holdout, source)
    eligible = cohort.loc[cohort['status'].eq('ELIGIBLE_PROXY')]
    validate_eligible(eligible)
    if set(eligible['family']) & bundle['family']:
        raise ValueError('Replay family crossover')
    predictions = eligible[[TARGET, RECIPE, CENTER, MONTH, 'family', 'anomaly_rows']].copy()
    predictions['Median'] = bundle['median']
    predictions['Mean'] = bundle['mean']
    for name, (_, columns) in PRIMARY.items():
        predictions[name] = predict(bundle['models'][name], eligible, columns)
    predictions['unseen_recipe'] = ~eligible[RECIPE].isin(bundle['recipe'])
    predictions['unseen_center'] = ~eligible[CENTER].isin(bundle['center'])
    predictions['unseen_physical_category'] = pd.DataFrame({
        c: ~eligible[c].isin(bundle['column_cardinality'][c]) for c in B if c in NOMINAL
    }).any(axis=1)
    predictions.to_csv(output / 'november_predictions.csv')
    cohort[[*OUTCOME, 'formula_pass_rows', 'anomaly_rows', 'label_conflict',
            'outcome_conflict', 'status']].to_csv(output / 'november_labels_audit.csv')
    metrics = performance(predictions, [*PRIMARY, *BASELINES], 'november_replay')
    for flag in ['unseen_recipe', 'unseen_center', 'unseen_physical_category']:
        for value, subgroup in predictions.groupby(flag):
            if len(subgroup) >= 30:
                metrics.extend(performance(subgroup, list(PRIMARY), f'november_{flag}_{int(value)}'))
    pd.DataFrame(metrics).to_csv(output / 'november_metrics.csv', index=False)
    write_json(seal, {'scored_utc': utc_now(), 'freeze_sha256': digest(manifest_path),
                      'eligible_replay': len(eligible), 'eligible_family_overlap': 0,
                      'unseen_recipe': int(predictions['unseen_recipe'].sum()),
                      'unseen_center': int(predictions['unseen_center'].sum()),
                      'unseen_physical_category': int(predictions['unseen_physical_category'].sum()),
                      'unseen_planned_month': 11 not in bundle['column_cardinality'][MONTH],
                      'artifacts_sha256': {p.name: digest(p) for p in output.glob('november_*.csv')}})
    LOGGER.info('November replay sealed')


def audit(output):
    if not (output / 'replay_seal.json').exists():
        raise ValueError('Replay must be sealed before the full snapshot audit')
    inputs = read_inputs()
    orders, ids = build_proxy(inputs)
    source = read_outcomes(ids, [*DEVELOPMENT, *REPLAY])
    summary = source_audit(inputs, ids, source, output)
    outcomes = label_proxies(orders, source)
    summary.update(proxy_orders=len(orders), proxy_families=orders['family'].nunique(),
                   proxy_cross_boundary=int(orders['cross_boundary'].sum()),
                   actual_vs_planned_month_rows=int(inputs[MONTH].ne(inputs[ACTUAL[-1]]).sum()),
                   proxy_label_conflicts=int(outcomes['label_conflict'].sum()))
    write_json(output / 'provenance_summary.json', summary)
    counts = outcomes.groupby([MONTH, 'status']).agg(
        proxies=(RECIPE, 'size'), rows=('source_rows', 'sum'), abnormal_rows=('anomaly_rows', 'sum'),
    )
    counts.to_csv(output / 'cohort_audit.csv')
    LOGGER.info('Source lineage audit completed')


def main():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('develop', 'score', 'audit'))
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    {'develop': develop, 'score': score, 'audit': audit}[args.stage](args.output)


if __name__ == '__main__':
    main()
