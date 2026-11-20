"""Development-only recipe cold-start and include/exclude anomaly refits."""

import argparse
import logging
from pathlib import Path

import pandas as pd

from step3_no_leakage import DEFAULT_OUTPUT, DEVELOPMENT, validate_eligible
from step3_no_leakage_data import MONTH, RECIPE, SEED, TARGET, assign_folds, build_proxy, label_proxies, read_inputs, read_outcomes
from step3_no_leakage_models import PRIMARY, performance, predict, tune_fit


MODELS = ['A_CatBoost', 'B_CatBoost']


def diagnostics(output):
    inputs = read_inputs()
    orders, ids = build_proxy(inputs)
    development = orders.loc[orders[MONTH].isin(DEVELOPMENT)]
    labels = label_proxies(development, read_outcomes(ids, DEVELOPMENT))
    eligible = labels.loc[labels['status'].eq('ELIGIBLE_PROXY')]
    validate_eligible(eligible)
    original_folds = assign_folds(development, 5, SEED).loc[eligible.index]
    recipe_folds = assign_folds(eligible.assign(family=eligible[RECIPE]), 5, SEED + 200)
    audit, statistics, predictions = [], [], []
    for name, folds in [('recipe_cold_start', recipe_folds), ('exclude_anomaly_refit', original_folds)]:
        collected = []
        for fold in range(5):
            train = eligible.loc[folds.ne(fold)]
            valid = eligible.loc[folds.eq(fold)]
            if name == 'recipe_cold_start':
                train = train.assign(family=train[RECIPE])
                if set(train[RECIPE]) & set(valid[RECIPE]):
                    raise ValueError('Recipe used by both training and evaluation')
            else:
                train = train.loc[train['anomaly_rows'].eq(0)]
            if set(train['family']) & set(valid['family']):
                raise ValueError('Family used by both training and evaluation')
            logging.info('%s outer fold %s: %s training and %s evaluation', name, fold, len(train), len(valid))
            inner = assign_folds(train, 3, SEED + 201 + fold)
            rows = valid[[TARGET, RECIPE, 'anomaly_rows']].copy().assign(experiment=name, fold=fold)
            for model_name in MODELS:
                engine, columns = PRIMARY[model_name]
                model = tune_fit(model_name, engine, columns, train, inner, audit, f'{name}_{fold}')
                rows[model_name] = predict(model, valid, columns)
            collected.append(rows)
            statistics.extend(performance(rows, MODELS, f'{name}_fold_{fold + 1}'))
        pooled = pd.concat(collected).sort_index()
        if not pooled.index.equals(eligible.index):
            raise ValueError('Incomplete sensitivity OOF')
        statistics.extend(performance(pooled, MODELS, name))
        predictions.append(pooled)
    pd.concat(predictions).to_csv(output / 'development_sensitivity_oof.csv')
    pd.DataFrame(audit).to_csv(output / 'development_sensitivity_fit_log.csv', index=False)
    pd.DataFrame(statistics).to_csv(output / 'development_sensitivity_metrics.csv', index=False)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    diagnostics(parser.parse_args().output)
