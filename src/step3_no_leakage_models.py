"""Nested development validation and frozen replay models."""

import json
import os
import pickle

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from step3_no_leakage_data import A, B, MODULES, MONTH, NOMINAL, PLAN, SEED, TARGET, assign_folds


CAT_OPTIONS = [{'iterations': 145, 'depth': 5}, {'iterations': 245, 'depth': 6}]
RIDGE_OPTIONS = [{'alpha': 1.0}, {'alpha': 15.0}, {'alpha': 100.0}]
THREADS = min(os.cpu_count() or 1, 6)
PRIMARY = {'A_CatBoost': ('CatBoost', A), 'B_CatBoost': ('CatBoost', B),
           'A_Ridge': ('Ridge', A), 'B_Ridge': ('Ridge', B),
           'A_no_month_CatBoost': ('CatBoost', [c for c in A if c != MONTH]),
           'Schedule_CatBoost': ('CatBoost', PLAN)}
LOCO = {f'LOCO_{name}': ('CatBoost', [c for c in B if c not in columns])
        for name, columns in MODULES.items()}
MODELS = PRIMARY | LOCO


def model_frame(orders, columns):
    frame = orders[columns].copy()
    for column in NOMINAL.intersection(columns):
        frame[column] = frame[column].astype('string').fillna('__MISSING__').astype(str)
    return frame


def fit_model(engine, columns, parameters, train):
    x = model_frame(train, columns)
    nominal = [c for c in columns if c in NOMINAL]
    if engine == 'CatBoost':
        model = CatBoostRegressor(
            **parameters, learning_rate=0.08, loss_function='RMSE', random_seed=SEED,
            one_hot_max_size=1000, thread_count=THREADS, verbose=False,
            allow_writing_files=False,
        )
        model.fit(x, train[TARGET], cat_features=nominal)
        return model
    numeric = [c for c in columns if c not in NOMINAL]
    transform = ColumnTransformer([
        ('numeric', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), numeric),
        ('nominal', OneHotEncoder(handle_unknown='ignore'), nominal),
    ])
    model = make_pipeline(transform, Ridge(**parameters, solver='lsqr'))
    return model.fit(x, train[TARGET])


def predict(model, frame, columns):
    return model.predict(model_frame(frame, columns))


def tune_fit(name, engine, columns, train, inner, audit, outer):
    options = CAT_OPTIONS if engine == 'CatBoost' else RIDGE_OPTIONS
    scores = []
    for parameters in options:
        absolute_error = 0.0
        for fold in sorted(inner.unique()):
            fit = train.loc[inner.ne(fold)]
            valid = train.loc[inner.eq(fold)]
            if set(fit['family']) & set(valid['family']):
                raise ValueError('Inner family crossover')
            model = fit_model(engine, columns, parameters, fit)
            error = np.abs(predict(model, valid, columns) - valid[TARGET])
            absolute_error += error.sum()
            audit.append({'outer': outer, 'inner': fold, 'model': name,
                          'params': json.dumps(parameters), 'n_train': len(fit),
                          'n_valid': len(valid), 'mae': error.mean()})
        scores.append(absolute_error / len(train))
    best = int(np.argmin(scores))
    audit.append({'outer': outer, 'inner': 'selected', 'model': name, 'n_train': len(train),
                  'params': json.dumps(options[best]), 'mae': scores[best]})
    return fit_model(engine, columns, options[best], train)


def metrics(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    error = pred - y
    mse = (error ** 2).mean()
    variance = y.var()
    return {'n': len(y), 'mae': np.abs(error).mean(), 'rmse': np.sqrt(mse),
            'r2': 1 - mse / variance if len(y) > 1 and variance > 0 else np.nan,
            'bias': error.mean()}


def performance(frame, names, subset):
    return [{'scope': subset, 'model': name, **metrics(frame[TARGET], frame[name])} for name in names]


def baseline(train, result):
    result['Median'] = train[TARGET].median()
    result['Mean'] = train[TARGET].mean()


def tree_shap(model, validation, columns, prediction):
    x = model_frame(validation, columns)
    pool = Pool(x, cat_features=[c for c in columns if c in NOMINAL])
    values = model.get_feature_importance(pool, type='ShapValues', thread_count=THREADS)
    residual = np.abs(values.sum(axis=1) - prediction)
    if not np.allclose(values.sum(axis=1), prediction, atol=1e-6, rtol=1e-8):
        raise ValueError('TreeSHAP additivity check failed')
    return pd.DataFrame(values, index=validation.index, columns=[*columns, 'baseline']), residual.max()


def save_bundle(path, value):
    with path.open('wb') as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
