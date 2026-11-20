"""Fold-local constraints, conditional blocks, ALE, and cluster intervals."""

import numpy as np
import pandas as pd

from step3_no_leakage_data import B, CENTER, MODULES, NOMINAL, RECIPE, SEED, TARGET
from step3_no_leakage_models import metrics, predict


REPEATS = 50
BOOTSTRAPS = 1000
MIN_DONORS = 20
MIN_COVERAGE = 0.8
MAX_DISTANCE = 0.14
ALE_COLUMNS = {
    '1_liner_-_width': 'Liner width',
    '1_liner_-_thickness': 'Liner thickness',
    '1_apex_-_width': 'Apex width',
    '1_carcass_ply_-_width': 'Carcass ply width',
    '1_sidewall_profile_-_weight': 'Sidewall profile weight',
    '2_belt_-_width': 'Second belt width',
    '1_tread-belt_assembly_-_weight': 'Tread-belt assembly weight',
}


def legal_carcass(frame):
    second = frame['Has_2nd_Carcass_Ply'].eq(1)
    reinforced = frame['Has_Bead_Reinforcement'].eq(1)
    return (
        second.eq(frame['2_carcass_ply_-_width'].gt(0))
        & second.eq(frame['2_carcass_ply_-_thickness'].gt(0))
        & second.eq(frame['2_carcass_ply_-_cf_code'].ne('Not_Present'))
        & reinforced.eq(frame['1_bead_reinforcement_-_width'].gt(0))
    )


def supported_segments(flags):
    segments = []
    for index, flag in enumerate(flags):
        if flag:
            if index == 0 or not flags[index - 1]:
                segments.append([])
            segments[-1].append(index)
    return segments


def profile_counts(frame, columns):
    return frame.groupby(columns, sort=True, dropna=False).size().rename('work_orders').reset_index()


def distance_matrix(left, right, columns, scales):
    distance = np.zeros((len(left), len(right)))
    for column in columns:
        x = left[column].to_numpy()[:, None]
        z = right[column].to_numpy()[None, :]
        if column in NOMINAL:
            distance += x != z
        else:
            distance += np.minimum(np.abs(x - z) / scales[column], 1)
    return distance / len(columns)


def local_context(train, columns):
    numeric = [c for c in columns if c not in NOMINAL]
    iqr90 = train[numeric].quantile(0.95) - train[numeric].quantile(0.05)
    spread = train[numeric].max() - train[numeric].min()
    return iqr90.where(iqr90.gt(0), spread).replace(0, 1)


def compatible_groups(frame):
    return frame.groupby([CENTER, 'rim_diameter'], sort=True).indices


def conditional_donors(train, validation, replaced):
    """Match on the complement within observed work-center and rim strata."""
    columns = [c for c in B if c not in replaced and c != 'rim_diameter']
    scales = local_context(train, columns)
    donors = profile_counts(train.loc[legal_carcass(train)], [CENTER, *B])
    donor_groups = compatible_groups(donors)
    candidates = [None] * len(validation)
    eligible_counts = np.zeros(len(validation), dtype=int)
    nearest = np.full(len(validation), np.nan)
    for stratum, positions in compatible_groups(validation).items():
        if stratum not in donor_groups:
            continue
        dpos = donor_groups[stratum]
        d = donors.iloc[dpos]
        v = validation.iloc[positions]
        distances = distance_matrix(v, d, columns, scales)
        different = np.zeros(distances.shape, dtype=bool)
        for column in replaced:
            different |= v[column].to_numpy()[:, None] != d[column].to_numpy()[None, :]
        weights = d['work_orders'].to_numpy()
        for local, pos in enumerate(positions):
            indices = np.flatnonzero(different[local] & (distances[local] <= MAX_DISTANCE))
            count = weights[indices].sum()
            eligible_counts[pos] = count
            if count >= MIN_DONORS:
                nearest[pos] = distances[local, indices].min()
                probability = weights[indices] * np.exp(-distances[local, indices] / 0.05)
                candidates[pos] = (dpos[indices], probability / probability.sum())
    return donors, candidates, eligible_counts, nearest


def conditional_permutation(model, train, validation, fold, baseline):
    results, losses, repetitions = [], [], []
    for module_index, (module, replaced) in enumerate(MODULES.items()):
        donors, candidates, counts, nearest = conditional_donors(train, validation, replaced)
        positions = np.flatnonzero([choices is not None for choices in candidates])
        support = validation.iloc[positions]
        n = len(support)
        results.append({'module': module, 'fold': fold, 'n_target': len(validation), 'n_supported': n,
                        'coverage': n / len(validation), 'n_recipes': support[RECIPE].nunique(),
                        'min_donors': counts[positions].min() if n else np.nan,
                        'median_donors': np.median(counts[positions]) if n else np.nan,
                        'median_nearest': np.median(nearest[positions]) if n else np.nan,
                        'known_constraint_violations': 0})
        if not n:
            continue
        generator = np.random.default_rng(SEED + 100 * fold + module_index)
        chosen = np.column_stack([generator.choice(candidates[i][0], REPEATS, p=candidates[i][1])
                                  for i in positions])
        perturbed = pd.concat([support[B]] * REPEATS, ignore_index=True)
        perturbed[replaced] = donors.iloc[chosen.ravel()][replaced].to_numpy()
        if not legal_carcass(perturbed).all():
            raise ValueError(f'Invalid carcass state after replacing {module}')
        predicted = predict(model, perturbed, B).reshape(REPEATS, n)
        observed = support[TARGET].to_numpy()
        before = baseline[positions]
        base = metrics(observed, before)
        error = predicted - observed
        for repeat in range(REPEATS):
            after = metrics(observed, predicted[repeat])
            repetitions.append({'module': module, 'fold': fold, 'repeat': repeat, 'n': n,
                                **{f'delta_{key}': after[key] - base[key] for key in ('mae', 'rmse', 'r2')}})
        losses.append(pd.DataFrame({'proxy_id': support.index, 'module': module, 'fold': fold,
                                    RECIPE: support[RECIPE].to_numpy(), 'y': observed,
                                    'mae_before': np.abs(before - observed),
                                    'mae_after': np.abs(error).mean(axis=0),
                                    'mse_before': (before - observed) ** 2,
                                    'mse_after': (error ** 2).mean(axis=0)}))
    return results, losses, repetitions


def endpoint_support(train, validation, column, endpoints):
    columns = [c for c in B if c != column and c != 'rim_diameter']
    scales = local_context(train, columns)
    donors = profile_counts(train.loc[legal_carcass(train)], [CENTER, *B])
    groups = compatible_groups(donors)
    counts = np.zeros((len(validation), len(endpoints)), dtype=int)
    for stratum, positions in compatible_groups(validation).items():
        if stratum not in groups:
            continue
        selected = donors.iloc[groups[stratum]]
        distances = distance_matrix(validation.iloc[positions], selected, columns, scales)
        compatible = distances <= MAX_DISTANCE
        for j, endpoint in enumerate(endpoints):
            hit = selected[column].eq(endpoint).to_numpy()
            counts[positions, j] = compatible[:, hit] @ selected.loc[hit, 'work_orders'].to_numpy()
    return counts


def ale(model, train, validation, fold):
    output = []
    for column in ALE_COLUMNS:
        original = train[column].to_numpy()
        edges = np.unique(np.quantile(original, np.linspace(0.05, 0.95, 11), method='nearest'))
        if len(edges) < 3:
            output.append({'feature': column, 'fold': fold, 'status': 'TOO_FEW_LEVELS'})
            continue
        # Merge sparse bins using only the training sample.
        cold = (original >= edges[0]) & (original <= edges[-1])
        while len(edges) > 3:
            memberships = np.digitize(original[cold], edges[1:-1])
            counts = np.bincount(memberships, minlength=len(edges) - 1)
            smallest = int(np.argmin(counts))
            if counts[smallest] >= 80:
                break
            edges = np.delete(edges, min(smallest + 1, len(edges) - 2))

        measurements = validation[column].to_numpy()
        inside = (measurements >= edges[0]) & (measurements <= edges[-1])
        support = endpoint_support(train, validation, column, edges)
        membership = np.digitize(measurements, edges[1:-1])
        rows, differences = [], []
        for j, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            original_bin = inside & (membership == j)
            valid = original_bin & (support[:, j] > 0) & (support[:, j + 1] > 0)
            selected = validation.loc[valid]
            valid_low = selected.assign(**{column: lo})
            valid_high = selected.assign(**{column: hi})
            if not legal_carcass(valid_low).all() or not legal_carcass(valid_high).all():
                raise ValueError('ALE endpoint violated a carcass constraint')
            n, clusters = len(selected), selected[RECIPE].nunique()
            approved = n >= 50 and clusters >= 5
            diffs = predict(model, valid_high, B) - predict(model, valid_low, B) if approved else np.array([])
            differences.append((selected[RECIPE].to_numpy(), diffs))
            rows.append({'feature': column, 'fold': fold, 'bin': j, 'low': lo, 'high': hi,
                         'n_target': len(validation), 'n_outside': int((~inside).sum()),
                         'n_bin': int(original_bin.sum()), 'n_supported': n, 'n_recipes': clusters,
                         'min_low_donors': support[valid, j].min() if n else np.nan,
                         'min_high_donors': support[valid, j + 1].min() if n else np.nan,
                         'median_low_donors': np.median(support[valid, j]) if n else np.nan,
                         'median_high_donors': np.median(support[valid, j + 1]) if n else np.nan,
                         'range_low': edges[0], 'range_high': edges[-1],
                         'status': 'SUPPORTED' if approved else 'LOW_SUPPORT',
                         'local_delta': diffs.mean() if approved else np.nan})
        for segment_id, indices in enumerate(supported_segments([r['status'] == 'SUPPORTED' for r in rows])):
            delta = np.array([rows[j]['local_delta'] for j in indices])
            weights = [rows[j]['n_supported'] for j in indices]
            curve = np.r_[0, delta.cumsum()]
            curve -= np.average((curve[:-1] + curve[1:]) / 2, weights=weights)
            for offset, j in enumerate(indices):
                rows[j].update(segment=segment_id, ale_low=curve[offset], ale_high=curve[offset + 1])
        output.extend(rows)
    return output


def cluster_weights(clusters, seed=SEED):
    levels, inverse = np.unique(np.asarray(clusters), return_inverse=True)
    probabilities = np.full(len(levels), 1 / len(levels))
    weights = np.random.default_rng(seed).multinomial(len(levels), probabilities, BOOTSTRAPS)
    return levels, inverse, weights


def cluster_mean_intervals(samples, clusters):
    levels, inverse, weights = cluster_weights(clusters)
    samples = np.asarray(samples)
    sums = np.zeros((len(levels), samples.shape[1]))
    np.add.at(sums, inverse, samples)
    draw = (weights @ sums) / (weights @ np.bincount(inverse))[:, None]
    return np.percentile(draw, [2.5, 97.5], axis=0)


def paired_intervals(y, abs_before, abs_after, sq_before, sq_after, clusters):
    levels, inverse, weights = cluster_weights(clusters)
    values = np.column_stack([abs_after - abs_before, sq_before, sq_after, y, y ** 2])
    sums = np.zeros((len(levels), values.shape[1]))
    np.add.at(sums, inverse, values)
    statistics = (weights @ sums) / (weights @ np.bincount(inverse))[:, None]
    variance = statistics[:, 4] - statistics[:, 3] ** 2
    draws = np.column_stack([statistics[:, 0], np.sqrt(statistics[:, 2]) - np.sqrt(statistics[:, 1]),
                             (statistics[:, 1] - statistics[:, 2]) / np.where(variance > 0, variance, np.nan)])
    low, high = np.nanpercentile(draws, [2.5, 97.5], axis=0)
    return {f'{metric}_{bound}': values[i] for i, metric in enumerate(('mae', 'rmse', 'r2'))
            for bound, values in (('lo', low), ('hi', high))}
