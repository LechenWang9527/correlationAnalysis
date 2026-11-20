"""Snapshot provenance, outcome-free proxy identity, and outcome eligibility."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / 'data/processed/clean_dataset.csv'
RAW = ROOT / 'data/raw/output.csv'
SCHEMA = ROOT / 'data/processed/feature_schema.json'
TARGET = 'ActualVsIEGap'
Q = 'QuantityDone'
IE = 'IEStandardCycleTimePer100Tire'
TOL = 'CycleTimeTolerancePer100Tire'
O = 'OperatorOverlapSeconds'
C = 'OperatorCoverageRatio'
MONTH = 'PlannedStartMonth'
RECIPE = 'RecipeID'
CENTER = 'WorkCenterNameCoupled'
PLAN = ['PlannedQuantity', 'PlannedStartHour', 'PlannedStartDayOfWeek', MONTH]
ACTUAL = ['ActualStartHour', 'ActualStartDayOfWeek', 'ActualStartMonth']
PLAN_KEY = [RECIPE, CENTER, *PLAN]
ACTUAL_KEY = [RECIPE, CENTER, 'PlannedQuantity', *ACTUAL]
OUTCOME = [Q, IE, TOL, TARGET]
FORBIDDEN = {O, C, *OUTCOME, *ACTUAL, 'ShiftNumber', 'OperatorUserID',
             'ExtTerminated', 'Comment', 'DestinationInfo', 'quantity gap '}

MODULES = {
    'geometry_bead': [
        'rim_diameter', 'tire_section_width', 'tire_aspect_ratio',
        'bead_-_bead_inner_diameter', 'bead_-_bead_construction',
    ],
    'liner': [
        '1_liner_-_width', '1_liner_-_thickness', '1_liner_-_profile_geometry_code',
        '1_liner_-_compound',
    ],
    'carcass_reinforcement': [
        '1_apex_-_width', '1_apex_-_thickness', '1_apex_-_compound',
        '1_apex_-_profile_geometry_code', '1_carcass_ply_-_width',
        'Has_2nd_Carcass_Ply', '2_carcass_ply_-_width', '2_carcass_ply_-_thickness',
        '2_carcass_ply_-_cf_code', 'Has_Bead_Reinforcement',
        '1_bead_reinforcement_-_width', 'ply_lock',
    ],
    'sidewall': [
        '1_flange_cushion_-_compound', '1_sidewall_-_bom_component_setting',
        '1_sidewall_profile_-_weight', '1_sidewall_-_combined_profile_-_width',
        '1_sidewall_profile_-_compound', '1_sidewall_-_combined_profile_-_profile_geometry_code',
    ],
    'belt_cap': [
        '1_belt_-_cf_epdm', '1_belt_ply_assembly_-_1_strip_rubber_-_width',
        '2_belt_-_angle', '2_belt_-_width', '1_belt_-_cf_code',
        '1_belt_-_cf_construction', '1_wound_cap_ply_-_cf_code',
        '1_wound_cap_ply_-_cf_construction', '1_wound_cap_ply_-_spooling_construction',
    ],
    'tread': [
        '1_tread_base_-_width', '1_tread_wing_-_width', '1_tread-belt_assembly_-_weight',
        '1_tread_base_-_compound', '1_tread_cap_-_compound', '1_tread_wing_-_profile_geometry_code',
    ],
}
B = [column for columns in MODULES.values() for column in columns]
A = B + PLAN
NOMINAL = {
    column for column in B
    if any(token in column for token in ('_code', '_compound', '_construction', 'Has_'))
} | {'ply_lock', 'PlannedStartDayOfWeek', MONTH}
SEED = 20261120
SNAPSHOTS = {
    CLEAN: '59f0b129820a4535d7e5bea15e45a3b5ab5608982dec368402b0494156a92b0f',
    RAW: 'eab24e2c4de56466943300799e1da20e5dbdb8bd93cb9d04e43972a51146bea6',
    SCHEMA: '1ac7931ac6dcb835fa92ecb61c0eddfc6947b4b1293cfc390b2a043c1ad27afd',
}


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def fingerprint(values):
    payload = json.dumps(list(values), ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def read_inputs():
    for path, expected in SNAPSHOTS.items():
        if digest(path) != expected:
            raise ValueError(f'Snapshot checksum mismatch: {path}')
    if len(set(A)) != 46 or len(set(B)) != 42 or set(A) & FORBIDDEN:
        raise ValueError('Invalid feature allowlist')
    return pd.read_csv(CLEAN, usecols=[*A, RECIPE, CENTER, *ACTUAL],
                       dtype={RECIPE: str, CENTER: str}, keep_default_na=False)


def build_proxy(rows):
    """Group on snapshot planning coordinates. Actual hours only link audit families."""
    planned = rows[PLAN_KEY].drop_duplicates().copy()
    planned['proxy_id'] = [fingerprint(item) for item in planned.itertuples(index=False, name=None)]
    if not planned['proxy_id'].is_unique:
        raise ValueError('Proxy identifier is not unique')
    mapping = rows[PLAN_KEY].merge(planned, on=PLAN_KEY, how='left', validate='many_to_one')
    ids = mapping['proxy_id']
    parents = dict(zip(planned['proxy_id'], planned['proxy_id']))

    def find(key):
        while key != parents[key]:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    links = rows[ACTUAL_KEY].assign(proxy_id=ids.to_numpy()).drop_duplicates()
    repeated = links.duplicated(ACTUAL_KEY, keep=False)
    for _, linked in links.loc[repeated].groupby(ACTUAL_KEY, sort=False):
        roots = {find(key) for key in linked['proxy_id']}
        root = min(roots)
        for key in roots:
            parents[key] = root

    planned = planned.set_index('proxy_id').sort_index()
    planned['family'] = [find(key) for key in planned.index]
    planned['cross_boundary'] = (
        planned[MONTH].eq(11).groupby(planned['family']).transform('nunique').gt(1)
    )
    crossed = rows[MONTH].eq(11).ne(rows['ActualStartMonth'].eq(11)).to_numpy()
    crossed_rows = ids[crossed]
    crossed_families = planned.loc[crossed_rows.unique(), 'family']
    planned['cross_boundary'] |= planned['family'].isin(crossed_families)
    grouped = rows[B + ACTUAL].groupby(ids.to_numpy(), sort=False)
    varied = grouped.nunique().reindex(planned.index)
    physical = grouped.first()[B].reindex(planned.index).where(varied[B].eq(1))
    planned['feature_conflict'] = varied[B].gt(1).any(axis=1)
    planned['actual_time_conflict'] = varied[ACTUAL].gt(1).any(axis=1)
    planned['source_rows'] = grouped.size().reindex(planned.index)
    return planned.join(physical), ids.rename('proxy_id')


def assign_folds(orders, n_splits, seed):
    counts = pd.crosstab(orders['family'], orders[MONTH]).sort_index()
    weights = counts.sum(axis=0).to_numpy() / n_splits
    random_order = np.random.default_rng(seed).permutation(len(counts))
    priority = sorted(random_order, key=lambda row: -sum(counts.iloc[row] / weights))
    loads = np.zeros((n_splits, len(weights)))
    folds = {}
    for row in priority:
        count = counts.iloc[row].to_numpy()
        cost = ((2 * loads + count) * count / weights ** 2).sum(axis=1)
        fold = int(np.argmin(cost))
        folds[counts.index[row]] = fold
        loads[fold] += count
    return orders['family'].map(folds).astype(int)


def identity_audit(rows):
    valid = rows[C].gt(0) & rows[Q].gt(0)
    runtime = rows[O].div(rows[C].where(valid))
    residual = runtime / rows[Q].where(valid) - (rows[IE] + rows[TOL]) / 100 - rows[TARGET]
    status = np.select([rows[C].eq(0) & rows[O].eq(0), ~valid, residual.abs().ge(1e-4)],
                       ['ZERO_COVERAGE', 'INVALID_DENOMINATOR', 'IDENTITY_RESIDUAL'],
                       default='PASS')
    return pd.DataFrame({'formula_residual': residual,
                         'runtime_proxy': runtime.where(status == 'PASS'),
                         'identity_status': status}, index=rows.index)


def read_outcomes(ids, months):
    """Access outcome cells only for the explicitly requested planning months."""
    columns = [*OUTCOME, O, C]
    outcomes = []
    with CLEAN.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        for record, row in enumerate(reader, 1):
            if int(row[MONTH]) in months:
                outcomes.append((record, ids.iat[record - 1], *(float(row[c]) for c in columns)))
    rows = pd.DataFrame(outcomes, columns=['clean_record', 'proxy_id', *columns])
    return rows.join(identity_audit(rows))


def label_proxies(orders, source):
    grouped = source.groupby('proxy_id', sort=False)
    counts = grouped[OUTCOME].nunique(dropna=False)
    values = grouped[OUTCOME].first().where(counts.eq(1))
    usable = source['identity_status'].eq('PASS').groupby(source['proxy_id']).sum()
    result = orders.join(values).join(usable.rename('formula_pass_rows'))
    result['anomaly_rows'] = result['source_rows'] - result['formula_pass_rows']
    result['label_conflict'] = counts[TARGET].gt(1).reindex(result.index, fill_value=False)
    result['outcome_conflict'] = counts.gt(1).any(axis=1).reindex(result.index, fill_value=False)
    result['status'] = np.select(
        [result['label_conflict'], result['outcome_conflict'], result['cross_boundary'],
         result['feature_conflict'], result['actual_time_conflict'], result['formula_pass_rows'].eq(0)],
        ['AMBIGUOUS_LABEL', 'AMBIGUOUS_OUTCOME', 'CROSS_BOUNDARY',
         'AMBIGUOUS_FEATURE', 'AMBIGUOUS_EXECUTION', 'NO_VALID_FORMULA_ROW'], default='ELIGIBLE_PROXY',
    )
    return result


def source_audit(inputs, ids, source, output):
    """Reproduce fixed Step 1 transforms solely for source-to-clean provenance."""
    from step1_data_cleaner import IndustrialDataCleaner

    schema = json.loads(SCHEMA.read_text(encoding='utf-8-sig'))
    categories = {column: str for column in schema['categorical_features']}
    clean = pd.read_csv(CLEAN, low_memory=False, float_precision='round_trip',
                        dtype=categories, keep_default_na=False)
    raw = pd.read_csv(RAW, low_memory=False, float_precision='round_trip')
    projected = raw.copy()
    for column in [O, C]:
        projected[column] = projected[column].clip(lower=0)
    for column, spec in schema['binary_and_ordinal_mappings'].items():
        values = projected[column].astype(str).str.strip()
        if 'mapping' in spec:
            values = values.replace(spec.get('dirty_value_fixes', {}))
            projected[column] = IndustrialDataCleaner._apply_strict_map(values, spec['mapping'], column)
        else:
            projected[column] = values.str.removeprefix('A=').astype(float)
    parsed = raw['tire_size'].map(IndustrialDataCleaner._parse_tire_size)
    projected['tire_section_width'], projected['tire_aspect_ratio'] = zip(*parsed)
    for column in schema['categorical_features']:
        projected[column] = projected[column].astype(str).str.strip()

    # The approved snapshot was serialized after pandas' original float parser.
    # Normalize decimal keys at a precision that preserves all distinct clean rows.
    columns = clean.columns.tolist()
    clean_keys = clean.copy()
    float_columns = clean.select_dtypes('float').columns
    decimals = dict.fromkeys(float_columns, 10)
    decimals[C] = 12
    for name in float_columns:
        clean_keys[name] = clean_keys[name].round(decimals[name])
        projected[name] = projected[name].round(decimals[name])
    if clean_keys.duplicated().any():
        raise ValueError('Audit normalization merged distinct clean records')
    matching = projected[columns].merge(
        clean_keys.assign(clean_record=np.arange(1, len(clean) + 1)),
        on=columns, how='left', validate='many_to_one',
    )
    if matching['clean_record'].isna().any() or matching['clean_record'].nunique() != len(clean):
        raise ValueError('Raw-to-clean conservation failed')
    raw_map = pd.DataFrame({'raw_record': np.arange(1, len(raw) + 1),
                            'export_index': raw.iloc[:, 0],
                            'clean_record': matching['clean_record'],
                            'raw_overlap': raw[O], 'raw_coverage': raw[C],
                            'raw_bead_structure': raw['bead_-_bead_construction']})
    raw_map['lineage_status'] = np.where(raw_map['clean_record'].duplicated(),
                                         'MULTIPLE_RAW_TO_CLEAN', 'FIRST_RAW_MATCH')
    raw_map.to_csv(output / 'raw_lineage.csv', index=False)

    aligned = inputs.copy()
    aligned['clean_record'] = np.arange(1, len(inputs) + 1)
    rows = aligned[[*ACTUAL_KEY, MONTH, 'clean_record']].merge(
        source, on='clean_record', validate='one_to_one',
    )
    assert len(rows) == len(inputs) and rows['proxy_id'].equals(ids)
    historical_key = ACTUAL_KEY + [Q]
    rows['historical_group'] = rows.groupby(historical_key, sort=True).ngroup() + 1
    historical = rows.groupby('historical_group').agg(
        source_rows=(TARGET, 'size'), target_levels=(TARGET, 'nunique'), completed_quantity=(Q, 'first'),
        first_clean_record=('clean_record', 'min'), actual_month=('ActualStartMonth', 'first'),
        proxy_candidates=('proxy_id', 'nunique'),
    )
    historical.to_csv(output / 'historical_groups_audit.csv')
    zero = rows.loc[rows['identity_status'].eq('ZERO_COVERAGE'), 'clean_record']
    original_zero = raw_map.loc[raw_map['clean_record'].isin(zero)].copy()
    original_zero['zero_source'] = np.select(
        [original_zero['raw_overlap'].lt(0) & original_zero['raw_coverage'].lt(0),
         original_zero['raw_overlap'].eq(0) & original_zero['raw_coverage'].eq(0)],
        ['NEGATIVE_CLIPPED', 'SOURCE_ZERO'], default='MIXED_OR_UNRESOLVED',
    )
    zero_groups = original_zero.groupby('clean_record')['zero_source']
    zero_status = zero_groups.first().where(zero_groups.nunique().eq(1), 'MIXED_OR_UNRESOLVED')
    rows['identity_status'] = rows['clean_record'].map(zero_status).fillna(rows['identity_status'])
    rows['historical_label_conflict'] = rows['historical_group'].map(historical['target_levels']).gt(1)
    rows.drop(columns=ACTUAL_KEY + [MONTH]).to_csv(output / 'clean_lineage_outcome_audit.csv', index=False)
    row_flags = rows['identity_status'].value_counts().to_dict()
    return {
        'clean_rows': len(rows), 'raw_rows': len(raw_map), 'raw_multi_matches': len(raw_map) - len(rows),
        'historical_groups': len(historical), 'historical_label_conflicts': int(historical['target_levels'].gt(1).sum()),
        'historical_conflict_rows': int(rows['historical_label_conflict'].sum()),
        'raw_quantity_sum_after_cleaning': int(rows[Q].sum()),
        'historical_quantity_once': int(historical['completed_quantity'].sum()), 'row_flags': row_flags,
        'anomaly_historical_groups': int(rows.loc[rows['identity_status'].ne('PASS'), 'historical_group'].nunique()),
    }
