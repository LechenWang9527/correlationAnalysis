import unittest

from step3_no_leakage_data import (
    A, B, FORBIDDEN, PLAN_KEY, assign_folds, build_proxy, identity_audit,
)
from step3_no_leakage_attribution import legal_carcass, supported_segments

import numpy as np
import pandas as pd


class SnapshotContractTest(unittest.TestCase):
    def test_feature_manifest(self):
        self.assertEqual((len(set(A)), len(set(B))), (46, 42))
        self.assertTrue(set(B) < set(A))
        self.assertFalse(set(A) & FORBIDDEN)
        self.assertFalse(set(PLAN_KEY) & FORBIDDEN)

    def test_ids_are_stable_and_actual_collision_only_links_families(self):
        samples = pd.DataFrame({column: [1, 1, 1] for column in A})
        samples['RecipeID'] = ['a', 'a', 'b']
        samples['WorkCenterNameCoupled'] = 'M'
        samples['PlannedStartMonth'] = [10, 11, 10]
        for column in ['ActualStartMonth', 'ActualStartDayOfWeek', 'ActualStartHour']:
            samples[column] = 10
        first, _ = build_proxy(samples)
        second, _ = build_proxy(samples.iloc[::-1])
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(len(first), 3)
        self.assertEqual(first['family'].nunique(), 2)
        self.assertEqual(first['cross_boundary'].sum(), 2)

    def test_group_folds_keep_family_together(self):
        orders = pd.DataFrame({'family': list('aabbccddeeffgghhiijj'),
                               'PlannedStartMonth': [9, 10] * 10})
        assigned = assign_folds(orders, 5, 17)
        self.assertEqual(set(assigned), set(range(5)))
        self.assertEqual(assigned.groupby(orders['family']).nunique().max(), 1)

    def test_identity_flags_preserve_original_target(self):
        rows = pd.DataFrame({'OperatorOverlapSeconds': [0, 10, 10],
                             'OperatorCoverageRatio': [0, 1, 1],
                             'QuantityDone': [10] * 3,
                             'IEStandardCycleTimePer100Tire': [100] * 3,
                             'CycleTimeTolerancePer100Tire': [0] * 3,
                             'ActualVsIEGap': [5, 0.001, 0]})
        result = identity_audit(rows)
        self.assertEqual(result['identity_status'].tolist(),
                         ['ZERO_COVERAGE', 'IDENTITY_RESIDUAL', 'PASS'])
        self.assertTrue(np.isnan(result['runtime_proxy'].iloc[0]))
        self.assertEqual(rows['ActualVsIEGap'].tolist(), [5, 0.001, 0])

    def test_ale_preserves_support_gaps_and_carcass_structure(self):
        self.assertEqual(supported_segments([True, True, False, True]), [[0, 1], [3]])
        frame = pd.DataFrame({'Has_2nd_Carcass_Ply': [0, 0, 1],
                              '2_carcass_ply_-_width': [0, 20, 20],
                              '2_carcass_ply_-_thickness': [0, 0, 2],
                              '2_carcass_ply_-_cf_code': ['Not_Present', 'Not_Present', 'C'],
                              'Has_Bead_Reinforcement': [0, 0, 1],
                              '1_bead_reinforcement_-_width': [0, 0, 20]})
        self.assertEqual(legal_carcass(frame).tolist(), [True, False, True])


if __name__ == '__main__':
    unittest.main()
