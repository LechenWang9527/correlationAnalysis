"""Regression checks for the screening report contracts."""
import unittest

from step2a_strip_a_screening import assign_traffic_light
from step2b_strip_b_team_and_simpson import check_simpson_triggers


class ScreeningContractTest(unittest.TestCase):
    def test_ratings_and_low_linear_signal(self):
        row = {
            "q_value_bh": 0.01,
            "effective_r2": 0.10,
            "dcor": 0.0,
            "track": "numeric",
            "feature_name": "measurement",
            "r2_linear_num": 0.009,
        }
        cases = [
            ({}, ("HIGH", "[LOW_LINEAR_SIGNAL]")),
            ({"effective_r2": 0.02}, ("MEDIUM", "[LOW_LINEAR_SIGNAL]")),
            ({"effective_r2": 0.003}, ("LOW", "[PASS]")),
            ({"effective_r2": 0.0, "dcor": 0.10}, ("LOW", "[PASS]")),
            ({"effective_r2": 0.0, "dcor": 0.08}, ("INSIGNIFICANT", "[PASS]")),
            ({"q_value_bh": 0.05}, ("INSIGNIFICANT", "[PASS]")),
            ({"r2_linear_num": 0.01}, ("HIGH", "[PASS]")),
            ({"feature_name": "__shadow_random_noise__"}, ("HIGH", "[NOISE_BASELINE]")),
            ({"track": "categorical", "eta2_num": 0.04, "omega2_num": 0.02},
             ("HIGH", "[CATEGORY_BIAS_ADJUSTMENT] η²−ω²=2.0 个百分点")),
        ]
        for changes, expected in cases:
            with self.subTest(changes=changes):
                self.assertEqual(assign_traffic_light(row | changes, 0.03), expected)

    def test_simpson_trigger_boundaries(self):
        cases = [
            ((0.04, 0.30, -0.21, 2), (True, True)),
            ((0.05, 0.30, -0.20, 2), (False, False)),
            ((0.02, 0.30, 0.20, 2), (False, True)),
            ((0.06, 0.25, -0.25, 2), (True, False)),
            ((0.04, 0.30, -0.21, 1), (False, False)),
        ]
        for values, expected in cases:
            with self.subTest(values=values):
                self.assertEqual(check_simpson_triggers(*values), expected)
