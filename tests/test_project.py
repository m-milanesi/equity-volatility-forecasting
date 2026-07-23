import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from volatility_forecasting import (  # noqa: E402
    FEATURES,
    HORIZON,
    load_prices,
    make_panel,
    purged_train_test_split,
)


class VolatilityProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = make_panel(load_prices())
        cls.train, cls.test = purged_train_test_split(cls.panel)

    def test_features_and_target_are_finite(self):
        values = self.panel[FEATURES + ["target_volatility"]].to_numpy()
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue((self.panel["target_volatility"] > 0).all())

    def test_split_is_chronological_and_disjoint(self):
        self.assertLess(self.train["Date"].max(), self.test["Date"].min())
        self.assertTrue(
            set(self.train["Date"].unique()).isdisjoint(
                set(self.test["Date"].unique())
            )
        )

    def test_split_has_a_full_horizon_gap(self):
        train_month = self.train["Date"].max().to_period("M").ordinal
        test_month = self.test["Date"].min().to_period("M").ordinal
        self.assertGreater(test_month - train_month, HORIZON)


if __name__ == "__main__":
    unittest.main()
