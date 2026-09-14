"""Self-check for the correlation/redundancy diagnostic's two pure functions.

python -m unittest strengthened.test_diagnostic_correlation -v

Fast, synthetic, no dependency on the 140 real checkpoints: verifies vif() and
clusters_at() do what the diagnostic's interpretation depends on, before
trusting either against real data.
"""
import unittest

import numpy as np

from .diagnostic_correlation import clusters_at, vif


class Vif(unittest.TestCase):
    def test_independent_columns_have_vif_near_one(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(2000, 6))
        scores = vif(x)
        self.assertTrue(np.all(scores < 1.2), scores)

    def test_near_duplicate_column_has_large_vif(self):
        rng = np.random.default_rng(0)
        base = rng.normal(size=(2000, 1))
        x = np.column_stack([base, base + rng.normal(scale=1e-3, size=(2000, 1)),
                             rng.normal(size=(2000, 4))])
        scores = vif(x)
        self.assertGreater(scores[0], 100)
        self.assertGreater(scores[1], 100)
        self.assertTrue(np.all(scores[2:] < 1.2), scores[2:])

    def test_exact_linear_combination_gives_infinite_vif(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=(500, 1))
        b = rng.normal(size=(500, 1))
        x = np.column_stack([a, b, a + b])
        self.assertTrue(np.isinf(vif(x)[2]))


class Clusters(unittest.TestCase):
    def test_correlated_pair_shares_a_cluster_independent_column_does_not(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=1000)
        corr = np.column_stack([a, a + rng.normal(scale=.05, size=1000), rng.normal(size=1000)])
        rho = np.corrcoef(corr, rowvar=False)
        labels = clusters_at(rho, 0.7)
        self.assertEqual(labels[0], labels[1])
        self.assertNotEqual(labels[0], labels[2])

    def test_all_independent_columns_are_singleton_clusters(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(2000, 5))
        rho = np.corrcoef(x, rowvar=False)
        labels = clusters_at(rho, 0.7)
        self.assertEqual(len(set(labels)), 5)


if __name__ == '__main__':
    unittest.main()
