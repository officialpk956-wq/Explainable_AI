"""Correctness tests for the v2 additions. Run before trusting any v2 result.

python -m unittest strengthened.test_experiment_v2 -v

These check construction, not outcomes: control shapes, label handling, the target
information boundary and the untouched original snapshot.
"""
import hashlib
import json
import unittest
from pathlib import Path

import numpy as np

from .experiment_v2 import (ROOT, config, coral_map, representations,
                            permute_within_groups, learner, fit_predict)


def toy(n=180, f=12, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, f))
    y = (rng.random(n) < .3).astype(int)
    g = np.asarray(['p%d' % (i % 3) for i in range(n)])
    return x, y, g


class Controls(unittest.TestCase):
    def setUp(self):
        self.cfg = config()
        self.cfg.update(teacher_trees=4, downstream_trees=4, oof_folds=3, outer_tuning_folds=3)

    def test_augmentation_control_is_dimension_matched(self):
        x, y, _ = toy()
        z, _, _ = toy(60, 12, 1)
        aa, bb, audit = representations(x, y, z, self.cfg, 101, 'RandomForest')
        self.assertEqual(aa['pn_aug'].shape[1], aa['ps_aug'].shape[1])
        self.assertEqual(bb['pn_aug'].shape[1], bb['ps_aug'].shape[1])
        self.assertEqual(audit['augmentation_columns']['pn_aug'],
                         audit['augmentation_columns']['ps_aug'])
        # the noise columns must not reproduce the SHAP columns
        width = aa['original'].shape[1]
        self.assertFalse(np.allclose(aa['pn_aug'][:, width + 1:], aa['ps_aug'][:, width + 1:]))

    def test_permutation_preserves_project_class_counts(self):
        _, y, g = toy()
        p = permute_within_groups(y, g, 101)
        for project in np.unique(g):
            m = g == project
            self.assertEqual(int(y[m].sum()), int(p[m].sum()))
        self.assertEqual(int(y.sum()), int(p.sum()))
        self.assertFalse(np.array_equal(y, p))

    def test_coral_matches_target_covariance_and_leaves_target_alone(self):
        a, _, _ = toy(400, 8, 2)
        b = toy(400, 8, 3)[0] * 3.0 + 5.0
        mapped = coral_map(a, b, self.cfg['coral_ridge'])
        self.assertEqual(mapped.shape, a.shape)
        before = np.linalg.norm(np.cov(a, rowvar=False) - np.cov(b, rowvar=False))
        after = np.linalg.norm(np.cov(mapped, rowvar=False) - np.cov(b, rowvar=False))
        self.assertLess(after, before)
        aa, bb, _ = representations(a, toy(400, 8, 2)[1], b, self.cfg, 101, 'RandomForest')
        np.testing.assert_allclose(bb['coral'], bb['original'])

    def test_prevalence_is_the_source_rate_and_learner_independent(self):
        x, y, g = toy()
        z = toy(60, 12, 1)[0]
        pred, audit = fit_predict(x, y, g, z, self.cfg, 101, 'RandomForest')
        keys = [k for k in pred if k.endswith('prevalence')]
        self.assertEqual(len(keys), len(self.cfg['models']) * len(self.cfg['tuning_regimes']))
        for k in keys:
            np.testing.assert_allclose(pred[k], y.mean())
        self.assertAlmostEqual(audit['source_positive_rate'], float(y.mean()))

    def test_fit_predict_cannot_receive_target_labels(self):
        import inspect
        params = inspect.signature(fit_predict).parameters
        self.assertNotIn('target_y', params)
        self.assertFalse(any('y' == p or p.endswith('_y') for p in list(params)[3:]))

    def test_every_declared_arm_is_predicted(self):
        x, y, g = toy()
        pred, _ = fit_predict(x, y, g, toy(60, 12, 1)[0], self.cfg, 101, 'RandomForest')
        produced = {k.split('__')[2] for k in pred}
        self.assertEqual(produced, set(self.cfg['arms']))

    def test_all_four_learners_build_and_balance(self):
        x, y, _ = toy()
        for name in self.cfg['models']:
            for c in range(2):
                m = learner(name, c, self.cfg, 101)
                m.fit(x, y)
                p = m.predict_proba(x[:20])[:, 1]
                self.assertTrue(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all())


class Snapshot(unittest.TestCase):
    def test_original_directory_unchanged(self):
        record = ROOT / 'revised' / 'original_snapshot.json'
        if not record.exists():
            self.skipTest('no original snapshot record present')
        entries = json.loads(record.read_text())
        entries = entries.get('files', entries)
        checked = 0
        for name, expected in entries.items():
            # keys are recorded relative to original/, not to the repository root
            p = ROOT / 'original' / name
            if not p.is_file():
                continue
            got = hashlib.sha256(p.read_bytes()).hexdigest()
            digest = expected if isinstance(expected, str) else expected.get('sha256')
            if digest:
                self.assertEqual(got, digest, f'{name} changed')
                checked += 1
        self.assertGreater(checked, 0)


if __name__ == '__main__':
    unittest.main()
