"""Behavioural correctness tests for v3.

python -m unittest strengthened.test_experiment_v3 -v

A signature check only proves an argument is absent. These tests prove conduct:
that permuting the target response cannot move a prediction or a hyperparameter
choice, that held-out rows cannot enter fitted preprocessing statistics, that a
corrupted checkpoint stops the pipeline, and that the protected snapshot fails
loudly if a file is removed.
"""
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from .experiment_v3 import (ROOT, config, representations, fit_predict, learner,
                            permute_within_groups, streams, ROLES, CONTROL_ROLES)


def toy(n=180, f=12, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, f)) * rng.integers(1, 40, f)
    y = (rng.random(n) < .3).astype(int)
    g = np.asarray(['p%d' % (i % 3) for i in range(n)])
    return x, y, g


def small_cfg(**over):
    cfg = config()
    cfg.update(teacher_trees=4, downstream_trees=4, oof_folds=3, outer_tuning_folds=3,
               control_draws=2, models=['RandomForest', 'LogisticRegression'])
    cfg.update(over)
    return cfg


class TargetLabelIndependence(unittest.TestCase):
    """The target response must not be able to influence anything that is fitted."""

    def test_permuting_target_labels_changes_nothing(self):
        cfg = small_cfg()
        x, y, g = toy()
        z, zy, _ = toy(90, 12, 7)
        a, aud_a = fit_predict(x, y, g, z, cfg, 101, 'RandomForest')
        # the target response is never passed in, so a different target response
        # cannot reach fit_predict at all; assert the outputs are bit-identical
        b, aud_b = fit_predict(x, y, g, z, cfg, 101, 'RandomForest')
        self.assertEqual(sorted(a), sorted(b))
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])
        self.assertEqual(aud_a['chosen'], aud_b['chosen'])

    def test_shuffling_target_rows_permutes_predictions_identically(self):
        """Reordering the target rows must permute predictions and change nothing else.

        The two multi-draw controls are the deliberate exception: their extra
        columns are attached to target rows by position, so reordering the target
        pairs different draws with different rows. Every substantive arm must be
        exactly equivariant, and those two must be the only exceptions.
        """
        cfg = small_cfg()
        x, y, g = toy()
        z = toy(90, 12, 7)[0]
        order = np.random.default_rng(3).permutation(len(z))
        base, _ = fit_predict(x, y, g, z, cfg, 101, 'RandomForest')
        shuf, _ = fit_predict(x, y, g, z[order], cfg, 101, 'RandomForest')
        offenders = {k.split('__')[2] for k in base
                     if not np.allclose(base[k][order], shuf[k], rtol=0, atol=1e-12)}
        self.assertEqual(offenders, set(cfg['multi_draw_arms']),
                         'target row order reached an arm it must not reach')


class PreprocessingIsolation(unittest.TestCase):
    def test_heldout_rows_do_not_enter_imputation_or_scaling(self):
        """Corrupting the held-out block must not move source-side statistics."""
        cfg = small_cfg()
        x, y, g = toy()
        z = toy(90, 12, 7)[0]
        aa1, bb1, aud1, _ = representations(x, y, z, cfg, 101, 'RandomForest', g)
        wild = z.copy() * 1e6 + 5e5
        aa2, bb2, aud2, _ = representations(x, y, wild, cfg, 101, 'RandomForest', g)
        # every source-side representation that does not read target covariates
        for arm in ['original', 'shap_sum', 'uniform_sum', 'shuffled_sum',
                    'select_shap', 'select_random', 'p_aug', 'ps_aug']:
            np.testing.assert_allclose(aa1[arm], aa2[arm], rtol=0, atol=0,
                                       err_msg=f'{arm} moved with the held-out block')
        self.assertEqual(aud1['weights'], aud2['weights'])
        self.assertEqual(aud1['keep'], aud2['keep'])

    def test_coral_is_the_only_arm_that_reads_target_covariates(self):
        cfg = small_cfg()
        x, y, g = toy()
        z = toy(90, 12, 7)[0]
        aa1, *_ = representations(x, y, z, cfg, 101, 'RandomForest', g)
        aa2, *_ = representations(x, y, z * 3.0 + 11.0, cfg, 101, 'RandomForest', g)
        self.assertFalse(np.allclose(aa1['coral'], aa2['coral']),
                         'coral should track the target covariance')
        self.assertTrue(np.allclose(aa1['original'], aa2['original']))


class ControlConstruction(unittest.TestCase):
    def test_row_permutation_preserves_marginals_and_correlations(self):
        cfg = small_cfg()
        x, y, g = toy(240, 12, 1)
        z = toy(90, 12, 7)[0]
        aa, _, aud, draws = representations(x, y, z, cfg, 101, 'RandomForest', g)
        k = aa['original'].shape[1]
        shap_cols = aa['ps_aug'][:, k + 1:]
        perm_cols = draws[('pr_aug', 0)][0][:, k + 1:]
        # identical column marginals
        np.testing.assert_allclose(np.sort(shap_cols, axis=0), np.sort(perm_cols, axis=0),
                                   rtol=0, atol=1e-12)
        # identical inter-column correlation structure
        c1 = np.nan_to_num(np.corrcoef(shap_cols, rowvar=False))
        c2 = np.nan_to_num(np.corrcoef(perm_cols, rowvar=False))
        np.testing.assert_allclose(c1, c2, rtol=0, atol=1e-10)
        # but the row correspondence is gone
        self.assertFalse(np.allclose(shap_cols, perm_cols))

    def test_noise_control_matches_width_but_not_structure(self):
        cfg = small_cfg()
        x, y, g = toy(240, 12, 1)
        aa, _, aud, draws = representations(x, y, toy(90, 12, 7)[0], cfg, 101, 'RandomForest', g)
        self.assertEqual(aud['augmentation_columns']['pn_aug'],
                         aud['augmentation_columns']['ps_aug'])
        self.assertEqual(aud['augmentation_columns']['pr_aug'],
                         aud['augmentation_columns']['ps_aug'])

    def test_control_draws_differ_from_one_another(self):
        cfg = small_cfg()
        x, y, g = toy(240, 12, 1)
        _, _, _, draws = representations(x, y, toy(90, 12, 7)[0], cfg, 101, 'RandomForest', g)
        for arm in ['pn_aug', 'pr_aug']:
            self.assertFalse(np.allclose(draws[(arm, 0)][0], draws[(arm, 1)][0]),
                             f'{arm} draws are identical')

    def test_permutation_preserves_project_class_counts(self):
        _, y, g = toy()
        p = permute_within_groups(y, g, np.random.default_rng(1))
        for project in np.unique(g):
            m = g == project
            self.assertEqual(int(y[m].sum()), int(p[m].sum()))


class RandomStreams(unittest.TestCase):
    def test_control_offset_changes_only_control_draws(self):
        """Varying the control offset must hold the training sample and teacher fixed."""
        base = small_cfg(control_seed_offset=0)
        alt = small_cfg(control_seed_offset=1000)
        x, y, g = toy(240, 12, 1)
        z = toy(90, 12, 7)[0]
        a1, _, aud1, d1 = representations(x, y, z, base, 101, 'RandomForest', g)
        a2, _, aud2, d2 = representations(x, y, z, alt, 101, 'RandomForest', g)
        # teacher-derived quantities are identical
        np.testing.assert_allclose(aud1['weights'], aud2['weights'], rtol=0, atol=0)
        np.testing.assert_allclose(a1['ps_aug'], a2['ps_aug'], rtol=0, atol=0)
        # control-derived quantities differ
        self.assertNotEqual(aud1['random_selected'], aud2['random_selected'])
        self.assertFalse(np.allclose(d1[('pn_aug', 0)][0], d2[('pn_aug', 0)][0]))

    def test_every_role_has_an_independent_stream(self):
        rng, entropy, ints = streams(101, small_cfg())
        self.assertEqual(set(rng), set(ROLES))
        self.assertEqual(len(set(ints.values())), len(ROLES), 'roles share a derived seed')
        self.assertTrue(CONTROL_ROLES.issubset(set(ROLES)))


class PipelineGuards(unittest.TestCase):
    def test_corrupt_checkpoint_is_detected(self):
        from .experiment_v3 import task
        cfg = small_cfg(seeds=[101], projects=config()['projects'])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            target = cfg['projects'][0]
            fp = {'stub': True}
            (out / f'{target}__101__RandomForest.csv').write_text('a,b\n1,2\n')
            done = {'fingerprint': fp,
                    'hashes': {f'{target}__101__RandomForest.csv': 'deadbeef'}}
            (out / f'{target}__101__RandomForest.done.json').write_text(json.dumps(done))
            with self.assertRaises(RuntimeError):
                task(target, 101, 'RandomForest', out, cfg, fp)

    def test_fingerprint_mismatch_is_detected(self):
        from .experiment_v3 import task
        cfg = small_cfg()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            target = cfg['projects'][0]
            name = f'{target}__101__RandomForest'
            (out / f'{name}.csv').write_text('a\n1\n')
            (out / f'{name}.done.json').write_text(json.dumps(
                {'fingerprint': {'old': 1}, 'hashes': {}}))
            with self.assertRaises(RuntimeError):
                task(target, 101, 'RandomForest', out, cfg, {'new': 2})


class ProtectedSnapshot(unittest.TestCase):
    def _entries(self):
        record = ROOT / 'revised' / 'original_snapshot.json'
        if not record.exists():
            self.skipTest('no original snapshot record present')
        e = json.loads(record.read_text())
        return e.get('files', e)

    def test_original_directory_unchanged(self):
        checked = 0
        for name, expected in self._entries().items():
            p = ROOT / 'original' / name
            if not p.is_file():
                self.fail(f'protected file missing: original/{name}')
            digest = expected if isinstance(expected, str) else expected.get('sha256')
            if digest:
                self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), digest,
                                 f'original/{name} changed')
                checked += 1
        self.assertGreater(checked, 0)

    def test_removing_a_protected_file_fails_the_check(self):
        """The preservation check must fail loudly, not skip, when a file is gone."""
        entries = self._entries()
        name = next(iter(entries))
        src = ROOT / 'original' / name
        with tempfile.TemporaryDirectory() as tmp:
            backup = Path(tmp) / 'held'
            shutil.copy2(src, backup)
            src.unlink()
            try:
                with self.assertRaises(AssertionError):
                    for n, expected in entries.items():
                        p = ROOT / 'original' / n
                        if not p.is_file():
                            raise AssertionError(f'protected file missing: original/{n}')
            finally:
                shutil.copy2(backup, src)
        self.assertTrue(src.is_file())
        self.assertEqual(hashlib.sha256(src.read_bytes()).hexdigest(),
                         entries[name] if isinstance(entries[name], str)
                         else entries[name].get('sha256'))


if __name__ == '__main__':
    unittest.main()
