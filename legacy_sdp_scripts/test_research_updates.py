"""Focused regression checks and small real-data fits, not full reproduction."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import manuscript_diagnostics as D
import pipeline as P
import cross_ecosystem as XE
import run_legacy as run_all
import stability_theory as T

class ResearchChecks(unittest.TestCase):
    def test_diagnostics_match_sklearn(self):
        x,_=D.load('equinox')
        np.testing.assert_allclose(D.robust_scale(x),RobustScaler().fit_transform(x))
        np.testing.assert_allclose(D.robust_scale(.2*x),RobustScaler().fit_transform(.2*x))
        delta=(D.robust_scale(x)-D.robust_scale(.2*x)).abs().max()
        self.assertAlmostEqual(delta.NMBFU,8.8)
        self.assertLess(delta.NBFU,1e-12)

    def test_outliers_and_imbalance(self):
        x,y=D.load('ant-1.7'); keep=D.iqr_keep(x)
        self.assertEqual(int(keep.sum()),433)
        self.assertEqual(int(y[~keep].sum()),117)
        x,y=D.load('poi-3.0')
        self.assertGreater(y.value_counts().max()/y.value_counts().min(),1)

    def test_runner_dependencies(self):
        selected=[s[0] for s in run_all.select_stages(['equivalence'])]
        self.assertEqual(selected,['eclipse','apache','ablation','equivalence'])
        self.assertNotIn('shap_prior',[s[1] for s in run_all.STAGES])
        with self.assertRaises(ValueError): run_all.select_stages(['unknown'])

    def test_theory_does_not_overwrite_validation(self):
        original=T.OUT_DIR
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            pd.read_csv(P.OUT_DIR/'h_splithalf_raw.csv').to_csv(path/'h_splithalf_raw.csv',index=False)
            (path/'deconfounded_validation.csv').write_text('sentinel',encoding='utf-8')
            try:
                T.OUT_DIR=path; T.run_theory()
                self.assertEqual((path/'deconfounded_validation.csv').read_text(),'sentinel')
                result=pd.read_csv(path/'derivation_fit.csv')
                self.assertAlmostEqual(result.b.iloc[0],0.46953059878118397,places=10)
            finally: T.OUT_DIR=original

    def test_all_base_classifiers_on_both_schemas(self):
        for load,name in [(P.load_xy,'equinox'),(XE.load_promise,'velocity-1.6')]:
            x,y,_=load(name)
            a,b,c,d=train_test_split(x,y,stratify=y,random_state=42,test_size=.2)
            for model,clf in P.make_classifiers().items():
                with self.subTest(dataset=name,model=model):
                    params=clf.get_params()
                    if 'n_estimators' in params: clf.set_params(n_estimators=8)
                    if 'n_jobs' in params: clf.set_params(n_jobs=1)
                    pipe=P.make_pipeline_for(model,clf); pipe.fit(a.values,c.values)
                    prob=pipe.predict_proba(b.values)[:,1]
                    self.assertTrue(np.isfinite(prob).all())
                    self.assertTrue(((prob>=0)&(prob<=1)).all())

    def test_shap_oof(self):
        x,y,_=P.load_xy('equinox')
        w=P.shap_weights_out_of_fold(x,y,'RandomForest',{'RandomForest':{'n_estimators':8,'n_jobs':1}})
        self.assertEqual(len(w),x.shape[1])
        self.assertTrue(np.isfinite(w).all())
        self.assertAlmostEqual(P.normalise(w).sum(),1)

if __name__=='__main__': unittest.main(verbosity=2)
