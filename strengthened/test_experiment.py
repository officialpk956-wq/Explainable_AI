import unittest
import numpy as np
from strengthened.experiment import config,fit_predict,representations,load

class IsolationTests(unittest.TestCase):
    def setUp(self):
        self.cfg=config();self.cfg.update(teacher_trees=3,downstream_trees=3)
        rng=np.random.default_rng(3)
        self.x=rng.normal(size=(90,6));self.y=np.tile([0,1],45)
        self.g=np.repeat(['a','b','c'],30);self.z=rng.normal(size=(12,6))
    def test_covariates_do_not_change_source_weights_or_tuning(self):
        _,a=fit_predict(self.x,self.y,self.g,self.z,self.cfg,4,'RandomForest')
        _,b=fit_predict(self.x,self.y,self.g,self.z*100+9,self.cfg,4,'RandomForest')
        self.assertEqual(a['candidate_auc'],b['candidate_auc'])
        self.assertEqual(a['chosen'],b['chosen'])
        self.assertEqual(a['final_representation']['weights'],b['final_representation']['weights'])
    def test_controls_and_oof_cover_every_source_row_once(self):
        aa,bb,a=representations(self.x,self.y,self.z,self.cfg,4,'ExtraTrees')
        np.testing.assert_allclose(sorted(a['weights']),sorted(a['shuffled_weights']))
        self.assertEqual(aa['select_shap'].shape,aa['select_random'].shape)
        seen=[]
        for f in a['oof']:
            self.assertFalse(set(f['train'])&set(f['validation']));seen+=f['validation']
        self.assertEqual(sorted(seen),list(range(len(self.y))))
        self.assertEqual(bb['ps_aug'].shape[1],2*self.x.shape[1]+1)
    def test_label_columns_excluded_and_documented_labels_used(self):
        self.assertEqual(len(self.cfg['features']),54)
        self.assertFalse(set(self.cfg['features'])&{'RealBug','RealBugCount','HeuBug','HeuBugCount','File','COMM'})
        x,y,ids=load(self.cfg['projects'][0],self.cfg)
        self.assertEqual(len(x),len(y));self.assertEqual(len(ids),len(set(ids)))

if __name__=='__main__':unittest.main()
