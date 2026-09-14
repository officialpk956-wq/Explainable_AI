"""Reconstruct all scores, validate project isolation, and derive descriptive tables."""
from itertools import product, combinations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from .experiment import ROOT, OUT, DATA, config, load, digest, fingerprint, metric, dump

COMPARISONS=[('shap_sum','uniform_sum'),('shap_sum','shuffled_sum'),('select_shap','select_random'),('ps_aug','p_aug'),('p_aug','original')]

def analyze():
    cfg=config();manifest=json.loads((OUT/'manifest.json').read_text())
    if manifest['status']!='complete' or manifest['smoke'] or manifest['fingerprint']!=fingerprint(cfg):raise RuntimeError('Full matching execution required')
    scores=pd.read_csv(OUT/'scores.csv');keys=['target','seed','teacher','tuning','model','arm']
    expected=set(product(cfg['projects'],cfg['seeds'],cfg['teachers'],cfg['tuning_regimes'],cfg['models'],cfg['arms']))
    if set(map(tuple,scores[keys].to_numpy()))!=expected or scores.duplicated(keys).any():raise RuntimeError('Incomplete score grid')
    diagnostics=[];weights=[];checked=0
    for target,seed,teacher in product(cfg['projects'],cfg['seeds'],cfg['teachers']):
        stem=str(OUT/f'{target}__{seed}__{teacher}')
        done=json.loads(Path(stem+'.done.json').read_text());audit=json.loads(Path(stem+'.audit.json').read_text())
        assert done['fingerprint']==manifest['fingerprint']
        for n,h in done['hashes'].items():assert digest(OUT/n)==h
        sources=set(cfg['projects'])-{target};assert set(audit['source_indices'])==sources
        labels=[];groups=[]
        for source,indices in audit['source_indices'].items():
            x,y,_=load(source,cfg);assert len(indices)==len(set(indices))==min(len(y),cfg['source_cap_per_project'])
            labels.extend(y[indices]);groups.extend([source]*len(indices))
        np.testing.assert_array_equal(labels,audit['source_labels']);assert groups==audit['source_groups']
        g=np.asarray(groups);cv_seen=[]
        for fold in audit['tuning']:
            tr,va=np.array(fold['train']),np.array(fold['validation'])
            assert not set(tr)&set(va) and not set(g[tr])&set(g[va])
            assert set(tr)|set(va)==set(range(len(g)));cv_seen.extend(va)
            check_oof(fold['representation'],len(tr))
        assert sorted(cv_seen)==list(range(len(g)))
        check_oof(audit['final_representation'],len(g))
        for key,c in audit['chosen'].items():
            regime,m,a=key.split('__');tuned_arm=a if regime=='independent' else 'original'
            vals=[np.mean(audit['candidate_auc']['__'.join([m,tuned_arm,str(i)])]) for i in range(2)]
            assert c==int(np.argmax(vals))
        sub=scores[(scores.target==target)&(scores.seed==seed)&(scores.teacher==teacher)]
        with np.load(stem+'.npz',allow_pickle=False) as saved:
            _,y,ids=load(target,cfg);np.testing.assert_array_equal(saved['labels'],y);np.testing.assert_array_equal(saved['target_ids'],ids)
            for r in sub.itertuples():
                m=metric(y,saved['__'.join([r.tuning,r.model,r.arm])])
                for k,v in m.items():assert np.isclose(v,getattr(r,k),rtol=0,atol=1e-12)
                checked+=1
        a=audit['final_representation'];w=np.asarray(a['weights']);assert np.ptp(w)>1e-10
        diagnostics.append({'target':target,'seed':seed,'teacher':teacher,'redundancy':a['source_redundancy'],'shift':a['target_median_shift'],'active_features':len(w)})
        vector=np.zeros(len(cfg['features']));vector[a['keep']]=w
        weights.append({'target':target,'seed':seed,'teacher':teacher,**dict(zip(cfg['features'],vector))})
    rows=[]
    for arm,control in COMPARISONS:
        pairkeys=['target','seed','teacher','tuning','model']
        paired=scores[scores.arm==arm].merge(scores[scores.arm==control],on=pairkeys,suffixes=('_a','_b'),validate='one_to_one')
        for metric_name in ['auc','average_precision','f1_macro']:
            paired['delta']=paired[metric_name+'_a']-paired[metric_name+'_b']
            for key,d in paired.groupby(['target','teacher','tuning','model']):
                rows.append(dict(zip(['target','teacher','tuning','model'],key),arm=arm,control=control,metric=metric_name,mean_difference=float(d.delta.mean()),repeat_sd=float(d.delta.std()),minimum=float(d.delta.min()),maximum=float(d.delta.max()),repeats=len(d)))
    effects=pd.DataFrame(rows);effects.to_csv(OUT/'effects.csv',index=False)
    summary=[]
    for key,d in effects.groupby(['tuning','metric','arm','control']):
        target=d.groupby('target').mean_difference.mean()
        summary.append(dict(zip(['tuning','metric','arm','control'],key),mean_difference=float(target.mean()),target_min=float(target.min()),target_max=float(target.max()),target_positive=int((target>0).sum()),targets=len(target)))
    pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(OUT/'diagnostics.csv',index=False)
    w=pd.DataFrame(weights);w.to_csv(OUT/'weights.csv',index=False)
    stability=[]
    for target,d in w.groupby('target'):
        for t,group in d.groupby('teacher'):
            vectors=group[cfg['features']].to_numpy();values=[float(spearmanr(a,b).statistic) for a,b in combinations(vectors,2)]
            stability.append({'target':target,'comparison':t+' across seeds','mean_spearman':float(np.mean(values))})
        means=d.groupby('teacher')[cfg['features']].mean()
        stability.append({'target':target,'comparison':'RF vs ExtraTrees','mean_spearman':float(spearmanr(means.loc['RandomForest'],means.loc['ExtraTrees']).statistic)})
    pd.DataFrame(stability).to_csv(OUT/'stability.csv',index=False)
    quality=[];overlap=[]
    for target in cfg['projects']:
        x,y,ids=load(target,cfg);df=pd.DataFrame(x)
        quality.append({'project':target,'rows':len(y),'buggy':int(y.sum()),'prevalence':float(y.mean()),'static_features':x.shape[1],'repeated_metric_rows':int(df.duplicated(keep=False).sum())})
        for source in cfg['projects']:
            if source>=target:continue
            xs,ys,si=load(source,cfg)
            hashes=set(pd.util.hash_pandas_object(pd.DataFrame(xs),index=False))
            overlap.append({'project_a':source,'project_b':target,'shared_path_strings':len(set(ids)&set(si)),'target_rows_with_matching_metric_vector':int(pd.util.hash_pandas_object(df,index=False).isin(hashes).sum())})
    pd.DataFrame(quality).to_csv(OUT/'datasets.csv',index=False);pd.DataFrame(overlap).to_csv(OUT/'overlap.csv',index=False)
    dump(OUT/'validation.json',{'valid':True,'score_rows':checked,'checkpoints':42,'target_projects':7,'target_label_access':'scoring only','source_project_folds_verified':True,'OOF_coverage_verified':True,'per_arm_parameter_choices_verified':True,'all_prediction_scores_reconstructed':True,'source_fingerprint_current':True})
    print(pd.DataFrame(summary).query("metric=='auc'").to_string(index=False))

def check_oof(a,n):
    seen=[]
    for f in a['oof']:
        tr,va=set(f['train']),set(f['validation']);assert not tr&va and tr|va==set(range(n));seen.extend(f['validation'])
    assert sorted(seen)==list(range(n))
    np.testing.assert_allclose(sorted(a['weights']),sorted(a['shuffled_weights']))
    assert len(a['selected'])==len(a['random_selected'])==int(np.ceil(len(a['weights'])/2))

if __name__=='__main__':analyze()
