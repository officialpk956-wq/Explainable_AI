"""Source-group nested, per-arm tuning with two cross-fitted SHAP teachers.

python -m strengthened.experiment --workers 2
No target responses are accepted by fit_predict(). All execution inputs are hashed.
"""
from pathlib import Path
from datetime import datetime, timezone
from importlib.metadata import version
import argparse
import concurrent.futures
import hashlib
import json
import time
import warnings
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GroupKFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/revised/strengthening/full'
DATA=ROOT/'data_external/Rnalytica/inst/extdata/jira'

def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,o):
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(o,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)
def config():return json.loads((Path(__file__).parent/'protocol.json').read_text())
def load(name,cfg):
    d=pd.read_csv(DATA/(name+'.csv'))
    raw=d[cfg['label']].astype(str).str.lower()
    if not raw.isin(['true','false']).all():raise ValueError('Unexpected response encoding')
    x=d[cfg['features']].astype(float).replace([np.inf,-np.inf],np.nan).to_numpy()
    y=(raw=='true').astype(int).to_numpy()
    if min(np.bincount(y,minlength=2))<3:raise ValueError('Insufficient classes')
    if not d.File.is_unique:raise ValueError('Duplicate file identifiers')
    return x,y,np.asarray(d.File.astype(str),dtype=str)

def metric(y,p):
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('Invalid probabilities')
    return {'auc':float(roc_auc_score(y,p)),'average_precision':float(average_precision_score(y,p)),
            'f1_macro':float(f1_score(y,p>=.5,average='macro',zero_division=0))}

def teacher(name,cfg,seed):
    cls={'RandomForest':RandomForestClassifier,'ExtraTrees':ExtraTreesClassifier}[name]
    return cls(n_estimators=cfg['teacher_trees'],max_depth=cfg['teacher_depth'],min_samples_leaf=2,class_weight='balanced',random_state=seed,n_jobs=1)
def shap_values(t,x):
    a=shap.TreeExplainer(t,feature_perturbation='tree_path_dependent').shap_values(x)
    if isinstance(a,list):a=a[1]
    a=np.asarray(a)
    if a.ndim==3:a=a[:,:,1]
    if a.shape!=x.shape or not np.isfinite(a).all():raise ValueError('SHAP output invalid')
    return a

def representations(x,y,z,cfg,seed,tname):
    """All transforms fit on x, never z; y belongs exclusively to x."""
    imputer=SimpleImputer(strategy='median',keep_empty_features=True).fit(x)
    a=imputer.transform(x);b=imputer.transform(z)
    keep=np.flatnonzero(np.std(a,axis=0)>0)
    if not len(keep):raise ValueError('No varying source metrics')
    a=a[:,keep];b=b[:,keep]
    pp=np.empty(len(y));ss=np.empty_like(a);folds=[]
    for tr,va in StratifiedKFold(cfg['oof_folds'],shuffle=True,random_state=seed).split(a,y):
        # Fit imputation inside each OOF teacher training fold as well.
        fold_imp=SimpleImputer(strategy='median',keep_empty_features=True).fit(x[tr])
        at=fold_imp.transform(x[tr])[:,keep];av=fold_imp.transform(x[va])[:,keep]
        t=teacher(tname,cfg,seed);t.fit(at,y[tr]);pp[va]=t.predict_proba(av)[:,1];ss[va]=shap_values(t,av)
        folds.append({'train':tr.tolist(),'validation':va.tolist()})
    t=teacher(tname,cfg,seed);t.fit(a,y);pt=t.predict_proba(b)[:,1];st=shap_values(t,b)
    w=np.maximum(np.abs(ss).mean(axis=0),1e-12);w=w/w.sum()
    rng=np.random.default_rng(seed);shuffle=w[rng.permutation(len(w))]
    k=int(np.ceil(len(w)*cfg['selection_fraction']));top=np.argsort(-w,kind='stable')[:k];random=rng.choice(len(w),k,replace=False)
    aa={'original':a,'shap_sum':a*w,'uniform_sum':a/len(w),'shuffled_sum':a*shuffle,'select_shap':a[:,top],'select_random':a[:,random],'p_aug':np.column_stack([a,pp]),'ps_aug':np.column_stack([a,pp,ss])}
    bb={'original':b,'shap_sum':b*w,'uniform_sum':b/len(w),'shuffled_sum':b*shuffle,'select_shap':b[:,top],'select_random':b[:,random],'p_aug':np.column_stack([b,pt]),'ps_aug':np.column_stack([b,pt,st])}
    corr=pd.DataFrame(a).corr(method='spearman').abs().fillna(0).to_numpy()
    scale=RobustScaler().fit(a);medshift=float(np.mean(np.abs(np.median(scale.transform(b),axis=0))))
    audit={'keep':keep.tolist(),'weights':w.tolist(),'shuffled_weights':shuffle.tolist(),'selected':top.tolist(),'random_selected':random.tolist(),'oof':folds,'source_redundancy':float(corr[np.triu_indices(len(w),1)].mean()),'target_median_shift':medshift}
    return aa,bb,audit

def signed_log(x):return np.sign(x)*np.log1p(np.abs(x))
def learner(name,choice,cfg,seed):
    if name=='RandomForest':return RandomForestClassifier(n_estimators=cfg['downstream_trees'],max_depth=cfg['candidates'][name][choice],min_samples_leaf=2,class_weight='balanced',random_state=seed,n_jobs=1)
    return Pipeline([('log',FunctionTransformer(signed_log)),('scale',RobustScaler()),('clf',LogisticRegression(C=cfg['candidates'][name][choice],class_weight='balanced',solver='liblinear',max_iter=20000,random_state=seed))])

def fit_predict(x,y,groups,target_x,cfg,seed,tname):
    """Deliberately no target_y argument: tuning and representations are source-only."""
    cv=list(GroupKFold(cfg['outer_tuning_folds']).split(x,y,groups))
    candidate={(m,a,c):[] for m in cfg['models'] for a in cfg['arms'] for c in range(2)}
    folds=[]
    for tr,va in cv:
        if set(groups[tr])&set(groups[va]):raise AssertionError('Project leakage in tuning')
        aa,bb,rep=representations(x[tr],y[tr],x[va],cfg,seed,tname)
        for m,a,c in candidate:
            model=learner(m,c,cfg,seed);model.fit(aa[a],y[tr]);candidate[m,a,c].append(float(roc_auc_score(y[va],model.predict_proba(bb[a])[:,1])))
        folds.append({'train':tr.tolist(),'validation':va.tolist(),'representation':rep})
    choice={(m,a):int(np.argmax([np.mean(candidate[m,a,c]) for c in range(2)])) for m in cfg['models'] for a in cfg['arms']}
    aa,bb,rep=representations(x,y,target_x,cfg,seed,tname)
    predictions={};selected={}
    for m in cfg['models']:
        for a in cfg['arms']:
            for c in set([choice[m,a],choice[m,'original']]):
                model=learner(m,c,cfg,seed);model.fit(aa[a],y);prob=model.predict_proba(bb[a])[:,1]
                for regime in cfg['tuning_regimes']:
                    wanted=choice[m,a] if regime=='independent' else choice[m,'original']
                    if wanted==c:
                        key='__'.join([regime,m,a]);predictions[key]=prob;selected[key]=c
    audit={'tuning':folds,'candidate_auc':{'__'.join([m,a,str(c)]):v for (m,a,c),v in candidate.items()},'chosen':selected,'final_representation':rep}
    return predictions,audit

def fingerprint(cfg):
    paths=[Path(__file__),Path(__file__).parent/'protocol.json',Path(__file__).parent/'SELECTION.json']+[DATA/(n+'.csv') for n in cfg['projects']]
    return {'files':{str(p.relative_to(ROOT)):digest(p) for p in paths},'packages':{n:version(n) for n in ['numpy','pandas','scikit-learn','scipy','shap']},'config':cfg}

def task(target,seed,tname,out,cfg,fp):
    warnings.filterwarnings('error',category=ConvergenceWarning)
    start=time.time();stem=out/f'{target}__{seed}__{tname}'
    def path(ext):return Path(str(stem)+ext)
    if path('.done.json').exists():
        d=json.loads(path('.done.json').read_text())
        if d['fingerprint']!=fp:raise RuntimeError('Checkpoint fingerprint mismatch')
        if any(digest(out/n)!=v for n,v in d['hashes'].items()):raise RuntimeError('Checkpoint corrupted')
        return f'cached {target} {seed} {tname}'
    xx=[];yy=[];gg=[];indices={}
    for name in cfg['projects']:
        if name==target:continue
        x,y,ids=load(name,cfg);ix=np.arange(len(y))
        if len(ix)>cfg['source_cap_per_project']:
            ix,_=train_test_split(ix,train_size=cfg['source_cap_per_project'],stratify=y,random_state=seed)
        xx.append(x[ix]);yy.append(y[ix]);gg.extend([name]*len(ix));indices[name]=ix.tolist()
    x=np.concatenate(xx);y=np.concatenate(yy);g=np.asarray(gg);xt,yt,it=load(target,cfg)
    with threadpool_limits(limits=1):pred,audit=fit_predict(x,y,g,xt,cfg,seed,tname)
    rows=[]
    for key,prob in pred.items():
        regime,m,a=key.split('__');rows.append({'target':target,'seed':seed,'teacher':tname,'tuning':regime,'model':m,'arm':a,**metric(yt,prob)})
    pred['labels']=yt;pred['target_ids']=it
    np.savez_compressed(path('.npz'),**pred)
    pd.DataFrame(rows).to_csv(path('.csv'),index=False)
    audit.update({'target':target,'seed':seed,'teacher':tname,'source_indices':indices,'source_groups':g.tolist(),'source_labels':y.tolist(),'target_rows':len(yt),'seconds':time.time()-start})
    dump(path('.audit.json'),audit)
    dump(path('.done.json'),{'fingerprint':fp,'hashes':{path(e).name:digest(path(e)) for e in ['.csv','.npz','.audit.json']}})
    return f'completed {target} seed={seed} teacher={tname} ({time.time()-start:.1f}s)'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=2);ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    cfg=config();out=OUT
    if args.smoke:
        cfg.update(teacher_trees=4,downstream_trees=4,seeds=[101]);out=OUT.parent/'smoke'
    out.mkdir(parents=True,exist_ok=True);fp=fingerprint(cfg);mp=out/'manifest.json'
    if mp.exists() and json.loads(mp.read_text())['fingerprint']!=fp:raise RuntimeError('Use a new output directory after execution changes')
    dump(mp,{'status':'running','fingerprint':fp,'started_utc':datetime.now(timezone.utc).isoformat(),'smoke':args.smoke})
    jobs=[(t,s,n,out,cfg,fp) for t in (cfg['projects'][:1] if args.smoke else cfg['projects']) for s in cfg['seeds'] for n in cfg['teachers']]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures=[ex.submit(task,*j) for j in jobs]
        for f in concurrent.futures.as_completed(futures):print(f.result(),flush=True)
    scores=pd.concat([pd.read_csv(Path(str(out/f'{t}__{s}__{n}')+'.csv')) for t,s,n,*_ in jobs],ignore_index=True)
    scores.to_csv(out/'scores.csv',index=False)
    dump(mp,{'status':'complete','fingerprint':fp,'completed_utc':datetime.now(timezone.utc).isoformat(),'smoke':args.smoke,'rows':len(scores)})
    print('COMPLETE',len(scores),flush=True)

if __name__=='__main__':main()
