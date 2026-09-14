"""Read-only checks of saved manuscript evidence; never retrains or alters CSV inputs."""
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'
AUDIT=OUT/'audit'

def inventory():
    rows=[]
    excluded={'sdp_env','venv','.venv-audit','.git','__pycache__','node_modules','catboost_info'}
    for directory,dirs,files in os.walk(ROOT):
        dirs[:]=[d for d in dirs if d not in excluded]
        for name in files:
            p=Path(directory)/name
            if AUDIT in p.parents: continue
            row=dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,
                     sha256=hashlib.sha256(p.read_bytes()).hexdigest(),kind=p.suffix,status='inventoried',details='')
            try:
                if p.suffix=='.py':
                    tree=ast.parse(p.read_text(encoding='utf-8-sig'))
                    row.update(status='syntax_ok',details=';'.join(n.name for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)))
                elif p.suffix=='.ipynb':
                    nb=json.loads(p.read_text(encoding='utf-8-sig'))
                    row['status']=f"notebook: {len(nb.get('cells',[]))} cells (not executed)"
                elif p.suffix=='.csv':
                    with p.open(encoding='utf-8-sig',newline='') as f:
                        reader=csv.reader(f); header=next(reader,[]); count=sum(1 for _ in reader)
                    row.update(status=f'csv: {count} rows',details=';'.join(header))
            except Exception as e:
                row['status']=f'ERROR: {e}'
            rows.append(row)
    pd.DataFrame(rows).to_csv(AUDIT/'file_inventory.csv',index=False)
    return rows

def counts(frame):
    sig=frame['significant_at_0.05'].astype(str).str.lower().eq('true')
    return dict(cells=len(frame), wins=int((sig & (frame.mean_diff>0)).sum()),
                losses=int((sig & (frame.mean_diff<0)).sum()))

def main():
    AUDIT.mkdir(parents=True,exist_ok=True)
    inv=inventory()
    evidence={}
    a=pd.read_csv(OUT/'ablation_tests.csv')
    evidence['ablation']=counts(a[a.comparison.isin(['shap_vs_uniform','shap_vs_shuffled'])])
    e=pd.read_csv(OUT/'ablation_equivalence.csv')
    evidence['equivalence']={str(k):dict(cells=len(g), equivalent_raw=int(g.equivalent_raw.sum()), equivalent_holm=int(g.equivalent_holm.sum())) for k,g in e.groupby('delta')}
    c=pd.read_csv(OUT/'cpdp_significance.csv')
    evidence['zero_shot']={str(k):counts(g) for k,g in c[c.setting=='zero-shot'].groupby('arm')}
    s=pd.read_csv(OUT/'h_specification_curve.csv')
    evidence['stability_specifications']=dict(rows=len(s),negative=int((s.median_rho<0).sum()),significant=int((s.p_value<.05).sum()),axes=['corr','n_inst','lodo','lomo'])
    for name in ['mechanism_ablation_verdict','derivation_fit','shap_prior_zeroshot_verdict','sfa_cross_project_verdict']:
        evidence[name]=pd.read_csv(OUT/f'{name}.csv').to_dict('records')
    issues=[]
    # Paired comparisons require one finite record for every arm and seed.
    for name in ['ablation_scores','scale_invariant_scores','reconcile_selection_scores','mechanism_ablation_scores','sfa_cross_project_scores','shap_prior_zeroshot_scores']:
        d=pd.read_csv(OUT/f'{name}.csv')
        keys=[x for x in ['ecosystem','dataset','target','model','seed','boot_seed','variant','arm','pipeline'] if x in d]
        metric=[x for x in ['f1','auc','prauc'] if x in d]
        if d.duplicated(keys).any(): issues.append(f'{name}: duplicate keys {keys}')
        for m in metric:
            if not d[m].between(0,1).all(): issues.append(f'{name}: invalid {m}')
        seed=next((x for x in ['seed','boot_seed'] if x in d),None)
        if seed:
            groups=[x for x in keys if x!=seed]
            if not d.groupby(groups)[seed].nunique().eq(20).all(): issues.append(f'{name}: not all groups have 20 seeds')
    evidence['integrity_issues']=issues
    evidence['inventory']=dict(files=len(inv),python=sum(r['kind']=='.py' for r in inv),notebooks=sum(r['kind']=='.ipynb' for r in inv),errors=[r['path'] for r in inv if r['status'].startswith('ERROR')])
    (AUDIT/'evidence_summary.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    print(json.dumps(evidence,indent=2))

if __name__=='__main__': main()
