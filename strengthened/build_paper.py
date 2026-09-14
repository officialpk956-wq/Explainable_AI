"""Generate the EMSE manuscript and every plot from validated new results."""
from pathlib import Path
import json
import re
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .experiment import ROOT, OUT, config, digest, dump

DEST=ROOT/'paper/emse'
def escape(s):
    return str(s).replace('&',r'\&').replace('_',r'\_').replace('%',r'\%')
def table(name,caption,headers,rows):
    cols=('l'+'Y'*(len(headers)-1)) if name in ['data','baseline','targeteffects'] else 'Y'*len(headers)
    tex='\n'.join([r'\begin{table}[!htbp]',r'\centering\small',r'\setlength{\tabcolsep}{3pt}',rf'\caption{{{caption}}}\label{{tab:{name}}}',rf'\begin{{tabularx}}{{\linewidth}}{{{cols}}}',r'\toprule',' & '.join(map(escape,headers))+r' \\',r'\midrule',*[' & '.join(map(escape,row))+r' \\' for row in rows],r'\bottomrule',r'\end{tabularx}',r'\end{table}'])
    (DEST/'tables'/f'{name}.tex').write_text(tex,encoding='utf-8')
    return rf'\input{{tables/{name}.tex}}'
def figblock(n,label,caption):
    return '\n'.join([r'\begin{figure}[!htbp]',r'\centering',rf'\includegraphics[width=\linewidth]{{figures/Fig{n}.pdf}}',rf'\caption{{{caption}}}\label{{fig:{label}}}',r'\end{figure}'])
def save(fig,n):
    for ext in ['pdf','eps','png']:
        fig.savefig(DEST/'figures'/f'Fig{n}.{ext}',dpi=600 if ext=='png' else None,bbox_inches='tight')
    plt.close(fig)

def build():
    # Superseded by build_paper_v2. Both wrote to paper/emse/main.tex, so running this
    # after the v2 build silently replaced the current manuscript with the earlier one.
    if (DEST/'main.tex').exists() and 'build_paper_v2' in (DEST/'main.tex').read_text(encoding='utf-8'):
        raise RuntimeError('paper/emse/main.tex is the current (v2) manuscript. This v1 builder '
                           'would overwrite it. Use strengthened.build_paper_v2 instead.')
    cfg=config();validation=json.loads((OUT/'validation.json').read_text())
    if not validation['valid']:raise RuntimeError('Valid full results required')
    for d in ['figures','tables','results']:(DEST/d).mkdir(exist_ok=True)
    scores=pd.read_csv(OUT/'scores.csv');eff=pd.read_csv(OUT/'effects.csv');summary=pd.read_csv(OUT/'summary.csv');data=pd.read_csv(OUT/'datasets.csv');stability=pd.read_csv(OUT/'stability.csv');diagnostics=pd.read_csv(OUT/'diagnostics.csv')
    assert len(scores)==1344
    targets=cfg['projects'];short=[x.split('-')[0].title() for x in targets]
    comparisons=[('shap_sum','uniform_sum','Weighting − uniform'),('select_shap','select_random','Selection − random'),('ps_aug','p_aug','Attribution + prediction − prediction')]
    plt.rcParams.update({'font.size':9,'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(5.1,3));ax.set_xlim(0,10);ax.set_ylim(0,4);ax.axis('off')
    boxes=[(2,3,'Six source projects\n400 sampled rows per project'),(7.5,3,'Three source-project validation folds\nRebuild teacher + controls inside each fold'),(2,1,'Three-fold source OOF teacher\nRandomForest / ExtraTrees'),(7.5,1,'Per-arm parameter choice; source refit\nHeld-out project: prediction, then scoring')]
    for x,y,t in boxes:ax.text(x,y,t,ha='center',va='center',fontsize=8,bbox=dict(boxstyle='round,pad=.45',fc='white',ec='black'))
    for a,b in [((4,3),(5,3)),((2,2.45),(2,1.55)),((7.5,2.45),(7.5,1.55)),((4,1),(5,1))]:ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','lw':1})
    fig.tight_layout();save(fig,1)
    primary=eff[(eff.metric=='auc')&(eff.tuning=='independent')]
    fig,axes=plt.subplots(1,3,figsize=(5.1,3.6),sharey=True)
    for i,(ax,(a,b,label)) in enumerate(zip(axes,comparisons)):
        d=primary[(primary.arm==a)&(primary.control==b)].groupby('target').mean_difference.mean().reindex(targets)
        ax.scatter(d,range(7),s=25,color='black',marker=['o','s','^'][i]);ax.axvline(0,color='gray',lw=.8,ls='--');ax.set_yticks(range(7),short);ax.invert_yaxis();ax.set_xlabel('Δ ROC-AUC');ax.set_title(['(a) Weighting','(b) Selection','(c) Augmentation'][i],fontsize=9)
        lim=max(.015,float(d.abs().max())*1.2);ax.set_xlim(-lim,lim)
    fig.tight_layout();save(fig,2)
    fig,axes=plt.subplots(1,3,figsize=(5.1,3.6),sharey=True)
    for i,(ax,(a,b,label)) in enumerate(zip(axes,comparisons)):
        for offset,t,marker,color in [(-.12,'RandomForest','o','black'),(.12,'ExtraTrees','s','#707070')]:
            d=primary[(primary.arm==a)&(primary.control==b)&(primary.teacher==t)].groupby('target').mean_difference.mean().reindex(targets)
            ax.scatter(d,np.arange(7)+offset,s=20,marker=marker,color=color,label=t)
        ax.axvline(0,color='gray',lw=.7,ls='--');ax.set_yticks(range(7),short);ax.invert_yaxis();ax.set_xlabel('Δ ROC-AUC');ax.set_title(['(a) Weighting','(b) Selection','(c) Augmentation'][i],fontsize=9)
    fig.legend(*axes[0].get_legend_handles_labels(),fontsize=8,loc='upper center',bbox_to_anchor=(.58,1.02),ncol=2,frameon=False);fig.tight_layout(rect=(0,0,1,.91));save(fig,3)
    fig,ax=plt.subplots(figsize=(5.1,3.6))
    for off,c,marker,label in [(-.15,'RandomForest across seeds','o','RF across source samples'),(0,'ExtraTrees across seeds','s','ET across source samples'),(.15,'RF vs ExtraTrees','^','Between teacher means')]:
        d=stability[stability.comparison==c].set_index('target').mean_spearman.reindex(targets)
        ax.scatter(d,np.arange(7)+off,label=label,marker=marker,s=28)
    ax.set_yticks(range(7),short);ax.invert_yaxis();ax.set_xlim(-.05,1.03);ax.set_xlabel('Spearman rank correlation');ax.legend(fontsize=8,frameon=False,loc='lower left');fig.tight_layout();save(fig,4)
    diagnostic=diagnostics.groupby('target')[['shift','redundancy']].mean()
    sel=primary[(primary.arm=='select_shap')&(primary.control=='select_random')].groupby('target').mean_difference.mean()
    fig,axes=plt.subplots(1,2,figsize=(5.1,3.3))
    for ax,feature,label in zip(axes,['shift','redundancy'],['Target median shift','Source metric redundancy']):
        ax.scatter(diagnostic[feature],sel.reindex(diagnostic.index),color='black',s=22)
        for i,t in enumerate(diagnostic.index):ax.annotate(chr(65+targets.index(t)),(diagnostic.loc[t,feature],sel.loc[t]),xytext=(4,4 if i%2==0 else -8),textcoords='offset points',fontsize=8)
        ax.axhline(0,color='gray',lw=.7,ls='--');ax.set_xlabel(label);ax.set_ylabel('Selection − random: Δ ROC-AUC')
        ax.margins(.25)
    fig.tight_layout();save(fig,5)
    def val(a,b,tuning='independent',column='mean_difference'):
        return float(summary[(summary.metric=='auc')&(summary.tuning==tuning)&(summary.arm==a)&(summary.control==b)].iloc[0][column])
    values={
      'weight':f"{val('shap_sum','uniform_sum'):+.4f}",
      'selection':f"{val('select_shap','select_random'):+.4f}",
      'augmentation':f"{val('ps_aug','p_aug'):+.4f}",
      'selection_min':f"{val('select_shap','select_random',column='target_min'):+.4f}",
      'selection_max':f"{val('select_shap','select_random',column='target_max'):+.4f}",
      'shared_selection':f"{val('select_shap','select_random','shared_baseline'):+.4f}",
      'teacher_stability':f"{stability[stability.comparison=='RF vs ExtraTrees'].mean_spearman.mean():.3f}"
    }
    values['augmentation_min']=f"{val('ps_aug','p_aug',column='target_min'):+.4f}"
    values['augmentation_max']=f"{val('ps_aug','p_aug',column='target_max'):+.4f}"
    def secondary(a,b,metric):
        return float(summary[(summary.tuning=='independent')&(summary.metric==metric)&(summary.arm==a)&(summary.control==b)].iloc[0].mean_difference)
    values['augmentation_ap']=f"{secondary('ps_aug','p_aug','average_precision'):+.4f}"
    values['augmentation_f1']=f"{secondary('ps_aug','p_aug','f1_macro'):+.4f}"
    values['selection_ap']=f"{secondary('select_shap','select_random','average_precision'):+.4f}"
    for name,key in [('RandomForest','rf_selection'),('ExtraTrees','et_selection')]:
        values[key]=f"{primary[(primary.arm=='select_shap')&(primary.control=='select_random')&(primary.teacher==name)].mean_difference.mean():+.4f}"
    delta=val('select_shap','select_random')-val('select_shap','select_random','shared_baseline')
    values['tuning_interpretation']=f'Independent rather than shared-baseline tuning changes the aggregate selection contrast by {delta:+.4f} ROC-AUC. This is a descriptive difference between two completed evaluation procedures. The per-target and per-model changes remain available, so the aggregate cannot conceal a reversal confined to one setting.'
    values['dataset_table']=table('data','External corpus. Prevalence is the fraction of files labelled defective; all inputs use the 54 static code metrics',['Release','Files','Defective','Prevalence'],[[r.project,str(r.rows),str(r.buggy),f'{r.prevalence:.3f}'] for r in data.itertuples()])
    values['arms_table']=table('arms','Complete external treatment/control design',['Arm','Representation','Purpose'],[
        ['Untreated','Original static metrics','Absolute baseline'],['SHAP weighting','Feature times unit-sum relevance','Attribution treatment'],['Uniform weighting','Feature divided by active feature count','Scale control'],['Shuffled weighting','Permuted relevance values','Assignment control'],['SHAP selection','Top half of active features','Attribution treatment'],['Random selection','Random half of active features','Dimension control'],['Prediction augmentation','Features + OOF probability','Stacking control'],['SHAP augmentation','Features + OOF probability + SHAP','Incremental attribution treatment']])
    baseline=scores[(scores.arm=='original')&(scores.tuning=='independent')&(scores.teacher=='RandomForest')].groupby(['target','model'])[['auc','average_precision','f1_macro']].mean()
    rows=[]
    for t in targets:
        row=[t]
        for metric in ['auc','average_precision','f1_macro']:
            row.extend(f'{baseline.loc[(t,m),metric]:.3f}' for m in cfg['models'])
        rows.append(row)
    values['baseline_table']=table('baseline','Untreated absolute scores. RF: RandomForest; LR: logistic regression; AP: average precision; F1: macro F1',['Release','RF AUC','LR AUC','RF AP','LR AP','RF F1','LR F1'],rows)
    rows=[]
    for a,b,label in comparisons+[('shap_sum','shuffled_sum','Weighting vs shuffled'),('p_aug','original','Prediction augmentation vs untreated')]:
        rows.append([label.replace(' − ',' vs '),f'{val(a,b):+.4f}',f'{val(a,b,"shared_baseline"):+.4f}',f'{val(a,b,column="target_min"):+.4f}',f'{val(a,b,column="target_max"):+.4f}'])
    values['effects_table']=table('effects','Paired ROC-AUC effects. Independent and shared settings receive equal two-candidate budgets. Range is across seven target means under independent tuning; it is not an uncertainty interval',['Comparison','Independent','Shared','Min target','Max target'],rows)
    values['secondary_table']=table('secondary','Independent-tuning secondary effects, equally averaged over targets. AP: average precision; F1: macro F1 at threshold 0.5',['Comparison','AP difference','F1 difference'],[[label.replace(' − ',' vs '),f'{secondary(a,b,"average_precision"):+.4f}',f'{secondary(a,b,"f1_macro"):+.4f}'] for a,b,label in comparisons])
    rows=[]
    for t in targets:
        row=[t]
        for a,b,label in comparisons:
            row.append(f'{primary[(primary.target==t)&(primary.arm==a)&(primary.control==b)].mean_difference.mean():+.4f}')
        rows.append(row)
    values['target_table']=table('targeteffects','Independent-tuning effects by target, averaged over two teachers, two learners and three source subsamples',['Release','Weight vs uniform','Select vs random','SHAP vs prediction augmentation'],rows)
    values['workflow_figure']=figblock(1,'workflow','External evaluation flow repeated for each held-out project. Target responses enter only scoring. Within each source-validation fold the representations and teacher are rebuilt')
    values['effects_figure']=figblock(2,'effects','Independent-tuning target means for the three principal comparisons. Each panel has its own horizontal scale. Points are descriptive means, with no confidence intervals')
    values['teacher_figure']=figblock(3,'teacher','Teacher sensitivity under independent tuning. Circles denote RandomForest and squares ExtraTrees; each point averages downstream models and source repeats')
    values['stability_figure']=figblock(4,'stability','Attribution ranking agreement across the three source subsamples and between teacher-mean vectors. Symbols distinguish comparisons; correlations describe rankings, not predictive equivalence')
    values['diagnostic_figure']=figblock(5,'diagnostics','Exploratory target-level diagnostics against the selection effect. Letters A--G follow the project order in Table~\\ref{tab:data}. No regression, significance test or subgroup selection is performed')
    template=(ROOT/'strengthened/manuscript.tex.in').read_text(encoding='utf-8')
    for key,v in values.items():template=template.replace('@@'+key+'@@',v)
    if '@@' in template:raise RuntimeError('Unresolved manuscript values')
    (DEST/'main.tex').write_text(template,encoding='utf-8')
    for name in ['scores.csv','effects.csv','summary.csv','datasets.csv','diagnostics.csv','stability.csv','overlap.csv','validation.json']:
        shutil.copy2(OUT/name,DEST/'results'/name)
    dump(DEST/'build_provenance.json',{'input_hashes':{str(p.relative_to(ROOT)):digest(p) for p in [OUT/'scores.csv',OUT/'effects.csv',OUT/'summary.csv',ROOT/'strengthened/manuscript.tex.in',Path(__file__)]},'values':values,'target_venue':'Empirical Software Engineering'})
    print(json.dumps({k:v for k,v in values.items() if not k.endswith(('_table','_figure'))},indent=2))

if __name__=='__main__':build()
