"""Prepare flat journal sources and the portable primary research artifact."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import zipfile
from .experiment import ROOT,OUT,digest,dump

PAPER=ROOT/'paper/emse'
def makezip(directory,target):
    files=sorted(p for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='SHA256.json' and not p.relative_to(directory).as_posix().startswith('data_external/Rnalytica/'))
    hashes={p.relative_to(directory).as_posix():digest(p) for p in files}
    dump(directory/'SHA256.json',hashes)
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files+[directory/'SHA256.json']:z.write(p,p.relative_to(directory).as_posix())
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        assert all(hashlib.sha256(z.read(n)).hexdigest()==v for n,v in hashes.items())
    return {'files':len(files)+1,'bytes':target.stat().st_size,'sha256':digest(target)}

def package():
    validation=json.loads((OUT/'validation.json').read_text());assert validation['valid']
    flat=PAPER/'upload';flat.mkdir(exist_ok=True)
    text=(PAPER/'main.tex').read_text(encoding='utf-8').replace('figures/','').replace('tables/','')
    (flat/'main.tex').write_text(text,encoding='utf-8')
    for name in ['references.bib','svjour3.cls','svglov3.clo','spbasic.bst','main.bbl','build.ps1','springer_readme.txt']:
        shutil.copy2(PAPER/name,flat/name)
    for sub in ['figures','tables']:
        for p in (PAPER/sub).iterdir():
            if p.is_file():shutil.copy2(p,flat/p.name)
    for path in re.findall(r'\\(?:input|includegraphics)(?:\[[^\]]*\])?\{([^}]+)\}',text):assert (flat/path).is_file(),path
    primary=PAPER/'artifact_primary';primary.mkdir(exist_ok=True)
    for folder in ['strengthened','data_external','paper/emse','outputs/revised/strengthening/full']:(primary/folder).mkdir(parents=True,exist_ok=True)
    for p in (ROOT/'strengthened').iterdir():
        if p.is_file() and p.name!='preview.py':shutil.copy2(p,primary/'strengthened'/p.name)
    shutil.copy2(ROOT/'requirements-audit.lock.txt',primary/'requirements-audit.lock.txt')
    for name in ['download_manifest.json','local_alignment.json']:
        shutil.copy2(ROOT/'data_external'/name,primary/'data_external'/name)
    for name in ['main.tex','references.bib','svjour3.cls','svglov3.clo','spbasic.bst','build.ps1','README.md','AUTHOR_ACTIONS.md','springer_readme.txt']:
        shutil.copy2(PAPER/name,primary/'paper/emse'/name)
    for folder in ['figures','tables']:
        shutil.copytree(PAPER/folder,primary/'paper/emse'/folder,dirs_exist_ok=True)
    for p in OUT.iterdir():
        if p.is_file() and p.suffix in ['.json','.npz','.csv']:shutil.copy2(p,primary/'outputs/revised/strengthening/full'/p.name)
    (primary/'README.md').write_text('''# Online Resource 1: primary external research artifact

Paper: Do Explanations Transfer? Separating Attribution Information from Transformation Effects in Defect Prediction
Intended journal: Empirical Software Engineering
Authors: Priyanshu Kumar, Kumar Rajnish, Shubham Kumar
Affiliation: Birla Institute of Technology, Mesra, Ranchi, India
Correspondence: priyanshu.kumar@bitmesra.ac.in

See strengthened/README.md for the full protocol and reproduction commands.
Start with a Python 3.12 environment and requirements-audit.lock.txt, then run
python -m strengthened.download_data. Downloads are pinned and SHA-256 verified.
Run python -m strengthened.analyze to verify the supplied complete predictions.
Run python -m strengthened.build_paper and compile in paper/emse to regenerate
the primary manuscript and figures. The completed checkpoint files are included;
an independent training run requires a fresh copy without the completed outputs.

This artifact includes all primary experiment source, inputs manifest, settings,
split indices, per-instance predictions, labels used for score validation, raw
score grids, diagnostics and presentation sources. Upstream CSVs are retrieved
from their authors rather than bundled under a new licence. The original local
Eclipse files are not needed to reproduce this primary study. Raw labels in the
predictions originate from the cited upstream dataset. Preserve that attribution.

This is a locally prepared artifact, not a public archive with an assigned DOI.
Pending author declarations in the draft must be resolved before submission.
''',encoding='utf-8')
    secondary=PAPER/'artifact_previous';secondary.mkdir(exist_ok=True)
    shutil.copy2(ROOT/'paper/submission/main.pdf',secondary/'Earlier_exploratory_manuscript.pdf')
    shutil.copy2(ROOT/'paper/Do_Explanations_Transfer_LaTeX.zip',secondary/'Earlier_source_and_results.zip')
    (secondary/'README.md').write_text('''# Online Resource 2: earlier exploratory experiments

Companion to: Do Explanations Transfer? Separating Attribution Information from Transformation Effects in Defect Prediction
Intended journal: Empirical Software Engineering
Authors: Priyanshu Kumar, Kumar Rajnish, Shubham Kumar
Birla Institute of Technology, Mesra, Ranchi, India
Correspondence: priyanshu.kumar@bitmesra.ac.in

The enclosed earlier manuscript and source/result summary package retain the
fixed-source experiments for traceability. They are supplementary, not the
current primary manuscript. Their Eclipse history-feature timing remains
unverified. They use a different protocol and must not be pooled with Online
Resource 1. No document inside this historical package overrides that distinction.
The complete earlier per-instance prediction files remain in the original
research workspace outputs/revised/full; this resource is the earlier summary
and source package and is not claimed to independently reproduce its unverified
raw-data acquisition.
''',encoding='utf-8')
    # Journal upload sources must contain no subdirectories or compiler by-products.
    upload_files=[p for p in flat.iterdir() if p.is_file() and p.suffix in ['.tex','.bib','.cls','.clo','.bst','.bbl','.ps1','.txt','.pdf','.eps','.png'] and p.name!='main.pdf']
    with zipfile.ZipFile(PAPER/'EMSE_LaTeX_Upload.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in upload_files:z.write(p,p.name)
    with zipfile.ZipFile(PAPER/'EMSE_LaTeX_Upload.zip') as z:
        assert z.testzip() is None and all('/' not in n for n in z.namelist())
    reports={'primary_artifact':makezip(primary,PAPER/'ESM_1.zip'),'earlier_artifact':makezip(secondary,PAPER/'ESM_2.zip'),'flat_latex_sha256':digest(PAPER/'EMSE_LaTeX_Upload.zip'),'flat_latex_files':len(upload_files)}
    dump(PAPER/'artifact_validation.json',reports);print(json.dumps(reports,indent=2))

if __name__=='__main__':package()
