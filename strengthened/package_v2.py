"""Build the EMSE submission package and the reproducibility artifact.

python -m strengthened.package_v2

EMSE does not accept subfolders in a LaTeX upload, so the upload directory is
flat and every \\input and \\includegraphics path is rewritten to a bare
filename. The local build under paper/emse keeps figures/ and tables/ for
convenience; the flat copy is what gets uploaded.
"""
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

from .experiment import ROOT, digest, dump
from .experiment_v2 import OUT, config

PAPER = ROOT / 'paper/emse'
UPLOAD = PAPER / 'upload'
ARTIFACT = PAPER / 'artifact'
CLASS_FILES = ['svjour3.cls', 'svglov3.clo', 'spbasic.bst']


def makezip(directory, target, skip=()):
    files = sorted(p for p in directory.rglob('*')
                   if p.is_file() and '__pycache__' not in p.parts
                   and p.name != 'SHA256.json' and p.name not in skip)
    hashes = {p.relative_to(directory).as_posix(): digest(p) for p in files}
    dump(directory / 'SHA256.json', hashes)
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in files + [directory / 'SHA256.json']:
            z.write(p, p.relative_to(directory).as_posix())
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None, 'archive CRC failure'
        assert not [n for n in z.namelist() if '\\' in n], 'windows separators in archive'
        for n, v in hashes.items():
            assert hashlib.sha256(z.read(n)).hexdigest() == v, f'{n} altered in archive'
    return {'files': len(files) + 1, 'bytes': target.stat().st_size, 'sha256': digest(target)}


def build_upload():
    if UPLOAD.exists():
        shutil.rmtree(UPLOAD)
    UPLOAD.mkdir(parents=True)
    text = (PAPER / 'main.tex').read_text(encoding='utf-8')
    flat = text.replace('figures/', '').replace('tables/', '')
    # inputs are inlined so the upload contains one editable source file per table
    (UPLOAD / 'main.tex').write_text(flat, encoding='utf-8')
    for name in CLASS_FILES + ['references.bib']:
        shutil.copy2(PAPER / name, UPLOAD / name)
    if (PAPER / 'main.bbl').exists():
        shutil.copy2(PAPER / 'main.bbl', UPLOAD / 'main.bbl')
    for sub in ['figures', 'tables']:
        for p in (PAPER / sub).iterdir():
            # ship vector PDF and EPS; the PNG previews are not needed for typesetting
            if p.is_file() and p.suffix.lower() != '.png':
                shutil.copy2(p, UPLOAD / p.name)
    missing = [r for r in re.findall(r'\\(?:input|includegraphics)(?:\[[^\]]*\])?\{([^}]+)\}', flat)
               if not (UPLOAD / r).is_file() and not (UPLOAD / (r + '.tex')).is_file()
               and not (UPLOAD / (r + '.pdf')).is_file()]
    if missing:
        raise RuntimeError(f'flat upload is missing referenced files: {missing}')
    if any(p.is_dir() for p in UPLOAD.iterdir()):
        raise RuntimeError('EMSE uploads must be flat; a subfolder is present')
    return makezip(UPLOAD, PAPER / 'EMSE_submission_flat.zip')


def build_artifact():
    if ARTIFACT.exists():
        shutil.rmtree(ARTIFACT)
    for folder in ['strengthened', 'results', 'paper']:
        (ARTIFACT / folder).mkdir(parents=True)
    for p in (ROOT / 'strengthened').iterdir():
        if p.is_file() and p.suffix in {'.py', '.json', '.md', '.in'}:
            shutil.copy2(p, ARTIFACT / 'strengthened' / p.name)
    for name in ['scores.csv', 'summary.csv', 'effects.csv', 'loto.csv', 'augmentation.csv',
                 'cell_signs.csv', 'datasets.csv', 'diagnostics.csv', 'stability.csv',
                 'weights.csv', 'validation.json', 'manifest.json']:
        if (OUT / name).exists():
            shutil.copy2(OUT / name, ARTIFACT / 'results' / name)
    cap = OUT.parent / 'cap800'
    if (cap / 'scores.csv').exists():
        shutil.copy2(cap / 'scores.csv', ARTIFACT / 'results' / 'scores_sourcecap800.csv')
    for name in ['main.tex', 'main.pdf', 'references.bib', 'RESULTS_CHANGE_REPORT.md',
                 'AUTHOR_ACTIONS.md', 'README.md'] + CLASS_FILES:
        if (PAPER / name).exists():
            shutil.copy2(PAPER / name, ARTIFACT / 'paper' / name)
    if (ROOT / 'requirements-audit.lock.txt').exists():
        shutil.copy2(ROOT / 'requirements-audit.lock.txt', ARTIFACT / 'requirements.lock.txt')
    for name in ['download_manifest.json']:
        src = ROOT / 'data_external' / name
        if src.exists():
            shutil.copy2(src, ARTIFACT / 'results' / name)
    return makezip(ARTIFACT, PAPER / 'EMSE_reproducibility_artifact.zip')


def package():
    validation = json.loads((OUT / 'validation.json').read_text())
    if not validation.get('valid'):
        raise RuntimeError('Valid v2 results are required before packaging')
    upload = build_upload()
    artifact = build_artifact()
    cfg = config()
    report = {'upload': upload, 'artifact': artifact,
              'upload_is_flat': True, 'figure_formats': ['pdf', 'eps'],
              'score_records': validation['score_records_reconstructed'],
              'checkpoints': validation['checkpoints'], 'arms': len(cfg['arms']),
              'seeds': len(cfg['seeds']), 'learners': len(cfg['models'])}
    dump(PAPER / 'PACKAGE_V2.json', report)
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    package()
