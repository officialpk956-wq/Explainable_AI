"""Package the v3 submission. Same flat-upload rules; v3 results and counts.

python -m strengthened.package_v3
"""
import json

from .experiment import dump
from .experiment_v3 import OUT, config
from .package_v2 import PAPER, build_upload, build_artifact


def package():
    validation = json.loads((OUT / 'validation.json').read_text())
    if not validation.get('valid'):
        raise RuntimeError('valid v3 results are required before packaging')
    upload = build_upload()
    artifact = build_artifact()
    cfg = config()
    report = {'upload': upload, 'artifact': artifact, 'upload_is_flat': True,
              'figure_formats': ['pdf', 'eps'],
              'score_records': validation['score_records_reconstructed'],
              'checkpoints': validation['checkpoints'], 'arms': len(cfg['arms']),
              'seeds': len(cfg['seeds']), 'learners': len(cfg['models']),
              'control_draws': cfg['control_draws'], 'teacher_cv': cfg['teacher_cv'],
              'sensitivity_runs': sorted(cfg['sensitivity_runs'])}
    dump(PAPER / 'PACKAGE_V3.json', report)
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    package()
