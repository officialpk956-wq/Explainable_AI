"""Hash the two frozen protocol specs and the preregistration document together
into one manifest, so the three files can be checked as a set later.

python -m strengthened.freeze_preregistration

This is a local, checkable record of what was frozen and when -- it is not a
substitute for a real OSF registration, which independently timestamps on
OSF's own platform. Run this once after PREREGISTRATION.md and both protocol
files are final; rerunning it after editing any of the three means the earlier
"frozen" record was not actually frozen, so a rerun that changes the recorded
hashes is flagged rather than silently overwritten.
"""
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .experiment import ROOT, dump

FILES = {
    'preregistration': ROOT / 'paper/emse/PREREGISTRATION.md',
    'protocol_decorrelation': ROOT / 'strengthened/protocol_decorrelation.json',
    'protocol_labelnoise': ROOT / 'strengthened/protocol_labelnoise.json',
}
MANIFEST = ROOT / 'paper/emse/PREREGISTRATION_MANIFEST.json'


def sha256(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git_commit():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def run():
    hashes = {name: sha256(p) for name, p in FILES.items()}
    if MANIFEST.exists():
        prior = json.loads(MANIFEST.read_text(encoding='utf-8'))
        changed = {n: (prior['hashes'].get(n), h) for n, h in hashes.items()
                  if prior['hashes'].get(n) != h}
        if changed:
            raise RuntimeError(
                f'{MANIFEST.name} already exists and records different hashes for '
                f'{list(changed)} -- one of the three files changed after it was first '
                f'frozen at {prior["frozen_utc"]}. If this is a deliberate, disclosed '
                f'revision, move the old manifest aside first; do not silently overwrite '
                f'a frozen record. Diffs: {changed}')
        print(f'manifest already frozen at {prior["frozen_utc"]}, all three hashes match; '
              'nothing to do')
        return prior

    manifest = {
        'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'git_commit': git_commit(),
        'hashes': hashes,
        'note': ('SHA-256 of the three files as committed at freeze time. This manifest is '
                'a local, checkable record -- not an OSF registration. Verify by '
                're-hashing each file and comparing to the values here.'),
    }
    dump(MANIFEST, manifest)
    print(f'wrote {MANIFEST}')
    for name, h in hashes.items():
        print(f'  {name:<24} {h}')
    return manifest


if __name__ == '__main__':
    run()
