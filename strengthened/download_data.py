"""Retrieve only the pinned primary data, then verify every SHA-256 digest.

Run from the artifact root: python -m strengthened.download_data
"""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'data_external/download_manifest.json').read_text())
for item in manifest:
    dest=(ROOT/item['path']).resolve()
    if not dest.is_relative_to((ROOT/'data_external').resolve()):raise ValueError('Unsafe destination')
    if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest()==item['sha256']:
        print('verified',dest.name);continue
    payload=urllib.request.urlopen(item['url'],timeout=60).read()
    if hashlib.sha256(payload).hexdigest()!=item['sha256']:raise RuntimeError('Upstream content mismatch')
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(payload);print('downloaded',dest.name)
