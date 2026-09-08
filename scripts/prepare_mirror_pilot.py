"""Flattened mirror is ONE unknown group, never a scene-separated benchmark."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import cv2
import numpy as np
from stereoscope.data import read_pfm, sceneflow_records, save_manifest

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--pairs',type=int,default=800)
a=p.parse_args()
root=Path('data/mirror-sceneflow')
if len(list((root/'left').glob('*.pfm')))<a.pairs:
    with tarfile.open('data/downloads/mirror-disparity.tar',mode='r|') as archive:
        n=0
        for member in archive:
            if member.isfile() and member.name.startswith('left/'):
                if n>=a.pairs: break
                archive.extract(member,root,filter='data'); n+=1
records=[]; hashes=[]
for left in sorted((root/'left').glob('*.png'))[:a.pairs]:
    gt=left.with_suffix('.pfm'); right=root/'right'/left.name
    if not gt.exists() or not right.exists(): raise ValueError('Incomplete mirror pair')
    l=cv2.imread(str(left),0); r=cv2.imread(str(right),0); d=read_pfm(gt)
    if l.shape!=r.shape or d.shape!=l.shape: raise ValueError('Pair shape mismatch')
    if not np.isfinite(d).any(): raise ValueError('No finite disparity')
    records.append(dict(id='mirror-'+left.stem,group='unknown_flattened_mirror_all',
        dataset='SceneFlowMirrorUnverified',left=str(left),right=str(right),gt=str(gt),split='train'))
    hashes.append(dict(id=left.stem,left_sha256=hashlib.sha256(left.read_bytes()).hexdigest(),
        right_sha256=hashlib.sha256(right.read_bytes()).hexdigest(),
        disparity_sha256=hashlib.sha256(gt.read_bytes()).hexdigest(),shape=list(d.shape)))
calibration=[dict(r,split='calibration') for r in sceneflow_records('data/sceneflow/Sampler')
             if 'FlyingThings3D/' in r['id'] or 'FlyingThings3D_' in r['id']]
assert len(calibration)==3
save_manifest('data/mirror-pilot-manifest.json',records+calibration)
Path('outputs/mirror-pilot-provenance.json').write_text(json.dumps(dict(
    mirror='https://huggingface.co/datasets/olivermao/sceneflow',
    image_url='https://huggingface.co/datasets/olivermao/sceneflow/resolve/main/frames_clean.tar',
    disparity_url='https://huggingface.co/datasets/olivermao/sceneflow/resolve/main/disparity.tar',
    original='https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html',
    limitation='Mirror has no card, original scene identifiers or declared license; attribution to SceneFlow not independently authenticated. Research-only pilot; no raw-data redistribution. Treat all training frames as ONE unknown group. No exact RGB hash overlap with official sampler was found; that alone does not prove independence.',
    training_pairs=len(records),calibration='3 official FlyingThings3D sampler frames, independently named source; no proven absence of semantic overlap with flattened mirror',
    files=hashes),indent=2)+'\n')
print(f'{len(records)} training pairs in ONE unknown group; {len(calibration)} calibration frames')
