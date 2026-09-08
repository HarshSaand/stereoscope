"""Fetch the explicitly unverified SceneFlow mirror for a one-group pilot.

Underlying SceneFlow research-only terms apply; raw assets are never published.
The mirror has no scene metadata. This command does not authenticate provenance.
"""
import argparse
from pathlib import Path
import tarfile
import urllib.request

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--pairs',type=int,default=300)
a=p.parse_args()
if not 1<=a.pairs<=800: raise ValueError('Mirror only contains 800 numbered pairs')
root=Path('data/mirror-sceneflow'); root.mkdir(parents=True,exist_ok=True)
for filename in ('frames_clean.tar','disparity.tar'):
    url='https://huggingface.co/datasets/olivermao/sceneflow/resolve/main/'+filename
    print('Streaming',url,flush=True)
    with urllib.request.urlopen(url,timeout=60) as response:
        with tarfile.open(fileobj=response,mode='r|') as archive:
            extracted=0
            for member in archive:
                if not member.isfile(): continue
                name=Path(member.name)
                if name.parent.name not in ('left','right'): continue
                if filename=='disparity.tar' and name.parent.name=='right': break
                if int(name.stem)>a.pairs: continue
                target=root/name
                if not target.resolve().is_relative_to(root.resolve()): raise ValueError('Unsafe path')
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(archive.extractfile(member).read())
                extracted+=1
                if extracted>=a.pairs*(2 if filename=='frames_clean.tar' else 1): break
    print('Extracted',extracted,'files',flush=True)
