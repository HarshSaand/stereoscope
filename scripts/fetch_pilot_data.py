"""Download official research assets; never upload datasets to GitHub."""
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request
import py7zr

SOURCES={
 'two_view_training.7z':'https://www.eth3d.net/data/two_view_training.7z',
 'two_view_training_gt.7z':'https://www.eth3d.net/data/two_view_training_gt.7z',
 'SceneFlow-Sampler.tar.gz':'https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlow/assets/Sampler.tar.gz'}


def main():
    root=Path('data/downloads'); root.mkdir(parents=True,exist_ok=True)
    records=[]
    for name,url in SOURCES.items():
        path=root/name
        if not path.exists():
            print('Downloading',url,flush=True); urllib.request.urlretrieve(url,path)
        records.append(dict(file=name,url=url,bytes=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        if name.endswith('.7z'):
            with py7zr.SevenZipFile(path) as archive: archive.extractall('data/eth3d')
        else:
            with tarfile.open(path) as archive: archive.extractall('data/sceneflow',filter='data')
    Path('outputs').mkdir(exist_ok=True)
    Path('outputs/data-provenance.json').write_text(json.dumps(records,indent=2)+'\n')


if __name__=='__main__': main()
