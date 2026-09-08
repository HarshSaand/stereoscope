"""Stream a bounded official Monkaa prefix, keeping complete scene groups.

No mirror or invented sample is used. The compressed label archive must be scanned
sequentially; abort safely at the byte budget rather than silently claim completion.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import time
import urllib.request

BASE='https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/Monkaa/'
URLS={'images': BASE+'raw_data/monkaa__frames_cleanpass_webp.tar',
      'labels': BASE+'derived_data/monkaa__disparity.tar.bz2'}


class BudgetReader:
    def __init__(self, response, budget):
        self.response=response; self.budget=budget; self.count=0
    def read(self, n=-1):
        if self.count >= self.budget: raise RuntimeError('Download byte budget reached')
        chunk=self.response.read(min(n if n>=0 else 65536,self.budget-self.count))
        self.count+=len(chunk); return chunk


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',choices=list(URLS),required=True)
    p.add_argument('--root',default='data/monkaa'); p.add_argument('--groups',type=int,default=4)
    p.add_argument('--max-gb',type=float,default=8)
    a=p.parse_args(); root=Path(a.root); root.mkdir(parents=True,exist_ok=True)
    seen=[]; files=[]; start=time.time(); status='incomplete'
    response=urllib.request.urlopen(URLS[a.kind],timeout=60)
    reader=BudgetReader(response,int(a.max_gb*1e9))
    try:
        with tarfile.open(fileobj=reader,mode='r|*') as archive:
            for member in archive:
                parts=Path(member.name).parts
                if len(parts)<2: continue
                group=parts[1]
                # x2 is an altered rendering of same scenario: require physical-group separation later.
                if group not in seen:
                    if len(seen)>=a.groups:
                        status='complete_prefix'; break
                    seen.append(group)
                    print(f'{a.kind}: scene {group}, {reader.count/1e6:.1f} MB, {(time.time()-start)/60:.1f} min',flush=True)
                if not member.isfile() or len(parts)<4: continue
                if a.kind=='labels' and parts[2]!='left': continue
                target=root.joinpath(*parts)
                if not target.resolve().is_relative_to(root.resolve()): raise ValueError('Unsafe archive path')
                target.parent.mkdir(parents=True,exist_ok=True)
                data=archive.extractfile(member).read()
                target.write_bytes(data)
                files.append(dict(path=str(target.relative_to(root)),sha256=hashlib.sha256(data).hexdigest()))
    finally:
        response.close()
        (root/f'provenance-{a.kind}.json').write_text(json.dumps(dict(url=URLS[a.kind],
            status=status,groups=seen,bytes_received=reader.count,elapsed_seconds=time.time()-start,files=files),indent=2)+'\n')


if __name__=='__main__': main()
