"""Render locally measured scene evidence and risk-coverage curves."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from stereoscope.cli import cache_record
from stereoscope.data import load_pair

root=Path('outputs'); root.mkdir(exist_ok=True)
records=json.loads(Path('data/eth3d-manifest.json').read_text())
r=next(r for r in records if r['id']=='delivery_area_1l')
s=cache_record(r,'outputs/cache'); left,_,_,_=load_pair(r,640)
fig,axes=plt.subplots(1,3,figsize=(14,4))
for ax,data,title in zip(axes,[left,np.where(s['disparity']>0,s['disparity'],np.nan),np.clip(s['consistency']/16,0,1)],
                        ['ETH3D real stereo scene','SGBM disparity (px)','Geometric inconsistency risk']):
    ax.imshow(data,cmap='gray' if title.startswith('ETH') else 'viridis'); ax.set_title(title); ax.axis('off')
fig.suptitle('StereoScope · measured stereo baseline, no neural-depth claim')
fig.tight_layout(); fig.savefig(root/'stereo-evidence.png',dpi=160); plt.close(fig)
