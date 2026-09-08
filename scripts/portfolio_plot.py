"""Metrics-only public figure: no source images or benchmark asset redistribution."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

out=Path('outputs')
sgbm=json.loads((out/'eth3d_sgbm.json').read_text())['summary_scene_macro']['sgbm_consistency']
raft=json.loads((out/'eth3d_raft.json').read_text())['summary_scene_macro']['raft_consistency']
fig,axes=plt.subplots(1,3,figsize=(13,4))
labels=['SGBM','RAFT-Stereo\npretrained']
for ax,values,title in zip(axes[:2],[[sgbm['epe'],raft['epe']],
    [100*sgbm['prediction_coverage'],100*raft['prediction_coverage']]],
    ['Disparity EPE (resized pixels)','Valid prediction coverage (%)']):
    bars=ax.bar(labels,values,color=['#999999','#222222'])
    ax.bar_label(bars,fmt='%.3f',padding=4); ax.set_title(title,fontsize=10)
    ax.spines[['top','right']].set_visible(False); ax.set_ylim(0,max(values)*1.22)
path=out/'eth3d_mirror_cnn.json'
if path.exists():
    report=json.loads(path.read_text()); methods=report['summary_scene_macro']
    names=[n for n in ('constant_training_prior','cnn','zero_error_risk') if n in methods]
    vals=[methods[n]['brier'] for n in names]
    bars=axes[2].bar(['Train prior','CNN','Zero-risk'][:len(names)],vals,color=['#999999','#222222','#cccccc'])
    axes[2].bar_label(bars,fmt='%.4f',padding=4)
    axes[2].set_ylim(0,max(vals)*1.22)
    axes[2].set_title('SGBM error-risk Brier (lower better)',fontsize=10)
else:
    axes[2].axis('off'); axes[2].text(.1,.5,'Confidence training not yet final')
fig.suptitle('StereoScope · 27 real ETH3D pairs · 640-pixel maximum width',fontsize=13)
fig.text(.5,.015,'Local scene-macro pilot, not official benchmark. CNN uses SGBM features; mirror training is one unknown group. No real-time or autonomy claim.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.065,1,.95)); fig.savefig(out/'portfolio-results.png',dpi=180)
