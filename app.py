"""Local inspection workbench: no uploaded assets leave this process."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import torch
from stereoscope.cli import cache_record, load_model
from stereoscope.data import load_pair
from stereoscope.geometry import read_eth_calibration, depth_from_disparity, point_cloud
from stereoscope.metrics import evaluate

st.set_page_config(page_title='StereoScope',layout='wide')
st.title('StereoScope · Confidence-Aware Stereo Perception')
st.caption('Inspect measured disparity and its failure regions. Research prototype—not navigation or collision avoidance.')
manifest=Path('data/eth3d-manifest.json')
if not manifest.exists():
    st.info('Download ETH3D and run the prepare command in README first.'); st.stop()
records=json.loads(manifest.read_text())
choice=st.sidebar.selectbox('Held-out scene',[r['id'] for r in records])
record=next(r for r in records if r['id']==choice)
width=st.sidebar.selectbox('Evaluation width',[640,480,1024])
backend=st.sidebar.selectbox('Disparity backend',['sgbm','raft'])
checkpoint_paths=sorted(Path('checkpoints').glob('*.pt'))
checkpoint=st.sidebar.selectbox('Confidence',['Geometry heuristic']+[str(p) for p in checkpoint_paths])
left,right,gt,scale=load_pair(record,width)
s=cache_record(record,'outputs/cache',width,backend)
confidence_name='Uncalibrated consistency heuristic'
risk=np.clip(s['consistency']/16,0,1)
if checkpoint != 'Geometry heuristic':
    model,metadata=load_model(checkpoint,torch.device('cpu'))
    if width!=metadata['max_width']:
        st.error('Choose checkpoint calibration width '+str(metadata['max_width'])); st.stop()
    if metadata.get('disparity_backend','sgbm')!=backend:
        st.error('Confidence checkpoint was trained for a different disparity backend.'); st.stop()
    with torch.no_grad():
        risk=torch.sigmoid(model(torch.from_numpy(s['features'])[None])/metadata['temperature'])[0].numpy()
    confidence_name=f"{metadata['architecture']}: {metadata['training_pairs']} training pairs, {metadata['training_groups']} scenes"
    st.warning('Small-sampler checkpoints only prove the training pipeline.' if metadata['training_pairs']<=9 else 'Mirror pilot: training frames form one unknown group; broad scene generalisation is not established.')
threshold=st.sidebar.slider('Retain risk ≤',0.,1.,.5)
cal=read_eth_calibration(Path(record['calibration']))
original_height = __import__('cv2').imread(record['left'],0).shape[0]
cal=cal.scaled(scale,left.shape[0]/original_height)
depth=depth_from_disparity(s['disparity'],cal)
depth[risk>threshold]=np.nan
columns=st.columns(2)
columns[0].image(left,caption='Left rectified camera',clamp=True)
columns[1].image(right,caption='Right rectified camera',clamp=True)
fig,axes=plt.subplots(1,3,figsize=(15,4))
for ax,data,title in zip(axes,[s['disparity'],risk,depth],[backend.upper()+' disparity (px)',confidence_name,'Retained depth (metres)']):
    shown=ax.imshow(data,vmin=0,vmax=1 if title==confidence_name else np.nanpercentile(data,95) if np.isfinite(data).any() else 1)
    ax.set_title(title,fontsize=9); ax.axis('off'); fig.colorbar(shown,ax=ax,fraction=.035)
st.pyplot(fig); plt.close(fig)
measures=evaluate(s['disparity'],gt,risk)
cols=st.columns(4)
cols[0].metric('Valid prediction EPE',f"{measures['epe']:.3f} px")
cols[1].metric('Labelled coverage',f"{measures['prediction_coverage']:.1%}")
cols[2].metric('All-labelled custom D1',f"{measures['d1_all_labelled']:.1%}")
cols[3].metric('Bad1 valid predictions',f"{measures['bad1_valid']:.1%}")
st.caption('Custom D1 uses >3px AND >5% error at displayed resolution; these are not official ETH3D benchmark scores. Invalid predictions count as failures in all-labelled D1.')
st.line_chart(pd.DataFrame(measures['risk_coverage']).set_index('retained_valid_coverage')[['d1']])
points=point_cloud(depth,cal,stride=8,max_depth=50)
if len(points):
    fig=plt.figure(figsize=(9,5)); ax=fig.add_subplot(projection='3d')
    ax.scatter(points[:,0],points[:,2],-points[:,1],c=points[:,2],s=.3,cmap='viridis')
    ax.set(xlabel='X (m)',ylabel='Depth Z (m)',zlabel='-Y (m)',title='Calibrated retained point cloud')
    st.pyplot(fig); plt.close(fig)
st.download_button('Download evaluation JSON',json.dumps(measures,indent=2),f'{choice}-evaluation.json')
