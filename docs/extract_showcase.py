from pathlib import Path
import argparse, json, hashlib, sys
p=argparse.ArgumentParser();p.add_argument('--source', type=Path, required=True);a=p.parse_args()
S=a.source.resolve(); D=Path(__file__).resolve().parent; R=D.parent
D.mkdir(exist_ok=True)
import numpy as np
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for c in iter(lambda:f.read(1048576),b''):h.update(c)
 return h.hexdigest()
def source(rel):return {'path':rel,'sha256':digest(S/rel)}
def write(data):
 import subprocess
 data['dataset_url']='https://www.eth3d.net/datasets'
 data['source_code_commit']=subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip()
 data['extractor_sha256']=digest(Path(__file__))
 data['sources']=sources
 data['source_repository']='https://github.com/HarshSaand/'+R.name
 data['extraction']='python docs/extract_showcase.py --source /path/to/reproduced/project'
 (D/'output-example.json').write_text(json.dumps(data,indent=2,ensure_ascii=False,default=str)+'\n')
import cv2
from PIL import Image
sys.path.insert(0,str(R/'src'));from stereoscope.geometry import stereo_sgbm,read_eth_calibration,depth_from_disparity,point_cloud
scene='delivery_area_1l';base=f'data/eth3d/{scene}';left=cv2.imread(str(S/base/'im0.png'),0);right=cv2.imread(str(S/base/'im1.png'),0);scale=min(1.,640/left.shape[1]);sz=(round(left.shape[1]*scale),round(left.shape[0]*scale));left=cv2.resize(left,sz);right=cv2.resize(right,sz)
disp,_=stereo_sgbm(left,right);cal=read_eth_calibration(S/base/'calib.txt').scaled(scale,scale);depth=depth_from_disparity(disp,cal);points=point_cloud(depth,cal,stride=16,max_depth=30)
valid=np.isfinite(depth);v=disp>0;lo,hi=np.percentile(disp[v],[2,98]);mapped=np.uint8(np.clip((disp-lo)/(hi-lo),0,1)*255);rgb=cv2.cvtColor(cv2.applyColorMap(mapped,cv2.COLORMAP_TURBO),cv2.COLOR_BGR2RGB);rgb[~v]=[12,18,22]
Image.fromarray(left).save(D/'left-input.png');Image.fromarray(right).save(D/'right-input.png');Image.fromarray(rgb).save(D/'disparity-output.png');np.savez_compressed(D/'disparity-depth-output.npz',disparity=disp,depth_metres=depth);np.savetxt(D/'point-cloud-sample.csv',points[:500],delimiter=',',header='x_metres,y_metres,z_metres',comments='')
sources=[source(base+'/'+f) for f in ['im0.png','im1.png','calib.txt']]
write(dict(title='StereoScope',subtitle='Two real images become a disparity and depth field',eyebrow='ACTUAL SGBM INFERENCE OUTPUT',context=f'ETH3D {scene} · calibrated rectified pair · {sz[0]} × {sz[1]} inference',images=[dict(path='left-input.png',label='Left input'),dict(path='right-input.png',label='Right input'),dict(path='disparity-output.png',label='Predicted disparity · near = warm')],raw=dict(scene=scene,method='repository OpenCV SGBM baseline',valid_fraction=float(v.mean()),median_valid_depth_metres=float(np.nanmedian(depth)),disparity_color_range=[float(lo),float(hi)],point_cloud_sample_rows=min(500,len(points)),calibration=cal.__dict__,array_file='disparity-depth-output.npz'),note='Dark pixels are invalid estimates, not zero depth. Color visualizes disparity. This is the implemented geometric baseline; it is not a newly trained stereo network. Full disparity/depth arrays and a sampled XYZ cloud are downloadable.',input='Real rectified left/right images and camera calibration',output='Disparity map, metric-depth array and 3D point sample'))
