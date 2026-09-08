from pathlib import Path
import numpy as np
import pytest
import torch
from stereoscope.geometry import Calibration, depth_from_disparity, point_cloud, stereo_sgbm, confidence_features
from stereoscope.data import read_pfm, grouped_split
from stereoscope.metrics import evaluate, bad_disparity
from stereoscope.model import ConfidenceCNN, LinearConfidence


def test_metric_units_and_offset():
    c=Calibration(100,100,1,1,.2,2)
    d=np.array([[8.,0.,-1.,np.nan]],dtype=np.float32)
    z=depth_from_disparity(d,c)
    assert z[0,0]==pytest.approx(2)
    assert np.isnan(z[0,1:]).all()


def test_resize_preserves_depth():
    c=Calibration(100,100,10,10,.2,2)
    assert depth_from_disparity(np.array([[8.]]),c)[0,0]==depth_from_disparity(np.array([[4.]]),c.scaled(.5,.5))[0,0]


def test_nonfinite_calibration_rejected():
    with pytest.raises(ValueError): Calibration(float('nan'),100,0,0,.2)
    with pytest.raises(ValueError): Calibration(100,100,0,0,.2).scaled(-1,1)


def test_pointcloud_projection():
    c=Calibration(100,100,0,0,.2)
    assert np.allclose(point_cloud(np.ones((2,2))*2,c,stride=1)[-1],[.02,.02,2])


def test_pfm_endian_and_vertical_flip(tmp_path):
    p=tmp_path/'x.pfm'; data=np.array([[1,2],[3,4]],dtype='<f4')
    p.write_bytes(b'Pf\n2 2\n-1.0\n'+data.tobytes())
    assert np.array_equal(read_pfm(p),[[3,4],[1,2]])


def test_split_no_group_leakage():
    records=[dict(group=str(g),id=f'{g}-{i}') for g in range(8) for i in range(3)]
    split=grouped_split(records)
    for g in range(8): assert len({r['split'] for r in split if r['group']==str(g)})==1
    assert {r['split'] for r in grouped_split([dict(group=str(g)) for g in range(3)])}=={'train','calibration','test'}


def test_bad_pixel_requires_both_conditions():
    assert not bad_disparity(np.array([104.]),np.array([100.]))[0]
    assert bad_disparity(np.array([110.]),np.array([100.]))[0]


def test_invalid_predictions_count_as_failures():
    result=evaluate(np.array([[-1.,10.]]),np.array([[10.,10.]]),np.array([[1.,0.]]))
    assert result['d1_all_labelled']==.5
    assert result['d1_valid']==0
    assert result['risk_coverage'][-1]['retained_labelled_coverage']==.5


def test_constant_risk_ties_cannot_exploit_pixel_order():
    result=evaluate(np.array([[10.,10.,20.,20.]]),np.array([[10.,10.,10.,10.]]),np.ones((1,4))*.5)
    assert all(r['d1']==.5 for r in result['risk_coverage'])


def test_confidence_forward_backward():
    for model in (ConfidenceCNN(),LinearConfidence()):
        output=model(torch.randn(2,6,16,16))
        assert output.shape==(2,16,16)
        output.mean().backward()
        assert all(p.grad is not None for p in model.parameters())


def test_stereo_known_shift_fixture():
    rng=np.random.default_rng(12)
    left=rng.integers(0,256,(80,256),dtype=np.uint8)
    right=np.roll(left,-8,axis=1)
    disp,rd=stereo_sgbm(left,right,32)
    assert abs(np.median(disp[10:-10,45:-20])-8)<.5
    features,consistency=confidence_features(left,right,disp,rd)
    assert features.shape==(6,80,256)
    assert np.isfinite(features).all()
    assert np.median(consistency[10:-10,45:-45])<1
