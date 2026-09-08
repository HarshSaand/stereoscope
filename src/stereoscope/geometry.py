"""Rectified stereo conventions: d=x_left-x_right, baseline positive."""
from dataclasses import dataclass
import re
import numpy as np
import cv2


@dataclass(frozen=True)
class Calibration:
    fx: float
    fy: float
    cx: float
    cy: float
    baseline: float
    doffs: float = 0.0
    unit: str = "metres"

    def __post_init__(self):
        if not all(np.isfinite(x) for x in (self.fx,self.fy,self.cx,self.cy,self.baseline,self.doffs)):
            raise ValueError("Calibration values must be finite")
        if self.fx <= 0 or self.fy <= 0 or self.baseline <= 0:
            raise ValueError("Focal lengths and baseline must be positive")

    def scaled(self, sx, sy):
        if not (np.isfinite(sx) and np.isfinite(sy) and sx>0 and sy>0):
            raise ValueError("Scale factors must be finite and positive")
        return Calibration(self.fx*sx, self.fy*sy, (self.cx+.5)*sx-.5,
                           (self.cy+.5)*sy-.5, self.baseline, self.doffs*sx, self.unit)


def read_eth_calibration(path):
    text = path.read_text()
    matrix = re.search(r"cam0=\[([^]]+)\]", text).group(1)
    values = [float(x) for x in re.split(r"[;\s]+", matrix.strip())]
    fields = dict(re.findall(r"(\w+)=([^\n]+)", text))
    return Calibration(values[0], values[4], values[2], values[5],
                       float(fields["baseline"])/1000, float(fields["doffs"]))


def depth_from_disparity(disparity, calibration):
    denominator = np.asarray(disparity, dtype=np.float32) + calibration.doffs
    valid = np.isfinite(disparity) & (disparity > 0) & (denominator > 1e-6)
    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    depth[valid] = calibration.fx * calibration.baseline / denominator[valid]
    return depth


def point_cloud(depth, calibration, stride=4, max_depth=100):
    y, x = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    z = depth[::stride, ::stride]
    points = np.stack(((x-calibration.cx)*z/calibration.fx,
                       (y-calibration.cy)*z/calibration.fy, z), axis=-1)
    return points[np.isfinite(z) & (z > 0) & (z < max_depth)]


def stereo_sgbm(left, right, max_disparity=128):
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("Expected equally sized grayscale rectified images")
    if left.dtype != np.uint8 or right.dtype != np.uint8:
        raise ValueError("SGBM expects uint8 images")
    nd = max(16, int(np.ceil(max_disparity/16))*16)
    if left.shape[1] <= nd + 5:
        raise ValueError("Image width must exceed disparity search range")
    matcher = cv2.StereoSGBM_create(minDisparity=0, numDisparities=nd,
        blockSize=5, P1=8*25, P2=32*25, disp12MaxDiff=1,
        uniquenessRatio=10, speckleWindowSize=50, speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    left_disp = matcher.compute(left, right).astype(np.float32)/16
    # Flipping both images turns the right-reference disparity positive.
    right_disp = np.fliplr(matcher.compute(np.fliplr(right).copy(),
                          np.fliplr(left).copy())).astype(np.float32)/16
    return left_disp, right_disp


def confidence_features(left, right, disparity, right_disparity):
    """No ground truth enters these six prediction-time feature channels."""
    y, x = np.indices(disparity.shape, dtype=np.float32)
    xr = x-disparity
    warped = cv2.remap(right.astype(np.float32)/255, xr, y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT)
    rd = cv2.remap(right_disparity, xr, y, cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
    valid = (disparity > 0) & (xr >= 0) & (xr <= left.shape[1]-1) & (rd > 0)
    consistency = np.where(valid, np.abs(disparity-rd), 16.)
    lum = left.astype(np.float32)/255
    texture = np.abs(cv2.Laplacian(lum, cv2.CV_32F))
    gradient = np.abs(cv2.Sobel(disparity, cv2.CV_32F, 1, 0, ksize=3))/8
    features = np.stack([lum, np.clip(disparity/128, 0, 4),
        np.abs(lum-warped), np.clip(consistency/16, 0, 1),
        np.clip(texture, 0, 1), np.clip(gradient/16, 0, 1)])
    return np.nan_to_num(features).astype(np.float32), consistency
