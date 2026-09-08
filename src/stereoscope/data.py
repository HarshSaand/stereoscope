"""Public dataset adapters; assets stay outside version control."""
from pathlib import Path
import hashlib
import json
import numpy as np
import cv2


def read_pfm(path):
    with open(path, "rb") as f:
        kind = f.readline().decode().strip()
        if kind not in ("Pf", "PF"):
            raise ValueError("Not a PFM file")
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()
        width, height = map(int, line.split())
        scale = float(f.readline())
        if width<=0 or height<=0 or width*height>100_000_000:
            raise ValueError("Invalid or excessive PFM dimensions")
        if not np.isfinite(scale) or scale == 0:
            raise ValueError("Invalid PFM scale")
        data = np.fromfile(f, dtype="<f4" if scale < 0 else ">f4")
        shape = (height, width, 3) if kind == "PF" else (height, width)
        return np.flipud(data.reshape(shape)).astype(np.float32) * abs(scale)


def eth_records(root):
    records = []
    for left in sorted(Path(root).rglob("im0.png")):
        scene = left.parent
        gt = scene/"disp0GT.pfm"
        if gt.exists():
            # All captures of one physical environment share a group.
            group = scene.name.rsplit("_", 1)[0]
            records.append(dict(id=scene.name, group=group, dataset="ETH3D",
                left=str(left), right=str(scene/"im1.png"), gt=str(gt),
                calibration=str(scene/"calib.txt")))
    return records


def sceneflow_records(root):
    """Original SceneFlow layout: frames_cleanpass/.../left/*.png + disparity/..."""
    root = Path(root)
    records = []
    for left in sorted(list(root.rglob("*.png"))+list(root.rglob("*.webp"))):
        if left.parent.name != "left":
            continue
        pass_name = next((p for p in ('frames_cleanpass','frames_cleanpass_webp','RGB_cleanpass') if p in left.parts),None)
        if not pass_name: continue
        right = left.parent.parent/"right"/left.name
        gt = Path(str(left).replace(pass_name, "disparity")).with_suffix(".pfm")
        if pass_name == 'RGB_cleanpass': gt=gt.parent.parent/gt.name
        if gt.exists() and right.exists():
            group = str(left.parent.parent.relative_to(root)).replace('_x2','')
            records.append(dict(id=str(left.relative_to(root)).replace("/", "_"),
                group=group, dataset="SceneFlow", left=str(left), right=str(right), gt=str(gt)))
    return records


def grouped_split(records, seed=42):
    groups = sorted({r["group"] for r in records},
                    key=lambda g: hashlib.sha256(f"{seed}:{g}".encode()).hexdigest())
    if len(groups) < 3:
        raise ValueError("At least three independent scene groups required")
    n_train = min(len(groups)-2, max(1, int(.7*len(groups))))
    n_val = max(1, int(.15*len(groups)))
    lookup = {g: "train" if i<n_train else "calibration" if i<n_train+n_val else "test"
              for i, g in enumerate(groups)}
    return [dict(r, split=lookup[r["group"]]) for r in records]


def load_pair(record, max_width=None):
    left = cv2.imread(record["left"], cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(record["right"], cv2.IMREAD_GRAYSCALE)
    gt = read_pfm(record["gt"])
    if left is None or right is None or gt.shape != left.shape:
        raise ValueError(f"Invalid image/disparity pair: {record['id']}")
    scale = 1.0
    if max_width and left.shape[1] > max_width:
        scale = max_width/left.shape[1]
        size = (max_width, round(left.shape[0]*scale))
        left = cv2.resize(left, size, interpolation=cv2.INTER_AREA)
        right = cv2.resize(right, size, interpolation=cv2.INTER_AREA)
        gt = cv2.resize(gt, size, interpolation=cv2.INTER_NEAREST)*scale
    return left, right, gt, scale


def save_manifest(path, records):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(records, indent=2)+"\n")
