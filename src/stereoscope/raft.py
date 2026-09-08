"""Optional upstream SceneFlow-only RAFT-Stereo comparator (not trained here)."""
from pathlib import Path
from types import SimpleNamespace
import sys
import time
import numpy as np
import torch


class RaftPredictor:
    def __init__(self, checkpoint, repo, device="auto", iters=32):
        repo = Path(repo).resolve()
        if not (repo / "core/raft_stereo.py").is_file():
            raise ValueError("Expected the official RAFT-Stereo repository")
        sys.path.insert(0, str(repo))
        from core.raft_stereo import RAFTStereo
        from core.utils.utils import InputPadder
        from .model import choose_device
        self.device = choose_device(device)
        self.iters = int(iters)
        if self.iters < 1:
            raise ValueError("iters must be positive")
        args = SimpleNamespace(hidden_dims=[128]*3, context_norm="batch",
            n_downsample=2, n_gru_layers=3, shared_backbone=False,
            corr_implementation="reg", corr_levels=4, corr_radius=4,
            slow_fast_gru=False, mixed_precision=False)
        self.model = RAFTStereo(args)
        weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict({k.removeprefix("module."): v for k, v in weights.items()}, strict=True)
        self.model.to(self.device).eval()
        self.padder = InputPadder

    def _sync(self):
        if self.device.type == "mps":
            torch.mps.synchronize()
        elif self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def predict(self, left, right):
        """Returns positive x_left-x_right disparity and timed forward seconds.

        Caller handles any resizing and associated calibration/disparity scaling.
        Loading, image preprocessing and host-output transfer excluded from timing.
        """
        if left.shape != right.shape or left.ndim != 2:
            raise ValueError("Expected same-sized rectified grayscale images")
        if left.dtype != np.uint8 or right.dtype != np.uint8:
            raise ValueError("Expected uint8 input, not pre-normalised tensors")
        def tensor(image):
            return torch.from_numpy(np.repeat(image[None], 3, axis=0).copy()).float()[None].to(self.device)
        left_t, right_t = tensor(left), tensor(right)
        pad = self.padder(left_t.shape, divis_by=32)
        left_t, right_t = pad.pad(left_t, right_t)
        self._sync()
        start = time.perf_counter()
        with torch.inference_mode():
            _, flow = self.model(left_t, right_t, iters=self.iters, test_mode=True)
        self._sync()
        elapsed = time.perf_counter()-start
        disparity = -pad.unpad(flow)[0, 0].float().cpu().numpy()
        if not np.isfinite(disparity).all():
            raise ValueError("Non-finite RAFT-Stereo output")
        return disparity.astype(np.float32), elapsed


def predict(left, right, checkpoint, repo, device="auto", iters=32):
    return RaftPredictor(checkpoint, repo, device, iters).predict(left, right)
