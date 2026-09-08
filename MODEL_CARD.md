# StereoScope confidence model card

## Identity and intended use

`mirror-cnn.pt` is a 4,369-parameter spatial error-confidence model, trained locally on Apple MPS. It predicts probability of custom bad disparity from six SGBM-derived channels. It is not a disparity estimator, autonomous-driving model, general confidence service or RAFT calibration head.

Checkpoint SHA-256: `32487c03fff4c246ac62447da2348bc0476a9b8639ffb1c17c53cdbf97202283`.

## Training and evaluation

- 300 mirror synthetic stereo pairs, **one unknown group**; 10 epochs, random 128x128 crops, seed 42, Adam learning rate 0.001.
- Calibration: 3 official FlyingThings3D sampler frames; selected temperature 0.6299605 on a fixed candidate grid.
- Width: maximum 640 pixels; confidence features and calibration depend on resolution.
- Training epoch-mean BCE: 0.45208 at epoch 1 → 0.38915 at epoch 10. These are training losses, not accuracy metrics.
- External evaluation: all 27 labelled ETH3D pairs (7 grouped physical environments), never used for fitting or calibration. Local resized-space metrics, not official hidden-test scores.
- CNN external Brier 0.05186; linear comparator 0.03828; zero-risk baseline 0.03518. **Calibration transfer failed to beat simple baselines.** The CNN's error ranking at retained coverage is separately reported, not conflated with calibrated probabilities.

## Provenance and limitations

Mirror: https://huggingface.co/datasets/olivermao/sceneflow. Original claimed dataset: https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html. The mirror has no dataset card, declared license or original scene identifiers. Its asserted source is not independently authenticated. All its frames are grouped together; semantic overlap with the independently named calibration sampler cannot be excluded solely by the absence of exact hashes. This is a bounded transfer pilot, not the approved full multi-scene training programme.

The official source is research-only and prohibits commercial dataset use. No permissive redistribution right for this mirror-derived checkpoint is asserted. The local checkpoint is intentionally excluded from public version control; reproducible source and measured metadata are available. Original source code's MIT license does not relicense datasets, pretrained RAFT weights or derived checkpoints.

## Frozen RAFT comparator

SceneFlow-only pretrained checkpoint, not trained by this project. Upstream source revision: `6e93ed2169bd858dbb43033988563f3b0bb49506`. Upstream checkpoint SHA-256: `ba75bdcb379b4c3942d1e714565d0e66e6b2450a3da8129f54073302d67e0a6d`. The adapter runs the regular PyTorch correlation implementation on MPS without CUDA extensions and converts negative horizontal flow to positive left-reference disparity. It is evaluated separately from the SGBM confidence checkpoint.

## Remaining work

Authenticated multi-scene training data; larger independent calibration; corruption and feature ablations; sequence-level uncertainty intervals; confidence models specific to frozen RAFT; measured deployment constraints. None is claimed complete.
