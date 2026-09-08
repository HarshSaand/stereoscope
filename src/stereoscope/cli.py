"""Explicit prepare/train/evaluate commands; evaluation never tunes a checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from .data import eth_records, sceneflow_records, grouped_split, save_manifest, load_pair
from .geometry import stereo_sgbm, confidence_features
from .metrics import bad_disparity, evaluate
from .model import ConfidenceCNN, LinearConfidence, choose_device


_RAFT_PREDICTORS = {}


def cache_record(record, cache_root, max_width=640, backend='sgbm'):
    key = hashlib.sha256(json.dumps([record, max_width, backend+"-v1"], sort_keys=True).encode()).hexdigest()[:20]
    path = Path(cache_root)/f"{key}.npz"
    if path.exists():
        return np.load(path)
    left, right, gt, scale = load_pair(record, max_width)
    start = time.perf_counter()
    if backend == 'sgbm':
        disparity, rd = stereo_sgbm(left, right)
    else:
        from .raft import RaftPredictor
        if backend not in _RAFT_PREDICTORS:
            _RAFT_PREDICTORS[backend]=RaftPredictor('data/models/raftstereo-sceneflow.pth',
                'data/vendor/RAFT-Stereo',device='auto',iters=32)
        predictor=_RAFT_PREDICTORS[backend]
        disparity,_=predictor.predict(left,right)
        rd,_=predictor.predict(np.fliplr(right).copy(),np.fliplr(left).copy())
        rd=np.fliplr(rd).copy()
    features, consistency = confidence_features(left, right, disparity, rd)
    elapsed = time.perf_counter()-start
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, features=features, consistency=consistency,
        disparity=disparity, gt=gt, seconds=elapsed, scale=scale)
    return np.load(path)


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = ConfidenceCNN() if checkpoint["architecture"] == "cnn" else LinearConfidence()
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint


def train(args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = choose_device(args.device)
    manifest_bytes=Path(args.manifest).read_bytes()
    records = json.loads(manifest_bytes)
    training = [r for r in records if r.get("split") == "train"]
    calibration = [r for r in records if r.get("split") == "calibration"]
    if not training or not calibration:
        raise ValueError("Need separate train and calibration scene groups")
    if {r['group'] for r in training} & {r['group'] for r in calibration}:
        raise ValueError("Scene-group leakage")
    if any(r['dataset'] == "ETH3D" for r in training+calibration):
        raise ValueError("ETH3D is reserved for external evaluation")
    samples = [cache_record(r, args.cache, args.max_width,args.backend) for r in training]
    prior_rates=[]
    for sample in samples:
        mask=np.isfinite(sample['gt']) & (sample['gt']>0) & (sample['disparity']>0)
        prior_rates.append(float(bad_disparity(sample['disparity'],sample['gt'])[mask].mean()))
    model = (ConfidenceCNN() if args.architecture == "cnn" else LinearConfidence()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    history = []
    start = time.perf_counter()
    for epoch in range(args.epochs):
        losses = []
        model.train()
        for i in rng.permutation(len(samples)):
            sample = samples[i]
            f, d, gt = sample['features'], sample['disparity'], sample['gt']
            h, w = d.shape
            size = min(args.crop, h, w)
            y, x = int(rng.integers(h-size+1)), int(rng.integers(w-size+1))
            sl = (slice(y,y+size), slice(x,x+size))
            mask = np.isfinite(gt[sl]) & (gt[sl]>0) & (d[sl]>0)
            if mask.sum() < 32:
                continue
            target = bad_disparity(d[sl], gt[sl]).astype(np.float32)
            batch = torch.from_numpy(f[:, sl[0], sl[1]].copy())[None].to(device)
            logits = model(batch)[0]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits[torch.from_numpy(mask).to(device)],
                torch.from_numpy(target).to(device)[torch.from_numpy(mask).to(device)])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(dict(epoch=epoch+1, loss=float(np.mean(losses)) if losses else None))
    # Temperature calibration on independent scenes only; architecture not selected on test.
    model.eval()
    logits_all, labels_all = [], []
    with torch.no_grad():
        for r in calibration:
            s = cache_record(r, args.cache, args.max_width,args.backend)
            mask = np.isfinite(s['gt']) & (s['gt']>0) & (s['disparity']>0)
            logits = model(torch.from_numpy(s['features'])[None].to(device))[0].cpu().numpy()
            logits_all.append(logits[mask][::20])
            labels_all.append(bad_disparity(s['disparity'], s['gt'])[mask][::20])
    logits = torch.tensor(np.concatenate(logits_all))
    labels = torch.tensor(np.concatenate(labels_all), dtype=torch.float32)
    candidates = np.geomspace(.25, 4, 25)
    temperature = float(min(candidates, key=lambda t: float(
        torch.nn.functional.binary_cross_entropy_with_logits(logits/t, labels))))
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(architecture=args.architecture, temperature=temperature,
        disparity_backend=args.backend,
        training_scene_macro_error_prior=float(np.mean(prior_rates)),
        training_pairs=len(training), training_groups=len({r['group'] for r in training}),
        calibration_pairs=len(calibration), seed=args.seed, epochs=args.epochs,
        max_width=args.max_width, elapsed_seconds=time.perf_counter()-start,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        training_ids=[r['id'] for r in training], calibration_ids=[r['id'] for r in calibration],
        fit_scene_groups=sorted({r['dataset']+':'+r['group'] for r in training+calibration}),
        device=str(device), history=history)
    torch.save(dict(metadata, state_dict={k:v.cpu() for k,v in model.state_dict().items()}), output)
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps({k:v for k,v in metadata.items() if k not in ('history','training_ids','calibration_ids')}, indent=2))


def evaluate_manifest(args):
    records = json.loads(Path(args.manifest).read_text())
    if args.split != "all":
        records = [r for r in records if r.get('split') == args.split]
    if not records:
        raise ValueError("No evaluation records")
    device = choose_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device) if args.checkpoint else (None, {})
    used = set(checkpoint.get('training_ids', [])+checkpoint.get('calibration_ids', []))
    if used & {r['id'] for r in records}:
        raise ValueError("Refusing evaluation on training/calibration IDs")
    if set(checkpoint.get('fit_scene_groups', [])) & {r['dataset']+':'+r['group'] for r in records}:
        raise ValueError("Refusing evaluation on training/calibration scene groups")
    if checkpoint and checkpoint['max_width'] != args.max_width:
        raise ValueError("Evaluation width must match checkpoint calibration width")
    if checkpoint and checkpoint.get('disparity_backend','sgbm')!=args.backend:
        raise ValueError("Confidence checkpoint must match the disparity backend")
    results = []
    for record in records:
        s = cache_record(record, args.cache, args.max_width,args.backend)
        risk = np.clip(s['consistency']/16, 0, 1)
        measures = {args.backend+"_consistency": evaluate(s['disparity'], s['gt'], risk)}
        if model:
            with torch.no_grad():
                risk = torch.sigmoid(model(torch.from_numpy(s['features'])[None].to(device))/
                    checkpoint['temperature'])[0].cpu().numpy()
            measures[checkpoint['architecture']] = evaluate(s['disparity'], s['gt'], risk)
            measures['zero_error_risk']=evaluate(s['disparity'],s['gt'],np.zeros_like(risk))
            if 'training_scene_macro_error_prior' in checkpoint:
                measures['constant_training_prior']=evaluate(s['disparity'],s['gt'],
                    np.full_like(risk,checkpoint['training_scene_macro_error_prior']))
        results.append(dict(id=record['id'], group=record['group'], dataset=record['dataset'],
            backend_seconds=float(s['seconds']), scale=float(s['scale']), methods=measures))
    summary = {method: {metric: float(np.mean([r['methods'][method][metric] for r in results]))
               for metric in ('epe','d1_valid','d1_all_labelled','bad1_valid','bad2_valid','bad4_valid','prediction_coverage','brier')}
               for method in results[0]['methods']}
    report = dict(protocol='Local held-out evaluation, not official hidden-test leaderboard',
        pairs=len(results), groups=len({r['group'] for r in results}), max_width=args.max_width,
        disparity_backend=args.backend,
        summary_scene_macro=summary, records=results, checkpoint=args.checkpoint,
        note='EPE/D1 measured at evaluation resolution. Filtering never changes disparity. Brier for consistency is heuristic, not calibrated.')
    output=Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(pairs=len(results), summary=summary), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--dataset', choices=['eth3d','sceneflow'], required=True)
    p.add_argument('--root', required=True); p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=42)
    for name in ('train', 'evaluate'):
        p = sub.add_parser(name)
        p.add_argument('--manifest', required=True); p.add_argument('--output', required=True)
        p.add_argument('--cache', default='outputs/cache')
        p.add_argument('--device', default='auto'); p.add_argument('--max-width', type=int, default=640)
        p.add_argument('--backend',choices=['sgbm','raft'],default='sgbm')
        if name == 'train':
            p.add_argument('--architecture', choices=['cnn','linear'], default='cnn')
            p.add_argument('--epochs', type=int, default=30); p.add_argument('--crop',type=int,default=128)
            p.add_argument('--seed', type=int, default=42)
        else:
            p.add_argument('--checkpoint'); p.add_argument('--split', default='test')
    args = parser.parse_args()
    if args.command == 'prepare':
        records = eth_records(args.root) if args.dataset == 'eth3d' else sceneflow_records(args.root)
        if not records: raise ValueError('No complete pairs found; check dataset layout')
        records = [dict(r, split='test') for r in records] if args.dataset == 'eth3d' else grouped_split(records,args.seed)
        save_manifest(args.output, records); print(f'{len(records)} pairs, {len({r["group"] for r in records})} groups')
    elif args.command == 'train': train(args)
    else: evaluate_manifest(args)


if __name__ == '__main__': main()
