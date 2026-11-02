#!/usr/bin/env python3
"""Compare paired tactile checkpoints on a persisted, shared validation manifest."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as pads
import torch
from scipy.spatial.transform import Rotation

try:
    from .data_relative_se3 import LEFT_KEY, RIGHT_KEY, TCP_KEY, TARGET_NAMES, _force_to_chw, local_se3_delta, signed_log1p
    from .model import PAIRWISE_MODEL_TYPES, load_pairwise_model
except ImportError:
    from data_relative_se3 import LEFT_KEY, RIGHT_KEY, TCP_KEY, TARGET_NAMES, _force_to_chw, local_se3_delta, signed_log1p
    from model import PAIRWISE_MODEL_TYPES, load_pairwise_model

SCALE = np.array([1000.] * 3 + [180 / np.pi] * 3)


def metrics(prediction, target):
    error = (prediction - target) * SCALE
    angle = (Rotation.from_rotvec(target[:, 3:]).inv()
             * Rotation.from_rotvec(prediction[:, 3:])).magnitude() * 180 / np.pi
    distance = np.linalg.norm(error[:, :3], axis=1)
    return {
        'samples': len(target),
        'mae_mm_deg': np.abs(error).mean(0).tolist(),
        'p95_abs_mm_deg': np.quantile(np.abs(error), .95, axis=0).tolist(),
        'rmse_mm_deg': np.sqrt((error ** 2).mean(0)).tolist(),
        'position_norm_mean_mm': float(distance.mean()),
        'position_norm_p95_mm': float(np.quantile(distance, .95)),
        'rotation_geodesic_mean_deg': float(angle.mean()),
        'rotation_geodesic_p95_deg': float(np.quantile(angle, .95)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, default=Path("/media/zhou/Elements SE/YBZHOU/nut_relative_ou_no_fts"))
    parser.add_argument('--checkpoint', type=Path, nargs='+', default=[
        Path('/home/zhou/vla/ur5-tactile-servo-control/tactile_policy/outputs/relative_se3_100/best_model.pt'),
        Path('/home/zhou/vla/ur5-tactile-servo-control/tactile_policy/outputs/relative_se3_200/best_model.pt'),
    ])
    parser.add_argument('--names', nargs='+', help='Unique labels, e.g. old new')
    parser.add_argument('--output-dir', type=Path, default=Path("tactile_policy/outputs/compare_relative_se3"))
    parser.add_argument('--manifest', type=Path, help='Existing manifest to reuse; otherwise create output-dir/validation_pairs.json')
    parser.add_argument('--pairs-per-episode', type=int, default=100)
    parser.add_argument('--pair-seed', type=int, default=2026)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    if args.pairs_per_episode < 1 or args.batch_size < 1:
        parser.error('pair count and batch size must be positive')
    names = args.names or [f'model_{i}' for i in range(len(args.checkpoint))]
    if len(names) != len(args.checkpoint) or len(set(names)) != len(names):
        parser.error('--names must contain one unique name per checkpoint')
    checkpoints = [torch.load(p, map_location='cpu', weights_only=False) for p in args.checkpoint]
    for cp in checkpoints:
        if cp.get('model_type') not in PAIRWISE_MODEL_TYPES:
            raise ValueError('Only paired SE3 checkpoints are supported')
        if cp.get('transform_convention') != 'inv(T_world_tcp_1) @ T_world_tcp_2; local xyz + rotvec':
            raise ValueError('Unsupported or missing checkpoint transform convention')

    root = args.dataset_root.expanduser().resolve()
    files = sorted((root / 'data').rglob('*.parquet'))
    if not files:
        raise ValueError('No parquet data found')
    source = pads.dataset([str(p) for p in files], format='parquet')
    meta = source.to_table(columns=['episode_index', 'frame_index', TCP_KEY])
    eps = np.asarray(meta['episode_index'].to_pylist(), dtype=np.int64)
    frames = np.asarray(meta['frame_index'].to_pylist(), dtype=np.int64)
    poses = np.asarray(meta[TCP_KEY].to_pylist(), dtype=np.float64)
    keys = list(zip(eps.tolist(), frames.tolist()))
    if len(set(keys)) != len(keys) or not np.isfinite(poses).all():
        raise ValueError('Duplicate episode/frame keys or nonfinite poses')
    lookup = {key: i for i, key in enumerate(keys)}
    order = np.lexsort((frames, eps))
    digest = hashlib.sha256()
    digest.update(np.stack((eps[order], frames[order]), 1).astype('<i8').tobytes())
    digest.update(poses[order].astype('<f8').tobytes())
    fingerprint = digest.hexdigest()
    unique = np.unique(eps)
    eligible = set(unique.tolist())
    for cp in checkpoints:
        info, config = cp['dataset_info'], cp['args']
        if info['episodes'] != len(unique):
            raise ValueError('Dataset episode count differs from training; cannot safely reconstruct held-out split')
        shuffled = np.random.default_rng(int(config['seed'])).permutation(unique)
        n = min(len(unique)-1, max(1, round(len(unique)*float(config['val_ratio']))))
        eligible &= set(shuffled[:n].tolist())
    if not eligible:
        raise ValueError('Checkpoints have no common held-out episodes')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.manifest or args.output_dir / 'validation_pairs.json'
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest['pose_index_sha256'] != fingerprint:
            raise ValueError('Manifest dataset pose/index fingerprint mismatch')
    else:
        if args.manifest:
            raise FileNotFoundError(path)
        rng = np.random.default_rng(args.pair_seed)
        pairs = []
        for ep in sorted(eligible):
            ff = sorted(frames[eps == ep].tolist())
            total = len(ff)*(len(ff)-1)//2
            if total < args.pairs_per_episode:
                raise ValueError(f'Episode {ep} has too few unique pairs')
            chosen = set()
            while len(chosen) < args.pairs_per_episode:
                a, b = sorted(rng.choice(ff, 2, replace=False).tolist())
                chosen.add((ep, a, b))
            pairs.extend(sorted(chosen))
        manifest = dict(version=1, dataset_root=str(root), pose_index_sha256=fingerprint,
                        pair_seed=args.pair_seed, pairs=pairs)
        path.write_text(json.dumps(manifest, indent=2))
    pairs = np.asarray(manifest['pairs'], dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 3 or not len(pairs):
        raise ValueError('Invalid manifest pairs')
    if not set(pairs[:, 0]).issubset(eligible) or np.any(pairs[:, 1] >= pairs[:, 2]):
        raise ValueError('Manifest includes training episodes or reversed/identical frames')
    indices = np.array([[lookup[(int(e), int(a))], lookup[(int(e), int(b))]] for e, a, b in pairs])
    selected, local = np.unique(indices, return_inverse=True)
    local = local.reshape(-1, 2)
    table = source.take(pa.array(selected), columns=[LEFT_KEY, RIGHT_KEY])
    tactile = signed_log1p(np.concatenate([_force_to_chw(table[k].to_pylist()) for k in (LEFT_KEY, RIGHT_KEY)], axis=1))
    if not np.isfinite(tactile).all():
        raise ValueError('Nonfinite tactile input')
    target = np.stack([local_se3_delta(poses[a], poses[b]) for a, b in indices])
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    report = dict(manifest=str(path.resolve()), axes=list(TARGET_NAMES),
                  note='Validation splits reconstructed from saved seed/ratio; use the original training dataset. Rotation axes are rotvec components.',
                  zero_motion_baseline=metrics(np.zeros_like(target), target), models={})
    with (args.output_dir / 'predictions.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model','episode','frame1','frame2'] + [f'{kind}_{axis}_{"mm" if i<3 else "deg"}' for kind in ('target','prediction','error') for i,axis in enumerate(TARGET_NAMES)])
        for name, checkpoint_path, cp in zip(names, args.checkpoint, checkpoints):
            norm = cp['normalization']
            mean, std, limit = [np.asarray(norm[k], dtype=np.float32) for k in ('input_mean','input_std','target_limit')]
            if any(x.shape != (6,) or not np.isfinite(x).all() for x in (mean,std,limit)) or np.any(std<=0) or np.any(limit<=0):
                raise ValueError('Invalid checkpoint normalization')
            model = load_pairwise_model(cp, device)
            outputs = []
            with torch.inference_mode():
                for start in range(0, len(local), args.batch_size):
                    x = tactile[local[start:start+args.batch_size]]
                    x = (x-mean[None,None,:,None,None])/std[None,None,:,None,None]
                    outputs.append(model(torch.from_numpy(x.astype(np.float32)).to(device)).cpu().numpy()*limit)
            pred = np.concatenate(outputs)
            if not np.isfinite(pred).all():
                raise ValueError(f'Nonfinite predictions: {name}')
            result = metrics(pred,target)
            result['checkpoint'] = str(checkpoint_path.resolve())
            result['model_type'] = model.model_type
            result['model_config'] = model.model_config
            result['target_out_of_range_fraction'] = (np.abs(target)>limit).mean(0).tolist()
            result['by_episode'] = {str(e): metrics(pred[pairs[:,0]==e], target[pairs[:,0]==e]) for e in np.unique(pairs[:,0])}
            report['models'][name] = result
            print(name, 'MAE [mm,deg]=', np.round(result['mae_mm_deg'],4), 'P95=',np.round(result['p95_abs_mm_deg'],4), flush=True)
            for pair, gt, pr in zip(pairs,target,pred):
                writer.writerow([name,*pair.tolist(),*(gt*SCALE),*(pr*SCALE),*((pr-gt)*SCALE)])
            del model
    (args.output_dir / 'summary.json').write_text(json.dumps(report,indent=2))
    print(f'Evaluated {len(pairs)} shared pairs; results: {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
