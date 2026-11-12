#!/usr/bin/env python3
"""Train reference/current dual-finger tactile regression for local SE(3)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    from .data_relative_se3 import TARGET_NAMES, load_pair_datasets
    from .model import PAIRWISE_BACKBONES, create_pairwise_model
except ImportError:
    from data_relative_se3 import TARGET_NAMES, load_pair_datasets
    from model import PAIRWISE_BACKBONES, create_pairwise_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("/media/zhou/Elements SE/YBZHOU/nut_relative_ou_217"))
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Defaults to the original directory for spatial, or a backbone-specific directory.")
    parser.add_argument(
        "--input-modality", choices=("distributed_force", "depth"),
        default="distributed_force",
        help=("Tactile input. depth uses embedded left/right depth PNGs, the "
              "training-set maximum divisor, and native image resolution."),
    )
    parser.add_argument("--backbone", choices=PAIRWISE_BACKBONES, default="spatial",
                        help="spatial preserves the original four-hand model; others encode each bilateral frame.")
    parser.add_argument("--feature-dim", type=int, default=128,
                        help="Per-frame feature size for create_model backbones; unused by spatial.")
    parser.add_argument("--grid-size", type=int, default=20,
                        help="Square input grid for the FCN backbone.")
    parser.add_argument("--image-size", type=int, default=128,
                        help="Square input size for CNN/ResNet/ViT backbones; tokenizer uses its native grid.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument(
        "--pairs-per-episode", type=int, default=10,
        help=("Fresh training pairs sampled per episode at every epoch; the same "
              "count of validation pairs is sampled once and then kept fixed. "
              "Use 20-50 for denser coverage of roughly 200-frame episodes."),
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=300)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--target-limit",
        default="0.0035,0.003,0.0015,0.030,0.035,0.040",
        help="Normalization limits dx,dy,dz(m), drx,dry,drz(rad), matching OU limits.",
    )
    args = parser.parse_args()
    if args.input_modality == "depth":
        if args.backbone not in ("spatial", "native_depth_cnn"):
            parser.error(
                "--input-modality depth uses the train_pose_real_image.py "
                "NativeDepthCNN; omit --backbone or set --backbone native_depth_cnn"
            )
        args.backbone = "native_depth_cnn"
    elif args.backbone == "native_depth_cnn":
        parser.error("--backbone native_depth_cnn requires --input-modality depth")
    for name in ("feature_dim", "grid_size", "image_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.output_dir is None:
        suffix = "" if args.backbone == "spatial" else f"_{args.backbone}"
        args.output_dir = Path(f"tactile_policy/outputs/relative_se3_600_200{suffix}")
    try:
        args.target_limit = np.asarray(
            [float(value) for value in args.target_limit.split(",")], dtype=np.float32
        )
    except ValueError as exc:
        parser.error(f"Invalid --target-limit: {exc}")
    if args.target_limit.shape != (6,) or np.any(args.target_limit <= 0):
        parser.error("--target-limit needs six positive numbers")
    if not 0 < args.val_ratio < 1:
        parser.error("--val-ratio must be in (0,1)")
    if args.pairs_per_episode < 1:
        parser.error("--pairs-per-episode must be positive")
    return args


def evaluate(model, loader, device, limit):
    model.eval()
    loss_sum, count, errors = 0.0, 0, []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            prediction = model(inputs)
            loss = nn.functional.mse_loss(prediction, targets)
            loss_sum += loss.item() * len(inputs)
            count += len(inputs)
            errors.append(((prediction - targets) * limit).abs().cpu().numpy())
    return loss_sum / count, np.concatenate(errors).mean(axis=0)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    train_set, val_set, normalization, info = load_pair_datasets(
        args.dataset_root, args.val_ratio, args.pairs_per_episode,
        args.target_limit, args.seed, args.input_modality,
    )
    print(json.dumps({key: value for key, value in info.items()
                      if key not in ("train_episode_ids", "val_episode_ids")}, indent=2))
    train_loader = DataLoader(
        train_set, args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set, args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    input_shape = info["input_shape_per_frame"]
    input_hw = tuple(input_shape[-2:])
    model = create_pairwise_model(
        args.backbone, args.feature_dim, args.grid_size, args.image_size,
        input_hw=input_hw,
    ).to(device)
    print(f"Model: {model.model_type} config={model.model_config}", flush=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
        min_lr=1e-6,
    )
    limit = torch.as_tensor(normalization.target_limit, device=device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = args.output_dir / "history.jsonl"
    history.write_text("", encoding="utf-8")
    best, stale = float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        # Deterministic across reruns, different across epochs. Validation set
        # is a separate immutable Dataset and is deliberately not resampled.
        train_set.resample_pairs((args.seed, epoch))
        model.train()
        loss_sum = count = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(inputs), targets)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(inputs)
            count += len(inputs)
        train_loss = loss_sum / count
        val_loss, mae = evaluate(model, val_loader, device, limit)
        scheduler.step(val_loss)
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_pairs": len(train_set),
            "val_pairs": len(val_set),
        }
        metrics.update({f"mae_{name}": float(value) for name, value in zip(TARGET_NAMES, mae)})
        with history.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics) + "\n")
        print(
            f"epoch {epoch:03d} train={train_loss:.6f} val={val_loss:.6f} | "
            f"position MAE mm={np.round(mae[:3] * 1000, 4)} | "
            f"rotation MAE deg={np.round(np.rad2deg(mae[3:]), 4)}"
        )
        if val_loss < best:
            best, stale = val_loss, 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_type": model.model_type,
                "model_config": model.model_config,
                "epoch": epoch,
                "normalization": normalization.as_dict(),
                "target_names": TARGET_NAMES,
                "transform_convention": "inv(T_world_tcp_1) @ T_world_tcp_2; local xyz + rotvec",
                "input_modality": args.input_modality,
                "input_keys": (
                    ["observation.tactile_left.depth", "observation.tactile_right.depth"]
                    if args.input_modality == "depth"
                    else ["observation.tactile_left.distributed_force",
                          "observation.tactile_right.distributed_force"]
                ),
                "input_shape_per_frame": input_shape,
                "input_preprocessing": (
                    {
                        "type": "depth_png_uint8_train_max",
                        "channel_order_per_time": ["left", "right"],
                        "time_order": ["reference", "current"],
                        "png_channel": 0,
                        "divisor": info["input_scale"],
                        "signed_log1p": False,
                        "per_image_normalization": False,
                        "resize": None,
                    }
                    if args.input_modality == "depth"
                    else {
                        "type": "distributed_force_signed_log1p_standardized",
                        "channel_order_per_time": ["left_xyz", "right_xyz"],
                        "time_order": ["reference", "current"],
                    }
                ),
                "dataset_info": info,
                "args": {key: str(value) if isinstance(value, Path) else value
                         for key, value in vars(args).items() if key != "target_limit"},
            }, args.output_dir / "best_model.pt")
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping after {stale} stale epochs")
                break


if __name__ == "__main__":
    main()
