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
    from .model import PairwiseTactileSE3Regressor
except ImportError:
    from data_relative_se3 import TARGET_NAMES, load_pair_datasets
    from model import PairwiseTactileSE3Regressor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("tactile_policy/outputs/relative_se3"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--pairs-per-episode", type=int, default=10)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--target-limit",
        default="0.00025,0.0005,0.0005,0.003490659,0.010471976,0.020943951",
        help="Normalization limits dx,dy,dz(m), drx,dry,drz(rad), matching OU limits.",
    )
    args = parser.parse_args()
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
        args.target_limit, args.seed,
    )
    print(json.dumps(info, indent=2))
    train_loader = DataLoader(
        train_set, args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set, args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = PairwiseTactileSE3Regressor().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    limit = torch.as_tensor(normalization.target_limit, device=device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = args.output_dir / "history.jsonl"
    history.write_text("", encoding="utf-8")
    best, stale = float("inf"), 0
    for epoch in range(1, args.epochs + 1):
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
        metrics = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
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
                "model_type": "pairwise_tactile_se3",
                "normalization": normalization.as_dict(),
                "target_names": TARGET_NAMES,
                "transform_convention": "inv(T_world_tcp_1) @ T_world_tcp_2; local xyz + rotvec",
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
