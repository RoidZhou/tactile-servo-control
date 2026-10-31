"""Episode-wise paired XENSE data for local SE(3) motion regression."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as pads
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset

LEFT_KEY = "observation.tactile_left.distributed_force"
RIGHT_KEY = "observation.tactile_right.distributed_force"
TCP_KEY = "observation.tcp_pose"
TARGET_NAMES = ("dx", "dy", "dz", "drx", "dry", "drz")


@dataclass
class PairNormalization:
    input_mean: np.ndarray
    input_std: np.ndarray
    target_limit: np.ndarray

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "input_mean": self.input_mean.tolist(),
            "input_std": self.input_std.tolist(),
            "target_limit": self.target_limit.tolist(),
        }


def signed_log1p(value: np.ndarray) -> np.ndarray:
    return np.sign(value) * np.log1p(np.abs(value))


def pose_to_matrix(pose: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_rotvec(pose[3:]).as_matrix()
    transform[:3, 3] = pose[:3]
    return transform


def local_se3_delta(first_pose: np.ndarray, second_pose: np.ndarray) -> np.ndarray:
    """Return log-like coordinates of inv(T_W1) @ T_W2.

    Translation is expressed in frame 1. Rotation is the shortest local
    rotation vector in radians.
    """
    relative = np.linalg.inv(pose_to_matrix(first_pose)) @ pose_to_matrix(second_pose)
    return np.r_[
        relative[:3, 3], Rotation.from_matrix(relative[:3, :3]).as_rotvec()
    ].astype(np.float32)


def _force_to_chw(values: list) -> np.ndarray:
    force = np.asarray(values, dtype=np.float32)
    if force.ndim != 4 or force.shape[-1] != 3:
        raise ValueError(f"Expected native XENSE force [N,H,W,3], got {force.shape}")
    return force.transpose(0, 3, 1, 2)


class PairedTactileSE3Dataset(Dataset):
    def __init__(
        self,
        frames: np.ndarray,
        poses: np.ndarray,
        pairs: np.ndarray,
        normalization: PairNormalization,
    ) -> None:
        self.frames = frames
        self.poses = poses
        self.pairs = np.asarray(pairs, dtype=np.int64)
        self.mean = normalization.input_mean[None, :, None, None]
        self.std = normalization.input_std[None, :, None, None]
        self.limit = normalization.target_limit

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        first, second = self.pairs[item]
        tactile = self.frames[[first, second]]
        tactile = (tactile - self.mean) / self.std
        target = local_se3_delta(self.poses[first], self.poses[second]) / self.limit
        return torch.from_numpy(tactile.astype(np.float32)), torch.from_numpy(target)


def load_pair_datasets(
    root: str | Path,
    val_ratio: float,
    pairs_per_episode: int,
    target_limit: np.ndarray,
    seed: int,
) -> tuple[PairedTactileSE3Dataset, PairedTactileSE3Dataset, PairNormalization, dict]:
    root = Path(root).expanduser().resolve()
    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files below {root / 'data'}")
    source = pads.dataset([str(path) for path in files], format="parquet")
    required = {LEFT_KEY, RIGHT_KEY, TCP_KEY, "episode_index", "frame_index"}
    missing = required - set(source.schema.names)
    if missing:
        raise ValueError(
            f"Dataset lacks paired-motion fields {sorted(missing)}. "
            "Collect into a new --root with the OU collector."
        )

    meta = source.to_table(columns=["episode_index", "frame_index"])
    episodes = np.asarray(meta["episode_index"].to_pylist(), dtype=np.int64)
    frame_indices = np.asarray(meta["frame_index"].to_pylist(), dtype=np.int64)
    unique_episodes = np.unique(episodes)
    if len(unique_episodes) < 2:
        raise ValueError("At least two complete episodes are required for train/validation")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_episodes)
    n_val = min(len(shuffled) - 1, max(1, round(len(shuffled) * val_ratio)))
    val_eps = set(shuffled[:n_val].tolist())

    global_pairs: dict[str, list[tuple[int, int]]] = {"train": [], "val": []}
    for episode in unique_episodes:
        rows = np.flatnonzero(episodes == episode)
        rows = rows[np.argsort(frame_indices[rows])]
        if len(rows) < 2:
            continue
        split = "val" if int(episode) in val_eps else "train"
        for _ in range(pairs_per_episode):
            selected = rng.choice(rows, size=2, replace=False)
            # Preserve temporal direction, as in a reference/current pair.
            first, second = sorted(selected.tolist())
            global_pairs[split].append((first, second))
    if not global_pairs["train"] or not global_pairs["val"]:
        raise ValueError("Could not create nonempty episode-wise train and validation pairs")

    selected_rows = np.unique(np.asarray(
        global_pairs["train"] + global_pairs["val"], dtype=np.int64
    ).reshape(-1))
    table = source.take(
        pa.array(selected_rows), columns=[LEFT_KEY, RIGHT_KEY, TCP_KEY]
    )
    left = _force_to_chw(table[LEFT_KEY].to_pylist())
    right = _force_to_chw(table[RIGHT_KEY].to_pylist())
    frames = signed_log1p(np.concatenate((left, right), axis=1)).astype(np.float32)
    poses = np.asarray(table[TCP_KEY].to_pylist(), dtype=np.float32)
    row_to_local = {int(row): index for index, row in enumerate(selected_rows)}

    def remap(pairs: list[tuple[int, int]]) -> np.ndarray:
        return np.asarray(
            [(row_to_local[a], row_to_local[b]) for a, b in pairs], dtype=np.int64
        )

    train_pairs, val_pairs = remap(global_pairs["train"]), remap(global_pairs["val"])
    train_rows = np.unique(train_pairs.reshape(-1))
    input_mean = frames[train_rows].mean(axis=(0, 2, 3), dtype=np.float64).astype(np.float32)
    input_std = frames[train_rows].std(axis=(0, 2, 3), dtype=np.float64).astype(np.float32)
    normalization = PairNormalization(
        input_mean, np.maximum(input_std, 1e-6), np.asarray(target_limit, dtype=np.float32)
    )
    info = {
        "episodes": len(unique_episodes), "train_episodes": len(unique_episodes) - n_val,
        "val_episodes": n_val, "loaded_unique_frames": len(selected_rows),
        "train_pairs": len(train_pairs), "val_pairs": len(val_pairs),
    }
    return (
        PairedTactileSE3Dataset(frames, poses, train_pairs, normalization),
        PairedTactileSE3Dataset(frames, poses, val_pairs, normalization),
        normalization, info,
    )
