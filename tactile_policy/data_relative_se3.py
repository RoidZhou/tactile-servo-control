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

try:
    from .data import DEPTH_KEYS, decode_depth_png
except ImportError:
    from data import DEPTH_KEYS, decode_depth_png

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


def sample_episode_pairs(
    episode_rows: list[np.ndarray],
    pairs_per_episode: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample unique, temporally ordered pairs independently inside each episode."""
    if pairs_per_episode < 1:
        raise ValueError("pairs_per_episode must be positive")
    sampled: list[tuple[int, int]] = []
    for rows in episode_rows:
        rows = np.asarray(rows, dtype=np.int64)
        possible = len(rows) * (len(rows) - 1) // 2
        if pairs_per_episode > possible:
            raise ValueError(
                f"Requested {pairs_per_episode} pairs from an episode with only "
                f"{len(rows)} frames ({possible} unique ordered pairs)"
            )
        chosen: set[tuple[int, int]] = set()
        while len(chosen) < pairs_per_episode:
            first_index, second_index = rng.choice(
                len(rows), size=2, replace=False
            ).tolist()
            # ``rows`` is already sorted by frame_index. Compare positions,
            # not global/local row IDs, so temporal direction remains correct
            # even if Parquet row order is unusual.
            if first_index > second_index:
                first_index, second_index = second_index, first_index
            chosen.add((int(rows[first_index]), int(rows[second_index])))
        sampled.extend(sorted(chosen))
    if not sampled:
        raise ValueError("No episode pairs could be sampled")
    return np.asarray(sampled, dtype=np.int64)


class ResampledPairedTactileSE3Dataset(PairedTactileSE3Dataset):
    """Training split whose within-episode frame pairs can change each epoch."""

    def __init__(
        self,
        frames: np.ndarray,
        poses: np.ndarray,
        episode_rows: list[np.ndarray],
        pairs_per_episode: int,
        normalization: PairNormalization,
        seed: int,
    ) -> None:
        self.episode_rows = [np.asarray(rows, dtype=np.int64) for rows in episode_rows]
        self.pairs_per_episode = int(pairs_per_episode)
        initial_pairs = sample_episode_pairs(
            self.episode_rows, self.pairs_per_episode, np.random.default_rng(seed)
        )
        super().__init__(frames, poses, initial_pairs, normalization)

    def resample_pairs(self, seed: int | tuple[int, ...]) -> None:
        """Replace this epoch's pairs reproducibly without changing validation."""
        self.pairs = sample_episode_pairs(
            self.episode_rows,
            self.pairs_per_episode,
            np.random.default_rng(seed),
        )


def load_pair_datasets(
    root: str | Path,
    val_ratio: float,
    pairs_per_episode: int,
    target_limit: np.ndarray,
    seed: int,
    input_modality: str = "distributed_force",
) -> tuple[PairedTactileSE3Dataset, PairedTactileSE3Dataset, PairNormalization, dict]:
    if input_modality not in ("distributed_force", "depth"):
        raise ValueError(
            "input_modality must be 'distributed_force' or 'depth', got "
            f"{input_modality!r}"
        )
    root = Path(root).expanduser().resolve()
    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files below {root / 'data'}")
    source = pads.dataset([str(path) for path in files], format="parquet")
    tactile_keys = (
        (LEFT_KEY, RIGHT_KEY)
        if input_modality == "distributed_force"
        else DEPTH_KEYS
    )
    required = {*tactile_keys, TCP_KEY, "episode_index", "frame_index"}
    missing = required - set(source.schema.names)
    if missing:
        raise ValueError(
            f"Dataset lacks paired-motion fields {sorted(missing)}. "
            "Collect into a new --root with the OU collector."
        )

    meta = source.to_table(columns=["episode_index", "frame_index"])
    episodes = np.asarray(meta["episode_index"].to_pylist(), dtype=np.int64)
    frame_indices = np.asarray(meta["frame_index"].to_pylist(), dtype=np.int64)
    episode_rows_global = {
        int(episode): rows[np.argsort(frame_indices[rows])]
        for episode in np.unique(episodes)
        if len(rows := np.flatnonzero(episodes == episode)) >= 2
    }
    unique_episodes = np.asarray(sorted(episode_rows_global), dtype=np.int64)
    if len(unique_episodes) < 2:
        raise ValueError("At least two complete episodes are required for train/validation")
    split_rng = np.random.default_rng(seed)
    shuffled = split_rng.permutation(unique_episodes)
    n_val = min(len(shuffled) - 1, max(1, round(len(shuffled) * val_ratio)))
    val_eps = set(shuffled[:n_val].tolist())

    # Dynamic training pairs need every frame from every eligible episode, not
    # merely the frames touched by one startup-time random draw.
    selected_rows = np.unique(np.concatenate(list(episode_rows_global.values())))
    table = source.take(pa.array(selected_rows), columns=[*tactile_keys, TCP_KEY])
    if input_modality == "distributed_force":
        left = _force_to_chw(table[LEFT_KEY].to_pylist())
        right = _force_to_chw(table[RIGHT_KEY].to_pylist())
        frames = signed_log1p(np.concatenate((left, right), axis=1)).astype(np.float32)
    else:
        sides = []
        expected_hw = None
        for key in DEPTH_KEYS:
            decoded = []
            for local_index, value in enumerate(table[key].to_pylist()):
                try:
                    depth = decode_depth_png(value)
                except (ValueError, OSError, TypeError) as exc:
                    global_row = int(selected_rows[local_index])
                    raise ValueError(
                        f"Invalid {key} at dataset row {global_row}: {exc}"
                    ) from exc
                if expected_hw is None:
                    expected_hw = depth.shape
                elif depth.shape != expected_hw:
                    raise ValueError(
                        f"Depth image shape changed from {expected_hw} to {depth.shape} "
                        f"at dataset row {int(selected_rows[local_index])}"
                    )
                decoded.append(depth)
            sides.append(np.stack(decoded))
        # Each frame is [left, right, H, W]. Dataset pairing adds the
        # reference/current dimension, yielding [2, 2, H, W].
        # Keep decoded depth compact in memory. Dataset normalization converts
        # a selected pair to float32 only when __getitem__ is called.
        frames = np.stack(sides, axis=1)
    poses = np.asarray(table[TCP_KEY].to_pylist(), dtype=np.float32)
    row_to_local = {int(row): index for index, row in enumerate(selected_rows)}
    train_episode_rows = [
        np.asarray([row_to_local[int(row)] for row in episode_rows_global[int(episode)]])
        for episode in unique_episodes if int(episode) not in val_eps
    ]
    val_episode_rows = [
        np.asarray([row_to_local[int(row)] for row in episode_rows_global[int(episode)]])
        for episode in unique_episodes if int(episode) in val_eps
    ]
    # Validation pairs are drawn exactly once and never mutated by the
    # training loop, so validation loss remains comparable across epochs.
    val_pairs = sample_episode_pairs(
        val_episode_rows, pairs_per_episode, np.random.default_rng((seed, 0x56414C))
    )
    train_rows = np.unique(np.concatenate(train_episode_rows))
    if input_modality == "distributed_force":
        input_mean = frames[train_rows].mean(
            axis=(0, 2, 3), dtype=np.float64
        ).astype(np.float32)
        input_std = frames[train_rows].std(
            axis=(0, 2, 3), dtype=np.float64
        ).astype(np.float32)
        input_scale = None
    else:
        # Same preprocessing as train_pose_real_image.py: use one training-set
        # maximum for both fingers, without per-image normalization or resize.
        input_scale = max(float(frames[train_rows].max()), 1.0)
        input_mean = np.zeros(frames.shape[1], dtype=np.float32)
        input_std = np.full(frames.shape[1], input_scale, dtype=np.float32)
    normalization = PairNormalization(
        input_mean, np.maximum(input_std, 1e-6), np.asarray(target_limit, dtype=np.float32)
    )
    info = {
        "episodes": len(unique_episodes), "train_episodes": len(unique_episodes) - n_val,
        "val_episodes": n_val, "loaded_unique_frames": len(selected_rows),
        "train_frames": int(sum(map(len, train_episode_rows))),
        "val_frames": int(sum(map(len, val_episode_rows))),
        "pairs_per_episode": int(pairs_per_episode),
        "train_pairs_per_epoch": len(train_episode_rows) * pairs_per_episode,
        "val_pairs": len(val_pairs),
        "train_pair_sampling": "resampled_each_epoch_without_replacement",
        "val_pair_sampling": "fixed_at_startup",
        "train_episode_ids": sorted(int(value) for value in unique_episodes
                                     if int(value) not in val_eps),
        "val_episode_ids": sorted(int(value) for value in val_eps),
        "input_modality": input_modality,
        "input_shape_per_frame": list(frames.shape[1:]),
        "input_scale": input_scale,
    }
    return (
        ResampledPairedTactileSE3Dataset(
            frames, poses, train_episode_rows, pairs_per_episode,
            normalization, seed,
        ),
        PairedTactileSE3Dataset(frames, poses, val_pairs, normalization),
        normalization, info,
    )
