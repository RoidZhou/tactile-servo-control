#!/usr/bin/env python3
"""Merge multiple compatible LeRobot v2.x datasets without modifying sources.

This utility is intended for datasets written by
``collect_lerobot_xenstactile_contactline.py``.  It rewrites episode/global/task
indices, preserves per-episode statistics and sidecar metadata, and rebuilds
``meta/info.json``.  Video-backed datasets are deliberately rejected because
their frame references require a separate video remux/copy path; the contact-line
collector stores its image features directly in parquet and has total_videos=0.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def replace_column(table: pa.Table, name: str, values: pa.Array) -> pa.Table:
    index = table.schema.get_field_index(name)
    if index < 0:
        raise ValueError(f"Required parquet column is missing: {name}")
    return table.set_column(index, name, values)


def constant_like(table: pa.Table, name: str, value: int) -> pa.Array:
    field = table.schema.field(name)
    return pa.array([value] * table.num_rows, type=field.type)


def check_compatible(reference: dict[str, Any], candidate: dict[str, Any], source: Path) -> None:
    for key in ("codebase_version", "robot_type", "fps", "features", "chunks_size"):
        if candidate.get(key) != reference.get(key):
            raise ValueError(f"Incompatible {key!r} in dataset {source}")
    if int(candidate.get("total_videos", 0)) != 0:
        raise ValueError(f"Video-backed dataset is not supported: {source}")


def find_episode_file(root: Path, info: dict[str, Any], episode_index: int) -> Path:
    chunk_size = int(info.get("chunks_size", 1000))
    relative = info["data_path"].format(
        episode_chunk=episode_index // chunk_size,
        episode_index=episode_index,
    )
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"Episode parquet not found: {path}")
    return path


def copy_episode_sidecars(source: Path, output: Path, old_ep: int, new_ep: int) -> None:
    """Copy files under meta subdirectories whose name contains the episode id."""
    meta = source / "meta"
    for path in meta.rglob(f"episode_{old_ep:06d}*"):
        if not path.is_file():
            continue
        relative = path.relative_to(meta)
        new_name = path.name.replace(
            f"episode_{old_ep:06d}", f"episode_{new_ep:06d}", 1
        )
        destination = output / "meta" / relative.parent / new_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)


def merge(sources: list[Path], output: Path, overwrite: bool) -> None:
    sources = [path.resolve() for path in sources]
    output = output.resolve()
    if len(sources) < 2:
        raise ValueError("Provide at least two --source datasets")
    if len(set(sources)) != len(sources):
        raise ValueError("The same source dataset was supplied more than once")
    if output in sources:
        raise ValueError("Output must not be one of the source directories")

    infos: list[dict[str, Any]] = []
    for source in sources:
        info_path = source / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"Not a LeRobot dataset (missing {info_path})")
        infos.append(read_json(info_path))
    for source, info in zip(sources[1:], infos[1:]):
        check_compatible(infos[0], info, source)

    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output}; pass --overwrite")
        shutil.rmtree(output)
    (output / "meta").mkdir(parents=True)

    # Build one shared task table, even if the input task indices differ.
    task_to_index: dict[str, int] = {}
    output_tasks: list[dict[str, Any]] = []
    source_task_maps: list[dict[int, int]] = []
    for source in sources:
        mapping: dict[int, int] = {}
        for row in read_jsonl(source / "meta" / "tasks.jsonl"):
            task = str(row["task"])
            if task not in task_to_index:
                task_to_index[task] = len(task_to_index)
                output_tasks.append({"task_index": task_to_index[task], "task": task})
            mapping[int(row["task_index"])] = task_to_index[task]
        source_task_maps.append(mapping)

    output_episodes: list[dict[str, Any]] = []
    output_stats: list[dict[str, Any]] = []
    next_episode = 0
    next_index = 0
    total_frames = 0
    chunk_size = int(infos[0].get("chunks_size", 1000))

    for source, info, task_map in zip(sources, infos, source_task_maps):
        episodes = sorted(
            read_jsonl(source / "meta" / "episodes.jsonl"),
            key=lambda row: int(row["episode_index"]),
        )
        stats_by_episode = {
            int(row["episode_index"]): row
            for row in read_jsonl(source / "meta" / "episodes_stats.jsonl")
        }
        if len(episodes) != int(info["total_episodes"]):
            raise ValueError(
                f"{source}: info total_episodes={info['total_episodes']} but "
                f"episodes.jsonl contains {len(episodes)} rows"
            )

        for episode_row in episodes:
            old_episode = int(episode_row["episode_index"])
            parquet_path = find_episode_file(source, info, old_episode)
            table = pq.read_table(parquet_path)
            frame_count = table.num_rows
            declared_length = int(episode_row["length"])
            if frame_count != declared_length:
                raise ValueError(
                    f"{parquet_path}: {frame_count} rows != declared length {declared_length}"
                )

            old_task_values = table.column("task_index").to_pylist()
            try:
                new_task_values = [task_map[int(value)] for value in old_task_values]
            except KeyError as error:
                raise ValueError(f"{source}: task_index {error.args[0]} absent from tasks.jsonl") from error

            table = replace_column(
                table, "episode_index", constant_like(table, "episode_index", next_episode)
            )
            index_type = table.schema.field("index").type
            table = replace_column(
                table,
                "index",
                pa.array(range(next_index, next_index + frame_count), type=index_type),
            )
            task_type = table.schema.field("task_index").type
            table = replace_column(
                table, "task_index", pa.array(new_task_values, type=task_type)
            )

            destination = (
                output
                / "data"
                / f"chunk-{next_episode // chunk_size:03d}"
                / f"episode_{next_episode:06d}.parquet"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, destination)

            new_episode_row = dict(episode_row)
            new_episode_row["episode_index"] = next_episode
            output_episodes.append(new_episode_row)

            if old_episode not in stats_by_episode:
                raise ValueError(f"{source}: statistics missing for episode {old_episode}")
            stat_row = dict(stats_by_episode[old_episode])
            stat_row["episode_index"] = next_episode
            # Keep metadata internally consistent with rewritten parquet columns.
            stats = stat_row.get("stats", {})
            if "episode_index" in stats:
                stats["episode_index"] = {
                    "min": [next_episode], "max": [next_episode],
                    "mean": [float(next_episode)], "std": [0.0],
                    "count": [frame_count],
                }
            if "index" in stats:
                first, last = next_index, next_index + frame_count - 1
                stats["index"]["min"] = [first]
                stats["index"]["max"] = [last]
                stats["index"]["mean"] = [(first + last) / 2.0]
                stats["index"]["count"] = [frame_count]
            output_stats.append(stat_row)
            copy_episode_sidecars(source, output, old_episode, next_episode)

            print(
                f"[{source.name}] episode {old_episode} -> {next_episode} "
                f"({frame_count} frames)"
            )
            next_episode += 1
            next_index += frame_count
            total_frames += frame_count

    write_jsonl(output / "meta" / "tasks.jsonl", output_tasks)
    write_jsonl(output / "meta" / "episodes.jsonl", output_episodes)
    write_jsonl(output / "meta" / "episodes_stats.jsonl", output_stats)

    merged_info = dict(infos[0])
    merged_info.update(
        total_episodes=next_episode,
        total_frames=total_frames,
        total_tasks=len(output_tasks),
        total_videos=0,
        total_chunks=(next_episode + chunk_size - 1) // chunk_size,
        splits={"train": f"0:{next_episode}"},
    )
    with (output / "meta" / "info.json").open("w", encoding="utf-8") as stream:
        json.dump(merged_info, stream, ensure_ascii=False, indent=4)
        stream.write("\n")

    print(
        f"Merged {len(sources)} datasets -> {output}\n"
        f"episodes={next_episode}, frames={total_frames}, tasks={len(output_tasks)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", action="append", required=True, type=Path,
        help="Input dataset root. Repeat this option for every dataset.",
    )
    parser.add_argument("--output", required=True, type=Path, help="New merged dataset root")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    merge(arguments.source, arguments.output, arguments.overwrite)
