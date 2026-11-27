"""Read raw nuScenes keyframes for single-sweep HEAL BEVFusion calibration.

Indexing needs only the standard library and does not open sensor payloads.
The root must contain the official metadata version directory and extracted
``samples/<sensor>`` directories; a partially extracted split is supported.
"""

from collections import defaultdict, deque
from contextlib import closing
from dataclasses import dataclass
import json
from json.decoder import WHITESPACE
import os
from pathlib import Path, PurePosixPath
import random
from typing import Iterator


CAMERA_ORDER = (
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT",
)
_CHANNELS = (*CAMERA_ORDER, "LIDAR_TOP")
_CHUNK_CHARS = 1 << 20
_MAX_RECORD_CHARS = 16 * _CHUNK_CHARS


@dataclass(frozen=True, slots=True)
class _SensorRecord:
    path: Path
    timestamp: int
    calibration: dict
    ego_pose: dict


@dataclass(frozen=True, slots=True)
class CalibrationFrame:
    """A complete keyframe, indexed without decoding its images or points."""

    token: str
    scene: str
    cameras: tuple[_SensorRecord, ...]
    lidar: _SensorRecord


def _json_rows(path: Path) -> Iterator[dict]:
    """Stream a JSON array of objects regardless of line formatting."""
    with path.open(encoding="utf-8-sig") as source:
        decoder = json.JSONDecoder()
        buffer = ""
        pos = 0

        def next_nonspace():
            nonlocal buffer, pos
            while True:
                pos = WHITESPACE.match(buffer, pos).end()
                if pos < len(buffer):
                    return buffer[pos]
                buffer = source.read(_CHUNK_CHARS)
                pos = 0
                if not buffer:
                    return ""

        if next_nonspace() != "[":
            raise ValueError(f"Expected a JSON array in {path}")
        pos += 1
        first = True
        while True:
            ch = next_nonspace()
            if ch == "]":
                pos += 1
                if next_nonspace():
                    raise ValueError(f"Unexpected content after JSON array in {path}")
                return
            if not first:
                if ch != ",":
                    raise ValueError(f"Expected ',' or ']' after JSON object in {path}")
                pos += 1
                ch = next_nonspace()
            if ch != "{":
                raise ValueError(f"Expected a JSON object in {path}")

            while True:
                try:
                    row, pos = decoder.raw_decode(buffer, pos)
                    break
                except json.JSONDecodeError as exc:
                    if len(buffer) - pos >= _MAX_RECORD_CHARS:
                        raise ValueError(f"Invalid or oversized JSON object in {path}") from exc
                    chunk = source.read(_CHUNK_CHARS)
                    if not chunk:
                        raise ValueError(f"Invalid or truncated JSON object in {path}") from exc
                    buffer = buffer[pos:] + chunk
                    pos = 0
            first = False
            yield row


def _spread_frames(candidates, sample_scenes, limit, seed):
    by_scene = defaultdict(list)
    for token, rows in candidates.items():
        if len(rows) == len(_CHANNELS):
            by_scene[sample_scenes[token]].append(token)

    # Sort before shuffling, so the result does not depend on metadata row order.
    rng = random.Random(seed)
    scenes = sorted(by_scene)
    rng.shuffle(scenes)
    for scene in scenes:
        by_scene[scene].sort()
        rng.shuffle(by_scene[scene])

    selected = []
    remaining = deque(scenes)
    while remaining and len(selected) < limit:
        scene = remaining.popleft()
        tokens = by_scene[scene]
        selected.append(tokens.pop())
        if tokens:
            remaining.append(scene)
    return selected


def index_calibration_frames(
    root: Path, *, version: str, limit: int, seed: int
) -> list[CalibrationFrame]:
    """Select up to ``limit`` complete frames, round-robin across shuffled scenes.

    Each selected scene contributes one random frame before any scene contributes
    another. Missing camera/LiDAR files are skipped using one directory listing
    per channel, not per-frame stats. No sweep or annotation data is required.
    """
    if limit < 0:
        raise ValueError("limit must be non-negative")
    if limit == 0:
        return []

    root = Path(root).resolve()
    meta = root / version
    sample_scenes = {
        row["token"]: row["scene_token"] for row in _json_rows(meta / "sample.json")
    }
    camera_and_lidar_files = {}
    for channel in _CHANNELS:
        try:
            names = os.listdir(root / "samples" / channel)
        except FileNotFoundError:
            return []
        if not names:
            return []
        camera_and_lidar_files[channel] = set(names)

    sensor_channels = {
        row["token"]: row["channel"] for row in _json_rows(meta / "sensor.json")
    }
    calibrations = {}
    channels_by_calibration = {}
    for row in _json_rows(meta / "calibrated_sensor.json"):
        channel = sensor_channels[row["sensor_token"]]
        if channel in camera_and_lidar_files:
            token = row["token"]
            calibrations[token] = row
            channels_by_calibration[token] = channel

    candidates = defaultdict(dict)
    for row in _json_rows(meta / "sample_data.json"):
        if not row["is_key_frame"]:
            continue
        calibration_token = row["calibrated_sensor_token"]
        channel = channels_by_calibration.get(calibration_token)
        token = row["sample_token"]
        if channel is None or token not in sample_scenes:
            continue
        filename = PurePosixPath(row["filename"])
        if (
            filename.parts[:-1] != ("samples", channel)
            or filename.name not in camera_and_lidar_files[channel]
        ):
            continue
        view = (row["filename"], row["timestamp"], calibration_token, row["ego_pose_token"])
        views = candidates[token]
        # Official metadata has one keyframe per channel. If duplicates are
        # present, prefer the earliest timestamp (and use the path to break ties).
        current = views.get(channel)
        if current is None or (view[1], view) < (current[1], current):
            views[channel] = view

    selected = _spread_frames(candidates, sample_scenes, limit, seed)
    if not selected:
        return []
    candidates = {token: candidates[token] for token in selected}
    wanted_poses = {view[3] for views in candidates.values() for view in views.values()}
    ego_poses = {}
    with closing(_json_rows(meta / "ego_pose.json")) as rows:
        for row in rows:
            token = row["token"]
            if token in wanted_poses:
                ego_poses[token] = row
                wanted_poses.remove(token)
                if not wanted_poses:
                    break
    if wanted_poses:
        raise ValueError(f"{meta / 'ego_pose.json'} lacks {len(wanted_poses)} referenced pose(s)")

    frames = []
    for token in selected:
        records = []
        for channel in _CHANNELS:
            filename, timestamp, calibration_token, ego_token = candidates[token][channel]
            records.append(_SensorRecord(
                root / filename, timestamp, calibrations[calibration_token], ego_poses[ego_token]
            ))
        frames.append(CalibrationFrame(token, sample_scenes[token], tuple(records[:-1]), records[-1]))
    return frames


def _rigid_transform(record):
    import numpy as np

    quat = np.asarray(record["rotation"], dtype=np.float64)
    translation = np.asarray(record["translation"], dtype=np.float64)
    if quat.shape != (4,) or translation.shape != (3,):
        raise ValueError("Expected a wxyz quaternion and xyz translation")
    norm = np.linalg.norm(quat)
    if not np.isfinite(norm) or norm == 0 or not np.isfinite(translation).all():
        raise ValueError("Invalid rigid transform in nuScenes metadata")
    w, x, y, z = quat / norm
    transform = np.eye(4)
    transform[:3, :3] = (
        (1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)),
        (2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)),
        (2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)),
    )
    transform[:3, 3] = translation
    return transform


def _sensor_to_global(sensor: _SensorRecord):
    # Ego poses belong to each sensor recording timestamp, not to the common
    # sample timestamp. Use the associated pose token even for the same keyframe.
    return _rigid_transform(sensor.ego_pose) @ _rigid_transform(sensor.calibration)


def load_calibration_frame(frame: CalibrationFrame, *, image_size: tuple[int, int]) -> dict:
    """Decode a keyframe as CPU tensors with the HEAL local demo preprocessing."""
    import numpy as np
    import torch
    from horizon_plugin_pytorch.nn.functional import bgr_to_yuv444
    from PIL import Image
    from torchvision.transforms.functional import pil_to_tensor, resize

    height, width = image_size
    if min(height, width) <= 0:
        raise ValueError("image_size must be positive (height, width)")
    lidar_to_global = _sensor_to_global(frame.lidar)
    projections, images = [], []
    for camera in frame.cameras:
        with Image.open(camera.path) as source:
            original_width, original_height = source.size
            rgb = pil_to_tensor(source.convert("RGB"))
        images.append(resize(rgb, [height, width], antialias=True))

        lidar_to_camera = np.linalg.solve(_sensor_to_global(camera), lidar_to_global)
        intrinsic = np.asarray(camera.calibration["camera_intrinsic"], dtype=np.float64)
        if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
            raise ValueError(f"Invalid intrinsics for {camera.path}")
        projection = np.eye(4)
        projection[:3, :3] = intrinsic
        # Scale image coordinates to the resized frame, without changing depth.
        projection[0] *= width / original_width
        projection[1] *= height / original_height
        projections.append(projection @ lidar_to_camera)

    img = (bgr_to_yuv444(torch.stack(images), True).float() - 128.0) / 128.0
    # Read the exact byte count: np.fromfile alone would ignore trailing bytes.
    raw = frame.lidar.path.read_bytes()
    if not raw or len(raw) % 20:
        raise ValueError(f"Invalid nuScenes point cloud (expected float32 Nx5): {frame.lidar.path}")
    points = np.frombuffer(raw, dtype="<f4").reshape(-1, 5).astype(np.float32, copy=True)
    if not np.isfinite(points).all():
        raise ValueError(f"Non-finite points in {frame.lidar.path}")
    # nuScenes raw column 5 is ring; HEAL uses time lag (0 for this keyframe).
    points[:, 4] = 0
    return {
        "img": img.contiguous(),
        "points": [torch.from_numpy(points)],
        "seq_meta": [{
            "lidar2img": np.asarray(projections, dtype=np.float32)[None],
            "meta": [{"scene": frame.scene}],
        }],
    }


def iter_calibration_frames(
    root: Path, *, version: str, limit: int, seed: int, image_size: tuple[int, int]
) -> Iterator[tuple[str, dict]]:
    """Yield ``(sample_token, HEAL model_input)`` for selected keyframes."""
    for frame in index_calibration_frames(root, version=version, limit=limit, seed=seed):
        yield frame.token, load_calibration_frame(frame, image_size=image_size)
