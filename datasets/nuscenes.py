"""
Read individual nuScenes LIDAR_TOP keyframes as raw NumPy point dictionaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._json import iter_json_records


def read_lidar_bin(path: str | Path) -> np.ndarray:
    """Return float32 ``[x, y, z, intensity, ring]`` records, shaped ``(N, 5)``.

    XYZ remains in the native LiDAR sensor frame. This function only decodes a
    scan; keyframe and sensor identity require the nuScenes metadata.
    """
    path = Path(path)
    # Compute the number of bytes per row and the size of the file
    record_bytes = 5 * np.dtype(np.float32).itemsize
    size = path.stat().st_size
    # Check if the file is empty or if the file does not have the requested format
    if size == 0:
        raise ValueError(f"LiDAR scan is empty: {path}")
    if size % record_bytes:
        raise ValueError(
            f"LiDAR scan must contain complete {record_bytes}-byte records "
            f"(5 float32 values), got {size} bytes: {path}"
        )
    values = np.fromfile(path, dtype=np.float32)
    if values.nbytes != size:
        raise ValueError(f"LiDAR scan size changed while reading: {path}")
    points = values.reshape(-1, 5)
    if not np.isfinite(points).all():
        raise ValueError(f"LiDAR scan contains non-finite values: {path}")
    return points


@dataclass(frozen=True)
class _Keyframe:
    sample_token: str
    sample_data_token: str
    scene_token: str
    timestamp_us: int
    lidar_path: Path


class NuScenesKeyframeReader:
    """Index and read one LIDAR_TOP keyframe for a nuScenes sample token.

    Construct once and reuse it. Metadata is streamed and only a small keyframe
    index is retained; scan files are opened on demand. All samples in the
    selected metadata version are indexed, without train/validation filtering.

    ``read`` returns ``{"point": ..., "meta": ...}``. Point fields are CPU
    NumPy arrays: coord/origin_coord ``(N, 3)``, strength ``(N, 1)``, feat
    ``(N, 4)`` (all float32), and offset ``[N]`` (int64). Intensity is divided
    by 255 exactly once. Ring indices are excluded from the model features.

    These are RAW scans: there is no cropping, augmentation, sweep aggregation,
    grid sampling, or tensor conversion. They are not ready for a LitePT forward
    until grid sampling has selected representatives and generated grid_coord.
    Future spatial transforms must rebuild feat from the transformed coord and
    strength, and preserve the corresponding unaugmented origin_coord rows.
    """

    def __init__(
        self, dataroot: str | Path, *, version: str = "v1.0-mini"
    ) -> None:
        self.dataroot = Path(dataroot).expanduser().resolve()
        if not self.dataroot.is_dir():
            raise NotADirectoryError(f"nuScenes root is not a directory: {self.dataroot}")
        if not version or Path(version).name != version or version in {".", ".."}:
            raise ValueError("version must name a metadata directory, e.g. 'v1.0-mini'")
        self.version = version
        metadata_dir = self.dataroot / version

        lidar_sensor_tokens = {
            record["token"]
            for record in iter_json_records(metadata_dir / "sensor.json")
            if record.get("channel") == "LIDAR_TOP"
            and record.get("modality") == "lidar"
        }
        if not lidar_sensor_tokens:
            raise ValueError(f"No LIDAR_TOP lidar sensor in {metadata_dir / 'sensor.json'}")

        lidar_calibrations = {
            record["token"]
            for record in iter_json_records(metadata_dir / "calibrated_sensor.json")
            if record["sensor_token"] in lidar_sensor_tokens
        }
        sample_scenes = {
            record["token"]: record["scene_token"]
            for record in iter_json_records(metadata_dir / "sample.json")
        }

        self._keyframes: dict[str, _Keyframe] = {}
        for record in iter_json_records(metadata_dir / "sample_data.json"):
            if not record.get("is_key_frame"):
                continue
            if record["calibrated_sensor_token"] not in lidar_calibrations:
                continue
            sample_token = record["sample_token"]
            if sample_token not in sample_scenes:
                raise ValueError(
                    f"LIDAR_TOP reading {record['token']!r} refers to unknown "
                    f"sample {sample_token!r}"
                )
            if sample_token in self._keyframes:
                raise ValueError(
                    f"Multiple LIDAR_TOP keyframes for sample {sample_token!r}"
                )
            filename = Path(record["filename"])
            if filename.is_absolute() or ".." in filename.parts:
                raise ValueError(f"Expected a dataset-relative LiDAR filename: {filename}")
            self._keyframes[sample_token] = _Keyframe(
                sample_token=sample_token,
                sample_data_token=record["token"],
                scene_token=sample_scenes[sample_token],
                timestamp_us=record["timestamp"],
                lidar_path=self.dataroot / filename,
            )

        if not self._keyframes:
            raise ValueError(f"No LIDAR_TOP keyframes found in {metadata_dir}")
        self.sample_tokens = tuple(
            token for token in sample_scenes if token in self._keyframes
        )

    def __len__(self) -> int:
        return len(self._keyframes)

    def read(self, sample_token: str) -> dict:
        """Read a fresh raw point dictionary and metadata for ``sample_token``."""
        try:
            frame = self._keyframes[sample_token]
        except KeyError:
            raise KeyError(
                f"No LIDAR_TOP keyframe for sample {sample_token!r} "
                f"in {self.version}"
            ) from None

        raw = read_lidar_bin(frame.lidar_path)
        coord = np.ascontiguousarray(raw[:, :3])
        strength = np.ascontiguousarray(raw[:, 3:4] / np.float32(255))
        point = {
            "coord": coord,
            "origin_coord": coord.copy(),
            "strength": strength,
            "feat": np.concatenate((coord, strength), axis=1),
            "offset": np.array([coord.shape[0]], dtype=np.int64),
        }
        meta = {
            "sample_token": frame.sample_token,
            "sample_data_token": frame.sample_data_token,
            "scene_token": frame.scene_token,
            "timestamp_us": frame.timestamp_us,
            "lidar_path": str(frame.lidar_path),
            "coordinate_frame": "LIDAR_TOP",
            "version": self.version,
        }
        return {"point": point, "meta": meta}
