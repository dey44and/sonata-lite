import json

import numpy as np
import pytest

from datasets import NuScenesKeyframeReader, read_lidar_bin


@pytest.fixture
def dataset(tmp_path):
    metadata = tmp_path / "v1.0-mini"
    metadata.mkdir()
    scans = tmp_path / "samples" / "LIDAR_TOP"
    scans.mkdir(parents=True)
    raw = np.array(
        [[1, -2, 3, 255, 31], [-4, 5, -6, 127.5, 0]], dtype=np.float32
    )
    raw.tofile(scans / "first.pcd.bin")
    raw[:1].tofile(scans / "second.pcd.bin")

    def reading(token, sample, calibration, filename, keyframe=True):
        return {
            "token": token,
            "sample_token": sample,
            "calibrated_sensor_token": calibration,
            "filename": filename,
            "is_key_frame": keyframe,
            "timestamp": 1_532_402_927_647_951,
        }

    tables = {
        "sensor": [
            {"token": "lidar", "channel": "LIDAR_TOP", "modality": "lidar"},
            {"token": "camera", "channel": "CAM_FRONT", "modality": "camera"},
        ],
        "calibrated_sensor": [
            {"token": "lidar-calib", "sensor_token": "lidar"},
            {"token": "camera-calib", "sensor_token": "camera"},
        ],
        "sample": [
            {"token": "first", "scene_token": "scene-a"},
            {"token": "second", "scene_token": "scene-b"},
        ],
        "sample_data": [
            reading("sweep", "first", "lidar-calib", "sweeps/missing.bin", False),
            # A misleading filename must not override the actual sensor metadata.
            reading("camera-data", "first", "camera-calib", "samples/LIDAR_TOP/fake.bin"),
            reading("lidar-first", "first", "lidar-calib", "samples/LIDAR_TOP/first.pcd.bin"),
            reading("lidar-second", "second", "lidar-calib", "samples/LIDAR_TOP/second.pcd.bin"),
        ],
    }
    for name, records in tables.items():
        (metadata / f"{name}.json").write_text(json.dumps(records), encoding="utf-8")
    return tmp_path, raw, tables


def test_keyframe_fields_and_metadata(dataset):
    root, raw, _ = dataset
    reader = NuScenesKeyframeReader(root)
    assert len(reader) == 2
    assert reader.sample_tokens == ("first", "second")
    item = reader.read("first")
    point, meta = item["point"], item["meta"]

    np.testing.assert_array_equal(point["coord"], raw[:, :3])
    np.testing.assert_array_equal(point["origin_coord"], raw[:, :3])
    np.testing.assert_array_equal(point["strength"], [[1], [0.5]])
    np.testing.assert_array_equal(point["feat"], [[1, -2, 3, 1], [-4, 5, -6, 0.5]])
    np.testing.assert_array_equal(point["offset"], [2])
    assert point["offset"].dtype == np.int64
    for key in ("coord", "origin_coord", "strength", "feat"):
        assert point[key].dtype == np.float32
        assert point[key].flags.c_contiguous
    assert "grid_coord" not in point
    assert "grid_size" not in point
    assert not np.shares_memory(point["coord"], point["origin_coord"])
    assert meta == {
        "sample_token": "first",
        "sample_data_token": "lidar-first",
        "scene_token": "scene-a",
        "timestamp_us": 1_532_402_927_647_951,
        "lidar_path": str(root / "samples/LIDAR_TOP/first.pcd.bin"),
        "coordinate_frame": "LIDAR_TOP",
        "version": "v1.0-mini",
    }

    point["coord"][0, 0] = 100
    assert point["origin_coord"][0, 0] == 1
    assert reader.read("first")["point"]["coord"][0, 0] == 1


@pytest.mark.parametrize("size", [0, 1, 16, 21])
def test_rejects_empty_or_truncated_records(tmp_path, size):
    path = tmp_path / "scan.bin"
    path.write_bytes(b"\0" * size)
    with pytest.raises(ValueError, match="empty|complete"):
        read_lidar_bin(path)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_rejects_nonfinite_scan(tmp_path, bad_value):
    path = tmp_path / "scan.bin"
    np.array([[1, 2, 3, bad_value, 0]], dtype=np.float32).tofile(path)
    with pytest.raises(ValueError, match="non-finite"):
        read_lidar_bin(path)


def test_raw_decoder_preserves_all_five_fields(dataset):
    root, raw, _ = dataset
    actual = read_lidar_bin(root / "samples/LIDAR_TOP/first.pcd.bin")
    np.testing.assert_array_equal(actual, raw)
    assert actual.dtype == np.float32


def test_unknown_sample_and_missing_scan(dataset):
    root, _, _ = dataset
    reader = NuScenesKeyframeReader(root)
    with pytest.raises(KeyError, match="unknown"):
        reader.read("unknown")
    (root / "samples/LIDAR_TOP/second.pcd.bin").unlink()
    with pytest.raises(FileNotFoundError):
        reader.read("second")


def test_duplicate_keyframe_is_not_silently_overwritten(dataset):
    root, _, tables = dataset
    records = tables["sample_data"]
    records.append({**records[2], "token": "duplicate"})
    (root / "v1.0-mini/sample_data.json").write_text(json.dumps(records))
    with pytest.raises(ValueError, match="Multiple LIDAR_TOP keyframes"):
        NuScenesKeyframeReader(root)


def test_keyframe_referencing_unknown_sample_is_rejected(dataset):
    root, _, tables = dataset
    records = tables["sample_data"]
    records[2]["sample_token"] = "missing-parent"
    (root / "v1.0-mini/sample_data.json").write_text(json.dumps(records))
    with pytest.raises(ValueError, match="unknown sample"):
        NuScenesKeyframeReader(root)
