"""Variable-length SSL batches, scene selection, and prefetch-safe schedules."""

import copy
import json
import random
import sys
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import DataLoader

from augmentation import MaskSchedule, NuScenesSSLViews
from trainer.data import (
    NuScenesSSLDataset,
    StepBatchSampler,
    collate_ssl_views,
    seed_worker,
)


@pytest.fixture
def keyframes(tmp_path):
    metadata = tmp_path / "v1.0-mini"
    metadata.mkdir()
    scans = tmp_path / "samples" / "LIDAR_TOP"
    scans.mkdir(parents=True)
    samples, readings = [], []
    for index, count in enumerate((96, 75, 81)):
        coord = np.column_stack(
            (
                np.arange(count) * 0.12,
                (np.arange(count) % 9) * 0.31,
                (np.arange(count) % 5) * 0.23,
            )
        )
        raw = np.column_stack((coord, np.arange(count) % 256, np.arange(count) % 32))
        filename = f"samples/LIDAR_TOP/{index}.pcd.bin"
        raw.astype(np.float32).tofile(tmp_path / filename)
        samples.append({"token": f"sample-{index}", "scene_token": f"scene-{index % 2}"})
        readings.append(
            {
                "token": f"data-{index}",
                "sample_token": f"sample-{index}",
                "calibrated_sensor_token": "calibration",
                "filename": filename,
                "is_key_frame": True,
                "timestamp": index,
            }
        )
    tables = {
        "sensor": [{"token": "sensor", "channel": "LIDAR_TOP", "modality": "lidar"}],
        "calibrated_sensor": [{"token": "calibration", "sensor_token": "sensor"}],
        "sample": samples,
        "sample_data": readings,
        "scene": [
            {"token": "scene-0", "name": "scene-0061"},
            {"token": "scene-1", "name": "scene-0103"},
        ],
    }
    for name, records in tables.items():
        (metadata / f"{name}.json").write_text(json.dumps(records))
    return tmp_path


def augmentation(**kwargs):
    return NuScenesSSLViews(
        global_view_scale=(0.7, 0.9),
        local_view_scale=(0.2, 0.3),
        min_points=4,
        **kwargs,
    )


def assert_views_equal(left, right):
    for group in ("global_views", "masked_global_views", "local_views"):
        assert len(left[group]) == len(right[group])
        for first, second in zip(left[group], right[group]):
            assert first.keys() == second.keys()
            for key in first:
                np.testing.assert_array_equal(first[key], second[key])
    assert left["training_step"] == right["training_step"]


def test_dataset_selects_scenes_and_reads_fresh_views(keyframes):
    dataset = NuScenesSSLDataset(
        keyframes, scene_names=["scene-0061"], augmentation=augmentation(), seed=17
    )
    assert dataset.sample_tokens == ("sample-0", "sample-2")
    assert len(dataset) == 2
    first = dataset[(0, 3)]
    expected = copy.deepcopy(first)
    assert first["meta"]["sample_token"] == "sample-0"
    assert first["meta"]["scene_token"] == "scene-0"
    first["global_views"][0]["feat"][:] = 12345
    assert_views_equal(dataset[(0, 3)], expected)
    assert dataset[(1, 3)]["meta"]["sample_token"] == "sample-2"
    limited = NuScenesSSLDataset(keyframes, max_samples=1)
    assert limited.sample_tokens == ("sample-0",)


def test_dataset_randomness_depends_on_explicit_step_and_preserves_callers(keyframes):
    dataset = NuScenesSSLDataset(keyframes, augmentation=augmentation(), seed=9)
    np.random.seed(12)
    random.seed(13)
    numpy_state, python_state = np.random.get_state(), random.getstate()
    first = dataset[(0, 1)]
    actual_state = np.random.get_state()
    assert actual_state[0] == numpy_state[0]
    np.testing.assert_array_equal(actual_state[1], numpy_state[1])
    assert actual_state[2:] == numpy_state[2:]
    assert random.getstate() == python_state
    np.random.seed(900)
    random.seed(901)
    assert_views_equal(first, dataset[(0, 1)])
    next_step = dataset[(0, 2)]
    assert not np.array_equal(
        first["global_views"][0]["coord"], next_step["global_views"][0]["coord"]
    )


def test_dataset_passes_requested_step_to_mask_schedule(keyframes):
    schedule = MaskSchedule(total_steps=100, warmup_ratio=0.2)
    dataset = NuScenesSSLDataset(
        keyframes, augmentation=augmentation(mask_schedule=schedule)
    )
    for step in (0, 7, 19, 20, 99):
        item = dataset[(0, step)]
        assert item["training_step"] == step
        for view in item["masked_global_views"]:
            assert view["mask_step"] == step
            assert (view["mask_size"], view["mask_ratio"]) == schedule(step)


def test_official_split_uses_devkit_scene_names(keyframes, monkeypatch):
    # Keep the test independent of the optional devkit installation while
    # checking the same public function called against the real devkit.
    nuscenes = types.ModuleType("nuscenes")
    utils = types.ModuleType("nuscenes.utils")
    splits = types.ModuleType("nuscenes.utils.splits")
    splits.create_splits_scenes = lambda verbose: {"mini_train": ["scene-0061"]}
    for name, module in (
        ("nuscenes", nuscenes), ("nuscenes.utils", utils), ("nuscenes.utils.splits", splits)
    ):
        monkeypatch.setitem(sys.modules, name, module)
    dataset = NuScenesSSLDataset(keyframes, split="mini_train")
    assert dataset.sample_tokens == ("sample-0", "sample-2")


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"scene_names": ["missing"]}, "absent"),
        ({"scene_names": [], "max_samples": 1}, "no LIDAR_TOP"),
        ({"split": "train"}, "requires"),
        ({"split": "unrecognized"}, "unsupported"),
        ({"split": "mini_train", "scene_names": ["scene-0061"]}, "either"),
    ],
)
def test_scene_selection_rejects_invalid_requests(keyframes, kwargs, message):
    with pytest.raises(ValueError, match=message):
        NuScenesSSLDataset(keyframes, **kwargs)


def make_view(count, identity, masked=False):
    coord = np.full((count, 3), identity, dtype=np.float64)
    view = {
        "coord": coord,
        "origin_coord": coord.copy(),
        "feat": np.full((count, 4), identity, dtype=np.float64),
        "grid_coord": np.arange(count * 3, dtype=np.int32).reshape(count, 3),
        "grid_size": 0.1,
        "offset": np.array([count]),
        "patch_id": np.arange(count),
        "strength": np.zeros((count, 1)),
    }
    if masked:
        view["mask"] = np.arange(count) % 2 == 0
    return view


def make_sample(identity, step=3):
    globals_ = [make_view(identity + 2, identity * 10), make_view(identity + 3, identity * 10 + 1)]
    masked = copy.deepcopy(globals_)
    for view in masked:
        view["mask"] = np.arange(len(view["coord"])) % 2 == 0
    return {
        "global_views": globals_,
        "masked_global_views": masked,
        "local_views": [make_view(1, identity * 10 + index + 2) for index in range(4)],
        "meta": {"sample_token": f"sample-{identity}"},
        "training_step": step,
    }


def test_collation_preserves_view_boundaries_and_scene_major_order():
    first, second = make_sample(0), make_sample(2)
    batch = collate_ssl_views([first, second])
    assert batch["step"] == 3
    assert batch["batch_size"] == 2
    assert batch["view_counts"] == {"global": 2, "masked_global": 2, "local": 4}
    assert batch["meta"] == [first["meta"], second["meta"]]
    packed = batch["global"]
    assert packed["offset"].tolist() == [2, 5, 9, 14]
    assert packed["feat"][:, 0].tolist() == [0] * 2 + [1] * 3 + [20] * 4 + [21] * 5
    assert batch["local"]["offset"].tolist() == list(range(1, 9))
    for group in ("global", "masked_global", "local"):
        for key in ("coord", "origin_coord", "feat"):
            assert batch[group][key].dtype == torch.float32
            assert batch[group][key].is_contiguous()
        assert batch[group]["grid_coord"].dtype == torch.int64
        assert batch[group]["offset"].dtype == torch.int64
        assert batch[group]["grid_size"] == 0.1
        assert "strength" not in batch[group]
        assert "patch_id" not in batch[group]
    torch.testing.assert_close(batch["global"]["feat"], batch["masked_global"]["feat"])
    assert batch["masked_global"]["mask"].dtype == torch.bool
    assert "mask" not in batch["global"]
    # Tensor storage belongs to concatenated batches, not worker sample arrays.
    batch["global"]["feat"][0, 0] = -999
    assert first["global_views"][0]["feat"][0, 0] == 0


def test_collation_supports_no_local_views_and_rejects_mixed_steps():
    sample = make_sample(0)
    sample["local_views"] = []
    assert collate_ssl_views([sample])["local"] is None
    with pytest.raises(ValueError, match="same training step"):
        collate_ssl_views([make_sample(0, step=1), make_sample(1, step=2)])


@pytest.mark.parametrize("drop_last, batches_per_epoch", [(False, 3), (True, 2)])
def test_step_sampler_covers_epochs_and_resumes_exactly(drop_last, batches_per_epoch):
    kwargs = dict(dataset_size=5, batch_size=2, total_steps=11, seed=19, drop_last=drop_last)
    complete = list(StepBatchSampler(**kwargs))
    sampler = StepBatchSampler(**kwargs, start_step=4)
    assert len(sampler) == 7
    assert list(sampler) == complete[4:]
    assert len(complete) == 11
    for step, batch in enumerate(complete):
        assert all(request[1] == step for request in batch)
        assert len(batch) == (1 if not drop_last and step % batches_per_epoch == 2 else 2)
    first_epoch = [index for batch in complete[:batches_per_epoch] for index, _ in batch]
    assert len(first_epoch) == len(set(first_epoch))
    if not drop_last:
        assert sorted(first_epoch) == list(range(5))


def test_loader_prefetch_and_worker_count_preserve_planned_batches(keyframes):
    dataset = NuScenesSSLDataset(
        keyframes,
        augmentation=augmentation(mask_schedule=MaskSchedule(total_steps=8)),
        seed=45,
    )

    def batches(workers, start_step=0):
        loader = DataLoader(
            dataset,
            batch_sampler=StepBatchSampler(3, 2, 4, seed=7, start_step=start_step),
            collate_fn=collate_ssl_views,
            worker_init_fn=seed_worker,
            num_workers=workers,
            generator=torch.Generator().manual_seed(5),
        )
        return list(loader)

    reference, prefetched, resumed = batches(0), batches(2), batches(2, start_step=2)
    for expected, actual in zip(reference, prefetched):
        assert expected["step"] == actual["step"]
        assert expected["meta"] == actual["meta"]
        for group in ("global", "masked_global", "local"):
            for key in expected[group]:
                if isinstance(expected[group][key], torch.Tensor):
                    torch.testing.assert_close(expected[group][key], actual[group][key], rtol=0, atol=0)
    for expected, actual in zip(reference[2:], resumed):
        assert expected["step"] == actual["step"]
        torch.testing.assert_close(
            expected["masked_global"]["mask"], actual["masked_global"]["mask"]
        )
