"""Geometric, correspondence, and sparse-grid contracts of the SSL views."""

import copy
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from augmentation import (
    Compose,
    GridSample,
    MultiViewGenerator,
    NuScenesSSLViews,
    RandomFlip,
    RandomJitter,
    RandomRotate,
    RandomScale,
)


def make_item(size=100):
    rng = np.random.default_rng(14)
    coord = (rng.normal(size=(size, 3)) * [5, 3, 1] + [10, 0, 2]).astype(np.float32)
    strength = np.arange(size, dtype=np.float32).reshape(-1, 1) / max(size, 1)
    return {
        "point": {
            "coord": coord,
            "origin_coord": coord.copy(),
            "strength": strength,
            # Ensure the pipeline rebuilds these stale coordinate channels.
            "feat": np.full((size, 4), -1234, dtype=np.float32),
            "offset": np.array([size], dtype=np.int64),
            "point_id": np.arange(size, dtype=np.int64),
        },
        "meta": {"sample_token": "example", "nested": {"version": "mini"}},
    }


def test_grid_sample_preserves_actual_representatives_and_correspondences():
    coord = np.array(
        [[-0.9, 0, 0], [-0.1, 0, 0], [0.1, 0, 0], [0.9, 0, 0], [1.1, 0, 0]],
        dtype=np.float32,
    )
    original = {
        "coord": coord,
        "origin_coord": coord + np.float32(100),
        "strength": np.arange(5, dtype=np.float32).reshape(-1, 1),
        "feat": np.arange(20, dtype=np.float32).reshape(5, 4),
        "point_id": np.arange(5, dtype=np.int64),
        "mask": np.array([False, True, True, False, True]),
        "offset": np.array([5], dtype=np.int64),
    }
    sampled = GridSample(1, return_inverse=True)(copy.deepcopy(original))
    assert len(sampled["coord"]) == 3
    assert sampled["grid_coord"].dtype == np.int64
    np.testing.assert_array_equal(
        sampled["grid_coord"], [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
    )
    for name in ("coord", "origin_coord", "strength", "feat", "point_id", "mask"):
        np.testing.assert_array_equal(sampled[name], original[name][sampled["index"]])
    np.testing.assert_array_equal(sampled["offset"], [3])
    raw_grid = np.floor(coord.astype(np.float64)).astype(np.int64)
    raw_grid -= raw_grid.min(axis=0)
    np.testing.assert_array_equal(sampled["grid_coord"][sampled["inverse"]], raw_grid)


def test_sampler_keeps_the_original_grid_phase():
    point = {"coord": np.array([[0.9, 0, 0], [1.1, 0, 0]], dtype=np.float32)}
    sampled = GridSample(1)(point)
    assert len(sampled["coord"]) == 2
    np.testing.assert_array_equal(sampled["grid_coord"][:, 0], [0, 1])


def test_sampler_gives_each_member_of_a_voxel_a_chance():
    # Unequal occupancies expose the upstream modulo-selection bias.
    coord = np.zeros((11, 3), dtype=np.float32)
    coord[:3, 0] = [0.1, 0.2, 0.3]
    coord[3:, 0] = np.linspace(1.1, 1.8, 8)
    np.random.seed(71)
    selected = np.zeros(3, dtype=np.int64)
    sampler = GridSample(1)
    for _ in range(2000):
        index = sampler({"coord": coord.copy()})["index"][0]
        selected[index] += 1
    assert np.all(np.abs(selected / selected.sum() - 1 / 3) < 0.04)


@pytest.mark.parametrize("grid_size", [0, -1, np.nan, np.inf])
def test_sampler_rejects_invalid_grid_sizes(grid_size):
    with pytest.raises(ValueError):
        GridSample(grid_size)


@pytest.mark.parametrize(
    "coord",
    [
        np.empty((0, 3), dtype=np.float32),
        np.zeros((3, 2), dtype=np.float32),
        np.array([[np.nan, 0, 0]], dtype=np.float32),
    ],
)
def test_sampler_rejects_invalid_coordinates(coord):
    with pytest.raises(ValueError):
        GridSample()({"coord": coord})


def test_geometric_transforms_leave_the_reference_frame_and_intensity_unchanged():
    point = make_item(12)["point"]
    original = copy.deepcopy(point)
    transformed = Compose(
        [
            RandomRotate(angle=(0.5, 0.5), center=(0, 0, 0), p=1),
            RandomScale(scale=(2, 2)),
            RandomFlip(p=1),
            RandomJitter(sigma=0, clip=0.02),
        ]
    )(point)
    expected = original["coord"][:, [1, 0, 2]] * [-1, 1, 1]
    expected *= [-2, -2, 2]
    np.testing.assert_allclose(transformed["coord"], expected, atol=1e-6)
    np.testing.assert_array_equal(transformed["origin_coord"], original["origin_coord"])
    np.testing.assert_array_equal(transformed["strength"], original["strength"])


def test_multiview_crops_use_point_count_fractions_and_keep_tiny_clouds_nonempty():
    point = make_item(20)["point"]
    point["coord"] = np.column_stack((np.arange(20), np.zeros((20, 2)))).astype(
        np.float32
    )
    point["origin_coord"] = point["coord"].copy()
    generator = MultiViewGenerator(
        global_view_scale=(0.5, 0.5), local_view_scale=(0.25, 0.25)
    )
    views = generator(point)
    assert [len(v["coord"]) for v in views["global_views"]] == [10, 10]
    assert [len(v["coord"]) for v in views["local_views"]] == [5, 5, 5, 5]
    for view in views["global_views"] + views["local_views"]:
        # Nearest-point crops on this one-dimensional cloud form contiguous intervals.
        np.testing.assert_array_equal(np.diff(np.sort(view["point_id"])), 1)
    tiny = generator(make_item(1)["point"])
    assert all(len(v["coord"]) == 1 for group in tiny.values() for v in group)


def test_shared_transform_cannot_silently_change_crop_geometry():
    def translate(point):
        point["coord"] += 1
        return point

    generator = MultiViewGenerator(global_shared_transform=[translate])
    with pytest.raises(ValueError):
        generator(make_item()["point"])


def test_ssl_views_are_aligned_independent_and_ready_for_tensor_conversion():
    item = make_item()
    original = copy.deepcopy(item)
    result = NuScenesSSLViews()(item)
    assert len(result["global_views"]) == 2
    assert len(result["local_views"]) == 4
    assert result["meta"] == original["meta"]
    views = result["global_views"] + result["local_views"]
    for view in views:
        size = len(view["coord"])
        assert size > 0
        for name in ("coord", "origin_coord", "strength", "feat"):
            assert view[name].dtype == np.float32
            assert view[name].flags.c_contiguous
            assert np.isfinite(view[name]).all()
        assert view["feat"].shape == (size, 4)
        np.testing.assert_array_equal(view["feat"][:, :3], view["coord"])
        np.testing.assert_array_equal(view["feat"][:, 3:], view["strength"])
        np.testing.assert_array_equal(
            view["origin_coord"], original["point"]["origin_coord"][view["point_id"]]
        )
        np.testing.assert_array_equal(
            view["strength"], original["point"]["strength"][view["point_id"]]
        )
        assert not np.shares_memory(view["coord"], view["origin_coord"])
        assert view["grid_size"] == 0.1
        assert view["grid_coord"].dtype == np.int64
        assert np.all(view["grid_coord"] >= 0)
        assert len(np.unique(view["grid_coord"], axis=0)) == size
        grid = np.floor(view["coord"].astype(np.float64) / view["grid_size"]).astype(
            np.int64
        )
        grid -= grid.min(axis=0)
        np.testing.assert_array_equal(view["grid_coord"], grid)
        assert view["offset"].dtype == np.int64
        np.testing.assert_array_equal(view["offset"], [size])
    for i, left in enumerate(views):
        for right in views[i + 1 :]:
            for name in ("coord", "origin_coord", "strength", "feat", "grid_coord"):
                assert not np.shares_memory(left[name], right[name])
    for name, value in item["point"].items():
        np.testing.assert_array_equal(value, original["point"][name])
    result["meta"]["nested"]["version"] = "changed"
    assert item["meta"] == original["meta"]


def test_pipeline_reproducible_with_both_upstream_random_seeds():
    pipeline = NuScenesSSLViews()
    item = make_item()
    random.seed(61)
    np.random.seed(61)
    first = pipeline(item)
    random.seed(61)
    np.random.seed(61)
    second = pipeline(item)
    for group in ("global_views", "local_views"):
        for left, right in zip(first[group], second[group]):
            for name in ("coord", "origin_coord", "strength", "grid_coord", "feat"):
                np.testing.assert_array_equal(left[name], right[name])


def test_pipeline_can_disable_geometry_and_support_a_single_raw_point():
    item = make_item(1)
    result = NuScenesSSLViews(global_transform=(), local_transform=())(item)
    for group in ("global_views", "local_views"):
        for view in result[group]:
            np.testing.assert_array_equal(view["coord"], item["point"]["coord"])
            np.testing.assert_array_equal(
                view["origin_coord"], item["point"]["origin_coord"]
            )
            np.testing.assert_array_equal(view["grid_coord"], [[0, 0, 0]])


def test_final_voxel_selection_happens_after_geometric_augmentation():
    item = make_item(2)
    item["point"]["coord"] = np.array([[0.9, 0, 0], [1.1, 0, 0]], dtype=np.float32)
    item["point"]["origin_coord"] = item["point"]["coord"].copy()
    pipeline = NuScenesSSLViews(
        grid_size=1,
        global_view_scale=(1, 1),
        local_view_scale=(1, 1),
        global_transform=[RandomScale(scale=(0.1, 0.1))],
        local_transform=[RandomScale(scale=(0.1, 0.1))],
    )
    result = pipeline(item)
    for group in ("global_views", "local_views"):
        for view in result[group]:
            # Distinct original cells become one cell after compression.
            assert len(view["coord"]) == 1
            np.testing.assert_array_equal(view["grid_coord"], [[0, 0, 0]])
            np.testing.assert_allclose(view["coord"], view["origin_coord"] * 0.1)


@pytest.mark.parametrize(
    ("options", "expected_transform_input_count"),
    [({}, 3), ({"presample": False}, 12)],
)
def test_shared_presampling_reduces_dense_crop_inputs_and_can_be_disabled(
    options, expected_transform_input_count
):
    item = make_item(12)
    coord = np.zeros((12, 3), dtype=np.float32)
    coord[:, 0] = np.r_[np.linspace(0.001, 0.045, 10), 1.01, 2.01]
    item["point"]["coord"] = coord
    item["point"]["origin_coord"] = coord.copy()
    counts_seen_by_geometry = []

    def capture_crop(point):
        counts_seen_by_geometry.append(len(point["coord"]))
        return point

    pipeline = NuScenesSSLViews(
        grid_size=0.1,
        global_view_num=1,
        local_view_num=1,
        global_view_scale=(1, 1),
        local_view_scale=(1, 1),
        global_transform=(capture_crop,),
        local_transform=(capture_crop,),
        **options,
    )
    result = pipeline(item)
    assert counts_seen_by_geometry == [expected_transform_input_count] * 2
    for group in ("global_views", "local_views"):
        # Final sampling still gives unique sites when presampling is disabled.
        assert len(result[group][0]["coord"]) == 3


@pytest.mark.parametrize(
    "kwargs",
    [
        {"grid_size": 0},
        {"max_size": 0},
        {"min_points": 5, "max_size": 4},
    ],
)
def test_pipeline_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        NuScenesSSLViews(**kwargs)


@pytest.mark.parametrize("name", ["coord", "origin_coord", "strength"])
def test_pipeline_rejects_misaligned_and_nonfinite_arrays(name):
    item = make_item()
    item["point"][name] = item["point"][name][:-1]
    with pytest.raises(ValueError):
        NuScenesSSLViews()(item)
    item = make_item()
    item["point"][name][0, 0] = np.inf
    with pytest.raises(ValueError):
        NuScenesSSLViews()(item)


def test_package_import_does_not_import_torch_or_cuda_libraries():
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import augmentation; "
            "assert 'torch' not in sys.modules; assert 'spconv' not in sys.modules",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
