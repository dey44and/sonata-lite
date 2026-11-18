"""Spatial patch masks and independent teacher/student view contracts."""

import copy
import pickle
import random

import numpy as np
import pytest

from augmentation import (
    MaskedViewsGenerator,
    NuScenesSSLViews,
    MaskSchedule,
    RandomScale,
)


def make_view():
    coord = np.array(
        [[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0], [1.1, 0, 0], [2.1, 0, 0], [2.2, 0, 0]],
        dtype=np.float32,
    )
    strength = np.arange(len(coord), dtype=np.float32).reshape(-1, 1) / 10
    return {
        "coord": coord,
        "origin_coord": coord.copy(),
        "strength": strength,
        "feat": np.concatenate((coord, strength), axis=1),
        "grid_coord": np.arange(18, dtype=np.int64).reshape(6, 3),
        "grid_size": 0.1,
        "offset": np.array([6], dtype=np.int64),
    }


def test_masks_cover_whole_occupied_patches_and_keep_views_separate():
    # The same three occupied cubes in two views must yield six independent patches.
    np.random.seed(19)
    masked_views = MaskedViewsGenerator(mask_size=0.4, mask_ratio=0.5)(
        [make_view(), make_view()]
    )
    selected_patch_ids = set()
    all_patch_ids = []
    for masked in masked_views:
        ids = masked["patch_id"]
        mask = masked["mask"]
        assert mask.dtype == np.bool_
        assert ids.dtype == np.int64
        assert mask.shape == ids.shape == (6,)
        assert len(np.unique(ids)) == 3
        assert len(np.unique(ids[:3])) == 1
        assert ids[4] == ids[5]
        for patch_id in np.unique(ids):
            assert len(np.unique(mask[ids == patch_id])) == 1
        all_patch_ids.append(set(ids))
        selected_patch_ids.update(ids[mask])
    assert all_patch_ids[0].isdisjoint(all_patch_ids[1])
    assert len(selected_patch_ids) == 3  # floor(0.5 * 6), counted across both views.


def test_masked_copies_keep_geometry_features_and_teacher_storage_independent():
    teacher = make_view()
    before = copy.deepcopy(teacher)
    students = MaskedViewsGenerator()([teacher, teacher])
    assert "mask" not in teacher
    for student in students:
        for key, original in before.items():
            if isinstance(original, np.ndarray):
                np.testing.assert_array_equal(student[key], original)
                assert not np.shares_memory(student[key], teacher[key])
            else:
                assert student[key] == original
        assert student["mask_size"] == 0.4
        assert student["mask_ratio"] == 0.3
    students[0]["coord"][:] = -123
    students[0]["feat"][:] = -456
    np.testing.assert_array_equal(teacher["coord"], before["coord"])
    np.testing.assert_array_equal(students[1]["coord"], before["coord"])
    np.testing.assert_array_equal(students[1]["feat"], before["feat"])


@pytest.mark.parametrize("ratio", [0, 1])
def test_mask_ratio_endpoints(ratio):
    masked = MaskedViewsGenerator(mask_ratio=ratio)([make_view()])[0]
    assert np.all(masked["mask"] == bool(ratio))


def test_tiny_cloud_preserves_upstream_floor_without_forcing_one_mask():
    generator = MaskedViewsGenerator(mask_ratio=0.3)
    mask, patch_id = generator.generate_mask(
        np.array([[12, -8, 2]], dtype=np.float32), np.array([1], dtype=np.int64)
    )
    np.testing.assert_array_equal(mask, [False])
    np.testing.assert_array_equal(patch_id, [0])
    assert generator([]) == []


def test_patch_lattice_uses_each_views_minimum_coordinate():
    coord = np.array([[0, 0, 0], [0.125, 0, 0], [0.5, 0, 0]], dtype=np.float32)
    translated = coord + np.array([100, -20, 5], dtype=np.float32)
    generator = MaskedViewsGenerator(mask_size=0.25, mask_ratio=0.5)
    offset = np.array([3], dtype=np.int64)
    np.random.seed(42)
    mask, ids = generator.generate_mask(coord, offset)
    np.random.seed(42)
    other_mask, other_ids = generator.generate_mask(translated, offset)
    np.testing.assert_array_equal(mask, other_mask)
    np.testing.assert_array_equal(ids, other_ids)


def make_item():
    view = make_view()
    return {"point": view, "meta": {"sample_token": "example"}}


def test_pipeline_masks_the_final_augmented_geometry_and_preserves_references():
    item = make_item()
    coord = np.zeros((4, 3), dtype=np.float32)
    coord[:, 0] = [0, 0.21, 0.42, 0.63]
    item["point"] = {
        "coord": coord,
        "origin_coord": coord.copy(),
        "strength": np.zeros((4, 1), dtype=np.float32),
    }
    result = NuScenesSSLViews(
        presample=False,
        global_view_num=1,
        local_view_num=0,
        global_view_scale=(1, 1),
        global_transform=(RandomScale(scale=(2, 2)),),
        mask_size=0.4,
        mask_ratio=0.5,
    )(item)
    teacher = result["global_views"][0]
    student = result["masked_global_views"][0]
    assert len(student["coord"]) == 4
    # Scaling doubles separations: four final occupied patches, versus two originally.
    assert len(np.unique(student["patch_id"])) == 4
    assert student["mask"].sum() == 2
    np.testing.assert_allclose(student["coord"], student["origin_coord"] * 2)
    for key in teacher:
        np.testing.assert_array_equal(student[key], teacher[key])
    assert result["local_views"] == []
    assert result["meta"] == item["meta"]


def test_pipeline_masks_are_repeatable_and_do_not_change_existing_view_sampling():
    item = make_item()

    def seeded_views(ratio):
        random.seed(81)
        np.random.seed(81)
        return NuScenesSSLViews(mask_ratio=ratio)(item)

    first = seeded_views(0.3)
    second = seeded_views(0.3)
    no_masks = seeded_views(0)
    assert len(first["masked_global_views"]) == 2
    assert len(first["local_views"]) == 4
    for group in ("global_views", "masked_global_views", "local_views"):
        for left, right in zip(first[group], second[group]):
            for key in left:
                np.testing.assert_array_equal(left[key], right[key])
    for group in ("global_views", "local_views"):
        for left, right in zip(first[group], no_masks[group]):
            for key in left:
                np.testing.assert_array_equal(left[key], right[key])
                assert "mask" not in left


@pytest.mark.parametrize("mask_size", [0, -0.1, np.nan, np.inf])
def test_mask_size_must_define_a_finite_positive_cube(mask_size):
    with pytest.raises(ValueError):
        MaskedViewsGenerator(mask_size=mask_size)


@pytest.mark.parametrize("mask_ratio", [-0.1, 1.1, np.nan, np.inf])
def test_mask_ratio_must_define_a_valid_fraction(mask_ratio):
    with pytest.raises(ValueError):
        MaskedViewsGenerator(mask_ratio=mask_ratio)


def test_progressive_patch_width_changes_groups_and_records_effective_values():
    schedule = MaskSchedule(total_steps=2000)
    generator = MaskedViewsGenerator(mask_schedule=schedule)
    initial = generator([make_view()], step=0)[0]
    final = generator([make_view()], step=99)[0]
    assert len(np.unique(initial["patch_id"])) == 6
    assert len(np.unique(final["patch_id"])) == 3
    assert len(np.unique(initial["patch_id"][initial["mask"]])) == 1
    assert len(np.unique(final["patch_id"][final["mask"]])) == 2
    for view, step, expected in ((initial, 0, (0.1, 0.3)), (final, 99, (0.4, 0.7))):
        assert (view["mask_size"], view["mask_ratio"]) == expected
        assert view["mask_step"] == step
        assert view["mask_total_steps"] == 2000
        for key, value in make_view().items():
            np.testing.assert_array_equal(view[key], value)
    # Scheduled calls must not update fixed configuration or a hidden step counter.
    assert (generator.mask_size, generator.mask_ratio) == (0.4, 0.3)
    assert schedule(0) == (0.1, 0.3)


def test_progressive_ratio_reaches_actual_patch_selection_at_rounding_boundary():
    coord = np.zeros((18, 3), dtype=np.float32)
    coord[:, 0] = np.arange(18)
    generator = MaskedViewsGenerator(mask_schedule=MaskSchedule(2000))
    # All 18 sites stay in separate cubes throughout the size schedule.
    for step, expected_count in ((0, 5), (22, 7), (99, 12), (2000, 12)):
        mask, ids = generator.generate_mask(coord, np.array([18]), step=step)
        assert len(np.unique(ids)) == 18
        assert mask.sum() == expected_count


def test_explicit_step_is_required_only_for_scheduled_masking():
    scheduled = MaskedViewsGenerator(mask_schedule=MaskSchedule(1000))
    with pytest.raises(ValueError, match="explicit training step"):
        scheduled([make_view()])
    with pytest.raises(ValueError, match="step requires a mask_schedule"):
        MaskedViewsGenerator()([make_view()], step=0)
    pipeline = NuScenesSSLViews(mask_schedule=MaskSchedule(1000))
    with pytest.raises(ValueError, match="explicit training step"):
        pipeline(make_item())


def test_scheduled_pipeline_preserves_crops_and_supports_copied_worker_instances():
    pipeline = NuScenesSSLViews(mask_schedule=MaskSchedule(1000))
    worker_copy = pickle.loads(pickle.dumps(pipeline))

    def seeded_views(generator, step):
        random.seed(16)
        np.random.seed(16)
        return generator(make_item(), step=step)

    initial = seeded_views(pipeline, 0)
    final = seeded_views(pipeline, 49)
    resumed = seeded_views(worker_copy, 49)
    repeated = seeded_views(pipeline, 0)
    for group in ("global_views", "local_views"):
        for left, right in zip(initial[group], final[group]):
            for key in left:
                np.testing.assert_array_equal(left[key], right[key])
    for left, right in zip(
        final["masked_global_views"], resumed["masked_global_views"]
    ):
        for key in left:
            np.testing.assert_array_equal(left[key], right[key])
    for left, right in zip(
        initial["masked_global_views"], repeated["masked_global_views"]
    ):
        assert left["mask_size"] == 0.1
        assert left["mask_ratio"] == 0.3
        for key in left:
            np.testing.assert_array_equal(left[key], right[key])
    for view in final["masked_global_views"]:
        assert view["mask_size"] == 0.4
        assert view["mask_ratio"] == 0.7
