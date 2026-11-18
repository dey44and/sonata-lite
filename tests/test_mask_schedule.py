"""Progressive masking follows Sonata's inclusive linear warmup exactly."""

import pickle

import numpy as np
import pytest

from augmentation.schedule import MaskSchedule


@pytest.mark.parametrize(
    "total_steps,warmup_ratio",
    [(2000, 0.05), (7, 0.5), (8, 1), (10, 0), (1, 0.05), (20, 0.05)],
)
def test_schedule_matches_upstream_warmup_and_constant_tail(total_steps, warmup_ratio):
    schedule = MaskSchedule(total_steps, warmup_ratio=warmup_ratio)
    warmup_steps = int(total_steps * warmup_ratio)
    expected = np.column_stack(
        [
            np.r_[
                np.linspace(0.1, 0.4, warmup_steps),
                np.full(total_steps - warmup_steps, 0.4),
            ],
            np.r_[
                np.linspace(0.3, 0.7, warmup_steps),
                np.full(total_steps - warmup_steps, 0.7),
            ],
        ]
    )
    actual = np.array([schedule(step) for step in range(total_steps)])
    np.testing.assert_array_equal(actual, expected)
    assert schedule(total_steps) == schedule(total_steps + 100) == (0.4, 0.7)


def test_warmup_boundaries_include_both_endpoints():
    schedule = MaskSchedule(total_steps=2000)
    assert schedule.warmup_steps == 100
    assert schedule(0) == (0.1, 0.3)
    assert schedule(99) == schedule(100) == (0.4, 0.7)
    size, ratio = schedule(50)
    assert size == np.linspace(0.1, 0.4, 100)[50]
    assert ratio == np.linspace(0.3, 0.7, 100)[50]
    assert type(size) is float
    assert type(ratio) is float


def test_zero_and_one_step_warmup_keep_numpy_linspace_semantics():
    no_warmup = MaskSchedule(total_steps=19)
    one_step = MaskSchedule(total_steps=20)
    assert no_warmup.warmup_steps == 0
    assert no_warmup(0) == (0.4, 0.7)
    assert one_step.warmup_steps == 1
    assert one_step(0) == (0.1, 0.3)
    assert one_step(1) == (0.4, 0.7)


def test_upstream_rounding_preserves_patch_count_at_integer_boundary():
    # Scalar interpolation with a different operation order gives a ratio just
    # below 7/18 here, changing floor(ratio * Q) from seven masked patches to six.
    schedule = MaskSchedule(total_steps=2000)
    _, ratio = schedule(22)
    assert ratio == np.linspace(0.3, 0.7, 100)[22]
    assert int(ratio * 18) == 7


def test_explicit_step_supports_resuming_reordered_calls_and_pickling():
    schedule = MaskSchedule(total_steps=np.int64(1000))
    restored = pickle.loads(pickle.dumps(schedule))
    for step in [999, 7, 0, 50, 49, 7, 1000]:
        assert restored(np.int64(step)) == schedule(step)
    assert restored(0) == (0.1, 0.3)


def test_configurable_endpoints_and_warmup_fraction():
    schedule = MaskSchedule(
        total_steps=40,
        mask_size_start=0.2,
        mask_size_base=0.8,
        mask_ratio_start=0,
        mask_ratio_base=1,
        warmup_ratio=0.25,
    )
    assert schedule.warmup_steps == 10
    assert schedule(0) == (0.2, 0)
    assert schedule(9) == schedule(40) == (0.8, 1)


@pytest.mark.parametrize("total_steps", [0, -1, 1.5, True])
def test_total_steps_must_be_a_positive_integer(total_steps):
    with pytest.raises(ValueError, match="total_steps"):
        MaskSchedule(total_steps)


@pytest.mark.parametrize("step", [-1, 0.5, True])
def test_step_must_be_a_nonnegative_integer(step):
    with pytest.raises(ValueError, match="step"):
        MaskSchedule(100)(step)


@pytest.mark.parametrize(
    "name,value",
    [
        ("mask_size_start", 0),
        ("mask_size_base", -0.1),
        ("mask_size_start", np.nan),
        ("mask_size_base", np.inf),
        ("mask_ratio_start", -0.1),
        ("mask_ratio_base", 1.1),
        ("mask_ratio_start", np.nan),
        ("mask_ratio_base", np.inf),
        ("warmup_ratio", -0.1),
        ("warmup_ratio", 1.1),
        ("warmup_ratio", np.nan),
    ],
)
def test_invalid_mask_ranges_are_rejected_at_construction(name, value):
    with pytest.raises(ValueError, match=name):
        MaskSchedule(100, **{name: value})
