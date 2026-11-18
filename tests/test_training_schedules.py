"""Check the scalar schedule conventions reused from official Pointcept."""

import numpy as np
import pytest

from trainer.schedules import ScalarSchedule


@pytest.mark.parametrize("steps,warmup", [(1000, .05), (20, .05), (19, .05), (8, 1)])
def test_teacher_temperature_is_linear_then_constant(steps, warmup):
    schedule = ScalarSchedule(steps, .07, start_value=.04, warmup_ratio=warmup)
    width = int(steps * warmup)
    expected = np.r_[np.linspace(.04, .07, width), np.full(steps - width, .07)]
    np.testing.assert_array_equal([schedule(i) for i in range(steps)], expected)


def test_momentum_uses_original_cosine_indexing():
    schedule = ScalarSchedule(1000, .994, final_value=1)
    assert schedule(0) == .994
    assert schedule(500) == 1 + .5 * (.994 - 1) * (1 + np.cos(np.pi / 2))
    assert schedule(999) < 1
    assert schedule(1000) == 1
