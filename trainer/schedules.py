"""Scalar schedules adapted from Pointcept's CosineScheduler (96e109d).

https://github.com/Pointcept/Pointcept/blob/96e109d103418a5daaa375c50e1665b9f04f4a4b/pointcept/utils/scheduler.py
Warmup includes both endpoints; cosine uses the original arange / length rule.
"""

import numpy as np


class ScalarSchedule:
    def __init__(self, total_steps, base_value, final_value=None,
                 start_value=None, warmup_ratio=0.0):
        if total_steps <= 0 or not 0 <= warmup_ratio <= 1:
            raise ValueError("total_steps must be positive and warmup_ratio in [0, 1]")
        self.total_steps = total_steps
        self.final_value = base_value if final_value is None else final_value
        start_value = base_value if start_value is None else start_value
        warmup = int(total_steps * warmup_ratio)
        ramp = np.linspace(start_value, base_value, warmup)
        remaining = total_steps - warmup
        tail = self.final_value + 0.5 * (base_value - self.final_value) * (
            1 + np.cos(np.pi * np.arange(remaining) / max(remaining, 1))
        )
        self.values = np.concatenate((ramp, tail))

    def __call__(self, step):
        return float(self.values[step]) if step < self.total_steps else self.final_value
