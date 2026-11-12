"""
Schedule for mask size and mask ratio for masked global views.
Author: Andrei-Iosif Ioja (iojaandrei11@gmail.com)

Adapted from:
https://github.com/Pointcept/Pointcept/blob/main/pointcept/utils/scheduler.py#L155-L189
https://github.com/Pointcept/Pointcept/blob/main/pointcept/models/sonata/sonata_v1m1_base.py#L191-L209
"""

import numpy as np


class MaskSchedule:
    """
    Return (patch width in meters, occupied-patch ratio) for a training step.

    The call method use an explicit zero-based step without advancing internal state. 
    The caller can therefore resume at a saved step or evaluate steps in any order.
    """

    def __init__(
        self,
        total_steps: int,
        mask_size_start: float = 0.1,
        mask_size_base: float = 0.4,
        mask_ratio_start: float = 0.3,
        mask_ratio_base: float = 0.7,
        warmup_ratio: float = 0.05,
    ):
        if total_steps <= 0:
            raise ValueError("total_steps must be a positive integer")
        for value in (mask_size_start, mask_size_base):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("Mask size must be finite and positive")
        for value in (mask_ratio_start, mask_ratio_base):
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Mask ratio must be between 0 and 1")
        if not np.isfinite(warmup_ratio) or not 0 <= warmup_ratio <= 1:
            raise ValueError("Warmup ratio must be between 0 and 1")

        self.total_steps = total_steps
        self.mask_size_base = mask_size_base
        self.mask_ratio_base = mask_ratio_base
        self.warmup_steps = int(self.total_steps * warmup_ratio)
        self._mask_size_warmup = np.linspace(
            mask_size_start, self.mask_size_base, self.warmup_steps
        )
        self._mask_ratio_warmup = np.linspace(
            mask_ratio_start, self.mask_ratio_base, self.warmup_steps
        )

    def __call__(self, step):
        """
        Returns a pair consisting of mask size and mask ratio.

        :param step: The current training step.
        """
        if step < 0 or step >= self.total_steps:
            raise ValueError(f"step must be a nonnegative integer between [0, {self.total_steps})")
        if step < self.warmup_steps:
            return self._mask_size_warmup[step], self._mask_ratio_warmup[step]
        return self.mask_size_base, self.mask_ratio_base
