"""
Occupied-patch masks, adapted to NumPy from Pointcept implementation.
Author: Andrei-Iosif Ioja (iojaandrei11@gmail.com)

Source: https://github.com/Pointcept/Pointcept/blob/main/pointcept/models/sonata/sonata_v1m1_base.py#L293-L310
"""

import copy
from typing import Tuple

import numpy as np


class MaskedViewsGenerator:
    """Copy global views and flag all rows in randomly selected occupied cubes.

    Cubes have width mask_size in meters and an origin at each view's minimum augmented
    coordinate. Mask floor(mask_ratio * Q) of the Q occupied patches across the supplied views,
    rather than a fixed fraction of points per view. Features are not changed, instead
    the backbone (e.g. LitePT) replaces the corresponding masked preembeddings with its learned
    token before the first encoder block.

    Optional mask_schedule supplies the width and ratio at an explicit training
    step, overriding the fixed values.
    """

    def __init__(self, mask_size=0.4, mask_ratio=0.3, *, mask_schedule=None):
        if not np.isfinite(mask_size) or mask_size <= 0:
            raise ValueError("mask_size must be finite and positive")
        if not np.isfinite(mask_ratio) or not 0 <= mask_ratio <= 1:
            raise ValueError("mask_ratio must be between 0 and 1")
        self.mask_size = mask_size
        self.mask_ratio = mask_ratio
        self.mask_schedule = mask_schedule

    def _parameters(self, step):
        if self.mask_schedule is not None:
            if step is None:
                raise ValueError("mask_schedule requires an explicit training step")
            return self.mask_schedule(step)
        if step is not None:
            raise ValueError("step requires a mask_schedule")
        return self.mask_size, self.mask_ratio

    def generate_mask(
            self,
            coord: np.ndarray,
            offset: np.ndarray,
            *,
            mask_size: float,
            mask_ratio: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return boolean flags and patch IDs for packed, nonempty NumPy views.

        offset contains cumulative point counts, as in the model's Point dict.
        View IDs distinguish patches at identical coordinates in different views.
        This helper can also receive all globals of a future collated batch.
        """
        starts = np.r_[0, offset[:-1]] # A simple way to build a new numpy array
        batch = np.repeat(np.arange(len(offset)), offset - starts)
        min_coord = np.minimum.reduceat(coord, starts, axis=0)
        grid_coord = np.floor((coord - min_coord[batch]) / mask_size).astype(np.int64)
        patch_keys = np.column_stack((batch, grid_coord))
        unique, patch_id = np.unique(patch_keys, axis=0, return_inverse=True)
        patch_num = len(unique)
        selected = np.random.permutation(patch_num)[: int(patch_num * mask_ratio)]
        mask = np.isin(patch_id, selected)
        return mask, patch_id.astype(np.int64, copy=False)

    def __call__(self, global_views, *, step=None):
        mask_size, mask_ratio = self._parameters(step)
        if not global_views:
            return []
        offset = np.cumsum([len(view["coord"]) for view in global_views], dtype=np.int64)
        coord = np.concatenate([view["coord"] for view in global_views])
        mask, patch_id = self.generate_mask(coord, offset, mask_size=mask_size, mask_ratio=mask_ratio)
        masked_views = []
        start = 0
        for view, stop in zip(global_views, offset):
            # We generate a copy of our global view, and select the corresponding mask (along with patch ids for tests)
            masked = copy.deepcopy(view)
            masked["mask"] = mask[start:stop].copy()
            masked["patch_id"] = patch_id[start:stop].copy()
            # We also keep the metadata (for debug purposes).
            masked["mask_size"] = mask_size
            masked["mask_ratio"] = mask_ratio
            if self.mask_schedule is not None:
                masked["mask_step"] = int(step)
                masked["mask_total_steps"] = self.mask_schedule.total_steps
            masked_views.append(masked)
            start = stop
        return masked_views
