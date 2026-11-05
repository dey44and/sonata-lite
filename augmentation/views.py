"""
Sonata crop generation for our LitePT SSL training.
Author: Andrei-Iosif Ioja (iojaandrei11@gmail.com)

Adapted from:
    Source: https://github.com/facebookresearch/sonata/blob/main/sonata/transform.py
    Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com).
"""

import copy
from collections.abc import Mapping

import numpy as np

from .transforms import (
    Compose,
    GridSample,
    RandomFlip,
    RandomJitter,
    RandomRotate,
    RandomScale,
)


def _validate_arrays(point, keys):
    if not isinstance(point, Mapping):
        raise TypeError("point must be a mapping of NumPy arrays")
    coord = point["coord"]
    if not isinstance(coord, np.ndarray) or not np.issubdtype(coord.dtype, np.floating):
        raise TypeError("coord must be a floating-point NumPy array")
    if coord.ndim != 2 or coord.shape[1] != 3 or len(coord) == 0:
        raise ValueError("coord must have nonempty shape (N, 3)")
    if not np.isfinite(coord).all():
        raise ValueError("coord must contain finite values")
    for key in keys:
        if key not in point:
            continue
        array = point[key]
        if not isinstance(array, np.ndarray) or array.ndim == 0 or len(array) != len(coord):
            raise ValueError(f"{key} must be a point-aligned NumPy array")


class MultiViewGenerator:
    """Select principal/global/local crops for SSL training.

    Crop scales are fractions of min(N, max_size).
    View_transform runs after independent global/local transforms, for example
    to produce the final LitePT voxel representation.
    """

    def __init__(
        self,
        global_view_num=2,
        global_view_scale=(0.4, 1.0),
        local_view_num=4,
        local_view_scale=(0.1, 0.4),
        global_shared_transform=None,
        global_transform=None,
        local_transform=None,
        max_size=65536,
        center_height_scale=(0, 1),
        shared_global_view=False,
        view_keys=("coord", "origin_coord", "strength", "point_id"),
        *,
        min_points=1,
        view_transform=None,
    ):
        self.global_view_num = global_view_num
        self.local_view_num = local_view_num
        self.global_view_scale = global_view_scale
        self.local_view_scale = local_view_scale
        self.max_size = max_size
        self.min_points = min_points
        # Check if the number of points exceeds the maximum size
        if self.min_points > self.max_size:
            raise ValueError("min_points must not exceed max_size")
        self.center_height_scale = center_height_scale
        self.global_shared_transform = Compose(global_shared_transform)
        self.global_transform = Compose(global_transform)
        self.local_transform = Compose(local_transform)
        self.view_transform = Compose(view_transform)
        self.shared_global_view = shared_global_view
        self.view_keys = tuple(view_keys)
        # Check if coord key is missing
        if "coord" not in self.view_keys:
            raise ValueError("view_keys must include coord")

    def get_view(self, point, center, scale):
        coord = point["coord"]
        max_size = min(self.max_size, coord.shape[0])
        size = int(np.random.uniform(*scale) * max_size)
        # I suppose the logic here is that we need the maximum between the minimum number of points and the size,
        # but at the same time the size should be smaller than the max_size.
        size = min(max_size, max(self.min_points, size))
        # Keep only the first `size` points which are closest to the center
        index = np.argsort(np.sum(np.square(coord - center), axis=-1))[:size]
        view = dict(index=index)
        for key in point.keys():
            if key in self.view_keys:
                view[key] = point[key][index]
        if "index_valid_keys" in point.keys():
            view["index_valid_keys"] = list(point["index_valid_keys"])
        return view

    def __call__(self, data_dict):
        _validate_arrays(data_dict, self.view_keys)
        coord = data_dict["coord"]

        # First apply the global shared transform
        point = self.global_shared_transform(copy.deepcopy(data_dict))
        if not np.array_equal(point["coord"], coord):
            raise ValueError("global_shared_transform must not change coordinates")

        z_min = coord[:, 2].min()
        z_max = coord[:, 2].max()
        z_min_ = z_min + (z_max - z_min) * self.center_height_scale[0]
        z_max_ = z_min + (z_max - z_min) * self.center_height_scale[1]
        center_mask = np.logical_and(coord[:, 2] >= z_min_, coord[:, 2] <= z_max_)
        if not np.any(center_mask):
            raise ValueError("center_height_scale contains no crop-center points")

        # Sonata: centers for subsequent crops come from the principal crop (major view).
        major_center = coord[np.random.choice(np.where(center_mask)[0])]
        major_view = self.get_view(point, major_center, self.global_view_scale)
        major_coord = major_view["coord"]
        # Here we decide to generate separate global views or copies of the same global view.
        # We should also be aware that the centers of the other global views may lie outside the center mask.
        if not self.shared_global_view:
            global_views = [
                self.get_view(
                    point=point,
                    center=major_coord[np.random.randint(major_coord.shape[0])],
                    scale=self.global_view_scale,
                )
                for _ in range(self.global_view_num - 1)
            ]
        else:
            global_views = [
                {key: value.copy() for key, value in major_view.items()}
                for _ in range(self.global_view_num - 1)
            ]
        global_views = [major_view] + global_views

        # Sonata: favor principal-crop points not covered by previous locals.
        cover_mask = np.zeros_like(major_view["index"], dtype=bool)
        local_views = []
        for _ in range(self.local_view_num):
            # If all the points are already covered, we reset the cover mask
            if sum(~cover_mask) == 0:
                cover_mask[:] = False
            # We generate a local view where the center is not part of an already generated local view.
            local_view = self.get_view(
                point=data_dict,
                center=major_coord[np.random.choice(np.where(~cover_mask)[0])],
                scale=self.local_view_scale,
            )
            # Append the new local view to the list and update the cover mask by checking which elements from
            # major view were also selected for that local view
            local_views.append(local_view)
            cover_mask[np.isin(major_view["index"], local_view["index"])] = True

        # Finally, we form two groups and apply the corresponding global/local transforms
        result = {"global_views": [], "local_views": []}
        for name, views, transform in (
            ("global_views", global_views, self.global_transform),
            ("local_views", local_views, self.local_transform),
        ):
            for view in views:
                view.pop("index")
                result[name].append(self.view_transform(transform(view)))
        return result


class _PrepareLitePTView:
    """Finalize one augmented crop without importing the model or PyTorch."""

    def __init__(self, grid_size):
        self.sampler = GridSample(grid_size=grid_size)

    def __call__(self, point):
        # Upstream rotation produces float64; establish the model's float32
        # coordinate representation before quantizing, not after it.
        for key in ("coord", "origin_coord", "strength"):
            point[key] = np.ascontiguousarray(point[key], dtype=np.float32)
            if not np.isfinite(point[key]).all():
                raise ValueError(f"{key} cannot be represented as finite float32")
        point = self.sampler(point)
        point.pop("index", None)
        point.pop("index_valid_keys", None)
        point["grid_size"] = self.sampler.grid_size
        point["feat"] = np.concatenate((point["coord"], point["strength"]), axis=1)
        point["offset"] = np.array([len(point["coord"])], dtype=np.int64)
        return point


def _nuscenes_geometry():
    # Official LitePT nuScenes recipe; intensity has already been divided by
    # 255 in the reader, and is not modified here.
    return (
        RandomRotate(angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
        RandomScale(scale=[0.9, 1.1]),
        RandomFlip(p=0.5),
        RandomJitter(sigma=0.005, clip=0.02),
    )


class NuScenesSSLViews:
    """Turn one reader item into two global and four local LitePT view dicts.

    The default 0.10 m grid is our experimental baseline; official LitePT uses
    0.05 m. Sonata's shared sampling pass is retained before crop generation to
    reduce the influence of dense near-sensor returns on point-count crops.
    Set presample=False to disable it. Final sampling after each view's geometry
    is still required for unique sparse sites. Both passes use grid_size.
    origin_coord is only indexed, never spatially transformed. feat is rebuilt
    from current coord and strength.

    Returns CPU NumPy arrays and a deep copy of metadata. Global views are
    teacher inputs and templates for a future masked-student branch. Patch
    masking, mask-token insertion, tensor conversion and batching belong to
    later training stages. A coordinate-corrupted masked branch must undergo
    its own final voxelization before the convolutional stem.

    Set global_transform/local_transform to an empty tuple to disable geometry;
    None selects the default recipe. Seed both random and numpy.random for
    reproducibility, following the upstream transforms' random-number usage.
    """

    def __init__(
        self,
        grid_size=0.10,
        global_view_num=2,
        local_view_num=4,
        global_view_scale=(0.4, 1.0),
        local_view_scale=(0.1, 0.4),
        max_size=65536,
        min_points=1,
        *,
        presample=True,
        global_transform=None,
        local_transform=None,
    ):
        if not isinstance(presample, bool):
            raise TypeError("presample must be a boolean")
        self.presampler = GridSample(grid_size) if presample else None
        self.generator = MultiViewGenerator(
            global_view_num=global_view_num,
            local_view_num=local_view_num,
            global_view_scale=global_view_scale,
            local_view_scale=local_view_scale,
            max_size=max_size,
            min_points=min_points,
            global_transform=_nuscenes_geometry()
            if global_transform is None
            else global_transform,
            local_transform=_nuscenes_geometry()
            if local_transform is None
            else local_transform,
            view_transform=(_PrepareLitePTView(grid_size),),
        )

    def __call__(self, item):
        if not isinstance(item, Mapping):
            raise TypeError("reader item must be a mapping with point and meta")
        point = item["point"]
        _validate_arrays(point, ("origin_coord", "strength", "point_id"))
        for key, width in (("origin_coord", 3), ("strength", 1)):
            array = point[key]
            if not np.issubdtype(array.dtype, np.floating):
                raise TypeError(f"{key} must be a floating-point NumPy array")
            if array.shape != (len(point["coord"]), width):
                raise ValueError(f"{key} must have shape (N, {width})")
            if not np.isfinite(array).all():
                raise ValueError(f"{key} must contain finite values")
        # Pass only base fields; stale raw feat/offset/grids cannot enter views.
        base = {key: point[key] for key in self.generator.view_keys if key in point}
        if self.presampler is not None:
            base = self.presampler(base)
            # Sampling's grid describes the unaugmented full cloud. Each crop
            # receives a fresh grid only after its independent geometry.
            base = {key: base[key] for key in self.generator.view_keys if key in base}
        result = self.generator(base)
        result["meta"] = copy.deepcopy(item["meta"])
        return result
