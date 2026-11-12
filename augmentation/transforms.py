"""
NumPy transforms adapted from Sonata for our Sonata Lite.
Author: Andrei-Iosif Ioja (iojaandrei11@gmail.com)

Adapted from:
    Source: https://github.com/facebookresearch/sonata/blob/main/sonata/transform.py
    Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com).
"""

import random
from numbers import Real

import numpy as np


class Compose:
    """Apply an iterable of callable transforms in order, without a registry."""

    def __init__(self, transforms=None):
        self.transforms = list(transforms) if transforms is not None else []
        if not all(callable(transform) for transform in self.transforms):
            raise TypeError("Compose expects callable transforms")

    def __call__(self, data_dict):
        for transform in self.transforms:
            data_dict = transform(data_dict)
        return data_dict


class RandomRotate:
    def __init__(self, angle=None, center=None, axis="z", always_apply=False, p=0.5):
        self.angle = [-1, 1] if angle is None else angle
        self.axis = axis
        self.always_apply = always_apply
        self.p = p if not self.always_apply else 1
        self.center = center

    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict
        angle = np.random.uniform(self.angle[0], self.angle[1]) * np.pi
        rot_cos, rot_sin = np.cos(angle), np.sin(angle)
        if self.axis == "x":
            rot_t = np.array([[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]])
        elif self.axis == "y":
            rot_t = np.array([[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]])
        elif self.axis == "z":
            rot_t = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
        else:
            raise NotImplementedError
        if "coord" in data_dict:
            if self.center is None:
                x_min, y_min, z_min = data_dict["coord"].min(axis=0)
                x_max, y_max, z_max = data_dict["coord"].max(axis=0)
                center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
            else:
                center = self.center
            # Here we have to subtract the center in order to place the origin of our 3D system in that point
            # After rotating the point cloud, we add back the center to each 3D point
            data_dict["coord"] -= center
            data_dict["coord"] = np.dot(data_dict["coord"], np.transpose(rot_t))
            data_dict["coord"] += center
        if "normal" in data_dict:
            data_dict["normal"] = np.dot(data_dict["normal"], np.transpose(rot_t))
        return data_dict


class RandomScale:
    def __init__(self, scale=None, anisotropic=False):
        self.scale = scale if scale is not None else [0.95, 1.05]
        self.anisotropic = anisotropic

    def __call__(self, data_dict):
        if "coord" in data_dict:
            scale = np.random.uniform(
                self.scale[0], self.scale[1], 3 if self.anisotropic else 1
            )
            data_dict["coord"] *= scale
        return data_dict


class RandomFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data_dict):
        # Applying random flip on X-axis
        if np.random.rand() < self.p:
            if "coord" in data_dict:
                data_dict["coord"][:, 0] = -data_dict["coord"][:, 0]
            if "normal" in data_dict:
                data_dict["normal"][:, 0] = -data_dict["normal"][:, 0]
        # Applying random flip on Y-axis
        if np.random.rand() < self.p:
            if "coord" in data_dict:
                data_dict["coord"][:, 1] = -data_dict["coord"][:, 1]
            if "normal" in data_dict:
                data_dict["normal"][:, 1] = -data_dict["normal"][:, 1]
        return data_dict


class RandomJitter:
    def __init__(self, sigma=0.01, clip=0.05):
        assert clip > 0
        self.sigma = sigma
        self.clip = clip

    def __call__(self, data_dict):
        if "coord" in data_dict:
            jitter = np.clip(
                self.sigma * np.random.randn(data_dict["coord"].shape[0], 3),
                -self.clip,
                self.clip,
            )
            data_dict["coord"] += jitter
        return data_dict


class GridSample:
    """Select one measured point per occupied voxel of one unbatched view.

    Quantization applies ``floor(coord / grid_size)`` and then
    subtracts the minimum integer cell on each axis. Continuous ``coord`` and
    ``origin_coord`` are only indexed.

    The returned ``index`` addresses input rows. Optional ``inverse`` maps every
    input row to its output voxel row.
    Listed per-point fields must be NumPy arrays with the input point count.
    Additional field names can be supplied through ``index_valid_keys``.
    """

    _point_keys = (
        "coord",
        "origin_coord",
        "strength",
        "feat",
        "color",
        "normal",
        "segment",
        "instance",
        "mask",
        "point_id",
    )

    def __init__(self, grid_size=0.1, return_inverse=False):
        if (
            isinstance(grid_size, (bool, np.bool_))
            or not isinstance(grid_size, Real)
            or not np.isfinite(grid_size)
            or grid_size <= 0
        ):
            raise ValueError("grid_size must be a positive finite scalar in meters")
        self.grid_size = float(grid_size)
        self.return_inverse = return_inverse

    def __call__(self, data_dict):
        coord = data_dict["coord"]
        # Check the integrity and format of the coord field
        if (
            not isinstance(coord, np.ndarray)
            or coord.ndim != 2
            or coord.shape[1] != 3
            or coord.shape[0] == 0
            or not np.issubdtype(coord.dtype, np.floating)
            or not np.isfinite(coord).all()
        ):
            raise ValueError(
                "coord must be a nonempty finite floating array shaped (N, 3)"
            )
        # Check if we have one unbatched view
        if "batch" in data_dict:
            raise ValueError("GridSample expects one unbatched view, without batch")

        point_count = len(coord)
        if "offset" in data_dict:
            offset = np.asarray(data_dict["offset"])
            if offset.shape != (1,) or offset[0] != point_count:
                raise ValueError("GridSample expects the offset of one unbatched view")

        # Check if point fields have matching row counts
        custom_keys = data_dict.get("index_valid_keys", ())
        if isinstance(custom_keys, str):
            raise TypeError("index_valid_keys must be an iterable of field names")
        point_keys = list(dict.fromkeys((*self._point_keys, *custom_keys)))
        for key in point_keys:
            if key in data_dict:
                value = data_dict[key]
                if not isinstance(value, np.ndarray) or value.ndim == 0 or value.shape[0] != point_count:
                    raise ValueError(
                        f"point field {key!r} must have {point_count} rows"
                    )

        # Float64 quantization and explicit bounds prevent integer wrap (not necessary for nuScenes however)
        scaled_coord = np.floor(coord.astype(np.float64) / self.grid_size)
        grid_coord = scaled_coord.astype(np.int64)
        grid_coord -= grid_coord.min(0)

        # Exact rows replace the upstream 64-bit hash, avoiding hash collisions.
        _, inverse, count = np.unique(
            grid_coord, axis=0, return_inverse=True, return_counts=True
        )

        idx_sort = np.argsort(inverse)
        idx_select = np.cumsum(np.insert(count, 0, 0)[:-1]) + np.random.randint(count)
        idx_unique = idx_sort[idx_select]
        for key in point_keys:
            if key in data_dict:
                data_dict[key] = data_dict[key][idx_unique]
        data_dict["index_valid_keys"] = point_keys
        data_dict["index"] = idx_unique.astype(np.int64, copy=False)
        data_dict["grid_coord"] = grid_coord[idx_unique]
        data_dict["grid_size"] = self.grid_size
        data_dict["offset"] = np.array([len(idx_unique)], dtype=np.int64)
        if self.return_inverse:
            data_dict["inverse"] = inverse.astype(np.int64, copy=False)
        else:
            data_dict.pop("inverse", None)
        return data_dict
