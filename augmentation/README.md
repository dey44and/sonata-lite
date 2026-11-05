# Sonata-style LiDAR view augmentation

This NumPy package adapts published Sonata/Pointcept code for the raw
single-keyframe nuScenes reader and LitePT's sparse-convolution input. It does
not import PyTorch, SpConv, SciPy, or the full Sonata/Pointcept packages.

```python
import random
import numpy as np

from augmentation import NuScenesSSLViews
from datasets import NuScenesKeyframeReader

random.seed(42)
np.random.seed(42)

reader = NuScenesKeyframeReader(
    "/home/cristian/Datasets/nuScenes", version="v1.0-mini"
)
augment = NuScenesSSLViews(grid_size=0.10)
item = reader.read(reader.sample_tokens[0])
views = augment(item)

teacher_global = views["global_views"]  # Two independently augmented crops.
student_local = views["local_views"]   # Four independently augmented crops.
meta = views["meta"]
```

The input item is unchanged. Each view owns independent arrays. Construct and
reuse the reader and augmentation objects; calling the augmentation samples new
random crops and transforms each time. The pipeline is pickleable. Both Python
and NumPy random generators must be seeded for reproducibility, as in upstream.

## Processing order

1. Select one real point per occupied **shared full-cloud voxel**. This follows
   Sonata's initial sampling stage and reduces near-sensor density bias before
   point-count crops. Set `presample=False` to compare cropping the raw scan.
2. Generate two global and four local crops using Sonata's principal-crop rule.
   Global point-count fractions are 40–100%; local fractions are 10–40% of
   `min(number_of_points_after_shared_sampling, max_size)`. `max_size=65536`.
   These are point-count fractions, not crop radii or volume fractions.
3. Independently augment each crop using the official LitePT nuScenes recipe:
   yaw in `[-pi, pi]` about the sensor origin with probability 0.5, isotropic
   scaling in `[0.9, 1.1]`, X/Y flips with probability 0.5 each, and Gaussian
   coordinate jitter with sigma 0.005 m clipped to ±0.02 m. No center shift,
   elastic distortion, RGB augmentation, or intensity augmentation is applied.
4. Convert coordinates to float32, then **sample each augmented view again** to
   remove final voxel collisions and generate its explicit sparse-grid indices.
   Both sampling passes use `grid_size`, which defaults to our experimental
   0.10 m baseline. Official LitePT outdoor segmentation uses 0.05 m.
5. Rebuild `feat = [coord, strength]` and the view's `offset = [N]`.

Centers for the second global and local crops lie in the principal global crop.
Crop membership is selected from the shared full cloud by nearest-point sorting;
the resulting crops are not necessarily subsets of the principal crop. Local
centers favor principal-crop points not covered by previous local crops.

`origin_coord` remains in the unaugmented sensor frame; both sampling passes
and cropping select its corresponding rows. `strength` is already intensity/255
from the reader and is not normalized again. The shared pass reduces density
imbalance but does not guarantee balanced distance bands or semantic coverage.

## Per-view fields

| Field | Shape / value | Type |
| --- | --- | --- |
| `coord` | `(N, 3)`, augmented XYZ | float32 array |
| `origin_coord` | `(N, 3)`, corresponding original XYZ | float32 array |
| `strength` | `(N, 1)`, unchanged normalized intensity | float32 array |
| `feat` | `(N, 4)`, current XYZ + strength | float32 array |
| `grid_coord` | `(N, 3)`, unique nonnegative voxel indices | int64 array |
| `grid_size` | Configured voxel width in meters | Python float |
| `offset` | `[N]` | int64 array |

An optional input `point_id` array is indexed consistently and retained for
debugging correspondence to the raw scan. Other reader metadata is deep-copied
under the output `meta`, outside the model's point dictionaries.

`min_points` prevents empty pre-voxel crops. It does not guarantee a minimum
number of unique voxels after sampling; sparse-view quality needs to be inspected
when changing crop ratios. `global_transform=()` and `local_transform=()` disable
geometry for debugging. Passing `None` selects the default geometric recipe.

## Training boundary

These are **CPU views before tensor conversion and collation**. The global views
are teacher inputs and templates for masked student globals. Creating patch
masks, disturbing masked coordinates, and suppressing input features before
LitePT's spatial stem remain model-side training work. This package does not
pretend that passing a boolean mask to the current backbone hides features.

Coordinate corruption in a future masked branch must precede its final sampling
or trigger another sampling pass. Teacher and student point counts can differ;
matching must use the preserved original-coordinate frame. Never reuse stale
serialization or sparse-convolution caches after changing geometry.

## Source provenance and necessary adaptations

| Component | Reuse / adaptation |
| --- | --- |
| `RandomRotate`, `RandomScale`, `RandomFlip`, `RandomJitter` | Original Sonata class bodies, with registry decorators removed. |
| `Compose` | Small callable-only adapter; avoids the source registry dependency. |
| `GridSample` | Source floor/group/sort/representative pattern; training mode only. Exact voxel rows replace hash grouping, per-voxel random selection is unbiased, and aligned reference fields are included. |
| `MultiViewGenerator` | Source principal/global/local crop and local coverage logic. Adds validation and nonempty crop clamping, returns separate dictionaries, and allows final per-view voxelization. |
| `NuScenesSSLViews` | Project-specific adapter for reader items, intensity features, default geometric recipe, and final LitePT fields. |

Quantization uses `floor(coord.astype(float64) / grid_size)`, followed by
subtraction of the minimum **integer** cell. Continuous coordinates are preserved,
not snapped to voxel centers or averaged. Float64 division avoids float32
rounding across voxel boundaries; final coordinates are cast to float32 before
this calculation. The resulting `grid_coord` must be passed into LitePT rather
than recomputed using a different lattice origin.

Pinned primary sources:

- [Sonata transforms, commit 18c09ff](https://github.com/facebookresearch/sonata/blob/18c09ff8d713494f78a8213792262b910977a65d/sonata/transform.py).
- [Pointcept Sonata view generator, commit 96e109d](https://github.com/Pointcept/Pointcept/blob/96e109d103418a5daaa375c50e1665b9f04f4a4b/pointcept/datasets/transform.py#L1011-L1127).
- [Sonata training configuration](https://github.com/Pointcept/Pointcept/blob/96e109d103418a5daaa375c50e1665b9f04f4a4b/configs/sonata/pretrain-sonata-v1m1-0-base.py#L110-L187).
- [LitePT nuScenes augmentation configuration, commit 436d048](https://github.com/prs-eth/LitePT/blob/436d04801c8151faebe66a1b2d368a9711e7e6aa/configs/nuscenes/semseg-litept-small-v1m1.py#L87-L123).
- [Sonata paper, Appendix A](https://arxiv.org/html/2503.16429v1#A1).

This combination is our adaptation, not an official Sonata nuScenes SSL recipe.
Upstream copyright and license texts are retained in `LICENSES/`.

Run `python -m pytest -q tests` from the repository root.
