# Sonata-style LiDAR view augmentation

This NumPy package adapts Sonata code for visualizing a single nuScenes frame, adapting
the input for LitePT backbone.

```python
import random
import numpy as np

from pathlib import Path
from augmentation import NuScenesSSLViews, MaskSchedule
from datasets import NuScenesKeyframeReader

random.seed(42)
np.random.seed(42)

home = Path.home()

reader = NuScenesKeyframeReader(
    home / "Datasets/nuScenes", version="v1.0-mini"
)
schedule = MaskSchedule(total_steps=1000)  # Example training length.
augment = NuScenesSSLViews(grid_size=0.10, mask_schedule=schedule)
item = reader.read(reader.sample_tokens[0])
views = augment(item, step=0)  # Explicit, zero-based training step.

teacher_global = views["global_views"]  # Two independently augmented crops.
student_masked = views["masked_global_views"]  # Two global masked views
student_local = views["local_views"]   # Four independently augmented crops.
meta = views["meta"]
```

For inspection, `augment(item, return_shared=True)` also returns `shared_input`,
an independent copy of the exact cloud used for crop generation. With presampling
enabled it includes the initial voxel coordinates and grid size.

### Processing order

1. Select one real point per occupied **shared full-cloud voxel**. This follows
   Sonata's initial sampling stage and reduces near-sensor density bias before
   point-count crops. Set `presample=False` to compare cropping the raw scan.
2. Generate two global and four local crops using Sonata's crop rule.
   Global point-count fractions are 40–100% and local fractions are 10–40% of
   `min(number_of_points_after_shared_sampling, max_size)`, where `max_size=65536`.
3. Independently augment each crop using the official LitePT nuScenes rules:
   yaw in `[-pi, pi]` about the sensor origin with probability 0.5, isotropic
   scaling in `[0.9, 1.1]`, X/Y flips with probability 0.5 each, and Gaussian
   coordinate jitter with sigma 0.005 m clipped to ±0.02 m.
4. Convert coordinates to float32, then **sample each augmented view again** to
   remove final voxel collisions and generate its explicit sparse-grid indices.
   Both sampling passes use `grid_size`, which defaults to our experimental
   0.10 m baseline.
5. Rebuild `feat = [coord, strength]`.
6. Copy each finished global view into `masked_global_views`. Group its points
   into occupied cubes of the effective `mask_size` and flag selected patches with
   a boolean `mask`.

Centers for the second global and local crops lie in the principal global crop.
Crop membership is selected from the shared full cloud by nearest-point sorting. Thus,
the resulting crops are not necessarily subsets of the principal crop. Local
centers favor principal-crop points not covered by previous local crops.

`origin_coord` remains in the unaugmented sensor frame, then both sampling passes
and cropping select its corresponding rows. `strength` is already `intensity/255`
from the dataset reader. The shared pass reduces density imbalance but does not guarantee balanced distance bands or semantic coverage.

### Per-view fields

| Field | Shape / value | Type |
| --- | --- | --- |
| `coord` | `(N, 3)`, augmented XYZ | float32 array |
| `origin_coord` | `(N, 3)`, corresponding original XYZ | float32 array |
| `strength` | `(N, 1)`, unchanged normalized intensity | float32 array |
| `feat` | `(N, 4)`, current XYZ + strength | float32 array |
| `grid_coord` | `(N, 3)`, unique nonnegative voxel indices | int64 array |
| `grid_size` | Configured voxel width in meters | Python float |
| `offset` | `[N]` | int64 array |

Masked global views also contain:

| Field | Shape / value | Type |
| --- | --- | --- |
| `mask` | `(N,)`; `True` means replace this row's preembedding | bool array |
| `patch_id` | `(N,)`; occupied-patch membership, distinct across supplied globals | int64 array |
| `mask_size` | Masking cube width, independent of `grid_size` | Python float |
| `mask_ratio` | Requested fraction of occupied patches | Python float |
| `mask_step` | Explicit training step, scheduled mode only | Python int |
| `mask_total_steps` | Configured training length, scheduled mode only | Python int |

An optional input `point_id` array is indexed consistently and retained for
debugging correspondence to the raw scan. Other reader metadata is deep-copied
under the output `meta`, outside the model's point dictionaries.

`min_points` prevents empty pre-voxel crops. It does not guarantee a minimum
number of unique voxels after sampling, and sparse-view quality needs to be inspected
when changing crop ratios. `global_transform=()` and `local_transform=()` disable
geometry for debugging, but passing `None` selects the default geometric recipe.

### Progressive masking

`MaskSchedule` follows Sonata's published defaults:

| Parameter | Start | End of warmup and thereafter |
| --- | --- | --- |
| Masking cube side length | 0.10 m | 0.40 m |
| Fraction of occupied patches selected | 0.30 | 0.70 |
| Warmup length | First 5% of training steps | Constant afterward |

For `T=total_steps`, warmup occupies `W=int(0.05*T)` steps. The implementation
uses the original `np.linspace(start, base, W)` for both parameters, including
both endpoints. Thus, with 1,000 total steps, step 0 gives `(0.10, 0.30)`, step
49 gives `(0.40, 0.70)`, and step 50 onward stays at `(0.40, 0.70)`. If `W=0`,
the base values apply immediately; if `W=1`, step 0 uses the starting values.

```python
schedule = MaskSchedule(total_steps=1000)
print(schedule(0))   # (0.1, 0.3)
print(schedule(49))  # (0.4, 0.7)

augment = NuScenesSSLViews(mask_schedule=schedule)
views = augment(item, step=25)
```

Size and ratio are evaluated at the supplied step and used for patch grouping
and selection. The mask grid is independent of the backbone's `grid_size`.
The schedule is a stateless lookup: calls do not advance a counter or mutate the
fixed generator settings. Resuming uses the saved training step and the same
schedule configuration. Endpoints and `warmup_ratio` are configurable.

Without `mask_schedule`, `NuScenesSSLViews(mask_size=0.4, mask_ratio=0.3)(item)`
retains the fixed-mask behavior. With a schedule, `step=` is required and the
schedule takes precedence over the fixed size and ratio.

### Patch selection and Training boundary

`MaskedViewsGenerator` adapts Sonata's `generate_mask`: subtract each view's
minimum augmented XYZ, floor-divide by the masking cube width, include the view
ID in each cube key, and group occupied keys. Of the `Q` occupied patches across
the supplied globals, choose exactly `floor(mask_ratio * Q)` uniformly without
replacement and flag every point in those patches. Empty cubes are never stored.
Tiny inputs can legitimately receive no selected patches. The percentage of
masked **points**, and each individual view's selected fraction, can differ from
the requested patch ratio.

The current wrapper selects across the two globals of **one keyframe**.
Official Sonata selects after packing the globals of a training batch. The
algorithm is the same in our case.

The caller supplies the intended training step, this implies that every item belonging to one
training batch must use the same step. Data-loader prefetching does not track
the optimizer's live step automatically. A future loader must attach the
intended step to queued work, or generate masks in the training process after
batching. Trainer / data-loader integration is not included yet.

Our `litept_v1.py` now uses:

```text
[XYZ, intensity] -> Linear(4, 36) -> LayerNorm -> GELU
                -> replace rows where mask=True with a learned 36D token
                -> existing spatial convolution stem -> LitePT encoder
```

### Source provenance and adaptations

| Component | Reuse / adaptation |
| --- | --- |
| `RandomRotate`, `RandomScale`, `RandomFlip`, `RandomJitter` | Original Sonata class bodies, with registry decorators removed. |
| `Compose` | Small callable-only adapter; avoids the source registry dependency. |
| `GridSample` | Source floor/group/sort/representative pattern; training mode only. Exact voxel rows replace hash grouping, per-voxel random selection is unbiased, and aligned reference fields are included. |
| `MultiViewGenerator` | Source principal/global/local crop and local coverage logic. Adds validation and nonempty crop clamping, returns separate dictionaries, and allows final per-view voxelization. |
| `MaskedViewsGenerator` | NumPy adaptation of Sonata's occupied-cube grouping and fixed-count random patch selection; copies each global and retains point-aligned masks and patch IDs. |
| `MaskSchedule` | Source `np.linspace` warmup and constant tail, evaluated with an explicit step; stores only warmup arrays. |
| `NuScenesSSLViews` | Project-specific adapter for reader items, intensity features, default geometric recipe, final LitePT fields and masked global copies. |

Quantization uses `floor(coord.astype(float64) / grid_size)`, followed by
subtraction of the minimum **integer** cell. Continuous coordinates are preserved,
not snapped to voxel centers or averaged. Float64 division avoids float32
rounding across voxel boundaries, and the final coordinates are cast to float32 before
this calculation.

### Code Sources

- [Sonata transforms, commit 18c09ff](https://github.com/facebookresearch/sonata/blob/18c09ff8d713494f78a8213792262b910977a65d/sonata/transform.py).
- [Pointcept Sonata view generator, commit 96e109d](https://github.com/Pointcept/Pointcept/blob/96e109d103418a5daaa375c50e1665b9f04f4a4b/pointcept/datasets/transform.py#L1011-L1127).
- [Sonata training configuration](https://github.com/Pointcept/Pointcept/blob/96e109d103418a5daaa375c50e1665b9f04f4a4b/configs/sonata/pretrain-sonata-v1m1-0-base.py#L110-L187).
- [Sonata masking curriculum parameters](https://github.com/Pointcept/Pointcept/blob/96e109d103418a5daaa375c50e1665b9f04f4a4b/configs/sonata/pretrain-sonata-v1m1-0-base.py#L60-L65).
- [Sonata mask scheduler construction](https://github.com/Pointcept/Pointcept/blob/96e109d103418a5daaa375c50e1665b9f04f4a4b/pointcept/models/sonata/sonata_v1m1_base.py#L191-L209).
- [Pointcept scheduler implementation](https://github.com/Pointcept/Pointcept/blob/96e109d103418a5daaa375c50e1665b9f04f4a4b/pointcept/utils/scheduler.py#L155-L189).
- [Sonata occupied-patch mask generation](https://github.com/Pointcept/Pointcept/blob/96e109d103418a5daaa375c50e1665b9f04f4a4b/pointcept/models/sonata/sonata_v1m1_base.py#L293-L310).
- [LitePT nuScenes augmentation configuration, commit 436d048](https://github.com/prs-eth/LitePT/blob/436d04801c8151faebe66a1b2d368a9711e7e6aa/configs/nuscenes/semseg-litept-small-v1m1.py#L87-L123).
- [Sonata paper, Appendix A](https://arxiv.org/html/2503.16429v1#A1)

### Tests

Run `python -m pytest -q tests` from the repository root.
