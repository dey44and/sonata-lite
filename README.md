# sonata-lite
Some experiments using Sonata from CVPR 2025, LitePT architecture and nuScenes dataset.

For installation, dataset configuration, H200 cluster jobs and optional Weights
& Biases logging, see [README_CLUSTER.md](README_CLUSTER.md).

## Single-keyframe nuScenes reader

Run from this repository's root. The reader needs NumPy; it does not import
PyTorch, SpConv, or the nuScenes SDK.

```python
from datasets import NuScenesKeyframeReader

reader = NuScenesKeyframeReader(
    "/home/cristian/Datasets/nuScenes", version="v1.0-mini"
)
sample_token = reader.sample_tokens[0]  # Or choose a specific sample token.
item = reader.read(sample_token)
point, meta = item["point"], item["meta"]

for name, array in point.items():
    print(name, array.shape, array.dtype)
print(meta)
```

Each item contains one `LIDAR_TOP` **keyframe**, selected through the sensor,
calibration, and sample-data metadata. Past sweeps are not aggregated. Construct
the reader once and reuse it: metadata is streamed into a compact keyframe index,
and scan files are read on demand. `v1.0-trainval` is also supported, but this reader
does not yet implement official train/validation scene splits or a PyTorch Dataset.

The raw scan format is `(N, 5)` float32: XYZ, intensity, ring index. Coordinates
remain in the native LiDAR sensor frame, in meters. The returned CPU arrays are:

| Field | Shape | dtype | Meaning |
| --- | --- | --- | --- |
| `coord` | `(N, 3)` | float32 | Raw XYZ |
| `origin_coord` | `(N, 3)` | float32 | Independent copy of the original XYZ |
| `strength` | `(N, 1)` | float32 | Intensity divided by 255 |
| `feat` | `(N, 4)` | float32 | `[x, y, z, strength]` |
| `offset` | `(1,)` | int64 | `[N]` |

`meta` contains sample, sensor-reading and scene tokens, the timestamp in
microseconds, the scan path, metadata version, and coordinate-frame name.
`read_lidar_bin(path)` is available separately when only binary decoding is needed;
it retains all five fields, but cannot establish keyframe identity by itself.

**This is the raw reading stage, not a model-ready batch.** Grid sampling must
select one representative per occupied cell and produce `grid_coord` before a
LitePT forward. `grid_size` belongs to that later stage. Spatial SSL transforms
must keep corresponding `origin_coord` rows in their original frame and rebuild
the XYZ channels of `feat` after changing `coord`. Tensor conversion and collation
are applied after those transforms by the training data loader.

Malformed, empty, and non-finite scans raise errors. Missing scan files are checked
on read, rather than requiring the entire dataset to be present during indexing.

Sources: [nuScenes binary format](https://github.com/nutonomy/nuscenes-devkit/blob/b40adc467b919192899405d9b77871afee8efa07/python-sdk/nuscenes/utils/data_classes.py#L236-L258),
[nuScenes sensor-frame coordinates](https://github.com/nutonomy/nuscenes-devkit/blob/b40adc467b919192899405d9b77871afee8efa07/python-sdk/nuscenes/nuscenes.py#L884-L903),
[LitePT intensity convention](https://github.com/prs-eth/LitePT/blob/436d04801c8151faebe66a1b2d368a9711e7e6aa/datasets/nuscenes.py#L49-L80).

Run the reader tests with `python -m pytest tests`.

## SSL view augmentation

The NumPy-only `augmentation` package reuses Sonata's geometric transforms and
principal/global/local crop generation, adapted for nuScenes intensity and
LitePT's unique sparse-grid inputs:

```python
from augmentation import NuScenesSSLViews, MaskSchedule

schedule = MaskSchedule(total_steps=1000)  # Example training length.
augment = NuScenesSSLViews(grid_size=0.10, mask_schedule=schedule)
views = augment(item, step=0)  # item is the raw reader result above.
global_views = views["global_views"]         # Two teacher crops.
masked_views = views["masked_global_views"]  # Two student copies with patch masks.
local_views = views["local_views"]           # Four unmasked student crops.
```

Shared voxel sampling precedes crop generation, and final sampling follows each
view's geometry. Each view includes explicit `grid_coord`, preserved
`origin_coord`, rebuilt four-channel `feat`, and `offset`. Intensity remains
unchanged. The experimental voxel width is configurable; official LitePT uses
0.05 m for nuScenes segmentation. Masked globals retain all points and add a
boolean `mask` selecting occupied patches. Sonata's progressive schedule grows
patch width from 0.10 to 0.40 m and the occupied-patch ratio from 30% to 70%
linearly during the first 5% of training steps, then holds both constant. The
ratio counts patches across this item's globals. No additional masked-coordinate
jitter is applied in this first version.

The caller passes the zero-based training step explicitly, including on resume;
there is no worker-local counter. The training sampler supplies the
intended batch step when prefetching. Omitting `mask_schedule` retains fixed
masking, configurable through `mask_size` and `mask_ratio`.

`litept_v1.py` replaces selected rows after your pointwise preembedding and
before the spatial convolution with a learned 36-dimensional token. The sparse
feature cache is updated with the replacement. Tensor conversion, batching,
teacher/student orchestration and SSL losses are implemented in the `trainer`
package described below.

See [augmentation/README.md](augmentation/README.md) for the exact order,
parameters, output contract, source attribution, and adaptations.

The [data preparation and masking report](reports/ssl_data_preparation_and_masking.md)
covers the proposed next phase, published masking mechanisms, measured patch
populations, and the staged implementation and verification plan.

## Browser view inspection

```bash
python visualize_views.py --sample-index 0 --seed 42
```

Inspect a scheduled training step with, for example,
`python visualize_views.py --total-steps 1000 --step 50 --seed 42`.

Opens a local 3D viewer with a dropdown for the original nuScenes keyframe, two
global views, their two masked student copies and four local views. Masked
features are highlighted in pink, with the actual selected point count shown.
Switch between augmented coordinates and
original positions to inspect crop coverage. Includes orbit/pan/zoom, point size
and a fit button; works offline with bundled Three.js files.

See [visualization/README.md](visualization/README.md) for dataset selection,
options, and displaying views that have already been generated.

## SSL training

The entry point is `train.py`, configured by
[configs/nuscenes_ssl_litept_s.json](configs/nuscenes_ssl_litept_s.json).
This first trainer supports one GPU. Its default dataset selection is the
official nuScenes **train** scene split from `v1.0-trainval`; validation and test
scenes are excluded. It reads individual LIDAR_TOP keyframes, without sweeps.

```bash
python train.py --config configs/nuscenes_ssl_litept_s.json
```

Start with a short real-data smoke test:

```bash
python train.py --version v1.0-mini --split mini_train \
  --steps 1000 --stop-after 2 --max-samples 2 --max-points 256 \
  --batch-size 1 --workers 0 --amp none --output-dir runs/smoke
python train.py --resume runs/smoke/last.pt --stop-after 2
```

`--steps` is the fixed schedule horizon; `--stop-after` limits this invocation
without changing the curriculum. Resume restores both networks, AdamW, the
gradient scaler, RNG states, and the next planned data step. Changing worker
count or output directory is supported; changing data selection, batch size,
model, schedules, precision, gradient clipping or drop-last behavior is rejected.

The pipeline is:

```text
nuScenes reader -> NumPy global/local/masked views -> CPU tensor collation
  -> CUDA -> unmasked teacher globals, masked student globals, student locals
  -> encoder features -> two concatenating upcasts -> 900-channel features
  -> separate mask/unmask OnlineCluster heads -> original-coordinate matching
  -> teacher Sinkhorn targets + student log-softmax -> three weighted losses
  -> backward -> gradient clipping -> AdamW -> teacher EMA
```

The sampler attaches the intended zero-based step to every sample request.
All samples in a batch share this step, even with worker prefetching. Views are
packed in scene-major order with one cumulative offset per view. Augmentation
RNG depends on `(seed, step, sample_index)`, so worker count does not change
the generated batch. CPU pinning and GPU transfer happen after collation.

The SSL model uses encoder-only LitePT-S, a 900 -> 4096 -> 256 projection,
unit-normalized embeddings, and 4096 unit-norm prototype weights per head.
Masked student globals compare to the corresponding teacher global and to the
other teacher global. Student locals compare to the principal teacher global.
Their respective weights are 0.25, 0.25, 0.50. Every spatially matched row
participates, including visible rows of a masked view. Matching is exact,
many-to-one, and limited to the intended scene/view pair. It uses chunked
Torch distance calculations rather than requiring PointOps. The configurable
0.32 m threshold is borrowed from Sonata's indoor configuration; its suitability
for our 0.10 m nuScenes voxel grid remains an experimental choice. Logs include
matched rows and valid-view counts for every branch.

Teacher scores are selected through matching before three FP32 Sinkhorn
iterations; repeated teacher matches remain repeated. Losses average within
valid views, then across valid views. Empty comparisons are excluded. If an
entire batch has no matches, both optimizer and EMA updates are skipped.
Sinkhorn supports distributed reductions, but this trainer does not yet wrap
the networks/data sampler for DDP.

The teacher is frozen and kept in evaluation mode. EMA updates all parameters
and floating buffers after successful optimizer updates; integer BN counters
are copied. Updating BN state is a LitePT-specific adaptation because the
reference Sonata PTv3 backbone uses LayerNorm. AMP overflow skips EMA as well
as the optimizer update. `global_step` counts consumed batches;
`optimizer_steps` separately counts successful updates.

Optional Weights & Biases logging records these metrics and the effective
configuration, with checkpointed run IDs and explicit training-step axes.
Install [requirements-wandb.txt](requirements-wandb.txt) and enable it with
`--wandb` or `--wandb-mode offline`; local JSONL logging remains active.
See the [cluster logging guide](README_CLUSTER.md#7-optional-weights--biases-logging)
for authentication, offline synchronization and resume behavior.

The example enables FP16 AMP for both backbone and heads (`backbone_amp=true`).
SpConv supports LitePT's channel widths in FP16. Its evaluation-mode implicit
GEMM path, however, bypasses the autocast handling used during training. A teacher
can consequently present FP16 features to FP32 convolution weights. The
`PointSequential` adapter casts the sparse features and temporary convolution
parameter values together for each evaluation call using
`torch.func.functional_call`. Stored parameters remain FP32 for EMA, and teacher
BatchNorm remains in evaluation mode. `backbone_amp=false` remains available for
experiments with an FP32 backbone. The verified native recipe uses FP16; BF16
sparse convolutions are not covered by this verification.

Mask size/ratio and teacher temperature use linear warmup for the first 5%
of steps, then remain constant. Teacher momentum uses Sonata's cosine schedule
from 0.994 toward 1. The example learning rate uses linear warmup followed by
cosine decay, with AdamW LR 1e-4 and weight decay 0.04. This small-batch LR
recipe is an explicit project baseline; the reference Sonata configuration
uses OneCycleLR, layer-wise LR scaling, and a weight-decay schedule, which this
first trainer does not reproduce. The crop cap of 16,384 is also our baseline.
Mask generation retains the existing per-keyframe NumPy selection scope.

`registry.py` now provides `MODELS`, `DATASETS`, `TRANSFORMS`, and `TRAINERS`.
The lazy constructors in `builder.py` build nested JSON configurations without
requiring CUDA dependencies for NumPy-only transform construction. For example:

```python
from builder import build_model, build_trainer

model = build_model({
    "type": "SSLModel",
    "backbone": {"type": "LitePT-v1", "enc_mode": True},
    "head_in_channels": 900,
})
```

Use the dedicated `.venv` in this repository for training. Its native stack
uses Python 3.11, PyTorch 2.4.1+cu121, torchvision 0.19.1+cu121,
SpConv-cu121 2.3.8, FlashAttention 2.7.4.post1, timm 1.0.31, torch-scatter and
NumPy 1.26.4. This follows LitePT's PyTorch 2.4.1 baseline with CUDA 12.1
wheels matching the installed toolkit. It does not require modifying the
environments belonging to other projects.

```bash
source .venv/bin/activate
python -m pip check
python -m pytest tests/test_native_attention.py tests/test_spconv_amp.py -q
```

The exact installed runtime and test dependencies are pinned in
[requirements-cu121.txt](requirements-cu121.txt), including the official
FlashAttention wheel URL and its SHA256. To recreate this Linux environment
with Python 3.11 in a fresh checkout:

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-cu121.txt
```

Native FlashAttention is used when installed and `enable_flash=true`, as in
the supplied configuration. Its official packed interface supports LitePT's
18-channel heads by internally padding and trimming the head dimension. The
native integration tests verify an actual extension call, agreement with the
reference attention path, finite gradients, and isolation between packed views.
`DropPath` comes directly from `timm.layers` in this environment. PointROPE
currently uses the existing Torch implementation; its separate CUDA extension
is not installed. The SDPA and stochastic-depth fallbacks remain available for
environments without the native dependencies.

Verification on the RTX 3090 passed 15 native CUDA integration tests and 48
focused trainer/data/loss/schedule/registry/attention regression tests. A real
`v1.0-mini` / `mini_train` smoke run used batch size 2, a 16,384-point crop cap,
two data workers and FP16 AMP throughout the eligible backbone/head operations.
Both initial AdamW/EMA updates completed with finite losses and gradients.
Forward hooks recorded FP16 teacher and student sparse convolutions and 48
native FlashAttention calls over those two updates. The teacher stayed in eval
mode with FP32 stored parameters. Peak CUDA allocated memory was 4,289.2 MiB
for this run; this is a workload-specific smoke measurement. It includes the
teacher/student networks, training state and batch intermediates, and is not
the size of model parameters alone. Checkpoint resume then completed a third
AdamW/EMA update after changing the data worker count from two to zero. The
versions, backend call counts and training metrics are recorded in
[reports/native_training_verification.json](reports/native_training_verification.json).

The older test suite still contains expectations for reader/mask/validation
interfaces that changed before trainer integration; those failures are separate
from the focused training tests.

Implementation references:
[Sonata model, pairing, losses and EMA](https://github.com/Pointcept/Pointcept/blob/96e109d103418a5daaa375c50e1665b9f04f4a4b/pointcept/models/sonata/sonata_v1m1_base.py),
[Sonata configuration](https://github.com/Pointcept/Pointcept/blob/96e109d103418a5daaa375c50e1665b9f04f4a4b/configs/sonata/pretrain-sonata-v1m1-0-base.py),
[Pointcept scalar scheduler](https://github.com/Pointcept/Pointcept/blob/96e109d103418a5daaa375c50e1665b9f04f4a4b/pointcept/utils/scheduler.py),
[nuScenes official scene splits](https://github.com/nutonomy/nuscenes-devkit/blob/b40adc467b919192899405d9b77871afee8efa07/python-sdk/nuscenes/utils/splits.py),
[PyTorch SDPA](https://docs.pytorch.org/docs/2.4/generated/torch.nn.functional.scaled_dot_product_attention.html),
[LitePT environment](https://github.com/prs-eth/LitePT#installation),
[SpConv evaluation/training paths](https://github.com/traveller59/spconv/blob/v2.3.8/spconv/pytorch/conv.py),
[SpConv CUDA 12.1 wheel](https://pypi.org/project/spconv-cu121/2.3.8/),
[PyTorch functional_call](https://docs.pytorch.org/docs/2.4/generated/torch.func.functional_call.html),
[FlashAttention packed head padding](https://github.com/Dao-AILab/flash-attention/blob/v2.7.4.post1/flash_attn/flash_attn_interface.py#L506-L596),
[PyG extension wheels](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).
