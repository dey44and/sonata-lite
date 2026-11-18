# Training Sonata-style LitePT-S on an H200 cluster

This guide recreates the project environment, checks the native GPU backends,
and starts or resumes single-keyframe nuScenes SSL training. Run commands from
the repository root, the directory containing `train.py`.

The pinned environment has been tested on an RTX 3090. **H200 execution has not
yet been tested in this project.** The H200 has compute capability 9.0, and
FlashAttention-2 supports Hopper. The installed Torch, FlashAttention, SpConv
and torch-scatter binaries include support for this architecture; the native
tests below verify execution on the allocated cluster GPU. See NVIDIA's
[GPU table](https://developer.nvidia.com/cuda/gpus) and the pinned
[FlashAttention documentation](https://github.com/Dao-AILab/flash-attention/blob/v2.7.4.post1/README.md#nvidia-cuda-support).

## 1. Copy the source and create a fresh environment

Copy the working directory, including files that have not been committed to
Git. A fresh clone alone can omit required local files. Do not copy `.venv`:
Python virtual environments contain paths to the machine where they were
created. Recreate it on the cluster instead, as described in Python's
[virtual-environment documentation](https://docs.python.org/3.11/library/venv.html#how-venvs-work).

For example, run this on the source machine, replacing the destination:

```bash
rsync -a \
  --exclude='.venv/' --exclude='__pycache__/' --exclude='.pytest_cache/' \
  --exclude='runs/' --exclude='wandb/' \
  /home/cristian/andrei-work/sonata-lite/ \
  USER@CLUSTER:/CLUSTER/PROJECT/sonata-lite/
```

The trailing slash copies the contents of `sonata-lite`. This example excludes
training outputs. If continuing an existing experiment, transfer its run
directory separately, including `last.pt`, `config.json`, `metrics.jsonl`, and
any offline W&B directories that should later be synchronized. The repository
copy does not include nuScenes; make the dataset available separately.

Use **Linux x86_64 and Python 3.11** for this exact requirements file. Check
`ldd --version` as well: use glibc 2.28 or newer for the complete binary recipe,
including the optional [W&B 0.31.0 wheel](https://pypi.org/project/wandb/0.31.0/#files). The
FlashAttention wheel is specifically for CPython 3.11, x86_64, Torch 2.4 and
the C++11 ABI setting used by the pinned Torch wheel. An Arm host, including
a Grace-based host, needs a different environment/build even when its GPU
has the same compute capability.

```bash
cd /CLUSTER/PROJECT/sonata-lite
uname -m
ldd --version
python3.11 --version
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-cu121.txt
python -m pip check
```

Load the site's Python module first if necessary; module names are specific
to the cluster. Install into a new `.venv`, without `--system-site-packages`,
so unrelated Torch/CUDA packages do not enter this environment.

The [pinned requirements](requirements-cu121.txt) install the following stack:

| Component | Pinned version |
| --- | --- |
| Python interpreter | 3.11; created separately |
| PyTorch / torchvision | 2.4.1+cu121 / 0.19.1+cu121 |
| SpConv / cumm | spconv-cu121 2.3.8 / cumm-cu121 0.7.11 |
| FlashAttention | 2.7.4.post1; official binary wheel with SHA256 |
| timm | 1.0.31 |
| torch-scatter | 2.1.2+pt24cu121 |
| NumPy | 1.26.4 |

These are prebuilt packages; the documented installation does not compile
FlashAttention or PointROPE. A system CUDA toolkit is needed if you later
choose to build extensions from source. Keep that as a separate environment
change rather than mixing packages into this recipe.

Installation needs access to PyPI, the PyTorch wheel index, GitHub releases,
and the PyG wheel index. If compute nodes have no internet, installation can
take place on a compatible login/build node with the same shared project
directory, subject to cluster policy. GPU execution checks must still run on
an allocated GPU node. The requirements file contains a direct GitHub wheel
URL; adding `--no-index` alone does not make installation fully offline.

## 2. Check the allocated H200 and native kernels

Request one H200 through your cluster scheduler before running GPU commands.
The current trainer uses **one process and one GPU**; it does not implement
DDP. Do not launch it using `torchrun` or multiple Slurm tasks.

Use the cluster-maintained H200 driver. NVIDIA's R550.90.07 release notes list
HGX H200 among supported configurations; a suitable newer site driver is also
appropriate. The CUDA version shown by `nvidia-smi` describes driver support,
whereas `torch.version.cuda` identifies the runtime used by Torch. A newer
driver does not require replacing our CUDA 12.1 wheels. See NVIDIA's
[R550 supported hardware](https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-550-90-07/index.html#hardware-software-support)
and [CUDA compatibility guidance](https://docs.nvidia.com/deploy/cuda-compatibility/latest/minor-version-compatibility.html).

```bash
source .venv/bin/activate
nvidia-smi
python - <<'PY'
import platform
import importlib.metadata as metadata
import torch
import litept_v1
from timm.layers import DropPath

print('Python:', platform.python_version(), platform.machine())
print('Torch:', torch.__version__, 'CUDA runtime:', torch.version.cuda)
assert torch.cuda.is_available(), 'Run inside an allocated GPU job'
print('GPU:', torch.cuda.get_device_name(0))
print('Compute capability:', torch.cuda.get_device_capability(0))
for name in ('spconv-cu121', 'cumm-cu121', 'flash-attn', 'timm', 'torch-scatter'):
    print(name, metadata.version(name))
assert litept_v1.DropPath is DropPath, 'Native timm layers are not active'
assert litept_v1.flash_attn is not None, 'FlashAttention failed to import'
PY
python -m pytest tests/test_native_attention.py tests/test_spconv_amp.py -q -rs
```

The native tests exercise the actual FlashAttention extension, its
18-dimensional LitePT attention heads, backward gradients, view boundaries,
and FP16 SpConv in both teacher evaluation and student training paths. Inspect
the result: skipped CUDA tests do not establish GPU compatibility. On the
verified RTX 3090 environment these two files passed 15 tests.

The configuration enables `amp: "float16"`, `model.backbone_amp: true`, and
`model.backbone.enable_flash: true`. Start with these values on H200. BF16
sparse convolution is not part of the current verification. The separate
FlashAttention-3 interface is not integrated; installing it does not switch
the model to FA3. PointROPE currently uses the existing Torch implementation.

For the focused trainer regression suite:

```bash
python -m pytest \
  tests/test_training_data.py tests/test_ssl_core.py \
  tests/test_training_engine.py tests/test_training_registry.py \
  tests/test_training_schedules.py tests/test_litept_attention.py -q
```

Some older tests retain expectations for previous reader and masking APIs;
the focused suite above covers the current training path. The existing
`trainer/schedules.py` remains unchanged.

## 3. Make the dataset available

Use the directory containing both the version metadata and the sensor files:

```text
/CLUSTER/DATA/nuScenes/
├── v1.0-mini/                 # for the mini smoke test, if available
│   ├── sample.json
│   ├── sample_data.json
│   ├── sensor.json
│   ├── calibrated_sensor.json
│   └── scene.json             # retain the full official metadata directory
├── v1.0-trainval/             # for the full training run
│   └── ...
└── samples/
    └── LIDAR_TOP/
        └── *.pcd.bin
```

Keep the official relative paths referenced in `sample_data.json`; the
LIDAR_TOP files for all selected scenes must be accessible. This project
reads raw keyframe scans and does not require detector `.pkl` information
files, image files, labels, or historical `sweeps/` for this training path.
The official split names are `mini_train` with `v1.0-mini` and `train` with
`v1.0-trainval`; validation/test scenes are excluded from those training
selections. Dataset format and download instructions are maintained by the
[official nuScenes devkit](https://github.com/nutonomy/nuscenes-devkit#nuscenes).

Pass the path explicitly with `--dataroot`; the supplied JSON defaults to
`~/Datasets/nuScenes`. For a new experiment, choose a stable shared path that
will remain accessible in future jobs. If using node-local scratch, a stable
symlink can keep the configured path unchanged between allocations.

## 4. Run a short real-data smoke test

First verify two full training updates, including backward, AdamW and EMA:

```bash
python -u train.py \
  --config configs/nuscenes_ssl_litept_s.json \
  --dataroot /CLUSTER/DATA/nuScenes \
  --version v1.0-mini --split mini_train \
  --steps 10000 --stop-after 2 --max-samples 2 \
  --batch-size 2 --workers 2 --amp float16 \
  --output-dir runs/h200_mini_smoke
```

This keeps the default 16,384-point crop cap. For debugging a failure, use a
new output directory with `--workers 0 --batch-size 1 --max-points 256`.
If mini data is unavailable, replace the version/split with
`--version v1.0-trainval --split train` and retain `--max-samples 2`.

Check that the loss and gradient norm are finite, `optimizer_steps` reaches 2,
and the branches report spatial matches. Then test checkpoint continuation:

```bash
python -u train.py --resume runs/h200_mini_smoke/last.pt \
  --stop-after 1 --workers 0
```

This should advance `optimizer_steps` to 3. `--steps` is the complete schedule
horizon; `--stop-after` limits the number of additional batches in this
invocation. Stopping a smoke test after two batches therefore does not
compress the warmup into two batches.

Use a fresh output directory for the full dataset run: the smoke checkpoint
contains its mini/two-sample dataset selection and should not initialize a
different experiment through `--resume`.

## 5. Start the full training run

```bash
python -u train.py \
  --config configs/nuscenes_ssl_litept_s.json \
  --dataroot /CLUSTER/DATA/nuScenes \
  --version v1.0-trainval --split train \
  --steps 10000 --batch-size 2 --workers 4 --amp float16 \
  --output-dir runs/h200_nuscenes_ssl
```

Training is configured by consumed batches, rather than a requested epoch
count. With batch size 2, 10,000 batches are a baseline experiment and may
not cover a complete nuScenes training epoch. The sampler reshuffles when
it exhausts the selected keyframes.

The JSON controls the 0.10 m voxel grid, two teacher globals, two corresponding
masked student globals, four student locals, mask curriculum, temperatures,
EMA momentum, matching radius and optimizer. CLI overrides are listed by
`python train.py --help`. Keep an experiment-specific JSON if changing
settings without CLI flags; the effective configuration is saved with the run.
This is the project's Sonata-style LitePT baseline, not an exact reproduction
of Sonata's published training recipe; see the main [training documentation](README.md#ssl-training).

Begin with batch size 2 even on H200. Increase batch size or crop cap only
after measuring the real workload, and start a new experiment when doing so.
GPU memory also includes prototype scores, attention/convolution buffers,
optimizer state and teacher/student activations. Large VRAM alone does not
establish an appropriate batch size or a learning-rate scaling rule. CPU
augmentation and dataset I/O can also limit utilization; tune `--workers`
within the CPU allocation, and use fast storage for the sensor files.

### Example Slurm job

Save the following example as `train_h200.sbatch` on the cluster. Replace both
paths and add your site's required partition/account/GPU-type directives;
`--gres=gpu:1` is only a common Slurm spelling. The memory/CPU/time values are
starting requests to adapt to site policy and measured usage.

```bash
#!/bin/bash
#SBATCH --job-name=sonata-lite-ssl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
# If required, load your site's Python module here before activating .venv.
cd /CLUSTER/PROJECT/sonata-lite
source .venv/bin/activate
export OMP_NUM_THREADS=1
nvidia-smi
srun --ntasks=1 python -u train.py \
  --config configs/nuscenes_ssl_litept_s.json \
  --dataroot /CLUSTER/DATA/nuScenes \
  --version v1.0-trainval --split train \
  --steps 10000 --batch-size 2 --workers 4 --amp float16 \
  --output-dir runs/h200_nuscenes_ssl
```

From the repository root, create `logs` **before** submission because Slurm
opens its output files before the script body runs:

```bash
mkdir -p logs
sbatch train_h200.sbatch
```

Slurm does not move the source or dataset onto compute nodes. They must be
available at the paths used in the job. Scheduler variables are not expanded
inside `#SBATCH` directives. See the official
[sbatch reference](https://slurm.schedmd.com/sbatch.html).

## 6. Outputs, checkpointing and continuation

Each output directory contains:

| File | Contents |
| --- | --- |
| `config.json` | Effective configuration after CLI overrides |
| `metrics.jsonl` | One metrics record per consumed training batch |
| `last.pt` | Latest saved teacher, student, optimizer, scaler, RNG and data step |
| `wandb/` | Local W&B files when that logger is enabled |

The supplied configuration saves `last.pt` every 1,000 batches and at normal
completion, including completion caused by `--stop-after`. Each save replaces
the previous `last.pt` atomically. The trainer has no scheduler-preemption
signal handler: a killed job resumes from its latest completed checkpoint,
not necessarily its last processed batch. For short allocations, reduce
`save_every` in the experiment JSON before starting, or run planned chunks
with `--stop-after` while retaining the same total horizon.

```bash
python -u train.py --resume runs/h200_nuscenes_ssl/last.pt --workers 4
```

With `--resume` and no `--config`, configuration is loaded from the checkpoint.
Worker count and output directory can change. Data selection, configured
dataset path, batch size, model, schedules, optimizer, precision, gradient
clipping and drop-last behavior must remain consistent. In particular,
`--resume ... --dataroot A_DIFFERENT_PATH` is currently rejected; use the
original configured path or a symlink resolving that path to the same data.
Do not change `--steps` to extend a finished experiment through this resume
path. Restoring RNG and planned batches does not promise bitwise-identical
numerics across different GPU architectures or library versions.

Keep each experiment's output directory distinct. Starting fresh in a directory
that already has `metrics.jsonl` appends metrics while replacing the effective
configuration and, when saved, its checkpoint.

## 7. Optional Weights & Biases logging

W&B is optional and disabled by default. Install its separate pinned
dependency only when using it:

```bash
python -m pip install -r requirements-wandb.txt
python -m pip check
python -m pytest tests/test_wandb_logging.py tests/test_training_cli.py -q
```

For an online run, authenticate on a node with internet access:

```bash
wandb login
```

Alternatively provide `WANDB_API_KEY` through the cluster's secret/environment
mechanism. Keep credentials out of committed JSON files and batch scripts.
For a site-hosted W&B service, use its prescribed login/server configuration.
The official [W&B CLI implementation](https://github.com/wandb/wandb/blob/v0.31.0/wandb/cli/cli.py)
documents login and synchronization options.

Add these flags to a fresh training command for online logging:

```bash
  --wandb --wandb-mode online --wandb-project sonata-lite \
  --wandb-name h200_nuscenes_ssl
```

Use `--wandb-entity YOUR_TEAM` when selecting a particular account/team.
The logger records training scalars and the effective experiment configuration;
local JSONL metrics and `.pt` checkpoints are still written. It does not upload
the point-cloud dataset or model checkpoints as W&B artifacts.

If compute nodes cannot reach the service, choose offline mode instead:

```bash
python -u train.py \
  --config configs/nuscenes_ssl_litept_s.json \
  --dataroot /CLUSTER/DATA/nuScenes \
  --version v1.0-trainval --split train \
  --output-dir runs/h200_nuscenes_ssl_offline \
  --wandb-mode offline --wandb-project sonata-lite \
  --wandb-name h200_nuscenes_ssl_offline
```

After the job, on a connected node, authenticate and synchronize the specific
offline run directory printed by W&B:

```bash
wandb login
wandb sync runs/h200_nuscenes_ssl_offline/wandb/offline-run-TIMESTAMP-RUN_ID
```

For online continuation, the project restores the W&B run identity from its
checkpoint. Keep its project and entity unchanged when continuing the same
server run. Offline continuation can create separate local recording segments;
the explicit `train/step` axis continues across those segments. Keep them all.
With the pinned SDK, synchronize all segments for one run ID together; replace
`RUN_ID` below with the ID in their directory names:

```bash
wandb sync runs/h200_nuscenes_ssl_offline/wandb/offline-run-*-RUN_ID
```

The SDK groups files belonging to the same run and processes them in start-time
order. See `wandb sync --help` in the pinned environment. To disable W&B for an
invocation, use `--wandb-mode disabled`. Local JSONL logging remains active.

The equivalent optional configuration block is:

```json
"wandb": {
  "enabled": true,
  "project": "sonata-lite",
  "entity": null,
  "name": "h200_nuscenes_ssl",
  "mode": "offline",
  "tags": ["nuscenes", "litept-s", "h200"],
  "log_every": 10
}
```

`wandb.log_every` controls remote/offline scalar logging frequency; the local
`metrics.jsonl` keeps every consumed batch. `log_every` at the top level
controls console output. W&B logging settings can change on checkpoint resume
without changing the model's training recipe.

Local verification with the pinned W&B SDK completed three real nuScenes
updates, including an offline checkpoint continuation. The two binary W&B
recording segments contained steps `[0, 1]` and `[2]`, with every logged scalar
matching `metrics.jsonl`. Online authentication and H200 execution need to be
checked in the target environment.

## Troubleshooting

| Symptom | First check |
| --- | --- |
| `torch.cuda.is_available()` is false | Activate the project environment and run inside a GPU allocation; inspect `nvidia-smi`. |
| FlashAttention wheel is unsupported | Check Python 3.11 and `uname -m`; this wheel requires Linux x86_64. |
| `undefined symbol` during an extension import | Check the exact Torch/extension versions with `pip check`; recreate the isolated pinned environment. |
| `no kernel image` or a CUDA launch error | Run the native tests on the allocated GPU and record GPU, driver and package versions; do not infer runtime compatibility from successful imports. |
| NumPy ABI warnings or `Numpy is not available` | Confirm NumPy 1.26.4 is active and no other environment/site-packages path is being injected. |
| Teacher sparse convolution fails under AMP | Confirm the copied `modules.py` includes the evaluation adapter and run `tests/test_spconv_amp.py`; retain the FP16 baseline. |
| Missing scan/metadata files | Check `--dataroot`, the matching version/split, and paths referenced by `sample_data.json`. |
| DataLoader worker exception | Retry the same new-run settings with `--workers 0` to expose the original traceback. |
| GPU out of memory | Use a fresh run with smaller batch size or crop cap, then measure; those changes cannot be applied to an exact resume. |
| W&B cannot authenticate/connect | Configure credentials before an online batch job, or use `--wandb-mode offline`. |

For a useful failure report, save the command, `config.json`, job traceback,
`nvidia-smi` output, `python -m pip freeze`, and native-test output. The local
[RTX 3090 verification record](reports/native_training_verification.json) is a
reference for the tested recipe; H200 results should be recorded separately.
