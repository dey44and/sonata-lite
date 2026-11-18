"""Single-GPU Sonata-style pretraining, with resumable planned data steps.

Reference orchestration: Pointcept Sonata v1m1 (commit 96e109d).
The optimizer, checkpoint format, and LitePT BN-buffer EMA are project adapters.
"""

import copy
import json
import random
import warnings
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from builder import build_dataset, build_model
from registry import TRAINERS
from .data import StepBatchSampler, collate_ssl_views, seed_worker
from .logging import WandbLogger
from .optim import layerwise_parameter_groups
from .schedules import ScalarSchedule


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


@TRAINERS.register_module()
class SSLTrainer:
    def __init__(self, config, *, model=None, dataset=None):
        self.config = copy.deepcopy(config)
        self.total_steps = int(config["total_steps"])
        self.seed = int(config.get("seed", 42))
        seed_everything(self.seed)
        self.device = torch.device(config.get("device", "cuda"))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for LitePT's sparse convolutions")
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            raise RuntimeError("SSLTrainer currently supports one process/GPU; DDP is not wired yet")
        self.model = (build_model(config["model"]) if model is None else model).to(self.device)
        self.dataset = build_dataset(config["dataset"]) if dataset is None else dataset
        optimizer_cfg = dict(config.get("optimizer", {}))
        optimizer_type = optimizer_cfg.pop("type", "AdamW")
        if optimizer_type != "AdamW":
            raise ValueError("The training recipe currently supports AdamW")
        optimizer_cfg.setdefault("lr", 1e-4)
        optimizer_cfg.setdefault("weight_decay", 0.04)
        layer_decay = optimizer_cfg.pop("layer_decay", None)
        parameters = (p for p in self.model.student.parameters() if p.requires_grad)
        if layer_decay is not None:
            parameters = layerwise_parameter_groups(
                self.model.student, optimizer_cfg["lr"], layer_decay
            )
        self.optimizer = torch.optim.AdamW(parameters, **optimizer_cfg)
        lr_cfg = dict(config.get("lr_schedule", {}))
        schedule_type = lr_cfg.pop("type", "ScalarSchedule")
        self.lr_scheduler = None
        self.lr_schedule = None
        if schedule_type == "OneCycleLR":
            # Construction installs step-zero LR and Adam beta1. The scheduler
            # advances only after successful optimizer updates, matching the
            # official trainer's AMP-overflow guard. Data curricula use step.
            self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=[group["lr"] for group in self.optimizer.param_groups],
                total_steps=self.total_steps, **lr_cfg,
            )
        elif schedule_type == "ScalarSchedule":
            self.lr_schedule = ScalarSchedule(
                self.total_steps, optimizer_cfg["lr"],
                final_value=lr_cfg.get("final_value", 1e-6),
                start_value=lr_cfg.get("start_value", 1e-6),
                warmup_ratio=lr_cfg.get("warmup_ratio", 0.05),
            )
        else:
            raise ValueError(f"Unsupported lr_schedule.type: {schedule_type}")
        wd_cfg = config.get("weight_decay_schedule")
        self.weight_decay_schedule = None
        if wd_cfg is not None:
            wd_cfg = dict(wd_cfg)
            wd_cfg.setdefault("base_value", optimizer_cfg["weight_decay"])
            self.weight_decay_schedule = ScalarSchedule(self.total_steps, **wd_cfg)
        self.temperature_schedule = ScalarSchedule(
            self.total_steps, **config.get("teacher_temperature", {
                "base_value": 0.07, "start_value": 0.04, "warmup_ratio": 0.05,
            })
        )
        self.momentum_schedule = ScalarSchedule(
            self.total_steps, **config.get("teacher_momentum", {
                "base_value": 0.994, "final_value": 1.0,
            })
        )
        self.amp = config.get("amp", "float16")
        if self.amp not in ("none", "float16", "bfloat16"):
            raise ValueError("amp must be none, float16 or bfloat16")
        if self.device.type == "cpu" and self.amp == "float16":
            raise ValueError("CPU smoke models require amp=none or bfloat16")
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.device.type == "cuda" and self.amp == "float16"
        )
        self.global_step = 0
        self.optimizer_steps = 0
        self.lr_scheduler_steps = 0
        self.wandb_run_id = None
        self.output_dir = Path(config.get("output_dir", "runs/nuscenes_ssl_litept_s"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def make_loader(self):
        sampler = StepBatchSampler(
            len(self.dataset), self.config.get("batch_size", 2), self.total_steps,
            seed=self.seed, start_step=self.global_step,
            drop_last=self.config.get("drop_last", False),
        )
        workers = int(self.config.get("num_workers", 4))
        options = {}
        if workers:
            options["prefetch_factor"] = self.config.get("prefetch_factor", 2)
            options["persistent_workers"] = True
        # Loader seed draws must not advance the model's dropout RNG on resume.
        generator = torch.Generator().manual_seed(self.seed)
        return DataLoader(
            self.dataset, batch_sampler=sampler, num_workers=workers,
            collate_fn=collate_ssl_views, worker_init_fn=seed_worker,
            generator=generator, pin_memory=self.device.type == "cuda", **options,
        )

    def train_step(self, batch):
        step = int(batch["step"])
        if step != self.global_step:
            raise ValueError(f"Expected planned step {self.global_step}, received {step}")
        if self.lr_schedule is not None:
            lr = self.lr_schedule(step)
            for group in self.optimizer.param_groups:
                group["lr"] = lr * group.get("lr_scale", 1.0)
        if self.weight_decay_schedule is not None:
            weight_decay = self.weight_decay_schedule(step)
            for group in self.optimizer.param_groups:
                group["weight_decay"] = weight_decay
        # Record values used for this update, before advancing OneCycleLR.
        learning_rates = [group["lr"] for group in self.optimizer.param_groups]
        lr = max(learning_rates)
        weight_decay = self.optimizer.param_groups[0]["weight_decay"]
        beta1 = self.optimizer.param_groups[0]["betas"][0]
        temperature = self.temperature_schedule(step)
        momentum = self.momentum_schedule(step)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        batch = move_to_device(batch, self.device)
        context = nullcontext() if self.amp == "none" else torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16 if self.amp == "float16" else torch.bfloat16,
        )
        with context:
            result = self.model(batch, teacher_temp=temperature)
            loss = result["loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite SSL loss at step {step}")
        match_counts = [v for key, v in result.items() if key.endswith("_matched")]
        has_matches = not match_counts or sum(int(v) for v in match_counts) > 0
        stepped = False
        if has_matches:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.student.parameters(), self.config.get("gradient_clip", 3.0)
            )
            old_scale = self.scaler.get_scale()
            if self.scaler.is_enabled():
                self.scaler.step(self.optimizer)
                self.scaler.update()
                stepped = self.scaler.get_scale() >= old_scale
            else:
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError(f"Non-finite gradient at step {step}")
                self.optimizer.step()
                stepped = True
            if stepped:
                self.model.update_teacher(momentum)
                self.optimizer_steps += 1
        else:
            grad_norm = loss.new_zeros(())
        self.global_step = step + 1
        if self.lr_scheduler is not None and stepped:
            self.lr_scheduler.step()
            self.lr_scheduler_steps += 1
        metrics = {key: float(value.detach()) if isinstance(value, torch.Tensor)
                   else float(value) for key, value in result.items()}
        metrics.update(step=step, lr=lr, teacher_temp=temperature, ema_momentum=momentum,
                       lr_min=min(learning_rates), lr_max=max(learning_rates),
                       weight_decay=weight_decay, adam_beta1=beta1,
                       grad_norm=float(grad_norm), optimizer_step=stepped,
                       optimizer_steps=self.optimizer_steps,
                       lr_scheduler_steps=self.lr_scheduler_steps)
        return metrics

    def save_checkpoint(self, path=None):
        path = Path(path) if path is not None else self.output_dir / "last.pt"
        state = {
            "format_version": 3, "config": self.config,
            "global_step": self.global_step, "optimizer_steps": self.optimizer_steps,
            "lr_scheduler_steps": self.lr_scheduler_steps,
            "wandb": {"run_id": self.wandb_run_id},
            "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        torch.save(state, temporary)
        temporary.replace(path)
        return path

    def load_checkpoint(self, path):
        state = torch.load(path, map_location="cpu", weights_only=False)
        # Schedule horizons and sampling must remain stable on resume.
        for key in ("total_steps", "seed", "batch_size", "model", "dataset",
                    "lr_schedule", "teacher_temperature", "teacher_momentum", "optimizer",
                    "weight_decay_schedule"):
            if state["config"].get(key) != self.config.get(key):
                raise ValueError(f"Resume configuration differs for {key}")
        for key, default in (("drop_last", False), ("amp", "float16"),
                             ("gradient_clip", 3.0)):
            if state["config"].get(key, default) != self.config.get(key, default):
                raise ValueError(f"Resume configuration differs for {key}")
        scheduler_state = state.get("lr_scheduler")
        scheduler_steps = 0
        if self.lr_scheduler is not None:
            if scheduler_state is None:
                raise ValueError("OneCycleLR checkpoint is missing lr_scheduler state")
            legacy_clock = "lr_scheduler_steps" not in state
            if legacy_clock and state.get("format_version", 1) >= 3:
                raise ValueError("OneCycleLR checkpoint is missing lr_scheduler_steps")
            # Legacy format 2 stepped on consumed batches. Preserve its saved
            # LR/beta1 position rather than rewinding previous optimizer work.
            scheduler_steps = state["global_step"] if legacy_clock else state["lr_scheduler_steps"]
            if (not isinstance(scheduler_steps, int)
                    or not 0 <= state["optimizer_steps"] <= scheduler_steps <= state["global_step"] <= self.total_steps
                    or scheduler_state.get("last_epoch") != scheduler_steps):
                raise ValueError("OneCycleLR state differs from the checkpoint's scheduler-step counter")
            if legacy_clock and scheduler_steps > state["optimizer_steps"]:
                warnings.warn(
                    "Legacy checkpoint advanced OneCycleLR on skipped updates; "
                    "preserving its saved LR position. Future skipped updates "
                    "will not advance the scheduler.", RuntimeWarning,
                )
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(scheduler_state)
        self.lr_scheduler_steps = scheduler_steps
        self.scaler.load_state_dict(state["scaler"])
        self.global_step = state["global_step"]
        self.optimizer_steps = state["optimizer_steps"]
        self.wandb_run_id = (state.get("wandb") or {}).get("run_id")
        rng = state["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if rng["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])

    def run(self, *, stop_after=None):
        (self.output_dir / "config.json").write_text(json.dumps(self.config, indent=2) + "\n")
        stop = self.total_steps if stop_after is None else min(
            self.total_steps, self.global_step + stop_after
        )
        if self.global_step >= stop:
            return []
        history = []
        log_every = int(self.config.get("log_every", 10))
        save_every = int(self.config.get("save_every", 1000))
        logger = WandbLogger(self.config, self.output_dir, run_id=self.wandb_run_id)
        succeeded = False
        try:
            logger.start()
            self.wandb_run_id = logger.run_id
            loader = self.make_loader()
            for batch in loader:
                metrics = self.train_step(batch)
                with (self.output_dir / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(metrics) + "\n")
                if metrics["step"] % log_every == 0 or self.global_step == stop:
                    print(json.dumps(metrics), flush=True)
                logger.log(metrics, final=self.global_step == stop)
                history.append(metrics)
                if self.global_step % save_every == 0:
                    self.save_checkpoint()
                if self.global_step >= stop:
                    break
            self.save_checkpoint()
            succeeded = True
            return history
        finally:
            logger.finish(exit_code=0 if succeeded else 1)
