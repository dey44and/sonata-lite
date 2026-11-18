"""Optional scalar logging through the public Weights & Biases Run API.

References: https://docs.wandb.ai/models/track/log/customize-logging-axes
and https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_init.py.
The SDK is imported only when logging is explicitly enabled.
"""

import copy
import importlib
import random
import warnings
from contextlib import contextmanager

import numpy as np
import torch


@contextmanager
def _preserve_training_rng():
    """Keep optional SDK operations independent of model/data randomness."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


class WandbLogger:
    """Log existing trainer scalars without making W&B a required dependency."""

    def __init__(self, config, output_dir, *, run_id=None):
        self.config = copy.deepcopy(config)
        self.options = dict(config.get("wandb") or {})
        self.mode = self.options.get("mode", "online")
        if self.mode not in ("online", "offline", "disabled"):
            raise ValueError("wandb.mode must be online, offline or disabled")
        self.enabled = bool(self.options.get("enabled", False)) and self.mode != "disabled"
        self.output_dir = output_dir
        self.run_id = run_id
        self.run = None
        self.log_every = int(self.options.get("log_every", config.get("log_every", 10)))
        if self.enabled and self.log_every < 1:
            raise ValueError("wandb.log_every must be positive")

    def start(self):
        if not self.enabled:
            return
        configured_id = self.options.get("id")
        if self.run_id is not None and configured_id not in (None, self.run_id):
            raise ValueError("wandb.id differs from the run ID saved in the checkpoint")
        with _preserve_training_rng():
            try:
                wandb = importlib.import_module("wandb")
            except ImportError as error:
                raise RuntimeError(
                    "W&B logging is enabled but wandb is not installed; "
                    "install it with 'python -m pip install wandb' or disable wandb.enabled"
                ) from error
            options = {
                "project": self.options.get("project", "sonata-lite"),
                "dir": str(self.output_dir),
                "mode": self.mode,
                "config": self.config,
            }
            for key in ("entity", "name", "tags"):
                if self.options.get(key) is not None:
                    options[key] = self.options[key]
            if self.run_id is not None or configured_id is not None:
                options["id"] = self.run_id or configured_id
            # W&B does not support resuming an offline session in place.
            if self.mode == "online":
                options["resume"] = "allow"
            self.run = wandb.init(**options)
            self.run_id = self.run.id
            self.run.define_metric("train/step")
            self.run.define_metric("train/*", step_metric="train/step")

    def log(self, metrics, *, final=False):
        if self.run is None:
            return
        if int(metrics["step"]) % self.log_every and not final:
            return
        # Custom coordinates survive a resumed SDK session or sparse logging.
        # Deliberately avoid Run.log(step=...), whose counter belongs to W&B.
        with _preserve_training_rng():
            self.run.log({f"train/{key}": value for key, value in metrics.items()})

    def finish(self, *, exit_code):
        if self.run is None:
            return
        with _preserve_training_rng():
            try:
                self.run.finish(exit_code=exit_code)
            except Exception as error:
                # A shutdown failure must not replace a training exception.
                warnings.warn(f"W&B could not finish logging: {error}", RuntimeWarning)
            finally:
                self.run = None
