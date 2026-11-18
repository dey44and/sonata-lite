"""Optional W&B lifecycle and checkpoint behavior, without network access."""

import builtins
import copy
import json
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from trainer.logging import WandbLogger
from test_training_engine import make_trainer


class FakeRun:
    def __init__(self, run_id):
        self.id = run_id
        self.metrics = []
        self.definitions = []
        self.exit_codes = []

    @staticmethod
    def consume_randomness():
        random.random()
        np.random.random()
        torch.rand(1)

    def define_metric(self, *args, **kwargs):
        self.definitions.append((args, kwargs))
        self.consume_randomness()

    def log(self, metrics):
        self.metrics.append(copy.deepcopy(metrics))
        self.consume_randomness()

    def finish(self, *, exit_code):
        self.exit_codes.append(exit_code)
        self.consume_randomness()


@pytest.fixture
def sdk(monkeypatch):
    calls, runs = [], []

    def init(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        run = FakeRun(kwargs.get("id", "generated-id"))
        runs.append(run)
        run.consume_randomness()
        return run

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init))
    return SimpleNamespace(calls=calls, runs=runs)


@pytest.mark.parametrize("options", [{}, {"enabled": False},
                                     {"enabled": True, "mode": "disabled"}])
def test_disabled_logging_never_imports_sdk(tmp_path, monkeypatch, options):
    real_import = builtins.__import__

    def import_without_wandb(name, *args, **kwargs):
        if name == "wandb" or name.startswith("wandb."):
            raise AssertionError("Disabled logging must not import the SDK")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_wandb)
    monkeypatch.delitem(sys.modules, "wandb", raising=False)
    logger = WandbLogger({"wandb": options}, tmp_path)
    logger.start()
    logger.log({"step": 0, "loss": 1.0})
    logger.finish(exit_code=0)
    assert logger.run is None


def test_numeric_metrics_custom_step_and_offline_options(tmp_path, sdk):
    config = {"log_every": 2, "wandb": {
        "enabled": True, "mode": "offline", "project": "test-project",
        "entity": "team", "name": "experiment", "tags": ["test"], "id": "explicit-id",
    }}
    logger = WandbLogger(config, tmp_path)
    logger.start()
    logger.log({"step": 0, "loss": 2.0, "optimizer_steps": 1})
    logger.log({"step": 1, "loss": 1.5})
    logger.log({"step": 3, "loss": 1.0}, final=True)
    logger.finish(exit_code=0)
    options = sdk.calls[0]
    assert options == {
        "config": config, "project": "test-project", "entity": "team",
        "name": "experiment", "tags": ["test"], "id": "explicit-id",
        "dir": str(tmp_path), "mode": "offline",
    }
    assert sdk.runs[0].definitions == [(("train/step",), {}),
        (("train/*",), {"step_metric": "train/step"})]
    assert sdk.runs[0].metrics == [
        {"train/step": 0, "train/loss": 2.0, "train/optimizer_steps": 1},
        {"train/step": 3, "train/loss": 1.0},
    ]
    assert sdk.runs[0].exit_codes == [0]


def test_training_metrics_and_checkpoint_resume_keep_run_id_and_rng(tmp_path, sdk):
    uninterrupted = make_trainer(tmp_path / "unlogged")
    uninterrupted.run()
    expected = {key: value.clone() for key, value in uninterrupted.model.state_dict().items()}
    options = {"enabled": True, "project": "test-project", "log_every": 1}
    partial = make_trainer(tmp_path / "partial", wandb=options)
    history = partial.run(stop_after=1)
    checkpoint = partial.output_dir / "last.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert state["wandb"] == {"run_id": "generated-id"}
    assert sdk.runs[0].metrics == [{f"train/{key}": value for key, value in history[0].items()}]
    assert json.loads((partial.output_dir / "metrics.jsonl").read_text()) == history[0]
    resumed = make_trainer(tmp_path / "resumed", wandb=options)
    resumed.load_checkpoint(checkpoint)
    resumed.run()
    assert sdk.calls[1]["id"] == "generated-id"
    assert sdk.calls[1]["resume"] == "allow"
    assert [entry["train/step"] for entry in sdk.runs[1].metrics] == [1, 2]
    assert all(run.exit_codes == [0] for run in sdk.runs)
    for name, value in resumed.model.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)


def test_old_checkpoint_loads_without_logging_metadata(tmp_path, sdk):
    original = make_trainer(tmp_path / "original")
    path = original.save_checkpoint()
    state = torch.load(path, map_location="cpu", weights_only=False)
    state.pop("wandb")
    torch.save(state, path)
    resumed = make_trainer(tmp_path / "resumed", wandb={"enabled": True})
    resumed.load_checkpoint(path)
    resumed.run(stop_after=1)
    assert "id" not in sdk.calls[0]


def test_training_exception_finishes_sdk_run(tmp_path, sdk, monkeypatch):
    trainer = make_trainer(tmp_path, wandb={"enabled": True})

    def fail(batch):
        raise FloatingPointError("deliberate training failure")

    monkeypatch.setattr(trainer, "train_step", fail)
    with pytest.raises(FloatingPointError, match="deliberate"):
        trainer.run()
    assert sdk.runs[0].exit_codes == [1]


def test_conflicting_checkpoint_run_id_is_rejected(tmp_path, sdk):
    logger = WandbLogger({"wandb": {"enabled": True, "id": "different"}},
                         tmp_path, run_id="saved-id")
    with pytest.raises(ValueError, match="checkpoint"):
        logger.start()
    assert not sdk.calls


def test_enabled_logging_reports_missing_optional_dependency(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)
    logger = WandbLogger({"wandb": {"enabled": True}}, tmp_path)
    with pytest.raises(RuntimeError, match="pip install wandb"):
        logger.start()
