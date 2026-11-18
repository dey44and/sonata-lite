"""Exercise W&B CLI overrides without constructing a CUDA model."""

import json
import sys
from types import SimpleNamespace

import pytest

import train


@pytest.mark.parametrize(
    "initial,arguments,expected",
    [
        ({"enabled": False}, [], {"enabled": False}),
        ({"enabled": False}, ["--wandb"], {"enabled": True}),
        ({}, ["--wandb-mode", "offline"], {"enabled": True, "mode": "offline"}),
        ({"enabled": True}, ["--wandb-mode", "disabled"],
         {"enabled": False, "mode": "disabled"}),
        ({"enabled": False, "mode": "disabled"}, ["--wandb"],
         {"enabled": True, "mode": "online"}),
        ({}, ["--wandb-project", "research", "--wandb-entity", "lab", "--wandb-name", "h200"],
         {"enabled": True, "project": "research", "entity": "lab", "name": "h200"}),
    ],
)
def test_wandb_overrides_reach_trainer(tmp_path, monkeypatch, initial, arguments, expected):
    config = {"total_steps": 100, "dataset": {"augmentation": {}}, "wandb": initial}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    received = {}

    def build_trainer(spec):
        received.update(spec["config"])
        return SimpleNamespace(run=lambda **kwargs: None)

    monkeypatch.setitem(sys.modules, "builder", SimpleNamespace(build_trainer=build_trainer))
    train.main(["--config", str(path), *arguments])
    assert received["wandb"] == expected


def test_legacy_configuration_accepts_wandb_and_dataset_overrides(tmp_path, monkeypatch):
    config = {"total_steps": 100, "dataset": {"augmentation": {}}}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    received = {}

    def build_trainer(spec):
        received.update(spec["config"])
        return SimpleNamespace(run=lambda **kwargs: None)

    monkeypatch.setitem(sys.modules, "builder", SimpleNamespace(build_trainer=build_trainer))
    train.main(["--config", str(path), "--wandb-mode", "offline", "--dataroot", "/cluster/nuScenes"])
    assert received["dataset"]["dataroot"] == "/cluster/nuScenes"
    assert received["wandb"] == {"enabled": True, "mode": "offline"}
