"""Official optimizer schedules, block grouping, skipped batches, and resume."""

import copy
import warnings

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from trainer.engine import SSLTrainer
from trainer.optim import layerwise_parameter_groups


class RecipeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.student = nn.ModuleDict({
            "backbone": nn.ModuleDict({
                "stem": nn.Linear(1, 1),
                "enc": nn.ModuleDict({
                    "enc0": nn.ModuleDict({"block0": nn.Linear(1, 1),
                                           "block1": nn.Linear(1, 1)}),
                    "enc1": nn.ModuleDict({"down": nn.Linear(1, 1),
                                           "block0": nn.Linear(1, 1)}),
                }),
            }),
            "head": nn.Linear(1, 1),
        })
        self.teacher = copy.deepcopy(self.student).requires_grad_(False)

    def forward(self, batch, teacher_temp):
        # Randomness verifies checkpoint RNG restoration as well as optimizer
        # moments, cycling beta1, block LR scales, and the teacher EMA.
        loss = sum(p.square().sum() for p in self.student.parameters()) * torch.rand(())
        return {"loss": loss, "test_matched": int(batch.get("matches", True))}

    @torch.no_grad()
    def update_teacher(self, momentum):
        for teacher, student in zip(self.teacher.parameters(), self.student.parameters()):
            teacher.mul_(momentum).add_(student, alpha=1 - momentum)


def make_recipe(path, **changes):
    torch.manual_seed(17)
    model = RecipeModel()
    config = {
        "seed": 17, "device": "cpu", "total_steps": 40, "amp": "none",
        "batch_size": 1, "num_workers": 0, "model": {}, "dataset": {},
        "output_dir": str(path),
        "optimizer": {"type": "AdamW", "lr": .004, "weight_decay": .04,
                      "layer_decay": .9},
        "lr_schedule": {"type": "OneCycleLR", "pct_start": .05,
                        "div_factor": 10, "final_div_factor": 1000,
                        "anneal_strategy": "cos"},
        "weight_decay_schedule": {"final_value": .2, "warmup_ratio": 0},
    }
    config.update(changes)
    return SSLTrainer(config, model=model, dataset=range(3))


def test_layer_groups_cover_each_parameter_once_and_keep_other_modules_at_base_lr():
    model = RecipeModel()
    groups = layerwise_parameter_groups(model.student, .004, .9)
    assert [g["lr_scale"] for g in groups] == pytest.approx([1, .81, .9, 1])
    grouped = [id(p) for group in groups for p in group["params"]]
    assert len(grouped) == len(set(grouped))
    assert set(grouped) == {id(p) for p in model.student.parameters()}
    base_ids = {id(p) for p in groups[0]["params"]}
    assert id(model.student["head"].weight) in base_ids
    assert id(model.student["backbone"]["stem"].weight) in base_ids
    assert id(model.student["backbone"]["enc"]["enc1"]["down"].weight) in base_ids
    assert all(group["lr"] == pytest.approx(.004 * group["lr_scale"]) for group in groups)


def test_onecycle_values_and_beta1_match_pytorch_and_wd_matches_cosine(tmp_path):
    trainer = make_recipe(tmp_path)
    params = [nn.Parameter(torch.ones(())) for _ in trainer.optimizer.param_groups]
    reference = torch.optim.AdamW(
        [{"params": [p], "lr": .004 * g["lr_scale"]}
         for p, g in zip(params, trainer.optimizer.param_groups)], lr=.004,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        reference, max_lr=[g["lr"] for g in reference.param_groups], total_steps=40,
        pct_start=.05, div_factor=10, final_div_factor=1000, anneal_strategy="cos",
    )
    for step in range(40):
        expected_lrs = [group["lr"] for group in reference.param_groups]
        expected_beta = reference.param_groups[0]["betas"][0]
        result = trainer.train_step({"step": step})
        assert result["lr_max"] == pytest.approx(max(expected_lrs))
        assert result["lr_min"] == pytest.approx(min(expected_lrs))
        assert result["adam_beta1"] == pytest.approx(expected_beta)
        expected_wd = .2 + .5 * (.04 - .2) * (1 + torch.cos(torch.tensor(torch.pi * step / 40)).item())
        assert result["weight_decay"] == pytest.approx(expected_wd, abs=1e-8)
        reference.step()
        scheduler.step()
    assert result["lr_max"] == pytest.approx(.004 / 10000)
    assert trainer.lr_scheduler.last_epoch == trainer.global_step == 40


def test_no_match_batch_advances_curriculum_without_optimizer_or_teacher_update(tmp_path):
    trainer = make_recipe(tmp_path)
    before = {name: value.clone() for name, value in trainer.model.state_dict().items()}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = trainer.train_step({"step": 0, "matches": False})
    assert not caught
    assert not result["optimizer_step"]
    assert trainer.optimizer_steps == 0
    assert trainer.lr_scheduler.last_epoch == trainer.global_step == 1
    for name, value in trainer.model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    result = trainer.train_step({"step": 1})
    assert result["lr"] == pytest.approx(.004)
    assert result["adam_beta1"] == pytest.approx(.85)


def test_onecycle_resume_matches_uninterrupted_training_after_a_skipped_batch(tmp_path):
    full = make_recipe(tmp_path / "full")
    for step in range(12):
        full.train_step({"step": step, "matches": step != 2})
    expected = {name: value.clone() for name, value in full.model.state_dict().items()}
    partial = make_recipe(tmp_path / "partial")
    for step in range(5):
        partial.train_step({"step": step, "matches": step != 2})
    checkpoint = partial.save_checkpoint()
    resumed = make_recipe(tmp_path / "resume")
    resumed.load_checkpoint(checkpoint)
    for step in range(5, 12):
        resumed.train_step({"step": step})
    assert resumed.global_step == 12
    assert resumed.optimizer_steps == 11
    assert resumed.lr_scheduler.state_dict() == full.lr_scheduler.state_dict()
    for actual, reference in zip(resumed.optimizer.param_groups, full.optimizer.param_groups):
        assert actual["lr"] == reference["lr"]
        assert actual["betas"] == reference["betas"]
        assert actual["weight_decay"] == reference["weight_decay"]
    for name, value in resumed.model.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)


def test_resume_rejects_changed_weight_decay_schedule(tmp_path):
    trainer = make_recipe(tmp_path / "before")
    checkpoint = trainer.save_checkpoint()
    changed = make_recipe(tmp_path / "changed", weight_decay_schedule={"final_value": .1})
    with pytest.raises(ValueError, match="weight_decay_schedule"):
        changed.load_checkpoint(checkpoint)


def test_legacy_baseline_checkpoint_without_scheduler_still_loads(tmp_path):
    changes = {"optimizer": {"lr": .001}, "lr_schedule": {},
               "weight_decay_schedule": None}
    trainer = make_recipe(tmp_path / "before", **changes)
    trainer.train_step({"step": 0})
    checkpoint = trainer.save_checkpoint()
    state = torch.load(checkpoint, weights_only=False)
    state.pop("lr_scheduler")
    state["config"].pop("weight_decay_schedule")
    state["format_version"] = 1
    torch.save(state, checkpoint)
    resumed = make_recipe(tmp_path / "after", **changes)
    resumed.load_checkpoint(checkpoint)
    assert resumed.global_step == resumed.optimizer_steps == 1
    assert resumed.lr_scheduler is None
    assert resumed.train_step({"step": 1})["optimizer_step"]
