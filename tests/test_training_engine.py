"""Actual optimizer/EMA/checkpoint orchestration with a tiny CPU SSL backbone."""

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn
from trainer.data import collate_ssl_views
from trainer.engine import SSLTrainer
from trainer.ssl import SSLModel


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 8)
        self.mask_token = nn.Parameter(torch.zeros(1, 8))
        self.bn = nn.BatchNorm1d(8)
        self.dropout = nn.Dropout(0.2)

    def forward(self, point):
        feat = self.linear(point["feat"])
        if "mask" in point:
            feat = torch.where(point["mask"][:, None], self.mask_token, feat)
        point["feat"] = self.dropout(self.bn(feat))
        return point


class TinyDataset:
    def __len__(self):
        return 3

    def __getitem__(self, request):
        index, step = request
        xyz = np.array([[0., 0., 0.], [.1, 0., 0.], [.2, 0., 0.]], dtype=np.float32)
        xyz[:, 1] = index
        view = {"coord": xyz, "origin_coord": xyz.copy(),
                "grid_coord": np.arange(9, dtype=np.int64).reshape(3, 3),
                "grid_size": .1,
                "feat": np.column_stack((xyz, np.ones(3, dtype=np.float32)))}
        masked = dict(view, mask=np.array([True, False, True]))
        return {"global_views": [copy.deepcopy(view) for _ in range(2)],
                "masked_global_views": [copy.deepcopy(masked) for _ in range(2)],
                "local_views": [copy.deepcopy(view) for _ in range(4)],
                "meta": {"sample_token": str(index)}, "training_step": step}


def make_trainer(path, **kwargs):
    torch.manual_seed(17)
    model = SSLModel(TinyBackbone(), head_in_channels=8, up_cast_level=0,
                     head_hidden_channels=16, head_embed_channels=6,
                     head_num_prototypes=7)
    config = {"seed": 17, "device": "cpu", "total_steps": 3,
              "batch_size": 1, "num_workers": 0, "amp": "none",
              "output_dir": str(path), "model": {}, "dataset": {},
              "optimizer": {"lr": .001}, **kwargs}
    return SSLTrainer(config, model=model, dataset=TinyDataset())


def test_real_backward_optimizer_and_ema(tmp_path):
    trainer = make_trainer(tmp_path)
    before = {k: v.clone() for k, v in trainer.model.teacher.state_dict().items()}
    result = trainer.train_step(collate_ssl_views([trainer.dataset[(0, 0)]]))
    assert result["optimizer_step"]
    assert trainer.global_step == trainer.optimizer_steps == 1
    assert result["mask_matched"] == 6
    assert result["roll_mask_matched"] == 6
    assert result["unmask_matched"] == 12
    assert all(p.grad is None for p in trainer.model.teacher.parameters())
    assert trainer.model.student.backbone.mask_token.grad.abs().sum() > 0
    for name, value in trainer.model.teacher.state_dict().items():
        student = trainer.model.student.state_dict()[name]
        expected = before[name] * .994 + student * .006 if value.is_floating_point() else student
        torch.testing.assert_close(value, expected)


def test_checkpoint_resume_matches_uninterrupted_dropout_training(tmp_path):
    uninterrupted = make_trainer(tmp_path / "full")
    uninterrupted.run()
    expected = {k: v.clone() for k, v in uninterrupted.model.state_dict().items()}
    partial = make_trainer(tmp_path / "partial")
    partial.run(stop_after=1)
    resumed = make_trainer(tmp_path / "resumed")
    resumed.load_checkpoint(partial.output_dir / "last.pt")
    resumed.run()
    assert resumed.global_step == resumed.optimizer_steps == 3
    for name, value in resumed.model.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)


def test_no_matches_does_not_update_optimizer_or_teacher(tmp_path):
    trainer = make_trainer(tmp_path)
    batch = collate_ssl_views([trainer.dataset[(0, 0)]])
    batch["masked_global"]["origin_coord"] += 100
    batch["local"]["origin_coord"] += 100
    before = {k: v.clone() for k, v in trainer.model.teacher.state_dict().items()}
    result = trainer.train_step(batch)
    assert not result["optimizer_step"]
    assert trainer.optimizer_steps == 0
    for name, value in trainer.model.teacher.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_scaler_overflow_does_not_apply_ema(tmp_path):
    trainer = make_trainer(tmp_path)

    class OverflowScaler:
        scale_value = 1024.
        def scale(self, loss): return loss
        def unscale_(self, optimizer): pass
        def get_scale(self): return self.scale_value
        def is_enabled(self): return True
        def step(self, optimizer): pass
        def update(self): self.scale_value /= 2

    trainer.scaler = OverflowScaler()
    before = {k: v.clone() for k, v in trainer.model.teacher.state_dict().items()}
    result = trainer.train_step(collate_ssl_views([trainer.dataset[(0, 0)]]))
    assert not result["optimizer_step"]
    assert trainer.optimizer_steps == 0
    for name, value in trainer.model.teacher.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_resume_rejects_changed_schedule_horizon(tmp_path):
    trainer = make_trainer(tmp_path / "first")
    path = trainer.save_checkpoint()
    changed = make_trainer(tmp_path / "changed", total_steps=4)
    with pytest.raises(ValueError, match="total_steps"):
        changed.load_checkpoint(path)


@pytest.mark.parametrize("key,value", [("drop_last", True), ("amp", "bfloat16"),
                                      ("gradient_clip", 1.0)])
def test_resume_rejects_changed_update_or_sampling_rules(tmp_path, key, value):
    trainer = make_trainer(tmp_path / "first")
    path = trainer.save_checkpoint()
    changed = make_trainer(tmp_path / "changed", **{key: value})
    with pytest.raises(ValueError, match=key):
        changed.load_checkpoint(path)


def test_model_rejects_inconsistent_view_configuration(tmp_path):
    trainer = make_trainer(tmp_path)
    batch = collate_ssl_views([trainer.dataset[(0, 0)]])
    batch["view_counts"]["global"] = 4
    with pytest.raises(ValueError, match="view counts differ"):
        trainer.train_step(batch)


def test_backbone_can_remain_fp32_while_heads_use_amp(tmp_path):
    trainer = make_trainer(tmp_path, amp="bfloat16")
    trainer.model.backbone_amp = False
    backbone_dtypes, head_dtypes = [], []
    handle1 = trainer.model.student.backbone.linear.register_forward_hook(
        lambda module, args, output: backbone_dtypes.append(output.dtype))
    handle2 = trainer.model.student.mask_head.mlp[0].register_forward_hook(
        lambda module, args, output: head_dtypes.append(output.dtype))
    try:
        result = trainer.train_step(collate_ssl_views([trainer.dataset[(0, 0)]]))
    finally:
        handle1.remove()
        handle2.remove()
    assert result["optimizer_step"]
    assert backbone_dtypes and set(backbone_dtypes) == {torch.float32}
    assert head_dtypes == [torch.bfloat16]
