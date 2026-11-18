"""Numerical, correspondence and gradient checks without sparse GPU dependencies."""

from datetime import timedelta
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from trainer.losses import (
    match_nearest,
    matched_distillation_loss,
    sinkhorn_knopp,
    up_cast,
)
from trainer.ssl import SSLModel


def reference_sinkhorn(scores, temperature, iterations=3):
    """Official single-process Sonata formula, independent of our helper."""
    q = torch.exp(scores.float() / temperature).t()
    n, k = scores.shape
    q /= q.sum()
    for _ in range(iterations):
        q = q / q.sum(dim=1, keepdim=True) / k
        q = q / q.sum(dim=0, keepdim=True) / n
    return (q * n).t()


def test_sinkhorn_matches_official_formula_and_stops_teacher_gradients():
    scores = torch.randn(17, 5, generator=torch.Generator().manual_seed(4), requires_grad=True)
    actual = sinkhorn_knopp(scores, 0.7)
    torch.testing.assert_close(actual, reference_sinkhorn(scores.detach(), 0.7))
    assert not actual.requires_grad
    torch.testing.assert_close(actual.sum(dim=1), torch.ones(17))


def test_sinkhorn_converges_to_balanced_mass_and_handles_large_logits():
    scores = torch.randn(23, 5, generator=torch.Generator().manual_seed(5))
    target = sinkhorn_knopp(scores, 0.5, num_iter=100)
    torch.testing.assert_close(target.sum(dim=0), torch.full((5,), 23 / 5))
    shifted = sinkhorn_knopp(scores + 10000, 0.5)
    assert torch.isfinite(shifted).all()
    torch.testing.assert_close(shifted.sum(dim=1), torch.ones(23))


def test_sinkhorn_empty_input():
    result = sinkhorn_knopp(torch.empty(0, 7), 0.04)
    assert result.shape == (0, 7)


def test_matching_is_many_to_one_and_radius_is_strict():
    student = torch.tensor([[0.01, 0, 0], [0.02, 0, 0], [1.0, 0, 0], [3.0, 0, 0]])
    teacher = torch.tensor([[0.0, 0, 0], [2.0, 0, 0]])
    pairs, view = match_nearest(
        student, torch.tensor([4]), teacher, torch.tensor([2]), [0], 1.0,
        query_chunk_size=1, reference_chunk_size=1,
    )
    assert pairs.tolist() == [[0, 0], [1, 0]]
    assert view.tolist() == [0, 0]


def test_matching_never_crosses_packed_views_and_can_swap_within_scene():
    coords = torch.tensor([[0.0, 0, 0], [10.0, 0, 0], [20.0, 0, 0], [30.0, 0, 0]])
    offset = torch.tensor([1, 2, 3, 4])
    # Same coordinates in a different scene/view must not be used implicitly.
    pairs, _ = match_nearest(coords.flip(0), offset, coords, offset, [0, 1, 2, 3], 0.1)
    assert len(pairs) == 0
    pairs, view = match_nearest(coords, offset, coords, offset, [1, 0, 3, 2], 10.1)
    assert pairs.tolist() == [[0, 1], [1, 0], [2, 3], [3, 2]]
    assert view.tolist() == [0, 1, 2, 3]


def test_upcast_gathers_all_three_levels_and_preserves_fine_metadata():
    fine = {"feat": torch.tensor([[1.0], [2.0], [3.0], [4.0]], requires_grad=True),
            "origin_coord": torch.randn(4, 3), "offset": torch.tensor([4])}
    middle = {"feat": torch.tensor([[10.0, 11.0], [20.0, 21.0]], requires_grad=True),
              "pooling_parent": fine, "pooling_inverse": torch.tensor([0, 0, 1, 1])}
    coarse = {"feat": torch.tensor([[100.0, 101.0, 102.0]], requires_grad=True),
              "pooling_parent": middle, "pooling_inverse": torch.tensor([0, 0])}
    output = up_cast(coarse, 2)
    assert output is fine
    assert output["feat"].tolist() == [
        [1, 10, 11, 100, 101, 102], [2, 10, 11, 100, 101, 102],
        [3, 20, 21, 100, 101, 102], [4, 20, 21, 100, 101, 102],
    ]
    output["feat"].sum().backward()
    torch.testing.assert_close(coarse["feat"].grad, torch.full((1, 3), 4.0))
    assert output["offset"].tolist() == [4]


def test_loss_balances_matched_teacher_rows_then_means_within_views():
    student_coord = torch.tensor([[0.0, 0, 0], [0.01, 0, 0], [10.0, 0, 0]])
    teacher_coord = torch.tensor([[0.0, 0, 0], [10.0, 0, 0], [100.0, 0, 0]])
    student = torch.tensor([[2.0, -1], [-0.5, 1], [0.2, -0.3]], requires_grad=True)
    teacher = torch.tensor([[1.0, 0], [-0.2, 0.7], [10.0, -10]], requires_grad=True)
    result = matched_distillation_loss(
        student, teacher,
        {"origin_coord": student_coord, "offset": torch.tensor([2, 3])},
        {"origin_coord": teacher_coord, "offset": torch.tensor([1, 3])},
        [0, 1], teacher_temp=0.6, student_temp=0.8, match_max_r=0.1,
    )
    # Teacher row 0 is gathered twice; the unmatched row 2 never enters SK.
    targets = reference_sinkhorn(teacher.detach()[[0, 0, 1]], 0.6)
    per_row = -(targets * torch.log_softmax(student / 0.8, dim=-1)).sum(-1)
    expected = (per_row[:2].mean() + per_row[2]) / 2
    torch.testing.assert_close(result.loss, expected)
    assert result.matched.item() == 3
    assert result.matched_views.item() == 2
    result.loss.backward()
    assert teacher.grad is None
    assert student.grad is not None and student.grad.abs().sum() > 0


def test_no_overlap_returns_finite_differentiable_zero():
    student = torch.randn(3, 4, requires_grad=True)
    result = matched_distillation_loss(
        student, torch.randn(2, 4),
        {"origin_coord": torch.zeros(3, 3), "offset": torch.tensor([3])},
        {"origin_coord": torch.full((2, 3), 100.0), "offset": torch.tensor([2])},
        [0], teacher_temp=0.04,
    )
    assert result.loss.item() == 0
    assert result.matched.item() == result.matched_views.item() == 0
    result.loss.backward()
    torch.testing.assert_close(student.grad, torch.zeros_like(student))


class TinyBackbone(nn.Module):
    """Real differentiable hierarchy, mask replacement and BatchNorm state."""

    enc_mode = True

    def __init__(self):
        super().__init__()
        self.input = nn.Linear(4, 2)
        self.bn = nn.BatchNorm1d(2)
        self.middle = nn.Linear(2, 3)
        self.coarse = nn.Linear(3, 4)
        self.mask_token = nn.Parameter(torch.zeros(1, 2))

    def _pool(self, point, layer):
        ends = [0] + point["offset"].tolist()
        feats, coords, inverse, offsets = [], [], [], []
        total = 0
        for first, last in zip(ends[:-1], ends[1:]):
            assert (last - first) % 2 == 0
            n = (last - first) // 2
            feats.append(layer(point["feat"][first:last]).reshape(n, 2, -1).mean(1))
            coords.append(point["origin_coord"][first:last].reshape(n, 2, 3).mean(1))
            inverse.append(torch.arange(total, total + n).repeat_interleave(2))
            total += n
            offsets.append(total)
        return {
            "feat": torch.cat(feats), "origin_coord": torch.cat(coords),
            "offset": torch.tensor(offsets), "pooling_parent": point,
            "pooling_inverse": torch.cat(inverse),
        }

    def forward(self, data):
        feat = self.input(data["feat"])
        if "mask" in data:
            feat = torch.where(data["mask"][:, None], self.mask_token, feat)
        point = {"feat": self.bn(feat), "origin_coord": data["origin_coord"],
                 "offset": data["offset"]}
        return self._pool(self._pool(point, self.middle), self.coarse)


def make_model(**kwargs):
    return SSLModel(
        backbone=TinyBackbone(), head_in_channels=9, head_hidden_channels=16,
        head_embed_channels=6, head_num_prototypes=8,
        num_local_view=2, match_max_r=0.3, **kwargs,
    )


def make_batch():
    def packed(views):
        coords = torch.cat([
            torch.tensor([[0.0, 0, 0], [0.1, 0, 0], [0.2, 0, 0], [0.3, 0, 0]])
            + torch.tensor([scene * 20.0, 0, 0])
            for scene in range(2) for _ in range(views)
        ])
        return {"feat": torch.cat([coords, torch.ones(len(coords), 1)], dim=1),
                "origin_coord": coords, "offset": torch.arange(1, 2 * views + 1) * 4}
    globals_ = packed(2)
    masked = {key: value.clone() for key, value in globals_.items()}
    masked["mask"] = torch.arange(len(masked["feat"])) % 2 == 0
    return {"global": globals_, "masked_global": masked, "local": packed(2)}


def test_full_ssl_forward_backward_freezes_teacher_and_trains_both_heads():
    torch.manual_seed(8)
    model = make_model().train()
    teacher_buffers = {name: value.clone() for name, value in model.teacher.named_buffers()}
    result = model(make_batch(), teacher_temp=0.07)
    assert torch.isfinite(result["loss"]) and result["loss"].item() > 0
    for name in ("mask", "roll_mask", "unmask"):
        assert result[name + "_matched"].item() == 16
        assert result[name + "_matched_views"].item() == 4
    torch.testing.assert_close(
        result["loss"], .25 * result["mask_loss"] + .25 * result["roll_mask_loss"]
        + .5 * result["unmask_loss"],
    )
    result["loss"].backward()
    assert all(parameter.grad is None for parameter in model.teacher.parameters())
    assert all(not parameter.requires_grad for parameter in model.teacher.parameters())
    for name in ("mask_head", "unmask_head"):
        assert model.student[name].mlp[0].weight.grad.abs().sum() > 0
    assert model.student["backbone"].mask_token.grad.abs().sum() > 0
    assert not model.teacher.training
    for name, value in model.teacher.named_buffers():
        torch.testing.assert_close(value, teacher_buffers[name])
    assert model.student["backbone"].bn.num_batches_tracked.item() == 2


def test_teacher_ema_updates_parameters_and_float_buffers_copies_integer_state():
    model = make_model()
    with torch.no_grad():
        for parameter in model.student.parameters():
            parameter.fill_(3)
        for parameter in model.teacher.parameters():
            parameter.fill_(1)
        model.student["backbone"].bn.running_mean.fill_(6)
        model.teacher["backbone"].bn.running_mean.fill_(2)
        model.student["backbone"].bn.num_batches_tracked.fill_(7)
    model.update_teacher(0.75)
    for parameter in model.teacher.parameters():
        torch.testing.assert_close(parameter, torch.full_like(parameter, 1.5))
    torch.testing.assert_close(model.teacher["backbone"].bn.running_mean, torch.full((2,), 3.0))
    assert model.teacher["backbone"].bn.num_batches_tracked.item() == 7
    model.eval().train()
    assert model.student.training and not model.teacher.training


def test_full_model_no_overlap_has_zero_loss_and_supports_backward():
    model = make_model()
    batch = make_batch()
    batch["masked_global"]["origin_coord"] += 1000
    batch["local"]["origin_coord"] += 1000
    result = model(batch)
    assert result["loss"].item() == 0
    result["loss"].backward()
    assert model.student["mask_head"].mlp[0].weight.grad is not None


def test_local_matching_uses_only_principal_global_from_same_scene():
    model = make_model()
    batch = make_batch()
    for start in (4, 12):
        batch["global"]["origin_coord"][start : start + 4] += torch.tensor([5.0, 0, 0])
        batch["masked_global"]["origin_coord"][start : start + 4] += torch.tensor([5.0, 0, 0])
    batch["local"]["origin_coord"] += torch.tensor([5.0, 0, 0])
    result = model(batch)
    assert result["mask_matched"].item() == 16
    assert result["roll_mask_matched"].item() == 0
    assert result["unmask_matched"].item() == 0


def test_masked_only_training_allows_no_local_views():
    model = SSLModel(
        backbone=TinyBackbone(), head_in_channels=9, head_hidden_channels=8,
        head_embed_channels=6, head_num_prototypes=4, num_local_view=0,
        unmask_loss_weight=0,
    )
    batch = make_batch()
    batch["local"] = None
    result = model(batch)
    result["loss"].backward()
    assert "unmask_head" not in model.student
    assert result["unmask_loss"].item() == 0


def _distributed_sinkhorn_worker(rank, filename):
    import torch.distributed as dist

    dist.init_process_group(
        "gloo", init_method="file://" + filename, rank=rank, world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        scores = torch.randn(7, 4, generator=torch.Generator().manual_seed(19))
        expected = reference_sinkhorn(scores, 0.7)
        for lengths in ((2, 5), (0, 7)):
            start = sum(lengths[:rank])
            end = start + lengths[rank]
            actual = sinkhorn_knopp(scores[start:end], 0.7)
            torch.testing.assert_close(actual, expected[start:end])
        assert sinkhorn_knopp(scores[:0], 0.7).shape == (0, 4)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_available(), reason="requires torch.distributed")
def test_distributed_sinkhorn_handles_unequal_counts_and_empty_ranks(tmp_path):
    torch.multiprocessing.spawn(
        _distributed_sinkhorn_worker, args=(str(Path(tmp_path) / "sinkhorn.store"),),
        nprocs=2, join=True,
    )
