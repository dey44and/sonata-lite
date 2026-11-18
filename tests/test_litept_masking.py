"""Exercise the real pre-embedding/stem without optional attention dependencies.

The development environment lacks timm and flash-attn. Extracting the two model
class definitions isolates this boundary while using the actual Point,
PointSequential and SpConv implementations. This is not a full-backbone test.
"""

import ast
from functools import partial
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
nn = torch.nn
spconv = pytest.importorskip("spconv.pytorch")
pytest.importorskip("addict")

from modules import PointModule, PointSequential
from utils.structure import Point


@pytest.fixture
def model_class():
    source = Path(__file__).resolve().parents[1] / "litept_v1.py"
    tree = ast.parse(source.read_text())
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in {"Embedding", "LitePT"}
    ]
    # Isolated definitions must not register a second copy of the real model.
    for node in classes:
        node.decorator_list = []
    namespace = {
        "torch": torch,
        "nn": nn,
        "spconv": spconv,
        "partial": partial,
        "Point": Point,
        "PointModule": PointModule,
        "PointSequential": PointSequential,
    }
    exec(
        compile(ast.Module(body=classes, type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace["LitePT"]


def make_model(model_class, **kwargs):
    return model_class(
        in_channels=4,
        stride=(),
        enc_depths=(0,),
        enc_channels=(36,),
        enc_num_head=(2,),
        enc_patch_size=(16,),
        enc_conv=(False,),
        enc_attn=(False,),
        enc_rope_freq=(100.0,),
        enc_mode=True,
        **kwargs,
    ).eval()


def make_point(feat, mask=None):
    grid = torch.tensor(
        [[10, 10, 10], [11, 10, 10], [10, 11, 10], [11, 11, 10]],
        dtype=torch.long,
        device=feat.device,
    )
    point = {
        "feat": feat,
        "coord": grid.float() * 0.1,
        "grid_coord": grid,
        "grid_size": 0.1,
        "offset": torch.tensor([4], device=feat.device),
    }
    if mask is not None:
        point["mask"] = mask
    return point


class StemReached(Exception):
    pass


def capture_stem_input(model, point):
    captured = {}

    def capture(_module, args):
        value = args[0]
        captured["feat"] = value.feat.detach().clone()
        captured["sparse_feat"] = value.sparse_conv_feat.features.detach().clone()
        raise StemReached

    handle = model.embedding.register_forward_pre_hook(capture)
    try:
        with pytest.raises(StemReached):
            model(point)
    finally:
        handle.remove()
    return captured


def test_masked_raw_features_cannot_reach_spatial_stem(model_class):
    model = make_model(model_class)
    with torch.no_grad():
        model.mask_token.fill_(0.25)
    feat = torch.randn(4, 4)
    mask = torch.tensor([False, True, True, False])
    changed = feat.clone()
    changed[mask] = torch.tensor([120.0, -90.0, 45.0, 0.9])

    first = capture_stem_input(model, make_point(feat, mask))
    second = capture_stem_input(model, make_point(changed, mask))

    torch.testing.assert_close(first["feat"], second["feat"], rtol=0, atol=0)
    torch.testing.assert_close(first["feat"], first["sparse_feat"], rtol=0, atol=0)
    torch.testing.assert_close(second["feat"], second["sparse_feat"], rtol=0, atol=0)
    torch.testing.assert_close(
        first["feat"][mask], model.mask_token.detach().expand(2, -1)
    )


def test_unmasked_views_keep_their_pointwise_embeddings(model_class):
    model = make_model(model_class)
    feat = torch.randn(4, 4)
    expected = model.pre_embed(Point(feat=feat.clone())).feat.detach()
    without_mask = capture_stem_input(model, make_point(feat.clone()))
    false_mask = capture_stem_input(
        model, make_point(feat.clone(), torch.zeros(4, dtype=torch.bool))
    )
    torch.testing.assert_close(without_mask["feat"], expected, rtol=0, atol=0)
    torch.testing.assert_close(false_mask["feat"], expected, rtol=0, atol=0)
    torch.testing.assert_close(without_mask["sparse_feat"], expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "mask",
    [torch.zeros(4), torch.zeros((4, 1), dtype=torch.bool)],
)
def test_mask_has_one_boolean_flag_per_point(model_class, mask):
    model = make_model(model_class)
    with pytest.raises(ValueError, match="boolean tensor shaped"):
        model(make_point(torch.randn(4, 4), mask))


def test_disabled_token_rejects_masked_inputs(model_class):
    model = make_model(model_class, mask_token=False)
    assert model.mask_token is None
    with pytest.raises(ValueError, match="mask_token=True"):
        model(make_point(torch.randn(4, 4), torch.ones(4, dtype=torch.bool)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="SpConv stem requires CUDA")
def test_actual_sparse_stem_backpropagates_to_mask_token(model_class):
    model = make_model(model_class).cuda()
    with torch.no_grad():
        model.mask_token.fill_(0.25)
    feat = torch.randn(4, 4, device="cuda", requires_grad=True)
    mask = torch.tensor([False, True, True, False], device="cuda")

    result = model(make_point(feat, mask))
    result.feat.square().sum().backward()

    assert model.mask_token.grad is not None
    assert torch.isfinite(model.mask_token.grad).all()
    assert model.mask_token.grad.abs().sum().item() > 0
    torch.testing.assert_close(feat.grad[mask], torch.zeros_like(feat.grad[mask]))
    assert feat.grad[~mask].abs().sum().item() > 0
