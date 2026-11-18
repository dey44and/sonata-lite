"""Real Torch attention fallback preserves serialized view/window boundaries."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("spconv")
pytest.importorskip("torch_scatter")
pytest.importorskip("addict")
from litept_v1 import PointROPEAttention
from utils.structure import Point


def point(feat):
    result = Point(feat=feat, offset=torch.tensor([5, 7]),
                   grid_coord=torch.tensor([[i, 0, 0] for i in range(7)]))
    result.serialization(order=["z"], shuffle_orders=False)
    return result


def test_fallback_matches_explicit_attention_and_backpropagates():
    torch.manual_seed(9)
    attention = PointROPEAttention(36, 2, 3, 100., enable_flash=False).eval()
    feat = torch.randn(7, 36, requires_grad=True)
    value = point(feat)
    pad, unpad, cu = attention.get_padding_and_inverse(value)
    order = value.serialized_order[0][pad]
    inverse = unpad[value.serialized_inverse[0]]
    q, k, v = attention.qkv(feat)[order].chunk(3, dim=-1)
    positions = value.grid_coord[order][None]
    q = attention.rope(q.reshape(-1, 2, 18).transpose(0, 1)[None], positions)[0]
    k = attention.rope(k.reshape(-1, 2, 18).transpose(0, 1)[None], positions)[0]
    v = v.reshape(-1, 2, 18).transpose(0, 1)
    parts = []
    bounds = cu.tolist()
    for begin, end in zip(bounds[:-1], bounds[1:]):
        weights = (q[:, begin:end] @ k[:, begin:end].transpose(-1, -2)) * attention.scale
        parts.append((weights.softmax(-1) @ v[:, begin:end]).transpose(0, 1).reshape(-1, 36))
    expected = attention.proj(torch.cat(parts)[inverse])
    result = attention(value).feat
    torch.testing.assert_close(result, expected)
    result.square().sum().backward()
    assert torch.isfinite(feat.grad).all()
    assert feat.grad.abs().sum() > 0


def test_attention_does_not_mix_distinct_views():
    torch.manual_seed(9)
    attention = PointROPEAttention(36, 2, 3, 100., enable_flash=False).eval()
    feat = torch.randn(7, 36)
    first = attention(point(feat.clone())).feat
    changed = feat.clone()
    changed[5:] += 100
    second = attention(point(changed)).feat
    torch.testing.assert_close(first[:5], second[:5], rtol=0, atol=0)
