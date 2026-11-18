"""CUDA integration checks for LitePT's real FlashAttention and timm layers."""

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("Native attention integration requires CUDA", allow_module_level=True)

pytest.importorskip("spconv")
pytest.importorskip("torch_scatter")
pytest.importorskip("addict")
pytest.importorskip("flash_attn")
timm_layers = pytest.importorskip("timm.layers")

import litept_v1
from utils.structure import Point


# Both views need padding at patch_size=16. Their coordinate ranges overlap, so
# their packed offsets, rather than spatial separation, must keep them apart.
VIEW_COUNTS = (37, 23)


def _point(features):
    positions = torch.cat([
        torch.stack((
            torch.arange(count, device="cuda"),
            torch.arange(count, device="cuda") * 7 % 13,
            torch.arange(count, device="cuda") * 5 % 11,
        ), dim=1)
        for count in VIEW_COUNTS
    ])
    value = Point(
        feat=features,
        offset=torch.tensor(VIEW_COUNTS, device="cuda").cumsum(0),
        grid_coord=positions,
    )
    value.serialization(order=["z"], shuffle_orders=False)
    return value


def _attention(enable_flash):
    # LitePT-S uses 18 channels per attention head, which FlashAttention pads
    # internally. Exercising this dimension catches unsupported-kernel paths.
    return litept_v1.PointROPEAttention(
        channels=252,
        num_heads=14,
        patch_size=16,
        rope_freq=100.0,
        enable_flash=enable_flash,
    ).cuda().eval()


def test_drop_path_uses_native_timm():
    assert litept_v1.DropPath is timm_layers.DropPath


def test_native_flash_matches_sdpa_and_backpropagates(monkeypatch):
    torch.manual_seed(23)
    native = _attention(enable_flash=True)
    reference = _attention(enable_flash=False)
    reference.load_state_dict(native.state_dict())
    features = torch.randn(sum(VIEW_COUNTS), 252, device="cuda", requires_grad=True)

    real_flash = litept_v1.flash_attn.flash_attn_varlen_qkvpacked_func
    calls = []

    def record_real_flash(qkv, cu_seqlens, **kwargs):
        calls.append((qkv.shape, qkv.dtype, cu_seqlens.detach().cpu()))
        return real_flash(qkv, cu_seqlens, **kwargs)

    monkeypatch.setattr(
        litept_v1.flash_attn, "flash_attn_varlen_qkvpacked_func", record_real_flash
    )
    with torch.autocast("cuda", dtype=torch.float16):
        actual = native(_point(features)).feat
        expected = reference(_point(features.detach().clone())).feat

    # The spy delegates to the real extension. A silent SDPA substitution would
    # fail this assertion even if its numerical output happened to agree.
    assert len(calls) == 1
    shape, dtype, boundaries = calls[0]
    assert shape[-3:] == (3, 14, 18)
    assert dtype == torch.float16
    assert boundaries.tolist() == [0, 16, 32, 48, 64, 80]
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-3)

    actual.float().square().mean().backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert features.grad.abs().sum() > 0
    for parameter in native.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_native_flash_preserves_view_boundaries_in_forward_and_backward():
    torch.manual_seed(29)
    attention = _attention(enable_flash=True)
    features = torch.randn(sum(VIEW_COUNTS), 252, device="cuda", requires_grad=True)
    changed = features.detach().clone()
    changed[VIEW_COUNTS[0]:] += 10.0

    with torch.autocast("cuda", dtype=torch.float16):
        original = attention(_point(features)).feat
        perturbed = attention(_point(changed)).feat

    torch.testing.assert_close(
        original[:VIEW_COUNTS[0]], perturbed[:VIEW_COUNTS[0]], rtol=0, atol=0
    )
    original[:VIEW_COUNTS[0]].float().square().mean().backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert features.grad[:VIEW_COUNTS[0]].abs().sum() > 0
    assert torch.count_nonzero(features.grad[VIEW_COUNTS[0]:]) == 0
