"""Normalization must not couple unrelated rows in packed point clouds."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("spconv")
pytest.importorskip("torch_scatter")
pytest.importorskip("addict")

from litept_v1 import LitePT


@pytest.mark.parametrize("enc_mode", [True, False])
def test_normalization_is_independent_of_other_packed_rows(enc_mode):
    model = LitePT(enc_mode=enc_mode, enable_flash=False)
    assert not any(
        isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
        for module in model.modules()
    )
    assert not any(
        name.endswith(("running_mean", "running_var", "num_batches_tracked"))
        for name in model.state_dict()
    )
    norms = [
        module for module in model.modules()
        if isinstance(module, torch.nn.LayerNorm)
    ]
    assert norms
    generator = torch.Generator().manual_seed(42)
    with torch.no_grad():
        for norm in norms:
            channels = norm.normalized_shape[0]
            features = torch.randn(3, channels, generator=generator)
            unrelated = torch.randn(7, channels, generator=generator) * 10 + 50
            expected = norm(features)
            packed = norm(torch.cat((features, unrelated)))[: len(features)]
            torch.testing.assert_close(packed, expected)
            norm.eval()
            torch.testing.assert_close(norm(features), expected)
