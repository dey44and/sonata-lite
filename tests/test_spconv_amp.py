"""Native CUDA SpConv AMP preserves teacher FP32 state and student gradients."""

import copy

import pytest

torch = pytest.importorskip("torch")
spconv = pytest.importorskip("spconv.pytorch")
pytest.importorskip("addict")

from modules import PointSequential
from utils.structure import Point


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Native SpConv AMP requires CUDA"
)


def sparse_input(features):
    index = torch.arange(features.shape[0], device=features.device, dtype=torch.int32)
    coordinates = torch.stack(
        [torch.zeros_like(index), index // 8 + 2, index % 8 + 2, index % 3 + 2],
        dim=1,
    )
    return spconv.SparseConvTensor(features, coordinates, [16, 16, 16], 1)


def wrap_input(features, point):
    sparse = sparse_input(features)
    return Point(feat=features, sparse_conv_feat=sparse) if point else sparse


def result_features(result, point):
    if point:
        assert result.feat is result.sparse_conv_feat.features
        return result.feat
    return result.features


@pytest.mark.parametrize("point", [False, True], ids=["sparse", "point"])
@pytest.mark.parametrize("channels", [36, 252])
@pytest.mark.parametrize("kernel_size", [3, 5], ids=["implicit", "native"])
def test_eval_amp_matches_half_reference_without_changing_teacher_state(
    point, channels, kernel_size
):
    torch.manual_seed(17)
    teacher = PointSequential(
        torch.nn.Linear(4, channels),
        torch.nn.BatchNorm1d(channels),
        spconv.SubMConv3d(channels, channels, kernel_size, bias=True),
    ).cuda().eval()
    reference = copy.deepcopy(teacher).half()
    state = {name: value.clone() for name, value in teacher.state_dict().items()}
    raw = torch.randn(48, 4, device="cuda")

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        actual = result_features(teacher(wrap_input(raw, point)), point)
    with torch.no_grad():
        expected = result_features(reference(wrap_input(raw.half(), point)), point)

    assert actual.dtype == torch.float16
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for name, value in teacher.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)
    assert all(parameter.dtype == torch.float32 for parameter in teacher.parameters())
    assert all(not module.training for module in teacher.modules())


@pytest.mark.parametrize("point", [False, True], ids=["sparse", "point"])
@pytest.mark.parametrize("kernel_size", [3, 5], ids=["implicit", "native"])
def test_training_amp_backpropagates_through_linear_and_sparse_convolution(
    point, kernel_size
):
    torch.manual_seed(23)
    student = PointSequential(
        torch.nn.Linear(4, 36),
        spconv.SubMConv3d(36, 36, kernel_size, bias=True),
    ).cuda().train()
    raw = torch.randn(48, 4, device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=torch.float16):
        output = result_features(student(wrap_input(raw, point)), point)
        loss = output.float().square().mean()
    loss.backward()

    assert output.dtype == torch.float16
    assert raw.grad is not None and torch.isfinite(raw.grad).all()
    assert raw.grad.abs().sum() > 0
    for parameter in student.parameters():
        assert parameter.dtype == torch.float32
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
