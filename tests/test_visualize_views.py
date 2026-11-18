"""Browser payload correspondence and the local viewer's HTTP integration."""

import json
import random
from threading import Thread
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
import pytest

from visualize_views import create_server, main, make_payload


def example_views():
    original = {
        "coord": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        "origin_coord": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        "strength": np.array([[0.25], [0.75]], dtype=np.float32),
    }
    augmented = {
        "coord": np.array([[10, 20, 30]], dtype=np.float32),
        "origin_coord": original["origin_coord"][[1]],
        "strength": original["strength"][[1]],
        "grid_size": 0.1,
    }
    return {
        "meta": {"sample_token": "example", "coordinate_frame": "LIDAR_TOP"},
        "global_views": [augmented, augmented],
        "local_views": [augmented] * 4,
    }, original


def test_payload_keeps_exact_view_points_and_original_correspondences():
    views, original = example_views()
    payload = json.loads(json.dumps(make_payload(views, original=original, seed=42)))
    assert len(payload["views"]) == 7
    assert payload["meta"]["seed"] == 42
    assert payload["meta"]["grid_size"] == 0.1
    np.testing.assert_array_equal(payload["views"][0]["coord"], original["coord"])
    for encoded in payload["views"][1:]:
        assert encoded["count"] == 1
        assert encoded["coord"] == [[10.0, 20.0, 30.0]]
        assert encoded["origin_coord"] == [[4.0, 5.0, 6.0]]
        assert encoded["strength"] == [0.75]
    # Preparing the display leaves the supplied metadata and arrays untouched.
    assert "seed" not in views["meta"]
    np.testing.assert_array_equal(original["coord"], [[1, 2, 3], [4, 5, 6]])
    assert len(make_payload(views)["views"]) == 6


def test_payload_includes_actual_masks_without_regenerating_or_removing_points():
    views, original = example_views()
    views["masked_global_views"] = []
    for mask in ([True, False], [False, True]):
        views["masked_global_views"].append(
            {
                **original,
                "mask": np.array(mask, dtype=bool),
                "mask_size": 0.4,
                "mask_ratio": 0.3,
            }
        )
    payload = json.loads(json.dumps(make_payload(views, original=original)))
    assert len(payload["views"]) == 9
    assert [view["kind"] for view in payload["views"]] == [
        "original",
        "global",
        "global",
        "masked_global",
        "masked_global",
        "local",
        "local",
        "local",
        "local",
    ]
    for encoded, expected in zip(payload["views"][3:5], views["masked_global_views"]):
        assert encoded["count"] == len(original["coord"])
        np.testing.assert_array_equal(encoded["coord"], expected["coord"])
        np.testing.assert_array_equal(encoded["origin_coord"], expected["origin_coord"])
        assert encoded["mask"] == expected["mask"].tolist()
        assert all(type(value) is bool for value in encoded["mask"])
        assert encoded["mask_size"] == 0.4
        assert encoded["mask_ratio"] == 0.3
    assert "mask" not in payload["views"][1]
    np.testing.assert_array_equal(
        views["masked_global_views"][0]["mask"], [True, False]
    )


def test_payload_includes_scheduled_step_and_effective_masking_values():
    views, original = example_views()
    views["masked_global_views"] = [
        {
            **original,
            "mask": np.array([True, False]),
            "mask_size": 0.25,
            "mask_ratio": 0.5,
            "mask_step": np.int64(25),
            "mask_total_steps": np.int64(1000),
        }
    ]
    encoded = json.loads(json.dumps(make_payload(views)))["views"][2]
    assert encoded["mask_step"] == 25
    assert encoded["mask_total_steps"] == 1000
    assert encoded["mask_size"] == 0.25
    assert encoded["mask_ratio"] == 0.5
    assert "mask_step" not in make_payload(views)["views"][0]


@pytest.mark.parametrize("presample, expected_count", [(True, 3), (False, 12)])
def test_shared_baseline_preserves_crop_points_masks_and_random_state(
    presample, expected_count
):
    from augmentation import NuScenesSSLViews

    coord = np.zeros((12, 3), dtype=np.float32)
    coord[:, 0] = np.r_[np.linspace(0.001, 0.045, 10), 1.01, 2.01]
    original = {
        "coord": coord,
        "origin_coord": coord.copy(),
        "strength": np.ones((12, 1), dtype=np.float32),
        "point_id": np.arange(12),
    }
    item = {"point": original, "meta": {"sample_token": "shared-baseline"}}

    random.seed(42)
    np.random.seed(42)
    plain = NuScenesSSLViews(presample=presample)(item)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    random.seed(42)
    np.random.seed(42)
    captured = NuScenesSSLViews(presample=presample)(item, return_shared=True)

    assert random.getstate() == python_state
    for actual, expected in zip(np.random.get_state(), numpy_state):
        np.testing.assert_equal(actual, expected)
    assert "shared_input" not in plain
    assert captured["meta"] == plain["meta"]
    for group in ("global_views", "local_views", "masked_global_views"):
        for actual, expected in zip(captured[group], plain[group]):
            assert actual.keys() == expected.keys()
            for key in expected:
                np.testing.assert_equal(actual[key], expected[key])

    shared = captured["shared_input"]
    assert len(shared["coord"]) == expected_count
    np.testing.assert_array_equal(shared["coord"], coord[shared["point_id"]])
    for group in ("global_views", "local_views"):
        for view in captured[group]:
            assert np.isin(view["point_id"], shared["point_id"]).all()
    if presample:
        assert len(np.unique(shared["grid_coord"], axis=0)) == expected_count
    payload = make_payload(captured, original=original)
    baseline = next(view for view in payload["views"] if view["kind"] == "shared")
    assert baseline["count"] == expected_count
    np.testing.assert_array_equal(baseline["coord"], shared["coord"])
    assert payload["views"][0]["name"] == "Original keyframe (raw)"
    if presample:
        assert baseline["grid_size"] == 0.1
        assert "0.1 m voxel sampled" in baseline["name"]
    else:
        assert "presampling disabled" in baseline["name"]


@pytest.mark.parametrize(
    "argv, expected_fixed, expected_step",
    [
        ([], (0.4, 0.3), None),
        (["--mask-size", "0.2", "--mask-ratio", "0.6"], (0.2, 0.6), None),
        (["--total-steps", "1000"], None, 0),
        (["--total-steps", "1000", "--step", "50"], None, 50),
    ],
)
def test_cli_forwards_fixed_or_progressive_masking_without_starting_server(
    monkeypatch, argv, expected_fixed, expected_step
):
    import augmentation
    import datasets
    import visualize_views

    calls = {}
    schedule = object()
    views, original = example_views()
    views["masked_global_views"] = []
    views["shared_input"] = original
    item = {"point": original}

    def make_schedule(total_steps):
        calls["total_steps"] = total_steps
        return schedule

    class FakeViews:
        def __init__(self, **kwargs):
            calls["config"] = kwargs

        def __call__(self, data, **kwargs):
            assert data is item
            calls["generate"] = kwargs
            return views

    class FakeReader:
        sample_tokens = ["sample"]

        def __init__(self, *args, **kwargs):
            pass

        def __len__(self):
            return 1

        def read(self, token):
            assert token == "sample"
            return item

    def fake_serve(data, **kwargs):
        assert data is views
        assert kwargs["original"] is original
        assert not kwargs["open_browser"]
        calls["served"] = True

    monkeypatch.setattr(augmentation, "NuScenesSSLViews", FakeViews)
    monkeypatch.setattr(
        augmentation, "MaskSchedule", make_schedule, raising=False
    )
    monkeypatch.setattr(datasets, "NuScenesKeyframeReader", FakeReader)
    monkeypatch.setattr(visualize_views, "serve_views", fake_serve)
    main([*argv, "--no-browser"])
    assert calls["served"]
    if expected_fixed is not None:
        assert calls["config"] == {
            "grid_size": 0.1,
            "mask_size": expected_fixed[0],
            "mask_ratio": expected_fixed[1],
        }
        assert calls["generate"] == {"return_shared": True}
        assert "total_steps" not in calls
    else:
        assert calls["total_steps"] == 1000
        assert calls["config"] == {"grid_size": 0.1, "mask_schedule": schedule}
        assert calls["generate"] == {"step": expected_step, "return_shared": True}


@pytest.mark.parametrize(
    "argv",
    [
        ["--step", "10"],
        ["--total-steps", "1000", "--mask-size", "0.4"],
        ["--total-steps", "1000", "--mask-ratio", "0.3"],
    ],
)
def test_cli_rejects_ambiguous_mask_configuration(argv):
    with pytest.raises(SystemExit) as error:
        main(argv)
    assert error.value.code == 2


def test_local_server_supplies_data_and_offline_modules_without_exposing_project():
    views, original = example_views()
    server = create_server(views, original=original, port=0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(f"{url}/api/views", timeout=5) as response:
            assert response.headers.get_content_type() == "application/json"
            assert json.load(response)["views"][1]["name"] == "Global 0 (principal)"
        for path, content_type in (
            ("/", "text/html"),
            ("/viewer.js", "text/javascript"),
            ("/vendor/three.module.min.js", "text/javascript"),
            ("/vendor/three.core.min.js", "text/javascript"),
            ("/vendor/OrbitControls.js", "text/javascript"),
        ):
            with urlopen(url + path, timeout=5) as response:
                assert response.headers.get_content_type() == content_type
                assert response.read()
        with pytest.raises(HTTPError) as error:
            urlopen(f"{url}/../datasets/nuscenes.py", timeout=5)
        assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
