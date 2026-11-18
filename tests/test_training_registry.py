"""Config construction must use one shared project registry."""

import pytest

from builder import build_transform
from registry import Registry, TRANSFORMS


def test_unknown_qualified_type_has_clean_lookup_failure():
    registry = Registry("example", scope="example")
    assert registry.get("missing.Object") is None
    with pytest.raises(KeyError, match="missing.Object"):
        registry.build({"type": "missing.Object"})


def test_registry_builds_nested_mask_schedule_without_modifying_config():
    config = {"type": "NuScenesSSLViews", "mask_schedule": {
        "type": "MaskSchedule", "total_steps": 1000}}
    transform = build_transform(config)
    assert transform.mask_generator.mask_schedule(0) == (.1, .3)
    assert config["mask_schedule"]["type"] == "MaskSchedule"
    assert "NuScenesSSLViews" in TRANSFORMS


def test_registry_builds_nested_transform_chains():
    transform = build_transform({"type": "NuScenesSSLViews",
        "global_transform": [{"type": "RandomFlip", "p": 0.5}],
        "local_transform": [{"type": "Compose", "transforms": [
            {"type": "RandomScale", "scale": [1, 1]}]}]})
    assert callable(transform.generator.global_transform)
    assert callable(transform.generator.local_transform)
