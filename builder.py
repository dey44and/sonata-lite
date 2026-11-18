"""Construct project components with the existing OpenMMLab-style registry."""

from copy import deepcopy

from registry import DATASETS, MODELS, TRAINERS, TRANSFORMS


def build_model(config):
    kind = config.get("type")
    if kind in ("LitePT", "LitePT-v1"):
        import litept_v1  # noqa: F401; registers the backbone lazily
    elif kind == "OnlineCluster":
        import cluster  # noqa: F401
    elif kind in ("SSLModel", "SonataLitePT"):
        import trainer.ssl  # noqa: F401
    return MODELS.build(deepcopy(config))


def build_transform(config):
    import augmentation

    for name in augmentation.__all__:
        cls = getattr(augmentation, name)
        if name not in TRANSFORMS:
            TRANSFORMS.register_module(name=name, module=cls)
    config = deepcopy(config)
    if isinstance(config.get("mask_schedule"), dict):
        config["mask_schedule"] = build_transform(config["mask_schedule"])
    for key in ("transforms", "global_transform", "local_transform",
                "global_shared_transform", "view_transform"):
        if isinstance(config.get(key), (list, tuple)):
            config[key] = [build_transform(item) if isinstance(item, dict) else item
                           for item in config[key]]
        elif isinstance(config.get(key), dict):
            config[key] = [build_transform(config[key])]
    return TRANSFORMS.build(config)


def build_dataset(config):
    from trainer.data import NuScenesSSLDataset

    if "NuScenesSSLDataset" not in DATASETS:
        DATASETS.register_module(module=NuScenesSSLDataset)
    config = deepcopy(config)
    if isinstance(config.get("augmentation"), dict):
        config["augmentation"] = build_transform(config["augmentation"])
    return DATASETS.build(config)


def build_trainer(config):
    from trainer.engine import SSLTrainer  # noqa: F401

    return TRAINERS.build(deepcopy(config))
