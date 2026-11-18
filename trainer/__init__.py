"""Training components, imported lazily to keep dataset-only use lightweight."""

from importlib import import_module

_EXPORTS = {
    "SSLTrainer": ("engine", "SSLTrainer"),
    "SSLModel": ("ssl", "SSLModel"),
    "NuScenesSSLDataset": ("data", "NuScenesSSLDataset"),
    "StepBatchSampler": ("data", "StepBatchSampler"),
    "collate_ssl_views": ("data", "collate_ssl_views"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(f".{module}", __name__), attribute)
    globals()[name] = value
    return value
