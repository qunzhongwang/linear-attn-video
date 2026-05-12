"""Drop-in mmcv 1.x compat shim backed by mmengine. Inference-only."""
from mmengine import Registry, Config
from mmengine.registry import build_from_cfg
from mmengine.utils import mkdir_or_exist
from mmengine import dump, load
from . import runner, utils
__all__ = ["Registry", "Config", "build_from_cfg", "mkdir_or_exist", "dump", "load", "runner", "utils"]
