# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

def configurable(func):
    """
    Minimal configurable decorator for compatibility.
    In Detectron2, this is used for config-based instantiation.
    This version is a no-op and simply returns the function unchanged.
    """
    return func

from .compat import downgrade_config, upgrade_config
from .config import CfgNode, get_cfg, global_cfg, set_global_cfg
from .config_additions import add_clogging_config

__all__ = [
    "CfgNode",
    "get_cfg",
    "global_cfg",
    "set_global_cfg",
    "downgrade_config",
    "upgrade_config",
    "add_clogging_config",
    "configurable",
]
