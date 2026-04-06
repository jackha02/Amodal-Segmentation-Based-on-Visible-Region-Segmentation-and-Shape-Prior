# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from .utils.env import setup_environment

setup_environment()

# Import C++/CUDA extension
try:
    from ._C import *
except ModuleNotFoundError:
    pass  # _C not compiled; pure-Python fallback paths will be used

__version__ = "0.1"
