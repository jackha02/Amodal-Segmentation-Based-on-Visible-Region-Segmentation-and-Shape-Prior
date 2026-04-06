# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from .utils.env import setup_environment

setup_environment()

# Import C++/CUDA extension
from ._C import *

__version__ = "0.1"
