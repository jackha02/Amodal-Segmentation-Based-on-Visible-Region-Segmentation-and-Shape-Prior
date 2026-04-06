# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .box_head import ROI_BOX_HEAD_REGISTRY, build_box_head
from .keypoint_head import ROI_KEYPOINT_HEAD_REGISTRY, build_keypoint_head
from .mask_head import ROI_MASK_HEAD_REGISTRY, build_mask_head
from .roi_heads import (
    ROI_HEADS_REGISTRY,
    ROIHeads,
    StandardROIHeads,
    build_roi_heads,
    select_foreground_proposals,
)

# For a model adapted for clogging extent estimation 
from .uncertainty_weighting import UncertaintyWeightedLoss
from .clogging_loss import clogging_extent_loss, soft_iou
from .unet_reconstruction import UNetReconstruction
from .mask_head_clogging import ParallelAmodalMaskHeadClogging


from .rotated_fast_rcnn import RROIHeads

from . import cascade_rcnn  # isort:skip

# All the functions will be called, when the roi_heads module is imported 
__all__ = [
    "UncertaintyWeightedLoss",
    "clogging_extent_loss",
    "soft_iou",
    "UNetReconstruction",
    "ParallelAmodalMaskHeadClogging",
]