# Config node additions for the clogging extension of VRSP-Net
#
# Placement in repo:
#   detectron2/config/config_additions.py
#
# Call add_clogging_config(cfg) immediately after get_cfg() and before merging any YAML file
# This is done automatically in the main training script (code/train_storm_drain.py)
# This function is used to specify U-Net dimensions and override the reclassification layer to support a single class
from detectron2.config import CfgNode as CN

def add_clogging_config(cfg: CN) -> None:
    """
    Extend the default Detectron2/VRSP-Net config with nodes required by ParallelAmodalMaskHeadClogging

    New nodes
    ---------
    MODEL.ROI_MASK_HEAD.CONV_DIM : int
        Channel width of all mask-head convolutions.  Default: 256.

    MODEL.ROI_MASK_HEAD.FM_LAMBDAS : list[float]
        Weights (lambda_4, lambda_5) for the feature-matching loss.
        Default: [0.01, 0.05] — values from Xiao et al. (2021).

    MODEL.ROI_MASK_HEAD.UNET_BASE_CHANNELS : int
        Base channel width of the U-Net reconstruction network.
        Doubles at each encoder depth level.  Default: 16.

    MODEL.ROI_MASK_HEAD.UNET_DEPTH : int
        Number of encoder/decoder stages in the U-Net.
        Default: 3 (bottleneck at H/8 × W/8, i.e. 3×3 for 28×28 ROIs).

    Notes
    -----
    * No lamdba_clog node is added — the clogging loss weight is learned automatically by UncertaintyWeightedLoss (Kendall et al., 2018)
    * No auto-encoder/codebook are added — the shape prior auto-encoder from the original VRSP-Net is replaced entirely by the U-Net reconstruction network.
    * The reclassification head config nodes are intentionally omitted — the reclass head is removed for this single-class task.
    """
    cfg.MODEL.ROI_MASK_HEAD.CONV_DIM            = 256
    cfg.MODEL.ROI_MASK_HEAD.FM_LAMBDAS          = [0.01, 0.05]
    cfg.MODEL.ROI_MASK_HEAD.UNET_BASE_CHANNELS  = 16
    cfg.MODEL.ROI_MASK_HEAD.UNET_DEPTH          = 3

# Make importable directly from detectron2.config
__all__ = ["add_clogging_config"]