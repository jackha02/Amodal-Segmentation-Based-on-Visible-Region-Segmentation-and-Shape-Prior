# =============================================================================
# mask_head_clogging.py
#
# Modified parallel amodal/visible mask head for storm drain clogging extent
# estimation.  This file should REPLACE (or be registered in place of) the
# existing mask head used by the full VRSP-Net model, i.e. the head invoked
# by the config:
#
#   configs/.../mask_rcnn_R_50_FPN_1x_parallel_CtRef_VAR_SPRef_SPRet_FM.yaml
#
# Placement in repo:
#   detectron2/modeling/roi_heads/mask_head_clogging.py
#
# Changes relative to the original VRSP-Net mask head (Xiao et al., 2021)
# ------------------------------------------------------------------------
# 1. Reclassification head (L_rc) REMOVED — single-class task.
#    Precedent: Kim et al. (2023), Computers and Electronics in Agriculture.
#
# 2. Auto-encoder + codebook shape prior REPLACED with U-Net reconstruction.
#    Kim et al. (2023) demonstrated that for single-class objects with high
#    intraclass shape variance, U-Net outperforms the auto-encoder baseline
#    (AP 50.06 vs 49.31) at faster inference speed (220 ms vs 233 ms).
#    Storm drain inlets share the same physical geometry but can appear very
#    different across viewpoints and scales, motivating the same substitution.
#    The U-Net input is the coarse amodal mask M^c_a; the output replaces
#    the k nearest shape-prior masks M^k_sp from the original Module 3.
#
# 3. Clogging extent loss (L_clog) ADDED — MAE between predicted and
#    ground-truth clogging ratios, gated by the soft amodal IoU quality
#    weight w_i (see clogging_loss.py for full derivation).
#
# 4. UncertaintyWeightedLoss wrapper ADDED around all learnable loss terms
#    (L^c_a, L^c_v, L^r_a, L^r_v, L_afm, L_vfm, L_unet_recon, L_clog),
#    eliminating all manual lambda hyperparameters for these terms.
#    Reference: Kendall, Gal & Cipolla, CVPR 2018.
#
# 5. Detection-head losses (L_cls, L_reg) are NOT wrapped — they belong to
#    the RPN/box head and are returned unchanged, consistent with standard
#    Detectron2 practice.
#
# Pipeline overview (three modules from Xiao et al., 2021)
# ---------------------------------------------------------
#   Module 1 — Coarse Mask Segmentation
#       Input : ROI feature F
#       Output: M^c_a (coarse amodal), M^c_v (coarse visible)
#       Losses: L^c_a (BCE), L^c_v (BCE)
#
#   Module 2 — Visible Mask Segmentation
#       Input : F * M^c_a  (cross-task attention, scheme (d) from Table 3)
#       Output: M^r_v (refined visible)
#       Losses: L^r_v (BCE), L_vfm (cosine feature matching)
#
#   Module 3 — Amodal Mask Segmentation  ← MODIFIED
#       Input : F * M^r_v  (visible-region attention)
#               U-Net(M^c_a)  (replaces auto-encoder shape prior)
#               cat(F * M^r_v, U-Net(M^c_a))
#       Output: M^r_a (refined amodal)
#       Losses: L^r_a (BCE), L_afm (cosine feature matching),
#               L_unet_recon (BCE between U-Net output and GT amodal)
#
#   Post-Module 3 — Clogging Extent  ← NEW
#       Inputs : M^r_a, M^r_v, GT masks
#       Loss   : L_clog (amodal-IoU-gated MAE)
# =============================================================================

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ConvTranspose2d, cat, interpolate
from detectron2.modeling.roi_heads.mask_head import ROI_MASK_HEAD_REGISTRY
from detectron2.utils.events import get_event_storage

# Local modules (same package directory as this file)
from .uncertainty_weighting import UncertaintyWeightedLoss
from .clogging_loss import clogging_extent_loss
from .unet_reconstruction import UNetReconstruction


# ---------------------------------------------------------------------------
# Helper: feature-matching loss (cosine similarity, Eq. 3 / Eq. 5 of paper)
# ---------------------------------------------------------------------------

def _feature_matching_loss(
    feats_plain:   list,            # [f^(j)(F_i)]        j = 4, 5
    feats_attn:    list,            # [f^(j)(F_i * M)]    j = 4, 5
    lambdas: tuple = (0.01, 0.05), # lambda_4, lambda_5 from Xiao et al.
) -> torch.Tensor:
    """
    Cosine-similarity feature matching loss between feature maps from the
    plain ROI feature F and the attention-masked feature F * M, evaluated
    at conv layers j = {4, 5} of the mask head.

    See Eq. 3 (visible head) and Eq. 5 (amodal head) in Xiao et al. (2021).
    Loss = 1 - cosine_similarity, so 0 is a perfect match.
    """
    assert len(feats_plain) == len(feats_attn) == len(lambdas)
    loss = feats_plain[0].new_zeros(1).squeeze()
    for f_plain, f_attn, lam in zip(feats_plain, feats_attn, lambdas):
        fp = f_plain.view(f_plain.shape[0], -1)   # (N, C*H*W)
        fa = f_attn.view(f_attn.shape[0], -1)
        cos_sim = F.cosine_similarity(fp, fa, dim=1).mean()
        loss = loss + lam * (1.0 - cos_sim)
    return loss


# ---------------------------------------------------------------------------
# Basic mask head block shared by amodal and visible heads
# ---------------------------------------------------------------------------

class _MaskHeadBlock(nn.Module):
    """
    4 × Conv2d(3×3, ReLU) + 1 × ConvTranspose2d(2×2, stride=2).

    Mirrors the mask head architecture in VRSP-Net.  The forward_with_features
    method additionally returns the intermediate activations at layers 3 and 4
    (0-indexed), corresponding to layers j=4 and j=5 in the paper's 1-indexed
    notation, required by the feature-matching loss.
    """

    def __init__(self, in_channels: int, num_classes: int, conv_dim: int = 256):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(4):
            self.convs.append(
                Conv2d(
                    in_channels if i == 0 else conv_dim,
                    conv_dim,
                    kernel_size=3,
                    padding=1,
                    activation=nn.ReLU(),
                )
            )
        self.deconv   = ConvTranspose2d(conv_dim, conv_dim, kernel_size=2, stride=2)
        self.out_conv = Conv2d(conv_dim, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = conv(x)
        x = self.deconv(x)
        return self.out_conv(x)

    def forward_with_features(self, x: torch.Tensor):
        """
        Returns (logits, [feat_j4, feat_j5]) where feat_j4 and feat_j5 are
        the activations after the 3rd and 4th conv layers (0-indexed 2 and 3),
        used for the feature-matching loss.
        """
        feats = []
        for i, conv in enumerate(self.convs):
            x = conv(x)
            if i in (2, 3):          # paper's j=4, j=5
                feats.append(x)
        x = self.deconv(x)
        return self.out_conv(x), feats   # (logits, [feat_j4, feat_j5])


# ---------------------------------------------------------------------------
# Main head — registered with Detectron2's mask head registry
# ---------------------------------------------------------------------------

@ROI_MASK_HEAD_REGISTRY.register()

class ParallelAmodalMaskHeadClogging(nn.Module):
    """
    Parallel amodal + visible mask head extended with:

      - U-Net reconstruction network replacing the auto-encoder shape prior
        (Kim et al., 2023)
      - Clogging extent loss L_clog (MAE, quality-gated by amodal IoU)
      - Automatic uncertainty-based loss weighting (Kendall et al., 2018)
      - No reclassification head (single-class task)
    """

    def __init__(self, cfg, input_shape):
        """
        Standard Detectron2 mask head constructor: (cfg, input_shape)
        """
        args = type(self).from_config(cfg, input_shape)
        super().__init__()
        in_channels       = args["input_shape"].channels
        num_classes       = args["num_classes"]
        conv_dim          = args["conv_dim"]
        fm_lambdas        = args["fm_lambdas"]
        unet_base_channels = args["unet_base_channels"]
        unet_depth        = args["unet_depth"]
        self.num_classes  = num_classes
        self.fm_lambdas   = fm_lambdas

        # ---- Coarse mask heads (Module 1) ----------------------------------
        self.coarse_amodal_head  = _MaskHeadBlock(in_channels, num_classes, conv_dim)
        self.coarse_visible_head = _MaskHeadBlock(in_channels, num_classes, conv_dim)

        # ---- Refined visible head (Module 2) --------------------------------
        # Receives F * M^c_a.  Weights shared with coarse visible head
        # (parameter sharing described in Xiao et al., §"Feature Matching").
        self.refined_visible_head = self.coarse_visible_head   # shared weights

        # ---- U-Net reconstruction network (replaces auto-encoder, Module 3) -
        # Input:  coarse amodal mask M^c_a  (N, 1, H, W)
        # Output: same-size refined amodal logit  (N, 1, H, W)
        self.unet = UNetReconstruction(
            base_channels=unet_base_channels,
            depth=unet_depth,
        )

        # ---- Refined amodal head (Module 3) ---------------------------------
        # Input: cat(F * M^r_v,  U-Net(M^c_a))
        # The concatenation adds 1 channel (U-Net output) to the ROI features.
        # in_channels + 1 accounts for this.
        self.refined_amodal_head = _MaskHeadBlock(
            in_channels + 1, num_classes, conv_dim
        )

        # ---- Uncertainty weighting (Kendall et al., 2018) ------------------
        # Task order must match the ordered list in _uncertainty_loss().
        # 'cls' for BCE-type losses, 'reg' for the MAE clogging ratio loss.
        self.uncertainty_weighter = UncertaintyWeightedLoss(
            task_types=[
                "cls",   # L^c_a        — coarse amodal BCE
                "cls",   # L^c_v        — coarse visible BCE
                "cls",   # L^r_v        — refined visible BCE
                "cls",   # L_vfm        — visible feature matching
                "cls",   # L_unet_recon — U-Net reconstruction BCE
                "cls",   # L^r_a        — refined amodal BCE
                "cls",   # L_afm        — amodal feature matching
                "reg",   # L_clog       — clogging extent MAE
            ]
        )

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "input_shape":         input_shape,
            "num_classes":         cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "conv_dim":            cfg.MODEL.ROI_MASK_HEAD.CONV_DIM,
            "fm_lambdas":          tuple(cfg.MODEL.ROI_MASK_HEAD.FM_LAMBDAS),
            "unet_base_channels":  cfg.MODEL.ROI_MASK_HEAD.UNET_BASE_CHANNELS,
            "unet_depth":          cfg.MODEL.ROI_MASK_HEAD.UNET_DEPTH,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bce_mask_loss(
        pred_logits: torch.Tensor,
        gt_masks:    torch.Tensor,
    ) -> torch.Tensor:
        """Binary cross-entropy mask loss, averaged over instances and pixels."""
        if gt_masks.shape[-2:] != pred_logits.shape[-2:]:
            gt_masks = interpolate(
                gt_masks.float(),
                size=pred_logits.shape[-2:],
                mode="nearest",
            )
        return F.binary_cross_entropy_with_logits(
            pred_logits, gt_masks.float(), reduction="mean"
        )

    def _uncertainty_loss(self, named_losses: dict) -> torch.Tensor:
        """
        Wraps the eight learnable losses with uncertainty weighting.

        named_losses must contain exactly these keys (order matters):
            'loss_mask_coarse_amodal'
            'loss_mask_coarse_visible'
            'loss_mask_refined_visible'
            'loss_vfm'
            'loss_unet_recon'
            'loss_mask_refined_amodal'
            'loss_afm'
            'loss_clog'
        """
        ordered = [
            named_losses["loss_mask_coarse_amodal"],
            named_losses["loss_mask_coarse_visible"],
            named_losses["loss_mask_refined_visible"],
            named_losses["loss_vfm"],
            named_losses["loss_unet_recon"],
            named_losses["loss_mask_refined_amodal"],
            named_losses["loss_afm"],
            named_losses["loss_clog"],
        ]
        return self.uncertainty_weighter(ordered)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        roi_features: torch.Tensor,   # (N, C, H, W)  ROIAlign output
        instances:    list,           # list[Instances]
    ):
        if self.training:
            return self._forward_train(roi_features, instances)
        else:
            return self._forward_inference(roi_features, instances)

    # ------------------------------------------------------------------

    def _forward_train(
        self,
        F: torch.Tensor,
        instances: list,
    ) -> dict:
        """
        Runs the three-module pipeline and returns the combined loss dict.

        Ground-truth masks are stored as PolygonMasks (or BitMasks) on each
        Instances object and are cropped + resized to the ROI output size here:
            instances[i].gt_masks_amodal   — PolygonMasks, amodal (complete) mask
            instances[i].gt_masks_visible  — PolygonMasks, visible portion mask
            instances[i].proposal_boxes    — Boxes, used for crop_and_resize
        """
        # ---- Crop and resize GT masks to the ROI mask output size ----------
        # The mask head deconv doubles the pooler-resolution spatial dims,
        # e.g. 14×14 ROI features → 28×28 mask predictions.
        mask_side_len = F.shape[-1] * 2

        gt_amodal_list:  list = []
        gt_visible_list: list = []
        for inst in instances:
            if len(inst) == 0:
                continue
            gt_amodal_list.append(
                inst.gt_masks_amodal.crop_and_resize(
                    inst.proposal_boxes.tensor, mask_side_len
                ).to(F.device)
            )
            gt_visible_list.append(
                inst.gt_masks_visible.crop_and_resize(
                    inst.proposal_boxes.tensor, mask_side_len
                ).to(F.device)
            )

        if not gt_amodal_list:
            # No foreground proposals — return a zero loss with grad.
            return {"loss_mask": F.sum() * 0.0}

        gt_amodal  = cat(gt_amodal_list,  dim=0).float().unsqueeze(1)  # (N, 1, H, W)
        gt_visible = cat(gt_visible_list, dim=0).float().unsqueeze(1)  # (N, 1, H, W)

        # ==================================================================
        # Module 1 — Coarse Mask Segmentation
        # ==================================================================
        Mc_a_logits, _ = self.coarse_amodal_head.forward_with_features(F)
        Mc_v_logits, _ = self.coarse_visible_head.forward_with_features(F)

        L_ca = self._bce_mask_loss(Mc_a_logits, gt_amodal)
        L_cv = self._bce_mask_loss(Mc_v_logits, gt_visible)

        Mc_a_soft = torch.sigmoid(Mc_a_logits)   # (N, 1, H, W) for attention
        # Downsample to match ROI feature spatial resolution for attention
        Mc_a_soft_attn = interpolate(Mc_a_soft, size=F.shape[-2:], mode="bilinear", align_corners=False)

        # ==================================================================
        # Module 2 — Visible Mask Segmentation
        # Cross-task attention scheme (d) from Table 3 of Xiao et al.:
        #   M^r_v = f_v(F * M^c_a)
        # ==================================================================
        F_vis_attn = F * Mc_a_soft_attn

        Mr_v_logits, feats_vis_attn  = self.refined_visible_head.forward_with_features(F_vis_attn)
        _,           feats_vis_plain = self.refined_visible_head.forward_with_features(F)

        L_rv  = self._bce_mask_loss(Mr_v_logits, gt_visible)
        L_vfm = _feature_matching_loss(feats_vis_plain, feats_vis_attn, self.fm_lambdas)

        Mr_v_soft = torch.sigmoid(Mr_v_logits)   # (N, 1, H, W)
        # Downsample to match ROI feature spatial resolution for attention
        Mr_v_soft_attn = interpolate(Mr_v_soft, size=F.shape[-2:], mode="bilinear", align_corners=False)

        # ==================================================================
        # Module 3 — Amodal Mask Segmentation (U-Net replaces auto-encoder)
        #
        # Step A: U-Net reconstruction
        #   Input : M^c_a (coarse amodal mask logits — single channel)
        #   Output: U-Net refined amodal logits (same spatial size)
        #   Loss  : BCE against GT amodal mask (trains U-Net end-to-end)
        #
        # Step B: Refined amodal prediction
        #   Input : cat(F * M^r_v,  sigmoid(unet_logits))
        #           = visible-region features concatenated with U-Net output
        #   Output: M^r_a (final refined amodal mask)
        #   Loss  : BCE against GT amodal mask + feature matching
        # ==================================================================

        # Step A — U-Net reconstruction
        unet_logits = self.unet(Mc_a_soft_attn)                   # (N, 1, H, W)
        L_unet_recon = self._bce_mask_loss(unet_logits, gt_amodal)
        unet_soft    = torch.sigmoid(unet_logits)                  # (N, 1, H, W)

        # Step B — Refined amodal head
        # Concatenate visible-region attended features with U-Net output
        F_amodal_attn  = F * Mr_v_soft_attn                       # (N, C, H, W)
        F_amodal_input = torch.cat([F_amodal_attn, unet_soft], dim=1)  # (N,C+1,H,W)

        # For feature matching we also need the plain (unattended) version
        F_amodal_plain_input = torch.cat(
            [F, torch.zeros_like(unet_soft)], dim=1
        )  # zeros stand in for U-Net output in the plain path

        Mr_a_logits, feats_amodal_attn  = self.refined_amodal_head.forward_with_features(F_amodal_input)
        _,           feats_amodal_plain = self.refined_amodal_head.forward_with_features(F_amodal_plain_input)

        L_ra  = self._bce_mask_loss(Mr_a_logits, gt_amodal)
        L_afm = _feature_matching_loss(feats_amodal_plain, feats_amodal_attn, self.fm_lambdas)

        # ==================================================================
        # Clogging Extent Loss
        # Computed from the *refined* masks (highest quality predictions).
        # Quality gate w_i = soft_IoU(M^r_a, M^g_a) — detached.
        # ==================================================================
        L_clog = clogging_extent_loss(
            pred_amodal_logits  = Mr_a_logits,
            pred_visible_logits = Mr_v_logits,
            gt_amodal_masks     = gt_amodal,
            gt_visible_masks    = gt_visible,
        )

        # ==================================================================
        # Uncertainty-Weighted Total (learnable losses only)
        # Detection losses (L_cls, L_reg) are returned by the box head and
        # added externally by GeneralizedRCNN — do not include them here.
        # ==================================================================
        named_losses = {
            "loss_mask_coarse_amodal":   L_ca,
            "loss_mask_coarse_visible":  L_cv,
            "loss_mask_refined_visible": L_rv,
            "loss_vfm":                  L_vfm,
            "loss_unet_recon":           L_unet_recon,
            "loss_mask_refined_amodal":  L_ra,
            "loss_afm":                  L_afm,
            "loss_clog":                 L_clog,
        }
        loss_total = self._uncertainty_loss(named_losses)

        # Log individual losses to TensorBoard/W&B via the event storage.
        # These are NOT included in the returned dict so they are never summed
        # into the optimiser step.
        try:
            storage = get_event_storage()
            storage.put_scalar("mask_head/loss_coarse_amodal",   float(L_ca))
            storage.put_scalar("mask_head/loss_coarse_visible",  float(L_cv))
            storage.put_scalar("mask_head/loss_refined_visible", float(L_rv))
            storage.put_scalar("mask_head/loss_vfm",             float(L_vfm))
            storage.put_scalar("mask_head/loss_unet_recon",      float(L_unet_recon))
            storage.put_scalar("mask_head/loss_refined_amodal",  float(L_ra))
            storage.put_scalar("mask_head/loss_afm",             float(L_afm))
            storage.put_scalar("mask_head/loss_clog",            float(L_clog))
        except Exception:
            pass  # storage not available outside training loop (e.g. unit tests)

        # Return ONLY the single combined loss key consumed by the optimiser.
        return {"loss_mask": loss_total}

    # ------------------------------------------------------------------

    def _forward_inference(
        self,
        F: torch.Tensor,
        instances: list,
    ) -> list:
        """
        Inference path — runs the full pipeline and attaches refined masks
        and predicted clogging extent to each Instances object.
        """
        # Module 1
        Mc_a_logits, _ = self.coarse_amodal_head.forward_with_features(F)
        Mc_a_soft = torch.sigmoid(Mc_a_logits)  # (N, 1, 2H, 2W) — 2× upsampled by deconv

        # Downsample coarse mask to ROI feature resolution before attention
        Mc_a_soft_attn = interpolate(Mc_a_soft, size=F.shape[-2:], mode="bilinear", align_corners=False)

        # Module 2
        F_vis_attn  = F * Mc_a_soft_attn
        Mr_v_logits, _ = self.refined_visible_head.forward_with_features(F_vis_attn)
        Mr_v_soft = torch.sigmoid(Mr_v_logits)  # (N, 1, 2H, 2W)

        Mr_v_soft_attn = interpolate(Mr_v_soft, size=F.shape[-2:], mode="bilinear", align_corners=False)

        # Module 3 — U-Net + refined amodal
        unet_logits = self.unet(Mc_a_soft_attn)  # consistent input size with training
        unet_soft   = torch.sigmoid(unet_logits)

        F_amodal_input = torch.cat([F * Mr_v_soft_attn, unet_soft], dim=1)
        Mr_a_logits, _ = self.refined_amodal_head.forward_with_features(F_amodal_input)
        Mr_a_soft = torch.sigmoid(Mr_a_logits)

        # Predicted clogging extent per instance
        eps = 1e-6
        N = Mr_a_soft.shape[0]
        p_a = Mr_a_soft.view(N, -1)
        p_v = Mr_v_soft.view(N, -1)
        occ_area  = (p_a - p_v).clamp(min=0).sum(dim=1)
        amod_area = p_a.sum(dim=1)
        clogging_extent = occ_area / (amod_area + eps)   # (N,) in [0, 1]

        # Attach to instances (split by image)
        num_per_image = [len(i) for i in instances]
        start = 0
        for i, count in enumerate(num_per_image):
            end = start + count
            # pred_masks (N, 1, H, W) is the standard field consumed by
            # COCOEvaluator / instances_to_coco_json and by
            # GeneralizedRCNN._postprocess (paste_masks_in_image).
            instances[i].pred_masks           = Mr_a_soft[start:end]        # (count, 1, H, W)
            instances[i].pred_masks_amodal    = Mr_a_soft[start:end, 0]    # (count, H, W)
            instances[i].pred_masks_visible   = Mr_v_soft[start:end, 0]    # (count, H, W)
            instances[i].pred_clogging_extent = clogging_extent[start:end]  # (count,)
            start = end

        return instances