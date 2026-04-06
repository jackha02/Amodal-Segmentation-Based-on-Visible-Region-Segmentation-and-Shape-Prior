# Mean Absolute Error for to quantify the amodal-quality-gated clogging extent loss
#
# Clogging extent for instance i:
#
#   gamma_i = (amodal_area_i - visible_area_i) / amodal_area_i
#           = occluded_area_i / amodal_area_i
#
# Loss (per batch):
#
#   L_clog = (1/N) * sum_i  w_i * |gamma_pred_i - gamma_gt_i|
#
# where w_i = soft_IoU(pred_amodal_i, gt_amodal_i),  detached from the
# gradient graph so it acts as a static quality gate rather than a
# differentiable term.
#
# Need for the quality gate

# The clogging ratio is derived from two masks, so a model could in principle
# achieve a near-zero ratio error with a geometrically incorrect amodal mask
# (ratio degeneracy). The soft IoU weight suppresses the ratio penalty when
# the amodal mask itself is poor, preventing the network from optimising the
# ratio at the expense of mask quality.
#
# Soft masks vs hard masks
# ------------------------
# sigmoid(logits) are used throughout rather than hard-thresholded binary
# masks, keeping all operations end-to-end differentiable so gradients
# propagate back through both mask heads.

import torch
import torch.nn.functional as F

_EPS = 1e-6   # numerical stability — prevents division by zero

def soft_iou(pred_logits: torch.Tensor,
             gt_masks: torch.Tensor) -> torch.Tensor:
    """
    Differentiable (soft) IoU between sigmoid-activated predictions and
    binary ground-truth masks, computed per instance.

    Parameters
    ----------
    pred_logits : (N, 1, H, W)  raw logits for the amodal mask
    gt_masks    : (N, 1, H, W)  binary ground-truth amodal mask

    Returns
    -------
    iou : (N,)  per-instance soft IoU in [0, 1]
    """
    p = torch.sigmoid(pred_logits).view(pred_logits.shape[0], -1)   # (N, H*W)
    g = gt_masks.float().view(gt_masks.shape[0], -1)                # (N, H*W)
    intersection = (p * g).sum(dim=1)                               # (N,)
    union = p.sum(dim=1) + g.sum(dim=1) - intersection              # (N,)
    return intersection / (union + _EPS)                             # (N,)

def clogging_extent_loss(
    pred_amodal_logits:  torch.Tensor,
    pred_visible_logits: torch.Tensor,
    gt_amodal_masks:     torch.Tensor,
    gt_visible_masks:    torch.Tensor,
) -> torch.Tensor:
    """
    Compute the amodal-quality-gated clogging extent MAE loss

    Parameters
    ----------
    pred_amodal_logits  : (N, 1, H, W)
        Raw (pre-sigmoid) logits for the *refined* amodal mask M^r_a,
        i.e. the output of the amodal mask head f_a after the amodal mask
        segmentation module

    pred_visible_logits : (N, 1, H, W)
        Raw (pre-sigmoid) logits for the *refined* visible mask M^r_v,
        i.e. the output of the visible mask head f_v after the visible mask
        segmentation module

    gt_amodal_masks     : (N, 1, H, W)
        Binary ground-truth amodal mask M^g_a  (dtype: bool or float)

    gt_visible_masks    : (N, 1, H, W)
        Binary ground-truth visible mask M^g_v  (dtype: bool or float)

    Returns
    -------
    loss : scalar torch.Tensor
        Mean (over N instances) quality-weighted absolute error between
        predicted and ground-truth clogging extent.
        Returns 0.0 (with gradient) if N == 0

    Notes
    -----
    * The reclassification head (L_rc) is omitted from the surrounding model
      because this is a single-class problem (storm drain inlets), following
      the precedent of Kim et al. (2023) for single-class cucumber datasets.
    * No external lambda hyperparameter appears here, and the scale control is 
      handled entirely by the UncertaintyWeightedLoss wrapper (Kendall et al., 2018).
    * The quality gate weight w_i is detached so it does not back-propagate
      through the IoU computation — it acts as a per-instance scalar
      multiplier only.
    """
    N = pred_amodal_logits.shape[0]
    if N == 0:
        # Return a differentiable zero that carries the correct device/dtype.
        return pred_amodal_logits.sum() * 0.0

    # Soft sigmoid activations
    p_a = torch.sigmoid(pred_amodal_logits)    # (N, 1, H, W)  predicted amodal
    p_v = torch.sigmoid(pred_visible_logits)   # (N, 1, H, W)  predicted visible

    # Quality gate, where the weight depends on soft IoU of predicted vs GT amodal mask
    # Detached so gradients do not flow through w_i itself
    w = soft_iou(pred_amodal_logits, gt_amodal_masks).detach()  # (N,)

    # Predicted clogging extent 
    # Occluded region = amodal - visible, clamped to avoid small negatives caused by independent sigmoids
    p_a_flat  = p_a.view(N, -1)                                  # (N, H*W)
    p_v_flat  = p_v.view(N, -1)                                  # (N, H*W)
    occ_pred  = (p_a_flat - p_v_flat).clamp(min=0.0).sum(dim=1)  # (N,)
    area_pred = p_a_flat.sum(dim=1)                              # (N,)
    gamma_pred = occ_pred / (area_pred + _EPS)                   # (N,) in [0,1]

    # Ground-truth clogging extent
    g_a = gt_amodal_masks.float().view(N, -1)                    # (N, H*W)
    g_v = gt_visible_masks.float().view(N, -1)                   # (N, H*W)
    occ_gt   = (g_a - g_v).clamp(min=0.0).sum(dim=1)             # (N,)
    area_gt  = g_a.sum(dim=1)                                    # (N,)
    gamma_gt = occ_gt / (area_gt + _EPS)                         # (N,) in [0,1]

    # w_i suppresses the penalty when the amodal mask is geometrically poor, preventing ratio accuracy from being 
    # achieved via mask degeneracy (i.e., poorly predicted amodal mask)
    mae = torch.abs(gamma_pred - gamma_gt)                        # (N,)
    loss = (w * mae).mean()                                       # scalar

    return loss