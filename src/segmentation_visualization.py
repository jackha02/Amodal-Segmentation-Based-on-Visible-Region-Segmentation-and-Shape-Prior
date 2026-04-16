# =============================================================================
# segmentation_visualization.py  —  Qualitative inference visualization
# =============================================================================
#
# PURPOSE
# -------
# Randomly selects N images from the held-out test dataset, runs inference
# using a randomly chosen trained fold model for each image, and saves
# overlay visualisations showing:
#
#   • Predicted AMODAL mask     — blue  semi-transparent overlay
#   • Predicted VISIBLE mask    — red   semi-transparent overlay
#   • Predicted clogging extent — value displayed in a small box per instance
#
# USAGE
# -----
#   Adjust ConfigArgs at the bottom, then run:
#
#       conda activate d2_amodal
#       python segmentation_visualization.py
#
# INPUT
# -----
#   Same COCO JSON format as used for training (synthetic.json / clogged.json).
#   Requires standard COCO fields plus:
#       "segmentation"         : polygon  — amodal mask
#       "visible_segmentation" : polygon  — visible mask
#       "clogging_extent"      : float [0, 1]
#
#   Each fold's trained checkpoint must exist at:
#       <trained_models_dir>/fold_<k>/model_final.pth
#
# OUTPUT
# ------
#   Saved images:
#       <trained_models_dir>/visualizations/
#         <img_stem>_fold<k>.jpg    — one file per visualised image
#
#   Overlay legend:
#       Blue  = predicted amodal mask  (semi-transparent filled polygon)
#       Red   = predicted visible mask (semi-transparent filled polygon)
#       Box   = predicted clogging extent label per instance (e.g. "Clog: 42%")
#
# =============================================================================

import copy
import json
import logging
import os
import random
import sys

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(
    os.path.dirname(SCRIPT_DIR),
    "Amodal-Segmentation-Based-on-Visible-Region-Segmentation-and-Shape-Prior",
)
if not os.path.isdir(REPO_DIR):
    raise RuntimeError(
        f"Cannot find the VRSP-Net repository at:\n  {REPO_DIR}\n"
        "Ensure the repo is cloned as a direct sibling of the code/ folder."
    )
sys.path.insert(0, REPO_DIR)

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import add_clogging_config
from detectron2.data import build_detection_test_loader
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.modeling.roi_heads import (
    ROI_HEADS_REGISTRY,
    StandardROIHeads,
    select_foreground_proposals,
)
from detectron2.structures import BoxMode, PolygonMasks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("segmentation_viz")


# ---------------------------------------------------------------------------
# Custom ROI Heads — identical to train_amodal_segmentation.py
# ---------------------------------------------------------------------------
@ROI_HEADS_REGISTRY.register()
class CloggingROIHeads(StandardROIHeads):
    """Thin subclass that forwards `instances` to the clogging mask head."""

    def _forward_mask(self, features, instances):
        if not self.mask_on:
            return {} if self.training else instances

        if self.training:
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            mask_features  = self.mask_pooler(features, proposal_boxes)
            return self.mask_head(mask_features, proposals)
        else:
            pred_boxes    = [x.pred_boxes for x in instances]
            mask_features = self.mask_pooler(features, pred_boxes)
            return self.mask_head(mask_features, instances)


# ---------------------------------------------------------------------------
# Dataset registration — identical loader to train_amodal_segmentation.py
# ---------------------------------------------------------------------------

def register_vis_dataset(dataset_name, json_path, images_dir):
    """
    Register a COCO JSON test dataset, preserving the custom fields
    (clogging_extent, visible_segmentation).  Safe to call multiple times.
    """
    if dataset_name in DatasetCatalog.list():
        return

    def _loader():
        with open(json_path) as f:
            data = json.load(f)
        anns_by_image = {}
        for ann in data["annotations"]:
            anns_by_image.setdefault(ann["image_id"], []).append(ann)
        dataset_dicts = []
        for img in data["images"]:
            img_id = img["id"]
            record = {
                "file_name":   os.path.join(images_dir, img["file_name"]),
                "image_id":    img_id,
                "height":      img["height"],
                "width":       img["width"],
                "annotations": [],
            }
            for ann in anns_by_image.get(img_id, []):
                obj = {
                    "id":           ann["id"],
                    "bbox":         ann["bbox"],
                    "bbox_mode":    BoxMode.XYWH_ABS,
                    "category_id":  0,
                    "segmentation": ann.get("segmentation", []),
                    "iscrowd":      ann.get("iscrowd", 0),
                    "area":         ann.get("area", 0),
                }
                if "clogging_extent" in ann:
                    obj["clogging_extent"] = ann["clogging_extent"]
                if "visible_segmentation" in ann:
                    obj["visible_segmentation"] = ann["visible_segmentation"]
                record["annotations"].append(obj)
            dataset_dicts.append(record)
        return dataset_dicts

    DatasetCatalog.register(dataset_name, _loader)
    MetadataCatalog.get(dataset_name).set(thing_classes=["storm_drain"])


# ---------------------------------------------------------------------------
# Config builder — eval-only, mirrors train_amodal_segmentation.py
# ---------------------------------------------------------------------------

def build_vis_cfg(args, checkpoint_path, dataset_name):
    cfg = get_cfg()
    add_clogging_config(cfg)
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.defrost()
    cfg.MODEL.DEVICE                = "cuda"
    cfg.VIS_PERIOD                  = 0
    cfg.MODEL.ROI_HEADS.NAME              = "CloggingROIHeads"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES       = 1
    cfg.MODEL.RPN.IOU_THRESHOLDS          = [0.3, 0.5]
    cfg.MODEL.RPN.IOU_LABELS              = [0, -1, 1]
    cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS    = [0.1] # minimum IoU between gt and predicted inference
    cfg.MODEL.ROI_HEADS.IOU_LABELS        = [0, 1]
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST   = 0.5
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.TEST.DETECTIONS_PER_IMAGE         = 2
    cfg.MODEL.ROI_MASK_HEAD.NAME          = "ParallelAmodalMaskHeadClogging"
    cfg.MODEL.MASK_ON                     = True
    cfg.MODEL.WEIGHTS                     = checkpoint_path
    cfg.DATASETS.TRAIN                    = ()
    cfg.DATASETS.TEST                     = (dataset_name,)
    cfg.OUTPUT_DIR                        = os.path.dirname(checkpoint_path)
    cfg.freeze()
    return cfg


# ---------------------------------------------------------------------------
# Core visualization helpers
# ---------------------------------------------------------------------------

def _overlay_mask(canvas, binary_mask, color_bgr, alpha=0.15, fill=False, contour_thickness=6):
    """
    Draw a colored segmentation mask polygon outline onto the canvas (in-place).
    Draws thick contour lines to show the actual polygon shape, not a filled box.

    Parameters
    ----------
    canvas             : np.ndarray  H x W x 3, uint8, BGR
    binary_mask        : np.ndarray  H x W, bool
    color_bgr          : tuple       (B, G, R) 0-255
    alpha              : float       transparency of light fill (0-1), default 0.15
    fill               : bool        if True, add light semi-transparent fill under contours
    contour_thickness  : int         thickness of contour line (default 6)
    """
    if np.sum(binary_mask) == 0:
        return  # Empty mask
    
    # Optional: add very light fill to make region visible but keep focus on boundary
    if fill:
        overlay = np.zeros_like(canvas)
        overlay[binary_mask] = color_bgr
        cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, canvas)
    
    # Draw boundary contours — THICK and BRIGHT to show polygon shape clearly
    mask_uint8 = (binary_mask.astype(np.uint8) * 255)
    findContours_ret = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Support both OpenCV 3.x and 4.x return signatures
    if len(findContours_ret) == 2:
        contours, _ = findContours_ret
    else:
        _, contours, _ = findContours_ret
    
    # Draw thick bright contours (6px) so polygon shapes are clear
    cv2.drawContours(canvas, contours, -1, color_bgr, thickness=contour_thickness, lineType=cv2.LINE_AA)


def _draw_clogging_box(canvas, x1, y1, clog_pred, clog_gt=None, font_scale=0.55, thickness=1):
    """
    Draw a small filled label box showing predicted and (optionally) ground-truth clogging extent.

    Parameters
    ----------
    canvas     : np.ndarray  BGR image (modified in-place)
    x1, y1     : int         top-left corner of the instance bounding box
    clog_pred  : float       predicted clogging extent in [0, 1]
    clog_gt    : float, optional  ground-truth clogging extent in [0, 1]
    """
    if clog_gt is not None:
        label = f"Pred: {clog_pred * 100:.1f}% | GT: {clog_gt * 100:.1f}%"
    else:
        label = f"Clog: {clog_pred * 100:.1f}%"
    font     = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    pad      = 4
    bx0      = max(x1, 0)
    by0      = max(y1 - th - 2 * pad, 0)
    bx1      = bx0 + tw + 2 * pad
    by1      = by0 + th + 2 * pad
    # Dark semi-transparent background
    roi      = canvas[by0:by1, bx0:bx1]
    if roi.size == 0:
        return
    bg       = np.zeros_like(roi)
    bg[:]    = (30, 30, 30)
    cv2.addWeighted(bg, 0.70, roi, 0.30, 0, roi)
    canvas[by0:by1, bx0:bx1] = roi
    # White text
    cv2.putText(
        canvas, label,
        (bx0 + pad, by1 - pad - baseline),
        font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
    )


def visualize_single(image_bgr, instances, gt_annotations=None, include_legend=True, include_clogging_text=True):
    """
    Produce a visualization overlay for one image.

    Layers applied (back to front):
        1. Original image
        2. Amodal mask        — blue  (BGR 255, 0, 0) boundary lines
        3. Visible mask       — red   (BGR 0, 0, 255) boundary lines
        4. Clogging extent box — per instance (showing pred + gt)

    Parameters
    ----------
    image_bgr : np.ndarray   H x W x 3, uint8, BGR
    instances : Instances    CPU Detectron2 Instances from model inference
    gt_annotations : list, optional  List of ground-truth annotation dicts with clogging_extent
    include_legend : bool    If True, draw legend in top-left corner (default True)
    include_clogging_text : bool  If True, draw clogging extent labels (default True)

    Returns
    -------
    np.ndarray  Annotated BGR image.
    """
    canvas = image_bgr.copy()
    h, w   = canvas.shape[:2]

    n = len(instances)
    if n == 0:
        return canvas

    # Debug: Print all available fields and their shapes/types for this Instances object
    field_info = {}
    for k, v in instances.get_fields().items():
        try:
            if hasattr(v, 'shape'):
                field_info[k] = f"shape={v.shape}, dtype={getattr(v, 'dtype', type(v))}"
            elif hasattr(v, 'tensor') and hasattr(v.tensor, 'shape'):
                field_info[k] = f"shape={v.tensor.shape}, dtype={getattr(v.tensor, 'dtype', type(v.tensor))}"
            else:
                field_info[k] = str(type(v))
        except Exception as e:
            field_info[k] = f"unreadable ({e})"
    logger.info(f"Instances fields: {field_info}")

    # Extract visible and amodal masks (if available)
    # NOTE: The model attaches masks with these field names:
    # - pred_masks_amodal (3D: count×H×W) — refined amodal mask
    # - pred_masks_visible (3D: count×H×W) — refined visible mask
    # - pred_masks (4D: count×1×H×W) — standard field (also amodal)
    
    # Extract visible masks — standard field attached by mask_head_clogging.py
    # CRITICAL: must threshold at 0.5, NOT use .astype(bool)!
    # sigmoid outputs are always > 0 (e.g. background pixel ≈ 0.01),
    # so .astype(bool) converts the entire 28×28 grid to all-True → filled rectangle.
    if instances.has("pred_masks_visible"):
        visible_masks = (instances.pred_masks_visible.numpy() > 0.5)
        vis_coverage = visible_masks.mean(axis=(1, 2)) * 100
        logger.info(f"pred_masks_visible: shape={visible_masks.shape}  coverage(%)={vis_coverage.round(1).tolist()}")
    else:
        logger.warning("No pred_masks_visible field found. Skipping visualization.")
        return canvas

    # Extract amodal masks — standard field attached by mask_head_clogging.py
    amodal_masks = None
    if instances.has("pred_masks_amodal"):
        amodal_masks = (instances.pred_masks_amodal.numpy() > 0.5)
        amod_coverage = amodal_masks.mean(axis=(1, 2)) * 100
        logger.info(f"pred_masks_amodal: shape={amodal_masks.shape}  coverage(%)={amod_coverage.round(1).tolist()}")
    else:
        logger.warning("No pred_masks_amodal field found in instances.")

    # Clogging extent (per-instance) 
    # NOTE: Clogging extent is computed internally by the model as:
    #   clogging_extent_i = 1 - (visible_area_i / amodal_area_i)
    # Each instance has its OWN amodal and visible mask, so clogging is per-instance.
    # Overlapping visible masks do NOT affect individual instance clogging values—
    # they remain independent because each instance's masks are separate.
    
    clogging_vals = (
        instances.pred_clogging_extent.numpy()
        if instances.has("pred_clogging_extent")
        else np.zeros(n, dtype=np.float32)
    )
    boxes = instances.pred_boxes.tensor.numpy().astype(int)  # [N, 4] xyxy

    # Filter to only instances with valid clogging extent predictions
    # (i.e., instances that have actual clogging extent values from the model)
    valid_indices = np.where((clogging_vals > 0) | (instances.has("pred_clogging_extent") and clogging_vals != 0))[0]
    
    if len(valid_indices) == 0 and instances.has("pred_clogging_extent"):
        # If all clogging values are zero but the field exists, show all instances
        # as they may all be valid predictions with zero clogging
        valid_indices = np.arange(n)
    elif len(valid_indices) == 0:
        # No valid clogging extent field, show all instances anyway
        valid_indices = np.arange(n)
    
    logger.info(f"Showing {len(valid_indices)} valid instances (out of {n} total detected)")

    def paste_mask_to_image(mask_small, box, img_shape):
        """
        Resize a small (28x28) mask to its bounding box region and paste into full image.
        
        Parameters
        ----------
        mask_small : np.ndarray  (H_small, W_small) binary mask at low resolution
        box : array-like         [x1, y1, x2, y2] in image coordinates
        img_shape : tuple        (img_h, img_w)
        
        Returns
        -------
        np.ndarray  (img_h, img_w) binary mask at image resolution
        """
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, img_shape[1])
        y2 = min(y2, img_shape[0])
        
        w = x2 - x1
        h = y2 - y1
        
        if w <= 0 or h <= 0:
            return np.zeros(img_shape, dtype=bool)
        
        # Resize mask to box size
        mask_resized = cv2.resize(
            mask_small.astype(np.float32), 
            (w, h), 
            interpolation=cv2.INTER_LINEAR
        )
        mask_bin = (mask_resized > 0.5)
        
        # Create full-size mask and paste
        full_mask = np.zeros(img_shape, dtype=bool)
        full_mask[y1:y2, x1:x2] = mask_bin
        
        return full_mask

    # Draw ONLY the valid instances with clogging extent predictions
    for i in valid_indices:
        x1, y1, x2, y2 = boxes[i]
        
        # Draw amodal mask FIRST (blue) so it appears under visible mask
        # Use thin, bright contour lines to show polygon shape clearly
        if amodal_masks is not None:
            mask_full_amodal = paste_mask_to_image(amodal_masks[i], [x1, y1, x2, y2], canvas.shape[:2])
            _overlay_mask(canvas, mask_full_amodal, (255, 0, 0), alpha=0.10, fill=False, contour_thickness=5)  # blue polygon outline
        
        # Draw visible mask (red) on top with thicker lines
        # This creates clear polygon boundaries for both masks
        mask_full_visible = paste_mask_to_image(visible_masks[i], [x1, y1, x2, y2], canvas.shape[:2])
        _overlay_mask(canvas, mask_full_visible, (0, 0, 255), alpha=0.10, fill=False, contour_thickness=6)  # red polygon outline
        
        
        # Get ground-truth clogging extent if available
        gt_clog = None
        if gt_annotations is not None and i < len(gt_annotations):
            gt_clog = gt_annotations[i].get('clogging_extent', None)
        
        # Draw clogging extent label box with predicted and ground-truth
        if include_clogging_text:
            _draw_clogging_box(canvas, int(x1), int(y1), float(clogging_vals[i]), clog_gt=gt_clog)

    # Draw legend in the top-left corner if requested
    if include_legend:
        lx, ly = 10, 10
        if amodal_masks is not None:
            cv2.line(canvas, (lx, ly + 7), (lx + 18, ly + 7), (255, 0, 0), 5)
            cv2.putText(canvas, "Amodal mask polygon", (lx + 24, ly + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
            ly += 22
        
        cv2.line(canvas, (lx, ly + 7), (lx + 18, ly + 7), (0, 0, 255), 5)
        cv2.putText(canvas, "Visible mask polygon", (lx + 24, ly + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
        ly += 22
        
        cv2.putText(canvas, "Clogging: Pred | GT", (lx, ly + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas


# ---------------------------------------------------------------------------
# Collage generation
# ---------------------------------------------------------------------------

def create_collage(imagesdata, cols=4, rows=6, max_img_width=256):
    """
    Create a collage from a list of image data tuples with legend header.

    Parameters
    ----------
    imagesdata : list of tuples
        Each tuple contains (image_bgr, filename_stem, pred_clogging, gt_clogging) where:
        - image_bgr : np.ndarray  BGR image
        - filename_stem : str     filename without extension
        - pred_clogging : float   predicted clogging extent (0-1) or None
        - gt_clogging : float     ground-truth clogging extent (0-1) or None
    cols : int
        Number of columns in collage (default 4)
    rows : int
        Number of rows in collage (default 6)
    max_img_width : int
        Maximum width for each image in collage (default 256)

    Returns
    -------
    np.ndarray  Collage image with header legend and all images arranged in grid.
    """
    num_images = len(imagesdata)
    num_cells = cols * rows
    
    # Limit to requested grid size
    if num_images > num_cells:
        logger.warning(f"Too many images ({num_images}) for {rows}x{cols} grid ({num_cells} cells). Using only first {num_cells}.")
        imagesdata = imagesdata[:num_cells]
    
    # Pad with blank images if needed
    if num_images < num_cells:
        blank = np.zeros((100, 100, 3), dtype=np.uint8)
        while len(imagesdata) < num_cells:
            imagesdata.append((blank, "", None, None))
    
    # Resize all images to fit in max_img_width, maintaining aspect ratio
    resized_images = []
    for item in imagesdata:
        if len(item) == 2:
            img_bgr, img_stem = item
            pred_clog, gt_clog = None, None
        else:
            img_bgr, img_stem, pred_clog, gt_clog = item
        
        h, w = img_bgr.shape[:2]
        if w > max_img_width:
            scale = max_img_width / w
            new_h = int(h * scale)
            img_resized = cv2.resize(img_bgr, (max_img_width, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            img_resized = img_bgr
        resized_images.append((img_resized, img_stem, pred_clog, gt_clog))
    
    # Find max dimensions per column and row
    col_widths = [0] * cols
    row_heights = [0] * rows
    
    for idx, (img, _, _, _) in enumerate(resized_images):
        row_idx = idx // cols
        col_idx = idx % cols
        h, w = img.shape[:2]
        col_widths[col_idx] = max(col_widths[col_idx], w)
        # +50 for filename and clogging labels
        row_heights[row_idx] = max(row_heights[row_idx], h + 50)
    
    grid_width = sum(col_widths) + (cols - 1) * 5  # 5px spacing
    grid_height = sum(row_heights) + (rows - 1) * 5
    
    # Add header for legend (60px tall)
    header_height = 60
    total_width = int(grid_width)
    total_height = int(header_height + grid_height)
    
    # Create blank canvas
    collage = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255
    
    # Draw header legend
    header = collage[:header_height, :]
    # Blue line for amodal
    cv2.line(header, (10, 30), (30, 30), (255, 0, 0), 4)
    cv2.putText(header, "Blue = Amodal mask", (40, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    # Red line for visible
    cv2.line(header, (300, 30), (320, 30), (0, 0, 255), 4)
    cv2.putText(header, "Red = Visible mask", (330, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    
    # Place images in grid (below header)
    for idx, (img, img_stem, pred_clog, gt_clog) in enumerate(resized_images):
        row_idx = idx // cols
        col_idx = idx % cols
        
        # Calculate position
        x_offset = sum(col_widths[:col_idx]) + col_idx * 5
        y_offset = header_height + sum(row_heights[:row_idx]) + row_idx * 5
        
        h, w = img.shape[:2]
        
        # Place image (centered in its cell if smaller than cell)
        cell_w = col_widths[col_idx]
        img_x = int(x_offset + (cell_w - w) / 2)
        img_y = int(y_offset + 50)  # Leave space at top for labels
        
        collage[img_y:img_y+h, img_x:img_x+w] = img
        
        # Draw filename at top of cell in small font (0.35 scale)
        if img_stem:
            font_scale = 0.35
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_thickness = 1
            (tw, th), _ = cv2.getTextSize(img_stem, font, font_scale, font_thickness)
            text_x = int(x_offset + (cell_w - tw) / 2)
            text_y = int(y_offset + 18)
            cv2.putText(collage, img_stem, (text_x, text_y), font, 
                       font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
        
        # Draw clogging extent in larger font (1.5x filename size = 0.525 scale)
        # Position at top left of image
        if pred_clog is not None:
            clog_font_scale = 0.35 * 1.5
            clog_thickness = 1
            if gt_clog is not None:
                clog_text = f"P:{pred_clog*100:.0f}% G:{gt_clog*100:.0f}%"
            else:
                clog_text = f"P:{pred_clog*100:.0f}%"
            
            # Draw semi-transparent background for text
            (tw, th), baseline = cv2.getTextSize(clog_text, cv2.FONT_HERSHEY_SIMPLEX, 
                                                  clog_font_scale, clog_thickness)
            text_x = img_x + 5
            text_y = img_y + th + 5
            
            # Semi-transparent background
            roi = collage[text_y-th-2:text_y+2, text_x:text_x+tw+4].copy()
            bg = np.zeros_like(roi)
            bg[:] = (200, 200, 200)
            cv2.addWeighted(bg, 0.7, roi, 0.3, 0, roi)
            collage[text_y-th-2:text_y+2, text_x:text_x+tw+4] = roi
            
            # Draw text
            cv2.putText(collage, clog_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, clog_font_scale, (0, 0, 0), clog_thickness, cv2.LINE_AA)
    
    return collage.astype(np.uint8)


# ---------------------------------------------------------------------------
# Scatter plot with absolute error bands
# ---------------------------------------------------------------------------

def plot_clogging_atp_scatter_absolute(clogging_pairs, output_path, error_thresholds=None):
    """
    Generate an Actual-to-Predicted (AtP) scatter plot with ABSOLUTE error bands.
    Error bands are parallel to the perfect prediction line.

    Parameters
    ----------
    clogging_pairs : list of tuples
        List of (actual, predicted) clogging extent pairs in [0, 1].
    output_path : str
        Absolute path where figure will be saved.
    error_thresholds : list of float, optional
        Absolute error values (in [0, 1]) to display as bands.
        Default: [0.1, 0.2] for ±10% and ±20% absolute error.
        At actual=60%, bands show 50%-70% and 40%-80% predicted ranges.

    Returns
    -------
    None
        Saves figure to output_path.
    """
    if error_thresholds is None:
        error_thresholds = [0.1, 0.2]
    
    error_thresholds = sorted(error_thresholds, reverse=True)
    
    if len(clogging_pairs) == 0:
        logger.warning(f"No clogging pairs provided. Skipping scatter plot.")
        return
    
    actual = np.array([p[0] for p in clogging_pairs]) * 100
    predicted = np.array([p[1] for p in clogging_pairs]) * 100
    
    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    
    axis_min = 0
    axis_max = 100
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
  # Plot error bands (absolute, parallel to perfect prediction line)
    # y = x + error for upper band
    # y = x - error for lower band
    # Bands are ordered from largest (lightest) to smallest (darker) for visual hierarchy
    
    # Using a slightly bluer/cooler gray to match the screenshot better
    colors = plt.cm.bone_r(np.linspace(0.1, 0.3, len(error_thresholds))) 
    
    for idx, error_abs in enumerate(error_thresholds):
        error_pct = error_abs * 100
        # Create parallel bands over a dense grid for all x values
        x_band = np.linspace(axis_min, axis_max, 500)
        y_lower = x_band - error_pct
        y_upper = x_band + error_pct
        # Clip to plot bounds
        y_lower = np.maximum(y_lower, axis_min)
        y_upper = np.minimum(y_upper, axis_max)
        
        # The screenshot only has hatching on the inner 10% band
        # Since error_thresholds are sorted descending (20% then 10%), 
        # idx 1 corresponds to the 10% band.
        hatch_pattern = '-' if error_pct == 10 else None
        
        # In Matplotlib, the hatch color is controlled by the 'edgecolor'
        edge_color = '#333333' if hatch_pattern else 'none'
        
        # Fill the band
        ax.fill_between(
            x_band, y_lower, y_upper, 
            facecolor=colors[idx], 
            edgecolor=edge_color,
            hatch=hatch_pattern,
            alpha=0.4 if hatch_pattern else 0.40, # Slightly darker alpha for the hatched band
            linewidth=0.5 if hatch_pattern else 0, # Thin border for the hatched area
            label=f'{error_pct:.0f}% Error', 
            zorder=1
        )
    
    # Plot perfect prediction line (slope=1) - changed 'r-' to 'r--' for dashed line
    perfect_line = np.array([axis_min, axis_max])
    ax.plot(perfect_line, perfect_line, 'r--', linewidth=2.5, label='Perfect Prediction', zorder=5)

    # Plot data points as black circles
    ax.scatter(actual, predicted, marker='o', s=50, color='black', 
               alpha=0.7, edgecolors='black', linewidth=0.5, label=f'Predictions (n={len(actual)})', zorder=10)
    
    # Labels and formatting
    ax.set_xlabel('Actual Clogging Extent (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Clogging Extent (%)', fontsize=12, fontweight='bold')
    ax.set_title('Actual-to-Predicted (AtP) Scatter Plot\nClogging Extent Predictions', 
                 fontsize=14, fontweight='bold', pad=20)
    
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"  AtP scatter plot (absolute, n={len(actual)}) saved to {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Error bar chart
# ---------------------------------------------------------------------------

def plot_clogging_error_bars(clogging_errors, actual_values, output_path, num_bins=4):
    """
    Generate error bar chart of MAE grouped by clogging extent ranges.
    Plots all individual data points as small circles with median highlighted.

    Parameters
    ----------
    clogging_errors : list of float
        Absolute errors (|predicted - actual|) for each instance.
    actual_values : list of float
        Actual clogging extent values in [0, 1] for each instance.
    output_path : str
        Absolute path where figure will be saved.
    num_bins : int
        Number of bins to split clogging range into (default 4).

    Returns
    -------
    None
        Saves figure to output_path.
    """
    if len(clogging_errors) == 0 or len(actual_values) == 0:
        logger.warning(f"No clogging data provided. Skipping error bar chart.")
        return
    
    # Convert to array and filter valid pairs
    errors = np.array(clogging_errors)
    actuals = np.array(actual_values) * 100  # Convert to percentage
    
    # Determine bin edges: round to nearest 5 for neatness
    min_val = np.floor(actuals.min() / 5) * 5
    max_val = np.ceil(actuals.max() / 5) * 5
    
    # Create bins
    bin_edges = np.linspace(min_val, max_val, num_bins + 1)
    bin_labels = []
    bin_stats = {"min": [], "max": [], "median": [], "x_positions": [], "all_errors": []}
    
    for i in range(len(bin_edges) - 1):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]
        mask = (actuals >= lower) & (actuals < upper)
        
        # Handle last bin inclusively
        if i == len(bin_edges) - 2:
            mask = (actuals >= lower) & (actuals <= upper)
        
        bin_label = f"{int(lower)}-{int(upper)}%"
        bin_labels.append(bin_label)
        
        if np.any(mask):
            bin_errors = errors[mask]
            bin_stats["min"].append(np.min(bin_errors))
            bin_stats["max"].append(np.max(bin_errors))
            bin_stats["median"].append(np.median(bin_errors))
            bin_stats["all_errors"].append(bin_errors)
        else:
            bin_stats["min"].append(0)
            bin_stats["max"].append(0)
            bin_stats["median"].append(0)
            bin_stats["all_errors"].append(np.array([]))
        
        bin_stats["x_positions"].append(i)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
    
    x_pos = np.array(bin_stats["x_positions"])
    min_vals = np.array(bin_stats["min"])
    max_vals = np.array(bin_stats["max"])
    median_vals = np.array(bin_stats["median"])
    
    # Plot transparent bars from min to max (black with high transparency)
    for i, (x, min_v, max_v) in enumerate(zip(x_pos, min_vals, max_vals)):
        if max_v > min_v:
            # Draw semi-transparent bar
            ax.plot([x, x], [min_v, max_v], 'k-', linewidth=12, alpha=0.15, zorder=1)
    
    # Plot individual data points as small circles (light black)
    for i, (x, errors_in_bin) in enumerate(zip(x_pos, bin_stats["all_errors"])):
        if len(errors_in_bin) > 0:
            # Add slight jitter to x position to avoid overlapping points
            jittered_x = x + np.random.normal(0, 0.02, len(errors_in_bin))
            ax.scatter(jittered_x, errors_in_bin, s=20, color='black', 
                      alpha=0.3, edgecolors='none', zorder=2)
    
    # Plot median points as larger, bold circles (dark black)
    ax.scatter(x_pos, median_vals, s=100, color='black', 
              alpha=1.0, edgecolors='black', linewidth=2, 
              zorder=10, label='Median MAE')
    
    # Labels and formatting
    ax.set_xlabel('Clogging Extent Range (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('MAE (Mean Absolute Error)', fontsize=12, fontweight='bold')
    ax.set_title('Error Bar Chart of Clogging Extent Prediction', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bin_labels)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y', linewidth=0.5)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
    
    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"  Error bar chart saved to {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main visualization loop
# ---------------------------------------------------------------------------

def run_visualization(args):
    """
    Run inference on ALL test dataset images, create collages of predictions,
    generate scatter plot with absolute error bands, and create error bar chart.
    
    Output structure:
        <trained_models_dir>/visualizations/
          collage_<batch_index>.jpg          — 4x6 grid collages
          clogging_atp_scatter_absolute.png  — scatter plot (absolute error bands, ALL data)
          clogging_error_bars.png            — error bar chart (ALL data)
    """
    out_dir = os.path.join(args.trained_models_dir, "visualizations")
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Visualization output directory: {out_dir}")

    # Register dataset
    ds_name = "storm_drain_vis"
    register_vis_dataset(ds_name, args.test_json, args.test_imgs)
    dataset_dicts = DatasetCatalog.get(ds_name)
    total_images = len(dataset_dicts)
    logger.info(f"Total images in dataset: {total_images}")

    # Verify available fold checkpoints
    available_folds = [
        fold for fold in range(args.kfold)
        if os.path.isfile(
            os.path.join(args.trained_models_dir, "model_0044999.pth")
        )
    ]
    if not available_folds:
        raise FileNotFoundError(
            "No model_final.pth found in any fold directory under:\n"
            f"  {args.trained_models_dir}\n"
            "Run train_amodal_segmentation.py first."
        )
    logger.info(f"Available fold checkpoints: {available_folds}")

    # Use ALL images from the dataset
    rng = random.Random(args.seed)
    selected = dataset_dicts  # Use ALL images

    # Cache loaded models per fold to avoid reloading the same checkpoint
    model_cache = {}
    

    # Data collection for plots and collages (IoU-matched pairs only)
    all_viz_images = []  # List of (image_bgr, filename_stem, pred_clogging, gt_clogging) tuples
    all_clogging_pairs = []  # List of (actual, predicted) tuples for scatter plot
    all_clogging_errors = []  # List of absolute errors for error bar chart
    all_actual_values = []  # List of actual clogging extent values

    from detectron2.structures import Boxes, pairwise_iou
    import torch

    for idx, record in enumerate(selected):
        fold = rng.choice(available_folds)
        checkpoint = os.path.join(args.trained_models_dir, "model_0044999.pth")

        # Load (or reuse cached) model
        if fold not in model_cache:
            logger.info(f"  Loading checkpoint for fold {fold}: {checkpoint}")
            cfg = build_vis_cfg(args, checkpoint, ds_name)
            model = DefaultTrainer.build_model(cfg)
            DetectionCheckpointer(model).load(checkpoint)
            model.eval()
            model_cache[fold] = (model, cfg)
        else:
            model, cfg = model_cache[fold]

        # Load image
        image_bgr = cv2.imread(record["file_name"])
        if image_bgr is None:
            logger.warning(f"  Could not read image: {record['file_name']} — skipping.")
            continue

        # Run inference (Detectron2 expects RGB input as a tensor)
        image_rgb = image_bgr[:, :, ::-1].copy()
        image_tensor = torch.as_tensor(
            np.ascontiguousarray(image_rgb.transpose(2, 0, 1)), dtype=torch.float32
        )
        with torch.no_grad():
            inputs = [{"image": image_tensor, "height": image_bgr.shape[0],
                       "width": image_bgr.shape[1]}]
            outputs = model(inputs)
        instances = outputs[0]["instances"].to("cpu")

        # Extract ground-truth annotations from the record
        gt_anns = record.get("annotations", [])

        # Build GT boxes and clogging extents
        gt_boxes = []
        gt_clogging = []
        for ann in gt_anns:
            bbox = ann["bbox"]
            # Convert XYWH to XYXY
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h
            gt_boxes.append([x1, y1, x2, y2])
            gt_clogging.append(float(ann.get("clogging_extent", 0.0)))

        if len(gt_boxes) == 0 or len(instances) == 0 or not instances.has("pred_clogging_extent"):
            # No GT or no predictions or no clogging field
            annotated = visualize_single(image_bgr, instances, gt_annotations=gt_anns,
                                        include_legend=False, include_clogging_text=False)
            img_stem = os.path.splitext(os.path.basename(record["file_name"]))[0]
            all_viz_images.append((annotated, img_stem, None, None))
            continue

        # Prepare predicted boxes and clogging extents
        pred_boxes = instances.pred_boxes.tensor.numpy()
        pred_clogging = instances.pred_clogging_extent.numpy()


        # IoU-based matching (as in performance_eval.py)
        gt_boxes_tensor = torch.tensor(gt_boxes, dtype=torch.float32)
        pred_boxes_tensor = torch.tensor(pred_boxes, dtype=torch.float32)
        iou_matrix = pairwise_iou(Boxes(pred_boxes_tensor), Boxes(gt_boxes_tensor))
        # For each GT, find best prediction
        max_ious, matched_pred_indices = iou_matrix.max(dim=0)

        # Use IoU threshold from config
        iou_thresh = cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS[0] if hasattr(cfg.MODEL.ROI_HEADS, 'IOU_THRESHOLDS') else 0.5
        matched_pairs = []  # (gt_idx, pred_idx)
        for gt_idx, pred_idx in enumerate(matched_pred_indices):
            if max_ious[gt_idx] > iou_thresh:
                matched_pairs.append((gt_idx, int(pred_idx)))

        # For overlays: only show matched pairs
        # For plots: only include matched pairs
        # Build overlay with only matched predictions
        # Create a new Instances object with only matched predictions
        from detectron2.structures import Instances
        matched_pred_indices_set = set(pred_idx for _, pred_idx in matched_pairs)
        if len(matched_pred_indices_set) > 0:
            matched_instances = instances[sorted(matched_pred_indices_set)]
        else:
            matched_instances = instances[:0]  # empty

        # For overlays, pass only matched predictions and GTs
        matched_gt_anns = [gt_anns[gt_idx] for gt_idx, _ in matched_pairs]
        annotated = visualize_single(image_bgr, matched_instances, gt_annotations=matched_gt_anns,
                                    include_legend=False, include_clogging_text=False)
        img_stem = os.path.splitext(os.path.basename(record["file_name"]))[0]

        # For plots, collect only matched pairs
        for gt_idx, pred_idx in matched_pairs:
            gt_clog = gt_clogging[gt_idx]
            pred_clog = float(pred_clogging[pred_idx])
            all_clogging_pairs.append((gt_clog, pred_clog))
            all_clogging_errors.append(abs(pred_clog - gt_clog))
            all_actual_values.append(gt_clog)

        # For collage, show per-image mean if multiple matches, else None
        if matched_pairs:
            mean_pred = float(np.mean([float(pred_clogging[pred_idx]) for _, pred_idx in matched_pairs]))
            mean_gt = float(np.mean([gt_clogging[gt_idx] for gt_idx, _ in matched_pairs]))
        else:
            mean_pred = None
            mean_gt = None
        all_viz_images.append((annotated, img_stem, mean_pred, mean_gt))

        # --- DEBUG: Log number of raw predictions before thresholding ---
        logger.info(f"Raw model predictions: {len(instances)} instances before IoU matching and thresholding.")
        if hasattr(instances, 'scores'):
            logger.info(f"Prediction scores: {instances.scores.cpu().numpy().round(3).tolist()}")
        else:
            logger.info("No 'scores' field in instances.")

        if (idx + 1) % 50 == 0 or idx + 1 == len(selected):
            logger.info(
                f"  [{idx + 1}/{len(selected)}] fold={fold}  "
                f"instances={len(instances)}  processed"
            )

    logger.info(f"Successfully processed {len(all_viz_images)} images")

    # Create collages (4 columns x 6 rows = 24 images per collage)
    collage_size = 24
    num_collages = (len(all_viz_images) + collage_size - 1) // collage_size
    
    for collage_idx in range(num_collages):
        start_idx = collage_idx * collage_size
        end_idx = min((collage_idx + 1) * collage_size, len(all_viz_images))
        batch = all_viz_images[start_idx:end_idx]
        
        collage = create_collage(batch, cols=4, rows=6, max_img_width=256)
        collage_name = f"collage_{collage_idx:02d}.jpg"
        collage_path = os.path.join(out_dir, collage_name)
        cv2.imwrite(collage_path, collage)
        logger.info(f"  Collage {collage_idx + 1}/{num_collages} saved ({len(batch)} images): {collage_path}")

    # Generate scatter plot with absolute error bands (using ALL data)
    scatter_path = os.path.join(out_dir, "clogging_atp_scatter_absolute.png")
    plot_clogging_atp_scatter_absolute(all_clogging_pairs, scatter_path, 
                                       error_thresholds=[0.1, 0.2])

    # Generate error bar chart (using ALL data)
    error_bar_path = os.path.join(out_dir, "clogging_error_bars.png")
    plot_clogging_error_bars(all_clogging_errors, all_actual_values, error_bar_path, num_bins=4)

    logger.info(f"\n✓ Done.")
    logger.info(f"  • {len(all_viz_images)} images visualized")
    logger.info(f"  • {num_collages} collage(s) created")
    logger.info(f"  • {len(all_clogging_pairs)} clogging predictions included in plots")
    logger.info(f"  Outputs saved to: {out_dir}")



# =============================================================================
# CONFIGURATION SETTINGS
# =============================================================================

class ConfigArgs:
    base_dir = "/run/user/1000/gvfs/smb-share:server=ecresearch.uwaterloo.ca,share=cviss/Jack/baf/360streetview"

    # Path to the test COCO JSON (same format as synthetic.json)
    test_json = os.path.join(base_dir, "segmentation_training", "synthetic_v2", "synthetic.json")

    # Path to the directory containing the test images
    test_imgs = os.path.join(base_dir, "segmentation_training", "synthetic_v2", "images")

    # Directory containing fold_0/model_final.pth … fold_4/model_final.pth
    trained_models_dir = os.path.join(base_dir, "trained_models", "synthetic_final")

    # Total number of folds used during training
    kfold = 1

    # Random seed for reproducible image and fold selection
    seed = 42

    # Model config — must match the config used at training time
    config_file = (
        "/home/cviss/jack/360streetview/"
        "Amodal-Segmentation-Based-on-Visible-Region-Segmentation-and-Shape-Prior/"
        "configs/StormDrain/mask_rcnn_R_50_FPN_1x_clogging.yaml"
    )


if __name__ == "__main__":
    args = ConfigArgs()
    run_visualization(args)