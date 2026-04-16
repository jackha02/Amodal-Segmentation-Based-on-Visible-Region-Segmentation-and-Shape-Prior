# =============================================================================
# performance_eval.py  —  Held-out test set evaluation across all trained folds
# =============================================================================
#
# PURPOSE
# -------
# After k-fold cross-validation training is complete (train_amodal_segmentation.py),
# this script evaluates each fold's saved model checkpoint against a single
# held-out test dataset that was never seen during training or validation. One
# full inference pass is performed per fold, producing per-fold metrics. A
# cross-fold summary (mean ± std) is then written in the same format as the
# training kfold_summary.json, enabling direct comparison.
#
# USAGE
# -----
#   Adjust the paths in ConfigArgs at the bottom of this file, then run:
#
#       conda activate d2_amodal
#       python performance_eval.py
#
# INPUT DATA FORMAT
# -----------------
#   The test dataset must be a standard COCO JSON file with the same two
#   extra per-instance fields used during training:
#
#       "segmentation"         : polygon  — the AMODAL (complete) mask
#       "visible_segmentation" : polygon  — the VISIBLE portion of the mask
#       "clogging_extent"      : float in [0, 1]
#                                = 1 - (visible_pixel_area / amodal_pixel_area)
#
#   All standard COCO fields (id, image_id, category_id, bbox, area, iscrowd)
#   must also be present.
#
# MODEL CHECKPOINTS
# -----------------
#   Each fold's final checkpoint is expected at:
#       <trained_models_dir>/fold_<k>/model_final.pth
#
#   This is the exact path written by train_amodal_segmentation.py.
#
# OUTPUT
# ------
#   Per-fold inference results and metrics are written to:
#       <eval_output_dir>/
#         fold_<k>/
#           inference/
#             coco_instances_results.json   <- raw COCO-format predictions
#           test_metrics.json               <- full metrics for this fold
#
#   AtP (Actual-to-Predicted) scatter plots for each fold are written to:
#       <eval_output_dir>/fold_<k>/clogging_atp_scatter.png
#       (showing clogging extent predictions vs ground truth with error bands)
#
#   A cross-fold summary matching the kfold_summary.json schema is written to:
#       <eval_output_dir>/test_summary.json
#
#   test_summary.json schema (mirrors kfold_summary.json):
#   {
#     "num_folds":          int,
#     "AP_mean":            float,   <- segm AP (IoU 0.5:0.95), mean across folds
#     "AP_std":             float,
#     "AP50_mean":          float,   <- segm AP at IoU 0.50, mean
#     "AP50_std":           float,
#     "clogging_mae_mean":  float,   <- clogging extent MAE, mean
#     "clogging_mae_std":   float,
#     "per_fold": [
#       {
#         "bbox":     {"AP": float, "AP50": float, "AP75": float,
#                      "APs": float, "APm": float, "APl": float},
#         "segm":     {"AP": float, "AP50": float, "AP75": float,
#                      "APs": float, "APm": float, "APl": float,
#                      "AR1": float, "AR10": float, "AR100": float},
#         "clogging": {"MAE": float}
#       },
#       ...  (one entry per fold)
#     ]
#   }
#
# =============================================================================

import copy
import json
import logging
import math
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

import numpy as np
import torch

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data import build_detection_test_loader
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.engine import DefaultTrainer, default_setup, launch
from detectron2.evaluation import COCOEvaluator
from detectron2.evaluation import inference_on_dataset
from detectron2.config import add_clogging_config
from detectron2.modeling.roi_heads import (
    ROI_HEADS_REGISTRY,
    StandardROIHeads,
    select_foreground_proposals,
)
from detectron2.structures import BoxMode, Instances, PolygonMasks, pairwise_iou


logger = logging.getLogger("detectron2")


# ---------------------------------------------------------------------------
# Custom ROI Heads  —  identical registration to train_amodal_segmentation.py
# ---------------------------------------------------------------------------
# ParallelAmodalMaskHeadClogging.forward() requires (roi_features, instances)
# rather than the single tensor that StandardROIHeads._forward_mask provides.
# This thin subclass overrides only _forward_mask; all other logic (box head,
# pooler initialisation, optimiser) is inherited unchanged.
@ROI_HEADS_REGISTRY.register()
class CloggingROIHeads(StandardROIHeads):
    """
    Thin subclass of StandardROIHeads whose only change is to forward
    `instances` (proposals in training, predicted instances in inference)
    to the mask head together with the pooled ROI features.
    """

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
# Dataset helpers
# ---------------------------------------------------------------------------

def load_coco_json(json_path):
    """
    Load and return a parsed COCO annotation JSON dict.
    """
    with open(json_path, "r") as f:
        return json.load(f)


def register_test_dataset(dataset_name, json_path, images_dir):
    """
    Register the test COCO JSON with Detectron2's DatasetCatalog, preserving
    the non-standard fields (clogging_extent, visible_segmentation) that
    Detectron2's built-in register_coco_instances would silently drop.

    The dataset is registered only once; if it already exists in the catalog
    the call is a no-op, so this function is safe to call multiple times.

    Parameters
    ----------
    dataset_name : str   Name to register under (e.g. "storm_drain_test").
    json_path    : str   Absolute path to the COCO JSON file.
    images_dir   : str   Absolute path to the directory containing images.
    """
    if dataset_name in DatasetCatalog.list():
        return  # already registered from a previous call

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
                    "category_id":  0,  # single class
                    "segmentation": ann.get("segmentation", []),
                    "iscrowd":      ann.get("iscrowd", 0),
                    "area":         ann.get("area", 0),
                }
                # Preserve the two custom fields used by CloggingDatasetMapper
                if "clogging_extent" in ann:
                    obj["clogging_extent"] = ann["clogging_extent"]
                if "visible_segmentation" in ann:
                    obj["visible_segmentation"] = ann["visible_segmentation"]
                record["annotations"].append(obj)
            dataset_dicts.append(record)
        return dataset_dicts

    DatasetCatalog.register(dataset_name, _loader)
    MetadataCatalog.get(dataset_name).set(thing_classes=["storm_drain"])
    logger.info(f"Registered test dataset '{dataset_name}'  ({json_path})")


# ---------------------------------------------------------------------------
# Dataset mapper  —  identical to training (is_train=False path)
# ---------------------------------------------------------------------------

class CloggingDatasetMapper:
    """
    Custom mapper that parses both the amodal segmentation field ("segmentation")
    and the extra custom fields ("visible_segmentation", "clogging_extent") into
    the target Instances object.

    For evaluation (is_train=False) no augmentation is applied.  The mapper
    still attaches gt_masks_amodal, gt_masks_visible, and gt_clogging_extent
    so that CloggingEvaluator can match predicted and ground-truth clogging
    values even in the absence of training transforms.
    """

    def __init__(self, cfg, is_train=False):
        self.is_train    = is_train
        self.image_format = cfg.INPUT.FORMAT
        # No augmentation transforms during evaluation
        self.tfm_gens   = []

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)

        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )

        if not dataset_dict.get("annotations"):
            return dataset_dict

        annos = dataset_dict.pop("annotations")
        annos = [obj for obj in annos if obj.get("iscrowd", 0) == 0]

        # Map visible_segmentation -> visible_mask BEFORE geometric transforms
        # so the polygon is spatially transformed alongside "segmentation" and
        # "bbox" when augmentation is enabled (training path).
        for obj in annos:
            if "visible_mask" not in obj:
                obj["visible_mask"] = obj.get(
                    "visible_segmentation", obj.get("segmentation", [])
                )

        annos = [
            utils.transform_instance_annotations(obj, transforms, image.shape[:2])
            for obj in annos
        ]

        # Build standard instances (Detectron2 uses "segmentation" as the mask).
        instances = utils.annotations_to_instances(annos, image.shape[:2])

        # Rename the generic gt_masks to gt_masks_amodal to avoid ambiguity.
        if instances.has("gt_masks"):
            instances.gt_masks_amodal = instances.gt_masks
            instances.remove("gt_masks")

        # Build gt_masks_visible from the already-transformed visible_mask field.
        h, w = image.shape[:2]
        visible_polygons = []
        clogging_extents = []
        for obj in annos:
            vis_seg = obj.get("visible_mask", obj.get("segmentation", []))
            visible_polygons.append(vis_seg)
            clogging_extents.append(float(obj.get("clogging_extent", 0.0)))

        instances.gt_masks_visible  = PolygonMasks(visible_polygons)
        instances.gt_clogging_extent = torch.tensor(clogging_extents, dtype=torch.float32)

        dataset_dict["instances"] = instances
        return dataset_dict


# ---------------------------------------------------------------------------
# Evaluator  —  identical to training
# ---------------------------------------------------------------------------

class CloggingEvaluator(COCOEvaluator):
    """
    Extends COCOEvaluator with clogging extent MAE.

    Metric (A) — Amodal mask mAP
    -----------------------------
    Inherited from COCOEvaluator.  Predicted instance masks are matched to
    ground-truth amodal masks (the "segmentation" field) using the COCO API
    at IoU thresholds 0.50:0.05:0.95.  Reported under the "segm" key:

        AP    — mean AP over IoU 0.50:0.05:0.95
        AP50  — AP at IoU 0.50
        AP75  — AP at IoU 0.75
        APs / APm / APl  — AP by instance area (small / medium / large)
        AR1 / AR10 / AR100 — Recall at max detections 1 / 10 / 100

    Metric (B) — Clogging extent MAE
    ----------------------------------
    pred_clogging_extent is read from each predicted Instances object
    (populated by ParallelAmodalMaskHeadClogging._forward_inference).
    gt_clogging_extent is read from the ground-truth Instances object
    (populated by CloggingDatasetMapper from the "clogging_extent" field).
    Instances are matched by bounding-box IoU (threshold 0.5) to handle
    images containing multiple storm drains.  Per-instance absolute errors
    are accumulated in self._clogging_errors and averaged in evaluate().
    Result is stored under the "clogging" key as {"MAE": float}.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._clogging_errors = []
        # Store (actual, predicted) pairs for AtP scatter plot generation
        self._clogging_pairs = []

    def reset(self):
        super().reset()
        self._clogging_errors = []
        self._clogging_pairs = []

    def process(self, inputs, outputs):
        # Metric (A): standard COCO AP accumulation
        super().process(inputs, outputs)

        # Metric (B): per-instance clogging MAE accumulation
        for inp, out in zip(inputs, outputs):
            pred_inst = out.get("instances", None)
            if pred_inst is None or not pred_inst.has("pred_clogging_extent"):
                continue
            gt_inst = inp.get("instances", None)
            if gt_inst is None or not gt_inst.has("gt_clogging_extent"):
                continue
            if len(pred_inst) == 0 or len(gt_inst) == 0:
                continue

            # IoU-based matching: each GT instance is matched to the highest-
            # overlapping predicted instance.  Only matches with IoU > 0.5 are
            # accepted, consistent with COCO's standard matching threshold.
            iou_matrix = pairwise_iou(
                pred_inst.pred_boxes.to("cpu"),
                gt_inst.gt_boxes.to("cpu"),
            )
            pred_clog = pred_inst.pred_clogging_extent.cpu()
            gt_clog   = gt_inst.gt_clogging_extent.cpu()

            max_ious, matched_pred_indices = iou_matrix.max(dim=0)
            for gt_idx, pred_idx in enumerate(matched_pred_indices):
                if max_ious[gt_idx] > 0.5:
                    error = abs(float(pred_clog[pred_idx] - gt_clog[gt_idx]))
                    self._clogging_errors.append(error)
                    # Store (actual, predicted) pair for AtP plotting
                    self._clogging_pairs.append((
                        float(gt_clog[gt_idx]),
                        float(pred_clog[pred_idx])
                    ))

    def evaluate(self):
        # Metric (A): COCO AP suite
        results = super().evaluate() or {}

        # Metric (B): clogging MAE
        if self._clogging_errors:
            mae = float(np.mean(self._clogging_errors))
            logger.info(
                f"Clogging extent MAE: {mae:.4f}  "
                f"({len(self._clogging_errors)} instances)"
            )
            results["clogging"] = {"MAE": mae}
        else:
            logger.warning(
                "No clogging extent predictions collected. "
                "Verify that gt_clogging_extent is present in the test JSON "
                "and pred_clogging_extent is produced by the mask head."
            )
            results["clogging"] = {"MAE": float("nan")}

        return results


# ---------------------------------------------------------------------------
# Config builder for evaluation
# ---------------------------------------------------------------------------

def build_eval_cfg(args, test_dataset, fold_dir):
    """
    Build a Detectron2 CfgNode for inference-only evaluation of one fold.

    The config is identical to training except:
    - MODEL.WEIGHTS points to <fold_dir>/model_final.pth
    - DATASETS.TEST is set to the unseen test dataset
    - DATASETS.TRAIN is set to an empty tuple (no training)
    - OUTPUT_DIR is set to <fold_dir> so COCOEvaluator writes results there

    Parameters
    ----------
    args        : ConfigArgs   Configuration object (paths, config_file, etc.).
    test_dataset: str          Name of the registered test dataset.
    fold_dir    : str          Path to the fold's output directory, which must
                               contain model_final.pth.
    """
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
    cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS    = [0.3]
    cfg.MODEL.ROI_HEADS.IOU_LABELS        = [0, 1]
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST   = 0.5
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.TEST.DETECTIONS_PER_IMAGE         = 2
    cfg.MODEL.ROI_MASK_HEAD.NAME          = "ParallelAmodalMaskHeadClogging"
    cfg.MODEL.MASK_ON                     = True
    cfg.MODEL.WEIGHTS                     = os.path.join(fold_dir, "model_final.pth")
    cfg.DATASETS.TRAIN                    = ()
    cfg.DATASETS.TEST                     = (test_dataset,)
    cfg.OUTPUT_DIR                        = fold_dir
    cfg.freeze()
    return cfg


# ---------------------------------------------------------------------------
# AtP (Actual-to-Predicted) Scatter Plot Generation
# ---------------------------------------------------------------------------

def plot_clogging_atp_scatter(clogging_pairs, output_path, error_thresholds=None):
    """
    Generate an Actual-to-Predicted (AtP) scatter plot for clogging extent predictions.

    Parameters
    ----------
    clogging_pairs : list of tuples
        List of (actual, predicted) clogging extent pairs. Values should be in [0, 1].
        These will be converted to percentages (multiplied by 100) for plotting.
    
    output_path : str
        Absolute path where the figure will be saved (e.g., 'fold_0_atp_scatter.png').
    
    error_thresholds : list of float, optional
        Relative error percentages to display as transparent error bands around the 
        perfect prediction line. Bands expand as actual values increase (cone shape).
        Default: [10, 20] for ±10% and ±20% relative error bands.
        The error is calculated as: (predicted - actual) / actual * 100
        Example: error_thresholds=[5, 10, 15] creates bands for ±5%, ±10%, ±15%.

    Returns
    -------
    None
        Saves the figure to output_path.
    
    Notes
    -----
    The plot displays:
    - Black circles: predicted data points
    - Red line: perfect prediction (slope=1, identity line)
    - Transparent shaded cone regions: relative error bands (configurable)
    - Axes in percentage (0-100%)
    """
    if error_thresholds is None:
        error_thresholds = [10, 20]
    
    # Sort error thresholds in descending order for layering (largest first)
    error_thresholds = sorted(error_thresholds, reverse=True)
    
    # Convert pairs to arrays and scale to percentages
    if len(clogging_pairs) == 0:
        logger.warning(f"No clogging pairs available for plotting to {output_path}")
        return
    
    actual = np.array([p[0] for p in clogging_pairs]) * 100
    predicted = np.array([p[1] for p in clogging_pairs]) * 100
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    
    # Set axis limits with some padding
    axis_min = 0
    axis_max = 100
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    
    # Plot error bands (transparent) in descending order so smaller bands are visible on top
    # Error bands form a cone shape: relative error expands with actual values
    colors = plt.cm.Greys(np.linspace(0.2, 0.5, len(error_thresholds)))
    for idx, error_pct in enumerate(error_thresholds):
        # Create x-axis values (actual clogging extent)
        actual_line = np.linspace(axis_min, axis_max, 100)
        
        # Relative error bands: predicted = actual * (1 ± error_pct/100)
        # Upper error band: predicted_upper = actual * (1 + error_pct/100)
        upper_band = actual_line * (1 + error_pct / 100)
        # Lower error band: predicted_lower = actual * (1 - error_pct/100)
        lower_band = actual_line * (1 - error_pct / 100)
        
        # Clip to axis limits
        upper_band = np.clip(upper_band, axis_min, axis_max)
        lower_band = np.clip(lower_band, axis_min, axis_max)
        
        ax.fill_between(
            actual_line, lower_band, upper_band,
            alpha=0.15, color=colors[idx],
            label=f'±{error_pct}% Relative Error'
        )
    
    # Plot perfect prediction line (slope=1)
    perfect_line = np.array([axis_min, axis_max])
    ax.plot(perfect_line, perfect_line, 'r-', linewidth=2.5, label='Perfect Prediction', zorder=5)
    
    # Plot data points as black circles
    ax.scatter(actual, predicted, marker='o', s=50, color='black', 
               alpha=0.7, edgecolors='black', linewidth=0.5, label='Predictions', zorder=10)
    
    # Labels and formatting
    ax.set_xlabel('Actual Clogging Extent (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Clogging Extent (%)', fontsize=12, fontweight='bold')
    ax.set_title('Actual-to-Predicted (AtP) Scatter Plot\nClogging Extent Predictions', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Set equal aspect ratio to ensure slope=1 line appears at 45 degrees
    ax.set_aspect('equal', adjustable='box')
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Legend
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"  AtP scatter plot saved to {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-fold evaluation
# ---------------------------------------------------------------------------

def evaluate_fold(args, fold, test_dataset, eval_output_dir):
    """
    Run inference with a single fold's trained model on the full test dataset
    and return a metrics dict with the same structure as the per-fold entries
    in kfold_summary.json.

    The fold's model_final.pth is loaded from:
        <args.trained_models_dir>/fold_<fold>/model_final.pth

    Results are written to:
        <eval_output_dir>/fold_<fold>/test_metrics.json
        <eval_output_dir>/fold_<fold>/clogging_atp_scatter.png

    Parameters
    ----------
    args          : ConfigArgs   Configuration object.
    fold          : int          Zero-based fold index (0 to kfold-1).
    test_dataset  : str          Name of the registered test DatasetCatalog entry.
    eval_output_dir: str         Root directory for evaluation outputs.

    Returns
    -------
    dict  Metrics dict with keys "bbox", "segm", "clogging" (mirrors one
          element of the kfold_summary.json "per_fold" array).
    """
    fold_ckpt_dir = os.path.join(args.trained_models_dir, f"fold_{fold}")
    checkpoint    = os.path.join(fold_ckpt_dir, "model_final.pth")
    fold_out_dir  = os.path.join(eval_output_dir, f"fold_{fold}")
    os.makedirs(fold_out_dir, exist_ok=True)

    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found for fold {fold}:\n  {checkpoint}\n"
            "Ensure training has completed for all folds before running evaluation."
        )

    logger.info(f"  Loading checkpoint: {checkpoint}")

    cfg = build_eval_cfg(args, test_dataset, fold_ckpt_dir)
    # Override OUTPUT_DIR to write this fold's inference results into the
    # evaluation output directory rather than the training output directory.
    cfg = cfg.clone()
    cfg.defrost()
    cfg.OUTPUT_DIR = fold_out_dir
    cfg.freeze()

    default_setup(cfg, args)

    model = DefaultTrainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=fold_out_dir).resume_or_load(
        checkpoint, resume=False
    )
    model.eval()

    # Build test loader using the custom mapper (no augmentation, attaches GT
    # clogging_extent for metric (B)).
    test_loader = build_detection_test_loader(
        cfg,
        test_dataset,
        mapper=CloggingDatasetMapper(cfg, is_train=False),
    )

    evaluator = CloggingEvaluator(
        test_dataset,
        cfg,
        distributed=False,
        output_dir=os.path.join(fold_out_dir, "inference"),
    )

    results = inference_on_dataset(model, test_loader, evaluator)

    # Write per-fold results to disk
    metrics_path = os.path.join(fold_out_dir, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"  Test metrics written to {metrics_path}")

    # Generate AtP scatter plot using collected clogging pairs
    if hasattr(evaluator, '_clogging_pairs') and len(evaluator._clogging_pairs) > 0:
        atp_plot_path = os.path.join(fold_out_dir, "clogging_atp_scatter.png")
        # Use default error thresholds [10, 20]; modify as needed
        plot_clogging_atp_scatter(evaluator._clogging_pairs, atp_plot_path)
    else:
        logger.warning(
            f"No clogging pairs available for fold {fold}. "
            "AtP scatter plot was not generated."
        )

    return results


# ---------------------------------------------------------------------------
# Cross-fold aggregation
# ---------------------------------------------------------------------------

def evaluate_all_folds(args):
    """
    Evaluate each of the k trained fold models on the same held-out test
    dataset, then aggregate per-fold metrics into a summary with the same
    schema as kfold_summary.json.

    Aggregation
    -----------
    Mean and standard deviation are computed across the k fold results for:
        - segm AP   (IoU 0.5:0.95)
        - segm AP50 (IoU 0.50)
        - clogging MAE

    Output
    ------
    <args.eval_output_dir>/
      fold_<k>/
        inference/
          coco_instances_results.json
        test_metrics.json
      test_summary.json
    """
    os.makedirs(args.eval_output_dir, exist_ok=True)

    # Register the test dataset once; the same registration is reused across
    # all fold evaluation loops.
    register_test_dataset(
        "storm_drain_test",
        args.test_json,
        args.test_imgs,
    )

    all_results = []
    for fold in range(args.kfold):
        logger.info(f"\n{'='*60}")
        logger.info(f"  EVALUATING FOLD {fold + 1} / {args.kfold}  on held-out test set")
        logger.info(f"{'='*60}")

        results = evaluate_fold(args, fold, "storm_drain_test", args.eval_output_dir)
        all_results.append(results)

        logger.info(
            f"  Fold {fold}: "
            f"AP={results.get('segm', {}).get('AP',  float('nan')):.2f}  "
            f"AP50={results.get('segm', {}).get('AP50', float('nan')):.2f}  "
            f"Clogging MAE={results.get('clogging', {}).get('MAE', float('nan')):.4f}"
        )

    # Compute mean and std across folds, ignoring NaN entries (e.g. folds that
    # had no test instances in a particular size bucket).
    def _stats(values):
        clean = [v for v in values if not math.isnan(v)]
        if not clean:
            return float("nan"), float("nan")
        return float(np.mean(clean)), float(np.std(clean))

    ap_vals   = [r.get("segm", {}).get("AP",   float("nan")) for r in all_results]
    ap50_vals = [r.get("segm", {}).get("AP50",  float("nan")) for r in all_results]
    mae_vals  = [r.get("clogging", {}).get("MAE", float("nan")) for r in all_results]

    ap_mean,   ap_std   = _stats(ap_vals)
    ap50_mean, ap50_std = _stats(ap50_vals)
    mae_mean,  mae_std  = _stats(mae_vals)

    summary = {
        "num_folds":          args.kfold,
        "AP_mean":            ap_mean,
        "AP_std":             ap_std,
        "AP50_mean":          ap50_mean,
        "AP50_std":           ap50_std,
        "clogging_mae_mean":  mae_mean,
        "clogging_mae_std":   mae_std,
        "per_fold":           all_results,
    }

    summary_path = os.path.join(args.eval_output_dir, "test_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("TEST SET SUMMARY (across all fold models)")
    logger.info("=" * 60)
    logger.info(f"  Amodal AP   : {ap_mean:.2f} +/- {ap_std:.2f}")
    logger.info(f"  Amodal AP50 : {ap50_mean:.2f} +/- {ap50_std:.2f}")
    logger.info(f"  Clogging MAE: {mae_mean:.4f} +/- {mae_std:.4f}")
    logger.info(f"  Full summary written to {summary_path}")


# =============================================================================
# CONFIGURATION SETTINGS
# =============================================================================

class ConfigArgs:
    base_dir = "/run/user/1000/gvfs/smb-share:server=ecresearch.uwaterloo.ca,share=cviss/Jack/baf/360streetview"

    # Path to the unseen held-out test COCO JSON (same format as synthetic.json)
    test_json = os.path.join(base_dir, "segmentation_training", "original", "dataset", "custom.json")

    # Path to the directory containing the test images referenced in test_json
    test_imgs = os.path.join(base_dir, "segmentation_training", "original", "dataset","images")

    # Directory where training saved the fold checkpoints
    # (contains fold_0/model_final.pth, fold_1/model_final.pth, ...)
    trained_models_dir = os.path.join(base_dir, "trained_models", "synthetic")

    # Directory to write evaluation outputs (per-fold metrics + test_summary.json)
    # Safe to point at the same location as trained_models_dir; outputs are
    # written to per-fold subdirectories and will not overwrite training artifacts.
    eval_output_dir = os.path.join(base_dir, "trained_models", "synthetic", "real_test", "final")

    # Number of folds — must match the value used during training
    kfold = 5

    # Model configuration — must be identical to the config used at training time
    config_file = (
        "/home/cviss/jack/360streetview/"
        "Amodal-Segmentation-Based-on-Visible-Region-Segmentation-and-Shape-Prior/"
        "configs/StormDrain/mask_rcnn_R_50_FPN_1x_clogging.yaml"
    )

    # Distributed inference settings (single GPU by default)
    num_gpus    = 1
    num_machines = 1
    machine_rank = 0
    dist_url    = "auto"
    opts        = []


if __name__ == "__main__":
    args = ConfigArgs()
    """
    launch(
        evaluate_all_folds,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
    """
    # For single fold evaluation (fold 0)
    fold = 0
    test_dataset = "storm_drain_test"
    eval_output_dir = args.eval_output_dir
    # Register the test dataset before running evaluation
    register_test_dataset(
        test_dataset,
        args.test_json,
        args.test_imgs,
    )
    launch(
        evaluate_fold,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args, fold, test_dataset, eval_output_dir),
    )
