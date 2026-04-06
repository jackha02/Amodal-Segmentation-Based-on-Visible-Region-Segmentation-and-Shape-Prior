#   Single 80/20 train/val split:
#       cd <parent>/code
#       python train_storm_drain.py \
#           --all-json   /path/to/annotations/all.json \
#           --all-imgs   /path/to/images \
#           --output-dir ./output/storm_drain
#
#   5-fold cross-validation:
#       python train_storm_drain.py \
#           --all-json   /path/to/annotations/all.json \
#           --all-imgs   /path/to/images \
#           --output-dir ./output/storm_drain \
#           --kfold 5
#
#   Evaluation only from a saved checkpoint:
#       python train_storm_drain.py \
#           --all-json   /path/to/annotations/all.json \
#           --all-imgs   /path/to/images \
#           --output-dir ./output/storm_drain \
#           --eval-only \
#           --weights    ./output/storm_drain/model_final.pth
#
# =============================================================================
# DATASET ANNOTATION FORMAT
# =============================================================================
#
#   Standard COCO JSON with two extra per-instance fields:
#
#       "segmentation"         : polygon  -- the AMODAL (complete) mask
#       "visible_segmentation" : polygon  -- the VISIBLE portion of the mask
#       "clogging_extent"      : float in [0, 1]
#                                = 1 - (visible_pixel_area / amodal_pixel_area)
#
#   All other standard COCO fields (id, image_id, category_id, bbox, area,
#   iscrowd) must be present as normal.
#
# =============================================================================
# MODEL DESCRIPTION
# =============================================================================
#
#   This model adapts VRSP-Net (Xiao et al., AAAI 2021) for single-class
#   storm drain clogging extent estimation. Four modifications are made:
#
#   1. Reclassification head removed.
#      The original VRSP-Net uses a reclassification regulariser (L_rc) to
#      disambiguate multi-class predictions using visible region features.
#      This is unnecessary for a single-class task and is removed, following
#      Kim et al. (2023) who made the same omission for their single-class
#      cucumber dataset.
#
#   2. Auto-encoder + codebook shape prior replaced with U-Net reconstruction.
#      The original Module 3 of VRSP-Net uses a pre-trained auto-encoder with
#      a K-Means codebook to supply category-specific shape prior embeddings.
#      Kim et al. (2023) showed that for single-class objects with high
#      intraclass shape variance, U-Net outperforms the auto-encoder in both
#      accuracy (AP 50.06 vs 49.31) and inference speed (220 ms vs 233 ms).
#      Storm drain inlets share the same physical geometry but vary in apparent
#      shape across viewpoints and scales, motivating the same substitution.
#      The U-Net receives the coarse amodal mask M^c_a as input and produces
#      a same-sized refined amodal logit, concatenated with the visible-
#      attention features F * M^r_v before the final amodal mask head.
#
#   3. Clogging extent loss added (quality-gated MAE).
#      A new loss term penalises the error in the predicted clogging ratio:
#
#         gamma_pred_i = sum(M^r_a_i - M^r_v_i) / sum(M^r_a_i)
#         L_clog = (1/N) * sum_i  w_i * |gamma_pred_i - gamma_gt_i|
#
#      where w_i = soft_IoU(M^r_a_i, M^g_a_i) is detached from the gradient
#      graph and acts as a quality gate, preventing the model from achieving a
#      low ratio error via a geometrically incorrect amodal mask.
#      MAE is used because the clogging extent target is approximately
#      uniformly distributed in [0, 1], making the constant-gradient L1 loss
#      more appropriate than the quadratically-penalising L2 loss.
#
#   4. Automatic loss weighting via homoscedastic uncertainty.
#      All eight learnable loss terms are automatically weighted by learning
#      per-task log-variance scalars s_k = log(sigma_k^2), eliminating manual
#      lambda hyperparameter search (Kendall, Gal & Cipolla, CVPR 2018):
#
#         cls-type (BCE): exp(-s_k) * L_k  +  0.5 * s_k
#         reg-type (MAE): 0.5 * exp(-s_k) * L_k  +  0.5 * s_k
#
# =============================================================================
# EVALUATION METRICS AND OUTPUT FILES
# =============================================================================
#
#   Two metrics are reported at the end of each fold (or single run):
#
#   (A) Amodal mask mAP
#       Computed by COCOEvaluator against the amodal ground-truth masks stored
#       in the standard "segmentation" field. Reports the full COCO AP suite:
#       AP (IoU 0.5:0.95), AP50, AP75, APs, APm, APl. These are identical to
#       the metrics in Xiao et al. (2021, Table 1) under "Amodal AP", enabling
#       direct comparison with the baseline.
#
#   (B) Clogging extent MAE
#       Mean absolute error between pred_clogging_extent (set by the mask head
#       inference path in mask_head_clogging.py) and gt_clogging_extent (loaded
#       from the "clogging_extent" annotation field by the dataset mapper).
#       Accumulated instance-wise across the full validation fold and averaged.
#
#   Output files written per fold (and for single runs):
#
#       <output_dir>/
#         fold_<k>/                         (fold_0/, fold_1/, ...)
#           inference/
#             coco_instances_results.json   <- raw COCO-format predictions
#             metrics.json                  <- COCOEvaluator AP metrics
#           metrics.json                    <- AP + clogging MAE (combined)
#           model_final.pth                 <- trained weights for this fold
#           log.txt                         <- full training + evaluation log
#
#   For k-fold runs, an additional summary file is written:
#
#       <output_dir>/
#         kfold_summary.json                <- mean +/- std across all folds
#
#   Reading results in Python:
#       import json
#       # Per-fold results
#       r = json.load(open("<output_dir>/fold_0/metrics.json"))
#       print(r["segm"]["AP"])          # amodal mask AP (IoU 0.5:0.95)
#       print(r["segm"]["AP50"])        # amodal mask AP50
#       print(r["clogging_mae"])        # clogging extent MAE
#
#       # Cross-validation summary
#       s = json.load(open("<output_dir>/kfold_summary.json"))
#       print(s["AP_mean"], s["AP_std"])
#       print(s["clogging_mae_mean"], s["clogging_mae_std"])
# =============================================================================

import json
import logging
import math
import os
import random
import sys
import copy
from copy import deepcopy

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
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, default_setup, launch
from detectron2.evaluation import COCOEvaluator
from detectron2.config import add_clogging_config 
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.structures import Instances, BitMasks      # remove unused Polygons import
import detectron2.modeling.roi_heads
from detectron2.modeling.roi_heads import (
    ROI_HEADS_REGISTRY,
    StandardROIHeads,
    select_foreground_proposals,
)
from detectron2.structures import pairwise_iou


# ---------------------------------------------------------------------------
# Custom ROI Heads — passes `instances` to the clogging mask head
# ---------------------------------------------------------------------------
# StandardROIHeads._forward_mask calls self.mask_head(mask_features) with a
# single tensor argument.  ParallelAmodalMaskHeadClogging.forward() requires
# (roi_features, instances) so the head can:
#   • crop GT masks from PolygonMasks using proposal_boxes (training)
#   • split predictions back into per-image Instances objects (inference)
# This subclass overrides only _forward_mask; everything else (box head,
# pooler init, optimizer, etc.) is inherited unchanged.
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
            # Keep only foreground proposals (same as StandardROIHeads).
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            mask_features  = self.mask_pooler(features, proposal_boxes)
            # Pass proposals so the head can access GT masks & proposal_boxes.
            return self.mask_head(mask_features, proposals)
        else:
            pred_boxes    = [x.pred_boxes for x in instances]
            mask_features = self.mask_pooler(features, pred_boxes)
            # Pass instances so the head can attach predictions per image.
            return self.mask_head(mask_features, instances)

logger = logging.getLogger("detectron2")

def load_coco_json(json_path):
    """
    Load and return a parsed COCO annotation JSON dict
    """
    with open(json_path, "r") as f:
        return json.load(f)

def split_coco_json_kfold(coco_data, k, fold, seed=42):
    """
    Split a COCO annotation dict into train/val subsets for one fold of
    k-fold cross-validation

    The split is performed at the image level so that all annotation instances
    belonging to an image stay together in the same split.

    Parameters
    ----------
    coco_data : dict   Full COCO annotation dict.
    k         : int    Total number of folds.
    fold      : int    Index of the validation fold (0-indexed, 0 to k-1).
    seed      : int    Random seed for reproducible shuffling across runs.

    Returns
    -------
    train_data : dict   COCO dict for the training split.
    val_data   : dict   COCO dict for the validation split.
    """
    images = coco_data["images"]
    annotations = coco_data["annotations"]

    rng = random.Random(seed)
    image_ids = [img["id"] for img in images]
    rng.shuffle(image_ids)

    fold_size = math.ceil(len(image_ids) / k)
    val_ids   = set(image_ids[fold * fold_size: (fold + 1) * fold_size])
    train_ids = set(image_ids) - val_ids

    def _subset(id_set):
        data = deepcopy(coco_data)
        data["images"]      = [img for img in images      if img["id"]       in id_set]
        data["annotations"] = [ann for ann in annotations if ann["image_id"] in id_set]
        return data

    return _subset(train_ids), _subset(val_ids)

def write_coco_json(coco_data, path):
    """
    Serialise a COCO dict to disk
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(coco_data, f)

def register_fold_datasets(train_data, val_data, img_dir, fold, tmp_dir):
    """
    Write per-fold COCO JSONs to tmp_dir, register them with Detectron2's
    DatasetCatalog, and return (train_name, val_name).

    DatasetCatalog.remove() is called before re-registering to allow the
    function to be called multiple times across folds without raising a
    "dataset already registered" error.
    """
    train_json = os.path.join(tmp_dir, f"fold_{fold}_train.json")
    val_json   = os.path.join(tmp_dir, f"fold_{fold}_val.json")
    write_coco_json(train_data, train_json)
    write_coco_json(val_data,   val_json)

    train_name = f"storm_drain_fold{fold}_train"
    val_name   = f"storm_drain_fold{fold}_val"

    for name, path in [(train_name, train_json), (val_name, val_json)]:
        if name in DatasetCatalog.list():
            DatasetCatalog.remove(name)
        register_coco_instances(name, {}, path, img_dir)
        MetadataCatalog.get(name).set(thing_classes=["storm_drain"])
        logger.info(f"Registered '{name}'  ({path})")

    return train_name, val_name

def build_custom_transforms(is_train):
    """
    Returns a list of Transform objects for data augmentation.
    """
    if is_train:
        return [
            # Geometric augmentations
            T.RandomApply(T.RandomRotation(angle=[-30, 30]), prob=1.0),  # ±30° rotation
            T.RandomApply(T.RandomExtent(scale_range=(0.8, 1.2), shift_range=(0.0, 0.0)), prob=1.0),  # ±20% scaling, no translation
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False),  # 50% horizontal flip
            T.RandomFlip(prob=0.5, horizontal=False, vertical=True),  # 50% vertical flip

            # Photometric augmentations
            T.RandomBrightness(intensity_min=0.7, intensity_max=1.3),  # ±30% brightness
            T.RandomSaturation(intensity_min=0.7, intensity_max=1.3),  # ±30% saturation
            T.RandomLighting(intensity_min=0.985, intensity_max=1.015), # ±1.5% hue (approximate)
        ]
    else:
        return [
            T.ResizeShortestEdge(short_edge_length=640, max_size=640)  # consistent max_size
        ]
    
class CloggingDatasetMapper:
    """
    A custom mapper that extends the default Detectron2 mapper to also parse
    'visible_segmentation' and 'clogging_extent' into the target Instances
    """
    def __init__(self, cfg, is_train=True):
        self.is_train = is_train
        self.image_format = cfg.INPUT.FORMAT
        self.tfm_gens = build_custom_transforms(is_train)

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)

        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)

        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if not self.is_train and not dataset_dict.get("annotations"):
            return dataset_dict

        annos = dataset_dict.pop("annotations")
        annos = [obj for obj in annos if obj.get("iscrowd", 0) == 0]

        # Apply geometric transforms to standard fields (bbox, amodal segmentation)
        annos = [
            utils.transform_instance_annotations(obj, transforms, image.shape[:2])
            for obj in annos
        ]

        # Map visible_segmentation -> visible_mask (expected by the repo's annotations_to_instances)
        for obj in annos:
            if "visible_mask" not in obj:
                obj["visible_mask"] = obj.get("visible_segmentation", obj.get("segmentation", []))

        # Build standard instances (uses "segmentation" as the amodal mask)
        instances = utils.annotations_to_instances(annos, image.shape[:2])

        # --- Parse visible_segmentation into gt_masks_visible ---
        # Rename amodal masks (built from "segmentation") to gt_masks_amodal
        if instances.has("gt_masks"):
            instances.gt_masks_amodal = instances.gt_masks
            instances.remove("gt_masks")  # remove the generic field to avoid ambiguity

        visible_polygons = []
        clogging_extents = []
        for obj in annos:
            # visible_segmentation uses the same polygon format as segmentation
            vis_seg = obj.get("visible_segmentation", obj.get("segmentation", []))
            visible_polygons.append(vis_seg)
            clogging_extents.append(float(obj.get("clogging_extent", 0.0)))

        # Convert visible polygons to BitMasks (same spatial size as amodal)
        h, w = image.shape[:2]
        from detectron2.structures import PolygonMasks
        vis_masks = PolygonMasks(visible_polygons)
        instances.gt_masks_visible = vis_masks

        # Attach clogging extents as a float tensor
        instances.gt_clogging_extent = torch.tensor(clogging_extents, dtype=torch.float32)

        dataset_dict["instances"] = instances

        return dataset_dict

# Model performance evaluator 
class CloggingEvaluator(COCOEvaluator):
    """
    Extends Detectron2's COCOEvaluator with clogging extent MAE.

    Metric (A) — Amodal mask mAP
    -----------------------------
    Inherited unchanged from COCOEvaluator. Predicted instance masks are
    matched to ground-truth amodal masks (the standard "segmentation" field
    in the COCO JSON) using the COCO API's IoU-based matching at IoU
    thresholds from 0.5 to 0.95 in steps of 0.05. The following metrics are
    reported under the "segm" key of the results dict:

        AP    — mean AP over IoU thresholds 0.50:0.05:0.95
        AP50  — AP at IoU threshold 0.50
        AP75  — AP at IoU threshold 0.75
        APs   — AP for small instances  (area < 32^2 pixels)
        APm   — AP for medium instances (32^2 < area < 96^2 pixels)
        APl   — AP for large instances  (area > 96^2 pixels)

    These are the same metrics reported by Xiao et al. (2021, Table 1) under
    "Amodal AP", enabling direct comparison with the published baseline.

    Metric (B) — Clogging extent MAE
    ----------------------------------
    In process(), pred_clogging_extent is read from each predicted Instances
    object (populated by ParallelAmodalMaskHeadClogging._forward_inference),
    and gt_clogging_extent is read from the ground-truth Instances object
    (populated by the dataset mapper from the "clogging_extent" annotation
    field). Per-instance absolute errors are accumulated in
    self._clogging_errors. In evaluate(), the mean is computed over all
    collected errors and stored under the "clogging_mae" key of the results
    dict alongside the standard "segm" metrics.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._clogging_errors = []

    def reset(self):
        super().reset()
        self._clogging_errors = []

    def process(self, inputs, outputs):
        # Metric (A): delegate to COCOEvaluator
        super().process(inputs, outputs)

        # Metric (B): accumulate per-instance absolute clogging errors
        for inp, out in zip(inputs, outputs):
            pred_inst = out.get("instances", None)
            if pred_inst is None or not pred_inst.has("pred_clogging_extent"):
                continue
            gt_inst = inp.get("instances", None)
            if gt_inst is None or not gt_inst.has("gt_clogging_extent"):
                continue

            if len(pred_inst) == 0 or len(gt_inst) == 0:
                continue

            # Handle spatial matching using IoU for images with multiple drains
            iou_matrix = pairwise_iou(pred_inst.pred_boxes, gt_inst.gt_boxes)
            
            pred_clog = pred_inst.pred_clogging_extent.cpu()
            gt_clog   = gt_inst.gt_clogging_extent.cpu()
            
            max_ious, matched_pred_indices = iou_matrix.max(dim=0)
            
            for gt_idx, pred_idx in enumerate(matched_pred_indices):
                if max_ious[gt_idx] > 0.5:
                    error = abs(float(pred_clog[pred_idx] - gt_clog[gt_idx]))
                    self._clogging_errors.append(error)

    def evaluate(self):
        # Metric (A): COCO AP suite under "segm" key
        results = super().evaluate() or {}

        # Metric (B): clogging MAE appended to the same dict
        if self._clogging_errors:
            mae = float(np.mean(self._clogging_errors))
            logger.info(
                f"Clogging extent MAE: {mae:.4f}  "
                f"({len(self._clogging_errors)} instances)"
            )
            results["clogging_mae"] = mae
        else:
            logger.warning(
                "No clogging extent predictions collected. "
                "Verify that gt_clogging_extent is set by the dataset mapper "
                "and pred_clogging_extent is set by the mask head inference path."
            )
            results["clogging_mae"] = float("nan")

        return results

# Model trainer

class StormDrainTrainer(DefaultTrainer):
    """
    DefaultTrainer subclass with CloggingEvaluator.

    No other customisation is needed:
    - ParallelAmodalMaskHeadClogging is resolved from Detectron2's registry
      via the MODEL.ROI_MASK_HEAD.NAME config node.
    - The UncertaintyWeightedLoss log_var parameters are nn.Parameters inside
      a registered sub-module and are automatically picked up by
      build_optimizer() without any extra configuration.
    """
    @classmethod
    def build_train_loader(cls, cfg):
        # Force the trainer to use your custom mapper so it doesn't delete the visible masks
        mapper = CloggingDatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        mapper = CloggingDatasetMapper(cfg, is_train=False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return CloggingEvaluator(
            dataset_name,
            cfg,
            distributed=True,
            output_dir=output_folder,
        )

# Config builder

def build_cfg(args, train_dataset, val_dataset, output_dir):
    """
    Build a Detectron2 CfgNode for one training run or one fold.
    """
    cfg = get_cfg()
    add_clogging_config(cfg) # add the custom loss function (i.e., clogging extent MAE)

    # 
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)

    cfg.defrost()
    cfg.MODEL.DEVICE                = "cuda"
    cfg.VIS_PERIOD                  = 0     # disable visualisation (gt_masks renamed to gt_masks_amodal)
    cfg.MODEL.ROI_HEADS.NAME        = "CloggingROIHeads"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_MASK_HEAD.NAME    = "ParallelAmodalMaskHeadClogging"
    cfg.MODEL.MASK_ON               = True
    cfg.DATASETS.TRAIN              = (train_dataset,)
    cfg.DATASETS.TEST               = (val_dataset,)
    cfg.OUTPUT_DIR                  = output_dir

    if args.weights:
        cfg.MODEL.WEIGHTS = args.weights
    elif not cfg.MODEL.WEIGHTS or cfg.MODEL.WEIGHTS.startswith("detectron2://"):
        cfg.MODEL.WEIGHTS = os.path.join(os.path.dirname(SCRIPT_DIR), "R-50.pkl")

    cfg.freeze()
    return cfg

# Single training + evaluation run

def run_single_fold(args, train_dataset, val_dataset, output_dir):
    """
    Train on train_dataset, evaluate on val_dataset, write metrics.json,
    and return the results dict.

    The returned dict has the structure:
        {
          "segm": {
              "AP": float, "AP50": float, "AP75": float,
              "APs": float, "APm": float, "APl": float
          },
          "clogging_mae": float
        }

    The same dict is also written to <output_dir>/metrics.json for later
    access without rerunning the script.
    """
    os.makedirs(output_dir, exist_ok=True)
    cfg = build_cfg(args, train_dataset, val_dataset, output_dir)
    default_setup(cfg, args)

    if args.eval_only:
        model = StormDrainTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=output_dir).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=False
        )
        results = StormDrainTrainer.test(cfg, model)
    else:
        trainer = StormDrainTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()
        results = StormDrainTrainer.test(cfg, trainer.model)

    # Write combined metrics (AP + clogging MAE) to disk
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Metrics written to {metrics_path}")

    return results

# K-fold cross-validation

def run_kfold(args):
    """
    Run k-fold cross-validation over the full annotated dataset and write
    a summary JSON with mean and standard deviation across folds.

    Design decisions
    ----------------
    - Image-level splitting prevents data leakage (see split_coco_json_kfold).
    - A fresh model is initialised from the same ImageNet weights at the start
      of every fold, ensuring fully independent training runs.
    - The same random seed is used for every fold's shuffle so that the fold
      assignment is deterministic and reproducible. Each fold uses a different
      held-out subset by varying the `fold` index, not the seed.

    Output
    ------
    Per-fold results in <output_dir>/fold_<i>/metrics.json, plus:

        <output_dir>/kfold_summary.json:
        {
          "num_folds":          int,
          "AP_mean":            float,
          "AP_std":             float,
          "AP50_mean":          float,
          "AP50_std":           float,
          "clogging_mae_mean":  float,
          "clogging_mae_std":   float,
          "per_fold": [
            {"segm": {...}, "clogging_mae": float},   # fold 0
            ...
          ]
        }
    """
    k = args.kfold
    logger.info(f"Starting {k}-fold cross-validation  (seed={args.seed})")

    coco_data = load_coco_json(args.all_json)
    tmp_dir   = os.path.join(args.output_dir, "_fold_jsons")
    all_results = []

    for fold in range(k):
        logger.info(f"\n{'='*60}")
        logger.info(f"  FOLD {fold + 1} / {k}")
        logger.info(f"{'='*60}")

        train_data, val_data = split_coco_json_kfold(
            coco_data, k=k, fold=fold, seed=args.seed
        )
        logger.info(
            f"  Train: {len(train_data['images'])} images  "
            f"Val: {len(val_data['images'])} images"
        )

        train_name, val_name = register_fold_datasets(
            train_data, val_data, args.all_imgs,
            fold=fold, tmp_dir=tmp_dir,
        )

        fold_dir = os.path.join(args.output_dir, f"fold_{fold}")
        results  = run_single_fold(args, train_name, val_name, fold_dir)
        all_results.append(results)

        logger.info(
            f"  Fold {fold}: "
            f"AP={results.get('segm', {}).get('AP',  float('nan')):.2f}  "
            f"AP50={results.get('segm', {}).get('AP50', float('nan')):.2f}  "
            f"Clogging MAE={results.get('clogging_mae', float('nan')):.4f}"
        )

    # ---- Compute mean and std across folds ---------------------------------
    def _stats(values):
        clean = [v for v in values if not math.isnan(v)]
        if not clean:
            return float("nan"), float("nan")
        return float(np.mean(clean)), float(np.std(clean))

    ap_vals   = [r.get("segm", {}).get("AP",   float("nan")) for r in all_results]
    ap50_vals = [r.get("segm", {}).get("AP50",  float("nan")) for r in all_results]
    mae_vals  = [r.get("clogging_mae", float("nan")) for r in all_results]

    ap_mean,   ap_std   = _stats(ap_vals)
    ap50_mean, ap50_std = _stats(ap50_vals)
    mae_mean,  mae_std  = _stats(mae_vals)

    summary = {
        "num_folds":          k,
        "AP_mean":            ap_mean,
        "AP_std":             ap_std,
        "AP50_mean":          ap50_mean,
        "AP50_std":           ap50_std,
        "clogging_mae_mean":  mae_mean,
        "clogging_mae_std":   mae_std,
        "per_fold":           all_results,
    }

    summary_path = os.path.join(args.output_dir, "kfold_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("K-FOLD SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Amodal AP   : {ap_mean:.2f} +/- {ap_std:.2f}")
    logger.info(f"  Amodal AP50 : {ap50_mean:.2f} +/- {ap50_std:.2f}")
    logger.info(f"  Clogging MAE: {mae_mean:.4f} +/- {mae_std:.4f}")
    logger.info(f"  Full summary written to {summary_path}")


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    if args.kfold > 1:
        # K-fold cross-validation
        run_kfold(args)

    else:
        # Single 80/20 train/val split (kfold=0 or kfold=1)
        logger.info("Running single train/val split (no k-fold).")
        coco_data = load_coco_json(args.all_json)
        # Use fold 0 of a 5-fold split to obtain an approximately 80/20 split.
        train_data, val_data = split_coco_json_kfold(
            coco_data, k=5, fold=0, seed=args.seed
        )
        tmp_dir = os.path.join(args.output_dir, "_split_jsons")
        train_name, val_name = register_fold_datasets(
            train_data, val_data, args.all_imgs, fold=99, tmp_dir=tmp_dir
        )
        results = run_single_fold(args, train_name, val_name, args.output_dir)

        logger.info("\nFinal results:")
        logger.info(f"  Amodal AP   : {results.get('segm', {}).get('AP',  'N/A')}")
        logger.info(f"  Amodal AP50 : {results.get('segm', {}).get('AP50','N/A')}")
        logger.info(f"  Clogging MAE: {results.get('clogging_mae','N/A')}")

# CONFIGURATION SETTINGS
class ConfigArgs:
    base_dir = "/run/user/1000/gvfs/smb-share:server=ecresearch.uwaterloo.ca,share=cviss/Jack/baf/360streetview/segmentation_training"
    # Required Paths
    all_json   = os.path.join(base_dir, 'original', 'dataset', 'custom.json')  # Path to the merged COCO annotation JSON (annotations_all.json)
    all_imgs   = os.path.join(base_dir, 'original', 'dataset', 'images')       # Path to the directory containing training images
    output_dir = "/home/wonny/baf/360streetview/trained_models/segmentation"
    
    # Training Mode
    kfold = 5
    seed = 42
    
    # Model parameters
    config_file = "/home/wonny/baf/360streetview/Amodal-Segmentation-Based-on-Visible-Region-Segmentation-and-Shape-Prior/configs/StormDrain/mask_rcnn_R_50_FPN_1x_clogging.yaml" # defualts to a ResNet-50 backbone
    weights = ""
    eval_only = False # set to True to immediately run inference and evaluation
    
    # Distributed training
    num_gpus = 1
    num_machines = 1
    machine_rank = 0
    dist_url = "auto"
    opts = []

if __name__ == "__main__":
    args = ConfigArgs()
    launch(
        main,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )