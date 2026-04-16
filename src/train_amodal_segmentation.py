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
import cv2

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.evaluation import inference_on_dataset

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
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
from detectron2.engine.hooks import HookBase


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


# ---------------------------------------------------------------------------
# Custom RandomRotation Transform — preserves both visible and amodal masks
# ---------------------------------------------------------------------------

class RandomRotation(T.Transform):
    """
    Custom rotation transform for Detectron2 v0.1 that rotates images and
    all associated polygons (both amodal and visible segmentation masks).
    
    Parameters
    ----------
    angle : float
        Rotation angle in degrees (positive = counter-clockwise)
    expand : bool
        If True, expand output image to fit the rotated content (may create
        canvas larger than input). If False, crop to input size.
    """
    def __init__(self, angle, expand=False):
        super().__init__()
        self.angle = angle
        self.expand = expand
        self._original_shape = None  # Will store (h, w) during apply_image
        self._set_attributes(locals())
    
    def apply_image(self, img):
        """Rotate image using cv2.warpAffine"""
        h, w = img.shape[:2]
        self._original_shape = (h, w)  # Store for use in apply_coords
        center = (w / 2.0, h / 2.0)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, self.angle, 1.0)
        
        if self.expand:
            # Compute the size of the rotated image
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            new_w = int(h * sin + w * cos)
            new_h = int(h * cos + w * sin)
            
            # Adjust rotation matrix to account for translation
            M[0, 2] += (new_w / 2.0) - center[0]
            M[1, 2] += (new_h / 2.0) - center[1]
            
            # Rotate and pad to new size
            rotated = cv2.warpAffine(img, M, (new_w, new_h),
                                    borderMode=cv2.BORDER_REFLECT_101)
            self._new_shape = (new_h, new_w)
            return rotated
        else:
            # Rotate and keep original size
            rotated = cv2.warpAffine(img, M, (w, h),
                                    borderMode=cv2.BORDER_REFLECT_101)
            self._new_shape = (h, w)
            return rotated
    
    def _get_rotation_matrix(self):
        """Compute the rotation matrix for coordinate transformation"""
        if self._original_shape is None:
            raise RuntimeError("apply_image must be called before apply_coords")
        
        h, w = self._original_shape
        center = np.array([w / 2.0, h / 2.0])
        
        M = cv2.getRotationMatrix2D(tuple(center), self.angle, 1.0)
        
        if self.expand:
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            new_w = int(h * sin + w * cos)
            new_h = int(h * cos + w * sin)
            
            M[0, 2] += (new_w / 2.0) - center[0]
            M[1, 2] += (new_h / 2.0) - center[1]
        
        return M
    
    def apply_coords(self, coords):
        """Rotate coordinate points using the rotation matrix"""
        if len(coords) == 0:
            return coords
        
        M = self._get_rotation_matrix()
        
        # Apply rotation to coordinates
        # Convert to homogeneous coordinates
        ones = np.ones((coords.shape[0], 1))
        coords_h = np.hstack([coords, ones])  # shape: (N, 3)
        
        # Apply transformation: (2x3) @ (3xN) = (2xN)
        rotated_coords = (M @ coords_h.T).T  # shape: (N, 2)
        
        # Clip to valid image range if not expanding
        if not self.expand and self._new_shape is not None:
            h, w = self._new_shape
            rotated_coords = np.clip(rotated_coords, 0, [w, h])
        
        return rotated_coords
    
    def apply_segmentation(self, segmentation):
        """
        Rotate polygon segmentation.
        Detectron2 uses list of lists format: [[x0, y0, x1, y1, ...], ...]
        """
        if not segmentation:
            return segmentation
        
        rotated_segs = []
        for polygon in segmentation:
            if len(polygon) < 6:  # Skip degenerate polygons
                rotated_segs.append(polygon)
                continue
            
            # Convert to coordinate array
            coords = np.array(polygon, dtype=np.float32).reshape(-1, 2)
            
            # Rotate coordinates
            rotated = self.apply_coords(coords)
            
            # Convert back to flat list
            rotated_segs.append(rotated.flatten().tolist())
        
        return rotated_segs


class ApplyRandomRotation(T.TransformGen):
    """
    Augmentation wrapper for RandomRotation that samples random angles
    (Detectron2 v0.1 compatible).
    
    Parameters
    ----------
    angle_range : tuple of (min_angle, max_angle)
        Range of rotation angles in degrees
    p : float
        Probability of applying rotation (0.0 to 1.0)
    expand : bool
        If True, expand canvas to fit rotated image; if False, keep original size
    """
    def __init__(self, angle_range=(-15, 15), p=0.5, expand=False):
        super().__init__()
        self.angle_range = angle_range
        self.prob = p
        self.expand = expand
        self._init(locals())
    
    def get_transform(self, image):
        if self._rand_range() > self.prob:
            # Return identity transform (no rotation)
            return T.NoOpTransform()
        
        angle = np.random.uniform(self.angle_range[0], self.angle_range[1])
        return RandomRotation(angle, expand=self.expand)


class ApplyRandomVerticalFlip(T.TransformGen):
    """
    Vertical flip augmentation (Detectron2 v0.1 compatible).
    
    Handles vertical flipping of images and associated polygon annotations
    to account for camera tilt variations. This is distinct from horizontal
    flipping and helps the model to be invariant to vertical camera angles.
    
    Parameters
    ----------
    prob : float
        Probability of applying vertical flip (0.0 to 1.0), default 0.5
    """
    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob
        self._init(locals())
    
    def get_transform(self, image):
        if self._rand_range() > self.prob:
            return T.NoOpTransform()
        h = image.shape[0]
        return T.VFlipTransform(h)


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

    for name, path, in [(train_name, train_json), (val_name, val_json)]:
        if name in DatasetCatalog.list():
            DatasetCatalog.remove(name)
        # Use a custom loader instead of register_coco_instances so that
        # non-standard annotation fields (clogging_extent, visible_segmentation)
        # are preserved — register_coco_instances uses Detectron2's COCO loader
        # which silently drops any field not in the official COCO spec.
        _path, _img_dir = path, img_dir  # capture loop vars
        def _make_loader(json_path, images_dir):
            def _loader():
                with open(json_path) as f:
                    data = json.load(f)
                id_to_img = {img["id"]: img for img in data["images"]}
                anns_by_image = {}
                for ann in data["annotations"]:
                    anns_by_image.setdefault(ann["image_id"], []).append(ann)
                dataset_dicts = []
                for img in data["images"]:
                    img_id = img["id"]
                    record = {
                        "file_name":    os.path.join(images_dir, img["file_name"]),
                        "image_id":     img_id,
                        "height":       img["height"],
                        "width":        img["width"],
                        "annotations":  [],
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
                        # Preserve custom fields
                        if "clogging_extent" in ann:
                            obj["clogging_extent"] = ann["clogging_extent"]
                        if "visible_segmentation" in ann:
                            obj["visible_segmentation"] = ann["visible_segmentation"]
                        record["annotations"].append(obj)
                    dataset_dicts.append(record)
                return dataset_dicts
            return _loader
        DatasetCatalog.register(name, _make_loader(_path, _img_dir))
        MetadataCatalog.get(name).set(thing_classes=["storm_drain"])
        logger.info(f"Registered '{name}'  ({path})")

    return train_name, val_name

def build_custom_transforms(is_train):
    """
    Returns a list of Transform objects for data augmentation.
    
    CRITICAL: Includes scale augmentation to handle domain shift between
    synthetic training (245px inlets) and real test (315px inlets).
    
    Augmentations include:
    - Scale: ±30% to handle domain shift (synthetic 245px vs real 315px inlets)
    - Rotation: ±15 degrees (50% probability) to handle viewing angle variations
    - Scaling: ±20% to handle focal length variations and scale mismatches
    - Horizontal flip: 50% probability for natural scene augmentation
    - Vertical flip: 50% probability to handle camera angle variations
    - Photometric: brightness, saturation, and lighting adjustments
    """
    if is_train:
        return [
            # ===== SCALE AUGMENTATION (CRITICAL FOR GENERALIZATION) =====
            # Synthetic inlets: ~245x245 px (area 60K px²)
            # Real inlets: ~315x315 px (area 99K px²) = 65% LARGER
            # Use T.RandomExtent with ±30% scale range (0.7-1.3)
            T.RandomExtent(scale_range=(0.7, 1.3), shift_range=(0.0, 0.0)),
            # ================================================================
            
            # Geometric augmentations
            ApplyRandomRotation(angle_range=(-15, 15), p=0.5, expand=False),  # Rotation to mimic viewpoint variations
            T.RandomFlip(prob=0.5),          # 50% horizontal flip
            ApplyRandomVerticalFlip(prob=0.5),  # 50% vertical flip to handle camera tilt variations

            # Photometric augmentations
            T.RandomBrightness(intensity_min=0.7, intensity_max=1.3),         # ±30% brightness
            T.RandomSaturation(intensity_min=0.7, intensity_max=1.3),         # ±30% saturation
            T.RandomLighting(scale=0.03)                                     # ±3% lighting variation (approximate)
        ]
    else:
        return []
    
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

        # Map visible_segmentation -> visible_mask BEFORE transforms so it gets
        # spatially transformed alongside "segmentation" and "bbox"
        for obj in annos:
            if "visible_mask" not in obj:
                obj["visible_mask"] = obj.get("visible_segmentation", obj.get("segmentation", []))

        # Apply geometric transforms to standard fields (bbox, amodal segmentation, visible_mask)
        # ===== CRITICAL: transform_instance_annotations ALSO handles visible_mask polygons =====
        # We must keep visible_mask as polygons for transforms to work, then pop before annotations_to_instances
        transformed_annos = []
        for obj in annos:
            obj_transformed = utils.transform_instance_annotations(obj, transforms, image.shape[:2])
            transformed_annos.append(obj_transformed)
        
        annos = transformed_annos

        # ===== CRITICAL: Extract visible_mask but KEEP it in obj for detection_utils =====
        # detection_utils.annotations_to_instances expects visible_mask to exist (line 261)
        # We extract it to handle manually, but also need to keep it for detection_utils
        # Otherwise: KeyError at detection_utils.py:261 trying to access obj["visible_mask"]
        
        extracted_visible_masks = []
        for obj in annos:
            vis_mask = obj.get("visible_mask", [])  # Extract a copy, keep original in obj
            extracted_visible_masks.append(vis_mask)

        # Build standard instances (only processes "segmentation" as amodal mask)
        instances = utils.annotations_to_instances(annos, image.shape[:2])


        # --- Parse visible_segmentation into gt_masks_visible ---
        # Rename amodal masks (built from "segmentation") to gt_masks_amodal
        if instances.has("gt_masks"):
            instances.gt_masks_amodal = instances.gt_masks
            instances.remove("gt_masks")  # remove the generic field to avoid ambiguity

        # Extract clogging extents
        clogging_extents = []
        for obj in annos:
            clogging_extents.append(float(obj.get("clogging_extent", 0.0)))

        # ===== Manually convert extracted visible_mask polygons to binary masks =====
        h, w = image.shape[:2]
        
        visible_masks_binary = []
        for vis_mask in extracted_visible_masks:
            if vis_mask is not None:
                # vis_mask should be a polygon list from transform_instance_annotations
                if isinstance(vis_mask, np.ndarray):
                    # Already a binary mask - keep as is
                    if vis_mask.ndim == 2:
                        visible_masks_binary.append(vis_mask.astype(np.uint8))
                    else:
                        logger.warning(f"visible_mask unexpected shape {vis_mask.shape}; using zero mask")
                        visible_masks_binary.append(np.zeros((h, w), dtype=np.uint8))
                elif isinstance(vis_mask, list) and len(vis_mask) > 0:
                    # It's a polygon list - rasterize to binary mask
                    try:
                        binary_mask = np.zeros((h, w), dtype=np.uint8)
                        # vis_mask is a list of polygons, each polygon is [[x1,y1], [x2,y2], ...]
                        for poly in vis_mask:
                            if isinstance(poly, np.ndarray):
                                poly = poly.tolist()
                            if isinstance(poly, list) and len(poly) >= 6:  # Minimum 3 points
                                pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                                cv2.fillPoly(binary_mask, [pts], 1)
                        visible_masks_binary.append(binary_mask)
                    except Exception as e:
                        logger.warning(f"Failed to rasterize visible_mask: {e}; using zero mask")
                        visible_masks_binary.append(np.zeros((h, w), dtype=np.uint8))
                else:
                    # Empty or invalid visible_mask
                    visible_masks_binary.append(np.zeros((h, w), dtype=np.uint8))
            else:
                visible_masks_binary.append(np.zeros((h, w), dtype=np.uint8))

        # Create BitMasks for visible masks
        if visible_masks_binary:
            vis_masks_stack = torch.stack([
                torch.from_numpy(m.astype(np.uint8))
                for m in visible_masks_binary
            ])
            instances.gt_masks_visible = BitMasks(vis_masks_stack)



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
            results["clogging"] = {"MAE": mae}
        else:
            logger.warning(
                "No clogging extent predictions collected. "
                "Verify that gt_clogging_extent is set by the dataset mapper "
                "and pred_clogging_extent is set by the mask head inference path."
            )
            results["clogging"] = {"MAE": float("nan")}

        return results

# Model trainer

class ValidationEvaluationHook(HookBase):
    """
    Custom hook to run validation evaluation periodically during training and
    log metrics to TensorBoard.
    
    This enables real-time monitoring of validation AP, AP50, and clogging MAE
    during training without waiting for the full training run to complete.
    """
    def __init__(self, cfg, eval_period, eval_function):
        """
        Parameters
        ----------
        cfg : CfgNode
            Detectron2 config node with DATASETS.TEST set
        eval_period : int
            Number of training iterations between validation runs
            Recommended: check validation every 500-1000 iterations
        eval_function : callable
            Function that performs evaluation (e.g., trainer.test)
            Should return a dict with metrics
        """
        self.cfg = cfg
        self.eval_period = eval_period
        self.eval_function = eval_function
        self._do_train = True

    def before_train(self):
        """Initialize storage hook for TensorBoard metrics"""
        self.storage = self.trainer.storage

    def after_step(self):
        """Run validation periodically during training"""
        next_iter = self.trainer.iter + 1
        
        # Run evaluation every eval_period iterations
        if (self.eval_period > 0) and (next_iter % self.eval_period) == 0:
            self._do_evaluate()

    def _do_evaluate(self):
        """Execute validation and log metrics to TensorBoard"""
        logger.info(
            f"Running validation evaluation at iteration {self.trainer.iter}..."
        )
        
        # Run evaluation
        results = self.eval_function(self.cfg, self.trainer.model)
        
        # Extract and log metrics
        if results and "segm" in results:
            # Log amodal mask metrics
            for metric_name in ["AP", "AP50", "AP75", "APs", "APm", "APl"]:
                if metric_name in results["segm"]:
                    metric_value = results["segm"][metric_name]
                    self.storage.put_scalar(
                        f"eval/segm_{metric_name}",
                        metric_value,
                        smoothing_hint=False
                    )
            
            logger.info(
                f"  Amodal AP: {results['segm'].get('AP', float('nan')):.2f}, "
                f"AP50: {results['segm'].get('AP50', float('nan')):.2f}"
            )
        
        # Log clogging extent MAE
        if results and "clogging" in results:
            mae = results["clogging"].get("MAE", float("nan"))
            self.storage.put_scalar(
                "eval/clogging_mae",
                mae,
                smoothing_hint=False
            )
            logger.info(f"  Clogging MAE: {mae:.4f}")


class StormDrainTrainer(DefaultTrainer):
    """
    DefaultTrainer subclass with CloggingEvaluator.

    No other customisation is needed:
    - ParallelAmodalMaskHeadClogging is resolved from Detectron2's registry
      via the MODEL.ROI_MASK_HEAD.NAME config node.
    - All mask losses are returned as a named dict from the mask head and
      are automatically summed by Detectron2's training loop.  lambda_clog
      (MODEL.ROI_MASK_HEAD.LAMBDA_CLOG) scales the clogging MAE term relative
      to the seven BCE terms; adjust via the config YAML if the clogging MAE
      plateaus before the mask AP has converged.
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
    
    def build_hooks(self):
        """
        Build a set of default hooks including periodic validation evaluation.
        
        This extends the default trainer hooks to include validation evaluation
        during training, enabling real-time monitoring of validation metrics
        via TensorBoard.
        """
        # Get the default hooks from parent class
        hooks = super().build_hooks()
        
        # Add validation evaluation hook
        # Runs validation every 500 iterations to monitor AP, AP50, and clogging MAE
        eval_period = self.cfg.TEST.EVAL_PERIOD if hasattr(self.cfg.TEST, "EVAL_PERIOD") else 500
        
        if eval_period > 0 and len(self.cfg.DATASETS.TEST) > 0:
            eval_hook = ValidationEvaluationHook(
                self.cfg,
                eval_period=eval_period,
                eval_function=self.__class__.test
            )
            hooks.insert(-1, eval_hook)  # Insert before the checkpointer hook
        
        return hooks
    
    # NOTE: BestCheckpointer not available in this detectron2 version.
    # The default trainer saves model_final.pth after training completes.
    # For tracking best validation metric, use the metrics.json output files.

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
    cfg.MODEL.ROI_HEADS.NAME              = "CloggingROIHeads"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES       = 1
    # Domain-specific IOU thresholds for storm drain detection.
    # Storm drain inlets are ~200×150 px.  With standard RPN thresholds
    # [0.3, 0.7] a 128×128 anchor (IoU ≈ 0.55 with the drain) falls in the
    # ignored gap and is never a positive example.  With ROI_HEADS threshold
    # of 0.5, partial-overlap visible-strip proposals (IoU ~0.25–0.35)
    # become background, removing the primary positive training signal.
    # These values are set in the YAML config and explicitly repeated here
    # to ensure they are not overridden by any base config inheritance.
    cfg.MODEL.RPN.IOU_THRESHOLDS          = [0.3, 0.5]
    cfg.MODEL.RPN.IOU_LABELS              = [0, -1, 1]
    cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS    = [0.3]
    cfg.MODEL.ROI_HEADS.IOU_LABELS        = [0, 1]
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST   = 0.5
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.TEST.DETECTIONS_PER_IMAGE         = 2   # max 2 inlets per image
    cfg.MODEL.ROI_MASK_HEAD.NAME          = "ParallelAmodalMaskHeadClogging"
    cfg.MODEL.MASK_ON                     = True
    cfg.DATASETS.TRAIN                    = (train_dataset,)
    cfg.DATASETS.TEST                     = (val_dataset,)
    cfg.OUTPUT_DIR                        = output_dir
    
    # ===== TensorBoard Logging Configuration =====
    # Enable TensorBoard logging to track training/validation metrics in real-time
    cfg.SOLVER.LOG_EVERY_N_ITERS          = 50        # log training loss every 50 iterations
    
    # Validation evaluation during training
    # Run validation every 500 iterations to monitor AP and clogging_mae
    cfg.TEST.EVAL_PERIOD                  = 500

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

    final_ckpt = os.path.join(output_dir, "model_final.pth")
    # IMPORTANT: do NOT auto-skip training when model_final.pth exists.
    # Previously "eval_only = args.eval_only or os.path.isfile(final_ckpt)" caused
    # every re-run in the same output_dir to silently skip training and evaluate the
    # old checkpoint — making code changes appear to have no effect and giving
    # identical "training" times for different datasets (only eval was running).
    # Training is now ALWAYS run unless the user explicitly sets args.eval_only=True.
    eval_only = args.eval_only

    if eval_only:
        if not os.path.isfile(final_ckpt):
            raise FileNotFoundError(
                f"eval_only=True but no checkpoint found at:\n  {final_ckpt}\n"
                "Run training first, or point args.weights at a specific checkpoint."
            )
        logger.info(f"  eval_only=True: loading {final_ckpt}, skipping training.")
        cfg2 = cfg.clone()
        cfg2.defrost()
        cfg2.MODEL.WEIGHTS = final_ckpt
        cfg2.freeze()
        model = StormDrainTrainer.build_model(cfg2)
        DetectionCheckpointer(model, save_dir=output_dir).resume_or_load(
            cfg2.MODEL.WEIGHTS, resume=False
        )
        results = StormDrainTrainer.test(cfg2, model)
    else:
        if os.path.isfile(final_ckpt):
            logger.info(
                f"model_final.pth exists in {output_dir}. Will resume from last checkpoint "
                "if available. Set args.eval_only=True to evaluate the existing model without training."
            )
        trainer = StormDrainTrainer(cfg)
        # CRITICAL: resume=True enables checkpoint resumption after interruptions
        # This will load the last checkpoint (stored in last_checkpoint file) if it exists
        trainer.resume_or_load(resume=True)
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

    start_fold = getattr(args, 'start_fold', 0)
    for fold in range(k):
        root_logger = logging.getLogger("detectron2")
        root_logger.handlers = [h for h in root_logger.handlers 
                         if not isinstance(h, logging.FileHandler)]
        
        if fold < start_fold:
            logger.info(f"  Skipping fold {fold} (start_fold={start_fold})")
            all_results.append({})
            continue
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
            f"Clogging MAE={results.get('clogging', {}).get('MAE', float('nan')):.4f}"
        )

    # ---- Compute mean and std across folds ---------------------------------
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
        logger.info(f"  Clogging MAE: {results.get('clogging', {}).get('MAE','N/A')}")

# CONFIGURATION SETTINGS
class ConfigArgs:
    base_dir = "/run/user/1000/gvfs/smb-share:server=ecresearch.uwaterloo.ca,share=cviss/Jack/baf/360streetview"
    # Required Paths
    all_json   = os.path.join(base_dir, 'segmentation_training', 'synthetic_v2', 'synthetic.json')  # v2: fixed distribution + uniform inlet sampling
    all_imgs   = os.path.join(base_dir, 'segmentation_training', 'synthetic_v2', 'images')
    output_dir = os.path.join(base_dir, 'trained_models', 'synthetic_final_test')
    
    # Training Mode
    kfold = 1
    seed = 42
    start_fold = 0   # set > 0 to skip already-completed folds (training auto-skipped if model_final.pth exists)
    
    # Model parameters
    config_file = "/home/cviss/jack/360streetview/Amodal-Segmentation-Based-on-Visible-Region-Segmentation-and-Shape-Prior/configs/StormDrain/mask_rcnn_R_50_FPN_1x_clogging.yaml" # defualts to a ResNet-50 backbone
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