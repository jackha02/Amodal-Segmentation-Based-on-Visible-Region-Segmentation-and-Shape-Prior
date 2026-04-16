# =============================================================================
# diagnose_model.py — Deep diagnostic tool for the amodal segmentation model
# =============================================================================
#
# This script investigates three concrete failure modes in the current model:
#
#   [D1] CLOGGING FROM EMPTY MASKS
#        Detects cases where pred_clogging_extent > 0 but the binary amodal
#        mask has zero confident pixels (< 5). These are "phantom" predictions
#        caused by the soft-sigmoid inference bug (now fixed in mask_head_clogging.py).
#
#   [D2] VISIBLE MASK QUALITY
#        For each TP detection (IoU > 0.5 with GT), measures the binary visible
#        mask coverage, its IoU with the GT visible polygon, and the resulting
#        clogging error. This tells us if the visible mask head is the bottleneck.
#
#   [D3] DETECTION FALSE POSITIVE RATE
#        Counts predictions with no matching GT (IoU < 0.5) across score bins,
#        identifying whether the RPN / ROI score threshold is miscalibrated.
#
#   [D4] CLOGGING CALIBRATION BY RANGE
#        Splits the [0,1] clogging range into 5 bins and reports MAE per bin.
#        A uniform bar plot of errors across bins indicates domain shift.
#
# OUTPUT
# ------
#   <output_dir>/
#     D1_phantom_clogging.json    — per-image list of phantom clogging instances
#     D2_visible_mask_quality.json
#     D3_false_positive_rate.json
#     D4_clogging_calibration.png
#     D4_per_range_mae.json
#     diagnosis_summary.txt       — human-readable summary of all findings
#
# USAGE
# -----
#   conda activate d2_amodal
#   python diagnose_model.py
#
# =============================================================================

import copy
import json
import logging
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR   = os.path.join(os.path.dirname(SCRIPT_DIR),
    "Amodal-Segmentation-Based-on-Visible-Region-Segmentation-and-Shape-Prior")
if not os.path.isdir(REPO_DIR):
    raise RuntimeError(f"Cannot find repo at {REPO_DIR}")
sys.path.insert(0, REPO_DIR)

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import add_clogging_config
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.modeling.roi_heads import (
    ROI_HEADS_REGISTRY, StandardROIHeads, select_foreground_proposals)
from detectron2.structures import BoxMode, PolygonMasks, pairwise_iou, Boxes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("diagnose")


# ---------------------------------------------------------------------------
# Custom ROI Heads registration (identical to train_amodal_segmentation.py)
# ---------------------------------------------------------------------------
@ROI_HEADS_REGISTRY.register()
class CloggingROIHeads(StandardROIHeads):
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
# Dataset registration (preserves clogging_extent + visible_segmentation)
# ---------------------------------------------------------------------------
def register_dataset(name, json_path, images_dir):
    if name in DatasetCatalog.list():
        return
    def _loader():
        with open(json_path) as f:
            d = json.load(f)
        anns_by_img = {}
        for a in d["annotations"]:
            anns_by_img.setdefault(a["image_id"], []).append(a)
        records = []
        for img in d["images"]:
            iid = img["id"]
            rec = {
                "file_name":   os.path.join(images_dir, img["file_name"]),
                "image_id":    iid,
                "height":      img["height"],
                "width":       img["width"],
                "annotations": [],
            }
            for a in anns_by_img.get(iid, []):
                obj = {
                    "id": a["id"], "bbox": a["bbox"],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "category_id": 0,
                    "segmentation": a.get("segmentation", []),
                    "iscrowd": a.get("iscrowd", 0),
                    "area": a.get("area", 0),
                }
                if "clogging_extent"      in a: obj["clogging_extent"]      = a["clogging_extent"]
                if "visible_segmentation" in a: obj["visible_segmentation"] = a["visible_segmentation"]
                rec["annotations"].append(obj)
            records.append(rec)
        return records
    DatasetCatalog.register(name, _loader)
    MetadataCatalog.get(name).set(thing_classes=["storm_drain"])


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------
def build_model(args, checkpoint_path, dataset_name):
    cfg = get_cfg()
    add_clogging_config(cfg)
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.defrost()
    cfg.MODEL.DEVICE                       = "cuda"
    cfg.VIS_PERIOD                         = 0
    cfg.MODEL.ROI_HEADS.NAME              = "CloggingROIHeads"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES       = 1
    cfg.MODEL.RPN.IOU_THRESHOLDS          = [0.3, 0.5]
    cfg.MODEL.RPN.IOU_LABELS              = [0, -1, 1]
    cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS    = [0.3]
    cfg.MODEL.ROI_HEADS.IOU_LABELS        = [0, 1]
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST   = 0.5
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3   # intentionally low to see all detections
    cfg.TEST.DETECTIONS_PER_IMAGE         = 5
    cfg.MODEL.ROI_MASK_HEAD.NAME          = "ParallelAmodalMaskHeadClogging"
    cfg.MODEL.MASK_ON                     = True
    cfg.MODEL.WEIGHTS                     = checkpoint_path
    cfg.DATASETS.TRAIN                    = ()
    cfg.DATASETS.TEST                     = (dataset_name,)
    cfg.OUTPUT_DIR                        = os.path.dirname(checkpoint_path)
    cfg.freeze()
    model = DefaultTrainer.build_model(cfg)
    DetectionCheckpointer(model).load(checkpoint_path)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Polygon → binary mask helper
# ---------------------------------------------------------------------------
def polygons_to_mask(polygons, h, w):
    """Rasterise a list of polygon point-arrays to a (h,w) binary mask."""
    mask = np.zeros((h, w), dtype=np.uint8)
    for poly in polygons:
        pts = np.array(poly, dtype=np.float32).reshape(-1, 1, 2).astype(np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


# ---------------------------------------------------------------------------
# paste_roi_mask — resize 28x28 ROI mask into full-image coordinates
# ---------------------------------------------------------------------------
def paste_roi_mask(mask_roi, box, img_h, img_w):
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, img_w), min(y2, img_h)
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return np.zeros((img_h, img_w), dtype=bool)
    resized = cv2.resize(mask_roi.astype(np.float32), (bw, bh), interpolation=cv2.INTER_LINEAR)
    full = np.zeros((img_h, img_w), dtype=bool)
    full[y1:y2, x1:x2] = resized > 0.5
    return full


# ---------------------------------------------------------------------------
# Core diagnostic pass
# ---------------------------------------------------------------------------
def run_diagnose(args):
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Output: {out_dir}")

    ds_name = "storm_drain_diag"
    register_dataset(ds_name, args.test_json, args.test_imgs)
    dataset = DatasetCatalog.get(ds_name)
    logger.info(f"Dataset: {len(dataset)} images")

    # Pick one fold model
    ckpt = os.path.join(args.trained_models_dir, f"fold_{args.diag_fold}", "model_0034999.pth")
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"No checkpoint: {ckpt}")
    logger.info(f"Checkpoint: {ckpt}")
    model = build_model(args, ckpt, ds_name)

    # ---------- accumulators ------------------------------------------------
    D1_phantom = []      # phantom clogging events
    D2_vis_iou = []      # (gt_clog, pred_clog, vis_iou, amod_iou) for TPs
    D3_fps     = []      # (score, is_fp) for all detections
    D4_pairs   = []      # (gt_clog, pred_clog) for IoU-matched pairs

    for record in dataset:
        img_bgr = cv2.imread(record["file_name"])
        if img_bgr is None:
            logger.warning(f"Missing image: {record['file_name']}")
            continue
        H, W = img_bgr.shape[:2]
        img_rgb = img_bgr[:, :, ::-1].copy()
        tensor  = torch.as_tensor(
            np.ascontiguousarray(img_rgb.transpose(2, 0, 1)), dtype=torch.float32)

        with torch.no_grad():
            outputs   = model([{"image": tensor, "height": H, "width": W}])
        inst      = outputs[0]["instances"].to("cpu")
        n_pred    = len(inst)
        gt_anns   = record.get("annotations", [])
        n_gt      = len(gt_anns)

        if n_pred == 0 or n_gt == 0:
            continue

        # Build GT Boxes in xyxy for IoU matching
        gt_boxes_xyxy = []
        for a in gt_anns:
            bx, by, bw, bh = a["bbox"]
            gt_boxes_xyxy.append([bx, by, bx + bw, by + bh])
        from detectron2.structures import Boxes as D2Boxes
        gt_boxes = D2Boxes(torch.tensor(gt_boxes_xyxy, dtype=torch.float32))
        iou_mat  = pairwise_iou(inst.pred_boxes, gt_boxes)   # (n_pred, n_gt)

        # ---- D3: false positive analysis -----------------------------------
        max_iou_per_pred = iou_mat.max(dim=1).values.numpy()
        scores = inst.scores.numpy()
        for score, miou in zip(scores, max_iou_per_pred):
            D3_fps.append({"score": float(score), "max_iou": float(miou), "fp": bool(miou < 0.5)})

        # ---- D1 / D2 / D4: per-GT matching ---------------------------------
        max_iou_per_gt, best_pred_idx = iou_mat.max(dim=0)

        pred_scores  = inst.scores.numpy()
        pred_clog    = inst.pred_clogging_extent.numpy() if inst.has("pred_clogging_extent") else np.zeros(n_pred)
        pred_amodal  = (inst.pred_masks_amodal.numpy()  > 0.5) if inst.has("pred_masks_amodal")  else None
        pred_visible = (inst.pred_masks_visible.numpy() > 0.5) if inst.has("pred_masks_visible") else None

        for gt_idx, (iou_val, pid) in enumerate(zip(max_iou_per_gt, best_pred_idx)):
            ann   = gt_anns[gt_idx]
            gt_clog = float(ann.get("clogging_extent", 0.0))
            pid   = int(pid)

            # Phantom clogging check (D1)
            if pred_amodal is not None:
                box_xyxy = inst.pred_boxes.tensor[pid].numpy()
                amod_full = paste_roi_mask(pred_amodal[pid], box_xyxy, H, W)
                amod_px   = int(amod_full.sum())
                if amod_px < 5 and pred_clog[pid] > 0.05:
                    D1_phantom.append({
                        "image":       record["file_name"],
                        "score":       float(pred_scores[pid]),
                        "clogging":    float(pred_clog[pid]),
                        "amodal_px_above_0.5": amod_px,
                        "iou_with_gt": float(iou_val),
                    })

            if float(iou_val) < 0.5:
                continue   # only process true positives below

            # D4: clogging calibration
            D4_pairs.append((gt_clog, float(pred_clog[pid])))

            # D2: visible mask quality
            if pred_visible is not None and pred_amodal is not None:
                box_xyxy = inst.pred_boxes.tensor[pid].numpy()
                vis_full  = paste_roi_mask(pred_visible[pid], box_xyxy, H, W)
                amod_full = paste_roi_mask(pred_amodal[pid], box_xyxy, H, W)

                # GT visible mask
                vis_segs = ann.get("visible_segmentation", ann.get("segmentation", []))
                gt_vis   = polygons_to_mask(vis_segs, H, W)
                # GT amodal mask
                amod_segs = ann.get("segmentation", [])
                gt_amod   = polygons_to_mask(amod_segs, H, W)

                def iou_masks(a, b):
                    inter = (a & b).sum()
                    union = (a | b).sum()
                    return float(inter) / (float(union) + 1e-6)

                vis_iou  = iou_masks(vis_full,  gt_vis)
                amod_iou = iou_masks(amod_full, gt_amod)
                D2_vis_iou.append({
                    "gt_clogging":   gt_clog,
                    "pred_clogging": float(pred_clog[pid]),
                    "visible_iou":   vis_iou,
                    "amodal_iou":    amod_iou,
                    "clogging_err":  abs(float(pred_clog[pid]) - gt_clog),
                })

    # ======================================================================
    # Write results
    # ======================================================================
    def jdump(obj, fname):
        path = os.path.join(out_dir, fname)
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)
        logger.info(f"Wrote {path}")

    jdump(D1_phantom,  "D1_phantom_clogging.json")
    jdump(D2_vis_iou,  "D2_visible_mask_quality.json")
    jdump(D3_fps,      "D3_false_positive_rate.json")

    # ---------- D4 calibration plot ----------------------------------------
    if D4_pairs:
        gt_vals   = np.array([p[0] for p in D4_pairs])
        pred_vals = np.array([p[1] for p in D4_pairs])
        bins      = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
        bin_labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
        per_bin   = {}
        for lo, hi, label in zip(bins[:-1], bins[1:], bin_labels):
            idx = (gt_vals >= lo) & (gt_vals < hi)
            if idx.sum() > 0:
                mae = float(np.abs(pred_vals[idx] - gt_vals[idx]).mean())
                per_bin[label] = {"mae": mae, "n": int(idx.sum())}
            else:
                per_bin[label] = {"mae": None, "n": 0}

        jdump(per_bin, "D4_per_range_mae.json")

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: AtP scatter
        ax = axes[0]
        ax.scatter(gt_vals * 100, pred_vals * 100, alpha=0.5, s=20, color="dimgray", zorder=3)
        ax.plot([0, 100], [0, 100], "r-", lw=2, label="Perfect prediction")
        ax.fill_between([0, 100], [0 - 10, 100 - 10], [0 + 10, 100 + 10], alpha=0.15, color="gray", label="±10%")
        ax.fill_between([0, 100], [0 - 20, 100 - 20], [0 + 20, 100 + 20], alpha=0.08, color="gray", label="±20%")
        ax.set_xlabel("Ground Truth Clogging Extent (%)")
        ax.set_ylabel("Predicted Clogging Extent (%)")
        ax.set_title(f"AtP Scatter (n={len(gt_vals)}, MAE={np.abs(pred_vals-gt_vals).mean()*100:.1f}%)")
        ax.set_xlim(0, 100); ax.set_ylim(0, 100)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Right: per-range MAE bar
        ax2 = axes[1]
        labels = [k for k, v in per_bin.items() if v["mae"] is not None]
        maes   = [per_bin[k]["mae"] * 100 for k in labels]
        ns     = [per_bin[k]["n"]          for k in labels]
        bars   = ax2.bar(labels, maes, color="steelblue", alpha=0.8)
        for bar, n in zip(bars, ns):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f"n={n}", ha="center", va="bottom", fontsize=8)
        ax2.axhline(np.abs(pred_vals - gt_vals).mean() * 100, color="red", linestyle="--",
                    label=f"Overall MAE {np.abs(pred_vals-gt_vals).mean()*100:.1f}%")
        ax2.set_ylabel("MAE (%)")
        ax2.set_title("Clogging MAE vs Ground-Truth Range")
        ax2.legend(fontsize=8)
        ax2.set_ylim(0, max(maes) * 1.3 if maes else 10)
        ax2.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        fig_path = os.path.join(out_dir, "D4_clogging_calibration.png")
        plt.savefig(fig_path, dpi=150)
        plt.close()
        logger.info(f"Wrote {fig_path}")

    # ---------- D3 analysis ------------------------------------------------
    if D3_fps:
        all_fps = sum(1 for x in D3_fps if x["fp"])
        all_det = len(D3_fps)
        score_bins = np.arange(0.3, 1.05, 0.1)
        fp_by_thresh = {}
        for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            tp2 = sum(1 for x in D3_fps if x["score"] >= thr and not x["fp"])
            fp2 = sum(1 for x in D3_fps if x["score"] >= thr and  x["fp"])
            tot = tp2 + fp2
            fp_by_thresh[str(round(thr, 1))] = {
                "tp": tp2, "fp": fp2, "total": tot,
                "fp_rate": round(fp2 / (tot + 1e-6), 3)
            }
        jdump({"total_detections": all_det, "total_fp": all_fps,
               "fp_rate": round(all_fps / (all_det + 1e-6), 3),
               "by_score_threshold": fp_by_thresh},
              "D3_false_positive_rate.json")

    # ---------- Summary text -----------------------------------------------
    summary_lines = [
        "=" * 70,
        "DIAGNOSIS SUMMARY",
        "=" * 70,
        "",
        f"[D1] PHANTOM CLOGGING (clog>5% but amodal mask empty after threshold):",
        f"     Count: {len(D1_phantom)}  (should be 0 after hard-mask fix)",
        "",
        f"[D2] VISIBLE MASK QUALITY (TP detections, IoU > 0.5):",
    ]
    if D2_vis_iou:
        vis_ious  = [x["visible_iou"] for x in D2_vis_iou]
        amod_ious = [x["amodal_iou"]  for x in D2_vis_iou]
        clog_errs = [x["clogging_err"] for x in D2_vis_iou]
        summary_lines += [
            f"     Instances: {len(D2_vis_iou)}",
            f"     Mean visible  mask IoU : {np.mean(vis_ious):.3f}  (want >0.5)",
            f"     Mean amodal   mask IoU : {np.mean(amod_ious):.3f}  (want >0.7)",
            f"     Mean clogging err (MAE): {np.mean(clog_errs)*100:.1f}%",
            "",
            "     Per visible-IoU bucket:",
        ]
        for lo, hi in [(0, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.01)]:
            idx = [x for x in D2_vis_iou if lo <= x["visible_iou"] < hi]
            if idx:
                mae = np.mean([x["clogging_err"] for x in idx]) * 100
                summary_lines.append(
                    f"       vis_IoU [{lo:.0%}-{hi:.0%}): n={len(idx):3d}  clog_mae={mae:.1f}%"
                )
    else:
        summary_lines.append("     No TP detections found.")

    summary_lines += [
        "",
        f"[D3] FALSE POSITIVE RATE:",
    ]
    if D3_fps:
        all_fps  = sum(1 for x in D3_fps if x["fp"])
        all_det  = len(D3_fps)
        summary_lines.append(
            f"     {all_fps}/{all_det} detections are FP at SCORE_THRESH=0.3  ({all_fps/all_det*100:.1f}%)"
        )
        summary_lines.append("     FP rate by score threshold:")
        for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            tp2 = sum(1 for x in D3_fps if x["score"] >= thr and not x["fp"])
            fp2 = sum(1 for x in D3_fps if x["score"] >= thr and x["fp"])
            tot = tp2 + fp2
            if tot:
                summary_lines.append(f"       thresh >= {thr:.1f}: {fp2}/{tot} FP ({fp2/tot*100:.1f}%)")

    summary_lines += [
        "",
        "[D4] CLOGGING MAE BY GT RANGE:",
    ]
    if D4_pairs and per_bin:
        for label, v in per_bin.items():
            if v["mae"] is not None:
                summary_lines.append(f"     GT {label}: MAE={v['mae']*100:.1f}%  n={v['n']}")

    summary_lines += [
        "",
        "=" * 70,
        "CRITICAL FINDINGS AND RECOMMENDED FIXES:",
        "=" * 70,
        "",
        "1. VISUALIZATION BUG (FIXED in segmentation_visualization.py):",
        "   .astype(bool) on sigmoid floats → entire 28×28 grid = True → rectangles.",
        "   Fix: (pred_masks_X.numpy() > 0.5)",
        "",
        "2. PHANTOM CLOGGING BUG (FIXED in mask_head_clogging.py):",
        "   Soft sigmoid probabilities never = 0, so every detection gets",
        "   a non-zero clogging value even with no confident mask pixels.",
        "   Fix: use hard binary masks (> 0.5) and gate on >= 5 amodal pixels.",
        "",
        "3. AMODAL MASKS ARE ALWAYS 4-POINT RECTANGLES:",
        "   Both synthetic and real datasets use 4-point quadrilaterals for",
        "   the amodal mask. This explains AP50=100% on synthetic (trivially easy)",
        "   and the AP drop on real data (domain shift only, not complexity).",
        "",
        "4. TRAINING TOO SHORT (6000 iters ≈ 7 epochs on 800 images):",
        "   A 3-module architecture needs ~60-100 epochs to converge.",
        "   RECOMMENDED: MAX_ITER = 60000, STEPS = (45000, 55000)",
        "",
        "5. SCORE THRESHOLD MAY BE TOO LOW (0.5):",
        "   Check D3 output. If FP rate is high, increase SCORE_THRESH_TEST to 0.7.",
        "",
        "6. VISIBLE MASK IS THE BOTTLENECK FOR CLOGGING:",
        "   Check D2: if visible_mask IoU < 0.4, the clogging estimate is",
        "   unreliable regardless of model score. The visible mask head needs",
        "   more training iterations and possibly dataset augmentation.",
    ]

    summary_path = os.path.join(out_dir, "diagnosis_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print("\n".join(summary_lines))
    logger.info(f"Summary written to {summary_path}")


# =============================================================================
# CONFIGURATION
# =============================================================================
class ConfigArgs:
    BASE = "/run/user/1000/gvfs/smb-share:server=ecresearch.uwaterloo.ca,share=cviss/Jack/baf/360streetview"

    # Test dataset (real images)
    test_json  = os.path.join(BASE, "segmentation_training", "clogged_inlets", "clogged.json")
    test_imgs  = os.path.join(BASE, "segmentation_training", "clogged_inlets", "images")

    # Trained model directory
    trained_models_dir = os.path.join(BASE, "trained_models", "synthetic")

    # Which fold to diagnose (0-4)
    diag_fold = 0

    # Config YAML
    config_file = os.path.join(
        os.path.dirname(SCRIPT_DIR),
        "Amodal-Segmentation-Based-on-Visible-Region-Segmentation-and-Shape-Prior",
        "configs", "StormDrain", "mask_rcnn_R_50_FPN_1x_clogging.yaml",
    )

    # Output directory for diagnostics
    output_dir = os.path.join(trained_models_dir, "diagnosis")


if __name__ == "__main__":
    args = ConfigArgs()
    run_diagnose(args)
