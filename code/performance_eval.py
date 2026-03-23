import os
import csv
import torch
import numpy as np
import pycocotools.mask as mask_util
from detectron2.evaluation import DatasetEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader, DatasetCatalog
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo

# Import Aistron configs
from aistron.config import add_aistron_config

class SemanticAmodalIoUEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name, class_names, output_csv="amodal_miou.csv"):
        self.dataset_name = dataset_name
        self.class_names = class_names
        self.output_csv = output_csv
        # THE FIX: Pre-load the raw catalog to bypass the dataloader stripping annotations
        self.dataset_dicts = {d["image_id"]: d for d in DatasetCatalog.get(dataset_name)}
        self.reset()

    def reset(self):
        self.intersections = {cls_id: 0.0 for cls_id in self.class_names.keys()}
        self.unions = {cls_id: 0.0 for cls_id in self.class_names.keys()}

    def process(self, inputs, outputs):
        for input_dict, output_dict in zip(inputs, outputs):
            height, width = input_dict["height"], input_dict["width"]
            img_id = input_dict["image_id"]
            
            # Retrieve raw GT dictionary using the image_id
            gt_dict = self.dataset_dicts[img_id]
            
            # 1. Parse Ground Truth
            gt_masks = {cls_id: np.zeros((height, width), dtype=bool) for cls_id in self.class_names.keys()}
            if "annotations" in gt_dict:
                for annot in gt_dict["annotations"]:
                    # Category IDs are already 0-indexed by Detectron2
                    cat_id = annot["category_id"] 
                    if cat_id in self.class_names and "segmentation" in annot and annot["segmentation"]:
                        # THE FIX: Correctly convert COCO polygon to boolean mask using pycocotools
                        rles = mask_util.frPyObjects(annot["segmentation"], height, width)
                        rle = mask_util.merge(rles)
                        mask = mask_util.decode(rle).astype(bool)
                        gt_masks[cat_id] = gt_masks[cat_id] | mask

            # 2. Parse Predictions
            pred_masks = {cls_id: np.zeros((height, width), dtype=bool) for cls_id in self.class_names.keys()}
            if "instances" in output_dict:
                instances = output_dict["instances"]
                for mask, cat_id in zip(instances.pred_masks, instances.pred_classes):
                    cat_id = cat_id.item()
                    mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
                    pred_masks[cat_id] = pred_masks[cat_id] | mask_np.astype(bool)

            # 3. Calculate IoU per class
            for cls_id in self.class_names.keys():
                intersection = np.logical_and(gt_masks[cls_id], pred_masks[cls_id]).sum()
                union = np.logical_or(gt_masks[cls_id], pred_masks[cls_id]).sum()
                
                self.intersections[cls_id] += intersection
                self.unions[cls_id] += union

    def evaluate(self):
        ious = {}
        with open(self.output_csv, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Class", "mIoU"])
            
            for cls_id, cls_name in self.class_names.items():
                if self.unions[cls_id] == 0:
                    iou = float('nan') 
                else:
                    iou = self.intersections[cls_id] / self.unions[cls_id]
                
                ious[cls_name] = iou
                writer.writerow([cls_name, f"{iou:.4f}"])
        
        print("\n--- Final Semantic mIoU ---")
        for name, val in ious.items():
             print(f"{name.capitalize()}: {val:.4f}")
        
        return {"semantic_miou": ious}

# --- Execution Setup ---
base_dir = "/run/user/1000/gvfs/smb-share:server=ecresearch.uwaterloo.ca,share=cviss/Jack/baf/360streetview/"
weights_dir = os.path.join(base_dir, "trained_models", "test", "model_final.pth")
dataset_dir = os.path.join(base_dir, "segmentation_training", "synthetic", "copy_paste", "clean")
output_dir = os.path.join(base_dir, "segmentation_training", "synthetic", "copy_paste", "miou.csv")
dataset_name = "validation"

# Register the validation dataset
register_coco_instances(dataset_name, {}, os.path.join(dataset_dir, "original.json"), os.path.join(dataset_dir, "images"))

# Configure the model
cfg = get_cfg()
add_aistron_config(cfg) 
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = weights_dir
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

# Build predictor, dataloader, and custom evaluator
predictor = DefaultPredictor(cfg)
eval_loader = build_detection_test_loader(cfg, dataset_name)
evaluator = SemanticAmodalIoUEvaluator(dataset_name=dataset_name, class_names={0: "inlet", 1: "debris"}, output_csv=output_dir)

# Run Evaluation
inference_on_dataset(predictor.model, eval_loader, evaluator)