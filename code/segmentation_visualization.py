import cv2
import os
import pandas as pd
import numpy as np
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from aistron.config import add_aistron_config
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.engine import DefaultPredictor


cfg = get_cfg()
add_aistron_config(cfg)
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

cfg.MODEL.WEIGHTS = "/home/wonny/orcnn/output/model_final.pth"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2 
cfg.MODEL.DEVICE = "cuda"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # Lowered to 0.5 to catch more debris

predictor = DefaultPredictor(cfg)

metadata = MetadataCatalog.get("inlet_val").set(
    thing_classes=["Inlet", "Debris"],
    thing_colors=[(255, 255, 255), (0, 0, 0)] 
)

def process_image_data(image_path, save_path):
    """
    Runs prediction once, extracts confidences, calculates the clogging extent, and saves the visualization.
    """
    im = cv2.imread(image_path)
    if im is None:
        return None
    
    # Run the model once
    outputs = predictor(im)
    instances = outputs["instances"].to("cpu")
    
    # Extract data from the GPU
    pred_classes = instances.pred_classes.numpy()
    scores = instances.scores.numpy()
    
    # AIStron specific masks
    amodal_masks = instances.pred_amodal_masks.numpy() if instances.has("pred_amodal_masks") else instances.pred_masks.numpy()
    visible_masks = instances.pred_visible_masks.numpy() if instances.has("pred_visible_masks") else instances.pred_masks.numpy()
    
    inlet_idx = np.where(pred_classes == 0)[0]
    debris_idx = np.where(pred_classes == 1)[0]
    
    inlet_conf = float(scores[inlet_idx[0]]) if len(inlet_idx) > 0 else 0.0
    debris_conf = float(np.mean(scores[debris_idx])) if len(debris_idx) > 0 else 0.0
    
    clogging_extent = 0.0
    if len(inlet_idx) > 0:
        inlet_amodal_mask = amodal_masks[inlet_idx[0]]
        
        # This mathematically groups all touching (and non-touching) debris into one master layer
        master_debris_mask = np.zeros_like(inlet_amodal_mask, dtype=bool)
        for idx in debris_idx:
            master_debris_mask = np.logical_or(master_debris_mask, visible_masks[idx])
            
        intersection_mask = np.logical_and(inlet_amodal_mask, master_debris_mask)
        area_amodal_inlet = np.sum(inlet_amodal_mask)
        area_intersection = np.sum(intersection_mask)
        
        if area_amodal_inlet > 0:
            clogging_extent = (area_intersection / area_amodal_inlet) * 100

    v = Visualizer(
        im[:, :, ::-1], 
        metadata=metadata, 
        scale=1.2, 
        instance_mode=ColorMode.SEGMENTATION  # This forces class-based coloring
    )
    out = v.draw_instance_predictions(instances)
    cv2.imwrite(save_path, out.get_image()[:, :, ::-1])
    
    # Return the dictionary row for our DataFrame
    return {
        "Image_Name": os.path.basename(image_path),
        "Clogging_Extent_%": round(clogging_extent, 2),
        "Inlet_Confidence": round(inlet_conf, 3),
        "Average_Debris_Confidence": round(debris_conf, 3),
        "Debris_Instances_Found": len(debris_idx)
    }

if __name__ == "__main__":
    test_img_dir = "/run/user/1000/gvfs/smb-share:server=ecresearch.uwaterloo.ca,share=cviss/Jack/baf/360streetview/segmentation_training/dataset/val_images"
    output_img_dir = os.path.join(test_img_dir, "results")
    csv_output_path = os.path.join(output_img_dir, "clogging_results.csv")
    
    os.makedirs(output_img_dir, exist_ok=True)
    
    data_rows = []

    print(f"Processing images in: {test_img_dir}")
    
    for img_name in os.listdir(test_img_dir):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(test_img_dir, img_name)
            save_path = os.path.join(output_img_dir, f"res_{img_name}")
            
            # Run our unified function
            row_data = process_image_data(img_path, save_path)
            
            if row_data:
                data_rows.append(row_data)
                print(f"Processed {img_name} -> Clogging: {row_data['Clogging_Extent_%']}%")

    # Generate the DataFrame and save to Excel/CSV
    df = pd.DataFrame(data_rows)
    df.to_csv(csv_output_path, index=False)
    
    print("\n" + "="*40)
    print(f"Done! Results saved to {csv_output_path}")
    print("DataFrame Preview:")
    print(df.head())