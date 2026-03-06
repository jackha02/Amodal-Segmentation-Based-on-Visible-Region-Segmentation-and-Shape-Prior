import numpy as np
import os
import cv2
import numpy as np

from detection_model_training import data_split
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from aistron.config import add_aistron_config

# AIStron is an open-source toolbox used for Amodal Instance Segmentation methods

def train_orcnn_model(dataset_dir):
    """
    Trains the ORCNN model to recognize inlets and debris.
    :param dataset_dir: directory where JSON files for training and validation are stored
    Returns a fine tuned ORCNN model
    """
    # Register the datasets, which are standard JSON file
    register_coco_instances("inlet_train", {}, f"{dataset_dir}/train.json", f"{dataset_dir}/train_images")
    register_coco_instances("inlet_val", {}, f"{dataset_dir}/val.json", f"{dataset_dir}/val_images")

    # Setup the Configuration
    cfg = get_cfg() # copy of the default configuration
    add_aistron_config(cfg) # add amodal segmentation capabilities
    
    # Find a model from detectron2's 
    cfg.merge_from_file("aistron_workspace/configs/ORCNN/mask_orcnn_R_50_FPN_1x.yaml")
    
    # Configure the datasets and GPU
    cfg.DATASETS.TRAIN = ("inlet_train",)
    cfg.DATASETS.TEST = ("inlet_val",)
    cfg.DATALOADER.NUM_WORKERS = 4
    
    # Initialize with pre-trained weights (transfers general image knowledge to your model)
    cfg.MODEL.WEIGHTS = "detectron2://ImageNetPretrained/MSRA/R-50.pkl" 
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # Class 0: Inlet, Class 1: Debris
    
    # Hyperparameters (How fast and how long the model studies)
    cfg.SOLVER.IMS_PER_BATCH = 4         # Number of images processed at once by the GPU
    cfg.SOLVER.BASE_LR = 0.001           # Learning rate (how quickly it updates its guesses)
    cfg.SOLVER.MAX_ITER = 3000           # Total training steps
    cfg.MODEL.DEVICE = "cuda"            # Force the use of the GPU

    # Start Training
    import os
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    
    return cfg

# Estimate the clogging extent
def calculate_clogging_extent(image_path, cfg):
    """
    Runs the trained model on a new image and calculates the clogging percentage
    :param image_path: path directory for an image to run the trained ORCNN model on
    :param cfg: configuration 
    """
    # Tell the configuration to use the weights we just trained
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # Only accept confident predictions (70%+)
    
    # Initialize the Predictor
    predictor = DefaultPredictor(cfg)
    
    # Load the image
    img = cv2.imread(image_path)
    outputs = predictor(img)
    
    # Extract the results from the GPU to standard CPU memory
    instances = outputs["instances"].to("cpu")
    pred_classes = instances.pred_classes.numpy()
    
    # AIStron provides 'pred_amodal_masks' (full shape) and 'pred_visible_masks'
    amodal_masks = instances.pred_amodal_masks.numpy() 
    visible_masks = instances.pred_visible_masks.numpy()
    
    # Find the indices for our classes
    inlet_idx = np.where(pred_classes == 0)[0]
    debris_idx = np.where(pred_classes == 1)[0]
    
    if len(inlet_idx) == 0:
        return "No inlet detected."
        
    # Isolate the amodal mask for the inlet (assuming 1 main inlet per cropped image)
    inlet_amodal_mask = amodal_masks[inlet_idx[0]]
    
    # Combine all debris masks into one master debris mask
    master_debris_mask = np.zeros_like(inlet_amodal_mask, dtype=bool)
    for idx in debris_idx:
        master_debris_mask = np.logical_or(master_debris_mask, visible_masks[idx])
        
    # MATH: Calculate the intersection of the debris falling EXACTLY over the amodal inlet
    intersection_mask = np.logical_and(inlet_amodal_mask, master_debris_mask)
    
    # Count the true pixels to get the areas
    area_amodal_inlet = np.sum(inlet_amodal_mask)
    area_intersection = np.sum(intersection_mask)
    
    # Calculate Clogging Extent (%)
    if area_amodal_inlet == 0:
        return 0.0
        
    clogging_extent = (area_intersection / area_amodal_inlet) * 100
    
    print(f"Total Inlet Area: {area_amodal_inlet} pixels")
    print(f"Clogged Area: {area_intersection} pixels")
    print(f"Final Clogging Extent: {clogging_extent:.2f}%")
    
    return clogging_extent

if __name__ == '__main__':

    # File paths
    base_dir = "/run/user/1000/gvfs/smb-share:server=ecresearch.uwaterloo.ca,share=cviss/Jack/baf/360streetview"
    training_images = os.path.join(base_dir, 'segmentation_training', 'images') 
    training_labels = os.path.join(base_dir, 'segmentation_training', 'labels')
    dataset_dir = os.path.join(base_dir, 'segmentation_training', 'dataset')
    new_img = os.path.join(base_dir, 'final', 'images')
    results = os.path.join(base_dir, 'final', 'results')
    
    # Split the datasets for training and validation 
    data_split(training_images, training_labels, dataset_dir, validation_ratio=0.2)  
   
    # Train the ORCNN model
    cfg = train_orcnn_model(dataset_dir)

    # Run the inference on new sets of images 
    for img in os.listdir(new_img):
        img_path = os.path.join(new_img, img)
        clogging_score = calculate_clogging_extent(img_path, cfg)