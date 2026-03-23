import numpy as np
import os
import cv2
import torch
import json
import shutil
import random

from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetMapper, build_detection_train_loader
import detectron2.data.transforms as T
from aistron.config import add_aistron_config

# Custom Trainer for Online Augmentation
class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        augmentations = [
            # Standard resize with scaling variance (Equivalent to scale=0.2 / translate=0.1)
            T.ResizeShortestEdge(
                short_edge_length=(640, 672, 704, 736, 768, 800), 
                max_size=1333, 
                sample_style="choice"
            ),
            # Geometric: 50% chance to flip horizontally (fliplr=0.5)
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
            
            # Photometric: Value/Brightness shift +/- 20% (hsv_v=0.2)
            T.RandomBrightness(0.8, 1.2),
            
            # Photometric: Saturation shift +/- 30% (hsv_s=0.3)
            T.RandomSaturation(0.7, 1.3),
            
            # Photometric: Contrast shift to help substitute for Hue (hsv_h)
            T.RandomContrast(0.8, 1.2)
        ]
        
        mapper = DatasetMapper(cfg, is_train=True, augmentations=augmentations)
        return build_detection_train_loader(cfg, mapper=mapper)

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
    
    # File paths
    base_dir = "/run/user/1000/gvfs/smb-share:server=ecresearch.uwaterloo.ca,share=cviss/Jack/baf/360streetview"
    cfg.OUTPUT_DIR = os.path.join(base_dir, 'trained_models', 'test')
    
    # Find a model from detectron2's 
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    
    # Configure the datasets and GPU
    cfg.DATASETS.TRAIN = ("inlet_train",)
    cfg.DATASETS.TEST = ("inlet_val",)
    cfg.DATALOADER.NUM_WORKERS = 4
    
    # Initialize with pre-trained weights (transfers general image knowledge to your model)
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # Class 0: Inlet, Class 1: Debris
    
    # Hyperparameters (How fast and how long the model studies)
    cfg.SOLVER.IMS_PER_BATCH = 4         # Number of images processed at once by the GPU
    cfg.SOLVER.BASE_LR = 0.001           # Learning rate (how quickly it updates its guesses)
    cfg.SOLVER.MAX_ITER = 3000           # Total training steps
    cfg.MODEL.DEVICE = "cuda"            # Force the use of the GPU

    # Start Training
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = CustomTrainer(cfg)
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
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Only accept confident predictions (70%+)
    
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

def data_split(images_dir, single_json_file, dataset_dir, validation_ratio=0.2):
    """
    Splits a single COCO JSON file and an image directory into training and validation sets.
    """
    print("Loading original COCO JSON...")
    with open(single_json_file, 'r') as f:
        coco_data = json.load(f)
        
    images = coco_data.get('images', [])
    annotations = coco_data.get('annotations', [])
    categories = coco_data.get('categories', [])
    info = coco_data.get('info', {})

    # 1. Shuffle the images to ensure a random split
    random.seed(42) # Set seed for reproducibility
    random.shuffle(images)

    # 2. Calculate the split index
    val_size = int(len(images) * validation_ratio)
    val_images = images[:val_size]
    train_images = images[val_size:]

    # Get the image IDs for quick lookup
    train_img_ids = set([img['id'] for img in train_images])
    val_img_ids = set([img['id'] for img in val_images])

    # 3. Split the annotations based on which image they belong to
    train_anns = [ann for ann in annotations if ann['image_id'] in train_img_ids]
    val_anns = [ann for ann in annotations if ann['image_id'] in val_img_ids]
    
    # Optional: Fix Category IDs to start at 0 (Detectron2 prefers 0-indexed classes)
    # Your uploaded JSON uses IDs 1 and 2. This maps them to 0 and 1.
    for cat in categories:
        cat['id'] = cat['id'] - 1
    for ann in train_anns + val_anns:
        ann['category_id'] = ann['category_id'] - 1

    # 4. Construct the new COCO dictionaries
    train_coco = {'info': info, 'images': train_images, 'annotations': train_anns, 'categories': categories}
    val_coco = {'info': info, 'images': val_images, 'annotations': val_anns, 'categories': categories}

    # 5. Create the output directories
    train_img_dir = os.path.join(dataset_dir, 'train_images')
    val_img_dir = os.path.join(dataset_dir, 'val_images')
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)

    # 6. Copy the images into their respective folders
    print("Copying training images...")
    for img in train_images:
        src = os.path.join(images_dir, img['file_name'])
        dst = os.path.join(train_img_dir, img['file_name'])
        if os.path.exists(src):
            shutil.copyfile(src, dst)
        else:
            print(f"Warning: Image {src} not found in source folder!")

    print("Copying validation images...")
    for img in val_images:
        src = os.path.join(images_dir, img['file_name'])
        dst = os.path.join(val_img_dir, img['file_name'])
        if os.path.exists(src):
            shutil.copyfile(src, dst)
        else:
            print(f"Warning: Image {src} not found in source folder!")

    # 7. Save the new JSON files
    with open(os.path.join(dataset_dir, 'train.json'), 'w') as f:
        json.dump(train_coco, f)
    with open(os.path.join(dataset_dir, 'val.json'), 'w') as f:
        json.dump(val_coco, f)
        
    print(f"Data split successful!")
    print(f"Training: {len(train_images)} images, {len(train_anns)} annotations.")
    print(f"Validation: {len(val_images)} images, {len(val_anns)} annotations.")

if __name__ == '__main__':

    torch.cuda.empty_cache()

    # File paths    
    base_dir = "/run/user/1000/gvfs/smb-share:server=ecresearch.uwaterloo.ca,share=cviss/Jack/baf/360streetview"
    training_images = os.path.join(base_dir, 'segmentation_training', 'synthetic', 'copy_paste', 'dataset', 'images') 
    training_labels = os.path.join(base_dir, 'segmentation_training', 'synthetic', 'copy_paste', 'dataset', 'synthetic.json')
    dataset_dir = os.path.join(base_dir, 'segmentation_training', 'synthetic', 'copy_paste', 'dataset')
 
    # Split the datasets for training and validation 

    data_split(training_images, training_labels, dataset_dir, validation_ratio=0.2)  

    
    # Train the ORCNN model
    cfg = train_orcnn_model(dataset_dir)
