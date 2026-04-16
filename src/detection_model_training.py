from ultralytics import YOLO
import cv2
import os
import random
import shutil
import albumentations as A

def data_split(img_dir, label_dir, dataset_dir, validation_ratio):
    """
    Split the dataset into train and validation folders.
    :param img_dir: Directory to the custom image dataset.
    :param label_dir: Directory to the image labels.
    :param dataset_dir: Directory where the split dataset will be stored.
    :param validation_ratio: Ratio of the validation dataset.
    """
    # Sort the image and label folders to align them
    img_filenames = sorted(os.listdir(img_dir))
    label_filenames = sorted(os.listdir(label_dir))

    # Zip the images and labels to keep them aligned
    dataset = list(zip(img_filenames, label_filenames))
    random.shuffle(dataset)

    # Calculate the split point
    val_split_point = int(validation_ratio * len(dataset))
    
    # Map split names to their respective file slices
    splits = {
        "val": dataset[:val_split_point],
        "train": dataset[val_split_point:]
    }

    # Iterate through both splits to create directories and copy files
    for split_name, files in splits.items():
        # Setup destination paths
        img_dest_dir = os.path.join(dataset_dir, split_name, "images")
        lab_dest_dir = os.path.join(dataset_dir, split_name, "labels")

        # Create the directories
        os.makedirs(img_dest_dir, exist_ok=True)
        os.makedirs(lab_dest_dir, exist_ok=True)

        # Copy the paired files
        for img_name, lab_name in files:
            shutil.copyfile(os.path.join(img_dir, img_name), os.path.join(img_dest_dir, img_name))
            shutil.copyfile(os.path.join(label_dir, lab_name), os.path.join(lab_dest_dir, lab_name))

base_dir = "/run/user/1000/gvfs/smb-share:server=ecresearch.uwaterloo.ca,share=cviss/Jack/baf/360streetview"

def train_model():
    # 1. Load the model. 
    model = YOLO('yolov8n.pt')  

    # 2. Train the model
    results = model.train(
        data=os.path.join(base_dir, 'object_detection_training', 'bbox_data.yaml'),
        epochs=30, 
        imgsz=640,                     
        batch=16,
        project=os.path.join(base_dir, 'trained_models'),                        
        name='new',      
        device=0,

        # Implement online augmentation 
        # Geometric (Spatial & Viewpoint Variance)
        fliplr=0.5,        # 50% chance to flip horizontally
        degrees=0.0,       # No rotation (or set to 15.0 if you want the +/- 15 deg rotation back)
        translate=0.1,     # Translate by +/- 10%
        scale=0.2,         # Scale by +/- 20%
        mosaic=0.0,        # Turn off YOLO's default mosaic
        mixup=0.0,         # Turn off YOLO's default mixup
        
        # Photometric (Lighting & Sensor Variance)
        hsv_h=0.04,        # Hue shift fraction (approx 15 degrees)
        hsv_s=0.3,         # Saturation shift fraction (+/- 30%)
        hsv_v=0.2          # Value/Brightness shift fraction (+/- 20%)           
    )

    # 3. Validate the model
    metrics = model.val()

if __name__ == '__main__':
    # File paths:
    base_dir = "/run/user/1000/gvfs/smb-share:server=ecresearch.uwaterloo.ca,share=cviss/Jack/baf/360streetview"
    training_images = os.path.join(base_dir, 'object_detection_training', 'images') 
    training_labels = os.path.join(base_dir, 'object_detection_training', 'labels')
    dataset_dir = os.path.join(base_dir, 'object_detection_training', 'dataset')
    train_img_dir = os.path.join(dataset_dir, "train/images")
    train_lab_dir = os.path.join(dataset_dir, "train/labels")

    # Splite the dataset into training and validation
    data_split(training_images, training_labels, dataset_dir, validation_ratio=0.2)  
    train_model()
