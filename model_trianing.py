from ultralytics import YOLO
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets
import random
import shutil

# Directory to images, labels, datasets
img_dir = os.path.join(os.path.dirname(__file__), "pre_datasplit/images")
label_dir = os.path.join(os.path.dirname(__file__), "pre_datasplit/labels")
dataset_dir = os.path.join(os.path.dirname(__file__), "dataset")
train_dir = os.path.join(dataset_dir, "train/images")
val_dir = os.path.join(dataset_dir, "val/images")
    
# Parameters for DataLoaders
batch_size = 36
epochs = 50
num_workers = os.cpu_count()

# Other input parameters
split_ratio = 0.8


data_split(img_dir, label_dir, dataset_dir, split_ratio)

"""
# Transforming steps using albumentations for tensor conversion
train_transform = A.Compose([
    A.RandomCrop(width=640, height=640),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.HueSaturationValue(p=0.2),
    ToTensorV2()
])

test_transform = A.Compose([
    A.RandomCrop(width=640, height=640),
    ToTensorV2()
])

# Transform images into tensors using ImageFolder
train_data = datasets.ImageFolder(root=train_dir, # target folder of images
                                  transform=train_transform, # transforms to perform on data (images)
                                  target_transform=None) # transforms to perform on labels (if necessary)

val_data = datasets.ImageFolder(root=val_dir, 
                                 transform=test_transform)

# Create DataLoaders
train_dataloader = DataLoader(dataset=train_data, 
                              batch_size=batch_size, # how many samples per batch?
                              num_workers=num_workers, # how many subprocesses to use for data loading? (higher = more)
                              shuffle=True) # shuffle the data?

test_dataloader = DataLoader(dataset=val_data, 
                             batch_size=batch_size, 
                             num_workers=num_workers, 
                             shuffle=False) # don't usually need to shuffle testing data

# Load a pretrained YOLOv8 model
model = YOLO('yolov8n.pt')
# Train on a custom dataset
model.train(data="../dataset/data.yaml", epochs=epochs)
# Evaluate performance
results = model.val()
# Optimize for deploymnet 
model.export(format='onnx')
"""

 