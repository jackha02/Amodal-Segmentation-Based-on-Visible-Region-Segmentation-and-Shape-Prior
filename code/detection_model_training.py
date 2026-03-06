from ultralytics import YOLO
import os
import random
import shutil

def data_split(img_dir, label_dir, dataset_dir, validation_ratio):
    """
    Split the dataset into train and validation folders
    :param img_dir, directory to the custom image dataset
    :param label_dir, directory to the image labels
    :param dataset_dir, directory to where the split dataset will be stored
    :param validation_ratio, ratio of validation dataset 
    :param testing_ratio, ratio of testing dataset
    """
    img_filenames = sorted([f for f in os.listdir(img_dir)])
    label_filenames = sorted([f for f in os.listdir(label_dir)])
    # Count the number of files in the folder
    total = len(img_filenames)
    # Create an array with a range of numbers starting at 1 to size of the folder
    index_list = list(range(0, total))
    random.shuffle(index_list)
    val_split_point = int(validation_ratio * total)
    val_index = index_list[:val_split_point]
    train_index = index_list[val_split_point:]
    # Using the index in the previous steps, split the images and labels accordingly to the coresponding folder
    train_img = os.path.join(dataset_dir, "train/images")
    train_lab = os.path.join(dataset_dir, "train/labels")
    val_img = os.path.join(dataset_dir, "val/images")
    val_lab = os.path.join(dataset_dir, "val/labels")
    test_img = os.path.join(dataset_dir, "testing/images")
    test_lab = os.path.join(dataset_dir, "testing/labels")
    os.makedirs(train_img, exist_ok=True)
    os.makedirs(train_lab, exist_ok=True)
    os.makedirs(val_img, exist_ok=True)
    os.makedirs(val_lab, exist_ok=True)
    os.makedirs(test_img, exist_ok=True)
    os.makedirs(test_lab, exist_ok=True)
    for idx in train_index:
        shutil.copy(os.path.join(img_dir, img_filenames[idx]), train_img)
        shutil.copy(os.path.join(label_dir, label_filenames[idx]), train_lab)
    # Copy validation samples
    for idx in val_index:
        shutil.copy(os.path.join(img_dir, img_filenames[idx]), val_img)
        shutil.copy(os.path.join(label_dir, label_filenames[idx]), val_lab)

# Splite the dataset into training and validation
training_images = os.path.join(os.path.dirname(__file__), '..', 'object_detection_training', 'images') 
training_labels = os.path.join(os.path.dirname(__file__), '..', 'object_detection_training', 'labels')
dataset_dir = os.path.join(os.path.dirname(__file__), '..', 'object_detection_training', 'dataset')

data_split(training_images, training_labels, dataset_dir, validation_ratio=0.2)  

def train_model():
    # 1. Load the model. 
    model = YOLO('yolo26n.pt')  

    # 2. Train the model
    results = model.train(
        data=os.path.join(os.path.dirname(__file__), '..', 'object_detection_training', 'bbox_data.yaml'),
        epochs=100, 
        imgsz=320,                     
        batch=32,
        project=os.path.join(os.path.dirname(__file__), '..', 'trained_models'),                        
        name='results',      
        device=0           
    )

    # 3. Validate the model
    metrics = model.val()

if __name__ == '__main__':
    train_model()