import os
from pathlib import Path
from ultralytics import YOLO

def train_amodal_clogging_model(yaml_path, model_weights = "yolov8n-seg.pt", epochs = 100, imgsz = 640, batch_size = 16):
    """
    Trains a YOLOv8 instance segmentation model for amodal inlet detection and visible debris segmentation\
        yaml_path (str): Absolute path to the dataset data.yaml file.
        model_weights (str): Pretrained YOLOv8 segmentation weights to initialize with.
        epochs (int): Number of training epochs.
        imgsz (int): Target image size for training.
        batch_size (int): Batch size (adjust based on VRAM).
    """
    model = YOLO(model_weights)

    train_args = {
        "data": yaml_path,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch_size,
        "device": 0,                   # Use GPU 0. Set to 'cpu' if no GPU is available.
        "project": "inlet_clogging",   # Directory name for saving runs.
        "overlap_mask": False,         # Set to false amodal segmentation
        "mask_ratio": 1,               # Use full-resolution masks for precise area calculation.
        "save": True,                  # Save the best model checkpoints.
    }
    
    results = model.train(**train_args)
    return results

if __name__ == "__main__":
    # Define absolute paths to prevent working directory issues
    BASE_DIR = Path(__file__).parent.resolve()
    YAML_PATH = str(BASE_DIR / "dataset" / "data.yaml")
    
    # Verify the yaml exists before launching the heavy training process
    if not os.path.exists(YAML_PATH):
        raise FileNotFoundError(f"Could not find data.yaml at {YAML_PATH}")
    
    # Run the training
    # Note: If memory is an issue, drop batch_size to 8 or 4.
    train_amodal_clogging_model(
        yaml_path=YAML_PATH,
        model_weights="yolov8s-seg.pt", # Using 'small' variant for a good speed/accuracy tradeoff
        epochs=150,
        imgsz=640,
        batch_size=16
    )