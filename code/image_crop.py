import numpy as np
import os
import cv2 
from ultralytics import YOLO
from typing import List, Tuple, Dict
import pandas as pd
from pathlib import Path

class MultiViewInletClassifier:
    """
    A multi-view classification architecture utilizing a YOLO model.
    This framework evaluates three distinct views of a single inlet and uses max-score fusion to extract the appropriate bounding box
    """
    def __init__(self, model_weights_path: str):
        """
        Initializes the classifier with the pre-trained YOLO model.
        :param model_weights_path (str): Path to the trained YOLO weights (e.g., 'best.pt').
        """
        # Load the trained YOLO classification model
        self.model = YOLO(model_weights_path)

    def bounding_box(self, src):
        """
        Save the center coordinates of the bounding box for each image and its corresponding confidence
        Also, save the view with the highest confidence for each inlet separately for further processing
        Warning: the image without the bounding box is discarded
        :param : 
        :param src (str): File path to the folder containing all sets of images
        Returns np.ndarray a 1D (1 x 2)array of confidence scores for each class.
        """
        records = []

        # Iterate only through directories within the source path
        folders = [os.path.join(src, d) for d in os.listdir(src) if os.path.isdir(os.path.join(src, d))]
        
        for folder_path in folders:
            folder_name = Path(folder_path).stem

            for img in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img)
                
                # Skip non-image files if any exist
                if not img.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                img_name = Path(img_path).stem
                # Safely attempt to extract view from filename, fallback if format varies
                view = img_name.split('_')[1] if '_' in img_name else 'unknown'

                results = self.model.predict(source=img_path, verbose=False)
                result = results[0]

                # Find the center coordinates and confidence associated with the bounding box
                box = result.boxes
                
                # If no bounding box is detected in the inference, discard the image
                if len(box) == 0:
                    continue

                x_center, y_center = box.xywh[0][:2].int().tolist()
                confidence = box.conf[0].item()

                # If the bounding box has a low confidence score, also discard the image
                if confidence < 0.5:
                    continue

                # Append to our records list
                records.append({
                    'inlet_id': folder_name,
                    'view': view,
                    'x_center': x_center,
                    'y_center': y_center,
                    'confidence': confidence,
                    'img_path': img_path  # Saved for easy access during crop
                })

        df = pd.DataFrame(records)

        # Filter to keep the view with the highest object detection confidence score
        if not df.empty:
            best_indices = df.groupby('inlet_id')['confidence'].idxmax()
            df = df.loc[best_indices].reset_index(drop=True)

        return df
    
def crop(df_src, image_folder, output_path):
    """
    Retrieve the extent of cropping size
    Assumption: Assume that the size of the inlet is always less than 224 pixels (* minimum distance between the vehicle and inlet is set at 1 m) 
    :param df_src: dataframe that contains properties of the bounding boxes
    :param image_folder: the folder that contains all rectilinear images
    :output_path: the folder to save the cropped images
    Crops the original rectilinear into 224 x 224 images
    """
    if df_src.empty:
        print("Dataframe is empty. No images to crop.")
        return

    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    crop_size = 224
    half_size = crop_size // 2

    for index, row in df_src.iterrows():
        img_path = row['img_path']
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Failed to read image: {img_path}")
            continue

        h, w = img.shape[:2]
        x_center = int(row['x_center'])
        y_center = int(row['y_center'])

        # Calculate initial crop boundaries
        x1 = x_center - half_size
        y1 = y_center - half_size
        x2 = x_center + half_size
        y2 = y_center + half_size

        # Add padding if the proposed boundary exceeds the image edges
        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - w)
        pad_bottom = max(0, y2 - h)

        # Adjust boundaries to be within the image size for slicing
        x1_safe = max(0, x1)
        y1_safe = max(0, y1)
        x2_safe = min(w, x2)
        y2_safe = min(h, y2)

        # Crop the valid region
        cropped_img = img[y1_safe:y2_safe, x1_safe:x2_safe]

        # Apply padding if the box exceeded bounds, ensuring the output is exactly 224x224
        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            cropped_img = cv2.copyMakeBorder(
                cropped_img, 
                pad_top, pad_bottom, pad_left, pad_right, 
                cv2.BORDER_CONSTANT, 
                value=[0, 0, 0] # Pad with black pixels
            )

        # Save the cropped image
        out_filename = f"{row['inlet_id']}_{row['view']}_cropped.jpg"
        out_filepath = os.path.join(output_path, out_filename)
        cv2.imwrite(out_filepath, cropped_img)
        print(f"Saved: {out_filepath}")

if __name__ == "__main__":
    model_path = os.path.join(os.path.dirname(__file__), '..', 'trained_models', 'bounding_box.pt')
    data_path = os.path.join(os.path.dirname(__file__), '..', 'validation_data', 'original')
    output_path = os.path.join(os.path.dirname(__file__), '..', 'validation_data', 'cropped')

    # Initialize the architecture with your trained weights
    classifier = MultiViewInletClassifier(model_weights_path=model_path)

    # Generate a dataframe that contains the center coordinates of the bounding boxes
    bb_coord = classifier.bounding_box(data_path)

    # Crop the rectilinear images 
    crop(bb_coord, data_path, output_path)

        