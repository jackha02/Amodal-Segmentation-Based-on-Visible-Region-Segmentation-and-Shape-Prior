import numpy as np
import os
import cv2 
import math
from ultralytics import YOLO
import pandas as pd
from pathlib import Path
from training_images import pano_to_rect

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
        :param src (str): File path to the folder containing all sets of images
        Returns np.ndarray a 1D (1 x 2)array of confidence scores for each class.
        """
        records = []

        # Ensure flat directories don't cause the loop to fail
        folders = [os.path.join(src, d) for d in os.listdir(src) if os.path.isdir(os.path.join(src, d))]
        if not folders:
            folders = [src]
        
        for folder_path in folders:
            for img in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img)
                
                if not img.lower().endswith(('.png')):
                    continue
                    
                img_name = Path(img_path).stem
                image = cv2.imread(img_path)
                if image is None:
                    continue
                h, w = image.shape[:2]

                results = self.model.predict(source=img_path, verbose=False)
                boxes = results[0].boxes

                # Discard images that have no or low-confidence bounding box
                valid_boxes = [b for b in boxes if b.conf[0].item() >= 0.3]
                if not valid_boxes:
                    continue

                # Function to check if two bounding boxes intersect (e.g., double inlets)
                def is_overlap(b1, b2):
                    return not (b1[2] < b2[0] or b1[0] > b2[2] or b1[3] < b2[1] or b1[1] > b2[3])

                box_coords = [b.xyxy[0].tolist() for b in valid_boxes]
                confs = [b.conf[0].item() for b in valid_boxes]

                groups = []
                for i, coords in enumerate(box_coords):
                    matched = False
                    for g in groups:
                        if any(is_overlap(coords, gc) for gc in g['coords']):
                            g['coords'].append(coords)
                            g['confs'].append(confs[i])
                            matched = True
                            break
                    if not matched:
                        groups.append({'coords': [coords], 'confs': [confs[i]]})

                # Merge each group and locate the one closest to the image center (i.e., projection direction)
                img_center_x, img_center_y = w / 2.0, h / 2.0
                best_group = None
                min_dist = float('inf')

                for g in groups:
                    min_x = min(c[0] for c in g['coords'])
                    min_y = min(c[1] for c in g['coords'])
                    max_x = max(c[2] for c in g['coords'])
                    max_y = max(c[3] for c in g['coords'])
                    
                    cx = (min_x + max_x) / 2.0
                    cy = (min_y + max_y) / 2.0
                    dist = (cx - img_center_x)**2 + (cy - img_center_y)**2
                    
                    # Target the merged box closest to the center line
                    if dist < min_dist:
                        min_dist = dist
                        best_group = {
                            'coords': g['coords'], 'confs,': g['confs'],
                            'x_center': cx, 'y_center': cy,
                            'bb_w': max_x - min_x, 'bb_h': max_y - min_y,
                            'confidence': max(g['confs'])
                        }

                records.append({
                    'filename': img_name, 
                    'x1': best_group['coords'][0][0],
                    'y1': best_group['coords'][0][1],
                    'x2': best_group['coords'][0][2],
                    'y2': best_group['coords'][0][3],
                    'x_center': best_group['x_center'],
                    'y_center': best_group['y_center'],
                    'bb_w': best_group['bb_w'],
                    'bb_h': best_group['bb_h'],
                    'img_w': w,
                    'img_h': h,
                    'confidence': best_group['confidence'],
                })

        df = pd.DataFrame(records)
        return df
    
def recompute_rectilinear(df_bboxes, cropping_properties_path, pano_dir, output_dir, orig_fov, buffer_pct, output_size):
    """
    Recalculate the correct projection heading and pitch using 3D vector rotation,
    and generate a new rectilinear image centered on the inlet directly from the filtered panorama.
    :param df_bboxes: dataframe containing bounding box center coordinates and image dimensions
    :param cropping_properties_path: path to the CSV containing original projection heading and pitch
    :param pano_dir: directory containing the filtered panorama images
    :param output_dir: directory to save the corrected rectilinear images
    :param orig_fov: original FOV used for the initial rectilinear images (in degrees)
    :param buffer_pct: percentage buffer to add around the inlet bounding box (e.g., 0.20 for 20%)
    :param output_size: tuple of (height, width) for the final corrected image
    """
    if df_bboxes.empty:
        print("Dataframe is empty. No images to process.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # Load original cropping properties to get the initial heading and pitch
    df_props = pd.read_csv(cropping_properties_path)

    for index, row in df_bboxes.iterrows():
        filename = row['filename']
        x_center = row['x_center']
        y_center = row['y_center']
        bb_w = row['bb_w']
        bb_h = row['bb_h']
        w = row['img_w']
        h = row['img_h']
        
        # Match with original cropping properties
        prop_row = df_props[df_props['filename'] == filename]
        if prop_row.empty:
            continue
            
        orig_heading = prop_row.iloc[0]['heading']
        orig_pitch = prop_row.iloc[0]['pitch']
        
        # 1. Compute Focal Length
        focal_length = (w / 2.0) / math.tan(math.radians(orig_fov / 2.0))
        
        # 2. Compute dynamic FOV based on bounding box size
        max_bb_dim = max(bb_w, bb_h)
        buffered_dim = max_bb_dim * (1.0 + buffer_pct)
        new_fov = 2 * math.degrees(math.atan((buffered_dim / 2.0) / focal_length))
        
        # 3. Create Local 3D Vector representing the target pixel
        # u: horizontal offset (positive is right)
        # v: vertical offset (positive is up, so we subtract y_center from h/2)
        u = x_center - (w / 2.0)
        v = (h / 2.0) - y_center 
        Vc = np.array([u, v, focal_length]) # Vector in camera's local coordinate system

        # 4. Define 3D Rotation Matrices
        pitch_rad = math.radians(orig_pitch)
        heading_rad = math.radians(orig_heading)

        # CORRECTED Pitch Rotation Matrix (Rotation around X-axis)
        # The signs are structured so that a negative pitch (looking down) forces 
        # the Y-coordinate of the forward vector to be negative.
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(pitch_rad), math.sin(pitch_rad)],
            [0, -math.sin(pitch_rad), math.cos(pitch_rad)]
        ])

        # Heading Rotation Matrix (Rotation around Y-axis)
        # Standard convention: clockwise heading is positive, rotating +Z towards +X
        Ry = np.array([
            [math.cos(heading_rad), 0, math.sin(heading_rad)],
            [0, 1, 0],
            [-math.sin(heading_rad), 0, math.cos(heading_rad)]
        ])

        # 5. Transform local vector to Global 3D Panoramic Vector
        # Apply pitch rotation first, then heading rotation
        Vg = Ry @ (Rx @ Vc)
        Xg, Yg, Zg = Vg

        # 6. Extract exact Global Heading and Pitch from the 3D Vector
        norm_Vg = np.linalg.norm(Vg)
        
        # Pitch is the angle of elevation from the XZ plane.
        # If Yg is negative (pointing at the ground), asin returns a negative angle.
        new_pitch = math.degrees(math.asin(Yg / norm_Vg))
        new_pitch = max(-90.0, min(90.0, new_pitch))
        
        # Heading is the angle in the XZ plane. atan2 handles the full 360-degree mapping natively
        new_heading = math.degrees(math.atan2(Xg, Zg))
        new_heading = ((new_heading + 180) % 360) - 180
        
        # Find the corresponding filtered panorama image
        pano_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            temp_path = os.path.join(pano_dir, f"{filename}{ext}")
            if os.path.exists(temp_path):
                pano_path = temp_path
                break
                
        if not pano_path:
            continue
            
        # Rerun pano_to_rect with the geometrically corrected angles and calculated FOV
        try:
            rect_img = pano_to_rect(new_fov, new_heading, new_pitch, output_size[0], output_size[1], pano_path)
            
            # Save the corrected rectilinear image
            out_filepath = os.path.join(output_dir, f"{filename}.png")
            cv2.imwrite(out_filepath, rect_img)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":

    # File paths
    base_dir = "/run/user/1000/gvfs/smb-share:server=ecresearch.uwaterloo.ca,share=cviss/Jack/baf/360streetview"
    model_path = os.path.join(base_dir, 'trained_models', 'yolov8.pt')
    data_path = os.path.join(base_dir, 'corrected_images', 'original') # path to initial rectilinear images
    filtered_pano_path = os.path.join(base_dir, 'filtered_data', 'filtered') # path to the filtered panoramas
    cropping_properties_path = os.path.join(base_dir, 'filtered_data', 'cropping_properties.csv') # path to the original cropping properties
    output_path = os.path.join(base_dir, 'corrected_images', 'corrected') # path to save the newly generated images
    df_path = os.path.join(base_dir, 'corrected_images') 

    # Adjustable configuration parameters
    ORIGINAL_FOV = 120        # The FOV used when the initial rectilinear images were created
    BUFFER_PCT = 0.5         # Percentage of buffer added around the bounding box (e.g., 0.20 for 20%)
    OUTPUT_SIZE = (640, 640)  # Desired dimensions for the final corrected images

    # Initialize the architecture with your trained weights
    classifier = MultiViewInletClassifier(model_weights_path=model_path)

    # Generate a dataframe that contains the center coordinates of the bounding boxes
    bb_coord = classifier.bounding_box(data_path)
    bb_coord.to_csv(os.path.join(df_path, 'bounding_boxes.csv'), index=False)

    # Recompute projection angles and extract the corrected rectilinear images
    recompute_rectilinear(
        df_bboxes=bb_coord, 
        cropping_properties_path=cropping_properties_path, 
        pano_dir=filtered_pano_path, 
        output_dir=output_path,
        orig_fov=ORIGINAL_FOV,
        buffer_pct=BUFFER_PCT,
        output_size=OUTPUT_SIZE
    )