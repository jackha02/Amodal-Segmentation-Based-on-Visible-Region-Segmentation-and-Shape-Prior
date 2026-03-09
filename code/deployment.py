import os
import pandas as pd
import cv2 
import utm
from pathlib import Path
from trainining_images import panorama_location, closest_panoramas_id, get_multiview, cropping_properties, save_images, pano_to_rect
from image_corrections import MultiViewInletClassifier, recompute_rectilinear
from segmentation_validate import process_image_data
from detectron2 import model_zoo
from detectron2.config import get_cfg
from aistron.config import add_aistron_config
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor

def extract_inlet_location(target_area):
    """
    Extract latitude and longtitude of the target inlets from the city database
    Warning: The function is specifically designed for City of Waterloo's database format
    The csv file can be downloaded here: https://data.waterloo.ca/maps/City-of-Waterloo::stormwater-catch-basins
    :param target_area: str, file path to a csv file containing the inlets within the roi
    Returns a geodetic coordinates of the drain inlet locations
    """
    target_area_df = pd.read_csv(target_area)

    # Note that Kitchener database uses the UTM coordinates - therefore we must convert it to latitude and longtitude prior to processing 
    inlet_locations = pd.DataFrame(target_area_df[["OBJECTID", "x", "y"]].values, columns=["inlet_id", "utme", "utmn"])
    inlet_locations = inlet_locations.dropna()
    inlet_locations[["lat", "lon"]] = inlet_locations.apply(lambda row: utm.to_latlon(row['utme'], row['utmn'], 17, northern=True), axis=1, result_type='expand')
    inlet_locations = inlet_locations.drop(columns=['utme', 'utmn'])

    return inlet_locations

if __name__ == '__main__':
    """
    # File paths:
    base_dir = "/run/user/1000/gvfs/smb-share:server=ecresearch.uwaterloo.ca,share=cviss/Jack/baf/360streetview/deploy"
    target_area = os.path.join(base_dir, 'test_area.csv') # path to csv file containing the locations of the target inlets
    raw_data = os.path.join(base_dir, '..', 'raw_data') # path to raw panorama images
    inlet_images = os.path.join(base_dir, 'filtered_data', 'filtered') # path to save the filtered panoramas
    cropping_properties_data = os.path.join(base_dir, 'filtered_data', 'cropping_properties.csv') # path to save the cropping properties
    intial_projection_images = os.path.join(base_dir, 'initial_projection_images') # path to save the intial rectilinear images

    # Panorama parameters:
    FOV = 120
    camera_height = 2.1  # height of the camera in meters
    output_size = (640, 640)
    pano_width = 4096 # in pixels
    vehicle_pixel_x = 2835 # vehicle's travelling direction offset from the left of pano

    inlet_properties = extract_inlet_location(target_area)

    best_matches = {}

    # Iterate through all subdirectories in raw_data
    for folder_name in os.listdir(raw_data):
        folder_path = os.path.join(raw_data, folder_name)
        
        # Skip files (like waterloo.csv and kitchener.csv)
        if not os.path.isdir(folder_path):
            continue

        pano_data = os.path.join(folder_path, 'camera_poses_transformed.txt')
        raw_images = os.path.join(folder_path, 'images')

        # Skip if the required files/folders do not exist in the current subdirectory
        if not os.path.exists(pano_data) or not os.path.exists(raw_images):
            continue
            
        pano_locations = panorama_location(pano_data)

        # Match inlets to closest panoramas for this specific folder
        matches = closest_panoramas_id(inlet_properties, pano_locations)

        # Calculate rectilinear cropping properties and save multiview images
        for inlet_id, closest_panorama_id, shortest_distance in matches:
            if inlet_id not in best_matches or shortest_distance < best_matches[inlet_id]['distance']:
                inlet_row = inlet_properties.loc[inlet_properties['inlet_id'] == inlet_id].iloc[0]
                inlet_locations = inlet_row[['lat', 'lon']]
                
                multi_view = get_multiview(inlet_locations, pano_locations, closest_panorama_id)
                props = cropping_properties(inlet_id, inlet_locations, multi_view, pano_locations, camera_height, pano_width, vehicle_pixel_x)
                # Store the winning data
                best_matches[inlet_id] = {
                    'distance': shortest_distance,
                    'props': props,
                    'multi_view': multi_view,
                    'raw_images_path': raw_images
                }

    all_heading_pitch = []
    
    for inlet_id, match_data in best_matches.items():
        all_heading_pitch.extend(match_data['props']) 
        save_images(match_data['multi_view'], inlet_id, match_data['raw_images_path'], inlet_images)
    df = pd.DataFrame(all_heading_pitch, columns = ['filename', 'heading', 'pitch'])     
    df.to_csv(cropping_properties_data, index = False)

    # Extract the panorama ID from image name, and find the corresponding heading and pitch from the cropping properties dataframe    
    for image_name in Path(inlet_images).iterdir():
        image_path = os.path.join(inlet_images, image_name)
        filename_str = image_name.stem
        inlet_id = filename_str.split('_')[0] # inlet id without the view title

        row = df[df['filename'] == filename_str]
        if row.empty:
            continue

        heading = row.iloc[0]['heading']
        pitch = row.iloc[0]['pitch']

        rect_img = pano_to_rect(FOV, heading, pitch, output_size[0], output_size[1], image_path)
        save_path = os.path.join(intial_projection_images + '/' + inlet_id)
        os.makedirs(save_path, exist_ok=True)
        cv2.imwrite(os.path.join(save_path, image_name.name), rect_img)

    # Correct the intial projection heading and pitch using the object detection model
    model_path = os.path.join(base_dir, 'image_reprojection', 'yolov8.pt') # path to the trained object detection model
    data_path = os.path.join(base_dir, 'initial_projection_images') # path to initial rectilinear images
    filtered_pano_path = os.path.join(base_dir, 'filtered_data', 'filtered') # path to the filtered panoramas
    reprojected_images_path = os.path.join(base_dir, 'image_reprojection') # path to save the newly generated images

    # Adjustable configuration parameters
    ORIGINAL_FOV = 120        # The FOV used when the initial rectilinear images were created
    BUFFER_PCT = 0.5         # Percentage of buffer added around the bounding box (e.g., 0.20 for 20%)
    OUTPUT_SIZE = (640, 640)  # Desired dimensions for the final corrected images

    # Initialize the architecture with your trained weights
    classifier = MultiViewInletClassifier(model_weights_path=model_path)

    # Generate a dataframe that contains the center coordinates of the bounding boxes
    bb_coord = classifier.bounding_box(data_path)
    bb_coord.to_csv(os.path.join(reprojected_images_path, 'bounding_boxes.csv'), index=False)

    # Recompute projection angles and extract the corrected rectilinear images
    recompute_rectilinear(
        df_bboxes=bb_coord, 
        cropping_properties_path=cropping_properties_data, 
        pano_dir=filtered_pano_path, 
        output_dir=os.path.join(reprojected_images_path, 'images'),
        orig_fov=ORIGINAL_FOV,
        buffer_pct=BUFFER_PCT,
        output_size=OUTPUT_SIZE
    )
    """
    # Extract clogging extent percetange and segmented images for visualization

    # File paths
    base_dir = "/run/user/1000/gvfs/smb-share:server=ecresearch.uwaterloo.ca,share=cviss/Jack/baf/360streetview/deploy"
    output_img_dir = os.path.join(base_dir, "results")
    csv_output_path = os.path.join(output_img_dir, "clogging_results.csv")
    reprojected_images_path = os.path.join(base_dir, 'image_reprojection')

    # Set up configurations to deploy pretrained ORCNN model
    cfg = get_cfg()
    add_aistron_config(cfg)
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

    cfg.MODEL.WEIGHTS = "/home/wonny/orcnn/output/model_final.pth"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2 
    cfg.MODEL.DEVICE = "cuda"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # Lowered to 0.5 to catch more debris

    predictor = DefaultPredictor(cfg)

    # Colour schemes for the inlet and debris class types
    metadata = MetadataCatalog.get("inlet_val").set(
        thing_classes=["Inlet", "Debris"],
        thing_colors=[(255, 255, 255), (0, 0, 0)] 
    )

    data_rows = []

    for img_name in os.listdir(reprojected_images_path):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(reprojected_images_path, img_name)
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