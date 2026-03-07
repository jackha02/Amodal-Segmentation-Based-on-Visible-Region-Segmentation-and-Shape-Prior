import os
import pandas as pd
import cv2 
import utm
from pathlib import Path
from trainining_images import panorama_location, closest_panoramas_id, get_multiview, cropping_properties, save_images, pano_to_rect
from image_corrections import MultiViewInletClassifier, recompute_rectilinear


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
    # File paths:
    base_dir = "/run/user/1000/gvfs/smb-share:server=ecresearch.uwaterloo.ca,share=cviss/Jack/baf/360streetview/deploy"
    target_area = os.path.join(base_dir, 'raw_data', 'test_area.csv')
    raw_data = os.path.join(base_dir, '..', 'raw_data')
    inlet_images = os.path.join(base_dir, 'filtered_data', 'filtered')
    cropping_properties_data = os.path.join(base_dir, 'filtered_data', 'cropping_properties.csv')
    intial_projection_images = os.path.join(base_dir, 'initial_projection_images') 

    # Panorama parameters:
    FOV = 120
    camera_height = 2.1  # height of the camera in meters
    output_size = (640, 640)
    pano_width = 4096 # in pixels
    vehicle_pixel_x = 2835 # vehicle's travelling direction offset from the left of pano

    inlet_properties = extract_inlet_location(target_area)
    
    # List to aggregate all heading and pitch data across folders
    all_heading_pitch = []

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
            
        print(f"Processing panoramas in: {folder_name}")
        
        pano_locations = panorama_location(pano_data)

        # Match inlets to closest panoramas for this specific folder
        matches = closest_panoramas_id(inlet_properties, pano_locations)

        # Calculate rectilinear cropping properties and save multiview images
        for inlet_id, closest_panorama_id, shortest_distance in matches:
            inlet_row = inlet_properties.loc[inlet_properties['inlet_id'] == inlet_id].iloc[0]
            inlet_locations = inlet_row[['lat', 'lon']]
            multi_view = get_multiview(inlet_locations, pano_locations, closest_panorama_id)
            props = cropping_properties(inlet_id, inlet_locations, multi_view, pano_locations, camera_height, pano_width, vehicle_pixel_x)
            all_heading_pitch.extend(props) # save all cropping properties in one dataframe
            save_images(multi_view, inlet_id, raw_images, inlet_images)

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
    data_path = os.path.join(base_dir, 'intial_projection_images') # path to initial rectilinear images
    filtered_pano_path = os.path.join(base_dir, 'filtered_data', 'filtered') # path to the filtered panoramas
    cropping_properties_path = os.path.join(base_dir, 'filtered_data', 'cropping_properties.csv') # path to the original cropping properties
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
        cropping_properties_path=cropping_properties_path, 
        pano_dir=filtered_pano_path, 
        output_dir=os.path.join(reprojected_images_path, 'images'),
        orig_fov=ORIGINAL_FOV,
        buffer_pct=BUFFER_PCT,
        output_size=OUTPUT_SIZE
    )