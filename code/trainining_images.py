import math
import shutil
import os
import random
import cv2 
import pandas as pd
import numpy as np
import py360convert
import utm
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import BallTree
from haversine import haversine, Unit
from pathlib import Path

def extract_inlet_location(waterloo_data, kitchener_data):
    """
    Extract latitude and longtitude of drain inlets from the city database
    Warning: The function is specifically designed for City of Waterloo's database format
    the csv file can be downloaded here: https://data.waterloo.ca/maps/City-of-Waterloo::stormwater-catch-basins
    :param file_path: str, file path to a csv file containing the city inlet database
    Returns a geodetic coordinates of the drain inlet locations
    """
    waterloo = pd.read_csv(waterloo_data)
    kitchener = pd.read_csv(kitchener_data)
    inlet_locations_waterloo = pd.DataFrame(waterloo[["ASSET_ID", "y", "x"]].values, columns=["inlet_id", "lat", "lon"])
    inlet_locations_waterloo = inlet_locations_waterloo.dropna()

    # Note that Kitchener database uses the UTM coordinates - therefore we must convert it to latitude and longtitude prior to processing 
    inlet_locations_kitchener = pd.DataFrame(kitchener[["OBJECTID", "x", "y"]].values, columns=["inlet_id", "utme", "utmn"])
    inlet_locations_kitchener = inlet_locations_kitchener.dropna()
    inlet_locations_kitchener[["lat", "lon"]] = inlet_locations_kitchener.apply(lambda row: utm.to_latlon(row['utme'], row['utmn'], 17, northern=True), axis=1, result_type='expand')
    inlet_locations_kitchener = inlet_locations_kitchener.drop(columns=['utme', 'utmn'])

    # Combine the databases
    inlet_locations = pd.concat([inlet_locations_waterloo, inlet_locations_kitchener], axis=0, ignore_index=True)
    return inlet_locations

import pandas as pd
import numpy as np

def panorama_location(file_path, lag_offset=1):
    """
    Assumption: The first row of the csv file represents the starting geodetic coordinates
    The csv file is structured as: [local translation N x 3| rotation quaternions N x 4]
    Warning: The input csv file must be delimited by space
    
    :param file_path: str, file path to a csv file containing the data collection vehicle poses
    :param lag_offset: int, number of rows to shift sensor data down to align with image IDs (default is 1)
    Returns geodetic coordinates of the panorama locations
    """
    df = pd.read_csv(file_path, sep=' ')
    
    # 1. Shift all columns EXCEPT the first one (panoramas_id) downwards
    data_columns = df.columns[1:] 
    df[data_columns] = df[data_columns].shift(lag_offset)
    
    # 2. Drop the initial rows that now contain NaN due to the downward shift
    df = df.iloc[lag_offset:].reset_index(drop=True)

    # Proceed with the original extraction logic
    pano_locations = df.iloc[:, [0, 5, 4, 6]].copy()
    pano_locations.columns = ['panoramas_id', 'lat', 'lon', 'alt']
      
    # Shift coordinates to calculate the vector to the NEXT panorama
    pano_locations['next_lat'] = pano_locations['lat'].shift(-1)
    pano_locations['next_lon'] = pano_locations['lon'].shift(-1)

    # Handle the final row by carrying over the previous distance delta
    pano_locations.loc[pano_locations.index[-1], 'next_lat'] = pano_locations['lat'].iloc[-1] + (pano_locations['lat'].iloc[-1] - pano_locations['lat'].iloc[-2])
    pano_locations.loc[pano_locations.index[-1], 'next_lon'] = pano_locations['lon'].iloc[-1] + (pano_locations['lon'].iloc[-1] - pano_locations['lon'].iloc[-2])

    # Calculate True North trajectory heading using spherical approximation
    dx = (pano_locations['next_lon'] - pano_locations['lon']) * np.cos(np.radians((pano_locations['lat'] + pano_locations['next_lat']) / 2))
    dy = pano_locations['next_lat'] - pano_locations['lat']
    
    # Calculate global heading clockwise from North
    pano_locations['camera_yaw'] = (np.degrees(np.arctan2(dx, dy)) + 360) % 360
    
    # Drop the temp columns to keep the dataframe clean
    pano_locations = pano_locations.drop(columns=['next_lat', 'next_lon'])
    
    return pano_locations

def closest_panoramas_id(inlet_locations, pano_locations):
    """
    Iterates through all inlet locations and finds the closest panorama id
    The distance between two points on the globe is calculated using haversine algorithm
    Assumption: Difference in the elevation is omitted in the haversine formula
    :param inlet_location: list of floats, latitude and longitude of the inlet
    :param pano_locations: dataframe, geodetic coordinates of the panorama
    Returns the inlet id, id of the closest panorama, and distance between the inlet and panorama
    """
    # Convert the coordaintes into radians for BallTree
    inlet_deg = inlet_locations[['lat', 'lon']].to_numpy()
    pano_deg  = pano_locations[['lat', 'lon']].to_numpy()
    inlet_rad = np.radians(inlet_deg)
    pano_rad  = np.radians(pano_deg)
    tree = BallTree(pano_rad, metric='haversine')
    results = []
    for i in range(len(inlet_rad)):    
        dist_rad, index = tree.query(inlet_rad[i].reshape(1,-1), k=1) # finds the closest panorama 
        dist_m = float(dist_rad[0][0]*6371000) # multiply the radians by radius of the earth to convert to meters
        if dist_m < 10:
            nearest_index = index[0][0]
            pano_id = float(pano_locations.iloc[nearest_index]['panoramas_id'])
            inlet_id = int(inlet_locations.iloc[i]['inlet_id'])
            results.append([inlet_id, pano_id, dist_m])
    return results

def get_multiview(inlet_locations, pano_locations, pano_id):
    """
    Given the id of the closest panorama (centeral pano) to the inlet and its distance to the inlet, identify two nearby pano_ids that meet the following conditions:
    1. At least 1 meters away from centeral pano
    2. Must be within 10 meters from the inlet
    3. One approaching and one leaving the centeral pano along the travelling direction
    Assumption: Condtion 1 is set given that panorma captures 10 images every second and it is travelling at approximately 50 km/hr
    :param inlet_locations: list of floats, latitude and longitude of the inlet 
    :param pano_locations: dataframe, ids and geodetic coordinates of the panorama locations
    :param pano_id: int, id of the closest panorama to the inlet
    :param shortest_distance: float, distance between the the panorama
    Returns a list of three panorama ids: [approaching pano id, centeral pano id, leaving pano id]
    """
    central_pano_index = pano_locations[pano_locations['panoramas_id'] == pano_id].index[0]
    central_lat, central_lon = pano_locations.iloc[central_pano_index, 1:3]

    # Restrict the search to a maximum of 10 panoramas before and after the central pano
    # The restriction is necessary to prevent the algorithm from grouping panoramas from a later revisit
    start_idx_approaching = max(0, central_pano_index - 10)
    end_idx_leaving = min(len(pano_locations), central_pano_index + 11)

    approaching_candidates = pano_locations.iloc[start_idx_approaching:central_pano_index][::-1] 
    leaving_candidates = pano_locations.iloc[central_pano_index+1:end_idx_leaving]

    pano_approaching = None
    pano_leaving = None
    
    # Find valid panoramas near central pano within the restricted window
    def find_valid_pano(candidates):
        for row in candidates.itertuples(index = False):
            distance_to_central_pano = haversine([row[1], row[2]], [central_lat, central_lon], Unit.METERS)
            distance_to_inlet = haversine([row[1], row[2]], [inlet_locations['lat'], inlet_locations['lon']], Unit.METERS)
            # Required conditions for valid panos
            if distance_to_central_pano >= 1 and distance_to_inlet <= 10:
                return row[0]
            
    pano_approaching = find_valid_pano(approaching_candidates)
    pano_leaving = find_valid_pano(leaving_candidates)

    return[pano_approaching, pano_id, pano_leaving] 

def compute_heading_pitch(camera_height, inlet_location, view_loc):
    """
    Compute compass heading (0 to 360 degrees) from panorama center to drain inlet
    Assumption: The haversine between inlet and panorama effectively becomes a straight line due to a short distance  
    :param camera_height: float, height of the camera in meters
    :param inlet_location: pandas Series/dict with 'lat' and 'lon'
    :param view_loc: tuple of (lat, lon) for the panorama
    Returns heading and heading from panorama center to drain inlet
    """
    inlet_lat, inlet_long = inlet_location['lat'], inlet_location['lon']
    pano_lat, pano_long = view_loc[0], view_loc[1]

    # Calculate the differences
    dx = (inlet_long - pano_long) * math.cos(math.radians((pano_lat + inlet_lat) / 2))
    dy = inlet_lat - pano_lat   

    # Calculate the global heading clockwise from North
    heading = (math.degrees(math.atan2(dx, dy)) + 360) % 360 # a positive value between 0 and 360

    # Calculate pitch
    distance = haversine((inlet_lat, inlet_long), (pano_lat, pano_long), Unit.METERS)
    pitch = math.degrees(math.atan2(-camera_height, distance)) # negative pitch for downward looking
    return heading, pitch

def cropping_properties(inlet_id, inlet_location, multi_view, pano_locations, camera_height, pano_width, vehicle_pixel_x):
    """
    Calculate the relative heading (theta) and pitch (phi) to center the drain inlet.
    Note that we account for a case where vehicle travelling direction is not aligned with the panorama direction
    :param inlet_id: int, the ID of the current inlet
    :param inlet_location: pandas Series/dict with 'lat' and 'lon'
    :param multi_view: list of floats, three panorama ids [approaching, center, leaving]
    :param pano_locations: dataframe, contains 'lat', 'lon', and 'camera_yaw'
    :param camera_height: float, height of camera in meters
    """
    cropping_data = []
    labels = ['approaching', 'center', 'leaving']
    
    # Calculate how many degrees the vehicle's travelling direction is offset from the panorama center [-180, 180]
    vehicle_offset_deg = ((vehicle_pixel_x / pano_width) - 0.5) * 360
    
    for view_id, label in zip(multi_view, labels):
        if not view_id:
            continue
            
        # Ensure pano_id format matches the image names exactly
        view_id_str = f"{view_id:.6f}"
        
        view_row = pano_locations[pano_locations['panoramas_id'] == view_id]
        if view_row.empty:
            continue
        
        view_loc = (view_row.iloc[0]['lat'], view_row.iloc[0]['lon'])
        vehicle_global_heading = view_row.iloc[0]['camera_yaw'] 
        inlet_global_heading, pitch = compute_heading_pitch(camera_height, inlet_location, view_loc)
        camera_center_heading = (vehicle_global_heading - vehicle_offset_deg) % 360

        if camera_center_heading < inlet_global_heading:
            alpha = (inlet_global_heading - camera_center_heading)
            pano_to_rect = (alpha + 180) % 360 - 180  
        elif camera_center_heading > inlet_global_heading:
            alpha = (camera_center_heading - inlet_global_heading)
            pano_to_rect = (-alpha + 180) % 360 - 180
        else:
            pano_to_rect = 0  

        # Include pano_id in the filename to prevent overwriting from intersecting trajectories
        cropping_data.append([f"{inlet_id}_{label}", pano_to_rect, pitch])
        
    return cropping_data

def save_images(multi_view, inlet_id, input_path, output_path):
    """
    Given a set of three panorama ids for the drain inlet, extract and save images in a seperate folder
    Assumption: All panorama images are saved in the folder with panorama ids as their names
    :param multi_view: list of floats, three panorama ids for the inlet
    :param inlet_id: int, inlet id of interest
    :param input_path: file path, directory to raw panorama images
    :param output_path: file path, directory to where the multiview images will be saved
    Three images are saved, named as {inlet_id}_{approaching/center/leaving}.jpg 
    """
    image_extensions = 'png'
    approaching, center, leaving = multi_view
    for label, pano_id in zip(['approaching', 'center', 'leaving'], [approaching, center, leaving]):
        if pano_id is None:
            continue
            
        pano_id_str = f"{pano_id:.6f}"
        input = os.path.join(input_path, f"{pano_id_str}.{image_extensions}")
        
        # Output name now includes pano_id_str to match the CSV
        output = os.path.join(output_path, f"{inlet_id}_{label}.{image_extensions}")
        
        os.makedirs(os.path.dirname(output), exist_ok=True)
        try:
            shutil.copy(input, output)
        except FileNotFoundError:
            print(f"File not found: {input}")
            pass    

def pano_to_rect(FOV, theta, phi, h, w, img_path):
    """
    Given a set of three panorama ids for the drain inlet, extract and save images in a seperate folder
    Assumption: All panorama images are saved in the folder with panorama ids as their names
    :param multi_view: list of floats, three panorama ids for the inlet
    :param inlet_id: int, inlet id of interest
    :param input_path: file path, directory to raw panorama images
    :param output_path: file path, directory to where the multiview images will be saved
    Three images are saved, named as {inlet_id}_{approaching/center/leaving}.jpg 
    """
    # Read the image from the given file path
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    # Check if the image loaded correctly to prevent silent failures
    if img is None:
        raise ValueError(f"Could not load image at {img_path}")
    persp = py360convert.e2p(img, fov_deg=FOV, u_deg=theta, v_deg=phi, out_hw=(int(h), int(w)), mode='bilinear')

    return persp

if __name__ == '__main__':
    # File paths:
    base_dir = "/run/user/1000/gvfs/smb-share:server=ecresearch.uwaterloo.ca,share=cviss/Jack/baf/360streetview"

    waterloo_data = os.path.join(base_dir, 'raw_data', 'waterloo.csv')
    kitchener_data = os.path.join(base_dir, 'raw_data', 'kitchener.csv')
    raw_data = os.path.join(base_dir, 'raw_data')
    inlet_images = os.path.join(base_dir, 'filtered_data', 'filtered')
    cropping_properties_data = os.path.join(base_dir, 'filtered_data', 'cropping_properties.csv')
    training_images = os.path.join(base_dir, 'corrected_images', 'original') 

    # Panorama parameters:
    FOV = 120
    camera_height = 2.1  # height of the camera in meters
    output_size = (640, 640)
    pano_width = 4096 # in pixels
    vehicle_pixel_x = 2835 # vehicle's travelling direction offset from the left of pano

    inlet_properties = extract_inlet_location(waterloo_data, kitchener_data)
    
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
    
    # Save cropped rectilinear images in pre_datasplit/images folder
    df = pd.read_csv(cropping_properties_data)
    if not os.path.exists(training_images):
        os.makedirs(training_images)

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
        save_path = os.path.join(training_images + '/' + inlet_id)
        os.makedirs(save_path, exist_ok=True)

        cv2.imwrite(os.path.join(save_path, image_name.name), rect_img)

