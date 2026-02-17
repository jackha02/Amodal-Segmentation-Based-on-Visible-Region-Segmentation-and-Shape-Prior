import random
import pandas as pd
from sklearn.neighbors import BallTree
from haversine import haversine, Unit
import math
from itertools import islice
import shutil
import os
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
import cv2 
from pathlib import Path


def extract_inlet_location(file_path):
    """
    Extract latitude and longtitude of drain inlets from the city database
    Warning: The function is specifically designed for City of Waterloo's database format
    the csv file can be downloaded here: https://data.waterloo.ca/maps/City-of-Waterloo::stormwater-catch-basins
    :param file_path: str, file path to a csv file containing the city inlet database
    Returns a geodetic coordinates of the drain inlet locations
    """
    df = pd.read_csv(file_path)
    inlet_locations = pd.DataFrame(df[["ASSET_ID", "y", "x"]].values, columns=["inlet_id", "lat", "lon"])
    inlet_locations = inlet_locations.dropna()
    return inlet_locations

def panorama_location(file_path):
    """
    Convert local poses of the panorama to geodetic coordinates using the enu2geodetic function
    Assumption: The first row of the csv file represents the starting geodetic coordinates
    The csv file is structured as: [local translation N x 3| rotation quaternions N x 4]
    Warning: The input csv file must be delimited by space
    :param file_path: str, file path to a csv file containing the data collection vehicle poses
    Returns a geodetic coordinates of the panorama locations
    """
    df = pd.read_csv(file_path, sep=' ')
    pano_locations = df.iloc[:, [0, 5, 4, 6]].copy()
    pano_locations.columns = ['panoramas_id', 'lat', 'lon', 'alt']
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
    1. At least 2 meters away from centeral pano
    2. Must be within 15 meters from the inlet
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
    approaching_candidates = pano_locations.iloc[:central_pano_index][::-1] # for approaching view, search from central to index 0
    leaving_candidates = pano_locations.iloc[central_pano_index+1:]
    pano_approaching = None
    pano_leaving = None
    
    def find_valid_pano(candidates):
        for row in candidates.itertuples(index = False):
            distance_to_central_pano = haversine([row[1], row[2]], [central_lat, central_lon], Unit.METERS)
            distance_to_inlet = haversine([row[1], row[2]], [inlet_locations[0], inlet_locations[1]], Unit.METERS)
            if distance_to_central_pano >= 1 and distance_to_inlet <= 15:
                return row[0]

    pano_approaching = find_valid_pano(approaching_candidates)
    pano_leaving = find_valid_pano(leaving_candidates)
    
    return[pano_approaching, pano_id, pano_leaving] 

def compute_heading_pitch(camera_height, inlet_location, view_loc):
    """
    Compute compass heading (0 to 360 degrees) from panorama center to drain inlet
    Assumption: The haversine between inlet and panorama effectively becomes a straight line due to a short distance  
    :param inlet_location: list of floats, latitude and longitude of the inlet
    :param view_loc: list of floats, latitude and longitude of the panorama
    Returns heading and heading from panorama center to drain inlet
    """
    inlet_lat, inlet_long = inlet_location[0], inlet_location[1]
    pano_lat, pano_long = view_loc[0], view_loc[1]
    inlet_loc = (inlet_lat, inlet_long)
    pano_loc = (pano_lat, pano_long)
    dx = (inlet_long - pano_long) * math.cos(math.radians((pano_lat + inlet_lat) / 2))
    dy = inlet_lat - pano_lat    
    heading = math.degrees(math.atan2(dx, dy))
    distance = haversine(inlet_loc, pano_loc, Unit.METERS)
    pitch = -abs(math.degrees(np.arctan2(camera_height, distance)))
    return heading, pitch

def cropping_properties(pano_heading, camera_height, inlet_id, inlet_location, multi_view, pano_locations):
    """
    To calculate the heading and pitch to center the drain inlet at the center of the rectilinear image
    :param inlet_location: list of floats, latitude and longitude of the inlet
    :param multi_view: list of floats, three panorama ids for the inlet
    """
    cropping_properties = []
    labels = ['approaching', 'center', 'leaving']
    for view_id, label in zip(multi_view, labels):
            if not view_id:
                continue
            view_row = pano_locations[pano_locations['panoramas_id'] == view_id]
            if view_row.empty:
                continue
            view_loc = (view_row.iloc[0]['lat'], view_row.iloc[0]['lon'])
            relative_heading, pitch = compute_heading_pitch(camera_height, inlet_location, view_loc)
            true_heading = (pano_heading + relative_heading) %360
            # Save the new filename stem so we can match it later
            cropping_properties.append([f"{inlet_id}_{label}", true_heading, pitch])
    return cropping_properties

def save_images(multi_view, inlet_id, input_path, output_path):
    """
    Given a set of three panorama ids for the drain inlet, extract and save images in a seperate folder
    Assumption: All panorama images are saved in the folder with panorama ids as their names
    :param multi_view: list of floats, three panorama ids for the inlet
    :param input_path: file path, directory to raw panorama images
    :param output_path: file path, directory to where the multiview images will be saved
    Three images are saved, named as {inlet_id}_{approaching/center/leaving}.jpg 
    """
    image_extensions = 'png'
    approaching, center, leaving = multi_view
    for label, pano_id in zip(['approaching', 'center', 'leaving'], [approaching, center, leaving]):
        pano_id = f"{pano_id:.6f}" # to enforce trailing zeros in panoramas ids
        input = os.path.join(input_path, f"{pano_id}.{image_extensions}")
        output = os.path.join(output_path, f"{inlet_id}_{label}.{image_extensions}")
        os.makedirs(os.path.dirname(output), exist_ok=True)
        try:
            shutil.copy(input, output)
        except FileNotFoundError:
            print(f"File not found: {input}")
            pass       

# The functions below were modified from https://github.com/fuenwang/Equirec2Perspec/blob/master/Equirec2Perspec.py
def xyz2lonlat(xyz):
    atan2 = np.arctan2
    asin = np.arcsin

    norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    xyz_norm = xyz / norm
    x = xyz_norm[..., 0:1]
    y = xyz_norm[..., 1:2]
    z = xyz_norm[..., 2:]

    lon = atan2(x, z)
    lat = asin(y)
    lst = [lon, lat]

    out = np.concatenate(lst, axis=-1)
    return out

def lonlat2XY(lonlat, shape):
    X = (lonlat[..., 0:1] / (2 * np.pi) + 0.5) * (shape[1] - 1)
    Y = (lonlat[..., 1:] / (np.pi) + 0.5) * (shape[0] - 1)
    lst = [X, Y]
    out = np.concatenate(lst, axis=-1)

    return out 

def pano_to_rect(FOV, theta, phi, h, w, img):
    # Determine the focal length
    f = w*1/(2*np.tan(np.radians(0.5*FOV)))

    # Construct the camera intrinsic matrix and then invert it
    cx = (w - 1) / 2.0 
    cy = (h - 1) / 2.0 # optical center of the image
    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0,  1],
    ], np.float32)
    K_inv = np.linalg.inv(K)

    # Project 2D pixel coordinate into a 3D direction vector in camera space
    x = np.arange(w)
    y = np.arange(h)
    x, y = np.meshgrid(x, y)
    z = np.ones_like(x)
    xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)
    xyz = xyz @ K_inv.T

    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    x_axis = np.array([1.0, 0.0, 0.0], np.float32)
    R1, _ = cv2.Rodrigues(y_axis * np.radians(theta))
    R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(phi))
    R = R2 @ R1
    xyz = xyz @ R.T
    lonlat = xyz2lonlat(xyz) 
    img = cv2.imread(img, cv2.IMREAD_COLOR)
    [height, wdith, _] = img.shape
    XY = lonlat2XY(lonlat, shape=img.shape).astype(np.float32)
    persp = cv2.remap(img, XY[..., 0], XY[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    return persp


def data_split(img_dir, label_dir, dataset_dir, validation_ratio, testing_ratio):
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
    test_split_point = int(testing_ratio * total) + val_split_point
    val_index = index_list[:val_split_point]
    test_index = index_list[val_split_point:test_split_point]
    train_index = index_list[test_split_point:]
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

if __name__ == '__main__':
    # File paths:
    inlet_data = os.path.join(os.path.dirname(__file__), '..', 'raw_data', 'waterloo.csv')
    pano_data = os.path.join(os.path.dirname(__file__), '..', 'raw_data', 'laurelwood1', 'camera_poses_transformed.txt')
    raw_images = os.path.join(os.path.dirname(__file__), '..', 'raw_data', 'laurelwood1', 'images')
    inlet_images = os.path.join(os.path.dirname(__file__), '..', 'filtered_data', 'filtered')
    cropping_properties_data = os.path.join(os.path.dirname(__file__), '..', 'filtered_data', 'cropping_properties.csv')
    training_images = os.path.join(os.path.dirname(__file__), '..', 'pre_datasplit', 'images') 
    training_labels = os.path.join(os.path.dirname(__file__), '..', 'pre_datasplit', 'labels')
    dataset_dir = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'train')

    # Panorama parameters:
    FOV = 120
    camera_height = 2.1  # height of the camera in meters
    vehicle_heading = 2850  # horizontal pixel coordinate from the center of the panorama

    # Output image parameters:
    output_size = (640, 640)
    
    # Extracting panorama images closest to the inlet
    inlet_properties = extract_inlet_location(inlet_data)
    pano_locations = panorama_location(pano_data)
    matches = closest_panoramas_id(inlet_properties, pano_locations)

    # Calculate rectilinear cropping properties and save multiview images
    heading_pitch = []
    for inlet_id, closest_panorama_id, shortest_distance in matches:
        inlet_index = inlet_properties[inlet_properties['inlet_id'] == inlet_id].index[0]
        inlet_locations = inlet_properties.iloc[inlet_index, 1:3]
        multi_view = get_multiview(inlet_locations, pano_locations, closest_panorama_id)
        heading_pitch.extend(cropping_properties(vehicle_heading, camera_height, inlet_id, inlet_locations, multi_view, pano_locations))
        print(heading_pitch)
        save_images(multi_view, inlet_id, raw_images, inlet_images)

    df = pd.DataFrame(heading_pitch, columns = ['filename', 'heading', 'pitch'])     
    df.to_csv(cropping_properties_data, index = False)
    
    # Save cropped rectilinear images in pre_datasplit/images folder
    df = pd.read_csv(cropping_properties_data)
    if not os.path.exists(training_images):
        os.makedirs(training_images)

    # Extract the panorama ID from image name, and find the corresponding heading and pitch from the cropping properties dataframe    
    for image_name in Path(inlet_images).iterdir():
        image_path = os.path.join(inlet_images, image_name)
        filename_str = image_name.stem
        stem = filename_str.split('.')[0]
        row = df[df['filename'] == stem]
        if row.empty:
            continue
        heading = row.iloc[0]['heading']
        pitch = row.iloc[0]['pitch']
        rect_img = pano_to_rect(FOV, heading, pitch, output_size[0], output_size[1], image_path)
        cv2.imwrite(training_images + '/' + image_name.name, rect_img)

    """
    # Split the data into training and validation folders
    split_ratio = 0.8
    data_split(training_images, training_labels, dataset_dir, validation_ratio=0.1, testing_ratio=0.1)  
    """