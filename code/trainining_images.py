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
import tqdm
import glob
import labelbox as lb

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
    pano_locations = df.iloc[:, 0:4].copy()
    pano_locations.columns = ['panoramas_id', 'lat', 'lon', 'alt']
    return pano_locations

def closest_panoramas_id(inlet_locations, pano_locations):
    """
    Iterates through all inlet locations and finds the closest panoramam id
    The distance between two points on the globe is calculated using haversine algorithm
    Assumption: Difference in the elevation is not considered in the haversine formula
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
        dist_rad, index = tree.query(inlet_rad[i].reshape(1,-1), k=1)
        dist_m = float(dist_rad[0][0]*6371000)
        if dist_m < 100:
            nearest_index = index[0][0]
            pano_id = float(pano_locations.iloc[nearest_index]['panoramas_id'])
            inlet_id = float(inlet_locations.iloc[i]['inlet_id'])
            results.append([inlet_id, pano_id, dist_m])
    return results

def get_multiview(pano_locations, pano_id, shortest_distance):
    """
    Given the id of the closest panorama (centeral pano) to the inlet and its distance to the inlet, identify two nearby pano_ids that meet the following conditions:
    1. At least 2 meters away from centeral pano
    2. Must be within 10 meters from the inlet
    3. One approaching and one leaving the centeral pano along the travelling direction
    Assumption: Condtion 1 is set given that panorma captures 10 images every second and it is travelling at approximately 50 km/hr
    :param pano_locations: dataframe, ids and geodetic coordinates of the panorama locations
    :param pano_id: int, id of the closest panorama to the inlet
    :param shortest_distance: float, distance between the the panorama
    Returns a list of three panorama ids: [approaching pano id, centeral pano id, leaving pano id]
    """
    central_pano_index = pano_locations[pano_locations['panoramas_id'] == pano_id].index[0]
    central_lat, central_lon = pano_locations.iloc[central_pano_index, 1:3]
    pano_approaching = []
    pano_leaving = []
    multi_view = []
    for row in reversed(list(islice(pano_locations.itertuples(index = False), central_pano_index - 6, central_pano_index))):
        distance_to_central_pano = haversine([row[1], row[2]], [central_lat, central_lon], Unit.METERS)
        distance_to_inlet = distance_to_central_pano + shortest_distance
        if distance_to_central_pano < 2:
            continue
        elif distance_to_inlet > 10:
            continue
        else: 
            pano_approaching = row[0]
            break
    for row in islice(pano_locations.itertuples(index = False), central_pano_index + 1, central_pano_index + 7):
        distance_to_central_pano = haversine([row[1], row[2]], [central_lat, central_lon], Unit.METERS)
        distance_to_inlet = distance_to_central_pano + shortest_distance
        if distance_to_central_pano < 2:
            continue
        elif distance_to_inlet > 10:
            continue
        else: 
            pano_leaving = row[0]
            break
    multi_view = [pano_approaching, pano_id, pano_leaving]
    return multi_view

def panorama_heading(u, width):
    """
    Determines the panorama heading by calculating the GPS bearing and adjusting for the camera's local boresight offset using Optical Flow
    :u: float, the horizontal pixel coordinate in the panorama image
    :image_size: tuple, the size of the panorama image (width, height)
    Returns the panorama heading in degrees
    """
    pano_heading =(u / width)%360
    return pano_heading

def compute_heading_pitch(camera_height, inlet_location, pano_location):
    """
    Compute compass heading (0 to 360 degrees) from panorama center to drain inlet
    Assumption: The haversine between inlet and panorama effectively becomes a straight line due to a short distance  
    :param inlet_location: list of floats, latitude and longitude of the inlet
    :param pano_locations: list of floats, latitude and longitude of the panorama
    Returns heading and heading from panorama center to drain inlet
    """
    inlet_lat, inlet_long = inlet_location[1], inlet_location[2]
    pano_lat, pano_long = pano_location[1], pano_location[2]
    dx = (inlet_long - pano_long) * math.cos(math.radians((pano_lat + inlet_lat) / 2))
    dy = inlet_lat - pano_lat    
    heading = math.degrees(math.atan2(dx, dy))
    distance = haversine(inlet_location, pano_location, Unit.METERS)
    pitch = np.arctan2(camera_height, distance)
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
    image_extensions = 'jpg'
    approaching, center, leaving = multi_view
    for label, pano_id in zip(['approaching', 'center', 'leaving'], [approaching, center, leaving]):
        if not pano_id:
            continue
        input = os.path.join(input_path, f"{pano_id}.{image_extensions}")
        output = os.path.join(output_path, f"{inlet_id}_{label}.{image_extensions}")
        os.makedirs(os.path.dirname(output), exist_ok=True)
        try:
            shutil.move(input, output)
        except FileNotFoundError:
            pass       

# The functions below were copied and adapted from https://blogs.codingballad.com/unwrapping-the-view-transforming-360-panoramas-into-intuitive-videos-with-python-6009bd5bca94
def map_to_sphere(x, y, z, W, H, f, yaw_radian, pitch_radian):
    """
    Converts a point from Cartesian coordinate to Spherical coordinate
    :param x, y, z: floats, Cartesian coordinate
    :param W, H, f: floats, width, height, and focal length of the rectilinear image
    :param yaw_radian, pitch_radian: floats, yaw and pitch of the rectilinear image
    Returns angle theta (angle from positive z-axis to the point) and phi (angle from positive x-axis in the xy-plane)
    """
    theta = np.arccos(z / np.sqrt(x ** 2 + y ** 2 + z ** 2))
    phi = np.arctan2(y, x)

    theta_prime = np.arccos(np.sin(theta) * np.sin(phi) * np.sin(pitch_radian) + np.cos(theta) * np.cos(pitch_radian))
    phi_prime = np.arctan2(np.sin(theta) * np.sin(phi) * np.cos(pitch_radian) - np.cos(theta) * np.sin(pitch_radian), np.sin(theta) * np.cos(phi))
    phi_prime += yaw_radian
    phi_prime = phi_prime % (2 * np.pi)
    return theta_prime.flatten(), phi_prime.flatten()

def interpolate_color(coords, img, method='bilinear'):
    """
    Samples RGB color values from an image coordinate
    :param coords: floats, a point on the image in Spherical coordinate
    :param img: np array, RGB color values of a point on the image
    Returns an np array of RGB color valuess 
    """
    order = {'nearest': 0, 'bilinear': 1, 'bicubic': 3}.get(method, 1)
    red = map_coordinates(img[:, :, 0], coords, order=order, mode='reflect')
    green = map_coordinates(img[:, :, 1], coords, order=order, mode='reflect')
    blue = map_coordinates(img[:, :, 2], coords, order=order, mode='reflect')
    return np.stack((red, green, blue), axis=-1)

def panorama_to_plane(panorama_path, FOV, output_size, yaw, pitch):
    """
    Transforms the panorama image into a rectilinear image
    :param panorama_path: file path, directory to the panorama image path
    :param FOV: int, field of view of the rectilinear image
    :param output_size: a list of int, width and height of the rectilinear image
    :param yaw, pitch: int, yaw and pitch angle of the rectilinear image 
    Returns the rectified image at a given size, yaw, and pitch
    """
    panorama = Image.open(panorama_path).convert('RGB')
    pano_width, pano_height = panorama.size
    pano_array = np.array(panorama)
    yaw_radian = np.radians(yaw)
    pitch_radian = np.radians(pitch)
    W, H = output_size
    f = (0.5 * W) / np.tan(np.radians(FOV) / 2)
    u, v = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    x = u - W / 2
    y = H / 2 - v
    z = f
    theta, phi = map_to_sphere(x, y, z, yaw_radian, pitch_radian)
    U = phi * pano_width / (2 * np.pi)
    V = theta * pano_height / np.pi
    U, V = U.flatten(), V.flatten()
    coords = np.vstack((V, U))
    colors = interpolate_color(coords, pano_array)
    output_image = Image.fromarray(colors.reshape((H, W, 3)).astype('uint8'), 'RGB')
    return output_image

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
    pano_data = os.path.join(os.path.dirname(__file__), '..', 'raw_data', 'poses.csv')
    raw_images = os.path.join(os.path.dirname(__file__), '..', 'raw_data', 'images')
    inlet_images = os.path.join(os.path.dirname(__file__), '..', 'filtered_data', 'filtered')
    cropping_properties_data = os.path.join(os.path.dirname(__file__), '..', 'filtered_data', 'cropping_properties.csv')
    training_images = os.path.join(os.path.dirname(__file__), '..', 'pre_datasplit', 'images') 
    training_labels = os.path.join(os.path.dirname(__file__), '..', 'pre_datasplit', 'labels')
    dataset_dir = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'train')

    # Panorama parameters:
    FOV = 90
    input_size = [2048, 1024]
    output_size = [640, 640]
    camera_height = 1.5  # height of the camera in meters
    lidar_offset = 461  # horizontal pixel coordinate from the center of the panorama

    # Panorama heading calculation
    vehicle_heading = panorama_heading(lidar_offset, input_size[0])
    print(f"Calculated Vehicle Heading: {vehicle_heading}")

    # Extracting panorama images closest to the inlet
    inlet_locations = extract_inlet_location(inlet_data)
    pano_locations = panorama_location(pano_data)
    matches = closest_panoramas_id(inlet_locations, pano_locations)

    # Calculate rectilinear cropping properties and save multiview images
    heading_pitch = []
    for inlet_id, closest_panorama_id, shortest_distance in matches:
        inlet_row = inlet_locations[inlet_locations['inlet_id'] == inlet_id].iloc[0]
        inlet_loc = (inlet_row['lat'], inlet_row['lon'])
        multi_view = get_multiview(pano_locations, closest_panorama_id, shortest_distance)
        heading_pitch.extend(cropping_properties(vehicle_heading, camera_height, inlet_id, inlet_loc, multi_view, pano_locations))
        save_images(multi_view, inlet_id, raw_images, inlet_images)
    df = pd.DataFrame(heading_pitch, columns = ['filename', 'heading', 'pitch'])     
    df.to_csv(cropping_properties_data, index = False)

    # Save cropped rectilinear images in pre_datasplit/images folder
    df = pd.read_csv(cropping_properties_data)
    if not os.path.exists(training_images):
        os.makedirs(training_images)

    # Extract the panorama ID from image name, and find the corresponding heading and pitch from the cropping properties dataframe    
    for image_name in tqdm(os.listdir(inlet_images)):
        image_path = os.path.join(inlet_images, image_name)
        stem = image_name.split('.')[0]
        row = df[df['filename'] == stem]
        if row.empty:
            continue
        heading = row.iloc[0]['heading']
        pitch = row.iloc[0]['pitch']
        pil_image = panorama_to_plane(image_path, FOV, output_size[0], output_size[1], heading, pitch)
        
        # Convert PIL RGB to OpenCV BGR
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        full_path = os.path.join(training_images, image_name)
        cv2.imwrite(full_path, opencv_image)

    # Split the data into training and validation folders
    split_ratio = 0.8
    data_split(training_images, training_labels, dataset_dir, validation_ratio=0.1, testing_ratio=0.1)