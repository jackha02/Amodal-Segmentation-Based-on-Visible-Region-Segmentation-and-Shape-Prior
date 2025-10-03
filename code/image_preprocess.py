import pandas as pd
import pymap3d as pm
from haversine import haversine, Unit
import math
from itertools import islice
import shutil
import os

# Current workflow:
# Input: panorama location and headings
# Output: move the panorama images closest to each inlet into a designated output folder for model input

def extract_inlet_location(file_path):
    """
    Extract latitude and longtitude of drain inlets from the city database
    Warning: The function is specifically designed for City of Waterloo's database format
    the csv file can be downloaded here: https://data.waterloo.ca/maps/City-of-Waterloo::stormwater-catch-basins
    :param file_path: str, file path to a csv file containing the city inlet database
    Returns a geodetic coordinates of the drain inlet locations
    """

    """
    Future edits: instead of hard coding, change so that the location is extracted based on the city name
    Ie. input: Waterloo ==> Find the columns that share the longtitude and latitude
    """
    df = pd.read_csv(file_path)
    inlet_locations = pd.DataFrame(df["ASSET_TYPE", "y", "x"].values, columns=["inlet_id", "latitude", "longitude"])
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
    pano_locations = []
    for row in df.itertuples(index = False):
        pano_id = row[0]
        if row == 0:
            lat, lon, alt = row[1:4]
        else: 
            east, north, up = row[1], row[2], row[3]
            lat0, lon0, h0 = pano_locations(['lat, lon, alt'])
            lat, lon, alt = pm.enu2geodetic(east, north, up, lat0, lon0, h0)
        pano_locations.append([pano_id, lat, lon, alt])
    pano_locations = pd.DataFrame(pano_locations, columns=['panoramas_id', 'lat', 'lon', 'alt'])
    return pano_locations

def closest_panoramas_id(inlet_locations, pano_locations):
    """
    Obtain the ids of the closest panoramas for a given inlet id
    The distance between two points on the globe is calculated using haversine algorithm
    Assumption: Difference in the elevation is not considered in the haversine formula
    :param inlet_location: list of floats, latitude and longitude of the inlet
    :param pano_locations: dataframe, geodetic coordinates of the panorama
    Returns the inlet id, id of the closest panorama, and distance between the inlet and panorama
    """
    inlet_coordinates = inlet_locations[1,2]
    shortest_distance = []
    for row in pano_locations.itertuples(index = False):
        pano_coordinate = pano_locations(['lat, lon'])
        distance = haversine(inlet_coordinates, pano_coordinate)
        if distance is None or distance <= shortest_distance:
            distance = shortest_distance
            pano_id = pano_locations['panoramas_id']
        else:
            continue
        return inlet_locations[0], pano_id, shortest_distance
            

def compute_heading(inlet_location, pano_location):
    """
    Compute compass heading (0 to 360 degrees) from panorama center to drain inlet
    :param inlet_location: list of floats, latitude and longitude of the inlet
    :param pano_locations: list of floats, latitude and longitude of the panorama
    Returns heading from panorama center to drain inlet
    """
    inlet_lat, inlet_long = inlet_location[1, 2]
    pano_lat, pano_long = pano_location[1, 2]
    dx = (inlet_long - pano_long) * math.cos(math.radians((pano_lat + inlet_lat) / 2))
    dy = inlet_lat - pano_lat    
    heading = math.degrees(math.atan2(dx, dy)) % 360
    return heading

def compute_pitch(inlet_location, pano_location):
    """
    Compute the pitch adjustment (-90 to 90 degrees) required to position the inlet at the center of the panorama
    Assumption: The haversine between inlet and panorama effectively becomes a straight line due to a short distance
    The height of the panorama relative to the ground is fixed at 2 meters
    :param inlet_location: list of floats, latitude and longitude of the inlet
    :param camera_locations: list of floats, latitude and longitude of the panorama
    Returns pitch from panorama center to drain inlet
    """
    distance = haversine(inlet_location, pano_location, Unit.METERS)
    pitch = math.degrees(math.atan(2, distance))
    return pitch

def get_multiview(pano_locations, pano_id, shortest_distance):
    """
    Given the id of the closest panorama (centeral pano) to the inlet and its distance to the inlet, identify two nearby pano_ids that meet the following conditions:
    1. At least 2 meters away from centeral pano
    2. Must be within 10 meters from the inlet
    3. One approaching and one leaving the centeral pano along the travelling direction
    Assumption: Condtion 1 is set given that panorma captures 10 images every second and it is travelling at approximately 50 km/hr
    :param pano_locations: dataframe, ids and geodetic coordinates of the panorama locations
    :param pano_id: int, id of the closest panorama to the inlet
    :param shortest_distance: flaoat, distance between the the panorana 
    """
    candidate_panos = []
    central_pano_index = pano_locations[pano_locations['panoramas_id'] == pano_id].index
    pano_approaching = []
    pano_leaving = []
    multi_view = []
    for row in reversed(list(islice(pano_locations.itertuples(index = False), [central_pano_index - 6], [central_pano_index]))):
        distance_to_central_pano = haversine([row[1], row[2]], [candidate_panos[1,0], candidate_panos[2,0]], Unit.METERS)
        distance_to_inlet = distance_to_central_pano + shortest_distance
        if distance_to_central_pano < 2:
            continue
        elif distance_to_inlet > 10:
            continue
        else: 
            pano_approaching.append(row[0])
            break
    for row in islice(pano_locations.itertuples(index = False), [central_pano_index + 1], [central_pano_index + 7]):
        distance_to_central_pano = haversine([row[1], row[2]], [candidate_panos[1,0], candidate_panos[2,0]], Unit.METERS)
        distance_to_inlet = distance_to_central_pano + shortest_distance
        if distance_to_central_pano < 2:
            continue
        elif distance_to_inlet > 10:
            continue
        else: 
            pano_leaving.append(row[0])
            break
    multi_view.append[pano_approaching, pano_id, pano_leaving]  
    return multi_view

def save_images(multi_view, inlet_id, input_path, output_path):
    """
    For a set of three panorama ids for the drain inlet, extract and save images in a seperate folder
    Assumption: All panorama images are saved in the folder with panorama ids as their names
    :param multi_view: list of floats, three panorama ids for the inlet
    :param input_path: file path, directory to raw panorama images
    :param output_path: file path, directory to where the multiview images will be saved 
    """
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
    for view in multi_view:
        if multi_view[0].endswith(image_extensions) in os.listdir(input_path):
            try:
                shutil.move(os.path.join(input_path, view.image_extensions), os.path.join(output_path, f"{inlet_id}_{approaching}.{image_extensions}"))
            except FileNotFoundError:
                pass
        if multi_view[1].endswith(image_extensions) in os.listdir(input_path):
            try:
                shutil.move(os.path.join(input_path, view.image_extensions), os.path.join(output_path, f"{inlet_id}_{central}.{image_extensions}"))
            except FileNotFoundError:
                pass
        if multi_view[2].endswith(image_extensions) in os.listdir(input_path):
            try:
                shutil.move(os.path.join(input_path, view.image_extensions), os.path.join(output_path, f"{inlet_id}_{leaving}.{image_extensions}"))
            except FileNotFoundError:
                pass

if __name__ == '__main__':
    inlet_locations = extract_inlet_location(file_path="../waterloo.csv")
    pano_locations = panorama_location(file_path="../poses.csv")
    raw_images = "../360streetview/raw"
    inlet_images = "../360streetview/filtered"
    for row in inlet_locations.itertuples(inded = False):
        inlet_id, closest_panorama_id, shortest_distance = closest_panoramas_id(row[0], pano_locations)
        multi_view = get_multiview(pano_locations, closest_panorama_id, shortest_distance)
        save_image(multi_view, raw_images, inlet_images)
    
    






