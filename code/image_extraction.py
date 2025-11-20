import pandas as pd
import pymap3d as pm
from sklearn.neighbors import BallTree
from haversine import haversine, Unit
import math
from itertools import islice
import shutil
import os
import numpy as np

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
    :param shortest_distance: flaoat, distance between the the panorana 
    """
    central_pano_index = pano_locations[pano_locations['panoramas_id'] == pano_id].index(0)
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
            pano_approaching.append(row[0])
            break
    for row in islice(pano_locations.itertuples(index = False), central_pano_index + 1, central_pano_index + 7):
        distance_to_central_pano = haversine([row[1], row[2]], [central_lat, central_lon], Unit.METERS)
        distance_to_inlet = distance_to_central_pano + shortest_distance
        if distance_to_central_pano < 2:
            continue
        elif distance_to_inlet > 10:
            continue
        else: 
            pano_leaving.append(row[0])
            break
    multi_view.append([pano_approaching, pano_id, pano_leaving])  
    return multi_view

def compute_heading_pitch(inlet_location, pano_location):
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
    heading = math.degrees(math.atan2(dx, dy)) % 360
    distance = haversine(inlet_location, pano_location, Unit.METERS)
    pitch = math.degrees(math.atan(2 / distance))
    return heading, pitch

def cropping_properties(inlet_location, multi_view):
    """
    To calculate the heading and pitch to center the drain inlet at the center of the rectilinear image
    :param inlet_location: list of floats, latitude and longitude of the inlet
    :param multi_view: list of floats, three panorama ids for the inlet
    """
    cropping_properties = []
    for view in multi_view:
            heading, pitch = compute_heading_pitch(inlet_location, view)
            cropping_properties.append([f"{view}", heading, pitch])
    return cropping_properties

def save_images(multi_view, inlet_id, input_path, output_path):
    """
    Given a set of three panorama ids for the drain inlet, extract and save images in a seperate folder
    Assumption: All panorama images are saved in the folder with panorama ids as their names
    :param multi_view: list of floats, three panorama ids for the inlet
    :param input_path: file path, directory to raw panorama images
    :param output_path: file path, directory to where the multiview images will be saved 
    """
    image_extensions = 'jpg'
    approaching, center, leaving = multi_view
    for label, pano_id in (zip(['approaching', 'center', 'leaving'], approaching, center, leaving)):
        input = os.path.join(input_path, f"{pano_id}.{image_extensions}")
        output = os.path.join(output_path, f"{inlet_id}_{label}.{image_extensions}")
        if os.path.exists(input):
            try:
                shutil.move(input, output)
            except FileNotFoundError:
                pass       

if __name__ == '__main__':
    inlet_data = os.path.join(os.path.dirname(__file__), 'waterloo.csv')
    pano_data = os.path.join(os.path.dirname(__file__), 'poses.csv')
    raw_images = os.path.join(os.path.dirname(__file__), 'pano_images', 'raw')
    filtered_images = os.path.join(os.path.dirname(__file__), 'pano_images', 'filtered')
    heading_pitch_path = os.path.join(os.path.dirname(__file__), 'cropping_properties.csv') 

    inlet_locations = extract_inlet_location(inlet_data)
    pano_locations = panorama_location(pano_data)

    heading_pitch = []
    for row in inlet_locations.itertuples(index = False):
        inlet_id, closest_panorama_id, shortest_distance = closest_panoramas_id(row, pano_locations)
        multi_view = get_multiview(pano_locations, closest_panorama_id, shortest_distance)
        heading_pitch.append(cropping_properties(row[1:4], multi_view))
        save_images(multi_view, inlet_id, raw_images, filtered_images)
    df = pd.DataFrame(heading_pitch, columns = ['pano_id', 'heading', 'pitch'])     
    df.to_csv(heading_pitch_path, index = False)
    




