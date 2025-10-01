import pandas as pd
import pymap3d as pm
from haversine import haversine, Unit
import math

# Input: panorama location and headings
# Output: Threee images closest to each inlet cropped and ready for annotation

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
    inlet_locations = pd.DataFrame(df[["y", "x"]].values, columns=["latitude", "longitude"])
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

def closest_panoramas_id(inlet_location, pano_locations):
    """
    Get the ids of the closest panoramas to drain inlets
    The distance between two points on the globe is calculated using haversine algorithm
    Assumption: Difference in the elevation is not considered in the haversine formula
    :param inlet_location: list of floats, latitude and longitude of the inlet
    :param pano_locations: dataframe, geodetic coordinates of the panorama
    Returns id of the closest panorama to the inlet    
    """
    shortest_distance = []
    for row in pano_locations.itertuples(index = False):
        pano_coordinate = pano_locations(['lat, lon'])
        distance = haversine(inlet_location, pano_coordinate)
        if distance is None or distance <= shortest_distance:
            distance = shortest_distance
            pano_id = pano_locations['panoramas_id']
        else:
            continue
    return pano_id

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

def get_multiview(pano_locations, pano_id):
    """
    Given the id of the closest panorama (pano_1) to the inlet, identify two nearby pano_ids that meet the following conditions:
    1. At least 2 meters away from pano_1
    2. Must be within 10 meters from the inlet
    3. One greater and one less than pano_1 (i.e., one just ahead and one just behind the closest panorama)
    Assumption: Condtion 1 is set with the asumption that panorma captures 10 images every second
    :pano_locations: dataframe, ids and geodetic coordinates of the panorama locations
    :param pano_id: int, id of the closest panorama to the inlet
    """
    candidate_panos = []
    for row in pano_locations.itertuples(index = False):
        proximity = abs(pano_id - row[0])
        if proximity == 0:
            continue
        else:
            candidate_panos.append(row[0], proximity) 
        candidate_panos = pd.DataFrame(candidate_panos, columns="pano_id", "proximity")
        candidate_panos.sort_values(by=['proximity'])


        
        

if __name__ == '__main__':
    inlet_locations = extract_inlet_location(file_path="../360streetview/waterloo.csv")
    camera_locations = panorama_location(file_path="../360streetview/poses.csv")
    






