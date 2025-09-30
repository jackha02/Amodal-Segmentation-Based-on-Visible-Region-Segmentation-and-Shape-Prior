import pandas as pd
from scipy.spatial.transform import Rotation as R
import pymap3d as pm
from itertools import islice
from haversine import haversine, Unit

# Input: panorama location and headings
# Output: number of images from closest to each inlet cropped and ready for annotation

def extract_inlet_location(file_path):
    """
    Extract latitude and longtitude of drain inlets from the city database
    Warning: The function is specifically designed for City of Waterloo's database format
    the csv file can be downloaded here: https://data.waterloo.ca/maps/City-of-Waterloo::stormwater-catch-basins
    :param file_path: str, file path to a csv file containing the city inlet database
    """

    """
    Future edits: instead of hard coding, change so that the location is extracted based on the city name
    Ie. input: Waterloo ==> Find the columns that share the longtitude and latitude
    """
    df = pd.read_csv(file_path)
    inlet_locations = pd.DataFrame(df[["y", "x"]].values, columns=["latitude", "longitude"])
    inlet_locations = inlet_locations.dropna()
    return inlet_locations

def camera_location(file_path, heading):
    """
    Convert local poses of the camera to geodetic coordinates and headings
    Assumption: The first row of the csv file represents the starting global coordinates and heading
    The csv file is structured as: [local translation | rotation quaternions]
    Warning: The input csv file must be delimited by space
    :param file_path: str, file path to a csv file containing the data collection vehicle poses
    :param heading: float, heading relative to true north at the start of data collection
    """
    df = pd.read_csv(file_path, sep=' ')
    origin = df.iloc[0, 1:4]
    camera_locations = []
    for index, row in islice(df.iterrows(), 1, None):
        panoramas_id = row.iloc[0]
        east, north, up = row.iloc[1], row.iloc[2], row.iloc[3]
        lat, lon, alt = pm.enu2geodetic(east, north, up, origin.iloc[0], origin.iloc[1], origin.iloc[2])
        w, x, y, z = row.iloc[4], row.iloc[5], row.iloc[6], row.iloc[7]
        r = R.from_quat([x, y, z, w])
        yaw, pitch, roll = r.as_euler('zxy', degrees=True)
        heading = yaw % 360
        camera_locations.append([panoramas_id, lat, lon, alt, heading])
    camera_locations = pd.DataFrame(camera_locations, columns=['panoramas_id', 'lat', 'lon', 'alt', 'heading'])
    return camera_locations

def closest_panoramas_id(inlet_location, camera_locations):
    """
    Get the ids of the closest panoramas to drain inlets
    The distance between two points on the globe is calculated using haversine algorithm
    Assumption: Difference in the elevation is not considered in the haversine formula
    :param inlet_location: list of floats, latitude and longitude of the inlet
    :param camera_locations: dataframe, geodetic coordinates of the camera
    """
    shortest_distance = []
    for index, row in camera_locations.iterrows():
        camera_coordinate = [row.iloc[1], row.iloc[2]]
        distance = haversine(inlet_location, camera_coordinate)
        if distance is None:
            distance = shortest_distance
            pano_id = row.iloc[0]
        elif distance >= shortest_distance:
            break
        else:
            distance = shortest_distance
            pano_id = row.iloc[0]
    return pano_id



if __name__ == '__main__':
    inlet_locations = extract_inlet_location(file_path="../360streetview/waterloo.csv")
    camera_locations = camera_location(file_path="../360streetview/poses.csv", heading = 0)
    print(inlet_locations.iloc[:, :])
    print(camera_locations.iloc[:, :])





