import os
import pandas as pd

# Camera intrinsics and extrinsics - all in meters
cam_height = 2 
pan_width = 2460
pan_height = 1230

# Multi view parameters
num_panos = 4

poses = pd.read_csv('poses.csv')
cb = pd.read_csv('city.csv')

# Projecton of image coordinates into real world coordinatess

# Extract latitude and longitude of stormwater inlets from the city database
data = pd.read_csv('waterloo.csv')
storm_inlets = data.iloc[1:8633, [18,17]]

# Extract Cartesian coordinates

enu = poses.iloc[0:,1:3]
print(enu)

'''
def glob_coord(ini_lat: float, ini_long: float, pano


# Compute compass heading from panorama center to stormwater inlet

def  +

# Local path to city's stormwater catch basins database
data = pd.read_csv('city.csv')


df = data.iloc[1601:1700, [18,17]]
'''

# Extract 