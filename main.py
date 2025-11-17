import os
import pandas as pd
from itertools import islice
from code.image_extraction import extract_inlet_location, panorama_location, closest_panoramas_id, get_multiview, cropping_properties, save_images



if __name__ == '__main__':
    # Input file paths
    inlet_data = os.path.join(os.path.dirname(__file__), 'input_data', 'waterloo.csv')
    pano_data = os.path.join(os.path.dirname(__file__), 'input_data', 'poses.txt')
    raw_images = os.path.join(os.path.dirname(__file__), 'input_data', 'images')

    # Output file paths
    filtered_images = os.path.join(os.path.dirname(__file__), 'output', 'filtered_images')
    crop_properties = os.path.join(os.path.dirname(__file__), 'output', 'cropping_properties.csv') 
    inlet_locations = extract_inlet_location(inlet_data)
    pano_locations = panorama_location(pano_data)

    inlet_ids = []
    closest_panorama_ids = []
    shortest_distances = []
    inlet_id, closest_panorama_id, shortest_distance = closest_panoramas_id(inlet_locations, pano_locations)
    print(inlet_ids, closest_panorama_ids, shortest_distances)

    """
    heading_pitch = []
    for row in inlet_locations.itertuples(index = False):
        inlet_id, closest_panorama_id, shortest_distance = closest_panoramas_id(row, pano_locations)
        multi_view = get_multiview(pano_locations, closest_panorama_id, shortest_distance)
        heading_pitch.append(cropping_properties(row[1:4], multi_view))
        save_images(multi_view, inlet_id, raw_images, filtered_images)
    df = pd.DataFrame(heading_pitch, columns = ['pano_id', 'heading', 'pitch'])     
    df.to_csv(crop_properties, index = False)
    """