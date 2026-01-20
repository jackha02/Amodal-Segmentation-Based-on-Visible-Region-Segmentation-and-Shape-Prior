import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
import os
import cv2
import pandas as pd


if __name__ == '__main__':
    filtered_images_folder = os.path.join(os.path.dirname(__file__), '..', 'pano_images', 'filterd')
    cropping_properties = os.path.join(os.path.dirname(__file__), '..', 'cropping_properties.csv')
    cropped_images = os.path.join(os.path.dirname(__file__), '..', 'cropped_images')
    FOV = 90
    output_size = [640, 640]
    df = pd.read_csv(cropping_properties)
    for image in os.listdir(filtered_images_folder):
        image_path = os.path.join(os.path.dirname(__file__), "{image}")
        index = df[df['pano_id'] == [image.split('.')[0]]].index(0)
        heading = df.iloc[1, index]
        pitch = df.iloc[2, index]
        image = panorama_to_plane(f"{image}", FOV, output_size[0], output_size[1], heading, pitch)
        full_path = os.path.join(cropped_images, f"{image}")
        cv2.imwrite(cropped_images, full_path)
        
