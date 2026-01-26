import statistics as stats
import pandas as pd
import cv2
import math
import random
from pathlib import Path

def panorama_heading(folder_path):
    """
    Determines the panorama heading by calculating the GPS bearing and adjusting for the camera's local boresight offset using Optical Flow
    :param folder_path: Path to the folder containing panorama images
    Returns: float, the absolute heading of the driving direction (0-360)
    """
    folder_path = Path(folder_path)
    # Randomly select 10 sets of 2 consecutive images from the image folder
    image_set = [] 
    pano_heading = []
    extensions = {'.jpg', '.jpeg', '.png'}
    if not folder_path.exists():
        print(f"Folder {folder_path} does not exist")
        return 0
    images = [f for f in folder_path.iterdir() if f.suffix.lower() in extensions]
    # Assume that all filenames are a float
    images.sort(key=lambda x: float(x.stem))
    while len(image_set) < 10 and len(image_set) < len(images) - 1:
        img1_index = random.choice(range(0, len(images)-1))
        img2_index = img1_index + 1
        img1 = images[img1_index]
        img2 = images[img2_index]
        image_set.append([img1, img2])
    # Use optical flow across the image to determine the vehicle travelling direction
    # OpenCV's optical flow function requires grayscale images
    for img1_path, img2_path in image_set:
        img1 = cv2.imread(str(img1_path), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(str(img2_path), cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None:
            continue
        flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[...,1])
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(magnitude)
        # Calculate the offset between the image center and vehicle travelling direction
        image_width = img1.shape[1]
        center_x = image_width / 2
        travelling_dir_x = min_loc[0]
        pixel_offset = travelling_dir_x - center_x
        angular_offset_u = (pixel_offset / image_width) * 360
        pano_heading.append(angular_offset_u)
    if not pano_heading:
        return 0
    median_pano_heading = stats.median(pano_heading)
    return median_pano_heading

if __name__ == "__main__":
    folder_path = "../raw_data/images"
    heading = panorama_heading(folder_path)
    print(heading)
