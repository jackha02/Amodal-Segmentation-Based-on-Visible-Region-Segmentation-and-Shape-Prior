import os
import cv2
import Equirec2Perspec as E2P
import glob
from tqdm import tqdm

if __name__ == '__main__':
    # Input parameters:
    panorama_file_path = os.path.join(os.path.dirname(__file__), '..', 'input_data', 'site2')
    output_path = os.path.join(os.path.dirname(__file__), '..', 'output', 'images', 'site2')
    pano_height = 1024
    pano_width = 2048
    image_per_pano = 3
    top_crop = 384
    image_width = 640
    image_height = 640
    FOV = 120
    for folder_path in tqdm(glob.glob(os.path.join(panorama_file_path, "*"))):
        folder_name = os.path.basename(folder_path)
        save_folder = os.path.join(output_path, folder_name)
        for image_path in tqdm(glob.glob(os.path.join(folder_path, "*.png"))):
            filename = os.path.basename(image_path).split('.')[0]
            theta = 0
            for i in range(image_per_pano):
                equ = E2P.Equirectangular(image_path)
                img = equ.GetPerspective(FOV, theta, 0, image_height, image_width) 
                theta += 360/image_per_pano
                save_path = os.path.join(save_folder, f"{filename}_view_{i}.png")
                cv2.imwrite(save_path, img)