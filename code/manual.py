import os
import cv2
import Equirec2Perspec as E2P
import glob

def pano_params(pano_height, pano_width, image_per_pano, top_crop):
    """
    Calculates the rectilinear image width and height based on the desired number of images per panorama  
    """
    image_height = pano_height - top_crop
    image_width = pano_width/image_per_pano
    FOV = 360/image_per_pano
    return image_width, image_height, FOV

if __name__ == '__main__':
    # Input parameters:
    panorama_file_path = os.path.join(os.path.dirname(__file__), 'input_data', 'site1', 'test')
    output_path = os.path.join(os.path.dirname(__file__), 'output', 'images', 'site1', 'seq8')
    pano_height = 1024
    pano_width = 2048
    image_per_pano = 3
    top_crop = 200
    image_width, image_height, FOV = pano_params(pano_height, pano_width, image_per_pano, top_crop)
    for image_path in glob.glob(os.path.join(panorama_file_path, "*.png")):
        filename = os.path.basename(image_path).split('.')[0]
        theta = 0
        for i in range(image_per_pano):
            equ = E2P.Equirectangular(image_path)
            img = equ.GetPerspective(FOV, theta, 0, image_height, image_width) 
            theta += 360/image_per_pano
            save_path = os.path.join(output_path, f"{filename}_view_{i}.png")
            cv2.imwrite(save_path, img)