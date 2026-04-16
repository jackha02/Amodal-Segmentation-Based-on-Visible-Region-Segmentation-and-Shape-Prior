import json
import os

base_dir = "/run/user/1000/gvfs/smb-share:server=ecresearch.uwaterloo.ca,share=cviss/Jack/baf/360streetview"
all_json   = os.path.join(base_dir, 'segmentation_training', 'synthetic', 'synthetic.json')  # Path to the merged COCO annotation JSON (annotations_all.json)

# Load the generated COCO JSON
with open(all_json, 'r') as f:
    data = json.load(f)

unique_segmentations = set()

# Loop through all annotations
for ann in data['annotations']:
    # COCO segmentations are lists of lists (e.g., [[x1, y1, ...]]). 
    # We convert them to nested tuples so they can be hashed and added to a set.
    seg_tuple = tuple(tuple(part) for part in ann['segmentation'])
    unique_segmentations.add(seg_tuple)

print(f"Total annotations generated: {len(data['annotations'])}")
print(f"Number of unique inlets used: {len(unique_segmentations)}")
import cv2
import numpy as np
from matplotlib import pyplot as plt


def draw_polygon_boundary(image, segmentation, color, thickness=2):
    """
    Draws only the boundary lines of a polygon mask on the image.
    segmentation: list of lists (COCO format)
    color: (B, G, R) tuple
    thickness: line thickness
    """
    for seg in segmentation:
        pts = np.array(seg, dtype=np.int32).reshape(-1, 2)
        if len(pts) >= 2:
            cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness)
    return image


def visualize_masks(
    json_path,
    images_dir,
    output_dir,
    num_images=10,
    random_seed=42
):
    """
    Visualize amodal and visible mask boundaries for a sample of images from a COCO JSON.
    Args:
        json_path: Path to COCO annotation JSON
        images_dir: Directory containing images
        output_dir: Directory to save visualizations
        num_images: Number of images to visualize
        random_seed: Seed for reproducibility
    """
    os.makedirs(output_dir, exist_ok=True)
    with open(json_path, 'r') as f:
        coco = json.load(f)

    # Build image_id to file_name mapping
    image_id_to_file = {img['id']: img['file_name'] for img in coco['images']}

    # Build image_id to annotations mapping
    from collections import defaultdict
    anns_by_image = defaultdict(list)
    for ann in coco['annotations']:
        anns_by_image[ann['image_id']].append(ann)

    # Select images
    rng = np.random.RandomState(random_seed)
    image_ids = list(image_id_to_file.keys())
    rng.shuffle(image_ids)
    selected_ids = image_ids[:num_images]

    for idx, img_id in enumerate(selected_ids):
        file_name = image_id_to_file[img_id]
        img_path = os.path.join(images_dir, file_name)
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue

        anns = anns_by_image[img_id]
        vis_img = img.copy()

        # Draw amodal mask boundaries in blue, visible mask boundaries in red
        for ann in anns:
            # Amodal mask boundary
            if 'segmentation' in ann and ann['segmentation']:
                vis_img = draw_polygon_boundary(vis_img, ann['segmentation'], color=(255, 0, 0), thickness=2)
            # Visible mask boundary
            if 'visible_segmentation' in ann and ann['visible_segmentation']:
                vis_img = draw_polygon_boundary(vis_img, ann['visible_segmentation'], color=(0, 0, 255), thickness=2)

        # Save visualization
        out_path = os.path.join(output_dir, f"viz_{idx:03d}_{file_name}")
        cv2.imwrite(out_path, vis_img)
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    # User parameters
    base_dir = "/run/user/1000/gvfs/smb-share:server=ecresearch.uwaterloo.ca,share=cviss/Jack/baf/360streetview"
    json_path = os.path.join(base_dir, 'segmentation_training', 'synthetic', 'synthetic.json')
    images_dir = os.path.join(base_dir, 'segmentation_training', 'synthetic', 'images')
    output_dir = os.path.join(base_dir, 'segmentation_training', 'synthetic', 'mask_visualizations')
    num_images = 20  # Set how many images to visualize

    visualize_masks(json_path, images_dir, output_dir, num_images=num_images)