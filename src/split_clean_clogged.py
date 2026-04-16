import os
import json
import shutil
from tqdm import tqdm
import pickle
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

def load_coco_json(json_path):
    """
    Load a COCO JSON file and return it as a dictionary
    """
    with open(json_path, 'r') as f:
        return json.load(f)

def poly_from_coco_seg(segmentation):
    """
    Build a Shapely geometry from a COCO segmentation field
    :param segmentation: List of flat coordinate lists [[x1,y1,x2,y2,...], ...]
    :returns: Shapely Polygon / MultiPolygon, or None if input is empty/invalid
    """
    if not segmentation: 
        return None

    polys = []
    for coords in segmentation:
        points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
        if len(points) >= 3:
            poly = Polygon(points)
            if poly.is_valid and not poly.is_empty:
                polys.append(poly)

    if not polys: 
        return None
    if len(polys) == 1: 
        return polys[0]
    return MultiPolygon(polys)

def clogging_ratio(inlet_poly, debris_poly):
    """
    Returns the fraction [0, 1] of the inlet area covered by debris
    :param inlet_poly: polygon of inlet
    :param debris_poly: polygon of debris (MultiPolygon for multiple debris in an image)
    """
    if debris_poly is None or debris_poly.is_empty:
        return 0.0
    return inlet_poly.intersection(debris_poly).area / inlet_poly.area

def split_clean_clogged(coco_data, clean_dir, source_images_dir, clean_threshold=0.10):
    """
    Split a COCO dataset into clean and clogged subsets based on the pre-computed
    clogging_extent field produced by coco_json_conversion.py.
    :param coco_data: Loaded COCO JSON dict (output of coco_json_conversion.py)
    :param clean_dir: Root output dir for clean split (images/ + clean.json written here)
    :param source_images_dir: Directory containing the original image files
    :param clean_threshold: clogging_extent threshold; >= this value means clogged, else clean
    Saves the clean images, clean.json, and clean_inlets_data.pkl inside clean_dir
    Also saves the clogged images, clogged.json, and clogged_inlets_data.pkl inside clogged_dir
    """
    # Ensure clogged_dir is in the same parent as clean_dir
    parent_dir = os.path.abspath(os.path.join(clean_dir, os.pardir))
    clogged_dir = os.path.join(parent_dir, 'clogged_inlets')
    clean_images_dir = os.path.join(clean_dir, 'images')
    clogged_images_dir = os.path.join(clogged_dir, 'images')
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(clean_images_dir, exist_ok=True)
    os.makedirs(clogged_dir, exist_ok=True)
    os.makedirs(clogged_images_dir, exist_ok=True)

    clean_coco   = {**coco_data, 'images': [], 'annotations': []}
    clogged_coco = {**coco_data, 'images': [], 'annotations': []}

    # Group the original annotations by image id for faster lookups
    anns_by_image = {}
    for ann in coco_data.get('annotations', []):
        anns_by_image.setdefault(ann['image_id'], []).append(ann)

    # Track the count for clean and clogged inlets
    clean_count = 0
    clogged_count = 0

    # Save data for clean and clogged inlets separately for return at the end
    clean_inlets_data = []
    clogged_inlets_data = []

    # Process each image id
    for img_info in tqdm(coco_data.get('images', []), desc="Splitting clean and clogged inlet images"):
        img_id = img_info['id']
        img_anns = anns_by_image.get(img_id, [])
        
        # Parse all storm_drain annotations in this image.
        # clogging_extent is pre-computed by coco_json_conversion.py — no debris
        # polygon arithmetic needed.
        inlets = []
        for ann in img_anns:
            inlet_poly   = poly_from_coco_seg(ann.get('segmentation'))
            visible_poly = poly_from_coco_seg(ann.get('visible_segmentation'))
            if inlet_poly is None:
                continue
            # Visible polygon falls back to amodal polygon for fully unobstructed drains
            if visible_poly is None:
                visible_poly = inlet_poly
            inlets.append((ann, inlet_poly, visible_poly))

        clogging_check = False
        temp_dicts = []


        # For single inlets
        if len(inlets) == 1:
            ann, inlet_poly, visible_poly = inlets[0]
            extent = float(ann.get('clogging_extent', 0.0))

            data_dict = {
                'image_info':      img_info,
                'inlet_ann':       ann,
                'inlet_poly':      inlet_poly,
                'visible_poly':    visible_poly,
                'clogging_extent': extent,
            }

            if extent >= clean_threshold:
                clogging_check = True
                clogged_inlets_data.append(data_dict)
            else:
                clean_inlets_data.append(data_dict)

        # For double inlets
        elif len(inlets) > 1:
            for ann, inlet_poly, visible_poly in inlets:
                extent = float(ann.get('clogging_extent', 0.0))
                temp_dicts.append({
                    'image_info':      img_info,
                    'inlet_ann':       ann,
                    'inlet_poly':      inlet_poly,
                    'visible_poly':    visible_poly,
                    'clogging_extent': extent,
                })

            # Image is considered fully clogged only if every inlet exceeds the threshold
            clogging_check = all(d['clogging_extent'] >= clean_threshold for d in temp_dicts)
            if clogging_check:
                clogged_inlets_data.extend(temp_dicts)
            else:
                clean_inlets_data.extend(temp_dicts)

        if clogging_check:
            clogged_count += 1
            # Copy image to the clogged images directory
            src_path = os.path.join(source_images_dir, img_info['file_name'])
            dest_path = os.path.join(clogged_images_dir, os.path.basename(img_info['file_name']))
            if os.path.exists(src_path):
                shutil.copyfile(src_path, dest_path)
            # Accumulate clogged image and its annotations into clogged_coco
            clogged_coco['images'].append(img_info)
            clogged_coco['annotations'].extend(img_anns)
        else:
            clean_count += 1
            # Copy image to the clean images directory
            src_path = os.path.join(source_images_dir, img_info['file_name'])
            dest_path = os.path.join(clean_images_dir, os.path.basename(img_info['file_name']))
            if os.path.exists(src_path):
                shutil.copyfile(src_path, dest_path)
            # Accumulate clean image and its annotations into clean_coco
            clean_coco['images'].append(img_info)
            clean_coco['annotations'].extend(img_anns)


    # Save clean split
    with open(os.path.join(clean_dir, 'clean.json'), 'w') as f:
        json.dump(clean_coco, f)

    with open(os.path.join(clean_dir, 'clean_inlets_data.pkl'), 'wb') as f:
        pickle.dump(clean_inlets_data, f)

    # Save clogged split
    with open(os.path.join(clogged_dir, 'clogged.json'), 'w') as f:
        json.dump(clogged_coco, f)

    with open(os.path.join(clogged_dir, 'clogged_inlets_data.pkl'), 'wb') as f:
        pickle.dump(clogged_inlets_data, f)

    print(f"Split complete — Clean images saved: {clean_count}, Clogged images saved: {clogged_count}")

if __name__ == "__main__":
    base_dir = "/run/user/1000/gvfs/smb-share:server=ecresearch.uwaterloo.ca,share=cviss/Jack/baf/360streetview/segmentation_training"    
    json_path         = os.path.join(base_dir, 'original', 'dataset', 'custom.json')   # Path to the merged COCO annotation JSON (annotations_all.json)
    source_images_dir = os.path.join(base_dir, 'original', 'dataset', 'images')   # Directory containing the original inlet images
    clean_dir         = os.path.join(base_dir, 'clean_inlets')   # Directory to save clean inlet data and pickle

    # Identify and save clean and clogged inlet images
    coco_data = load_coco_json(json_path)
    split_clean_clogged(
        coco_data,
        clean_dir,
        source_images_dir,
        clean_threshold=0.2,
    )

    