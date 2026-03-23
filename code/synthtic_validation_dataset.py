import os 
import json
import shutil
from shapely.geometry import Polygon, MultiPolygon

# Read the json file
def load_coco_json(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def poly_from_coco_seg(segmentation):
    """Converts a COCO segmentation array into a Shapely Polygon."""
    if not segmentation:
        return None
    
    # In standard COCO, segmentation can have multiple parts if occluded
    # For simplicity in intersection math, we'll take the primary outer boundary
    coords = segmentation[0] 
    points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
    
    if len(points) < 3:
        return None
        
    poly = Polygon(points)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly

def coco_seg_from_poly(poly):
    """
    Converts a Shapely Polygon or MultiPolygon back to COCO segmentation format.
    Returns a list of coordinate lists.
    """
    if poly.is_empty:
        return []
        
    segmentations = []
    
    # Handle both single Polygons and MultiPolygons (which occur during complex intersections)
    polys = poly.geoms if isinstance(poly, MultiPolygon) else [poly]
    
    for p in polys:
        # COCO expects flattened lists: [x1, y1, x2, y2, ...]
        coords = []
        for x, y in p.exterior.coords:
            coords.extend([x, y])
        segmentations.append(coords)
        
    return segmentations

def get_bbox_from_poly(poly):
    """Calculates the COCO bounding box [x_min, y_min, width, height] from a Shapely geometry."""
    minx, miny, maxx, maxy = poly.bounds
    return [minx, miny, maxx - minx, maxy - miny]

def distill_dataset(input_json_path, output_json_path, threshold=0.10, inlet_cat_id=1, debris_cat_id=2):
    """
    Filters the dataset to only include clogged inlets (>= threshold) and 
    crops debris to the exact intersection of the inlet.
    """
    print("Loading original dataset...")
    coco_data = load_coco_json(input_json_path)
    
    images_dict = {img['id']: img for img in coco_data['images']}
    annotations_by_image = {}
    
    # Group annotations by image
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        annotations_by_image.setdefault(img_id, []).append(ann)

    new_images = []
    new_annotations = []
    ann_id_counter = 1 # Re-indexing annotations from 1 for a clean JSON
    
    print("Processing geometries and filtering...")
    
    for img_id, img_info in images_dict.items():
        anns = annotations_by_image.get(img_id, [])
        
        # Separate inlets and debris, and convert to Shapely polygons
        inlets = []
        debris_list = []
        
        for ann in anns:
            poly = poly_from_coco_seg(ann['segmentation'])
            if not poly:
                continue
            
            if ann['category_id'] == inlet_cat_id:
                inlets.append((ann, poly))
            elif ann['category_id'] == debris_cat_id:
                debris_list.append((ann, poly))

        image_has_valid_data = False
        
        for inlet_ann, inlet_poly in inlets:
            inlet_area = inlet_poly.area
            total_intersection_area = 0
            intersecting_debris_polys = []
            
            # Check interactions with all debris in the image
            for debris_ann, debris_poly in debris_list:
                intersection = inlet_poly.intersection(debris_poly)
                
                if not intersection.is_empty and intersection.area > 0:
                    total_intersection_area += intersection.area
                    intersecting_debris_polys.append(intersection)
            
            # Check if this inlet meets the clogging threshold
            if total_intersection_area / inlet_area >= threshold:
                image_has_valid_data = True
                
                # 1. Save the clogged inlet annotation
                new_inlet_ann = inlet_ann.copy()
                new_inlet_ann['id'] = ann_id_counter
                
                # Ensure the occluded attribute exists and is flagged
                if 'attributes' not in new_inlet_ann:
                    new_inlet_ann['attributes'] = {}
                new_inlet_ann['attributes']['occluded'] = True
                
                new_annotations.append(new_inlet_ann)
                ann_id_counter += 1
                
                # 2. Save the cropped debris annotations
                for intersection_poly in intersecting_debris_polys:
                    cropped_seg = coco_seg_from_poly(intersection_poly)
                    cropped_bbox = get_bbox_from_poly(intersection_poly)
                    cropped_area = intersection_poly.area
                    
                    new_debris_ann = {
                        "id": ann_id_counter,
                        "image_id": img_id,
                        "category_id": debris_cat_id,
                        "segmentation": cropped_seg,
                        "area": cropped_area,
                        "bbox": cropped_bbox,
                        "iscrowd": 0,
                        "attributes": {"occluded": False}
                    }
                    new_annotations.append(new_debris_ann)
                    ann_id_counter += 1

        # Only add the image to the new JSON if it contains at least one clogged inlet
        if image_has_valid_data:
            new_images.append(img_info)

    # Reconstruct the final COCO dictionary
    new_coco = {
        "info": coco_data.get("info", {}),
        "licenses": coco_data.get("licenses", []),
        "images": new_images,
        "annotations": new_annotations,
        "categories": coco_data.get("categories", [])
    }
    
    # Save to disk
    print(f"Saving distilled dataset to {output_json_path}...")
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(new_coco, f)
        
    print("--- Processing Complete ---")
    print(f"Original images: {len(coco_data['images'])} | Distilled images kept: {len(new_images)}")
    print(f"Original annotations: {len(coco_data['annotations'])} | Distilled annotations kept: {len(new_annotations)}")

def matching_files(clogged_json_data, img_folder_path, output_path):
    file_name = {img['file_name']: img for img in clogged_json_data['images']}
    for img in os.listdir(img_folder_path):
        img_path = os.path.join(img_folder_path, img)
        if img in file_name:
            shutil.copyfile(img_path, os.path.join(output_path, img))

if __name__ == "__main__":
    base_dir = "/run/user/1000/gvfs/smb-share:server=ecresearch.uwaterloo.ca,share=cviss/Jack/baf/360streetview/segmentation_training/original/dataset"
    input_json = os.path.join(base_dir, "original.json")
    output_json = os.path.join(base_dir, "test", "clogged.json")
    img_folder = os.path.join(base_dir, "images")
    output_folder = os.path.join(base_dir, "test", "images") 

    # New set of images of clogged inlets
    read_input_json = load_coco_json(input_json)

    distill_dataset(
        input_json_path=input_json, 
        output_json_path=output_json, 
        threshold=0.10  # Keeps inlets with >= 10% occlusion
    )

    read_output_json = load_coco_json(output_json)
    matching_files(read_output_json, img_folder, output_folder)