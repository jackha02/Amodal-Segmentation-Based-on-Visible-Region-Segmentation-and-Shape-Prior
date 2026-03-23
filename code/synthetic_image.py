import os
import cv2
import json
import random
import numpy as np
from copy import deepcopy
from shapely.affinity import scale, rotate as shapely_rotate, translate
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

def load_coco_json(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def poly_from_coco_seg(segmentation):
    if not segmentation: return None
    # COCO segmentation can have multiple lists if it's a disjoint object
    polys = []
    for coords in segmentation:
        points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
        if len(points) >= 3:
            poly = Polygon(points)
            if not poly.is_valid: poly = poly.buffer(0)
            polys.append(poly)
            
    if not polys: return None
    if len(polys) == 1: return polys[0]
    return MultiPolygon(polys)

def coco_seg_from_poly(poly):
    """Updated to correctly map MultiPolygons to COCO's list-of-lists format."""
    if poly.is_empty: return []
    
    if poly.geom_type == 'Polygon':
        polys = [poly]
    elif poly.geom_type == 'MultiPolygon':
        polys = poly.geoms
    else:
        return []

    segmentation = []
    for p in polys:
        coords = []
        for x, y in p.exterior.coords:
            coords.extend([x, y])
        if len(coords) >= 6: # Ensure at least 3 points
            segmentation.append(coords)
    return segmentation

def get_clean_inlets(coco_data, threshold=0.20, inlet_cat_id=1, debris_cat_id=2):
    """Extracts clean background inlets and their associated background debris."""
    images = {img['id']: img for img in coco_data['images']}
    inlets_by_image = {}
    debris_by_image = {}
    
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        poly = poly_from_coco_seg(ann['segmentation'])
        if not poly: continue
            
        if ann['category_id'] == inlet_cat_id:
            inlets_by_image.setdefault(img_id, []).append((ann, poly))
        elif ann['category_id'] == debris_cat_id:
            debris_by_image.setdefault(img_id, []).append((ann, poly))
                
    clean_inlets = []
    for img_id, inlets in inlets_by_image.items():
        img_debris = debris_by_image.get(img_id, [])
        img_debris_polys = [p for _, p in img_debris]
        
        for ann, inlet_poly in inlets:
            total_intersection_area = sum([inlet_poly.intersection(dp).area for dp in img_debris_polys])
            if total_intersection_area / inlet_poly.area < threshold:
                clean_inlets.append({
                    'image_info': images[img_id],
                    'inlet_ann': ann,
                    'inlet_poly': inlet_poly,
                    'bg_debris_polys': img_debris_polys # Save background debris for unioning later
                })
                
    return clean_inlets, images

def get_debris_sources(coco_data, inlet_cat_id=1, debris_cat_id=2):
    images = {img['id']: img for img in coco_data['images']}
    inlets_by_image = {}
    debris_by_image = {}
    valid_debris = []
    
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        poly = poly_from_coco_seg(ann['segmentation'])
        if not poly: continue
            
        if ann['category_id'] == inlet_cat_id:
            inlets_by_image.setdefault(img_id, []).append((ann, poly))
        elif ann['category_id'] == debris_cat_id:
            debris_by_image.setdefault(img_id, []).append((ann, poly))
            
    for img_id, debris_list in debris_by_image.items():
        inlets = inlets_by_image.get(img_id, [])
        if inlets:
            ref_inlet = max([p for _, p in inlets], key=lambda x: x.area)
            for ann, poly in debris_list:
                valid_debris.append((img_id, ann, poly, ref_inlet))
                
    return valid_debris, images

def get_warp_params(src_inlet, dst_inlet):
    def get_top_edge_angle(poly):
        # Extract coordinates and find the 4 corners of the bounding box
        coords = np.array(poly.exterior.coords, dtype=np.float32)
        rect = cv2.minAreaRect(coords)
        box = cv2.boxPoints(rect)
        
        # Sort the 4 corners by their Y-coordinate (lowest Y is the top of the image)
        box = box[np.argsort(box[:, 1])]
        
        # The top edge is defined by the two points with the smallest Y
        pt1, pt2 = box[0], box[1]
        
        # Enforce a Left-to-Right vector so the angle calculation is consistent
        if pt1[0] > pt2[0]:
            pt1, pt2 = pt2, pt1
            
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        
        # arctan2 natively returns the angle between -180 and 180 degrees
        return np.degrees(np.arctan2(dy, dx))

    # Calculate the exact angle of the top curb-line for both inlets
    src_angle = get_top_edge_angle(src_inlet)
    dst_angle = get_top_edge_angle(dst_inlet)
    
    # The required rotation is simply the difference between the two
    base_angle = dst_angle - src_angle
    
    # Calculate scale factor based on total surface area
    scale_factor = np.sqrt(dst_inlet.area / src_inlet.area)
    
    return scale_factor, base_angle

def iterative_debris_placement(debris_poly, src_inlet, dst_inlet, cat_min, cat_max, max_attempts=15):
    warp_scale, base_angle = get_warp_params(src_inlet, dst_inlet)
    minx, miny, maxx, maxy = dst_inlet.bounds
    
    scaled_debris = scale(debris_poly, xfact=warp_scale, yfact=warp_scale, origin='centroid')
    warped_debris = shapely_rotate(scaled_debris, base_angle, origin='centroid')
    
    for _ in range(max_attempts):
        # Only calculate new random drop coordinates
        drop_x = random.uniform(minx, maxx)
        drop_y = random.uniform(miny, maxy)
        dx = drop_x - warped_debris.centroid.x
        dy = drop_y - warped_debris.centroid.y
        
        placed_debris = translate(warped_debris, xoff=dx, yoff=dy)
        
        intersection_area = placed_debris.intersection(dst_inlet).area
        clogging_ratio = intersection_area / dst_inlet.area
        
        if cat_min <= clogging_ratio <= cat_max:
            return placed_debris, base_angle, warp_scale, dx, dy
        
    return None, None, None, None, None

def copy_and_paste_debris(dst_img, src_img, orig_debris_poly, placed_poly, total_angle, scale_factor, dx, dy):
    dst_h, dst_w = dst_img.shape[:2]
    src_h, src_w = src_img.shape[:2]
    
    # 1. Enforce Spillover Limit
    img_poly = Polygon([(0, 0), (dst_w, 0), (dst_w, dst_h), (0, dst_h)])
    final_debris_poly = placed_poly.intersection(img_poly)
    
    if final_debris_poly.is_empty: return None, None
    if final_debris_poly.geom_type == 'MultiPolygon':
        final_debris_poly = max(final_debris_poly.geoms, key=lambda a: a.area)

    # 2. Extract bounding box with padding
    minx, miny, maxx, maxy = orig_debris_poly.bounds
    pad = int(max(maxx-minx, maxy-miny) * scale_factor) 
    sx1, sy1 = max(0, int(minx)-pad), max(0, int(miny)-pad)
    sx2, sy2 = min(src_w, int(maxx)+pad), min(src_h, int(maxy)+pad)
    
    src_crop = src_img[sy1:sy2, sx1:sx2]
    if src_crop.size == 0: return None, None

    # 3. Create the strict irregular mask (The "Cookie-Cutter")
    local_mask = np.zeros(src_crop.shape[:2], dtype=np.uint8)
    local_coords = np.array(orig_debris_poly.exterior.coords) - np.array([sx1, sy1])
    cv2.fillPoly(local_mask, [np.int32(local_coords)], 255)
    
    # Isolate debris, drop source pavement
    src_crop_clean = cv2.bitwise_and(src_crop, src_crop, mask=local_mask)
    
    # 4. Calculate Affine Transform Matrix (Canvas Expansion)
    cx = orig_debris_poly.centroid.x - sx1
    cy = orig_debris_poly.centroid.y - sy1
    
    M = cv2.getRotationMatrix2D((cx, cy), total_angle, scale_factor) 
    cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
    new_w = int((src_crop.shape[1] * cos) + (src_crop.shape[0] * sin))
    new_h = int((src_crop.shape[1] * sin) + (src_crop.shape[0] * cos))
    M[0, 2] += (new_w / 2) - cx
    M[1, 2] += (new_h / 2) - cy
    
    warped_crop = cv2.warpAffine(src_crop_clean, M, (new_w, new_h), flags=cv2.INTER_LINEAR)
    warped_mask = cv2.warpAffine(local_mask, M, (new_w, new_h), flags=cv2.INTER_NEAREST)
    
    # 5. Construct Full-Size Layers
    full_src = np.zeros_like(dst_img)
    full_mask = np.zeros((dst_h, dst_w), dtype=np.uint8)
    
    dest_cx = int(orig_debris_poly.centroid.x + dx)
    dest_cy = int(orig_debris_poly.centroid.y + dy)
    
    start_y, start_x = dest_cy - new_h // 2, dest_cx - new_w // 2
    end_y, end_x = start_y + new_h, start_x + new_w
    
    crop_sy1, crop_sy2, crop_sx1, crop_sx2 = 0, new_h, 0, new_w
    if start_y < 0: crop_sy1 -= start_y; start_y = 0
    if start_x < 0: crop_sx1 -= start_x; start_x = 0
    if end_y > dst_h: crop_sy2 -= (end_y - dst_h); end_y = dst_h
    if end_x > dst_w: crop_sx2 -= (end_x - dst_w); end_x = dst_w
        
    if start_y >= end_y or start_x >= end_x: return None, None
    
    full_src[start_y:end_y, start_x:end_x] = warped_crop[crop_sy1:crop_sy2, crop_sx1:crop_sx2]
    full_mask[start_y:end_y, start_x:end_x] = warped_mask[crop_sy1:crop_sy2, crop_sx1:crop_sx2]
    
    # 6. Simple Copy-Paste (Pure Alpha Blending)
    # No Poisson, no Gaussian blurring. Just a direct pixel overwrite based on the binary mask.
    alpha = (full_mask / 255.0)[..., np.newaxis]
    dst_img_copy = dst_img.copy()
    dst_img_copy = (alpha * full_src + (1 - alpha) * dst_img_copy).astype(np.uint8)
        
    return dst_img_copy, final_debris_poly

def create_color_mask(h, w, inlet_poly, debris_poly, output_path):
    """Draws a standard visualization mask: Black BG, White Inlet, Red Debris."""
    # Initialize black background (OpenCV uses BGR)
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Draw Inlet in White (255, 255, 255)
    if inlet_poly and not inlet_poly.is_empty:
        polys = inlet_poly.geoms if inlet_poly.geom_type == 'MultiPolygon' else [inlet_poly]
        for p in polys:
            coords = np.array(p.exterior.coords, dtype=np.int32)
            cv2.fillPoly(mask, [coords], (255, 255, 255))
            
    # Draw Debris in Red (BGR: 0, 0, 255)
    if debris_poly and not debris_poly.is_empty:
        polys = debris_poly.geoms if debris_poly.geom_type == 'MultiPolygon' else [debris_poly]
        for p in polys:
            coords = np.array(p.exterior.coords, dtype=np.int32)
            cv2.fillPoly(mask, [coords], (0, 0, 255))
            
    cv2.imwrite(output_path, mask)

def main(target_images):
    base_dir = "/run/user/1000/gvfs/smb-share:server=ecresearch.uwaterloo.ca,share=cviss/Jack/baf/360streetview/segmentation_training/synthetic/copy_paste"
    
    # Background (Clean Inlets) Paths
    bg_images_dir = os.path.join(base_dir, 'clean', 'images')
    bg_json_path = os.path.join(base_dir, 'clean', 'clean.json')
    
    # Debris Mask Paths (UPDATE THESE TO YOUR ACTUAL DIRECTORIES)
    debris_images_dir = os.path.join(base_dir, 'debris_mask', 'images')
    debris_json_path = os.path.join(base_dir, 'debris_mask', 'debris.json')
    
    # Output Paths
    output_images_dir = os.path.join(base_dir, 'dataset', 'images')
    output_json_path = os.path.join(base_dir, 'dataset', 'synthetic.json')
    
    # New Mask Output Directories
    output_orig_masks_dir = os.path.join(base_dir, 'synthetic_masks', 'masks_original')
    output_synth_masks_dir = os.path.join(base_dir, 'synthetic_masks', 'masks_synthetic')
    
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    os.makedirs(output_orig_masks_dir, exist_ok=True)
    os.makedirs(output_synth_masks_dir, exist_ok=True)

    bg_coco_data = load_coco_json(bg_json_path)
    clean_inlets, bg_images = get_clean_inlets(bg_coco_data, threshold=0.20)
    
    debris_coco_data = load_coco_json(debris_json_path)
    valid_debris_sources, debris_images = get_debris_sources(debris_coco_data)

    new_coco = deepcopy(bg_coco_data)
    new_coco['images'], new_coco['annotations'] = [], []
    
    img_id_counter = max(bg_images.keys()) + 1
    ann_id_counter = max([ann['id'] for ann in bg_coco_data['annotations']]) + 1
    
    categories = [(0.2, 0.4), (0.4, 0.6), (0.6, 0.8)]
    target_extents_queue = []
    samples_per_cat = target_images // len(categories)
    for cat in categories: target_extents_queue.extend([cat] * samples_per_cat)
    while len(target_extents_queue) < target_images: target_extents_queue.append(categories[-1])
    random.shuffle(target_extents_queue)

    while target_extents_queue:
        current_cat = target_extents_queue.pop(0)
        
        inlet_data = random.choice(clean_inlets)
        dst_img_info, dst_inlet_poly = inlet_data['image_info'], inlet_data['inlet_poly']
        bg_debris_polys = inlet_data['bg_debris_polys'] # Retrieve background debris
        
        src_img_id, _, orig_debris_poly, src_inlet_poly = random.choice(valid_debris_sources)
        
        placed_debris, total_angle, scale_factor, dx, dy = iterative_debris_placement(
            orig_debris_poly, src_inlet_poly, dst_inlet_poly, current_cat[0], current_cat[1]
        )
        
        if not placed_debris:
            target_extents_queue.append(current_cat)
            continue

        dst_img = cv2.imread(os.path.join(bg_images_dir, dst_img_info['file_name']))
        src_img = cv2.imread(os.path.join(debris_images_dir, debris_images[src_img_id]['file_name']))
        
        if dst_img is None or src_img is None:
            target_extents_queue.append(current_cat)
            continue
            
        synthetic_img, final_debris_poly = copy_and_paste_debris(
            dst_img, src_img, orig_debris_poly, placed_debris, total_angle, scale_factor, dx, dy
        )
        
        if synthetic_img is None:
            target_extents_queue.append(current_cat)
            continue
            
        # Merge newly pasted debris with existing background debris
        all_debris_polys = [final_debris_poly] + bg_debris_polys
        merged_debris_poly = unary_union(all_debris_polys).buffer(0) # Buffer(0) cleans up invalid geometries post-union
            
        # --- File Saving & Export ---
        img_h, img_w = dst_img_info['height'], dst_img_info['width']
        base_name = f"synthetic_{img_id_counter}"
        
        # 1. Save Synthetic Image
        cv2.imwrite(os.path.join(output_images_dir, f"{base_name}.png"), synthetic_img)
        
        # 2. Save Original Mask (Inlet + Original Background Debris)
        orig_bg_union = unary_union(bg_debris_polys).buffer(0) if bg_debris_polys else None
        create_color_mask(img_h, img_w, dst_inlet_poly, orig_bg_union, 
                          os.path.join(output_orig_masks_dir, f"{base_name}_orig_mask.png"))
        
        # 3. Save Synthetic Mask (Inlet + Merged Debris)
        create_color_mask(img_h, img_w, dst_inlet_poly, merged_debris_poly, 
                          os.path.join(output_synth_masks_dir, f"{base_name}_synth_mask.png"))

        # --- Update COCO Data ---
        new_img_id = img_id_counter
        new_coco["images"].append({
            "id": new_img_id, "width": img_w, "height": img_h,
            "file_name": f"{base_name}.png", "license": 0, "date_captured": 0
        })
        img_id_counter += 1
        
        inlet_bbox = dst_inlet_poly.bounds
        new_coco["annotations"].append({
            "id": ann_id_counter, "iscrowd": 0, "image_id": new_img_id, "category_id": 1, 
            "segmentation": coco_seg_from_poly(dst_inlet_poly), "bbox": [inlet_bbox[0], inlet_bbox[1], inlet_bbox[2]-inlet_bbox[0], inlet_bbox[3]-inlet_bbox[1]],
            "area": dst_inlet_poly.area, "attributes": {"occluded": True}
        })
        ann_id_counter += 1
        
        if not merged_debris_poly.is_empty:
            debris_bbox = merged_debris_poly.bounds
            new_coco["annotations"].append({
                "id": ann_id_counter, "iscrowd": 0, "image_id": new_img_id, "category_id": 2, 
                "segmentation": coco_seg_from_poly(merged_debris_poly), "bbox": [debris_bbox[0], debris_bbox[1], debris_bbox[2]-debris_bbox[0], debris_bbox[3]-debris_bbox[1]],
                "area": merged_debris_poly.area, "attributes": {"occluded": False}
            })
            ann_id_counter += 1

    with open(output_json_path, 'w') as f:
        json.dump(new_coco, f)
        
    print(f"Successfully generated {target_images} synthetic images and their associated ground-truth masks.")

if __name__ == "__main__":
    main(target_images=500)





    