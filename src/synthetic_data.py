import random
import os
import tqdm
import cv2
import pickle
import numpy as np
import json
import math
import matplotlib.pyplot as plt
import shutil
from shapely.geometry import Polygon, MultiPolygon, LineString, Point, box
from shapely.affinity import rotate as shapely_rotate, scale as shapely_scale, translate as shapely_translate
from shapely.ops import unary_union
from split_clean_clogged import load_coco_json, poly_from_coco_seg, clogging_ratio

# ─────────────────────────────────────────────────────────────────────────────
# To generate synthetic clogged inlet images by copy-pasting debris onto clean inlets
# The ultimate goal of this procedure is improve the performance of the amodal segmentation mode
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Step 1. Define the geometric constraints of where debris can be pasted within the inlet patch
# To make the pasted debris look realistic, we must first find where the inlet and concrete curb ares so that debris doesn't float everywhere in the image
# ─────────────────────────────────────────────────────────────────────────────

def get_inlet_points(ann):
    """
    Extract the four corner points of an inlet annotation in order

    :param ann: COCO annotation dict with a flat-coordinate segmentation.
    :returns: List of (x, y) tuples, length 4 (or more for non-quads).
    """
    seg = ann['segmentation']
    if not seg:
        return []
    coords = seg[0]
    return [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]

def get_valid_paste_region(ann, img_w=640, img_h=640):
    """
    Build a Shapely Polygon representing the valid region in which debris may be pasted.

    The valid region is the quadrilateral whose four corners are formed by extending
    the concrete curb edge (pts[1]–pts[2]) and the opposite inlet edge (pts[0]–pts[3])
    to the full image width, then taking the four intersection points with x=0 and x=img_w.
    The resulting quad is clipped to the image bounding box.

    Annotation convention (must be followed exactly):
      - pts[1] and pts[2] are the two points on the concrete curb edge.
      - pts[0] and pts[3] are the two points on the opposite (road-side) edge.
      - Inlets must be annotated with exactly 4 points.

    :param ann:   COCO annotation dict for an inlet.
    :param img_w: Image width in pixels (640 by default)
    :param img_h: Image height in pixels (640 by default)
    :returns:     Shapely Polygon of the valid paste region, clipped to image bounds.
    """
    pts = get_inlet_points(ann)
    if len(pts) != 4:
        raise ValueError(f"Inlet annotation id={ann.get('id')} has {len(pts)} points. Expected exactly 4.")

    curb_p1, curb_p2 = pts[1], pts[2]   # concrete curb edge
    opp_p1,  opp_p2  = pts[0], pts[3]   # opposite (road-side) edge

    def x_intercepts(p1, p2):
        """
        Return (left_pt, right_pt) where the line through p1,p2 crosses x=0 and x=img_w
        """
        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])
        dx = x2 - x1
        if abs(dx) < 1e-6:   
            return (x1, 0.0), (x1, float(img_h))
        slope = (y2 - y1) / dx
        y_left  = y1 + slope * (0      - x1)
        y_right = y1 + slope * (img_w  - x1)
        return (0.0, y_left), (float(img_w), y_right)

    curb_left, curb_right = x_intercepts(curb_p1, curb_p2)
    opp_left,  opp_right  = x_intercepts(opp_p1,  opp_p2)

    # Four corners of the valid pastable region
    band = Polygon([curb_left, curb_right, opp_right, opp_left])
    image_box = box(0, 0, img_w, img_h)
    return band.intersection(image_box)

def crop_patch_to_valid_region(patch_img, patch_mask, offset_x, offset_y, valid_region):
    """
    Zero out any pixel in the patch that falls outside the valid paste region
    CRITICAL: Returns both cropped pixels and a mask representing what was actually kept

    The valid region is a pre-built Shapely Polygon (from get_valid_paste_region).
    Each pixel's position in the full image is computed from its local patch index
    plus the patch offset, then tested for containment by rasterising the valid
    region into a temporary mask.

    :param patch_img:    (H, W, 3) uint8 patch image.
    :param patch_mask:   (H, W)    uint8 debris mask (255 inside debris polygon).
    :param offset_x:     X coordinate of the patch top-left corner in the full image.
    :param offset_y:     Y coordinate of the patch top-left corner in the full image.
    :param valid_region: Shapely Polygon defining where debris may be pasted.
    Returns (patch_img, patch_mask, kept_mask) where kept_mask indicates which pixels were retained
    """
    h, w = patch_mask.shape

    # Translate the valid region into patch-local coordinates
    local_region = shapely_translate(valid_region, xoff=-offset_x, yoff=-offset_y)

    # Rasterise the local valid region into a boolean mask the same size as the patch
    if local_region.is_empty:
        return np.zeros_like(patch_img), np.zeros_like(patch_mask), np.zeros((h, w), dtype=np.uint8)

    # Extract exterior ring coords and rasterise with fillPoly
    coords = np.array(local_region.exterior.coords, dtype=np.float32)
    pts_cv = coords.reshape((-1, 1, 2)).astype(np.int32)
    valid_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(valid_mask, [pts_cv], 255)

    # Handle potential interior holes (rare, but possible after image-box clipping)
    for interior in local_region.interiors:
        hole_coords = np.array(interior.coords, dtype=np.float32)
        hole_pts = hole_coords.reshape((-1, 1, 2)).astype(np.int32)
        cv2.fillPoly(valid_mask, [hole_pts], 0)

    invalid = valid_mask == 0

    patch_img  = patch_img.copy()
    patch_mask = patch_mask.copy()
    patch_img[invalid]  = 0
    patch_mask[invalid] = 0
    
    kept_mask = valid_mask.copy()

    return patch_img, patch_mask, kept_mask

def extend_line_to_image(p1, p2, img_w, img_h):
    """
    Extend the selected inlet edges to the full-width image boundary
    This is used to create a hard "upper" boundary at the concrete curb line
    and a "lower" boundary at the opposite inlet edge, between which debris will be pasted 

    Finds intersection with x=0 and x=img_w, clipped to [0, img_h] in y

    :param p1: (x, y) first point on the line.
    :param p2: (x, y) second point on the line.
    :param img_w: Image width in pixels.
    :param img_h: Image height in pixels.
    :returns: ((x0, y0), (x1, y1)) two boundary points defining the extended line.
    """
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    dx = x2 - x1
    dy = y2 - y1

    if abs(dx) < 1e-6:
        # Vertical line
        return (x1, 0), (x1, img_h)

    slope = dy / dx

    # y = y1 + slope * (x - x1)
    y_at_0    = y1 + slope * (0   - x1)
    y_at_imgw = y1 + slope * (img_w - x1)

    return (0, y_at_0), (img_w, y_at_imgw)

# ─────────────────────────────────────────────────────────────────────────────
# Functions for debris patch extraction, transformation, and placement
# ─────────────────────────────────────────────────────────────────────────────

def load_debris_json(debris_json_path):
    """
    Load a debris-patch COCO JSON to build a list of debris annotation dicts.

    The JSON is expected to contain a single category named 'debris'. The
    category ID is resolved by name so the caller does not need to know or
    pass a numeric ID.

    :param debris_json_path: Path to the manually annotated debris JSON file.
    Returns (debris_by_image, img_info_by_id) where
              debris_by_image  is {image_id: [ann, ...]} for debris annotations,
              img_info_by_id   is {image_id: img_info_dict}.
    """
    data = load_coco_json(debris_json_path)
    img_info_by_id = {img['id']: img for img in data.get('images', [])}

    # Resolve the 'debris' category ID by name
    debris_cat_id = None
    for cat in data.get('categories', []):
        if cat['name'] == 'debris':
            debris_cat_id = cat['id']
            break
    if debris_cat_id is None:
        raise ValueError(
            f"No category named 'debris' found in '{debris_json_path}'. "
            f"Available categories: {[c['name'] for c in data.get('categories', [])]}"
        )

    debris_by_image = {}
    for ann in data.get('annotations', []):
        if ann['category_id'] == debris_cat_id:
            debris_by_image.setdefault(ann['image_id'], []).append(ann)
    return debris_by_image, img_info_by_id

def extract_debris_patch(src_img, debris_ann):
    """
    Cut a debris patch (image crop + binary mask) from a source image

    :param src_img: Full source image as (H, W, 3) uint8 ndarray.
    :param debris_ann: COCO annotation dict for one debris instance
    Returns (patch_bgr, mask, poly) where
        patch_bgr is (crop_h, crop_w, 3) uint8,
        mask is (crop_h, crop_w) uint8 (255 inside polygon),
        poly is the Shapely Polygon of the debris in source-image coordinates
    """
    poly = poly_from_coco_seg(debris_ann.get('segmentation'))
    if poly is None:
        return None, None, None

    # Bounding box of the polygon in the source image
    minx, miny, maxx, maxy = map(int, poly.bounds)
    minx = max(0, minx)
    miny = max(0, miny)
    maxx = min(src_img.shape[1], maxx + 1)
    maxy = min(src_img.shape[0], maxy + 1)

    patch_bgr = src_img[miny:maxy, minx:maxx].copy()
    h, w      = patch_bgr.shape[:2]

    # Rasterise the polygon into a local mask
    pts_local = np.array(
        [[[int(x - minx), int(y - miny)] for x, y in poly.exterior.coords]],
        dtype=np.int32,
    )
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, pts_local, 255)

    # Translate the debris polygon to match the local path coordinates (origin at top-left of the patch)
    poly = shapely_translate(poly, xoff=-minx, yoff=-miny)

    return patch_bgr, mask, poly

def transform_patch(patch_bgr, mask, poly,
                     angle_range=(-30, 30),
                     scale_range=(0.7, 1.3)):
    """
    Apply random rotation and scaling to a debris patch and its mask

    :param patch_bgr:    (H, W, 3) uint8 patch.
    :param mask:         (H, W) uint8 mask.
    :param poly:         Shapely Polygon of the debris (used for the returned transformed poly).
    :param angle_range:  (min_deg, max_deg) rotation range.
    :param scale_range:  (min_s, max_s) uniform scale range.
    
    Returns (t_patch, t_mask, t_poly, new_w, new_h)
    where t_patch is the transformed patch image,
        t_mask is the transformed mask,
        t_poly is the transformed Shapely Polygon in the patch-local coordinate system,
        new_w/new_h are the dimensions of the transformed patch
    """
    angle = random.uniform(*angle_range)
    scale = random.uniform(*scale_range)

    h, w = patch_bgr.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    # Build rotation+scale matrix
    M = cv2.getRotationMatrix2D((cx, cy), angle, scale)

    # Compute new canvas size for the debris patch after rotation
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)

    # Adjust translation so the full patch fits
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    t_patch = cv2.warpAffine(patch_bgr, M, (new_w, new_h))
    t_mask  = cv2.warpAffine(mask,       M, (new_w, new_h))
    # Binarize: warpAffine with INTER_LINEAR produces antialiased edge values (0-254).
    # These intermediate values corrupt downstream bitwise-OR mask accumulation and
    # the _ratio() polygon computation.  A hard threshold at 127 restores a clean binary mask.
    _, t_mask = cv2.threshold(t_mask, 127, 255, cv2.THRESH_BINARY)

    # Transform the shapely polygon (relative to patch origin = 0,0)
    t_poly = shapely_rotate(poly, -angle, origin=(cx, cy), use_radians=False)
    t_poly = shapely_scale(t_poly, xfact=scale, yfact=scale, origin=(cx, cy))

    # Apply the exact same canvas expansion shift to the polygon
    shift_x = (new_w - w) / 2
    shift_y = (new_h - h) / 2
    t_poly = shapely_translate(t_poly, xoff=shift_x, yoff=shift_y)

    return t_patch, t_mask, t_poly, new_w, new_h

# ─────────────────────────────────────────────────────────────────────────────
# Smart debris patch placement using a spatial grid and geometric constraints
# Crop patches at the curb line and opposite edge to maintain realism
# ─────────────────────────────────────────────────────────────────────────────

def build_candidate_grid(inlet_poly, grid_step=10):
    """
    Generate a grid of candidate centre points that lie inside the inlet polygon

    :param inlet_poly: Shapely Polygon of the inlet region
    :param grid_step:  Spacing between candidate points in pixels

    Returns a list of (x, y) integer candidate centres, shuffled randomly
    Random shuffling ensures that the debris patches are placed at a randomly selected grid point
    """
    minx, miny, maxx, maxy = map(int, inlet_poly.bounds)
    candidates = []
    for y in range(miny, maxy, grid_step):
        for x in range(minx, maxx, grid_step):
            pt = Point(x, y)
            if inlet_poly.contains(pt):
                candidates.append((x, y))
    random.shuffle(candidates)
    return candidates

def random_interior_point(poly, max_tries=50):
    """
    Sample a random point guaranteed to lie strictly inside a polygon
    Uses rejection sampling within the bounding box

    :param poly:      Shapely Polygon.
    :param max_tries: Maximum rejection-sampling attempts

    Returns a Shapely Point inside the polygon, or None if sampling fails
    """
    minx, miny, maxx, maxy = poly.bounds
    for _ in range(max_tries):
        pt = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if poly.contains(pt):
            return pt
    return poly.centroid if poly.contains(poly.centroid) else None

def place_patch_on_grid(
    t_patch, t_mask, t_poly, nw, nh,
    p_bgr, p_mask, p_poly,
    candidates,
    inlet_poly,
    forbidden,
    angle_range,
    scale_range,
    max_attempts=200):
    """
    Find a valid placement for a debris patch by aligning a random interior point of the debris polygon to a candidate grid point

    A placement is valid when the he aligned polygon does not intersect the forbidden zone 
    The forbidden zone includes existing background debris areas and previously placed patches
    Note that this forbidden zone expands as we place more patches within the image, ensuring no overlaps between them

    For each candidate, a fresh random rotation/scale is also tried up to max_attempts times so that a patch that fails in one orientation may
    succeed in another, rather than being skipped entirely

    :param t_patch/t_mask/t_poly/nw/nh: Initial transformed patch (pre-computed).
    :param p_bgr/p_mask/p_poly:         Original extracted patch for re-transforms.
    :param candidates:   List of (cx, cy) grid points inside the inlet (shuffled).
    :param inlet_poly:   Shapely Polygon of the inlet.
    :param forbidden:    Shapely geometry of already-occupied area (or None).
    :param angle_range:  Rotation range for re-tries.
    :param scale_range:  Scale range for re-tries.
    :param max_attempts: Total candidate+transform attempts before giving up.
    Returns (offset_x, offset_y, placed_poly, final_patch, final_mask) or (None, None, None, None, None) on failure after max_attempts
    """
    current_patch, current_mask, current_poly, cur_nw, cur_nh = (t_patch, t_mask, t_poly, nw, nh)

    attempt = 0
    cand_idx = 0

    while attempt < max_attempts:
        # Cycle through candidates; when exhausted, re-transform and restart
        if cand_idx >= len(candidates):
            current_patch, current_mask, current_poly, cur_nw, cur_nh = transform_patch(
                p_bgr, p_mask, p_poly, angle_range=angle_range, scale_range=scale_range
            )
            cand_idx = 0
            if not candidates:
                break
        
        # A random candidate point (cx, cy) from the grid to attempt placement
        cx, cy = candidates[cand_idx]
        cand_idx += 1
        attempt += 1

        # Pick a random interior point of the debris polygon and align it to the candidate point
        interior_pt = random_interior_point(current_poly)
        if interior_pt is None:
            continue

        dx = cx - interior_pt.x
        dy = cy - interior_pt.y

        offset_x    = int(round(dx))
        offset_y    = int(round(dy))

        placed_poly = shapely_translate(current_poly, xoff=offset_x, yoff=offset_y)

        # Ensure that the placed debris patch intersects with the inlet polygon
        if not placed_poly.intersects(inlet_poly):
            continue
        
        # The pasted debris patch can cover up to 90% of the inlet
        intersection_area = placed_poly.intersection(inlet_poly).area
        if intersection_area / inlet_poly.area > 0.9:
            continue

        # Enesure no overlap with forbidden zone
        if forbidden is not None and not forbidden.is_empty:
            if placed_poly.intersects(forbidden):
                continue


        return offset_x, offset_y, placed_poly, current_patch, current_mask

    return None, None, None, None, None

# ──────────────────────────────────────────────────────────────────────────────
# Apply Poisson blending to ensure the pasted debris patches match the lighting and pixel texture of the inlet surface
# ──────────────────────────────────────────────────────────────────────────────

def poisson_blend_patch(canvas, patch_bgr, mask, offset_x, offset_y):
    """
    Blend one debris patch into the canvas using Poisson blending

    :param canvas:    (H, W, 3) uint8 destination image (not modified in-place).
    :param patch_bgr: (ph, pw, 3) uint8 source patch in BGR colour space.
    :param mask:      (ph, pw) uint8 binary mask — 255 inside the debris polygon,
                      0 outside.
    :param offset_x:  X pixel coordinate of the patch's top-left corner in the
                      full canvas.
    :param offset_y:  Y pixel coordinate of the patch's top-left corner in the
                      full canvas.
    :returns: New (H, W, 3) uint8 canvas with the patch blended in.
    """
    img_h, img_w = canvas.shape[:2]
    ph, pw       = patch_bgr.shape[:2]

    # Clip patch region to canvas bounds
    x1 = max(0, offset_x);          y1 = max(0, offset_y)
    x2 = min(img_w, offset_x + pw); y2 = min(img_h, offset_y + ph)

    if x2 <= x1 or y2 <= y1:
        return canvas  # Patch entirely outside the canvas — nothing to do

    # Corresponding slice within the local patch arrays
    px1 = x1 - offset_x; py1 = y1 - offset_y
    px2 = x2 - offset_x; py2 = y2 - offset_y

    roi_patch = patch_bgr[py1:py2, px1:px2]
    roi_mask  = mask[py1:py2, px1:px2]

    # seamlessClone requires ≥1 non-zero mask pixel that is not on the ROI border.
    if cv2.countNonZero(roi_mask) < 9:
        print(f"EXIT EARLY: Mask is empty! Non-zero pixels = {cv2.countNonZero(roi_mask)}")
        return canvas

    roi_canvas = canvas[y1:y2, x1:x2].copy()

    # Compute the center of the mask to use as the center point for seamlessClone
    br_x, br_y, br_w, br_h = cv2.boundingRect(roi_mask)
    roi_centre = (br_x + br_w // 2, br_y + br_h // 2)

    try:
        blended_roi = cv2.seamlessClone(
            roi_patch,
            roi_canvas,
            roi_mask,
            roi_centre,
            cv2.NORMAL_CLONE,   # smooth gradient propagation from source
        )

    except cv2.error:
        # Poisson blending failed for this patch — return canvas with no blend applied.
        # The annotation geometry is already recorded; only this patch's visual blend is skipped.
        return canvas

    result = canvas.copy()
    result[y1:y2, x1:x2] = blended_roi
    return result

# ──────────────────────────────────────────────────────────────────────────────
# Functions to generate COCO annotations for synthetic images
# ──────────────────────────────────────────────────────────────────────────────

def _is_clockwise(contour):
    """Return True if contour points are ordered clockwise."""
    value = 0
    n = len(contour)
    for i in range(n):
        p1 = contour[i][0]
        p2 = contour[(i + 1) % n][0]
        value += (p2[0] - p1[0]) * (p2[1] + p1[1])
    return value < 0

def _get_merge_point_idx(c1, c2):
    """Return indices of the closest point pair between contours c1 and c2."""
    idx1, idx2, dmin = 0, 0, -1
    for i, p1 in enumerate(c1):
        for j, p2 in enumerate(c2):
            d = (p2[0][0] - p1[0][0]) ** 2 + (p2[0][1] - p1[0][1]) ** 2
            if dmin < 0 or d < dmin:
                dmin, idx1, idx2 = d, i, j
    return idx1, idx2

def _merge_contours(c1, c2, idx1, idx2):
    """Bridge c1 and c2 at the given indices into a single flat contour."""
    merged = []
    for i in range(idx1 + 1):
        merged.append(c1[i])
    for i in range(idx2, len(c2)):
        merged.append(c2[i])
    for i in range(idx2 + 1):
        merged.append(c2[i])
    for i in range(idx1, len(c1)):
        merged.append(c1[i])
    return np.array(merged)

def mask2polygon(mask):
    """
    Convert a binary mask to a COCO flat-coordinate segmentation list.

    Handles masks with interior holes (e.g. a ring-shaped visible region where
    debris sits fully inside the inlet). For each parent contour every child
    (hole) contour is merged back into the parent by bridging the nearest pair
    of points.  The result is a set of simple, non-self-intersecting polygons
    that fillPoly reproduces exactly — no hole is accidentally filled.

    Returns a list of flat [x0,y0, x1,y1, ...] coordinate lists, one per
    connected visible region.
    """
    result = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    contours, hierarchies = result[-2], result[-1]
    if hierarchies is None:
        return []

    # Approximate each contour
    approx = []
    for c in contours:
        eps = 0.001 * cv2.arcLength(c, True)
        approx.append(cv2.approxPolyDP(c, eps, True))

    # Collect parent slots (outer contours); holes start as empty placeholders
    parents = []
    for i, c in enumerate(approx):
        if hierarchies[0][i][3] < 0 and len(c) >= 3:   # no parent → outer
            parents.append(c)
        else:
            parents.append([])

    # Merge each hole into its parent contour
    for i, c in enumerate(approx):
        parent_idx = hierarchies[0][i][3]
        if parent_idx >= 0 and len(c) >= 3:             # has a parent → hole
            parent = parents[parent_idx]
            if len(parent) == 0:
                continue
            # Parent clockwise, hole counter-clockwise
            if not _is_clockwise(parent):
                parent = parent[::-1]
            if _is_clockwise(c):
                c = c[::-1]
            idx1, idx2 = _get_merge_point_idx(parent, c)
            parents[parent_idx] = _merge_contours(parent, c, idx1, idx2)

    polygons = []
    for c in parents:
        if len(c) == 0:
            continue
        polygons.append(c.flatten().tolist())
    return polygons

def poly_to_coco_seg(poly):
    """Convert a Shapely Polygon to a COCO flat-coordinate segmentation list."""
    if poly is None or poly.is_empty:
        return []
    coords = list(poly.exterior.coords)[:-1]  # drop repeated closing point
    flat   = [v for pt in coords for v in pt]
    return [flat]

def bbox_from_poly(poly):
    """Return COCO-format [x, y, width, height] bounding box from a Shapely Polygon."""
    minx, miny, maxx, maxy = poly.bounds
    return [minx, miny, maxx - minx, maxy - miny]

def shapely_to_coco_seg(geom):
    """
    Convert any Shapely geometry to a COCO flat-coordinate segmentation list.
    Handles both Polygon and MultiPolygon; holes are ignored.
    Required for visible_segmentation which may become a MultiPolygon after
    subtracting placed debris (e.g. debris in the centre of the inlet).
    """
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == 'Polygon':
        polys = [geom]
    elif geom.geom_type in ('MultiPolygon', 'GeometryCollection'):
        polys = [g for g in geom.geoms if g.geom_type == 'Polygon' and not g.is_empty]
    else:
        return []
    result = []
    for poly in polys:
        coords = list(poly.exterior.coords[:-1])
        if len(coords) >= 3:
            result.append([v for pt in coords for v in pt])
    return result

def mask_to_polygon(mask, offset_x, offset_y):
    """
    Convert a binary mask to a Shapely Polygon at image coordinates.
    
    :param mask: (H, W) uint8 binary mask (255 = pixels to include)
    :param offset_x: X offset of mask top-left in image coordinates
    :param offset_y: Y offset of mask top-left in image coordinates
    :returns: Shapely Polygon or None if mask is empty
    """
    # Find contours of non-zero regions
    # Handle CV2 version differences
    result = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = result[-2] if len(result) == 3 else result[0]
    
    if not contours:
        return None
    
    # Use largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Convert contour to coordinate list
    contour_points = largest_contour.squeeze()
    if len(contour_points.shape) == 1:  # Single point
        return None
    
    if len(contour_points) < 3:  # Need at least 3 points for polygon
        return None
    
    # Translate to image coordinates
    points_img = contour_points + np.array([offset_x, offset_y])
    
    try:
        poly = Polygon(points_img)
        if poly.is_valid and not poly.is_empty:
            return poly
    except:
        pass
    
    return None

# ──────────────────────────────────────────────────────────────────────────────
# Internal helper: place debris on a single inlet to hit a target clogging range
# ──────────────────────────────────────────────────────────────────────────────

def _place_debris_on_inlet(
    inlet_poly,
    all_donor_data,
    candidates,
    angle_range,
    scale_range,
    max_placement_attempts,
    target_lo,
    target_hi,
    max_donor_rounds=6,
):
    """
    Place debris patches onto a single inlet polygon until the clogging extent
    falls within [target_lo, target_hi].

    Iterates over up to max_donor_rounds randomly-selected donor images. Within
    each round, every debris annotation in the donor image is attempted. Placement
    stops for the round as soon as the ceiling (target_hi) is reached. If target_lo
    is reached at any round boundary, the function returns immediately.

    The forbidden zone grows with each successfully placed patch so patches never
    overlap each other — identical to the behaviour in the original placement loop.

    :param inlet_poly            : Shapely Polygon — the inlet to clog.
    :param all_donor_data        : list of (donor_img_bgr, donor_anns) pairs
                                   pre-loaded upfront to avoid repeated disk reads.
    :param candidates            : list of (x, y) grid points inside inlet_poly (shuffled).
    :param angle_range           : (min_deg, max_deg) for debris rotation.
    :param scale_range           : (min_s, max_s) for debris scaling.
    :param max_placement_attempts: max candidate+transform tries per single patch.
    :param target_lo             : minimum acceptable clogging ratio.
    :param target_hi             : maximum acceptable clogging ratio (hard ceiling).
    :param max_donor_rounds      : number of donor images to draw before giving up.

    :returns: (cropped_patches, placed_union, actual_ratio)
              cropped_patches is a list of (ox, oy, cropped_poly, c_patch, c_mask).
              placed_union is the Shapely union of all placed debris (or None).
              actual_ratio is the float clogging ratio achieved.
    """
    forbidden       = None
    cropped_patches = []

    def _ratio():
        if not cropped_patches:
            return 0.0
        return clogging_ratio(inlet_poly, unary_union([p for _, _, p, _, _ in cropped_patches]))

    for _ in range(max_donor_rounds):
        if _ratio() >= target_lo:
            break  # already in the target range — done

        donor_img, donor_anns = random.choice(all_donor_data)

        for d_ann in donor_anns:
            if _ratio() >= target_hi:
                break  # hit the ceiling — stop adding patches

            p_bgr, p_mask, p_poly = extract_debris_patch(donor_img, d_ann)
            if p_bgr is None:
                continue

            t_patch, t_mask, t_poly, nw, nh = transform_patch(
                p_bgr, p_mask, p_poly,
                angle_range=angle_range,
                scale_range=scale_range,
            )

            ox, oy, placed_poly, final_patch, final_mask = place_patch_on_grid(
                t_patch, t_mask, t_poly, nw, nh,
                p_bgr, p_mask, p_poly,
                list(candidates),
                inlet_poly,
                forbidden,
                angle_range=angle_range,
                scale_range=scale_range,
                max_attempts=max_placement_attempts,
            )
            if ox is None:
                continue

            c_patch, c_mask, kept_mask = crop_patch_to_valid_region(
                final_patch, final_mask, ox, oy, inlet_poly
            )
            
            # Compute polygon from the debris pixels (c_mask), not from the valid region
            # shape (kept_mask). Using kept_mask inflated _ratio() to the inlet-polygon
            # footprint of the patch bounding box, causing early termination and accepting
            # images well below the target clogging threshold.
            actual_painted_poly = mask_to_polygon(c_mask, ox, oy)
            if actual_painted_poly is None:
                continue  # No actual debris pixels landed inside the inlet — skip
            
            # Intersect with inlet to handle edge cases
            cropped_poly = actual_painted_poly.intersection(inlet_poly)
            
            # Skip degenerate debris patches: area too small or not enough points
            if cropped_poly is None or cropped_poly.is_empty:
                continue
            if isinstance(cropped_poly, MultiPolygon):
                cropped_poly = max(cropped_poly.geoms, key=lambda g: g.area)
            coords = list(cropped_poly.exterior.coords)
            if len(coords) < 4 or cropped_poly.area < 10:
                # <4 because the last point repeats the first; so <4 means <3 unique points
                continue

            forbidden = unary_union([forbidden, cropped_poly]) if forbidden else cropped_poly
            cropped_patches.append((ox, oy, cropped_poly, c_patch, c_mask))

    if cropped_patches:
        placed_union = unary_union([p for _, _, p, _, _ in cropped_patches])
        actual_ratio = clogging_ratio(inlet_poly, placed_union)
        return cropped_patches, placed_union, actual_ratio

    return [], None, 0.0

# ──────────────────────────────────────────────────────────────────────────────
# Main function used to implement the copy-paste pipeline for generating synthetic clogged inlet images
# ──────────────────────────────────────────────────────────────────────────────

def paste_debris_patches(
    synthetic_image_count,
    clean_inlet_data,
    debris_json_path,
    source_images_dir,
    debris_images_dir,
    output_images_dir,
    output_json_path,
    angle_range=(-30, 30),
    scale_range=(0.7, 1.3),
    grid_step=5, # step size to generate candidate placement points inside the inlet polygon
    max_placement_attempts=200,
    target_clogging_min=0.20,
    target_clogging_max=0.90):
    """
    Generate synthetic clogged inlet images by copy-pasting debris onto clean inlets.

    Pipeline
    --------
    1. Randomly pick a clean inlet image and its inlet polygon.
    2. Randomly pick a clogged inlet image and extract its debris polygons/patches.
    3. Apply random rotation and scaling to each debris patch.
    4. Use a spatial grid to find valid, non-overlapping placement positions inside
       the inlet, excluding background debris zones.
    5. Crop patches at the curb line (upper boundary) and opposite inlet edge
       (lower boundary) so debris looks physically plausible.
    6. Blend each patch into the canvas with Poisson (NORMAL_CLONE) blending,
       with an alpha-composite fallback.
    7. Save the synthetic image and accumulate COCO annotations.

    Design notes
    ------------
    - Grid-based placement (Step 4) is O(G) per patch where G = grid cells inside
      the inlet, avoiding expensive random-retry loops.
    - Forbidden zones grow incrementally so later patches cannot overlap earlier ones.
    - `target_clogging_min` ensures the resulting synthetic image is actually clogged;
      if too few patches were placed, the image is skipped rather than mislabelled.

    :param synthetic_image_count:  Number of synthetic images to generate.
    :param clean_inlet_data:       List of dicts from split_clean_clogged (clean side).
    :param debris_json_path:       Path to the manually annotated debris-patch COCO JSON
    :param debris_images_dir:      Directory containing the original (clogged) inlet images with debris annotations
    :param source_images_dir:      Directory containing the original images.
    :param output_images_dir:      Directory to save synthetic images.
    :param output_json_path:       File path to save the output COCO JSON.
    :param angle_range:            (min_deg, max_deg) rotation for augmentation.
    :param scale_range:            (min_s, max_s) scale factor for augmentation.
    :param grid_step:              Pixel spacing of the placement candidate grid.
    :param max_placement_attempts: Max candidate positions tried per patch.
    :param target_clogging_min:    Lower bound of the clogging distribution (start of bin 0).
    :param target_clogging_max:    Upper bound of the clogging distribution (end of bin 4).
                                   Matches the 90 % single-patch cap in place_patch_on_grid.
                                   The range [target_clogging_min, target_clogging_max] is
                                   divided into 5 equal-width bins, each receiving
                                   synthetic_image_count // 5 images so the generated
                                   dataset is uniformly distributed across clogging extents.
                                   Secondary inlets in double-drain images that are below
                                   target_clogging_min will also have debris placed on them
                                   so no annotation has clogging_extent = 0.
    :returns: COCO dict of all generated synthetic annotations.
    """
    os.makedirs(output_images_dir, exist_ok=True)

    # Collect the maximum existing annotation/image IDs to avoid collisions
    # Category ID is resolved by name 'debris' inside load_debris_json
    debris_by_image, debris_img_info = load_debris_json(debris_json_path)
    debris_image_ids = list(debris_by_image.keys())

    all_ann_ids   = [d['inlet_ann']['id'] for d in clean_inlet_data]
    next_ann_id   = (max(all_ann_ids) + 1) if all_ann_ids else 1

    next_image_id = next_ann_id + 100_000  # ensure image IDs don't clash

    # Pre-load all donor images upfront to avoid repeated disk reads during generation
    all_donor_data = []
    for did in debris_image_ids:
        dpath = os.path.join(debris_images_dir, debris_img_info[did]['file_name'])
        dimg  = cv2.imread(dpath)
        if dimg is not None:
            all_donor_data.append((dimg, debris_by_image[did]))
    if not all_donor_data:
        raise RuntimeError("No donor images could be loaded from the debris directory.")

    # ── Bin setup ─────────────────────────────────────────────────────────────
    # Divide [target_clogging_min, target_clogging_max] into 5 equal-width bins.
    # Each bin receives an equal share of synthetic_image_count images so the
    # generated dataset is uniformly distributed across clogging extents.
    #
    # Example (target_clogging_min=0.20, target_clogging_max=0.90, count=1120):
    #   Bin 0: [0.20, 0.34)  — 280 images
    #   Bin 1: [0.34, 0.48)  — 280 images
    #   Bin 2: [0.48, 0.62)  — 280 images
    #   Bin 3: [0.62, 0.76)  — 280 images
    #   Bin 4: [0.76, 0.90]  — 280 images
    #
    # To adjust the thresholds edit target_clogging_min / target_clogging_max
    # in the ConfigArgs / call-site; the bins recompute automatically.
    num_bins  = 4
    bin_width = (target_clogging_max - target_clogging_min) / num_bins
    bin_edges = [round(target_clogging_min + i * bin_width, 6) for i in range(num_bins + 1)]
    bin_edges[-1] = target_clogging_max  # ensure exact upper bound

    # Per-bin quotas (remainder distributed to the first bins)
    # These are SOFT targets with ±10% flexibility to avoid infinite loops.
    bin_quota  = [synthetic_image_count // num_bins] * num_bins
    for i in range(synthetic_image_count % num_bins):
        bin_quota[i] += 1
    bin_counts = [0] * num_bins
    
    # Quota flexibility: allow each bin to be 10% below/above target
    def _quota_range(quota):
        """Return (min_acceptable, max_acceptable) for a bin quota."""
        flexibility = max(1, int(quota * 0.10))  # 10% tolerance, min 1 image
        return (quota - flexibility, quota + flexibility)

    def _get_bin(ratio):
        """Return the bin index for ratio, or None if outside [min, max]."""
        for i in range(num_bins):
            lo = bin_edges[i]
            hi = bin_edges[i + 1]
            if lo <= ratio < hi or (i == num_bins - 1 and lo <= ratio <= hi):
                return i
        return None

    phase_1_to_2_logged = [False]  # Track if we've announced the phase transition
    
    def _pick_target_bin():
        """Pick a bin needing images, using a two-phase strategy for max safety & inlet usage.
        
        Phase 1 (Safe): Fill all bins to their flexible minimum. Once all are ≥min,
        no infinite loop is possible regardless of later rejection rates.
        
        Phase 2 (Exhaust): Fill bins up to flexible maximum until synthetic_image_count
        is reached, maximizing unique inlet usage without risk.
        """
        # Phase 1: Fill any bin below its flexible minimum
        below_min = [i for i in range(num_bins)
                     if bin_counts[i] < _quota_range(bin_quota[i])[0]]
        if below_min:
            return random.choice(below_min)

        if not phase_1_to_2_logged[0]:
            phase_1_to_2_logged[0] = True
            print(f"\n  >>> Transitioning to Phase 2: All bins at flexible min. Exhausting to max... <<<\n")

        # Phase 2: Fill any bin below its flexible maximum
        below_max = [i for i in range(num_bins)
                     if bin_counts[i] < _quota_range(bin_quota[i])[1]]
        if below_max:
            return random.choice(below_max)
        return None  # All bins satisfied

    # ── Bootstrap the output COCO structure ───────────────────────────────────
    output_coco = {
        'info': {
            'contributor':  '',
            'date_created': '',
            'description':  '',
            'url':          '',
            'version':      '',
            'year':         '',
        },
        'licenses':    [{'name': '', 'id': 0, 'url': ''}],
        'categories':  [{'id': 1, 'name': 'storm_drain', 'supercategory': ''}],
        'images':      [],
        'annotations': [],
    }

    generated = 0

    # Group all clean inlet entries by their image ID to safely handle double inlets
    inlets_by_image_id = {}
    for entry in clean_inlet_data:
        img_id = entry['image_info']['id']
        inlets_by_image_id.setdefault(img_id, []).append(entry)

    # Each unique inlet image may contribute at most ceil(total / n_images)
    # successful synthetic images. Once that limit is reached the image is
    # retired from the active pool and is no longer selected. This ensures 
    # that the all available inlets are used equally 
    #
    # Within the active pool, candidates for a given target bin are ranked by:
    #   (1) fewest total successful uses                     ← equal distribution
    #   (2) fewest bin-mismatch failures for the target bin  ← bin-awareness
    # This naturally routes low-clogging bins to small inlets and high-clogging
    # bins to large inlets without hard-coding any inlet geometry.
    #
    # Infinite-loop guard: a per-bin consecutive-failure counter triggers a
    # forced bin completion after n_images × 3 consecutive failures, so the
    # loop is always bounded even when the remaining active pool cannot
    # physically reach a particular clogging target.

    img_group_list   = list(inlets_by_image_id.values())  # one sublist per unique image
    n_unique_images  = len(img_group_list)                 # unique source photos
    # Count total individual inlets (double-drain images each contribute 2 inlets).
    # Per-inlet usage target: synthetic_image_count / n_total_inlets uses per inlet.
    # Since selecting an image always processes ALL its inlets together, the per-image
    # limit is derived from the total-inlet count rather than the image count.
    n_total_inlets   = sum(len(g) for g in img_group_list)
    per_image_limit  = math.ceil(synthetic_image_count / n_total_inlets)
    img_use_count    = [0] * n_unique_images   # successful synthetics per source image
    # Per-image per-bin mismatch tally: high value → this image rarely reaches this bin
    img_bin_failures = [[0] * num_bins for _ in range(n_unique_images)]
    # Safety valve: abort if total failed attempts across all bins exceeds this threshold.
    # Prevents truly infinite loops when the dataset physically cannot fill a bin.
    MAX_TOTAL_FAILED_ATTEMPTS = synthetic_image_count * 50
    total_failed_attempts = 0

    def _pick_inlet_for_bin(target_bin):
        """Select the best inlet image for the target bin using a SOFT per-image limit.

        Images below per_image_limit are always preferred (equal-usage distribution),
        but if all images have hit the limit the least-used image is still selected.
        This ensures bin filling is NEVER hard-blocked by the usage limit — the limit
        is a preference, not an absolute gate.

        Selection priority:
          (1) fewest total successful uses  ← equal-distribution preference
          (2) fewest bin-mismatch failures for the target bin  ← bin-capability awareness

        Within the top-ranked tier a random choice spreads load evenly.
        Returns (entry, group_idx).
        """
        all_idx = list(range(n_unique_images))
        all_idx.sort(key=lambda i: (img_use_count[i], img_bin_failures[i][target_bin]))

        best_use  = img_use_count[all_idx[0]]
        best_fail = img_bin_failures[all_idx[0]][target_bin]
        tier = [i for i in all_idx
                if img_use_count[i] == best_use
                and img_bin_failures[i][target_bin] == best_fail]
        group_idx = random.choice(tier)
        entry     = random.choice(img_group_list[group_idx])
        return entry, group_idx

    print("\nClogging bin targets (±10% flexible quotas):")
    for i in range(num_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        bracket = ']' if i == num_bins - 1 else ')'
        min_cnt, max_cnt = _quota_range(bin_quota[i])
        print(f"  Bin {i}: [{lo:.2f}, {hi:.2f}{bracket}  — target {bin_quota[i]} (accept {min_cnt}–{max_cnt})")
    print(f"\nInlet image sampling: {n_unique_images} unique images / {n_total_inlets} total inlets, "
          f"soft limit {per_image_limit} uses/image")
    print(f"Safety valve: abort after {MAX_TOTAL_FAILED_ATTEMPTS} total failed attempts")
    print()

    with tqdm.tqdm(total=synthetic_image_count, desc="Generating synthetic images") as pbar:
        while generated < synthetic_image_count:

            # Select the next target bin; stop if all quotas are satisfied
            target_bin = _pick_target_bin()
            if target_bin is None:
                break
            
            bin_lo = bin_edges[target_bin]
            bin_hi = bin_edges[target_bin + 1]

            # Select the best inlet for this bin (soft limit — always returns an inlet)
            clean_entry, group_idx = _pick_inlet_for_bin(target_bin)
            clean_info  = clean_entry['image_info']
            inlet_ann   = clean_entry['inlet_ann']
            inlet_poly  = clean_entry['inlet_poly']
            # Existing clogging on the primary inlet (may be >0 for "clean" inlets
            # defined as <20% clogged, not necessarily 0%).
            existing_clog = float(inlet_ann.get('clogging_extent', 0.0))

            clean_img_path = os.path.join(source_images_dir, clean_info['file_name'])
            if not os.path.exists(clean_img_path):
                continue
            canvas = cv2.imread(clean_img_path)

            if canvas is None:
                continue

            img_h, img_w = canvas.shape[:2]

            # Step 2–5: Place debris on the PRIMARY inlet to hit the target bin.
            # Adjust the placement targets downward by the existing clogging so that
            # synthetic debris only needs to bridge the gap to the desired bin range.
            primary_candidates = build_candidate_grid(inlet_poly, grid_step=grid_step)
            synth_target_lo = max(0.0, bin_lo - existing_clog)
            synth_target_hi = max(0.0, bin_hi - existing_clog)
            primary_patches, primary_union, primary_ratio = _place_debris_on_inlet(
                inlet_poly,
                all_donor_data,
                primary_candidates,
                angle_range,
                scale_range,
                max_placement_attempts,
                target_lo=synth_target_lo,
                target_hi=synth_target_hi,
                max_donor_rounds=6 + target_bin * 2,  # harder bins get more donor rounds
            )

            # Accept/reject using Shapely polygon extent (fast — no rasterization on
            # rejected attempts, which are the vast majority of all iterations).
            # Each patch's cropped_poly was derived from the actual painted pixel mask
            # via mask_to_polygon, so the Shapely ratio closely tracks true pixel extent
            # without a full-image rasterization pass.
            # Also accounts for any pre-existing debris on the clean inlet.
            _orig_vis = clean_entry.get('visible_poly', inlet_poly)
            _orig_deb = (inlet_poly.difference(_orig_vis)
                         if _orig_vis and not _orig_vis.is_empty else None)
            _deb_geoms = [g for g in [_orig_deb, primary_union] if g and not g.is_empty]
            _total_deb = unary_union(_deb_geoms) if _deb_geoms else None
            if _total_deb and not _total_deb.is_empty and inlet_poly.area > 0:
                primary_extent = min(1.0, _total_deb.intersection(inlet_poly).area / inlet_poly.area)
            else:
                primary_extent = existing_clog
            if _get_bin(primary_extent) != target_bin:
                img_bin_failures[group_idx][target_bin] += 1
                total_failed_attempts += 1
                if total_failed_attempts >= MAX_TOTAL_FAILED_ATTEMPTS:
                    print(f"\n  CRITICAL: Reached {MAX_TOTAL_FAILED_ATTEMPTS} total failed attempts "
                          f"({generated} images generated). The dataset may be physically unable "
                          f"to fill remaining bins. Stopping generation.\n")
                    break
                continue

            secondary_patches_by_id = {}   # inlet_ann_id -> list of cropped patches
            secondary_unions_by_id  = {}   # inlet_ann_id -> placed union (or None)

            for sec_entry in inlets_by_image_id[clean_info['id']]:
                sec_id   = sec_entry['inlet_ann']['id']
                sec_poly = sec_entry['inlet_poly']

                # Skip the primary inlet — already handled above
                if sec_id == inlet_ann['id']:
                    continue
                if sec_poly is None or sec_poly.is_empty:
                    continue

                # Existing clogging on the secondary inlet ("clean" inlets may have
                # up to target_clogging_min worth of real debris already present).
                sec_existing_clog = float(sec_entry['inlet_ann'].get('clogging_extent', 0.0))
                if sec_existing_clog >= target_clogging_min:
                    # Already adequately clogged — leave as is
                    secondary_patches_by_id[sec_id] = []
                    secondary_unions_by_id[sec_id]  = None
                    continue

                # Secondary inlet is below the minimum: place debris independently.
                # Adjust placement targets to bridge only the gap above existing clogging.
                sec_candidates = build_candidate_grid(sec_poly, grid_step=grid_step)
                sec_patches, sec_union, _ = _place_debris_on_inlet(
                    sec_poly,
                    all_donor_data,
                    sec_candidates,
                    angle_range,
                    scale_range,
                    max_placement_attempts,
                    target_lo=max(0.0, target_clogging_min - sec_existing_clog),
                    target_hi=max(0.0, target_clogging_max - sec_existing_clog),
                )
                secondary_patches_by_id[sec_id] = sec_patches
                secondary_unions_by_id[sec_id]  = sec_union

            # Step 6: Poisson blending — primary inlet first, then any secondary inlets
            result = canvas.copy()

            for ox, oy, _, c_patch, c_mask in primary_patches:
                result = poisson_blend_patch(result, c_patch, c_mask, ox, oy)

            for sec_id, sec_patches in secondary_patches_by_id.items():
                for ox, oy, _, c_patch, c_mask in sec_patches:
                    result = poisson_blend_patch(result, c_patch, c_mask, ox, oy)

            # Save synthetic image with a new filename
            synth_fname   = f"synthetic_{generated:04d}.png"
            synth_outpath = os.path.join(output_images_dir, synth_fname)
            cv2.imwrite(synth_outpath, result)

            # Build COCO records
            synth_image_id = next_image_id
            next_image_id += 1

            output_coco['images'].append({
                'id':           synth_image_id,
                'width':        img_w,
                'height':       img_h,
                'file_name':    synth_fname,
                'license':      0,
                'flickr_url':   '',
                'coco_url':     '',
                'date_captured': 0,
            })

            # Build one storm_drain annotation per inlet in this source image.
            #
            # Both clogging_extent and visible_segmentation are derived from the
            # same Shapely polygon operations so they are perfectly consistent
            # with each other and with the acceptance-bin check.
            #
            # Formula:  clogging_extent = debris_area / inlet_area
            #                           = (inlet - visible) / inlet
            # This matches the model's inference formula:
            #   pred_clogging = (amodal_pixels - visible_pixels) / amodal_pixels
            # (same concept; model operates at 28×28 ROI resolution, annotation uses
            # exact polygon areas — the same unavoidable difference exists for all
            # real hand-labelled annotations in the training set.)
            
            # Build deterministic map of inlet_id -> placed_union to ensure correct
            # assignment regardless of iteration order (fixes double-inlet bug).
            placed_union_by_id = {inlet_ann['id']: primary_union}
            placed_union_by_id.update(secondary_unions_by_id)
            
            # Compute image-level clogging extent as max of all inlets (conservative).
            # This single value represents the synthetic image in the bin and metrics.
            #
            # DESIGN DECISION (for double-inlet images):
            #   - Each inlet gets its own per-inlet clogging_extent in the JSON
            #   - The synthetic image is BINNED and COUNTED based on the PRIMARY inlet
            #   - If secondary inlet has higher clogging, the image-level max reflects that
            #   - This ensures bins reflect the most conservative clogging present in any inlet
            all_inlet_extents = []
            
            for double_entry in inlets_by_image_id[clean_info['id']]:
                all_inlet_poly = double_entry['inlet_poly']
                if all_inlet_poly is None or all_inlet_poly.is_empty:
                    continue

                entry_id = double_entry['inlet_ann']['id']

                # Placed-debris polygon for this specific inlet (deterministic lookup).
                placed_union = placed_union_by_id.get(entry_id)

                # Original (pre-existing) debris from the clean source image.
                # visible_poly is the non-occluded region; everything inside
                # inlet_poly but outside visible_poly is original debris.
                orig_vis = double_entry.get('visible_poly', all_inlet_poly)
                orig_deb = (all_inlet_poly.difference(orig_vis)
                            if orig_vis and not orig_vis.is_empty else None)

                # Total debris = original + synthetic, clipped to the inlet boundary
                deb_geoms = [g for g in [orig_deb, placed_union]
                             if g is not None and not g.is_empty]
                total_deb = (unary_union(deb_geoms).intersection(all_inlet_poly)
                             if deb_geoms else None)

                # clogging_extent: fraction of the inlet covered by debris
                if total_deb and not total_deb.is_empty and all_inlet_poly.area > 0:
                    extent = min(1.0, total_deb.area / all_inlet_poly.area)
                else:
                    extent = float(double_entry['inlet_ann'].get('clogging_extent', 0.0))
                
                all_inlet_extents.append(extent)

                # visible_segmentation: the inlet region NOT covered by debris.
                # Derived from the same geometry as clogging_extent so the two
                # fields are guaranteed to be mutually consistent.
                if total_deb and not total_deb.is_empty:
                    visible_geom = all_inlet_poly.difference(total_deb)
                elif orig_vis and not orig_vis.is_empty:
                    visible_geom = orig_vis
                else:
                    visible_geom = all_inlet_poly
                visible_segmentation = (shapely_to_coco_seg(visible_geom)
                                        if visible_geom and not visible_geom.is_empty else [])

                output_coco['annotations'].append({
                    'id':                   next_ann_id,
                    'image_id':             synth_image_id,
                    'category_id':          1,
                    'segmentation':         poly_to_coco_seg(all_inlet_poly),
                    'visible_segmentation': visible_segmentation,
                    'area':                 float(all_inlet_poly.area),
                    'bbox':                 bbox_from_poly(all_inlet_poly),
                    'iscrowd':              0,
                    'clogging_extent':      round(float(extent), 4),
                })
                next_ann_id += 1

            bin_counts[target_bin] += 1
            img_use_count[group_idx] += 1        # mark this source image as used
            generated += 1
            pbar.update(1)
            sec_count = sum(len(v) for v in secondary_patches_by_id.values())
            img_extent = max(all_inlet_extents) if all_inlet_extents else 0.0
            extent_str = ",".join(f"{e:.2f}" for e in all_inlet_extents)
            print(f"  [{generated}/{synthetic_image_count}] {synth_fname} "
                  f"| bin={target_bin} [{bin_edges[target_bin]:.2f},{bin_edges[target_bin+1]:.2f}) "
                  f"| extents=[{extent_str}] (max={img_extent:.2f}) "
                  f"| img={group_idx}(x{img_use_count[group_idx]}/{per_image_limit}) "
                  f"| primary_patches={len(primary_patches)} "
                  f"| secondary_patches={sec_count}")

    # ← while loop (generated < synthetic_image_count) ends here

    with open(output_json_path, 'w') as f:
        json.dump(output_coco, f, indent=2)

    print(f"\nDone — {generated} synthetic images saved to '{output_images_dir}'.")
    print("Bin distribution (with ±10% flexible quotas):")
    for b in range(num_bins):
        min_cnt, max_cnt = _quota_range(bin_quota[b])
        status = "✓" if min_cnt <= bin_counts[b] <= max_cnt else "UNDER"
        print(f"  bin {b} [{bin_edges[b]:.2f}, {bin_edges[b+1]:.2f}): "
              f"{bin_counts[b]} images (target {bin_quota[b]}, accept {min_cnt}-{max_cnt}) [{status}]")

    use_arr = np.array(img_use_count)
    print(f"\nInlet usage (soft limit={per_image_limit}/image, "
          f"{n_unique_images} images / {n_total_inlets} total inlets):")
    print(f"  Unique images used : {(use_arr > 0).sum()}/{n_unique_images}")
    if use_arr.sum() > 0:
        cv_val = float(use_arr.std()) / float(use_arr.mean())
        print(f"  Uses: max={use_arr.max()}  min={use_arr.min()}  "
              f"mean={use_arr.mean():.2f}  CV={cv_val:.3f}")
        print(f"  (Ideal: max={per_image_limit}  min={per_image_limit - 1}  CV≈0)")

    return output_coco


# ─────────────────────────────────────────────────────────────────────────────
# Merge clean inlet JSON into synthetic JSON
# ─────────────────────────────────────────────────────────────────────────────

def merge_clean_into_synthetic(
    output_json_path,
    output_images_dir,
    clean_json_path,
    clean_images_dir):
    """
    After synthetic images are generated, merge clean inlet data into the synthetic JSON.
    
    This function:
    1. Loads the synthetic JSON that was just saved
    2. Loads the clean.json file
    3. Copies clean images from clean_images_dir to output_images_dir
    4. Re-IDs clean data to avoid conflicts with synthetic IDs
    5. Merges clean annotations into the synthetic JSON
    6. Saves the merged JSON back
    
    :param output_json_path:   Path to the synthetic JSON (will be updated with merged data)
    :param output_images_dir:  Directory where synthetic images are saved (clean images copied here)
    :param clean_json_path:    Path to the clean.json file
    :param clean_images_dir:   Directory containing clean inlet images
    """
    print(f"\n{'='*80}")
    print("Merging clean inlet JSON into synthetic JSON...")
    print(f"{'='*80}")
    
    # Load the synthetic JSON that was just saved
    with open(output_json_path, 'r') as f:
        synthetic_coco = json.load(f)
    
    # Load clean.json
    with open(clean_json_path, 'r') as f:
        clean_coco = json.load(f)
    
    # Find max IDs in synthetic to offset clean IDs
    max_image_id = max([img['id'] for img in synthetic_coco['images']]) if synthetic_coco['images'] else 0
    max_annotation_id = max([ann['id'] for ann in synthetic_coco['annotations']]) if synthetic_coco['annotations'] else 0
    
    print(f"Synthetic JSON: {len(synthetic_coco['images'])} images, {len(synthetic_coco['annotations'])} annotations")
    print(f"  Max image ID: {max_image_id}")
    print(f"  Max annotation ID: {max_annotation_id}")
    
    print(f"Clean JSON: {len(clean_coco['images'])} images, {len(clean_coco['annotations'])} annotations")
    
    # Create mapping from old clean image IDs to new IDs
    image_id_map = {}
    
    # Copy clean images and update IDs
    for clean_img in clean_coco['images']:
        old_img_id = clean_img['id']
        new_img_id = old_img_id + max_image_id
        image_id_map[old_img_id] = new_img_id
        
        # Copy the image file from clean_images_dir to output_images_dir
        src_image_path = os.path.join(clean_images_dir, clean_img['file_name'])
        dst_image_path = os.path.join(output_images_dir, clean_img['file_name'])
        
        if os.path.exists(src_image_path):
            shutil.copyfile(src_image_path, dst_image_path)
            print(f"  Copied: {clean_img['file_name']}")
        else:
            print(f"  WARNING: Source image not found: {src_image_path}")
        
        # Update image ID
        clean_img['id'] = new_img_id
        # Add to synthetic images
        synthetic_coco['images'].append(clean_img)
    
    # Update and add annotations
    for clean_ann in clean_coco['annotations']:
        old_img_id = clean_ann['image_id']
        new_img_id = image_id_map[old_img_id]
        
        # Update annotation IDs
        clean_ann['id'] = clean_ann['id'] + max_annotation_id
        clean_ann['image_id'] = new_img_id
        
        # Add to synthetic annotations
        synthetic_coco['annotations'].append(clean_ann)
    
    # Save the merged JSON
    with open(output_json_path, 'w') as f:
        json.dump(synthetic_coco, f, indent=2)
    
    print(f"\n✓ Merge complete!")
    print(f"  Final counts:")
    print(f"    Images: {len(synthetic_coco['images'])}")
    print(f"    Annotations: {len(synthetic_coco['annotations'])}")
    print(f"  Merged JSON saved to: {output_json_path}")


# ─────────────────────────────────────────────────────────────────────────────
# ConfigArgs: Configuration for synthetic image generation
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    base_dir = "/run/user/1000/gvfs/smb-share:server=ecresearch.uwaterloo.ca,share=cviss/Jack/baf/360streetview/segmentation_training"
    clean_inlet_images_dir = os.path.join(base_dir, 'clean_inlets', 'images')   # Directory containing the clean (non-clogged) inlet images
    clean_inlet_json_path = os.path.join(base_dir, 'clean_inlets', 'clean.json')   # Path to the clean.json COCO annotations for the clean inlets
    output_images_dir = os.path.join(base_dir, 'synthetic_v2', 'images')   # Directory to save the synthetic images
    output_json_path  = os.path.join(base_dir, 'synthetic_v2', 'synthetic.json')   # File path to save the output COCO JSON
    debris_images_dir = os.path.join(base_dir, 'debris_masks', 'images')   # Directory containing the original (clogged) inlet images with debris annotations
    debris_json_path  = os.path.join(base_dir, 'debris_masks', 'debris.json')   # Path to the manually annotated debris-patch COCO JSON
    clean_inlets_pkl  = os.path.join(base_dir, 'clean_inlets', 'clean_inlets_data.pkl')   # Path to the clean_inlets_data.pkl file

    with open(clean_inlets_pkl, 'rb') as f:
        clean_inlets_data = pickle.load(f)

    # Generate synthetic clogged images
    synthetic_image_count = 1126

    paste_debris_patches(
        synthetic_image_count  = synthetic_image_count,
        clean_inlet_data       = clean_inlets_data,
        debris_json_path       = debris_json_path,
        source_images_dir      = clean_inlet_images_dir,
        debris_images_dir      = debris_images_dir,
        output_images_dir      = output_images_dir,
        output_json_path       = output_json_path,
        angle_range            = (-30, 30),
        scale_range            = (0.7, 1.3),
        grid_step              = 5,
        max_placement_attempts = 100,
        target_clogging_min    = 0.20,
        target_clogging_max    = 0.90,
    )
    
    # Merge clean inlet JSON into the synthetic JSON
    merge_clean_into_synthetic(
        output_json_path=output_json_path,
        output_images_dir=output_images_dir,
        clean_json_path=clean_inlet_json_path,
        clean_images_dir=clean_inlet_images_dir,
    )

