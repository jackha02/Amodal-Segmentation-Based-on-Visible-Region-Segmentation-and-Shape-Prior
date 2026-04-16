"""
Quick audit of synthetic.json:
  - Q1: clogging distribution (bins vs intended)
  - Q2: clean inlet image reuse distribution
  - Q3: consistency between geometric and pixel clogging
"""
import json, collections, numpy as np, pickle

SYNTH_JSON = "/run/user/1000/gvfs/smb-share:server=ecresearch.uwaterloo.ca,share=cviss/Jack/baf/360streetview/segmentation_training/synthetic_v2/synthetic.json"
CLEAN_PKL  = "/run/user/1000/gvfs/smb-share:server=ecresearch.uwaterloo.ca,share=cviss/Jack/baf/360streetview/segmentation_training/clean_inlets/clean_inlets_data.pkl"

d = json.load(open(SYNTH_JSON))
anns = d["annotations"]
print(f"Total images: {len(d['images'])}, Total annotations: {len(anns)}")

# ── Load clean inlet pkl to map polygon → source filename ─────────────────
with open(CLEAN_PKL, 'rb') as f:
    clean_data = pickle.load(f)
print(f"Clean inlets in pkl: {len(clean_data)}")
print(f"Unique clean image filenames: {len(set(e['image_info']['file_name'] for e in clean_data))}")

fp_to_fname = {}
for e in clean_data:
    seg = e['inlet_ann'].get('segmentation', [[]])
    if seg and seg[0]:
        pts = seg[0]
        fp = tuple(round(v/5)*5 for v in pts[:8])
        fp_to_fname[fp] = e['image_info']['file_name']
print(f"Mapped {len(fp_to_fname)} polygon fingerprints to clean image filenames")

# ── Q1 ─────────────────────────────────────────────────────────────────────
clogging = [a["clogging_extent"] for a in anns if "clogging_extent" in a]
print("\n=== Q1: CLOGGING DISTRIBUTION (by synthetic bin) ===")
synth_bins = [(0.20,0.34),(0.34,0.48),(0.48,0.62),(0.62,0.76),(0.76,0.90)]
for lo, hi in synth_bins:
    n = sum(1 for c in clogging if lo <= c < hi)
    if hi == 0.90:
        n += sum(1 for c in clogging if c >= 0.90)
    print(f"  [{lo:.2f},{hi:.2f}): {n:4d}  (expected ~200)")
print(f"  Below 0.20: {sum(1 for c in clogging if c < 0.20)}")
print(f"  Above 0.90: {sum(1 for c in clogging if c > 0.90)}")

# ── Q2 ─────────────────────────────────────────────────────────────────────
fname_usage = collections.Counter()
for a in anns:
    seg = a.get("segmentation", [[]])
    if seg and seg[0]:
        fp = tuple(round(v/5)*5 for v in seg[0][:8])
        fname = fp_to_fname.get(fp, "UNKNOWN")
        fname_usage[fname] += 1

print("\n=== Q2: CLEAN IMAGE REUSE ===")
usage_counts = sorted(fname_usage.values(), reverse=True)
N = len(usage_counts)
print(f"Total distinct base clean inlets: {N}")
print(f"Unmapped: {fname_usage.get('UNKNOWN', 0)}")
print(f"Max / Mean / Min uses: {max(usage_counts)} / {np.mean(usage_counts):.1f} / {min(usage_counts)}")
print(f"Coefficient of variation: {np.std(usage_counts)/np.mean(usage_counts):.2f}  (0=uniform, >0.5=poor)")
print(f"Expected uniform: {len(anns)/N:.1f} uses per inlet")
print(f"Inlets used <=5x:  {sum(1 for c in usage_counts if c<=5)}")
print(f"Inlets used 6-10x: {sum(1 for c in usage_counts if 5<c<=10)}")
print(f"Inlets used >10x:  {sum(1 for c in usage_counts if c>10)}")
print(f"\nTop 15 most-used base clean images:")
for fname, cnt in fname_usage.most_common(15):
    print(f"  {cnt:3d}x  {fname}")
print(f"\nBottom 10 least-used:")
for fname, cnt in fname_usage.most_common()[-10:]:
    print(f"  {cnt:3d}x  {fname}")

# Also check per-bin diversity
print(f"\nUnique base inlets per clogging bin (annotation-level):")
for lo, hi in synth_bins:
    hi_check = 1.01 if hi == 0.90 else hi
    bin_anns = [a for a in anns if lo <= a.get("clogging_extent", 0) < hi_check]
    polys = set()
    for a in bin_anns:
        seg = a.get("segmentation", [[]])
        if seg and seg[0]:
            polys.add(tuple(round(v/5)*5 for v in seg[0][:8]))
    print(f"  [{lo:.2f},{hi:.2f}): {len(bin_anns):4d} anns from {len(polys):3d} unique base inlets")
