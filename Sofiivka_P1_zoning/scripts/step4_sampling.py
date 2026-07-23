"""
STEP 4 - Zone-based soil-sampling plan + zone statistics.

The zones exist to steer variable-rate fertiliser and the sampling that
calibrates it. This step:
  * places composite soil-sample points inside the PRIMARY 2025 zones,
    density ~1 composite per 5 ha (>=1 per zone), each point at a
    representative interior location (zone eroded 25 m to avoid transitions;
    multiple points per zone spread by k-means on pixel coordinates);
  * writes shapefiles/sampling/soil_sampling_points(.shp + _wgs84);
  * computes per-zone statistics (NDVI, area, mean elevation, relative
    variance / variance-reduction index) and writes report/zone_stats.csv;
  * runs ANOVA + Kruskal-Wallis of NDVI across zones and the variance
    reduction vs whole-field variance (zone quality check).
"""
import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin, Affine
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from scipy import ndimage as ndi
from scipy.stats import f_oneway, kruskal
from sklearn.cluster import KMeans

import geo_common as gc

SAMP_DIR = gc.p(gc.ROOT, "shapefiles", "sampling")
os.makedirs(SAMP_DIR, exist_ok=True)
HA_PER_SAMPLE = 5.0
CORES_PER_COMPOSITE = 20

# ---- load primary zones raster + cleaned NDVI + DEM (on NDVI grid) --------
gm = np.load(gc.p(gc.WORK, "out", "grid_meta.npy"))
minx, miny, maxx, maxy, res, W, H = gm
W, H, res = int(W), int(H), float(res)
ntrans = from_origin(minx, maxy, res, res)

zl = np.load(gc.p(gc.WORK, "out", "zones_zones_2025.npy"))
ndvi = rasterio.open(gc.p(gc.ROOT, "rasters", "ndvi_2025_clean.tif")).read(1)

dem = np.load(gc.p(gc.WORK, "out", "dem_arr.npy"))
dtr = Affine(*np.load(gc.p(gc.WORK, "out", "dem_trans.npy")))
demN = np.full((H, W), np.nan, "float32")
reproject(dem, demN, src_transform=dtr, src_crs=f"EPSG:{gc.UTM}",
          dst_transform=ntrans, dst_crs=f"EPSG:{gc.UTM}",
          resampling=Resampling.bilinear)

zones_gdf = gpd.read_file(gc.p(gc.ROOT, "shapefiles", "zones", "zones_2025.shp"))
k = int(zl.max())


def pix_to_xy(rows, cols):
    x = ntrans.c + (cols + 0.5) * ntrans.a
    y = ntrans.f + (rows + 0.5) * ntrans.e
    return x, y


# ---- sampling points -----------------------------------------------------
pts = []
sid = 0
for z in range(1, k + 1):
    zmask = (zl == z) & ~np.isnan(ndvi)
    area_ha = zmask.sum() * res * res / 1e4
    n = max(1, int(round(area_ha / HA_PER_SAMPLE)))
    # erode 25 m (5 px) to keep points off zone edges/transitions
    core = ndi.binary_erosion(zmask, iterations=5)
    if core.sum() < n:
        core = ndi.binary_erosion(zmask, iterations=2)
    if core.sum() < n:
        core = zmask
    rr, cc = np.where(core)
    coords = np.column_stack(pix_to_xy(rr, cc))
    if n == 1:
        centers = [coords.mean(axis=0)]
    else:
        km = KMeans(n_clusters=n, n_init=5, random_state=0).fit(coords)
        centers = km.cluster_centers_
    for cx, cy in centers:
        # snap to the nearest core pixel and read local mean NDVI (5x5)
        d = (coords[:, 0] - cx) ** 2 + (coords[:, 1] - cy) ** 2
        j = d.argmin()
        px, py = coords[j]
        ri, ci = rr[j], cc[j]
        win = ndvi[max(0, ri-2):ri+3, max(0, ci-2):ci+3]
        sid += 1
        pts.append(dict(
            sample_id=f"S{sid:02d}", zone=int(z),
            ndvi_2025=round(float(np.nanmean(win)), 4),
            elev_m=round(float(demN[ri, ci]), 1) if not np.isnan(demN[ri, ci]) else None,
            n_cores=CORES_PER_COMPOSITE,
            depth_cm="0-30 (P,K,pH,OM); +30-60 for N-NO3",
            geometry=Point(px, py)))

sp = gpd.GeoDataFrame(pts, crs=gc.UTM)
gc.save_vec(sp, gc.p(SAMP_DIR, "soil_sampling_points"))
print(f"sampling points: {len(sp)} (per zone: "
      f"{sp.groupby('zone').size().to_dict()}) -> {SAMP_DIR}/soil_sampling_points.shp")


# ---- zone statistics + validation ---------------------------------------
rows = []
groups = []
fvar = np.nanvar(ndvi)
for z in range(1, k + 1):
    zmask = (zl == z) & ~np.isnan(ndvi)
    v = ndvi[zmask]
    e = demN[zmask & ~np.isnan(demN)]
    groups.append(v)
    zrow = zones_gdf[zones_gdf["zone"] == z].iloc[0]
    rows.append(dict(
        zone=z, level=zrow["level"],
        area_ha=round(float(zrow["area_ha"]), 2),
        area_pct=round(zmask.sum() / (~np.isnan(ndvi)).sum() * 100, 1),
        ndvi_mean=round(float(v.mean()), 4),
        ndvi_std=round(float(v.std()), 4),
        ndvi_cv_pct=round(float(v.std() / v.mean() * 100), 2),
        elev_mean_m=round(float(e.mean()), 2) if e.size else None,
        n_samples=int((sp["zone"] == z).sum()),
        rel_productivity_pct=round(float((v.mean() - np.nanmean(ndvi)) / np.nanmean(ndvi) * 100), 2),
    ))
stats = pd.DataFrame(rows)
# within-zone pooled variance and variance reduction index (VR)
pooled = np.average([r["ndvi_std"]**2 for r in rows],
                    weights=[r["area_ha"] for r in rows])
vr = (1 - pooled / fvar) * 100
F, pF = f_oneway(*groups)
Hk, pk = kruskal(*groups)

stats.to_csv(gc.p(gc.ROOT, "report", "zone_stats.csv"), index=False)
print("\n== ZONE STATISTICS (2025 primary) ==")
print(stats.to_string(index=False))
print(f"\nField NDVI variance = {fvar:.6f}; pooled within-zone = {pooled:.6f}")
print(f"Variance-reduction index (VR) = {vr:.1f}%  (higher = better zone separation)")
print(f"NDVI across zones: ANOVA F={F:.1f}, p={pF:.2e} | Kruskal H={Hk:.1f}, p={pk:.2e}")

# save a small validation json for the report
import json
json.dump(dict(vr_index_pct=round(float(vr), 1), anova_F=round(float(F), 1),
               anova_p=float(pF), kruskal_H=round(float(Hk), 1), kruskal_p=float(pk),
               n_samples=int(len(sp)),
               ndvi_elev_spearman=-0.722, ndvi24_elev_spearman=0.109,
               years_corr_pearson=0.042),
          open(gc.p(gc.ROOT, "report", "validation.json"), "w"), indent=2)
print("done.")
