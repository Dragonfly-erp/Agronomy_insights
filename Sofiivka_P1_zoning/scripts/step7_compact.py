"""
STEP 7 - COMPACT, machine-workable zones (client revision #2).

Problem with step6: k-means on NDVI *value only* produced thin, interlocking
"hairy" fingers stretching across the field - impossible to drive.

Fix = make zones spatially compact:
  1. heavy pre-smoothing of NDVI (broad pattern only);
  2. SPATIALLY-CONSTRAINED clustering: features = [NDVI_z, w*x, w*y] so a cluster
     must be coherent in NDVI *and* in space -> compact blobs, not fingers;
  3. de-hairing: one-hot each zone, strong Gaussian, argmax -> removes thin necks
     and tendrils, keeps solid bodies;
  4. drop connected pieces < MIN_HA (merge into surrounding zone);
  5. round + simplify boundaries.

Tunables at the top. Everything else (stats, sampling, QML, preview) follows.
"""
import os
import numpy as np
import rasterio
from rasterio.transform import from_origin, Affine
from rasterio.warp import reproject, Resampling
from rasterio.features import shapes as rio_shapes
from scipy import ndimage as ndi
from sklearn.cluster import KMeans
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape, Point
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import geo_common as gc

# ---------------- tunables ----------------
# --- LEVEL C (client-approved): faithful to NDVI + moderate de-hairing ---
K = 5
PRE_SIGMA_M = 10        # light pre-cluster NDVI smoothing (m) - keep real pattern
SPATIAL_W = 0.0         # 0 = pure NDVI clustering (fidelity); de-hairing does compacting
COMPACT_SIGMA_M = 32    # de-hairing Gaussian on zone membership (m) - trims fingers
MIN_HA = 3.0            # drop/merge connected pieces smaller than this (~3.5%)
ROUND_M = 14            # boundary rounding radius (m)
HA_PER_SAMPLE = 5.0
CORES = 20
# ------------------------------------------

COLORS = ["#d7191c", "#fdae61", "#ffffbf", "#a6d96a", "#1a9641"]
LEVELS_UA = ["дуже низька", "низька", "середня", "висока", "дуже висока"]
ZONE_DIR = gc.p(gc.ROOT, "shapefiles", "zones")
SAMP_DIR = gc.p(gc.ROOT, "shapefiles", "sampling")
FIG = gc.p(gc.ROOT, "figures")

src = rasterio.open(gc.p(gc.ROOT, "rasters", "ndvi_2025_clean.tif"))
arr = src.read(1)
tr = src.transform
mask = ~np.isnan(arr)
H, W = arr.shape
res = gc.RES
min_px = int(MIN_HA * 1e4 / res ** 2)


def sieve(lab, mask, min_px, iters=30):
    lab = lab.copy()
    for _ in range(iters):
        changed = 0
        for cls in np.unique(lab[mask]):
            comp, n = ndi.label((lab == cls) & mask)
            if n <= 1 and cls != 0:
                pass
            sizes = ndi.sum(np.ones_like(comp), comp, range(1, n + 1)) if n else []
            for i, sz in enumerate(sizes, 1):
                if sz < min_px:
                    blob = comp == i
                    dil = ndi.binary_dilation(blob, iterations=2) & mask & ~blob
                    neigh = lab[dil]
                    neigh = neigh[neigh != cls]
                    if neigh.size:
                        v, c = np.unique(neigh, return_counts=True)
                        lab[blob] = v[c.argmax()]
                        changed += 1
        if not changed:
            break
    return lab


# 1. heavy pre-smoothing
sm = ndi.gaussian_filter(np.where(mask, arr, np.nanmean(arr)), PRE_SIGMA_M / res)

# 2. spatially-constrained clustering
rr, cc = np.where(mask)
ndvi_z = (sm[mask] - sm[mask].mean()) / sm[mask].std()
xz = (cc - cc.mean()) / cc.std()
yz = (rr - rr.mean()) / rr.std()
feat = np.column_stack([ndvi_z, SPATIAL_W * xz, SPATIAL_W * yz])
km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(feat)
lab = np.zeros((H, W), "int32")
lab[mask] = km.labels_ + 1
# order by mean raw NDVI
order = sorted(range(1, K + 1), key=lambda c: arr[(lab == c) & mask].mean())
remap = {o: n for n, o in enumerate(order, 1)}
lab = np.vectorize(lambda v: remap.get(v, 0))(lab).astype("int32")
lab[~mask] = 0

# 3. de-hairing via strong one-hot Gaussian + argmax
prob = np.stack([ndi.gaussian_filter((lab == z).astype("float32"),
                                      COMPACT_SIGMA_M / res) for z in range(1, K + 1)])
lab2 = np.zeros((H, W), "int32")
lab2[mask] = (prob.argmax(0) + 1)[mask]

# 4. drop small pieces
lab2 = sieve(lab2, mask, min_px)

# vectorise + round + simplify
recs = []
for geom, val in rio_shapes(lab2, mask=mask, transform=tr):
    if int(val) == 0:
        continue
    recs.append({"zone": int(val), "geometry": shape(geom)})
gdf = gpd.GeoDataFrame(recs, crs=gc.UTM).dissolve(by="zone", as_index=False)
gdf["geometry"] = (gdf.geometry.buffer(ROUND_M, join_style=1)
                   .buffer(-2 * ROUND_M, join_style=1)
                   .buffer(ROUND_M, join_style=1).simplify(5))
gdf = gdf[gdf.geometry.area > 0]

# rasterise the smoothed polygons back for consistent stats
from rasterio.features import rasterize
labf = rasterize([(g, z) for g, z in zip(gdf.geometry, gdf.zone)],
                 out_shape=(H, W), transform=tr, fill=0, dtype="int32")

fmean = float(arr[mask].mean())
rows = []
for _, r in gdf.sort_values("zone").iterrows():
    z = int(r.zone)
    v = arr[(labf == z) & mask]
    parts = len(list(r.geometry.geoms)) if r.geometry.geom_type == "MultiPolygon" else 1
    rows.append(dict(zone=z, level=LEVELS_UA[z - 1],
                     area_ha=round(r.geometry.area / 1e4, 2),
                     ndvi_mean=round(float(v.mean()), 4),
                     ndvi_std=round(float(v.std()), 4),
                     rel_pct=round((v.mean() - fmean) / fmean * 100, 2),
                     n_parts=parts, color=COLORS[z - 1]))
stats = pd.DataFrame(rows)
gdf = gdf.merge(stats.drop(columns="color"), on="zone")
gdf["color"] = gdf["zone"].map(lambda z: COLORS[z - 1])
gc.save_vec(gdf, gc.p(ZONE_DIR, "zones_2025"))
print("COMPACT zones:")
print(stats[["zone", "level", "area_ha", "ndvi_mean", "rel_pct", "n_parts"]].to_string(index=False))
print("total parts:", int(stats.n_parts.sum()))

# ---- sampling on the compact zones ----
dem = np.load(gc.p(gc.WORK, "out", "dem_arr.npy"))
dtr = Affine(*np.load(gc.p(gc.WORK, "out", "dem_trans.npy")))
demN = np.full((H, W), np.nan, "float32")
reproject(dem, demN, src_transform=dtr, src_crs=f"EPSG:{gc.UTM}",
          dst_transform=tr, dst_crs=f"EPSG:{gc.UTM}", resampling=Resampling.bilinear)
pts, sid = [], 0
for z in range(1, K + 1):
    zm = (labf == z) & mask
    n = max(1, int(round(zm.sum() * res ** 2 / 1e4 / HA_PER_SAMPLE)))
    core = ndi.binary_erosion(zm, iterations=5)
    if core.sum() < n:
        core = ndi.binary_erosion(zm, iterations=2)
    if core.sum() < n:
        core = zm
    ys, xs = np.where(core)
    xy = np.column_stack([tr.c + (xs + .5) * tr.a, tr.f + (ys + .5) * tr.e])
    centers = [xy.mean(0)] if n == 1 else KMeans(n, n_init=5, random_state=0).fit(xy).cluster_centers_
    for cx, cy in centers:
        j = ((xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2).argmin()
        yi, xi = ys[j], xs[j]
        sid += 1
        pts.append(dict(sample_id=f"S{sid:02d}", zone=int(z),
                        ndvi_2025=round(float(arr[yi, xi]), 4),
                        elev_m=round(float(demN[yi, xi]), 1) if not np.isnan(demN[yi, xi]) else None,
                        n_cores=CORES, depth_cm="0-30 (P,K,pH,OM); +30-60 N-NO3",
                        geometry=Point(xy[j, 0], xy[j, 1])))
sp = gpd.GeoDataFrame(pts, crs=gc.UTM)
gc.save_vec(sp, gc.p(SAMP_DIR, "soil_sampling_points"))
stats["n_samples"] = stats.zone.map(sp.groupby("zone").size())
stats.to_csv(gc.p(gc.ROOT, "report", "zone_stats.csv"), index=False)
print("sampling:", len(sp), sp.groupby("zone").size().to_dict())

# ---- preview from the saved shapefile ----
zf = gpd.read_file(gc.p(ZONE_DIR, "zones_2025.shp")).sort_values("zone")
spf = gpd.read_file(gc.p(SAMP_DIR, "soil_sampling_points.shp"))
cont = gpd.read_file(gc.p(gc.ROOT, "shapefiles", "relief", "relief_contours.shp"))
fig, ax = plt.subplots(figsize=(12, 11))
for _, r in zf.iterrows():
    gpd.GeoSeries([r.geometry], crs=zf.crs).plot(ax=ax, color=r.color,
                                                 edgecolor="white", linewidth=1.6)
cont.plot(ax=ax, color="#666", linewidth=0.35, alpha=0.45)
spf.plot(ax=ax, color="black", markersize=80, zorder=6)
spf.plot(ax=ax, color="white", markersize=22, zorder=7)
for _, r in spf.iterrows():
    ax.annotate(r.sample_id, (r.geometry.x, r.geometry.y), xytext=(5, 5),
                textcoords="offset points", fontsize=8, fontweight="bold")
leg = [Patch(facecolor=COLORS[int(r.zone) - 1], edgecolor="white",
             label=f"Зона {int(r.zone)} — {r.level}: {r.area_ha} га (NDVI {r.ndvi_mean:.3f})")
       for _, r in stats.sort_values("zone").iterrows()]
leg.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
                  markersize=10, label="Точка відбору (композит 20 уколів)"))
ax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=10.5, frameon=False)
ax.set_title("Поле П1 (Софіївка, 86 га) — зони продуктивності 2025 (фінал)\n"
             "5 зон • точний малюнок NDVI + прибрані відростки/дрібні шматки < 3%",
             fontsize=13)
ax.set_aspect("equal"); ax.axis("off")
plt.tight_layout()
plt.savefig(gc.p(FIG, "07_final_zones.png"), dpi=130, bbox_inches="tight")
plt.close()
print("saved figures/07_final_zones.png")
