"""
STEP 6 - Final simplified / smoothed zones (client revision).

Changes requested:
  * zones simpler and MORE smoothed (natural-looking boundaries);
  * drop small zone fragments smaller than MIN_FRAC of the field
    (~3%; merged into the surrounding zone);
  * keep 5 zones; re-place soil-sampling points on the new zones;
  * emit QGIS .qml styles so layers open pre-coloured;
  * render a preview straight from the saved shapefile.

Partition-preserving smoothing: sieve small clumps at raster level, then
one-hot upsample + Gaussian + argmax (curved, natural boundaries that still
tile the field with no gaps/overlaps).
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

K = 5
MIN_FRAC = 0.03          # drop/merge zone fragments smaller than 3% of field
EXTRA_SIGMA = 3.0        # extra pre-cluster smoothing (px) for simpler zones
UPSCALE = 3              # boundary-smoothing supersampling
BND_SIGMA = 6.0          # gaussian sigma (upsampled px) for natural edges
HA_PER_SAMPLE = 5.0
CORES = 20

ZONE_DIR = gc.p(gc.ROOT, "shapefiles", "zones")
SAMP_DIR = gc.p(gc.ROOT, "shapefiles", "sampling")
QML_DIR = gc.p(gc.ROOT, "shapefiles")
FIG = gc.p(gc.ROOT, "figures")

COLORS = ["#d7191c", "#fdae61", "#ffffbf", "#a6d96a", "#1a9641"]
LEVELS_UA = ["дуже низька", "низька", "середня", "висока", "дуже висока"]

# ---- load cleaned 2025 canopy + grid ------------------------------------
src = rasterio.open(gc.p(gc.ROOT, "rasters", "ndvi_2025_clean.tif"))
arr = src.read(1)
transform = src.transform
mask = ~np.isnan(arr)
H, W = arr.shape
field_ha = 86.05
min_px = int(MIN_FRAC * field_ha * 1e4 / (gc.RES ** 2))


def sieve(lab, mask, min_px):
    lab = lab.copy()
    for _ in range(20):
        changed = 0
        for cls in np.unique(lab[mask]):
            comp, n = ndi.label((lab == cls) & mask)
            if n == 0:
                continue
            sizes = ndi.sum(np.ones_like(comp), comp, range(1, n + 1))
            for i, sz in enumerate(sizes, start=1):
                if sz < min_px:
                    blob = comp == i
                    dil = ndi.binary_dilation(blob, iterations=2) & mask & ~blob
                    neigh = lab[dil]
                    if neigh.size:
                        v, c = np.unique(neigh, return_counts=True)
                        lab[blob] = v[c.argmax()]
                        changed += 1
        if not changed:
            break
    return lab


# ---- cluster on extra-smoothed NDVI -------------------------------------
sm = ndi.gaussian_filter(np.where(mask, arr, np.nanmean(arr)), EXTRA_SIGMA)
km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(sm[mask].reshape(-1, 1))
lab = np.zeros((H, W), "int32")
lab[mask] = km.labels_ + 1
order = sorted(range(1, K + 1), key=lambda c: arr[(lab == c) & mask].mean())
remap = {o: n for n, o in enumerate(order, 1)}
lab = np.vectorize(lambda v: remap.get(v, 0))(lab).astype("int32")
lab[~mask] = 0

# remove small fragments
lab = sieve(lab, mask, min_px)

# ---- natural boundary smoothing: one-hot upsample + gaussian + argmax ----
big = np.zeros((H * UPSCALE, W * UPSCALE), "int32")
prob = np.zeros((K, H * UPSCALE, W * UPSCALE), "float32")
for zi in range(1, K + 1):
    oneh = (lab == zi).astype("float32")
    up = np.kron(oneh, np.ones((UPSCALE, UPSCALE), "float32"))
    prob[zi - 1] = ndi.gaussian_filter(up, BND_SIGMA)
maskbig = np.kron(mask, np.ones((UPSCALE, UPSCALE))).astype(bool)
big[maskbig] = (prob.argmax(axis=0) + 1)[maskbig]
big[~maskbig] = 0
# one more sieve at fine scale to kill any slivers the smoothing produced
big = sieve(big, maskbig, min_px * UPSCALE * UPSCALE)

bigtrans = from_origin(transform.c, transform.f, gc.RES / UPSCALE, gc.RES / UPSCALE)

# ---- vectorise + light simplify -----------------------------------------
recs = []
for geom, val in rio_shapes(big, mask=maskbig, transform=bigtrans):
    if int(val) == 0:
        continue
    recs.append({"zone": int(val), "geometry": shape(geom)})
gdf = gpd.GeoDataFrame(recs, crs=gc.UTM).dissolve(by="zone", as_index=False)
gdf["geometry"] = gdf.geometry.simplify(3).buffer(4).buffer(-4)
gdf = gdf[gdf.geometry.area > 0]

# ---- stats --------------------------------------------------------------
# labels back on the original grid for stats/sampling
labf = np.zeros((H, W), "int32")
labf[mask] = big[::UPSCALE, ::UPSCALE][mask]
fmean = float(arr[mask].mean())
rows = []
for _, r in gdf.sort_values("zone").iterrows():
    z = int(r["zone"])
    zm = (labf == z) & mask
    v = arr[zm]
    rows.append(dict(zone=z, level=LEVELS_UA[z - 1],
                     area_ha=round(r.geometry.area / 1e4, 2),
                     ndvi_mean=round(float(v.mean()), 4),
                     ndvi_std=round(float(v.std()), 4),
                     rel_pct=round((v.mean() - fmean) / fmean * 100, 2),
                     n_parts=len(list(r.geometry.geoms)) if r.geometry.geom_type == "MultiPolygon" else 1,
                     color=COLORS[z - 1]))
stats = pd.DataFrame(rows)
gdf = gdf.merge(stats.drop(columns="color"), on="zone")
gdf["color"] = gdf["zone"].map(lambda z: COLORS[z - 1])
gc.save_vec(gdf, gc.p(ZONE_DIR, "zones_2025"))
print("FINAL zones:")
print(stats[["zone", "level", "area_ha", "ndvi_mean", "rel_pct", "n_parts"]].to_string(index=False))
print("total parts:", int(stats["n_parts"].sum()), "(was ~30+ before)")

# ---- re-place sampling points on new zones ------------------------------
dem = np.load(gc.p(gc.WORK, "out", "dem_arr.npy"))
dtr = Affine(*np.load(gc.p(gc.WORK, "out", "dem_trans.npy")))
demN = np.full((H, W), np.nan, "float32")
reproject(dem, demN, src_transform=dtr, src_crs=f"EPSG:{gc.UTM}",
          dst_transform=transform, dst_crs=f"EPSG:{gc.UTM}", resampling=Resampling.bilinear)

pts, sid = [], 0
for z in range(1, K + 1):
    zm = (labf == z) & mask
    n = max(1, int(round(zm.sum() * gc.RES ** 2 / 1e4 / HA_PER_SAMPLE)))
    core = ndi.binary_erosion(zm, iterations=5)
    if core.sum() < n:
        core = ndi.binary_erosion(zm, iterations=2)
    if core.sum() < n:
        core = zm
    rr, cc = np.where(core)
    xy = np.column_stack([transform.c + (cc + .5) * transform.a,
                          transform.f + (rr + .5) * transform.e])
    centers = [xy.mean(0)] if n == 1 else KMeans(n, n_init=5, random_state=0).fit(xy).cluster_centers_
    for cx, cy in centers:
        j = ((xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2).argmin()
        ri, ci = rr[j], cc[j]
        sid += 1
        pts.append(dict(sample_id=f"S{sid:02d}", zone=int(z),
                        ndvi_2025=round(float(arr[ri, ci]), 4),
                        elev_m=round(float(demN[ri, ci]), 1) if not np.isnan(demN[ri, ci]) else None,
                        n_cores=CORES, depth_cm="0-30 (P,K,pH,OM); +30-60 N-NO3",
                        geometry=Point(xy[j, 0], xy[j, 1])))
sp = gpd.GeoDataFrame(pts, crs=gc.UTM)
gc.save_vec(sp, gc.p(SAMP_DIR, "soil_sampling_points"))
stats["n_samples"] = stats["zone"].map(sp.groupby("zone").size())
stats.to_csv(gc.p(gc.ROOT, "report", "zone_stats.csv"), index=False)
print("sampling points:", len(sp), sp.groupby("zone").size().to_dict())


# ---- QGIS .qml styles ----------------------------------------------------
def hexrgb(h):
    h = h.lstrip("#")
    return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},255"

def categorized_qml(attr, cats, out, outline="255,255,255,255", ow="0.26"):
    syms, catx = [], []
    for i, (val, lab, hx) in enumerate(cats):
        catx.append(f'<category value="{val}" symbol="{i}" label="{lab}" render="true"/>')
        syms.append(f'''<symbol type="fill" name="{i}"><layer class="SimpleFill">
      <Option type="Map">
        <Option name="color" type="QString" value="{hexrgb(hx)}"/>
        <Option name="outline_color" type="QString" value="{outline}"/>
        <Option name="outline_width" type="QString" value="{ow}"/>
        <Option name="outline_width_unit" type="QString" value="MM"/>
        <Option name="style" type="QString" value="solid"/>
      </Option></layer></symbol>''')
    xml = f'''<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28" styleCategories="Symbology">
  <renderer-v2 type="categorizedSymbol" attr="{attr}" forceraster="0" symbollevels="0" enableorderby="0">
    <categories>
      {''.join(catx)}
    </categories>
    <symbols>
      {''.join(syms)}
    </symbols>
  </renderer-v2>
  <blendMode>0</blendMode>
</qgis>'''
    open(out, "w", encoding="utf-8").write(xml)

# zones style
categorized_qml("zone",
    [(int(r.zone), f"Зона {int(r.zone)} — {r.level} ({r.area_ha} га)", COLORS[int(r.zone)-1])
     for _, r in stats.sort_values("zone").iterrows()],
    gc.p(ZONE_DIR, "zones_2025.qml"))
# relief styles (reuse colors from their shapefiles)
for name, attr in [("relief/relief_elevation_zones", "elev_cls"),
                   ("relief/relief_slope_zones", "slope_cls"),
                   ("relief/relief_tpi_zones", "tpi_cls")]:
    g = gpd.read_file(gc.p(gc.ROOT, "shapefiles", name + ".shp"))
    cats = [(int(r[attr]), str(r["label"]), str(r["color"]))
            for _, r in g.sort_values(attr).iterrows()]
    categorized_qml(attr, cats, gc.p(gc.ROOT, "shapefiles", name + ".qml"), ow="0")
# sampling points style (single black marker)
open(gc.p(SAMP_DIR, "soil_sampling_points.qml"), "w", encoding="utf-8").write(
    '''<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28" styleCategories="Symbology">
  <renderer-v2 type="singleSymbol">
    <symbols><symbol type="marker" name="0"><layer class="SimpleMarker">
      <Option type="Map">
        <Option name="color" type="QString" value="0,0,0,255"/>
        <Option name="name" type="QString" value="circle"/>
        <Option name="outline_color" type="QString" value="255,255,255,255"/>
        <Option name="outline_width" type="QString" value="0.4"/>
        <Option name="size" type="QString" value="3"/>
      </Option></layer></symbol></symbols>
  </renderer-v2>
</qgis>''')
print("QML styles written")


# ---- preview from the SAVED shapefile ------------------------------------
zf = gpd.read_file(gc.p(ZONE_DIR, "zones_2025.shp")).sort_values("zone")
spf = gpd.read_file(gc.p(SAMP_DIR, "soil_sampling_points.shp"))
cont = gpd.read_file(gc.p(gc.ROOT, "shapefiles", "relief", "relief_contours.shp"))
fig, ax = plt.subplots(figsize=(12, 11))
for _, r in zf.iterrows():
    gpd.GeoSeries([r.geometry], crs=zf.crs).plot(ax=ax, color=r["color"],
                                                 edgecolor="white", linewidth=1.4)
cont.plot(ax=ax, color="#555", linewidth=0.4, alpha=0.5)
spf.plot(ax=ax, color="black", markersize=80, zorder=6)
spf.plot(ax=ax, color="white", markersize=22, zorder=7)
for _, r in spf.iterrows():
    ax.annotate(r["sample_id"], (r.geometry.x, r.geometry.y), xytext=(5, 5),
                textcoords="offset points", fontsize=8, fontweight="bold")
leg = [Patch(facecolor=COLORS[int(r.zone)-1], edgecolor="white",
             label=f"Зона {int(r.zone)} — {r.level}: {r.area_ha} га  (NDVI {r.ndvi_mean:.3f})")
       for _, r in stats.sort_values("zone").iterrows()]
leg.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
                  markersize=10, label="Точка відбору (композит 20 уколів)"))
ax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, -0.02),
          fontsize=10.5, frameon=False)
ax.set_title("Поле П1 (Софіївка, 86 га) — фінальні зони (спрощені та згладжені)\n"
             "5 зон • дрібні шматки < 3% злиті • з горизонталями рельєфу",
             fontsize=13)
ax.set_aspect("equal"); ax.axis("off")
plt.tight_layout()
plt.savefig(gc.p(FIG, "07_final_zones.png"), dpi=130, bbox_inches="tight")
plt.close()
print("saved figures/07_final_zones.png")
