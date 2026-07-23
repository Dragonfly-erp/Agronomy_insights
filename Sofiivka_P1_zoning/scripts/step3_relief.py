"""
STEP 3 - Terrain / relief from the field contour.

Uses the free Copernicus GLO-30 DEM (ESA, 30 m, downloaded from the public AWS
Open-Data bucket) for the 1x1 deg tile N50/E031 that contains the field.

Pipeline:
  * window the tile to the field bbox (+buffer), reproject to UTM 36N and
    resample to a 10 m grid (cubic) for smooth contours;
  * clip to the field outline;
  * derive elevation, slope (deg & %), aspect, and TPI (topographic position
    index: + = ridge/dry, - = depression/wet-accumulation);
  * export as SHAPEFILES (the client asked for shape format):
      - relief_contours       : 1 m contour lines (elev attribute)
      - relief_elevation_zones : 5 elevation classes (polygons)
      - relief_slope_zones     : slope classes (polygons)
      - relief_tpi_zones       : depression / slope / ridge (polygons)
  * save the clipped DEM GeoTIFF + a hillshade PNG figure.
"""
import os
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.features import shapes as rio_shapes
import geopandas as gpd
from shapely.geometry import shape, LineString, MultiLineString
from shapely.ops import unary_union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import geo_common as gc

DEM_SRC = gc.p(gc.WORK, "dem", "cop30_N50_E031.tif")
RELIEF_DIR = gc.p(gc.ROOT, "shapefiles", "relief")
os.makedirs(RELIEF_DIR, exist_ok=True)
DEM_RES = 10.0   # target grid, m

field = gc.field_geometry()
fg_wgs = gpd.GeoSeries([field], crs=gc.UTM).to_crs(gc.WGS)
minx, miny, maxx, maxy = fg_wgs.total_bounds


# ---- 1. window + reproject to UTM 10 m ---------------------------------
with rasterio.open(DEM_SRC) as src:
    win = from_bounds(minx - 0.004, miny - 0.004, maxx + 0.004, maxy + 0.004,
                      src.transform)
    dem = src.read(1, window=win).astype("float32")
    wtrans = src.window_transform(win)
    src_crs = src.crs

dst_trans, dw, dh = calculate_default_transform(
    src_crs, f"EPSG:{gc.UTM}", dem.shape[1], dem.shape[0],
    left=minx - 0.004, bottom=miny - 0.004, right=maxx + 0.004, top=maxy + 0.004,
    resolution=DEM_RES)
demu = np.empty((dh, dw), "float32")
reproject(dem, demu, src_transform=wtrans, src_crs=src_crs,
          dst_transform=dst_trans, dst_crs=f"EPSG:{gc.UTM}",
          resampling=Resampling.cubic)

# clip to field (+ small buffer so contours reach the edge)
fmask = gc.mask_from_geom(field.buffer(10), dst_trans, dw, dh)
demu = np.where(fmask, demu, np.nan)

# save clipped DEM
os.makedirs(gc.p(gc.ROOT, "rasters"), exist_ok=True)
with rasterio.open(gc.p(gc.ROOT, "rasters", "dem_field_10m.tif"), "w",
                   driver="GTiff", height=dh, width=dw, count=1,
                   dtype="float32", crs=f"EPSG:{gc.UTM}", transform=dst_trans,
                   nodata=np.nan) as dst:
    dst.write(demu, 1)

# ---- 2. derivatives -----------------------------------------------------
from scipy import ndimage as ndi
# nearest-neighbour extrapolate outside the field so gradients have no false
# cliff at the boundary, then mask back
valid = ~np.isnan(demu)
idx = ndi.distance_transform_edt(~valid, return_distances=False, return_indices=True)
dem_fill = demu[tuple(idx)]
gy, gx = np.gradient(dem_fill, DEM_RES)
slope_deg = np.degrees(np.arctan(np.sqrt(gx**2 + gy**2)))
slope_pct = np.tan(np.radians(slope_deg)) * 100
aspect = (np.degrees(np.arctan2(-gx, gy)) % 360)
tpi = dem_fill - ndi.uniform_filter(dem_fill, size=9)  # 90 m neighbourhood
for a in (slope_deg, slope_pct, aspect, tpi):
    a[~fmask] = np.nan

zmin, zmax = np.nanmin(demu), np.nanmax(demu)
print(f"DEM 10 m grid {dw}x{dh} | elevation {zmin:.1f}-{zmax:.1f} m "
      f"(range {zmax-zmin:.1f} m) | slope max {np.nanmax(slope_pct):.1f}%")


# ---- 3a. contour lines (1 m) -------------------------------------------
xs = dst_trans.c + (np.arange(dw) + 0.5) * dst_trans.a
ys = dst_trans.f + (np.arange(dh) + 0.5) * dst_trans.e
X, Y = np.meshgrid(xs, ys)
lo = np.floor(zmin) + 1
hi = np.ceil(zmax)
levels = np.arange(lo, hi, 1.0)
cs = plt.contour(X, Y, np.where(np.isnan(demu), -9999, demu), levels=levels)
recs = []
fbuf = field.buffer(0)
for lev, segs in zip(cs.levels, cs.allsegs):
    for seg in segs:
        if len(seg) < 2:
            continue
        ln = LineString(seg)
        ln = ln.intersection(fbuf)
        if ln.is_empty:
            continue
        geoms = ln.geoms if ln.geom_type == "MultiLineString" else [ln]
        for g in geoms:
            if g.geom_type == "LineString" and g.length > 5:
                recs.append({"elev_m": float(lev), "geometry": g})
plt.close()
gdf_c = gpd.GeoDataFrame(recs, crs=gc.UTM)
gc.save_vec(gdf_c, gc.p(RELIEF_DIR, "relief_contours"))
print(f"  contours: {len(gdf_c)} lines at {levels[0]:.0f}..{levels[-1]:.0f} m (1 m)")


# ---- 3b. helper: classify raster -> dissolved polygons ------------------
def classify_to_polys(data, edges, names, out, colors, valcol="class"):
    cls = np.digitize(data, edges).astype("float32")
    # smooth at raster level (majority filter) -> clean, non-overlapping partition
    filled = cls.copy()
    filled[np.isnan(cls)] = -1
    sm = ndi.median_filter(filled, size=3, mode="nearest")
    cls = np.where(fmask & ~np.isnan(data), sm, np.nan)
    intcls = np.where(np.isnan(cls), -1, cls).astype("int32")
    rr = []
    for geom, val in rio_shapes(intcls, mask=(intcls >= 0), transform=dst_trans):
        rr.append({valcol: int(val), "geometry": shape(geom)})
    g = gpd.GeoDataFrame(rr, crs=gc.UTM).dissolve(by=valcol, as_index=False)
    # simplify only (no buffering) so classes stay a partition of the field
    g["geometry"] = g.geometry.simplify(4)
    g = g[g.geometry.area > 0]
    g = g.clip(field)
    g["label"] = g[valcol].map(lambda i: names[int(i)] if int(i) < len(names) else "na")
    g["color"] = g[valcol].map(lambda i: colors[int(i)] if int(i) < len(colors) else "#888")
    g["area_ha"] = (g.geometry.area / 1e4).round(2)
    gc.save_vec(g, out)
    return g


# elevation zones (5 quantile classes)
qs = np.nanquantile(demu, [0.2, 0.4, 0.6, 0.8])
gz = classify_to_polys(demu, qs,
        [f"<{qs[0]:.1f}m", f"{qs[0]:.1f}-{qs[1]:.1f}m", f"{qs[1]:.1f}-{qs[2]:.1f}m",
         f"{qs[2]:.1f}-{qs[3]:.1f}m", f">{qs[3]:.1f}m"],
        gc.p(RELIEF_DIR, "relief_elevation_zones"),
        ["#2b83ba", "#abdda4", "#ffffbf", "#fdae61", "#d7191c"], "elev_cls")
print(f"  elevation zones: {len(gz)} polys, areas={list(gz.sort_values('elev_cls')['area_ha'])}")

# slope zones (agronomic breaks: <1,1-2,2-3,3-5,>5 %)
gs = classify_to_polys(slope_pct, np.array([1, 2, 3, 5.0]),
        ["<1% flat", "1-2% gentle", "2-3% moderate", "3-5% strong", ">5% steep"],
        gc.p(RELIEF_DIR, "relief_slope_zones"),
        ["#f7fcf5", "#c7e9c0", "#74c476", "#fe9929", "#cc4c02"], "slope_cls")
print(f"  slope zones: {len(gs)} polys, areas={list(gs.sort_values('slope_cls')['area_ha'])}")

# TPI zones (depression / mid-slope / ridge)
tstd = np.nanstd(tpi)
gt = classify_to_polys(tpi, np.array([-0.5*tstd, 0.5*tstd]),
        ["depression (wet/accum.)", "mid-slope", "ridge/knoll (dry/erosive)"],
        gc.p(RELIEF_DIR, "relief_tpi_zones"),
        ["#2166ac", "#f7f7f7", "#b2182b"], "tpi_cls")
print(f"  TPI zones: {len(gt)} polys, areas={list(gt.sort_values('tpi_cls')['area_ha'])}")

# save arrays for cross-analysis / figures
np.save(gc.p(gc.WORK, "out", "dem_arr.npy"), demu)
np.save(gc.p(gc.WORK, "out", "dem_trans.npy"),
        np.array([dst_trans.a, dst_trans.b, dst_trans.c,
                  dst_trans.d, dst_trans.e, dst_trans.f]))
print("done.")
