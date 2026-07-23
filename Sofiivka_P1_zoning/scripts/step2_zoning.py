"""
STEP 2 - Management-zone delineation.

For each cleaned NDVI layer:
  * choose the zone count k in 4..6 using silhouette + within-cluster variance
    (the client asked for "5 or 6, or 4 - whatever comes out");
  * k-means on the cleaned NDVI value -> k classes;
  * post-process for machine-usable zones: majority filter + sieve out clumps
    smaller than MIN_ZONE_HA so no tiny slivers remain;
  * re-label zones by ascending mean NDVI (Zone 1 = lowest vigour ...
    Zone k = highest), which is the order a VR fertiliser map needs;
  * vectorise, dissolve per zone, lightly smooth boundaries;
  * write shapefile (+WGS84 twin) with per-zone statistics.

Layers:
  2025 canopy  -> PRIMARY productivity zones  (shapefiles/zones/zones_2025_*)
  2024 bare-soil -> reference SOIL-SURFACE pattern (shapefiles/zones/soil_2024_*)

NOTE: the two dates are spatially uncorrelated (r~0.04), so they are NOT
blended - see the report. 2025 is the layer to drive VRA and soil sampling.
"""
import os
import numpy as np
import rasterio
from scipy import ndimage as ndi
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import geopandas as gpd
from shapely.geometry import shape
from rasterio.features import shapes as rio_shapes

import geo_common as gc

MIN_ZONE_HA = 0.5          # merge clumps smaller than this into neighbours
ZONE_DIR = gc.p(gc.ROOT, "shapefiles", "zones")
os.makedirs(ZONE_DIR, exist_ok=True)

PALETTE = {   # low -> high vigour, for the report/QGIS
    1: "#d7191c", 2: "#fdae61", 3: "#ffffbf", 4: "#a6d96a",
    5: "#1a9641", 6: "#006837",
}


def choose_k(values, kmin=4, kmax=6):
    """Pick k in [kmin,kmax] by best silhouette; report elbow too."""
    x = values.reshape(-1, 1)
    samp = x if len(x) <= 6000 else x[np.random.RandomState(0).choice(len(x), 6000, replace=False)]
    stats = {}
    for k in range(kmin, kmax + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(x)
        sil = silhouette_score(samp, KMeans(n_clusters=k, n_init=10,
                               random_state=0).fit_predict(samp))
        stats[k] = (sil, km.inertia_)
    best = max(stats, key=lambda k: stats[k][0])
    print("    k selection (silhouette | inertia):",
          {k: (round(v[0], 3), round(v[1], 4)) for k, v in stats.items()},
          "-> k =", best)
    return best, stats


def sieve(labels, mask, min_px):
    """Remove clumps < min_px by reassigning them to the majority neighbour.
    Iterates until stable."""
    lab = labels.copy()
    for _ in range(12):
        changed = 0
        for cls in np.unique(lab[mask]):
            comp, n = ndi.label((lab == cls) & mask)
            for i in range(1, n + 1):
                blob = comp == i
                if blob.sum() < min_px:
                    dil = ndi.binary_dilation(blob) & mask & ~blob
                    neigh = lab[dil]
                    if neigh.size:
                        vals, cnts = np.unique(neigh, return_counts=True)
                        lab[blob] = vals[cnts.argmax()]
                        changed += 1
        if not changed:
            break
    return lab


def zonify(tif, layer_name, out_prefix, is_primary, fixed_k=None):
    with rasterio.open(tif) as src:
        arr = src.read(1)
        transform = src.transform
    mask = ~np.isnan(arr)
    vals = arr[mask]

    if fixed_k is None:
        k, stats = choose_k(vals)
    else:
        _, stats = choose_k(vals)
        k = fixed_k
        print(f"    using requested k = {k}")
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(vals.reshape(-1, 1))
    lab = np.full(arr.shape, 0, dtype="int32")
    lab[mask] = km.labels_ + 1

    # relabel by ascending mean NDVI
    order = sorted(range(1, k + 1),
                   key=lambda c: arr[(lab == c) & mask].mean())
    remap = {old: new for new, old in enumerate(order, start=1)}
    lab2 = np.zeros_like(lab)
    for old, new in remap.items():
        lab2[lab == old] = new

    # majority filter (smooth) then sieve small clumps
    lab3 = lab2.copy()
    filt = ndi.median_filter(lab2, size=3, mode="nearest")
    lab3[mask] = filt[mask]
    min_px = int(MIN_ZONE_HA * 1e4 / (gc.RES ** 2))
    lab3 = sieve(lab3, mask, min_px)
    lab3[~mask] = 0

    # vectorise + dissolve
    recs = []
    for geom, val in rio_shapes(lab3.astype("int32"), mask=mask, transform=transform):
        if int(val) == 0:
            continue
        recs.append({"zone": int(val), "geometry": shape(geom)})
    gdf = gpd.GeoDataFrame(recs, crs=gc.UTM)
    gdf = gdf.dissolve(by="zone", as_index=False)
    # light boundary smoothing: close gaps, round corners, simplify
    gdf["geometry"] = gdf.geometry.buffer(6).buffer(-6).simplify(4)
    gdf = gdf[gdf.geometry.area > 0]

    # per-zone stats from the cleaned raster
    rows = []
    for _, r in gdf.iterrows():
        z = int(r["zone"])
        zmask = (lab3 == z) & mask
        zv = arr[zmask]
        rows.append(dict(
            zone=z,
            ndvi_mean=round(float(zv.mean()), 4),
            ndvi_min=round(float(zv.min()), 4),
            ndvi_max=round(float(zv.max()), 4),
            ndvi_std=round(float(zv.std()), 4),
            area_ha=round(float(r.geometry.area / 1e4), 2),
            px=int(zmask.sum()),
            color=PALETTE.get(z, "#888888"),
        ))
    import pandas as pd
    sdf = pd.DataFrame(rows).set_index("zone")
    gdf = gdf.merge(sdf, left_on="zone", right_index=True)
    # relative productivity vs field mean
    fmean = float(vals.mean())
    gdf["rel_pct"] = ((gdf["ndvi_mean"] - fmean) / fmean * 100).round(1)
    lo, hi = gdf["ndvi_mean"].min(), gdf["ndvi_mean"].max()
    labels_txt = {1: "very low", 2: "low", 3: "moderate",
                  4: "high", 5: "very high", 6: "highest"}
    # map textual level relative to k
    gdf["level"] = gdf["zone"].map(
        lambda z: ["low", "below-avg", "average", "above-avg", "high", "very high"][
            int(round((z - 1) / (k - 1) * 5))])
    gdf["layer"] = layer_name

    prefix = gc.p(ZONE_DIR, out_prefix)
    gc.save_vec(gdf.to_crs(gc.UTM), prefix)
    print(f"    {layer_name}: {k} zones, "
          f"areas(ha)={list(gdf.sort_values('zone')['area_ha'])} -> {prefix}.shp")

    # save label raster for figures
    np.save(gc.p(gc.WORK, "out", f"zones_{out_prefix}.npy"), lab3)
    return gdf, k, lab3, stats


if __name__ == "__main__":
    t2025 = gc.p(gc.ROOT, "rasters", "ndvi_2025_clean.tif")
    print("PRIMARY  - 2025 canopy productivity zones (k=5)")
    zonify(t2025, "2025 canopy productivity", "zones_2025", True, fixed_k=5)
    print("ALT - 2025 canopy, k=4")
    zonify(t2025, "2025 canopy productivity (4-zone)", "zones_2025_k4", False, fixed_k=4)
    print("ALT - 2025 canopy, k=6")
    zonify(t2025, "2025 canopy productivity (6-zone)", "zones_2025_k6", False, fixed_k=6)
    print("REFERENCE - 2024 bare-soil surface pattern (k=5)")
    zonify(gc.p(gc.ROOT, "rasters", "ndvi_2024_clean.tif"),
           "2024 bare-soil surface", "soil_2024", False, fixed_k=5)
    print("done.")
