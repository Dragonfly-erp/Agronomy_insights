"""
STEP 1 - Extract the clean vegetation-distribution pattern.

Goal (from the client): "maximally discard irrelevant data - possible machinery
passes, tillage - and isolate the clean vegetation-distribution pattern so the
field can be zoned correctly."

Cleaning cascade (per year), all in UTM 36N on a 5 m grid:
  1. rasterise NDVI polygons -> 5 m grid
  2. clip to the INNER CORE (field eroded by 18 m) -> drops headlands, field-edge
     grass/tree strips and turning zones (the bright edge artefacts seen in 2024)
  3. robust outlier removal (MAD): pixels further than 3.5*MAD from the field
     median are dropped as anomalies (roads, wet spots, residual edge strips)
  4. de-stripe / de-tramline: a small median filter removes thin linear features
     (wheel tracks, technological tracks, tillage striping) while preserving
     broad zones
  5. gap-fill (nearest) so the surface is continuous for smoothing/clustering
  6. low-pass smoothing (Gaussian) -> the clean, broad productivity pattern
  7. re-mask to the core and save cleaned GeoTIFF + before/after PNG.

Outputs: rasters/ndvi_<yr>_clean.tif, figures/02_clean_<yr>.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import rasterio
from rasterio.transform import from_origin

import geo_common as gc

os.makedirs(gc.p(gc.ROOT, "rasters"), exist_ok=True)
os.makedirs(gc.p(gc.ROOT, "figures"), exist_ok=True)

field = gc.field_geometry()
core = field.buffer(-gc.EDGE_BUFFER)
if core.geom_type == "MultiPolygon":
    core = max(core.geoms, key=lambda g: g.area)

# shared grid for the whole field (so both years align pixel-for-pixel)
transform, W, H, bounds = gc.grid_for(field)
core_mask = gc.mask_from_geom(core, transform, W, H)


def nearest_fill(arr, valid):
    """Fill NaN by nearest valid neighbour (for smoothing continuity)."""
    idx = ndi.distance_transform_edt(~valid, return_distances=False,
                                     return_indices=True)
    return arr[tuple(idx)]


def clean_year(year):
    gdf = gc.read_ndvi(year)
    raw = gc.rasterize_field(gdf, "NDVI", transform, W, H)

    # (2) clip to inner core
    a = np.where(core_mask & ~np.isnan(raw), raw, np.nan)

    # (3) robust MAD outlier removal
    v = a[~np.isnan(a)]
    med = np.median(v)
    mad = np.median(np.abs(v - med)) or 1e-6
    lo, hi = med - 3.5 * 1.4826 * mad, med + 3.5 * 1.4826 * mad
    n_out = int(np.nansum((a < lo) | (a > hi)))
    a = np.where((a < lo) | (a > hi), np.nan, a)

    valid0 = ~np.isnan(a)
    # (5) fill then (4) de-stripe with a median filter (25 m = 5 px window)
    filled = nearest_fill(np.where(valid0, a, med), valid0)
    destripe = ndi.median_filter(filled, size=5, mode="nearest")

    # (6) low-pass smoothing (Gaussian sigma 2 px = 10 m)
    smooth = ndi.gaussian_filter(destripe, sigma=2.0, mode="nearest")

    # keep only the core
    clean = np.where(core_mask, smooth, np.nan).astype("float32")

    # ---- save GeoTIFF
    out_tif = gc.p(gc.ROOT, "rasters", f"ndvi_{year}_clean.tif")
    with rasterio.open(out_tif, "w", driver="GTiff", height=H, width=W,
                       count=1, dtype="float32", crs=f"EPSG:{gc.UTM}",
                       transform=transform, nodata=np.nan) as dst:
        dst.write(clean, 1)

    # also keep the masked-but-unsmoothed core for reference/stats
    core_raw = np.where(core_mask, np.where(valid0, a, np.nan), np.nan)

    # ---- before/after figure
    fig, ax = plt.subplots(1, 3, figsize=(19, 6))
    vmin, vmax = np.nanpercentile(raw, 2), np.nanpercentile(raw, 98)
    for x, dat, t in [(0, raw, f"RAW {year} (all pixels)"),
                      (1, core_raw, f"core + outliers removed ({n_out} px)"),
                      (2, clean, f"CLEAN {year} (de-striped + smoothed)")]:
        im = ax[x].imshow(dat, cmap="RdYlGn", vmin=vmin, vmax=vmax)
        ax[x].set_title(t)
        ax[x].axis("off")
        plt.colorbar(im, ax=ax[x], fraction=0.046)
    plt.tight_layout()
    plt.savefig(gc.p(gc.ROOT, "figures", f"02_clean_{year}.png"),
                dpi=110, bbox_inches="tight")
    plt.close()

    cv = np.nanstd(clean) / np.nanmean(clean) * 100
    print(f"[{year}] core px={int(np.nansum(core_mask))}  outliers removed={n_out}"
          f"  clean mean={np.nanmean(clean):.4f} std={np.nanstd(clean):.4f}"
          f"  CV={cv:.1f}%  -> {out_tif}")
    return clean


if __name__ == "__main__":
    print(f"Field {field.area/1e4:.2f} ha | core {core.area/1e4:.2f} ha | "
          f"grid {W}x{H} @ {gc.RES} m")
    for yr in ("2024", "2025"):
        clean_year(yr)
    # persist grid metadata for later steps
    np.save(gc.p(gc.WORK, "out", "grid_meta.npy"),
            np.array([bounds[0], bounds[1], bounds[2], bounds[3], gc.RES, W, H]))
    print("done.")
