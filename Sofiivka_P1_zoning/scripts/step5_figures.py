"""
STEP 5 - Final report figures.
  05_main_zones_sampling.png : primary 2025 zones + soil-sampling points + legend
  06_validation.png          : NDVI-elevation relationship & per-zone summary
  04_relief.png (rebuild)    : elevation+contours / TPI / slope  (fixed colours)
"""
import os
import numpy as np
import rasterio
from rasterio.transform import Affine, from_origin
from rasterio.warp import reproject, Resampling
import geopandas as gpd
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import geo_common as gc

FIG = gc.p(gc.ROOT, "figures")
SHP = gc.p(gc.ROOT, "shapefiles")


def plot_gdf_colored(ax, gdf, catcol, edge="white", lw=0.7, alpha=1.0):
    for _, r in gdf.iterrows():
        gpd.GeoSeries([r.geometry], crs=gdf.crs).plot(
            ax=ax, color=r["color"], edgecolor=edge, linewidth=lw, alpha=alpha)


# ---------- 05 main map ----------
zones = gpd.read_file(gc.p(SHP, "zones", "zones_2025.shp")).sort_values("zone")
pts = gpd.read_file(gc.p(SHP, "sampling", "soil_sampling_points.shp"))
stats = pd.read_csv(gc.p(gc.ROOT, "report", "zone_stats.csv"))

fig, ax = plt.subplots(figsize=(12, 11))
plot_gdf_colored(ax, zones, "zone", edge="white", lw=1.2)
pts.plot(ax=ax, color="black", marker="o", markersize=90, zorder=5)
pts.plot(ax=ax, color="white", marker="o", markersize=30, zorder=6)
for _, r in pts.iterrows():
    ax.annotate(r["sample_id"], (r.geometry.x, r.geometry.y),
                xytext=(6, 6), textcoords="offset points",
                fontsize=8, fontweight="bold", zorder=7)
lvl = {1: "very low", 2: "low", 3: "moderate", 4: "high", 5: "very high"}
leg = []
for _, s in stats.iterrows():
    z = int(s["zone"])
    leg.append(Patch(facecolor=zones[zones.zone == z]["color"].iloc[0],
                     edgecolor="white",
                     label=(f"Zone {z} - {lvl.get(z,'')}: {s['area_ha']} ha "
                            f"| NDVI {s['ndvi_mean']:.3f} | {s['elev_mean_m']:.1f} m "
                            f"| {int(s['n_samples'])} sample(s)")))
leg.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
                  markersize=10, label="Soil-sampling point (20-core composite)"))
ax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, -0.02),
          fontsize=10, frameon=False, ncol=1)
ax.set_title("Field П1 (Sofiivka, ~86 ha) - management zones 2025 (canopy NDVI)\n"
             "Zone 1 = lowest vigour  ...  Zone 5 = highest vigour  |  variable-rate & sampling basis",
             fontsize=13)
ax.set_aspect("equal"); ax.axis("off")
plt.tight_layout()
plt.savefig(gc.p(FIG, "05_main_zones_sampling.png"), dpi=130, bbox_inches="tight")
plt.close()
print("saved 05_main_zones_sampling.png")


# ---------- 06 validation ----------
gm = np.load(gc.p(gc.WORK, "out", "grid_meta.npy"))
minx, miny, maxx, maxy, res, W, H = gm
W, H, res = int(W), int(H), float(res)
ntrans = from_origin(minx, maxy, res, res)
ndvi = rasterio.open(gc.p(gc.ROOT, "rasters", "ndvi_2025_clean.tif")).read(1)
zl = np.load(gc.p(gc.WORK, "out", "zones_zones_2025.npy"))
dem = np.load(gc.p(gc.WORK, "out", "dem_arr.npy"))
dtr = Affine(*np.load(gc.p(gc.WORK, "out", "dem_trans.npy")))
demN = np.full((H, W), np.nan, "float32")
reproject(dem, demN, src_transform=dtr, src_crs=f"EPSG:{gc.UTM}",
          dst_transform=ntrans, dst_crs=f"EPSG:{gc.UTM}", resampling=Resampling.bilinear)

fig, ax = plt.subplots(1, 2, figsize=(16, 6.5))
m = ~np.isnan(ndvi) & ~np.isnan(demN)
# density-ish scatter (subsample)
idx = np.random.RandomState(0).choice(m.sum(), min(6000, m.sum()), replace=False)
xe, yn = demN[m][idx], ndvi[m][idx]
sc = ax[0].scatter(xe, yn, c=zl[m][idx], cmap="RdYlGn", s=6, alpha=0.5)
# trend (center x first - raw polyfit is ill-conditioned at elevation~130 m)
xc = demN[m] - demN[m].mean()
z = np.polyfit(xc, ndvi[m], 1)
xx = np.linspace(demN[m].min(), demN[m].max(), 50)
ax[0].plot(xx, np.polyval(z, xx - demN[m].mean()), "k--", lw=2,
           label="linear trend")
ax[0].legend(loc="upper right", fontsize=9)
ax[0].set_xlabel("elevation, m"); ax[0].set_ylabel("clean NDVI 2025")
ax[0].set_title("Productivity vs terrain: Spearman rho = -0.72\n"
                "(lower ground = more productive)")
cb = plt.colorbar(sc, ax=ax[0]); cb.set_label("zone")

colors = [zones[zones.zone == z]["color"].iloc[0] for z in stats["zone"]]
ax[1].bar(stats["zone"], stats["ndvi_mean"], color=colors, edgecolor="k",
          yerr=stats["ndvi_std"], capsize=4)
for _, s in stats.iterrows():
    ax[1].text(s["zone"], s["ndvi_mean"] + 0.0007,
               f"{s['area_ha']:.0f} ha", ha="center", fontsize=9)
ax[1].set_xlabel("zone (1=low ... 5=high vigour)")
ax[1].set_ylabel("mean clean NDVI 2025")
ax[1].set_ylim(0.842, 0.873)
ax[1].set_title("Zone separation - variance reduction 93.3% (ANOVA p<1e-300)")
plt.tight_layout()
plt.savefig(gc.p(FIG, "06_validation.png"), dpi=130, bbox_inches="tight")
plt.close()
print("saved 06_validation.png")


# ---------- 04 relief rebuild (fixed colours) ----------
def hillshade(zz, az=315, alt=45):
    z2 = np.where(np.isnan(zz), np.nanmean(zz), zz)
    gxx, gyy = np.gradient(z2, 10, 10)
    slope = np.pi/2 - np.arctan(np.sqrt(gxx**2 + gyy**2))
    asp = np.arctan2(-gxx, gyy)
    azr = np.radians(360-az+90); altr = np.radians(alt)
    return np.sin(altr)*np.sin(slope) + np.cos(altr)*np.cos(slope)*np.cos(azr-asp)

mask = ~np.isnan(dem)
hs = np.where(mask, hillshade(dem), np.nan)
extent = [dtr.c, dtr.c+dtr.a*dem.shape[1], dtr.f+dtr.e*dem.shape[0], dtr.f]
cont = gpd.read_file(gc.p(SHP, "relief", "relief_contours.shp"))
tpi = gpd.read_file(gc.p(SHP, "relief", "relief_tpi_zones.shp"))
slp = gpd.read_file(gc.p(SHP, "relief", "relief_slope_zones.shp")).sort_values("slope_cls")

fig, ax = plt.subplots(1, 3, figsize=(22, 7.2))
ax[0].imshow(hs, extent=extent, cmap="gray", alpha=0.55)
im = ax[0].imshow(np.where(mask, dem, np.nan), extent=extent, cmap="terrain", alpha=0.6)
cont.plot(ax=ax[0], color="black", linewidth=0.6)
for lev in cont["elev_m"].unique():
    sub = cont[cont.elev_m == lev]
    g = sub.geometry.iloc[0]
    pt = g.interpolate(0.5, normalized=True)
    ax[0].annotate(f"{lev:.0f}", (pt.x, pt.y), fontsize=6, color="black")
plt.colorbar(im, ax=ax[0], fraction=0.046, label="elevation, m")
ax[0].set_title("Elevation + 1 m contours (Copernicus GLO-30)")
ax[0].set_aspect("equal"); ax[0].axis("off")

plot_gdf_colored(ax[1], tpi, "tpi_cls", edge="none", lw=0)
leg2 = [Patch(facecolor=r["color"], label=f"{r['label']} ({r['area_ha']} ha)")
        for _, r in tpi.sort_values("tpi_cls").iterrows()]
ax[1].legend(handles=leg2, loc="lower center", bbox_to_anchor=(0.5, -0.12),
             fontsize=9, frameon=False)
ax[1].set_title("Topographic position (depression / mid-slope / ridge)")
ax[1].set_aspect("equal"); ax[1].axis("off")

plot_gdf_colored(ax[2], slp, "slope_cls", edge="none", lw=0)
leg3 = [Patch(facecolor=r["color"], label=f"{r['label']} ({r['area_ha']} ha)")
        for _, r in slp.sort_values("slope_cls").iterrows()]
ax[2].legend(handles=leg3, loc="lower center", bbox_to_anchor=(0.5, -0.15),
             fontsize=9, frameon=False)
ax[2].set_title("Slope classes")
ax[2].set_aspect("equal"); ax[2].axis("off")
plt.tight_layout()
plt.savefig(gc.p(FIG, "04_relief.png"), dpi=120, bbox_inches="tight")
plt.close()
print("saved 04_relief.png (fixed)")
print("done.")
