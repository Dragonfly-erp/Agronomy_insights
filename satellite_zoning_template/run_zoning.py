#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
ШАБЛОН: супутниковий NDVI  ->  зони продуктивності + рельєф  (рівно 2 шейпфайли)
==============================================================================
Як користуватись:
    1) впишіть шляхи у блоці CONFIG нижче;
    2) запустіть:  python run_zoning.py
    3) у теці OUT_DIR зʼявиться РІВНО два шейпфайли:

       zones.shp   - зони продуктивності, покривають ПОВНУ площу поля.
                     Атрибути:
                        zone_id   - номер зони (1 = найслабша ... N = найсильніша)
                        veg_min   - мінімальна вегетація (NDVI) в зоні
                        veg_max   - максимальна вегетація (NDVI) в зоні
                        veg_mean  - середня вегетація (NDVI) в зоні
                        area_ha   - площа зони, га
                        corr_elev - коеф. кореляції NDVI<->висота в межах зони
                                    (Пірсон; показує, наскільки зона зумовлена рельєфом)
       relief.shp  - рельєф поля (горизонталі за замовчуванням).

Логіка (чому так):
  * аналіз робиться на "продуктивному ядрі" (поле, стягнуте всередину на
    EDGE_BUFFER_M) — щоб прибрати крайовий ефект, розворотні смуги, узбіччя;
  * зони кластеризуються за малюнком NDVI, тонкі "відростки" прибираються, дрібні
    шматки (< MIN_FRAC площі) зливаються -> компактні, придатні для техніки зони;
  * ГОТОВІ зони потім РОЗТЯГУЮТЬСЯ на повний контур поля (кожен крайовий піксель
    отримує найближчу зону) -> вихід покриває всю площу, яку ви дали;
  * рельєф береться з безкоштовної ЦМР Copernicus GLO-30 (ESA, ~30 м) за контуром.

Залежності: geopandas, rasterio, scikit-learn, scipy, shapely  (pip install ...)
Для рельєфу потрібен інтернет (або вкажіть локальний DEM у DEM_PATH).
==============================================================================
"""
import os
import ssl
import urllib.request
import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.features import rasterize, shapes as rio_shapes
from rasterio.transform import from_origin, Affine
from rasterio.warp import calculate_default_transform, reproject, Resampling
from scipy import ndimage as ndi
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from shapely.geometry import shape, Polygon, MultiPolygon, LineString
from shapely.ops import unary_union
import warnings
warnings.filterwarnings("ignore")

# ============================== CONFIG ======================================
INPUT_NDVI    = "INPUT/ndvi.shp"   # шейпфайл NDVI-полігонів (поле NDVI_FIELD) АБО растр .tif
NDVI_FIELD    = "NDVI"             # назва атрибута зі значенням NDVI (для полігонів)
FIELD_CONTOUR = None               # шейпфайл контуру поля; None -> взяти обвід NDVI
OUT_DIR       = "output"           # тека для 2 вихідних шейпфайлів

N_ZONES       = 5                  # кількість зон (4..6 типово)
SMOOTH_LEVEL  = "C"                # A(точно) B C(рекоменд.) D(сильно згладжено)
EDGE_BUFFER_M = 18.0               # стягування поля для аналізу (крайовий ефект), м
GRID_M        = 5.0                # робоча сітка, м

RELIEF_TYPE   = "contours"         # "contours" (горизонталі) або "elevation_zones"
CONTOUR_STEP  = 1.0                # крок горизонталей, м
DEM_PATH      = None               # локальний DEM .tif; None -> авто Copernicus GLO-30

OUT_CRS       = 4326               # CRS виходу (4326 = WGS84, універсально)
WRITE_QML     = True               # покласти поряд .qml (стиль QGIS) — не обовʼязково
# ============================================================================

# рівні згладжування (pre=перед-згладж., compact=прибирання відростків, m; min_frac=дрібні шматки; round=округлення меж)
LEVELS = {
    "A": dict(pre=10, compact=0,  min_frac=0.02,  round_m=6),
    "B": dict(pre=10, compact=20, min_frac=0.025, round_m=10),
    "C": dict(pre=10, compact=32, min_frac=0.03,  round_m=14),
    "D": dict(pre=25, compact=50, min_frac=0.04,  round_m=18),
}
ZCOLORS = ["#d7191c", "#fdae61", "#ffffbf", "#a6d96a", "#1a9641",
           "#006837", "#8c510a"]


# --------------------------------------------------------------------- utils
def utm_epsg(lon, lat):
    return (32600 if lat >= 0 else 32700) + int((lon + 180) / 6) + 1


def load_ndvi_to_utm(path, field):
    """Return (GeoDataFrame in UTM, utm_epsg). Accepts polygon shp or raster."""
    if path.lower().endswith((".tif", ".tiff")):
        raise SystemExit("Для растрового NDVI розширте load_ndvi_to_utm (тут — полігони).")
    g = gpd.read_file(path)
    if g.crs is None:
        g = g.set_crs(4326)
    c = g.to_crs(4326).unary_union.centroid
    epsg = utm_epsg(c.x, c.y)
    return g.to_crs(epsg), epsg


def field_from(contour_path, ndvi_gdf, epsg):
    if contour_path:
        f = gpd.read_file(contour_path).to_crs(epsg)
        geom = unary_union(f.geometry.values)
    else:
        geom = unary_union(ndvi_gdf.geometry.values).buffer(3).buffer(-3)
    if isinstance(geom, MultiPolygon):
        geom = max(geom.geoms, key=lambda p: p.area)
    return Polygon(geom.exterior)


def nearest_fill(a, valid):
    idx = ndi.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return a[tuple(idx)]


def sieve(lab, mask, min_px):
    lab = lab.copy()
    for _ in range(25):
        changed = 0
        for cls in np.unique(lab[mask]):
            comp, n = ndi.label((lab == cls) & mask)
            if not n:
                continue
            sizes = ndi.sum(np.ones_like(comp), comp, range(1, n + 1))
            for i, sz in enumerate(sizes, 1):
                if sz < min_px:
                    blob = comp == i
                    dil = ndi.binary_dilation(blob, iterations=2) & mask & ~blob
                    nb = lab[dil]; nb = nb[nb != cls]
                    if nb.size:
                        v, c = np.unique(nb, return_counts=True)
                        lab[blob] = v[c.argmax()]; changed += 1
        if not changed:
            break
    return lab


def download_cop30(lat, lon, cache="dem_cache"):
    """Download the 1x1 deg Copernicus GLO-30 tile containing (lat,lon)."""
    os.makedirs(cache, exist_ok=True)
    ns = f"N{int(np.floor(lat)):02d}" if lat >= 0 else f"S{int(-np.floor(lat)):02d}"
    ew = f"E{int(np.floor(lon)):03d}" if lon >= 0 else f"W{int(-np.floor(lon)):03d}"
    name = f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM"
    out = os.path.join(cache, name + ".tif")
    if os.path.exists(out):
        return out
    url = f"https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif"
    ca = "/root/.ccr/ca-bundle.crt"
    ctx = ssl.create_default_context(cafile=ca) if os.path.exists(ca) else ssl.create_default_context()
    print("  downloading DEM tile:", name)
    with urllib.request.urlopen(url, context=ctx, timeout=120) as r, open(out, "wb") as f:
        f.write(r.read())
    return out


def hex2rgb(h):
    h = h.lstrip("#"); return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},255"


def write_qml_categorized(path, attr, cats):
    syms = "".join(
        f'<symbol type="fill" name="{i}"><layer class="SimpleFill"><Option type="Map">'
        f'<Option name="color" type="QString" value="{hex2rgb(c)}"/>'
        f'<Option name="outline_color" type="QString" value="255,255,255,255"/>'
        f'<Option name="outline_width" type="QString" value="0.26"/>'
        f'<Option name="style" type="QString" value="solid"/></Option></layer></symbol>'
        for i, (_, _, c) in enumerate(cats))
    catx = "".join(f'<category value="{v}" symbol="{i}" label="{l}" render="true"/>'
                   for i, (v, l, _) in enumerate(cats))
    open(path, "w", encoding="utf-8").write(
        "<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>\n"
        '<qgis version="3.28" styleCategories="Symbology">'
        f'<renderer-v2 type="categorizedSymbol" attr="{attr}" forceraster="0" symbollevels="0" enableorderby="0">'
        f'<categories>{catx}</categories><symbols>{syms}</symbols></renderer-v2></qgis>')


# ------------------------------------------------------------------- pipeline
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    P = LEVELS[SMOOTH_LEVEL]
    ndvi_gdf, EPSG = load_ndvi_to_utm(INPUT_NDVI, NDVI_FIELD)
    field = field_from(FIELD_CONTOUR, ndvi_gdf, EPSG)
    core = field.buffer(-EDGE_BUFFER_M)
    if isinstance(core, MultiPolygon):
        core = max(core.geoms, key=lambda p: p.area)
    print(f"CRS UTM EPSG:{EPSG} | field {field.area/1e4:.2f} ha | core {core.area/1e4:.2f} ha")

    # grid over full field
    minx, miny, maxx, maxy = field.bounds
    W = int(np.ceil((maxx - minx) / GRID_M)); H = int(np.ceil((maxy - miny) / GRID_M))
    tr = from_origin(minx, maxy, GRID_M, GRID_M)
    field_mask = rasterize([(field, 1)], (H, W), transform=tr, fill=0, dtype="uint8").astype(bool)
    core_mask = rasterize([(core, 1)], (H, W), transform=tr, fill=0, dtype="uint8").astype(bool)
    ndvi = rasterize([(g, v) for g, v in zip(ndvi_gdf.geometry, ndvi_gdf[NDVI_FIELD])],
                     (H, W), transform=tr, fill=np.nan, dtype="float32")

    # ---- clean on core: MAD outliers -> median de-stripe -> gaussian ----
    a = np.where(core_mask & ~np.isnan(ndvi), ndvi, np.nan)
    v = a[~np.isnan(a)]; med = np.median(v); mad = np.median(np.abs(v - med)) or 1e-6
    lo, hi = med - 3.5 * 1.4826 * mad, med + 3.5 * 1.4826 * mad
    a = np.where((a < lo) | (a > hi), np.nan, a)
    valid = ~np.isnan(a)
    filled = nearest_fill(np.where(valid, a, med), valid)
    clean = ndi.gaussian_filter(ndi.median_filter(filled, size=5), sigma=2.0)
    clean = np.where(core_mask, clean, np.nan).astype("float32")

    # ---- DEM / relief ----
    cwgs = gpd.GeoSeries([field], crs=EPSG).to_crs(4326)
    bx = cwgs.total_bounds
    if DEM_PATH:
        dem_src = DEM_PATH
    else:
        cc = cwgs.unary_union.centroid
        dem_src = download_cop30(cc.y, cc.x)
    demu, dtr, slope_pct = build_relief(dem_src, field, EPSG, bx)
    demN = np.full((H, W), np.nan, "float32")
    reproject(demu, demN, src_transform=dtr, src_crs=f"EPSG:{EPSG}",
              dst_transform=tr, dst_crs=f"EPSG:{EPSG}", resampling=Resampling.bilinear)

    # ---- cluster (faithful NDVI) + de-hair + sieve ----
    sm = ndi.gaussian_filter(np.where(core_mask, clean, np.nanmean(clean)), P["pre"] / GRID_M)
    cm = core_mask & ~np.isnan(clean)
    km = KMeans(N_ZONES, n_init=10, random_state=0).fit(sm[cm].reshape(-1, 1))
    lab = np.zeros((H, W), "int32"); lab[cm] = km.labels_ + 1
    order = sorted(range(1, N_ZONES + 1), key=lambda c: clean[(lab == c) & cm].mean())
    remap = {o: n for n, o in enumerate(order, 1)}
    lab = np.vectorize(lambda x: remap.get(x, 0))(lab).astype("int32"); lab[~cm] = 0
    if P["compact"] > 0:
        prob = np.stack([ndi.gaussian_filter((lab == z).astype("float32"), P["compact"] / GRID_M)
                         for z in range(1, N_ZONES + 1)])
        lab2 = np.zeros((H, W), "int32"); lab2[cm] = (prob.argmax(0) + 1)[cm]
    else:
        lab2 = lab.copy()
    lab2 = sieve(lab2, cm, int(P["min_frac"] * field.area / GRID_M ** 2))

    # ---- EXTEND zones to the FULL field (nearest zone for every field pixel) ----
    labz = lab2.copy()
    labz[~cm] = 0
    full = nearest_fill(labz, labz > 0)          # nearest non-zero label everywhere
    full = np.where(field_mask, full, 0).astype("int32")

    # ---- smooth full-field labels (partition-preserving) then vectorise ----
    # one-hot Gaussian + argmax keeps a clean tiling (no gaps / no overlaps),
    # unlike per-polygon buffering. Boundaries come out rounded and natural.
    if P["round_m"] > 0:
        sig = P["round_m"] / GRID_M
        prob = np.stack([ndi.gaussian_filter((full == zz).astype("float32"), sig)
                         for zz in range(1, N_ZONES + 1)])
        fulls = np.zeros_like(full); fulls[field_mask] = (prob.argmax(0) + 1)[field_mask]
        fulls = sieve(fulls, field_mask, int(0.3 * 1e4 / GRID_M ** 2))
    else:
        fulls = full
    recs = [{"zone_id": int(val), "geometry": shape(g)}
            for g, val in rio_shapes(fulls, mask=field_mask, transform=tr) if int(val)]
    z = gpd.GeoDataFrame(recs, crs=EPSG).dissolve(by="zone_id", as_index=False)
    z["geometry"] = z.geometry.simplify(4)
    z = z[z.geometry.area > 0]

    # ---- per-zone attributes (from cleaned core pixels) ----
    labf = rasterize([(g, i) for g, i in zip(z.geometry, z.zone_id)], (H, W),
                     transform=tr, fill=0, dtype="int32")
    rows = []
    for _, row in z.sort_values("zone_id").iterrows():
        zid = int(row.zone_id)
        pm = (labf == zid) & cm
        nd = clean[pm]
        el = demN[pm & ~np.isnan(demN)]
        nde = clean[pm & ~np.isnan(demN)]
        if len(nde) > 5 and np.std(el) > 0 and np.std(nde) > 0:
            cr = float(pearsonr(nde, el)[0])
        else:
            cr = float("nan")
        rows.append(dict(zone_id=zid,
                         veg_min=round(float(nd.min()), 4),
                         veg_max=round(float(nd.max()), 4),
                         veg_mean=round(float(nd.mean()), 4),
                         area_ha=round(float(row.geometry.area / 1e4), 2),
                         corr_elev=round(cr, 3) if cr == cr else None))
    attr = pd.DataFrame(rows)
    z = z.merge(attr, on="zone_id")
    z = z[["zone_id", "veg_min", "veg_max", "veg_mean", "area_ha", "corr_elev", "geometry"]]

    # ---- write zones.shp ----
    zone_path = os.path.join(OUT_DIR, "zones.shp")
    z.to_crs(OUT_CRS).to_file(zone_path)
    if WRITE_QML:
        write_qml_categorized(os.path.join(OUT_DIR, "zones.qml"), "zone_id",
                              [(int(rr.zone_id), f"Зона {int(rr.zone_id)} (NDVI {rr.veg_mean})",
                                ZCOLORS[int(rr.zone_id) - 1]) for _, rr in z.iterrows()])
    print(f"zones.shp -> {len(z)} зон, {z['area_ha'].sum():.2f} га (повне поле {field.area/1e4:.2f} га)")

    # ---- write relief.shp ----
    write_relief(demu, dtr, slope_pct, field, EPSG, OUT_DIR)
    print("Готово: 2 шейпфайли у", OUT_DIR)


def build_relief(dem_src, field, EPSG, bbox_wgs):
    """Read DEM, reproject to UTM 10 m, clip to field. Return (dem, transform, slope%)."""
    with rasterio.open(dem_src) as s:
        from rasterio.windows import from_bounds
        win = from_bounds(bbox_wgs[0] - .004, bbox_wgs[1] - .004,
                          bbox_wgs[2] + .004, bbox_wgs[3] + .004, s.transform)
        dem = s.read(1, window=win).astype("float32")
        wtr = s.window_transform(win); scrs = s.crs
    dtr, dw, dh = calculate_default_transform(
        scrs, f"EPSG:{EPSG}", dem.shape[1], dem.shape[0],
        left=bbox_wgs[0] - .004, bottom=bbox_wgs[1] - .004,
        right=bbox_wgs[2] + .004, top=bbox_wgs[3] + .004, resolution=10.0)
    demu = np.empty((dh, dw), "float32")
    reproject(dem, demu, src_transform=wtr, src_crs=scrs, dst_transform=dtr,
              dst_crs=f"EPSG:{EPSG}", resampling=Resampling.cubic)
    fmask = rasterize([(field.buffer(10), 1)], (dh, dw), transform=dtr, fill=0, dtype="uint8").astype(bool)
    demu = np.where(fmask, demu, np.nan)
    fill = nearest_fill(demu, ~np.isnan(demu))
    gy, gx = np.gradient(fill, 10.0)
    slope = np.tan(np.radians(np.degrees(np.arctan(np.sqrt(gx ** 2 + gy ** 2))))) * 100
    slope[~fmask] = np.nan
    return demu, dtr, slope


def write_relief(demu, dtr, slope, field, EPSG, out_dir):
    dh, dw = demu.shape
    zmin, zmax = np.nanmin(demu), np.nanmax(demu)
    if RELIEF_TYPE == "contours":
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = dtr.c + (np.arange(dw) + .5) * dtr.a
        ys = dtr.f + (np.arange(dh) + .5) * dtr.e
        X, Y = np.meshgrid(xs, ys)
        levels = np.arange(np.floor(zmin) + 1, np.ceil(zmax), CONTOUR_STEP)
        cs = plt.contour(X, Y, np.where(np.isnan(demu), -9999, demu), levels=levels)
        recs = []
        for lev, segs in zip(cs.levels, cs.allsegs):
            for seg in segs:
                if len(seg) < 2:
                    continue
                ln = LineString(seg).intersection(field)
                for gg in (ln.geoms if ln.geom_type == "MultiLineString" else [ln]):
                    if gg.geom_type == "LineString" and gg.length > 5:
                        recs.append({"elev_m": round(float(lev), 1), "geometry": gg})
        plt.close()
        gpd.GeoDataFrame(recs, crs=EPSG).to_crs(OUT_CRS).to_file(os.path.join(out_dir, "relief.shp"))
    else:  # elevation_zones
        qs = np.nanquantile(demu, [.2, .4, .6, .8])
        cls = np.digitize(demu, qs).astype("float32")
        cls[np.isnan(demu)] = -1
        cls = ndi.median_filter(cls, size=3)
        intc = np.where((cls < 0) | np.isnan(demu), -1, cls).astype("int32")
        fmask = intc >= 0
        recs = [{"elev_cls": int(val), "geometry": shape(g)}
                for g, val in rio_shapes(intc, mask=fmask, transform=dtr) if int(val) >= 0]
        g = gpd.GeoDataFrame(recs, crs=EPSG).dissolve(by="elev_cls", as_index=False)
        g["geometry"] = g.geometry.simplify(4)
        g = g[g.geometry.area > 0].clip(field)
        names = [f"<{qs[0]:.1f}", f"{qs[0]:.1f}-{qs[1]:.1f}", f"{qs[1]:.1f}-{qs[2]:.1f}",
                 f"{qs[2]:.1f}-{qs[3]:.1f}", f">{qs[3]:.1f}"]
        g["label"] = g["elev_cls"].map(lambda i: names[int(i)] if int(i) < len(names) else "")
        g.to_crs(OUT_CRS).to_file(os.path.join(out_dir, "relief.shp"))
        if WRITE_QML:
            cols = ["#2b83ba", "#abdda4", "#ffffbf", "#fdae61", "#d7191c"]
            write_qml_categorized(os.path.join(out_dir, "relief.qml"), "elev_cls",
                                  [(int(rr.elev_cls), rr.label + " м", cols[int(rr.elev_cls)])
                                   for _, rr in g.iterrows()])


if __name__ == "__main__":
    main()
