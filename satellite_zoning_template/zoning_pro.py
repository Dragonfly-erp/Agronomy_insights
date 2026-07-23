#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
zoning_pro — ПРОФЕСІЙНЕ зонування поля з власних Sentinel-2 даних.
==============================================================================
На відміну від run_zoning.py (працює з готовим NDVI-експортом), цей модуль
САМ ТЯГНЕ безхмарні знімки Sentinel-2 (10 м, ESA/Copernicus, безкоштовно, через
відкритий STAC Earth Search) за кілька років і рахує КІЛЬКА індексів:
  * NDVI — вигор рослин,
  * NDRE — червоний край (не насичується у щільному покриві),
  * NDMI — вологозабезпеченість.
З них будується СТАБІЛЬНИЙ багаторічний композит -> надійні зони продуктивності.

Вихід на поле (у теці <FARM>/<FIELD>/):
  * zones.shp   — зони, повне поле, без щілин; атрибути zone_id, product,
                  ndvi, ndre, ndmi, area_ha, corr_elev
  * relief.shp  — точки висот (Copernicus GLO-30)
  * report.pdf  — короткий звіт (1–2 стор.): які дані, за який період, 6 карт

Спільні функції беруться з run_zoning.py (очищення, цілісність покриття, рельєф).
==============================================================================
"""
import os, json, ssl, urllib.request
os.environ.setdefault('GDAL_HTTP_CAINFO', '/root/.ccr/ca-bundle.crt')
os.environ.setdefault('CURL_CA_BUNDLE', '/root/.ccr/ca-bundle.crt')
os.environ.setdefault('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
os.environ.setdefault('CPL_VSIL_CURL_ALLOWED_EXTENSIONS', '.tif')
os.environ.setdefault('GDAL_HTTP_MAX_RETRY', '3')
os.environ.setdefault('GDAL_HTTP_RETRY_DELAY', '2')
import numpy as np
import geopandas as gpd
import pandas as pd
import shapely
import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.windows import from_bounds
from rasterio.transform import from_origin, Affine
from rasterio.features import rasterize, shapes as rio_shapes
from scipy import ndimage as ndi
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from shapely.geometry import shape, Polygon, MultiPolygon, Point
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import base64, subprocess
import warnings
warnings.filterwarnings("ignore")

import run_zoning as rz   # reuse: clean_layer, sieve, clean_coverage, fill_coverage_gaps, ...

# ============================== CONFIG ======================================
FARM         = "ГЕВОРГ"
FIELD_NAME   = "P4"
BOUNDARY_SRC = None            # shp контуру поля АБО будь-який NDVI-shp поля (для меж)
OUT_ROOT     = "output_pro"
N_ZONES      = 5
SMOOTH_LEVEL = "C"
YEARS        = range(2019, 2025)
MONTHS       = (6, 8)          # вікно піку вегетації (міс. від/до, включно)
MAX_PER_YEAR = 3               # найменш хмарних сцен на рік
CLOUD_MAX    = 25              # % хмарності сцени
CANOPY_NDVI  = 0.40            # медіана NDVI сцени > -> сцена канопі (інакше голий ґрунт)
GRID_M       = 10.0            # Sentinel-2 native
EDGE_BUFFER_M = 18.0
POINT_STEP_M = 20
OUT_CRS      = 4326
# ============================================================================

STAC = "https://earth-search.aws.element84.com/v1/search"
CTX = ssl.create_default_context(cafile="/root/.ccr/ca-bundle.crt") \
    if os.path.exists("/root/.ccr/ca-bundle.crt") else ssl.create_default_context()
COLORS = ["#d7191c", "#fdae61", "#ffffbf", "#a6d96a", "#1a9641", "#006837"]
CHROME = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"


# --------------------------------------------------------------- geometry
def field_boundary(src, epsg=None):
    g = gpd.read_file(src)
    if g.crs is None:
        g = g.set_crs(4326)
    if epsg is None:
        b = g.to_crs(4326).total_bounds
        epsg = rz.utm_epsg((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    g = g.to_crs(epsg)
    geom = shapely.union_all(g.geometry.buffer(0).values).buffer(3).buffer(-3)
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda p: p.area)
    return Polygon(geom.exterior), epsg


# --------------------------------------------------------------- Sentinel-2
def stac_scenes(bwgs):
    feats = []
    for y in YEARS:
        body = json.dumps({"collections": ["sentinel-2-l2a"], "bbox": list(bwgs),
                           "datetime": f"{y}-{MONTHS[0]:02d}-05T00:00:00Z/{y}-{MONTHS[1]:02d}-28T00:00:00Z",
                           "query": {"eo:cloud_cover": {"lt": CLOUD_MAX}}, "limit": 40}).encode()
        req = urllib.request.Request(STAC, data=body, headers={"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, context=CTX, timeout=60))
        byday = {}
        for ft in r["features"]:
            d = ft["properties"]["datetime"][:10]
            if d not in byday or ft["properties"]["eo:cloud_cover"] < byday[d]["properties"]["eo:cloud_cover"]:
                byday[d] = ft
        feats += sorted(byday.values(), key=lambda x: x["properties"]["eo:cloud_cover"])[:MAX_PER_YEAR]
    return feats


def read_band(url, bwgs, epsg, ttr, H, W):
    with rasterio.open("/vsicurl/" + url) as s:
        b = transform_bounds("EPSG:4326", s.crs, *bwgs)
        win = from_bounds(*b, s.transform).round_offsets().round_lengths()
        arr = s.read(1, window=win, boundless=True, fill_value=0).astype("float32")
        wtr = s.window_transform(win)
    out = np.zeros((H, W), "float32")
    reproject(arr, out, src_transform=wtr, src_crs=f"EPSG:{epsg}",
              dst_transform=ttr, dst_crs=f"EPSG:{epsg}", resampling=Resampling.bilinear)
    return out


def fetch_composites(bwgs, epsg, ttr, H, W):
    feats = stac_scenes(bwgs)
    print(f"  candidate S2 scenes (<= {MAX_PER_YEAR}/yr, cloud<{CLOUD_MAX}%): {len(feats)}")
    NDVI, NDRE, NDMI, used, rgb = [], [], [], [], None
    for ft in feats:
        a = ft["assets"]; date = ft["properties"]["datetime"][:10]
        try:
            scl = read_band(a["scl"]["href"], bwgs, epsg, ttr, H, W)
            red = read_band(a["red"]["href"], bwgs, epsg, ttr, H, W)
            nir = read_band(a["nir"]["href"], bwgs, epsg, ttr, H, W)
            re1 = read_band(a["rededge1"]["href"], bwgs, epsg, ttr, H, W)
            sw1 = read_band(a["swir16"]["href"], bwgs, epsg, ttr, H, W)
        except Exception:
            continue
        good = np.isin(np.round(scl), [4, 5, 7])
        if good.mean() < 0.5:
            continue
        red /= 1e4; nir /= 1e4; re1 /= 1e4; sw1 /= 1e4
        ndvi = np.where(good, (nir - red) / (nir + red + 1e-6), np.nan)
        if np.nanmedian(ndvi) < CANOPY_NDVI:
            continue
        NDVI.append(ndvi)
        NDRE.append(np.where(good, (nir - re1) / (nir + re1 + 1e-6), np.nan))
        NDMI.append(np.where(good, (nir - sw1) / (nir + sw1 + 1e-6), np.nan))
        used.append(date)
        if rgb is None:                          # true-colour from 1st good scene
            try:
                g = read_band(a["green"]["href"], bwgs, epsg, ttr, H, W) / 1e4
                b = read_band(a["blue"]["href"], bwgs, epsg, ttr, H, W) / 1e4
                rgb = np.dstack([np.clip(red / 0.3, 0, 1), np.clip(g / 0.3, 0, 1),
                                 np.clip(b / 0.3, 0, 1)])
            except Exception:
                pass
    if len(used) < 2:
        raise SystemExit(f"Замало канопі-сцен ({len(used)}) — розширте YEARS/MONTHS.")
    comp = dict(ndvi=np.nanmedian(np.stack(NDVI), 0),
                ndre=np.nanmedian(np.stack(NDRE), 0),
                ndmi=np.nanmedian(np.stack(NDMI), 0))
    print(f"  USED canopy scenes: {len(used)} ({min(used)}..{max(used)})")
    return comp, sorted(used), rgb


# --------------------------------------------------------------- zoning core
def zonify(stack_feats, ndvi_c, core_mask, field_mask, field, tr, H, W):
    """stack_feats: (N,H,W) standardized index composites (core). Returns zones gdf + label raster."""
    P = rz.LEVELS[SMOOTH_LEVEL]
    cm = core_mask & np.all(~np.isnan(stack_feats), axis=0)
    X = np.column_stack([f[cm] for f in stack_feats])
    Xs = StandardScaler().fit_transform(X)
    # light spatial smoothing of features for coherence
    km = KMeans(N_ZONES, n_init=10, random_state=0).fit(Xs)
    lab = np.zeros((H, W), "int32"); lab[cm] = km.labels_ + 1
    order = sorted(range(1, N_ZONES + 1), key=lambda c: np.nanmean(ndvi_c[(lab == c) & cm]))
    remap = {o: n for n, o in enumerate(order, 1)}
    lab = np.vectorize(lambda x: remap.get(x, 0))(lab).astype("int32"); lab[~cm] = 0
    # de-hair (compact) via one-hot gaussian argmax
    if P["compact"] > 0:
        prob = np.stack([ndi.gaussian_filter((lab == z).astype("float32"), P["compact"] / GRID_M)
                         for z in range(1, N_ZONES + 1)])
        lab2 = np.zeros((H, W), "int32"); lab2[cm] = (prob.argmax(0) + 1)[cm]
    else:
        lab2 = lab.copy()
    lab2 = rz.sieve(lab2, cm, int(P["min_frac"] * field.area / GRID_M ** 2))
    # extend to FULL field
    full = rz.nearest_fill(lab2, lab2 > 0)
    full = np.where(field_mask, full, 0).astype("int32")
    if P["round_m"] > 0:
        sig = P["round_m"] / GRID_M
        prob = np.stack([ndi.gaussian_filter((full == z).astype("float32"), sig) for z in range(1, N_ZONES + 1)])
        fulls = np.zeros_like(full); fulls[field_mask] = (prob.argmax(0) + 1)[field_mask]
        fulls = rz.sieve(fulls, field_mask, int(0.3 * 1e4 / GRID_M ** 2))
    else:
        fulls = full
    recs = [{"zone_id": int(v), "geometry": shape(g)}
            for g, v in rio_shapes(fulls, mask=field_mask, transform=tr) if int(v)]
    z = gpd.GeoDataFrame(recs, crs=field.crs if hasattr(field, "crs") else None)
    return z, fulls


# --------------------------------------------------------------- report
def build_pdf(paths, meta, out_pdf):
    imgs = "".join(
        f'<figure><img src="data:image/png;base64,{base64.b64encode(open(p,"rb").read()).decode()}"/>'
        f'<figcaption>{cap}</figcaption></figure>' for p, cap in paths)
    rows = "".join(f"<tr><td>{r['zone_id']}</td><td>{r['product']}</td><td>{r['area_ha']}</td>"
                   f"<td>{r['ndvi']}</td><td>{r['ndre']}</td></tr>" for r in meta["zone_rows"])
    html = f"""<!doctype html><html lang="uk"><head><meta charset="utf-8"><style>
@page {{ size:A4; margin:12mm; }}
body {{ font-family:'DejaVu Sans',Arial,sans-serif; font-size:9.5pt; color:#1a1a1a; }}
h1 {{ font-size:16pt; color:#14532d; margin:0 0 2px; }}
.sub {{ color:#555; font-size:9pt; margin-bottom:6px; }}
.grid {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:6px; }}
figure {{ margin:0; text-align:center; }} img {{ width:100%; border:1px solid #ddd; }}
figcaption {{ font-size:7.5pt; color:#555; }}
table {{ border-collapse:collapse; width:100%; font-size:8.5pt; margin-top:6px; }}
th,td {{ border:1px solid #bbb; padding:2px 5px; text-align:center; }} th {{ background:#e8f3ea; }}
.box {{ background:#f2f8f3; border-left:4px solid #86c28c; padding:5px 10px; margin:6px 0; font-size:8.5pt; }}
</style></head><body>
<h1>Зонування поля {meta['field']} — {meta['farm']}</h1>
<div class="sub">Площа {meta['area_ha']} га · {meta['zones']} зон · {meta['date']}</div>
<div class="box"><b>Дані для зонування:</b> Sentinel-2 L2A (ESA/Copernicus, 10&nbsp;м) —
<b>{meta['n_scenes']} безхмарних сцен</b> за <b>{meta['period']}</b> (пік вегетації, міс. {meta['months']}).
Індекси: <b>NDVI</b> (вигор), <b>NDRE</b> (червоний край, без насичення), <b>NDMI</b> (волога).
Рельєф: Copernicus GLO-30. Зони — стабільний багаторічний медіанний композит, кластеризація.</div>
<div class="grid">{imgs}</div>
<table><tr><th>Зона</th><th>Продуктив.<br>(поле=100)</th><th>Площа, га</th><th>NDVI</th><th>NDRE</th></tr>{rows}</table>
<div class="sub" style="margin-top:6px">Дати сцен: {meta['dates']}</div>
</body></html>"""
    hp = out_pdf.replace(".pdf", "_tmp.html")
    open(hp, "w", encoding="utf-8").write(html)
    subprocess.run([CHROME, "--headless", "--no-sandbox", "--disable-gpu",
                    "--no-pdf-header-footer", f"--print-to-pdf={out_pdf}", "file://" + os.path.abspath(hp)],
                   check=True, capture_output=True)
    os.remove(hp)


def main():
    out = os.path.join(OUT_ROOT, FARM, FIELD_NAME)
    os.makedirs(out, exist_ok=True)
    field, EPSG = field_boundary(BOUNDARY_SRC)
    minx, miny, maxx, maxy = field.bounds
    minx, miny = np.floor(minx / GRID_M) * GRID_M, np.floor(miny / GRID_M) * GRID_M
    maxx, maxy = np.ceil(maxx / GRID_M) * GRID_M, np.ceil(maxy / GRID_M) * GRID_M
    W = int((maxx - minx) / GRID_M); H = int((maxy - miny) / GRID_M)
    tr = from_origin(minx, maxy, GRID_M, GRID_M)
    bwgs = transform_bounds(f"EPSG:{EPSG}", "EPSG:4326", minx, miny, maxx, maxy)
    core = field.buffer(-EDGE_BUFFER_M)
    if isinstance(core, MultiPolygon):
        core = max(core.geoms, key=lambda p: p.area)
    field_mask = rasterize([(field, 1)], (H, W), transform=tr, fill=0, dtype="uint8").astype(bool)
    core_mask = rasterize([(core, 1)], (H, W), transform=tr, fill=0, dtype="uint8").astype(bool)
    print(f"{FARM}/{FIELD_NAME}: {field.area/1e4:.1f} ha | grid {W}x{H} EPSG:{EPSG}")

    comp, used, rgb = fetch_composites(bwgs, EPSG, tr, H, W)
    ndvi = rz.clean_layer(comp["ndvi"], core_mask)
    ndre = rz.clean_layer(comp["ndre"], core_mask)
    ndmi = rz.clean_layer(comp["ndmi"], core_mask)

    z, fulls = zonify(np.stack([ndvi, ndre, ndmi]), ndvi, core_mask, field_mask, field, tr, H, W)
    z.set_crs(EPSG, inplace=True)
    z = z.dissolve(by="zone_id", as_index=False)
    z = rz.clean_coverage(z, field, tol=max(4.0, rz.LEVELS[SMOOTH_LEVEL]["round_m"] * 0.5))
    z = z[z.geometry.area > 0]
    uniq = sorted(z["zone_id"].unique())
    z["zone_id"] = z["zone_id"].map({o: n for n, o in enumerate(uniq, 1)})
    z = z.sort_values("zone_id").reset_index(drop=True)

    # DEM / relief
    dem_src = rz.download_cop30((bwgs[1] + bwgs[3]) / 2, (bwgs[0] + bwgs[2]) / 2)
    demu, dtr, slope = rz.build_relief(dem_src, field, EPSG, bwgs)
    demN = np.full((H, W), np.nan, "float32")
    reproject(demu, demN, src_transform=dtr, src_crs=f"EPSG:{EPSG}", dst_transform=tr,
              dst_crs=f"EPSG:{EPSG}", resampling=Resampling.bilinear)

    # attributes
    labf = rasterize([(g, i) for g, i in zip(z.geometry, z.zone_id)], (H, W), transform=tr, fill=0, dtype="int32")
    cm = core_mask & ~np.isnan(ndvi)
    fmean = float(np.nanmean(ndvi[cm]))
    from scipy.stats import pearsonr
    rows = []
    for _, r in z.sort_values("zone_id").iterrows():
        zid = int(r.zone_id); pm = (labf == zid) & cm
        el = demN[pm & ~np.isnan(demN)]; nde = ndvi[pm & ~np.isnan(demN)]
        cr = float(pearsonr(nde, el)[0]) if len(nde) > 5 and np.std(el) > 0 else float("nan")
        rows.append(dict(zone_id=zid, product=round(float(np.nanmean(ndvi[pm])) / fmean * 100, 1),
                         ndvi=round(float(np.nanmean(ndvi[pm])), 3), ndre=round(float(np.nanmean(ndre[pm])), 3),
                         ndmi=round(float(np.nanmean(ndmi[pm])), 3),
                         area_ha=round(float(r.geometry.area / 1e4), 2),
                         corr_elev=round(cr, 3) if cr == cr else None))
    z = z.merge(pd.DataFrame(rows), on="zone_id")
    z = z[["zone_id", "product", "ndvi", "ndre", "ndmi", "area_ha", "corr_elev", "geometry"]]

    # write zones (snap + gap-fill + verify)
    zpath = os.path.join(out, "zones.shp")
    zo = z.to_crs(OUT_CRS)
    zo["geometry"] = shapely.set_precision(zo.geometry.values, grid_size=1e-7)
    zo = rz.fill_coverage_gaps(zo[~zo.geometry.is_empty]); zo["geometry"] = zo.geometry.buffer(0)
    zo = rz.fill_coverage_gaps(zo)
    zo.to_file(zpath)
    holes = 99
    for _ in range(4):                       # write -> re-read -> heal until 0 gaps
        check = gpd.read_file(zpath)
        holes, ov, _ = rz.coverage_holes_overlap(check)
        if holes == 0:
            break
        rz.fill_coverage_gaps(check).to_file(zpath)
    if holes > 0:
        raise SystemExit(f"!! ЦІЛІСНІСТЬ ПОРУШЕНА: {holes} щілин — файл НЕ прийнято.")
    print(f"  zones: {len(zo)} | integrity gaps={holes} | {check.to_crs(EPSG).geometry.area.sum()/1e4:.1f} ha")
    rz.write_qml_categorized(os.path.join(out, "zones.qml"), "zone_id",
                             [(int(rr.zone_id), f"Зона {int(rr.zone_id)} · продукт.{rr['product']}",
                               COLORS[int(rr.zone_id) - 1]) for _, rr in zo.iterrows()])

    # relief points
    rz.RELIEF_TYPE = "points"; rz.POINT_STEP_M = POINT_STEP_M; rz.OUT_CRS = OUT_CRS; rz.WRITE_QML = True
    rz.write_relief(demu, dtr, slope, field, EPSG, out)

    # ---- report figures ----
    figdir = os.path.join(out, "_fig"); os.makedirs(figdir, exist_ok=True)
    fmaskf = np.where(core_mask, 1.0, np.nan)
    def savemap(arr, cmap, title, fn, pts=False):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(np.where(core_mask, arr, np.nan), cmap=cmap,
                  vmin=np.nanpercentile(arr[cm], 3), vmax=np.nanpercentile(arr[cm], 97))
        ax.set_title(title, fontsize=9); ax.axis("off")
        plt.tight_layout(); p = os.path.join(figdir, fn); plt.savefig(p, dpi=90, bbox_inches="tight"); plt.close()
        return p
    p_ndvi = savemap(ndvi, "RdYlGn", "NDVI (вигор)", "ndvi.png")
    p_ndre = savemap(ndre, "RdYlGn", "NDRE (черв. край)", "ndre.png")
    p_ndmi = savemap(ndmi, "BrBG", "NDMI (волога)", "ndmi.png")
    # relief hillshade
    zz = np.where(np.isnan(demN), np.nanmean(demN), demN)
    gy, gx = np.gradient(zz, GRID_M)
    hs = np.cos(np.radians(45)) * np.cos(np.arctan(np.sqrt(gx**2 + gy**2))) + \
         np.sin(np.radians(45)) * np.sin(np.arctan(np.sqrt(gx**2 + gy**2)))
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(np.where(core_mask, demN, np.nan), cmap="terrain"); ax.set_title("Рельєф (висота)", fontsize=9); ax.axis("off")
    plt.tight_layout(); p_dem = os.path.join(figdir, "dem.png"); plt.savefig(p_dem, dpi=90, bbox_inches="tight"); plt.close()
    # zones
    fig, ax = plt.subplots(figsize=(4, 4))
    zu = zo.to_crs(EPSG).sort_values("zone_id")
    for _, r in zu.iterrows():
        gpd.GeoSeries([r.geometry], crs=EPSG).plot(ax=ax, color=COLORS[int(r.zone_id)-1], edgecolor="white", linewidth=0.7)
    ax.set_title("ЗОНИ (підсумок)", fontsize=9); ax.set_aspect("equal"); ax.axis("off")
    plt.tight_layout(); p_zon = os.path.join(figdir, "zones.png"); plt.savefig(p_zon, dpi=90, bbox_inches="tight"); plt.close()
    # rgb
    figs = [(p_ndvi, "NDVI — вигор рослин"), (p_ndre, "NDRE — червоний край"),
            (p_ndmi, "NDMI — волога"), (p_dem, "Рельєф — Copernicus GLO-30")]
    if rgb is not None:
        fig, ax = plt.subplots(figsize=(4, 4)); ax.imshow(np.where(core_mask[..., None], rgb, np.nan))
        ax.set_title("Знімок (true-color)", fontsize=9); ax.axis("off")
        plt.tight_layout(); p_rgb = os.path.join(figdir, "rgb.png"); plt.savefig(p_rgb, dpi=90, bbox_inches="tight"); plt.close()
        figs.append((p_rgb, "Sentinel-2 (натуральний колір)"))
    figs.append((p_zon, "Підсумкові зони продуктивності"))

    meta = dict(farm=FARM, field=FIELD_NAME, area_ha=round(field.area/1e4, 1), zones=len(zo),
                date=max(used),
                n_scenes=len(used), period=f"{used[0][:4]}–{used[-1][:4]}",
                months=f"{MONTHS[0]}–{MONTHS[1]}", dates=", ".join(used),
                zone_rows=z.drop(columns="geometry").sort_values("zone_id").to_dict("records"))
    build_pdf(figs, meta, os.path.join(out, "report.pdf"))
    import shutil; shutil.rmtree(figdir, ignore_errors=True)
    print(f"  report.pdf + zones.shp + relief.shp -> {out}")


if __name__ == "__main__":
    main()
