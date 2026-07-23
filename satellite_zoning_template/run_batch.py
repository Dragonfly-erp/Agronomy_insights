#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ПАКЕТНА обробка багатьох полів.

Сканує теку з NDVI-шейпфайлами (експорт платформи, назви виду
"<власник>_<господарство>_<ПОЛЕ>_<РІК>_NO Product_1_poly.shp"), групує за полем,
сам відбирає роки повного покриву (канопі) і для КОЖНОГО поля робить 2 файли:
    <OUT_ROOT>/<Поле>/zones.shp   (багаторічні стабільні зони)
    <OUT_ROOT>/<Поле>/relief.shp  (точки висот)

Роки з голим ґрунтом (низький NDVI) відкидаються. Якщо канопі-років кілька —
вони нормалізуються й усереднюються (стабільний патерн). Якщо жодного —
береться найкращий наявний рік (позначається як низька впевненість).

Запуск:  python run_batch.py
"""
import os
import re
import glob
import numpy as np
import geopandas as gpd
import run_zoning as R

# ============================== CONFIG ======================================
INPUT_DIR   = "INPUT_FIELDS"                 # тека з NDVI-шейпфайлами всіх полів
OUT_ROOT    = "output_batch"                 # куди складати результати по полях
FIELD_YEAR_RE = r'_([^_]+)_(\d{4})_NO Product'   # regex: (поле)(рік) з назви файлу
CANOPY_MIN  = 0.55                           # NDVI сер. >= -> рік придатний (канопі)
R.SMOOTH_LEVEL = "C"
R.RELIEF_TYPE  = "points"
R.N_ZONES      = 5
R.DEM_PATH     = None                        # авто Copernicus GLO-30 (кеш dem_cache)
# ============================================================================


def dec(s):
    return re.sub(r'#U([0-9A-Fa-f]{4})', lambda m: chr(int(m.group(1), 16)), s)


def safe(name):
    return (name.replace("Поле", "Pole").replace("П", "P")
            .replace(" ", "_").replace("/", "_"))


def collect(input_dir):
    """-> {field: [(year, mean_ndvi, path), ...]}"""
    from collections import defaultdict
    fields = defaultdict(list)
    for f in glob.glob(os.path.join(input_dir, "**", "*.shp"), recursive=True):
        b = dec(os.path.basename(f))
        m = re.search(FIELD_YEAR_RE, b)
        if not m:
            print("  [skip] не розпізнано:", b); continue
        field, year = m.group(1), m.group(2)
        try:
            g = gpd.read_file(f)
            mean = float(np.nanmean(g[R.NDVI_FIELD].values))
        except Exception as e:
            print("  [skip] помилка читання", b, e); continue
        fields[field].append((year, mean, f))
    return fields


def main():
    fields = collect(INPUT_DIR)
    print(f"Знайдено полів: {len(fields)}\n")
    summary = []
    for field in sorted(fields):
        yrs = sorted(fields[field])
        canopy = [(y, m, p) for y, m, p in yrs if m >= CANOPY_MIN]
        note = ""
        if not canopy:                       # жодного канопі -> найкращий рік
            best = max(yrs, key=lambda x: x[1])
            canopy = [best]
            note = f"НИЗЬКА ВПЕВНЕНІСТЬ (немає повного покриву; best {best[0]} NDVI {best[1]:.2f})"
        used_years = [y for y, _, _ in canopy]
        out = os.path.join(OUT_ROOT, safe(field))
        print(f"=== {field} -> {safe(field)} | роки {used_years} {note}")
        R.INPUT_NDVI = [p for _, _, p in canopy]
        R.FIELD_CONTOUR = None
        R.OUT_DIR = out
        try:
            R.main()
            z = gpd.read_file(os.path.join(out, "zones.shp"))
            r = gpd.read_file(os.path.join(out, "relief.shp"))
            summary.append((field, used_years, len(z),
                            round(z.to_crs(z.estimate_utm_crs()).geometry.area.sum() / 1e4, 1),
                            len(r), note))
        except Exception as e:
            print("  !! ПОМИЛКА:", e)
            summary.append((field, used_years, -1, -1, -1, "ПОМИЛКА: " + str(e)))
        print()

    print("=" * 70, "\nПІДСУМОК:")
    for field, yrs, nz, area, npts, note in summary:
        print(f"  {field:12s} зон={nz} площа={area}га точок={npts} роки={yrs} {note}")


if __name__ == "__main__":
    main()
