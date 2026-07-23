#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ПАКЕТНА професійна обробка полів одного господарства через zoning_pro
(власні Sentinel-2, кілька індексів, PDF-звіт). Потрібні лише КОНТУРИ полів —
беруться з наявних шейпфайлів (значення NDVI з них не використовуються, знімки
тягнуться самостійно). Вихід: <OUT_ROOT>/<FARM>/<Поле>/{zones,relief,report.pdf}.

Запуск:  python run_batch_pro.py
"""
import os, re, glob
import geopandas as gpd
import zoning_pro as Z

# ============================== CONFIG ======================================
INPUT_DIR = "INPUT_FIELDS"      # тека з шейпфайлами полів (для контурів + назв)
FARM      = "ГЕВОРГ"
OUT_ROOT  = "output_pro"
FIELD_RE  = r'_([^_]+)_(\d{4})_NO Product'
# ============================================================================


def dec(s):
    return re.sub(r'#U([0-9A-Fa-f]{4})', lambda m: chr(int(m.group(1), 16)), s)


def safe(n):
    return (n.replace("Поле", "Pole").replace("П", "P").replace(" ", "_").replace("/", "_"))


def main():
    # one boundary shapefile per field (any year)
    byfield = {}
    for f in glob.glob(os.path.join(INPUT_DIR, "**", "*.shp"), recursive=True):
        m = re.search(FIELD_RE, dec(os.path.basename(f)))
        if m:
            byfield.setdefault(m.group(1), f)   # first seen year is enough for the outline
    print(f"Полів: {len(byfield)} -> {list(byfield)}\n")
    summary = []
    for field, shp in sorted(byfield.items()):
        print(f"===== {field} =====")
        Z.FARM = FARM
        Z.FIELD_NAME = safe(field)
        Z.BOUNDARY_SRC = shp
        Z.OUT_ROOT = OUT_ROOT
        try:
            Z.main()
            summary.append((field, "OK"))
        except SystemExit as e:
            print("  !!", e); summary.append((field, f"skip: {e}"))
        except Exception as e:
            print("  !! ERROR", e); summary.append((field, f"ERROR: {e}"))
        print()
    print("=" * 60, "\nПІДСУМОК:")
    for f, s in summary:
        print(f"  {f:12s} {s}")


if __name__ == "__main__":
    main()
