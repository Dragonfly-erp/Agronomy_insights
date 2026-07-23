"""
Common configuration and helper functions for the Sofiivka P1 field
management-zoning pipeline.

Field: "Геворг ПОСП" / Софіївка / П1  (~86 ha), near 50.52 N, 31.59 E, Ukraine.
Input: vectorised single-date NDVI (2024 bare-soil snapshot, 2025 full-canopy
snapshot), EPSG:4326.

All raster processing is done in UTM zone 36N (EPSG:32636), 5 m grid.
"""
import os
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, shape, mapping
from shapely.ops import unary_union
from rasterio.features import rasterize, shapes as rio_shapes
from rasterio.transform import from_origin
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- config
UTM = 32636          # WGS84 / UTM zone 36N  (metres)
WGS = 4326
RES = 5.0            # analysis grid resolution, metres
EDGE_BUFFER = 18.0   # inward erosion to drop headlands / field-edge effects, m

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # Sofiivka_P1_zoning/
# processing scratch (rasters/intermediate); final vectors go to shapefiles/
WORK = os.environ.get("ZONING_WORK",
                      "/tmp/claude-0/-home-user-Agronomy-insights/"
                      "353665f6-43ea-5915-ad8d-e06f82b9b1a2/scratchpad/work")
INPUT = os.path.join(WORK, "input")

def p(*a):
    return os.path.join(*a)

# ---------------------------------------------------------------- io helpers
def read_ndvi(year):
    """Return the raw NDVI polygons for a year, reprojected to UTM."""
    return gpd.read_file(p(INPUT, f"ndvi_{year}.shp")).to_crs(UTM)


def field_geometry():
    """Clean field outline = union of both NDVI footprints, holes filled,
    largest part kept. Returns a shapely Polygon in UTM."""
    geoms = []
    for yr in ("2024", "2025"):
        geoms.append(unary_union(read_ndvi(yr).geometry.values))
    foot = unary_union(geoms).buffer(3).buffer(-3)
    if isinstance(foot, MultiPolygon):
        foot = max(foot.geoms, key=lambda g: g.area)
    return Polygon(foot.exterior)


def grid_for(geom, res=RES):
    """Return (transform, width, height, bounds) covering geom on a res grid."""
    minx, miny, maxx, maxy = geom.bounds
    w = int(np.ceil((maxx - minx) / res))
    h = int(np.ceil((maxy - miny) / res))
    transform = from_origin(minx, maxy, res, res)
    return transform, w, h, (minx, miny, maxx, maxy)


def rasterize_field(gdf, value_col, transform, w, h, fill=np.nan):
    shapes = [(g, v) for g, v in zip(gdf.geometry, gdf[value_col])]
    return rasterize(shapes, out_shape=(h, w), transform=transform,
                     fill=fill, dtype="float32", all_touched=False)


def mask_from_geom(geom, transform, w, h):
    """Boolean raster mask (True inside geom)."""
    m = rasterize([(geom, 1)], out_shape=(h, w), transform=transform,
                  fill=0, dtype="uint8", all_touched=True)
    return m.astype(bool)


def polygonize(arr, transform, mask=None):
    """Vectorise an integer label raster to a GeoDataFrame (UTM).
    Returns columns [value, geometry]."""
    if mask is None:
        mask = ~np.isnan(arr)
    a = np.where(mask, arr, -9999).astype("int32")
    recs = []
    for geom, val in rio_shapes(a, mask=(a != -9999), transform=transform):
        recs.append({"value": int(val), "geometry": shape(geom)})
    gdf = gpd.GeoDataFrame(recs, crs=UTM)
    return gdf


def save_vec(gdf, path_noext, also_wgs=True):
    """Save a GeoDataFrame as shapefile (UTM) and optional WGS84 twin."""
    os.makedirs(os.path.dirname(path_noext), exist_ok=True)
    gdf.to_file(path_noext + ".shp")
    if also_wgs:
        gdf.to_crs(WGS).to_file(path_noext + "_wgs84.shp")
    return path_noext + ".shp"
