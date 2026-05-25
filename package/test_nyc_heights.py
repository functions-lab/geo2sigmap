"""Print building height source and value for a bbox (matches scenegen logic)."""
import warnings
warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS")

import osmnx as ox
from shapely.geometry import shape
from pyproj import Transformer

from scene_generation.utils import (
    init_overture_lookup,
    random_building_height,
    is_float,
    get_utm_epsg_code_from_gps,
)

# Your box: (lat, lon) SW → NE  →  min_lon, min_lat, max_lon, max_lat
MIN_LON, MIN_LAT = -73.988155, 40.756979
MAX_LON, MAX_LAT = -73.984597, 40.759673
BBOX = (MIN_LON, MIN_LAT, MAX_LON, MAX_LAT)

center_lon = (MIN_LON + MAX_LON) / 2
center_lat = (MIN_LAT + MAX_LAT) / 2

utm_epsg = get_utm_epsg_code_from_gps(center_lon, center_lat)
to_utm = Transformer.from_crs("EPSG:4326", utm_epsg, always_xy=True)
to_wgs84 = Transformer.from_crs(utm_epsg, "EPSG:4326", always_xy=True)

print("Loading Overture (may take ~1 min first time)...")
init_overture_lookup(bbox=BBOX, verbose=True)

print("\nFetching OSM buildings...")
gdf = ox.features.features_from_bbox(bbox=BBOX, tags={"building": True})
gdf = gdf.to_crs(utm_epsg)

def height_source(building: dict) -> str:
    if "building:height" in building and is_float(building["building:height"]):
        return "osm:building:height"
    if "height" in building and is_float(building["height"]):
        return "osm:height"
    if "building:levels" in building and is_float(building["building:levels"]):
        return "osm:building:levels"
    if "level" in building and is_float(building["level"]):
        return "osm:level"
    return "overture_or_random"


print(f"\n{'idx':>4}  {'source':<22}  {'height_m':>10}  osm_id")
print("-" * 55)

shown = 0
for idx, row in enumerate(gdf.to_dict("records")):
    geom = shape(row["geometry"])
    if geom.geom_type != "Polygon":
        continue

    src = height_source(row)
    h = random_building_height(row, geom, to_wgs84=to_wgs84)

    # Print first 20 buildings, plus any that used Overture/random
    if shown < 20 or src == "overture_or_random":
        osm_id = row.get("id", row.get("osmid", "?"))
        print(f"{idx:4d}  {src:<22}  {h:10.2f}  {osm_id}")
        shown += 1

print(f"\nTotal polygon buildings: {sum(1 for r in gdf.to_dict('records') if shape(r['geometry']).geom_type == 'Polygon')}")