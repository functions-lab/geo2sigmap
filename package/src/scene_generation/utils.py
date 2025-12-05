"""
Utilities sub-package for the Scene Generation library.
"""

import pyproj
from pyproj import Transformer, CRS

from shapely.geometry import Polygon, MultiPolygon, LinearRing, Point

import numpy as np
import math
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds, transform
import rasterio

from typing import List, Tuple


import json
from typing import Optional, Dict, Any
import pdal
import shapely
from shapely import wkt


import logging

# Creates a logger named 'duke_lidar.height'
logger = logging.getLogger(__name__)


try:
    from importlib.metadata import version as pkg_version, PackageNotFoundError
except ImportError:
    # For Python < 3.8, use importlib_metadata backport
    from importlib_metadata import version as pkg_version, PackageNotFoundError
import math

PACKAGE_NAME = "scene_generation"


def get_package_version() -> str:
    """
    Attempt to retrieve the installed package version from metadata.
    Falls back to a default if the package isn't found (not installed).
    """
    try:
        return pkg_version(PACKAGE_NAME)
    except PackageNotFoundError:
        return "0.0.0.dev (uninstalled)"


# -------------------------------------------------------------------
# 1) Geographic Coordinate System Related
# -------------------------------------------------------------------


def top_left_to_center(x, y, width, height):
    """
    Converts coordinates from a top-left origin system to a center-origin system.

    Parameters:
    x (float): X-coordinate in the top-left system.
    y (float): Y-coordinate in the top-left system.
    width (float): Total width of the coordinate space.
    height (float): Total height of the coordinate space.

    Returns:
    (float, float): Transformed (x', y') in the center-origin system.
    """
    x_center = x - width / 2
    y_center = (height / 2) - y
    return x_center, y_center


def get_utm_epsg_code_from_gps(lon: float, lat: float) -> CRS:
    """
    Determine the UTM coordinate reference system (CRS) appropriate for a given
    longitude/latitude using WGS84 as the datum.

    This function queries pyproj's database for the UTM zone that best fits
    the point of interest (defined by lon/lat).

    Parameters:
    ----------
    lon : float
        Longitude in decimal degrees.
    lat : float
        Latitude in decimal degrees.

    Returns:
    -------
    utm_crs : CRS
        A pyproj CRS object representing the best matching UTM projection
        (e.g., EPSG:32633).
    """

    # Query for possible UTM CRS definitions covering our point of interest
    utm_crs_list = pyproj.database.query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=pyproj.aoi.AreaOfInterest(
            west_lon_degree=lon,
            south_lat_degree=lat,
            east_lon_degree=lon,
            north_lat_degree=lat,
        ),
    )
    # Typically, the first element is the most relevant match
    utm_crs = pyproj.CRS.from_epsg(utm_crs_list[0].code)
    return utm_crs


def gps_to_utm_xy(lon: float, lat: float, utm_epsg):
    """
    Convert GPS coordinates (longitude, latitude) in WGS84 to UTM coordinates.

    Parameters:
    ----------
    lon : float
        Longitude in decimal degrees (WGS84).
    lat : float
        Latitude in decimal degrees (WGS84).
    utm_epsg : int
        The EPSG code for the desired UTM zone (e.g., 32633).

    Returns:
    -------
    (utm_x, utm_y, epsg_code) : (float, float, int)
        utm_x  : easting in the specified UTM zone
        utm_y  : northing in the specified UTM zone
        epsg_code : same as the input `utm_epsg`, returned for convenience
    """

    # Create a transformer from WGS84 (EPSG:4326) to the specified UTM zone
    transformer = Transformer.from_crs("EPSG:4326", utm_epsg, always_xy=True)

    # Transform (longitude, latitude) into (easting, northing) in the UTM zone
    utm_x, utm_y = transformer.transform(lon, lat)

    # Return the results, including the EPSG code for clarity
    return (utm_x, utm_y, utm_epsg)


def rect_from_point_and_size(
    lon: float, lat: float, position: str, width: float, height: float
) -> List[Tuple[float, float]]:
    """
    Create a rectangular polygon (as a list of coordinates) given a GPS point and
    desired rectangle size in a UTM projection.

    :param lon: Longitude of the reference point (in EPSG:4326).
    :param lat: Latitude of the reference point (in EPSG:4326).
    :param position: One of the following strings:
                     ["top-left", "top-right", "bottom-left", "bottom-right", "center"]
                     indicating how (lon, lat) is interpreted relative to the rectangle.
    :param width:  The width of the rectangle in UTM projection units (e.g., meters).
    :param height: The height of the rectangle in UTM projection units (e.g., meters).
    :return:       A list of (longitude, latitude) coordinates (EPSG:4326)
                   forming the rectangle boundary. The last point is repeated
                   to close the polygon.

    .. note::
       This function currently does NOT handle edge cases such as crossing
       the International Date Line or spanning multiple UTM zones. Those must
       be addressed separately.
    """

    # TODO: Check for the coner case such as rossing the International Date Line at ±180°, crossing multiple UTM zones

    # Get the UTM EPSG code based on the given longitude/latitude.
    utm_epsg = get_utm_epsg_code_from_gps(lon, lat)

    # Convert the reference GPS point to UTM (x, y).
    point_utm = gps_to_utm_xy(lon, lat, utm_epsg)

    # Prepare a transformer to go from UTM back to EPSG:4326.
    transformer = Transformer.from_crs(utm_epsg, "EPSG:4326", always_xy=True)

    if position == "top-left":
        min_lon_left = point_utm[0]
        max_lon_right = point_utm[0] + width
        max_lat_top = point_utm[1]
        min_lat_bottom = point_utm[1] - height

    elif position == "top-right":

        min_lon_left = point_utm[0] - width
        max_lon_right = point_utm[0]
        max_lat_top = point_utm[1]
        min_lat_bottom = point_utm[1] - height

    elif position == "bottom-right":

        min_lon_left = point_utm[0] - width
        max_lon_right = point_utm[0]
        max_lat_top = point_utm[1] + height
        min_lat_bottom = point_utm[1]

    elif position == "bottom-left":

        min_lon_left = point_utm[0]
        max_lon_right = point_utm[0] + width
        max_lat_top = point_utm[1] + height
        min_lat_bottom = point_utm[1]

    elif position == "center":
        min_lon_left = point_utm[0] - width / 2
        max_lon_right = point_utm[0] + width / 2
        max_lat_top = point_utm[1] + height / 2
        min_lat_bottom = point_utm[1] - height / 2

    else:
        raise ValueError(
            f"Unknown position: {position}. "
            "Must be one of [top-left, top-right, bottom-left, bottom-right, center]."
        )

    points_utm = [
        [min_lon_left, min_lat_bottom],
        [min_lon_left, max_lat_top],
        [max_lon_right, max_lat_top],
        [max_lon_right, min_lat_bottom],
        [min_lon_left, min_lat_bottom],
    ]

    points_gps = [transformer.transform(x, y) for x, y in points_utm]
    return points_gps


# -------------------------------------------------------------------
# 2) Polygon/Coordinates Related
# -------------------------------------------------------------------
def round_polygon_coordinates(polygon: Polygon, decimal_places: int = 0) -> Polygon:
    """
    Round the exterior and interior coordinates of a single Polygon to the specified
    number of decimal places.

    Parameters
    ----------
    polygon : Polygon
        A shapely Polygon whose coordinates should be rounded.
    decimal_places : int, optional
        Number of decimal places to round to (default 0 = integer rounding).

    Returns
    -------
    Polygon
        A new Polygon with rounded exterior and interior coordinates.
    """
    rounded_exterior = LinearRing(
        [
            (round(x, decimal_places), round(y, decimal_places))
            for x, y in polygon.exterior.coords
        ]
    )
    rounded_interiors = [
        LinearRing(
            [
                (round(x, decimal_places), round(y, decimal_places))
                for x, y in interior.coords
            ]
        )
        for interior in polygon.interiors
    ]
    return Polygon(rounded_exterior, rounded_interiors)


def round_geometry_coords(geometry, decimal_places: int = 0):
    """
    Round the coordinates of a geometry (Polygon or MultiPolygon) to the
    specified number of decimal places.

    Parameters
    ----------
    geometry : Polygon or MultiPolygon
        Shapely geometry whose coordinates should be rounded.
    decimal_places : int, optional
        Number of decimal places for rounding (default 0 = integer).

    Returns
    -------
    Polygon or MultiPolygon
        The same geometry type with rounded coordinates.
    """
    if geometry.geom_type == "Polygon":
        return round_polygon_coordinates(geometry, decimal_places)
    elif geometry.geom_type == "MultiPolygon":
        return MultiPolygon(
            [round_polygon_coordinates(poly, decimal_places) for poly in geometry]
        )
    else:
        # If not a Polygon or MultiPolygon, return unchanged
        return geometry


def generate_random_points(poly: Polygon, num_points: int):
    """
    Generate a given number of random points that lie within the Polygon (including holes).

    Parameters
    ----------
    poly : Polygon
        The polygon in which to generate random points.
    num_points : int
        Number of random points to generate.

    Returns
    -------
    list of Point
        A list of shapely Point objects within the polygon.
    """
    min_x, min_y, max_x, max_y = poly.bounds
    points = []
    while len(points) < num_points:
        rand_x = np.random.uniform(min_x, max_x)
        rand_y = np.random.uniform(min_y, max_y)
        random_point = Point(rand_x, rand_y)
        if random_point.within(poly):
            points.append(random_point)
    return points


def unique_coords(input_coords):
    """
    Given a list of (x, y) coordinates, return a new list with duplicate
    coordinates removed, preserving the original order of first occurrences.

    Parameters
    ----------
    input_coords : list of (float, float)
        A list of 2D coordinate pairs.

    Returns
    -------
    list of (float, float)
        The same coordinates but with duplicates removed in order of appearance.
    """
    unique_coords_res = []
    seen_coords = set()
    for coord in input_coords:
        if coord not in seen_coords:
            unique_coords_res.append(coord)
            seen_coords.add(coord)
    return unique_coords_res


def reorder_localize_coords(input_coords, center_x: float, center_y: float):
    """
    Reverse coordinates if polygon is counterclockwise, then translate
    them relative to a given center.

    Parameters
    ----------
    input_coords : LinearRing or Sequence of coordinates
        A shapely LinearRing or any sequence of (x, y) coords.
        Must support `.is_ccw` and `.reverse()`, or adapt as needed.
    center_x : float
        X coordinate to translate from.
    center_y : float
        Y coordinate to translate from.

    Returns
    -------
    list of (float, float)
        The re-ordered, localized (translated) coordinates.
    """
    # If the ring is in CCW order, reverse it so we have consistent winding
    if hasattr(input_coords, "is_ccw") and input_coords.is_ccw:
        input_coords.reverse()

    # Translate coords to local origin at (center_x, center_y)
    res_coords = [
        (coord[0] - center_x, coord[1] - center_y)
        for coord in list(input_coords.coords)
    ]
    return res_coords


import hashlib

# import numpy as np

# A pre-calculated lookup table representing a Normal Distribution (Mean=5, Std=1)
# We map a uniform random integer (0-99) to a building height.
# Probabilities:
# Level 3: ~6%  (Indices 0-5)
# Level 4: ~24% (Indices 6-29)
# Level 5: ~38% (Indices 30-67)  <- Peak of the bell curve
# Level 6: ~24% (Indices 68-91)
# Level 7: ~6%  (Indices 92-97)
# Outliers (2, 8): ~2% (Indices 98-99)
HEIGHT_LOOKUP_TABLE = np.array(
    [3] * 6 + [4] * 24 + [5] * 38 + [6] * 24 + [7] * 6 + [2, 8]
)

# def random_building_height(building: dict,  building_polygon: Polygon) -> float:
#     """
#     Returns a building height that is GUARANTEED to be identical
#     across all operating systems, CPUs, and Python versions.
#     """

#     # ... [Explicit tag checks same as before] ...
#     if 'building:height' in building and is_float(building['building:height']):
#         return float(building['building:height'])

#     # --- Deterministic Fallback ---

#     # 1. Get the ID (String consistency is key)
#     osm_id = str(building.get('id', building.get('osmid', 0)))

#     # 2. Hash it (MD5 is standard across all platforms)
#     id_hash = hashlib.md5(osm_id.encode('utf-8')).hexdigest()

#     # 3. Convert to Integer Seed
#     # We strip to the last 8 chars to ensure it fits in 32-bit int safely
#     seed = int(id_hash[-8:], 16)

#     # 4. Use Legacy RandomState (Guaranteed stability)
#     rng = np.random.RandomState(seed)

#     # 5. INTEGER ONLY GENERATION
#     # We pick an index from 0 to 99. This uses only integer logic.
#     idx = rng.randint(0, 100)

#     # 6. Map to Height
#     levels = HEIGHT_LOOKUP_TABLE[idx]

#     return float(levels) * 3.5

# def is_float(value):
#     try:
#         float(value)
#         return True
#     except:
#         return False


def estimate_building_height_from_osm(
    building: dict
) -> float:
    """
    Determine a building's height from OSM tags if available, else random.

    Parameters
    ----------
    building : dict
        A record (row) from an OSM data source containing building attributes,
        e.g. 'building:height', 'height', 'building:levels', etc.

    Returns:
        dict: {
            "building_height": float,
            "method": str ("osm_tag_X", "osm_fallback_random"),
        }
    """

    # --- 1. Explicit Height (Best Quality) ---
    if "building:height" in building and is_float(building["building:height"]):
        res = float(building["building:height"])
        return {
            "building_height": res,
            "method": "osm_tag_building:height",
        }

    if "height" in building and is_float(building["height"]):
        res = float(building["height"])
        return {
            "building_height": res,
            "method": "osm_tag_height",
        }

    # --- 2. Explicit Levels (Medium Quality) ---
    # Standard OSM tag is 'building:levels'.
    if "building:levels" in building and is_float(building["building:levels"]):
        res = float(building["building:levels"]) * 3.5
        return {
            "building_height": res,
            "method": "osm_tag_building:levels",
        }

    # Some mappers just use 'levels'
    if "levels" in building and is_float(building["levels"]):
        res = float(building["levels"]) * 3.5
        return {
            "building_height": res,
            "method": "osm_tag_levels",
        }

    # --- Deterministic Fallback ---

    # 1. Get the ID (String consistency is key)
    osm_id = str(building.get("id", building.get("osmid", 0)))

    logger.debug(
        f"Building (OSM way ID: {osm_id}) missing height or level tags. "
        "Falling back to random height."
    )

    # 2. Hash it (MD5 is standard across all platforms)
    id_hash = hashlib.md5(osm_id.encode("utf-8")).hexdigest()

    # 3. Convert to Integer Seed
    # We strip to the last 8 chars to ensure it fits in 32-bit int safely
    seed = int(id_hash[-8:], 16)

    # 4. Use Legacy RandomState (Guaranteed stability)
    rng = np.random.RandomState(seed)

    # 5. INTEGER ONLY GENERATION
    # We pick an index from 0 to 99. This uses only integer logic.
    idx = rng.randint(0, 100)

    # 6. Map to Height
    levels = HEIGHT_LOOKUP_TABLE[idx]

    res = float(levels) * 3.5

    return {
        "building_height": res,
        "method": "osm_fallback_random",
    }


def calculate_building_height_from_lidar(
    laz_path: str,
    polygon_wkt: str,
    osm_id: Optional[int] = None,
    ground_buffer: float = 2.0,
    roof_erosion: float = -1.5,
) -> Optional[Dict[str, float]]:
    """
    Estimates building height from LiDAR data using robust geometric and statistical filtering.

    This function implements a "Superset/Subset" approach to solve the urban canyon problem.
    It reads a single buffered crop of the LiDAR data to minimize I/O overhead, then
    uses vectorized geometric operations to separate ground context from roof points.

    The height is calculated as: Height = Mode(Roof Z) - Median(Ground Z).

    Args:
        laz_path (str): The file path to the input .laz or .las file.
        polygon_wkt (str): The building footprint in Well-Known Text (WKT) format.
            Must use the same coordinate reference system (CRS) as the LAZ file.
        osm_id (Optional[int], optional): The OSM way ID of the building polygon.
            Used for logging and debugging purposes. Defaults to None.
        ground_buffer (float, optional): The distance in meters to expand (dilate)
            the polygon to capture surrounding ground points. Defaults to 2.0.
            Note: Values < 1.0 may fail to capture ground in sparse datasets.
        roof_erosion (float, optional): The distance in meters to shrink (erode)
            the polygon for roof point extraction. Defaults to -1.5.
            A negative value is required. This erosion removes points near the
            building edge, filtering out vertical wall points and interference
            from adjacent taller structures.

    Returns:
        Optional[Dict[str, float]]: A dictionary containing height metrics, or None
        if insufficient points were found. The dictionary keys are:
            - 'ground_mean': The median elevation of the surrounding ground.
            - 'roof_z': The estimated elevation of the roof (calculated via Mode/95th voting).
            - 'height': The calculated building height (roof_z - ground_mean).
            - 'method': The statistical method used for the roof ('mode' or '95th').

    Raises:
        pdal.PipelineError: If the PDAL pipeline fails to execute (e.g., file not found).
        shapely.errors.WKTReadingError: If the input WKT string is invalid.
    """

    # --- 1. Geometry Preparation ---
    try:
        poly_geom = wkt.loads(polygon_wkt)
    except Exception as e:
        # Re-raising with context is helpful for library users debugging bad inputs
        raise ValueError(f"Invalid WKT provided: {e}")

    # Create the 'Superset' polygon: Used to fetch all necessary context (Ground + Roof)
    poly_ground_geom = poly_geom.buffer(ground_buffer)
    poly_ground_wkt = poly_ground_geom.wkt

    # Create the 'Subset' polygon: Used to strictly isolate the roof core
    # We erode the polygon to avoid "bleeding" points from adjacent tall walls
    poly_roof_geom = poly_geom.buffer(roof_erosion)

    # Fallback: If erosion removes the entire polygon (e.g., small shed < 3m wide),
    # revert to the original footprint to ensure we get *some* data.
    if poly_roof_geom.is_empty:
        poly_roof_geom = poly_geom
        # In a logging environment, we would use logger.warning() here
        logger.warning(
            f"Building polygon (OSM way ID: {osm_id}) became empty after erosion ({roof_erosion}m). "
            "Falling back to original footprint."
        )

    # Optimization: 'prepare' accelerates subsequent vectorized queries in Shapely 2.0+
    shapely.prepare(poly_roof_geom)

    # --- 2. Single-Pass PDAL Execution (I/O Bound) ---
    # We define a pipeline to crop only the expanded area.
    # This prevents reading the entire file or running the pipeline twice.
    pipeline_json = {
        "pipeline": [laz_path, {"type": "filters.crop", "polygon": poly_ground_wkt}]
    }

    pipeline = pdal.Pipeline(json.dumps(pipeline_json))

    # Execute the pipeline. This is the most computationally expensive step.
    pipeline.execute()
    arr = pipeline.arrays[0]

    # Early exit if the crop yielded no points (e.g., wrong coordinates)
    if len(arr) == 0:
        return None

    # --- 3. Data Segmentation (CPU Bound) ---

    # A. Ground Extraction Strategy
    # We rely on ASPRS Standard Class 2 (Ground).
    # The crop is already buffered, so these are the points immediately surrounding the building.
    ground_points = arr[arr["Classification"] == 2]

    if len(ground_points) == 0:
        # Fallback: Assume 0.0 or handle based on specific project requirements.
        # Returning 0.0 allows the script to continue, but data quality should be flagged.
        ground_z_mean = 0.0
    else:
        # Use Median instead of Mean to be robust against underground noise/basement reflections
        ground_z_mean = np.median(ground_points["Z"])

    # B. Roof Extraction Strategy
    # First, filter out known ground points
    non_ground_subset = arr[arr["Classification"] != 2]

    if len(non_ground_subset) == 0:
        return None

    # Vectorized Point-in-Polygon Check
    # shapely.contains_xy is a ufunc (Universal Function) optimized in C.
    # It creates a boolean mask much faster than iterating through Point objects.
    mask_inside_roof = shapely.contains_xy(
        poly_roof_geom, non_ground_subset["X"], non_ground_subset["Y"]
    )

    actual_roof_points = non_ground_subset[mask_inside_roof]

    if len(actual_roof_points) == 0:
        return None

    roof_z = actual_roof_points["Z"]

    # --- 4. Statistical Height Determination ---

    # Calculate the 95th Percentile (Standard approach, sensitive to outliers)
    z_95 = np.percentile(roof_z, 95)

    # Calculate the Mode (Robust approach)
    # Roofs are flat surfaces (high density), walls are vertical (low density).
    # We bin Z values by 20cm to find the "densest" vertical slice.
    bins = np.arange(np.min(roof_z), np.max(roof_z) + 0.2, 0.2)
    hist, bin_edges = np.histogram(roof_z, bins=bins)

    peak_idx = np.argmax(hist)
    z_mode = (bin_edges[peak_idx] + bin_edges[peak_idx + 1]) / 2.0

    # Voting Logic:
    # If the 95th percentile is > 3.0m higher than the mode, it indicates
    # we likely caught a "spike" (noise) or a neighbor's wall.
    # In that case, the Mode is the safer physical estimate.
    if (z_95 - z_mode) > 3.0:
        final_roof_z = z_mode
        method = "mode"
    else:
        final_roof_z = z_95
        method = "95th"

    return {
        "ground_mean": float(ground_z_mean),
        "roof_z": float(final_roof_z),
        "building_height": float(final_roof_z - ground_z_mean),
        "method": method,
    }


def get_center_subarray(arr, x, y):
    """
    Extracts the center subarray of size (x, y) from a 2D NumPy array.

    Parameters:
    - arr: 2D NumPy array
    - x: Number of rows in the subarray
    - y: Number of columns in the subarray

    Returns:
    - Center subarray of shape (x, y)
    """
    H, W = arr.shape  # Get original array size

    # Compute starting indices
    start_x = (H - x) // 2
    start_y = (W - y) // 2

    # Extract the subarray
    return arr[start_x : start_x + x, start_y : start_y + y]


def error_tolerance_rate(y_true, y_pred, threshold, relative=False):
    """
    Computes the percentage of predictions that fall within a specified error threshold.

    This function calculates the proportion of predictions that have errors within the given threshold,
    supporting both absolute and relative error calculations.

    Parameters:
    ----------
    y_true : np.ndarray
        Array of actual (ground truth) values.
    y_pred : np.ndarray
        Array of predicted values.
    threshold : float
        The maximum allowed error for a prediction to be considered within tolerance.
    relative : bool, optional (default=False)
        If True, computes relative errors (normalized by `y_true`). If False, uses absolute errors.

    Returns:
    -------
    float
        The percentage of predictions within the specified error tolerance.

    Notes:
    ------
    - If `relative=True`, zero values in `y_true` are ignored to avoid division by zero.
    - The function returns a percentage (0 to 100) rather than a fraction.
    """

    if relative:
        # Mask to exclude zero values in `y_true` to prevent division by zero
        valid_mask = y_true != 0
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]

        # Compute relative errors
        errors = np.abs((y_true - y_pred) / y_true)
    else:
        # Compute absolute errors
        errors = np.abs(y_true - y_pred)

    # Count predictions within the specified error threshold
    within_tolerance = np.sum(errors <= threshold)

    # Return the percentage of values within tolerance
    return (within_tolerance / len(y_true)) * 100 if len(y_true) > 0 else 0.0


def is_float(element) -> bool:
    """
    Check if `element` can be safely cast to a float and is not NaN or inf.

    Parameters
    ----------
    element : any
        The value to check.

    Returns
    -------
    bool
        True if element is a valid float, otherwise False.
    """
    if element is None:
        return False
    try:
        val = float(element)
        return not (math.isnan(val) or math.isinf(val))
    except (TypeError, ValueError):
        return False


def print_if_int(num):
    """Prints the number as an integer if it's an integer, otherwise prints the original number."""
    if math.isclose(num, int(num)):
        return int(num)
    else:
        return num


class GeoTIFFHandler:
    """
    Class for opening and querying a GeoTIFF file for height (HAG, DEM, etc.).
    """

    def __init__(self, filepath: str):
        """
        Parameters
        ----------
        filepath : str
            Path to the GeoTIFF file.
        """
        self.filepath = filepath
        self.src = self._open_geotiff()

    def _open_geotiff(self):
        """
        Open the rasterio dataset from the given file.

        Returns
        -------
        rasterio.io.DatasetReader
            An open raster dataset.
        """
        return rasterio.open(self.filepath)

    def get_info(self):
        """
        Print or return some metadata info about the GeoTIFF.
        This includes bounds, nodata values, CRS, etc.
        """
        logger.info("Metadata: %s", self.src.meta)
        logger.info("NoData Value: %s", self.src.nodatavals)

        bounds = self.src.bounds
        gps_bounds = transform_bounds(
            self.src.crs,
            "EPSG:4326",
            bounds.left,
            bounds.bottom,
            bounds.right,
            bounds.top,
        )
        logger.info("GPS Bounds [lon_min, lat_min, lon_max, lat_max]: %s", gps_bounds)

    def query(self, gps_coordinate, reverse_xy: bool = False):
        """
        Query the GeoTIFF for height data at the given GPS coordinate.

        Parameters
        ----------
        gps_coordinate : (float, float)
            (longitude, latitude) in decimal degrees. Or (lat, lon) if reverse_xy is True.
        reverse_xy : bool, optional
            If True, interpret gps_coordinate as (lat, lon).

        Returns
        -------
        float or np.ndarray
            The height value(s) from the raster at the nearest pixel.
        """
        if reverse_xy:
            gps_coordinate = (gps_coordinate[1], gps_coordinate[0])

        # Transform from WGS84 to raster's CRS
        transformed_coordinates = transform(
            {"init": "epsg:4326"},
            self.src.crs,
            [gps_coordinate[0]],
            [gps_coordinate[1]],
        )
        x, y = transformed_coordinates[0][0], transformed_coordinates[1][0]

        # Convert to pixel row/col
        row, col = self.src.index(x, y)

        # Read the single pixel at (row, col)
        hag_value = self.src.read(
            1,
            window=rasterio.windows.Window(col, row, 1, 1),
            resampling=Resampling.nearest,
        )
        return hag_value.squeeze()

    def __del__(self):
        """
        Ensure the dataset is closed when the object is deleted.
        """
        try:
            self.src.close()
        except AttributeError:
            pass
