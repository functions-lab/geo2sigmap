"""
Utilities sub-package for the Scene Generation library.
"""

import pyproj
from pyproj import Transformer, CRS

from shapely.geometry import (
    Polygon,
    MultiPolygon,
    LinearRing,
    Point
)

import numpy as np
import math
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds, transform
import rasterio


# -------------------------------------------------------------------
# 1) Geographic Coordinate System Related
# -------------------------------------------------------------------
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


def gps_to_utm_xy(lon: float, lat: float, utm_epsg: int):
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
    rounded_exterior = LinearRing([
        (round(x, decimal_places), round(y, decimal_places))
        for x, y in polygon.exterior.coords
    ])
    rounded_interiors = [
        LinearRing([
            (round(x, decimal_places), round(y, decimal_places))
            for x, y in interior.coords
        ])
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
    if geometry.geom_type == 'Polygon':
        return round_polygon_coordinates(geometry, decimal_places)
    elif geometry.geom_type == 'MultiPolygon':
        return MultiPolygon([
            round_polygon_coordinates(poly, decimal_places)
            for poly in geometry
        ])
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

def random_building_height(building: dict, building_polygon: Polygon) -> float:
    """
    Determine a building's height from OSM tags if available, else random.

    Parameters
    ----------
    building : dict
        A record (row) from an OSM data source containing building attributes,
        e.g. 'building:height', 'height', 'building:levels', etc.
    building_polygon : Polygon
        The polygon geometry of this building (unused in this function's fallback).

    Returns
    -------
    float
        The estimated building height in meters.
    """
    if 'building:height' in building and is_float(building['building:height']):
        building_height = float(building['building:height'])
    elif 'height' in building and is_float(building['height']):
        building_height = float(building['height'])
    elif 'building:levels' not in building or not is_float(building['building:levels']):
        # Fallback random height (units: meters)
        building_height = 3.5 * max(1, min(15, int(np.random.normal(loc=5, scale=1))))
    elif 'level' not in building or not is_float(building['level']):
        building_height = 3.5 * max(1, min(15, int(np.random.normal(loc=5, scale=1))))
    else:
        building_height = float(building['building:levels']) * 3.5

    return building_height


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
            self.src.crs, "EPSG:4326",
            bounds.left, bounds.bottom,
            bounds.right, bounds.top
        )
        logger.info(
            "GPS Bounds [lon_min, lat_min, lon_max, lat_max]: %s", gps_bounds
        )

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
            {'init': 'epsg:4326'}, self.src.crs,
            [gps_coordinate[0]], [gps_coordinate[1]]
        )
        x, y = transformed_coordinates[0][0], transformed_coordinates[1][0]

        # Convert to pixel row/col
        row, col = self.src.index(x, y)

        # Read the single pixel at (row, col)
        hag_value = self.src.read(
            1,
            window=rasterio.windows.Window(col, row, 1, 1),
            resampling=Resampling.nearest
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