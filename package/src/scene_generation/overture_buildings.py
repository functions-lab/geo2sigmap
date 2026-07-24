"""
Helpers for loading Overture Maps building and part footprints for a scene AOI.

The scene generator already works in two coordinate spaces:
  * EPSG:4326 for external data lookups.
  * A local UTM CRS for mesh generation and footprint matching.

This module keeps the Overture query in EPSG:4326, then optionally reprojects
the returned GeoDataFrame to the caller's target CRS.
"""

from __future__ import annotations

import logging
import math
import re
import duckdb
from typing import Dict, Optional, Sequence, Tuple, Union

import geopandas as gpd
import numpy as np
import urllib.request
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from .utils import generate_random_points

logger = logging.getLogger(__name__)

OVERTURE_BUILDINGS_S3_TEMPLATE = (
    "s3://overturemaps-us-west-2/release/{release}/theme=buildings/type={feature_type}/*"
)
OVERTURE_BUILDINGS_AZURE_TEMPLATE = (
    "az://overturemapswestus2.blob.core.windows.net/release/"
    "{release}/theme=buildings/type={feature_type}/*"
)
OVERTURE_BUILDING_TYPE = "building"
OVERTURE_BUILDING_PART_TYPE = "building_part"

# method from Josef Ullrich's work
def get_latest_release_url(con, feature):
        """
        Query Overture STAC catalog for latest release URL.

        Note:
            Overture's docs recommend using the STAC catalog to find the
            latest release instead of hardcoding a release path.
        """

        latest = con.execute(
            "SELECT latest FROM 'https://stac.overturemaps.org/catalog.json'"
        ).fetchone()[0]

        release = str(latest).strip()

        return _overture_buildings_path(release, "s3", feature)

def load_overture_buildings_for_aoi(
    bbox_4326: Sequence[float],
    target_crs=None,
    *,
    source: str = "s3",
    parquet_path: Optional[str] = None,
    duckdb_connection=None,
) -> gpd.GeoDataFrame:
    """
    Load Overture building footprints intersecting a WGS84 bounding box.

    Parameters
    ----------
    bbox_4326
        Bounding box in ``(min_lon, min_lat, max_lon, max_lat)`` order.
    target_crs
        Optional CRS to reproject the returned GeoDataFrame into. Pass the
        scene's UTM CRS when creating local meshes.
    source
        Cloud source for the default path. Supported values: ``"s3"`` and
        ``"azure"``. Default: ``"s3"``.
    parquet_path
        Override path for testing or pinned local/cloud data. This should point
        at the Overture buildings GeoParquet partition.
    duckdb_connection
        Optional existing DuckDB connection, mostly useful for tests.

    Returns
    -------
    geopandas.GeoDataFrame
        Overture buildings in EPSG:4326, or ``target_crs`` when provided.
        Columns include ``id``, ``height``, ``num_floors``, ``min_height``,
        ``min_floor``, ``roof_height``, ``subtype``, ``class``, ``has_parts``,
        ``is_underground``, ``overture_feature_type``, and ``geometry`` when
        present in the source schema.
    """

    return _load_overture_building_features_for_aoi(
        bbox_4326,
        target_crs,
        source=source,
        parquet_path=parquet_path,
        duckdb_connection=duckdb_connection,
        feature_type=OVERTURE_BUILDING_TYPE,
    )


def load_overture_building_parts_for_aoi(
    bbox_4326: Sequence[float],
    target_crs=None,
    *,
    source: str = "s3",
    parquet_path: Optional[str] = None,
    duckdb_connection=None,
) -> gpd.GeoDataFrame:
    """
    Load Overture building part footprints intersecting a WGS84 bounding box.

    Building parts are associated with parent buildings by ``building_id`` and
    may carry their own ``height``/``num_floors`` plus ``min_height``/``min_floor``
    vertical offsets for stacked or floating geometry.
    """

    return _load_overture_building_features_for_aoi(
        bbox_4326,
        target_crs,
        source=source,
        parquet_path=parquet_path,
        duckdb_connection=duckdb_connection,
        feature_type=OVERTURE_BUILDING_PART_TYPE,
    )


def _load_overture_building_features_for_aoi(
    bbox_4326: Sequence[float],
    target_crs=None,
    *,
    release: str = None,
    source: str,
    parquet_path: Optional[str],
    duckdb_connection,
    feature_type: str,
) -> gpd.GeoDataFrame:
    min_lon, min_lat, max_lon, max_lat = _normalize_bbox_4326(bbox_4326)

    con, owns_connection = _get_duckdb_connection(duckdb_connection)
    if release is None:
        release = get_latest_release_url(con, feature_type)
    print(f"Using Overture release: {release}")
    path = parquet_path or release
    select_columns = _overture_select_columns(feature_type)
    try:
        _load_duckdb_extensions(con)
        if source == "s3" and parquet_path is None:
            con.execute("SET s3_region='us-west-2'")
            con.execute("SET s3_url_style='vhost'")
            con.execute("SET s3_use_ssl=true")
            con.execute("SET s3_requester_pays=false")

        query = f"""
            SELECT
                {select_columns},
                bbox.xmin AS bbox_xmin,
                bbox.ymin AS bbox_ymin,
                bbox.xmax AS bbox_xmax,
                bbox.ymax AS bbox_ymax,
                geometry AS geometry_wkb
            FROM read_parquet(?, hive_partitioning=1)
            WHERE
                bbox.xmin <= ?
                AND bbox.xmax >= ?
                AND bbox.ymin <= ?
                AND bbox.ymax >= ?
                AND COALESCE(is_underground, false) = false
        """

        df = con.execute(
            query,
            [path, max_lon, min_lon, max_lat, min_lat],
        ).fetchdf()
    finally:
        if owns_connection:
            con.close()

    if df.empty:
        return gpd.GeoDataFrame(
            df.drop(columns=["geometry_wkb"], errors="ignore"),
            geometry=[],
            crs="EPSG:4326",
        )

    geometry_wkb = df.pop("geometry_wkb").map(_duckdb_wkb_to_bytes)
    geometry = gpd.GeoSeries.from_wkb(geometry_wkb, crs="EPSG:4326")
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    # The bbox predicate is intentionally broad for Parquet pruning. Do an exact
    # geometry intersection locally before returning candidates for footprint
    # matching.
    aoi = box(min_lon, min_lat, max_lon, max_lat)
    gdf = gdf[gdf.intersects(aoi)].copy()
    gdf = _explode_polygonal_geometries(gdf)

    if target_crs is not None and not gdf.empty:
        gdf = gdf.to_crs(target_crs)

    logger.info("Loaded %d Overture %s candidates", len(gdf), feature_type)
    return gdf


def _overture_select_columns(feature_type: str) -> str:
    common_columns = """
                id,
                building_id,
                height,
                num_floors,
                min_height,
                min_floor,
                roof_height,
                is_underground
    """
    if feature_type == OVERTURE_BUILDING_TYPE:
        return f"""
                'building' AS overture_feature_type,
                NULL::VARCHAR AS building_id,
                subtype,
                class,
                has_parts,
                {common_columns}
        """
    if feature_type == OVERTURE_BUILDING_PART_TYPE:
        return f"""
                'building_part' AS overture_feature_type,
                building_id,
                NULL::VARCHAR AS subtype,
                NULL::VARCHAR AS class,
                false AS has_parts,
                {common_columns}
        """
    raise ValueError(f"Unsupported Overture buildings feature type: {feature_type}")


def _explode_polygonal_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf

    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    gdf = gdf[gdf.geometry.geom_type.isin(("Polygon", "MultiPolygon"))].copy()
    if gdf.empty:
        return gdf

    try:
        exploded = gdf.explode(index_parts=False, ignore_index=True)
    except TypeError:
        exploded = gdf.explode(index_parts=False).reset_index(drop=True)
    return exploded[exploded.geometry.geom_type == "Polygon"].reset_index(drop=True)


def resolve_building_height(
    building: dict,
    building_polygon: BaseGeometry,
    *,
    hag_handler=None,
    to_4326=None,
    floor_height_m: float = 3.5,
    hag_sample_count: int = 30,
    min_hag_height_m: float = 2.0,
    lidar_height_calibration: bool,
    return_source: bool = False,
) -> Union[float, Tuple[float, Dict[str, object]]]:
    """
    Resolve the extrusion height for one building footprint.

    Parameters
    ----------
    building
        Building record from ``GeoDataFrame.to_dict("records")``, either OSM record or Overture record, depending on the data source used.
    building_polygon
        Building footprint in the scene's projected CRS.
    hag_handler
        Optional ``GeoTIFFHandler`` used for LiDAR height-above-ground samples.
    to_4326
        Transformer from the scene CRS to EPSG:4326. Required with
        ``hag_handler`` because ``GeoTIFFHandler.query`` expects GPS coords.
    floor_height_m : float, optional
        Multiplier (in m) for number of floors to obtain height. Default: 3.5.
    hag_sample_count : int, optional
        Number of points to sample from LiDAR HAG. Default: 30.
    min_hag_height_m : float, optional
        Minimum height above ground in meters to be considered a valid sample when performing LiDAR HAG averaging. Default: 2.0.
    lidar_height_calibration : boolean
        If True, use LiDAR as a step in the height determination hierarchy.
    return_source
        If True, return ``(height, metadata)``. Otherwise return just height.
    """

    for source_type in _height_mode_steps(lidar_height_calibration):
        height, source, metadata = _height_from_source(
            source_type,
            building,
            building_polygon,
            hag_handler=hag_handler,
            to_4326=to_4326,
            floor_height_m=floor_height_m,
            hag_sample_count=hag_sample_count,
            min_hag_height_m=min_hag_height_m,
        )
        if height is not None:
            return _height_result(height, source, return_source, metadata)

    height = _random_fallback_height(floor_height_m)
    return _height_result(height, "fallback:random", return_source)


def resolve_building_base_height(
    building: dict,
    *,
    floor_height_m: float = 3.5,
) -> float:
    """
    Resolve the above-ground base offset for an Overture building or part.

    Overture ``min_height`` is already in meters. When it is absent, ``min_floor``
    can be used as a floor-count proxy for the same vertical offset.
    """

    min_height = _nonnegative_float(building.get("min_height"))
    if min_height is not None:
        return min_height

    min_floor = _positive_float(building.get("min_floor"))
    if min_floor is not None:
        return min_floor * floor_height_m

    return 0.0


def resolve_overture_top_height(
    building: dict,
    *,
    floor_height_m: float = 3.5,
    use_num_floors: bool = True,
) -> Optional[float]:
    """
    Resolve the top elevation for an Overture building or building part.

    Returns ``None`` when the feature has no declared height or usable floor
    count. This deliberately avoids LiDAR/random fallbacks so parent/part
    footprint selection stays deterministic.
    """

    height, _ = _height_from_overture_row(
        building,
        floor_height_m=floor_height_m,
        use_height=True,
        use_num_floors=use_num_floors,
    )
    if height is None:
        return None
    return resolve_building_base_height(
        building,
        floor_height_m=floor_height_m,
    ) + height


def should_include_overture_parent_footprint(
    parent_building: dict,
    associated_parts: Sequence[dict],
    *,
    floor_height_m: float = 3.5,
) -> bool:
    """
    Decide whether a parent Overture footprint should be kept with its parts.

    Building parts are normally preferred over their parent footprint. Keep the
    parent as well when its declared top height is shorter than every associated
    part's declared top height, because the parent footprint still contributes
    useful lower-volume geometry in that case.
    """

    if not associated_parts:
        return True

    parent_top = resolve_overture_top_height(
        parent_building,
        floor_height_m=floor_height_m,
    )
    if parent_top is None:
        return False

    part_tops = [
        resolve_overture_top_height(part, floor_height_m=floor_height_m)
        for part in associated_parts
    ]
    if any(part_top is None for part_top in part_tops):
        return False

    return all(parent_top < part_top for part_top in part_tops)


def _normalize_bbox_4326(bbox_4326: Sequence[float]) -> Tuple[float, float, float, float]:
    if len(bbox_4326) != 4:
        raise ValueError("bbox_4326 must contain (min_lon, min_lat, max_lon, max_lat)")

    min_lon, min_lat, max_lon, max_lat = [float(v) for v in bbox_4326]
    if min_lon > max_lon:
        min_lon, max_lon = max_lon, min_lon
    if min_lat > max_lat:
        min_lat, max_lat = max_lat, min_lat

    if not (-180.0 <= min_lon <= 180.0 and -180.0 <= max_lon <= 180.0):
        raise ValueError("bbox longitudes must be in EPSG:4326 degrees")
    if not (-90.0 <= min_lat <= 90.0 and -90.0 <= max_lat <= 90.0):
        raise ValueError("bbox latitudes must be in EPSG:4326 degrees")

    return min_lon, min_lat, max_lon, max_lat


def _overture_buildings_path(release: str, source: str, feature_type: str) -> str:
    if source == "s3":
        return OVERTURE_BUILDINGS_S3_TEMPLATE.format(
            release=release,
            feature_type=feature_type,
        )
    if source == "azure":
        return OVERTURE_BUILDINGS_AZURE_TEMPLATE.format(
            release=release,
            feature_type=feature_type,
        )
    raise ValueError("source must be 's3' or 'azure'")


def _get_duckdb_connection(duckdb_connection):
    if duckdb_connection is not None:
        return duckdb_connection, False
    return duckdb.connect(database=":memory:"), True


def _load_duckdb_extensions(con) -> None:
    # INSTALL is harmless when an extension is already present, and LOAD is
    # needed for cloud-backed Parquet reads and spatial SQL.
    for extension in ("httpfs", "spatial"):
        try:
            con.execute(f"INSTALL {extension}")
        except Exception as exc:
            logger.debug(
                "DuckDB INSTALL %s failed or was unnecessary: %s",
                extension,
                exc,
            )
        con.execute(f"LOAD {extension}")


def _duckdb_wkb_to_bytes(value):
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    return value


def _height_from_overture_row(
    row,
    *,
    floor_height_m: float,
    use_height: bool,
    use_num_floors: bool,
) -> Tuple[Optional[float], Optional[str]]:
    if use_height:
        explicit_height = _positive_float(row.get("height"))
        if explicit_height is not None:
            return explicit_height, "height"

    if use_num_floors:
        num_floors = _positive_float(row.get("num_floors"))
        if num_floors is not None:
            return num_floors * floor_height_m, "num_floors"

    return None, None


def _height_mode_steps(lidar_height_calibration: bool) -> Sequence[str]:
    if lidar_height_calibration:
        return ["overture_height", "overture_num_floors", "hag"]
    return ["overture_height", "overture_num_floors"]

def _height_from_source(
    source_type: str,
    building: dict,
    building_polygon: BaseGeometry,
    *,
    hag_handler,
    to_4326,
    hag_sample_count: int,
    min_hag_height_m: float,
    floor_height_m: float,
) -> Tuple[Optional[float], Optional[str], Optional[Dict[str, object]]]:
    if source_type == "hag":
        height = _height_from_hag(
            building_polygon,
            hag_handler=hag_handler,
            to_4326=to_4326,
            sample_count=hag_sample_count,
            min_height_m=min_hag_height_m,
        )
        if height is not None:
            return height, "hag", None
        return None, None, None

    if source_type == "overture_height":
        height, source = _height_from_overture_row(
            building,
            floor_height_m=floor_height_m,
            use_height=True,
            use_num_floors=False,
        )
        if height is not None:
            return height, f"overture:{source}", _overture_row_metadata(
                building,
                source,
                height,
            )
        return None, None, None

    if source_type == "overture_num_floors":
        height, source = _height_from_overture_row(
            building,
            floor_height_m=floor_height_m,
            use_height=False,
            use_num_floors=True,
        )
        if height is not None:
            return height, f"overture:{source}", _overture_row_metadata(
                building,
                source,
                height,
            )
        return None, None, None

    raise ValueError(f"Unsupported building height source: {source_type}")


def _overture_row_metadata(
    row,
    height_source: str,
    height: float,
) -> Dict[str, object]:
    return {
        "overture_id": row.get("id"),
        "height_source": height_source,
        "height": height,
    }


def _height_from_hag(
    building_polygon: BaseGeometry,
    *,
    hag_handler,
    to_4326,
    sample_count: int,
    min_height_m: float,
) -> Optional[float]:
    if hag_handler is None or to_4326 is None:
        return None
    if building_polygon is None or building_polygon.is_empty:
        return None

    try:
        random_points = generate_random_points(building_polygon, sample_count)
    except Exception as exc:
        logger.debug("Unable to sample HAG points for building: %s", exc)
        return None

    heights = []
    for point in random_points:
        try:
            value = hag_handler.query(to_4326.transform(point.x, point.y), False)
        except Exception as exc:
            logger.debug("Unable to query HAG value for building point: %s", exc)
            continue

        numeric_value = _positive_float(np.asarray(value).squeeze())
        if numeric_value is not None and numeric_value > min_height_m:
            heights.append(numeric_value)

    if not heights:
        return None

    height = float(np.mean(heights))
    if math.isnan(height) or math.isinf(height):
        return None
    return height


def _random_fallback_height(floor_height_m: float) -> float:
    floors = max(1, min(15, int(np.random.normal(loc=5, scale=1))))
    return floor_height_m * floors


def _height_result(
    height: float,
    source: str,
    return_source: bool,
    metadata: Optional[Dict[str, object]] = None,
) -> Union[float, Tuple[float, Dict[str, object]]]:
    if not return_source:
        return height

    result = {"source": source, "height": height}
    if metadata:
        result.update(metadata)
    return height, result


def _positive_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric_value) or math.isinf(numeric_value) or numeric_value <= 0:
        return None
    return numeric_value


def _nonnegative_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric_value) or math.isinf(numeric_value) or numeric_value < 0:
        return None
    return numeric_value

""" 
# for querying footprint centroid coordinates by building ID

bbox_4326 = [-74.007201, 40.712267, -74.000018, 40.717733]
building_id = "f8878ea8-656e-4c15-a66e-0ebc7710cf0f"
buildings = load_overture_buildings_for_aoi(bbox_4326)

building = buildings.loc[buildings["id"] == building_id]

if building.empty:
    raise LookupError(f"No building found with id '{building_id}'.")

centroid = building.geometry.iloc[0].centroid

lon = centroid.x
lat = centroid.y

print(f"Longitude: {lon:.8f}, Latitude: {lat:.8f}")
"""