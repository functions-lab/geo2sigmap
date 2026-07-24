"""
Helpers for loading Overture Maps road centerlines for a scene AOI.

Overture models roads as ``theme=transportation``, ``type=segment`` LineString
features. A single ``segment`` row can carry ``subtype`` values of ``road``,
``rail``, or ``water`` (ferry routes); this module only ever queries
``subtype = 'road'``.

Unlike buildings, Overture road segments rarely carry an explicit width
(``width_rules``) — it is only populated for a minority of segments, mostly
dedicated cycleways in the sampled data. ``resolve_road_width`` falls back to
a per-``class`` default width (loosely based on typical OSM highway widths)
when no explicit value is present.

This module keeps the Overture query in EPSG:4326, then optionally reprojects
the returned GeoDataFrame to the caller's target CRS, mirroring
``overture_buildings.py``.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Optional, Sequence

import duckdb
import geopandas as gpd
from shapely.geometry import Polygon, box
from shapely.geometry.base import BaseGeometry

logger = logging.getLogger(__name__)

OVERTURE_TRANSPORTATION_S3_TEMPLATE = (
    "s3://overturemaps-us-west-2/release/{release}/theme=transportation/type={feature_type}/*"
)
OVERTURE_TRANSPORTATION_AZURE_TEMPLATE = (
    "az://overturemapswestus2.blob.core.windows.net/release/"
    "{release}/theme=transportation/type={feature_type}/*"
)
OVERTURE_SEGMENT_TYPE = "segment"
ROAD_SUBTYPE = "road"

# Edge-to-edge road width fallback (meters) used when a segment has no
# explicit `width_rules` value and its `class` isn't in
# DEFAULT_ROAD_CLASS_WIDTHS_M.
DEFAULT_ROAD_WIDTH_M = 5.0

# Loosely based on typical OSM `highway` width conventions. Overture's
# `class` values mirror OSM highway tags.
DEFAULT_ROAD_CLASS_WIDTHS_M: Dict[str, float] = {
    "motorway": 15.0,
    "motorway_link": 8.0,
    "trunk": 12.0,
    "trunk_link": 7.0,
    "primary": 10.0,
    "primary_link": 6.0,
    "secondary": 9.0,
    "secondary_link": 6.0,
    "tertiary": 8.0,
    "tertiary_link": 5.0,
    "residential": 6.0,
    "living_street": 5.5,
    "unclassified": 5.5,
    "service": 4.0,
    "pedestrian": 4.0,
    "footway": 2.0,
    "sidewalk": 2.0,
    "crosswalk": 2.0,
    "cycleway": 2.0,
    "path": 1.5,
    "steps": 1.5,
    "track": 3.0,
    "bridleway": 2.0,
    "unknown": DEFAULT_ROAD_WIDTH_M,
}


def load_overture_roads_for_aoi(
    bbox_4326: Sequence[float],
    target_crs=None,
    *,
    source: str = "s3",
    parquet_path: Optional[str] = None,
    road_subtype: str = ROAD_SUBTYPE,
    duckdb_connection=None,
) -> gpd.GeoDataFrame:
    """
    Load Overture road centerlines intersecting a WGS84 bounding box.

    Parameters
    ----------
    bbox_4326
        Bounding box in ``(min_lon, min_lat, max_lon, max_lat)`` order.
    target_crs
        Optional CRS to reproject the returned GeoDataFrame into. Pass the
        scene's UTM CRS before buffering centerlines into road footprints.
    source
        Cloud source for the default path. Supported values: ``"s3"`` and
        ``"azure"``. Defaults to ``"s3"``.
    parquet_path
        Override path for testing or pinned local/cloud data. This should
        point at the Overture transportation segment GeoParquet partition.
    road_subtype
        Overture ``subtype`` value to filter on. Defaults to ``"road"``,
        which excludes ``rail`` and ``water`` (ferry) segments.
    duckdb_connection
        Optional existing DuckDB connection, mostly useful for tests.

    Returns
    -------
    geopandas.GeoDataFrame
        Overture road segments in EPSG:4326, or ``target_crs`` when
        provided. Columns include ``id``, ``class``, ``subclass``,
        ``road_surface``, ``width_rules``, and ``geometry``.
    """

    min_lon, min_lat, max_lon, max_lat = _normalize_bbox_4326(bbox_4326)

    con, owns_connection = _get_duckdb_connection(duckdb_connection)
    try:
        _load_duckdb_extensions(con)
        if source == "s3" and parquet_path is None:
            con.execute("SET s3_region='us-west-2'")
            con.execute("SET s3_url_style='vhost'")
            con.execute("SET s3_use_ssl=true")
            con.execute("SET s3_requester_pays=false")

        if parquet_path is None:
            release = _get_latest_transportation_release_url(con, OVERTURE_SEGMENT_TYPE)
        else:
            release = None
        path = parquet_path or release
        print(f"Using Overture release: {path}")

        query = """
            SELECT
                id,
                class,
                subclass,
                road_surface,
                width_rules,
                bbox.xmin AS bbox_xmin,
                bbox.ymin AS bbox_ymin,
                bbox.xmax AS bbox_xmax,
                bbox.ymax AS bbox_ymax,
                geometry AS geometry_wkb
            FROM read_parquet(?, hive_partitioning=1)
            WHERE
                subtype = ?
                AND bbox.xmin <= ?
                AND bbox.xmax >= ?
                AND bbox.ymin <= ?
                AND bbox.ymax >= ?
        """

        df = con.execute(
            query,
            [path, road_subtype, max_lon, min_lon, max_lat, min_lat],
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

    # The bbox predicate is intentionally broad for Parquet pruning. Do an
    # exact geometry intersection locally before returning candidates.
    aoi = box(min_lon, min_lat, max_lon, max_lat)
    gdf = gdf[gdf.intersects(aoi)].copy()
    gdf = _explode_line_geometries(gdf)

    if target_crs is not None and not gdf.empty:
        gdf = gdf.to_crs(target_crs)

    logger.info("Loaded %d Overture road segment candidates", len(gdf))
    return gdf


def resolve_road_width(
    road: dict,
    *,
    default_widths_by_class: Optional[Dict[str, float]] = None,
    default_width_m: float = DEFAULT_ROAD_WIDTH_M,
) -> float:
    """
    Resolve a road segment's edge-to-edge width in meters.

    Prefers an explicit Overture ``width_rules`` value. Falls back to a
    per-``class`` default width, then to ``default_width_m`` for
    unrecognized or missing classes.
    """

    explicit_width = _positive_float(_first_rule_value(road.get("width_rules")))
    if explicit_width is not None:
        return explicit_width

    widths_by_class = default_widths_by_class or DEFAULT_ROAD_CLASS_WIDTHS_M
    road_class = road.get("class")
    if isinstance(road_class, str):
        width = widths_by_class.get(road_class.lower())
        if width is not None:
            return width

    return default_width_m


def build_road_polygon(line: BaseGeometry, width_m: float) -> Optional[Polygon]:
    """
    Buffer a road centerline into a flat polygon footprint for meshing.

    Uses flat end caps so a segment's buffered footprint does not balloon
    past its endpoints (shared ``connector`` points with adjoining segments).
    """

    if line is None or line.is_empty or width_m <= 0:
        return None

    buffered = line.buffer(width_m / 2.0, cap_style="flat", join_style="round")
    if buffered.is_empty:
        return None

    if buffered.geom_type == "MultiPolygon":
        buffered = max(buffered.geoms, key=lambda geom: geom.area)

    if buffered.geom_type != "Polygon":
        return None

    return buffered


def _first_rule_value(rules):
    # `width_rules` / `road_surface` are lists of {"value": ..., "between": ...}
    # scoped-rule structs. Prefer a rule that applies to the whole segment
    # (`between` is null); otherwise, fall back to the first rule.
    if rules is None:
        return None
    try:
        rule_list = list(rules)
    except TypeError:
        return None
    if not rule_list:
        return None

    for rule in rule_list:
        if isinstance(rule, dict) and rule.get("between") is None and rule.get("value") is not None:
            return rule.get("value")

    first_rule = rule_list[0]
    if isinstance(first_rule, dict):
        return first_rule.get("value")
    return None


def _explode_line_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf

    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    gdf = gdf[gdf.geometry.geom_type.isin(("LineString", "MultiLineString"))].copy()
    if gdf.empty:
        return gdf

    try:
        exploded = gdf.explode(index_parts=False, ignore_index=True)
    except TypeError:
        exploded = gdf.explode(index_parts=False).reset_index(drop=True)
    return exploded[exploded.geometry.geom_type == "LineString"].reset_index(drop=True)


def _normalize_bbox_4326(bbox_4326: Sequence[float]):
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


def _overture_transportation_path(release: str, source: str, feature_type: str) -> str:
    if source == "s3":
        return OVERTURE_TRANSPORTATION_S3_TEMPLATE.format(
            release=release,
            feature_type=feature_type,
        )
    if source == "azure":
        return OVERTURE_TRANSPORTATION_AZURE_TEMPLATE.format(
            release=release,
            feature_type=feature_type,
        )
    raise ValueError("source must be 's3' or 'azure'")


def _get_latest_transportation_release_url(con, feature_type: str) -> str:
    latest = con.execute(
        "SELECT latest FROM 'https://stac.overturemaps.org/catalog.json'"
    ).fetchone()[0]
    release = str(latest).strip()
    return _overture_transportation_path(release, "s3", feature_type)


def _get_duckdb_connection(duckdb_connection):
    if duckdb_connection is not None:
        return duckdb_connection, False
    return duckdb.connect(database=":memory:"), True


def _load_duckdb_extensions(con) -> None:
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
