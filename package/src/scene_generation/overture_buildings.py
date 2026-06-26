"""
Helpers for loading Overture Maps building footprints for a scene AOI.

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
from typing import Dict, Optional, Sequence, Tuple, Union

import geopandas as gpd
import numpy as np
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from .utils import generate_random_points

logger = logging.getLogger(__name__)


OVERTURE_BUILDINGS_RELEASE = "2026-05-20.0"
OVERTURE_BUILDINGS_S3_TEMPLATE = (
    "s3://overturemaps-us-west-2/release/{release}/theme=buildings/type={feature_type}/*"
)
OVERTURE_BUILDINGS_AZURE_TEMPLATE = (
    "az://overturemapswestus2.blob.core.windows.net/release/"
    "{release}/theme=buildings/type={feature_type}/*"
)
OVERTURE_BUILDING_TYPE = "building"
OVERTURE_BUILDING_PART_TYPE = "building_part"

HEIGHT_MODE_LIDAR_OSM = "lidar-osm"
HEIGHT_MODE_OVERTURE = "overture"
BUILDING_HEIGHT_MODE_OPTIONS = (
    HEIGHT_MODE_LIDAR_OSM,
    HEIGHT_MODE_OVERTURE,
)

_HEIGHT_MODE_ALIASES = {
    "1": HEIGHT_MODE_LIDAR_OSM,
    "lidar-osm": HEIGHT_MODE_LIDAR_OSM,
    "2": HEIGHT_MODE_OVERTURE,
    "overture": HEIGHT_MODE_OVERTURE,
}


def normalize_building_height_mode(mode: str) -> str:
    normalized = _HEIGHT_MODE_ALIASES.get(str(mode).strip().lower())
    if normalized is None:
        valid = ", ".join(BUILDING_HEIGHT_MODE_OPTIONS)
        raise ValueError(
            f"Invalid building height mode '{mode}'. Valid values: {valid}"
        )
    return normalized


def load_overture_buildings_for_aoi(
    bbox_4326: Sequence[float],
    target_crs=None,
    *,
    release: str = OVERTURE_BUILDINGS_RELEASE,
    source: str = "s3",
    parquet_path: Optional[str] = None,
    require_height: bool = False,
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
    release
        Overture release string used when ``parquet_path`` is not supplied.
    source
        Cloud source for the default path. Supported values: ``"s3"`` and
        ``"azure"``.
    parquet_path
        Override path for testing or pinned local/cloud data. This should point
        at the Overture buildings GeoParquet partition.
    require_height
        If true, keep only rows with a positive explicit ``height`` value.
        Leave false if you also want to use floor counts as a fallback.
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
        release=release,
        source=source,
        parquet_path=parquet_path,
        require_height=require_height,
        duckdb_connection=duckdb_connection,
        feature_type=OVERTURE_BUILDING_TYPE,
    )


def load_overture_building_parts_for_aoi(
    bbox_4326: Sequence[float],
    target_crs=None,
    *,
    release: str = OVERTURE_BUILDINGS_RELEASE,
    source: str = "s3",
    parquet_path: Optional[str] = None,
    require_height: bool = False,
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
        release=release,
        source=source,
        parquet_path=parquet_path,
        require_height=require_height,
        duckdb_connection=duckdb_connection,
        feature_type=OVERTURE_BUILDING_PART_TYPE,
    )


def _load_overture_building_features_for_aoi(
    bbox_4326: Sequence[float],
    target_crs=None,
    *,
    release: str,
    source: str,
    parquet_path: Optional[str],
    require_height: bool,
    duckdb_connection,
    feature_type: str,
) -> gpd.GeoDataFrame:
    min_lon, min_lat, max_lon, max_lat = _normalize_bbox_4326(bbox_4326)
    path = parquet_path or _overture_buildings_path(release, source, feature_type)
    select_columns = _overture_select_columns(feature_type)

    con, owns_connection = _get_duckdb_connection(duckdb_connection)
    try:
        _load_duckdb_extensions(con)
        if source == "s3" and parquet_path is None:
            con.execute("SET s3_region='us-west-2'")
            con.execute("SET s3_url_style='vhost'")
            con.execute("SET s3_use_ssl=true")
            con.execute("SET s3_requester_pays=false")

        height_filter = "AND height IS NOT NULL AND height > 0" if require_height else ""
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
                {height_filter}
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

# DEPRECATED function: used in an earlier iteration to match Overture buildings to OSM footprints,
# but we are now completely isolating Overture data from OSM data (if Overture data is used, no OSM data is used).
# def lookup_overture_height(
#     building_polygon: BaseGeometry,
#     overture_buildings: gpd.GeoDataFrame,
#     *,
#     min_iou: float = 0.3,
#     min_coverage: float = 0.5,
#     floor_height_m: float = 3.5,
#     use_num_floors: bool = True,
#     use_height: bool = True,
#     return_match: bool = False,
# ) -> Union[Optional[float], Tuple[Optional[float], Optional[Dict[str, object]]]]:
#     """
#     Match one local building footprint to Overture candidates and return a height.

#     ``building_polygon`` and ``overture_buildings`` must already be in the same
#     CRS.

#     The match accepts a candidate when either:
#       * intersection-over-union is at least ``min_iou``; or
#       * the candidate covers at least ``min_coverage`` of the local footprint.

#     Parameters
#     ----------
#     building_polygon
#         Local building footprint in the same CRS as ``overture_buildings``.
#     overture_buildings
#         GeoDataFrame returned by ``load_overture_buildings_for_aoi``.
#     min_iou
#         Minimum intersection-over-union needed for a match.
#     min_coverage
#         Minimum fraction of the local footprint covered by the candidate.
#     floor_height_m
#         Height per floor used when Overture has floor-count metadata but no
#         explicit ``height``.
#     use_num_floors
#         If true, use Overture num_floors times ``floor_height_m`` as a fallback height.
#     use_height
#         If true, use explicit Overture ``height`` values.
#     return_match
#         If true, return ``(height, metadata)``. Otherwise return just height.

#     Returns
#     -------
#     float or None
#         Building height in meters when a matching Overture candidate has a
#         usable height; otherwise None.
#     """

#     no_match = (None, None) if return_match else None
#     if building_polygon is None or building_polygon.is_empty:
#         return no_match
#     if overture_buildings is None or len(overture_buildings) == 0:
#         return no_match
#     if "geometry" not in overture_buildings:
#         return no_match

#     building_area = building_polygon.area
#     if building_area <= 0:
#         return no_match

#     candidate_idx = overture_buildings.sindex.query(
#         building_polygon,
#         predicate="intersects",
#     )
#     if len(candidate_idx) == 0:
#         return no_match

#     best = None
#     candidates = overture_buildings.iloc[candidate_idx]
#     for row_idx, row in candidates.iterrows():
#         candidate_geom = row.geometry
#         if candidate_geom is None or candidate_geom.is_empty:
#             continue

#         intersection_area = building_polygon.intersection(candidate_geom).area
#         if intersection_area <= 0:
#             continue

#         candidate_area = candidate_geom.area
#         union_area = building_area + candidate_area - intersection_area
#         if candidate_area <= 0 or union_area <= 0:
#             continue

#         iou = intersection_area / union_area
#         building_coverage = intersection_area / building_area
#         candidate_coverage = intersection_area / candidate_area
#         if iou < min_iou and building_coverage < min_coverage:
#             continue

#         height, source = _height_from_overture_row(
#             row,
#             floor_height_m=floor_height_m,
#             use_height=use_height,
#             use_num_floors=use_num_floors,
#         )
#         if height is None:
#             continue

#         # Prefer explicit Overture heights over floor-derived estimates, then
#         # prefer the strongest geometric match.
#         score = (
#             1 if source == "height" else 0,
#             iou,
#             building_coverage,
#             candidate_coverage,
#             intersection_area,
#         )
#         if best is None or score > best["score"]:
#             best = {
#                 "height": height,
#                 "score": score,
#                 "metadata": {
#                     "overture_id": row.get("id"),
#                     "height_source": source,
#                     "height": height,
#                     "iou": iou,
#                     "building_coverage": building_coverage,
#                     "candidate_coverage": candidate_coverage,
#                     "intersection_area": intersection_area,
#                     "row_index": row_idx,
#                 },
#             }

#     if best is None:
#         return no_match
#     if return_match:
#         return best["height"], best["metadata"]
#     return best["height"]


def resolve_building_height(
    building: dict,
    building_polygon: BaseGeometry,
    *,
    hag_handler=None,
    to_4326=None,
    floor_height_m: float = 3.5,
    hag_sample_count: int = 30,
    min_hag_height_m: float = 2.0,
    use_overture_num_floors: bool = True,
    use_osm_levels: bool = True,
    height_mode: str = HEIGHT_MODE_LIDAR_OSM,
    return_source: bool = False,
) -> Union[float, Tuple[float, Dict[str, object]]]:
    """
    Resolve the extrusion height for one building footprint.

    ``height_mode`` controls the source hierarchy:

    ``"lidar-osm"`` or ``"1"``
        LiDAR HAG sampling, OSM explicit height tags, OSM floor-count tags,
        then random fallback.
    ``"overture"`` or ``"2"``
        Overture explicit height, Overture num_floors times ``floor_height_m``,
        LiDAR HAG sampling, then random fallback.

    Parameters
    ----------
    building
        Building record from ``GeoDataFrame.to_dict("records")``. In
        ``lidar-osm`` mode this is an OSM record. In ``overture`` mode this is
        an Overture record.
    building_polygon
        Building footprint in the scene's projected CRS.
    hag_handler
        Optional ``GeoTIFFHandler`` used for LiDAR height-above-ground samples.
    to_4326
        Transformer from the scene CRS to EPSG:4326. Required with
        ``hag_handler`` because ``GeoTIFFHandler.query`` expects GPS coords.
    height_mode
        One of ``"lidar-osm"`` or ``"overture"``. Numeric aliases ``"1"``
        and ``"2"`` are also accepted.
    return_source
        If true, return ``(height, metadata)``. Otherwise return just height.
    """

    height_mode = normalize_building_height_mode(height_mode)

    # if height_mode == HEIGHT_MODE_LIDAR_OSM:
    #     height = _height_from_hag(
    #         building_polygon,
    #         hag_handler=hag_handler,
    #         to_4326=to_4326,
    #         sample_count=hag_sample_count,
    #         min_height_m=min_hag_height_m,
    #     )
    #     if height is not None:
    #         return _height_result(height, "hag", return_source)

    for source_type in _height_mode_steps(height_mode, use_osm_levels):
        height, source, metadata = _height_from_source(
            source_type,
            building,
            building_polygon,
            hag_handler=hag_handler,
            to_4326=to_4326,
            floor_height_m=floor_height_m,
            hag_sample_count=hag_sample_count,
            min_hag_height_m=min_hag_height_m,
            use_overture_num_floors=use_overture_num_floors,
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

    try:
        import duckdb
    except ImportError as exc:
        raise ImportError(
            "load_overture_buildings_for_aoi requires duckdb. "
            "Install it with `pip install duckdb`."
        ) from exc

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


def _height_mode_steps(height_mode: str, use_osm_levels: bool) -> Sequence[str]:
    if height_mode == HEIGHT_MODE_LIDAR_OSM:
        steps = ["hag", "osm_explicit"]
        if use_osm_levels:
            steps.append("osm_levels")
        return steps

    if height_mode == HEIGHT_MODE_OVERTURE:
        return ["overture_height", "overture_num_floors", "hag"]

    raise ValueError(f"Unsupported building height mode: {height_mode}")

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
    use_overture_num_floors: bool,
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
    if source_type == "osm_explicit":
        height, source = _explicit_height_from_osm(building)
        return height, source, None

    if source_type == "osm_levels":
        height, source = _height_from_osm_levels(building, floor_height_m)
        return height, source, None

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
        if not use_overture_num_floors:
            return None, None, None
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


def _explicit_height_from_osm(building: dict) -> Tuple[Optional[float], Optional[str]]:
    for key in ("building:height", "height"):
        height = _height_value_to_meters(building.get(key))
        if height is not None:
            return height, f"osm:{key}"
    return None, None


def _height_from_osm_levels(
    building: dict,
    floor_height_m: float,
) -> Tuple[Optional[float], Optional[str]]:
    for key in ("building:levels", "levels"):
        levels = _positive_float(building.get(key))
        if levels is not None:
            return levels * floor_height_m, f"osm:{key}"
    return None, None


def _height_value_to_meters(value) -> Optional[float]:
    if value is None:
        return None

    numeric_value = _positive_float(value)
    if numeric_value is not None:
        return numeric_value

    if not isinstance(value, str):
        return None

    normalized = value.strip().lower().replace(",", ".")
    if not normalized:
        return None

    match = re.search(r"[-+]?\d*\.?\d+", normalized)
    if not match:
        return None

    numeric_value = _positive_float(match.group(0))
    if numeric_value is None:
        return None

    if "ft" in normalized or "feet" in normalized or "foot" in normalized:
        return numeric_value * 0.3048
    return numeric_value


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
