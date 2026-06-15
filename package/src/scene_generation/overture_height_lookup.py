"""
overture_height_lookup.py

Helper functions for using Overture Maps building data as a fallback
height source for Geo2SigMap.

Main idea:
    - Load Overture buildings/building_parts once for the scene bounding box.
    - For each OSM building polygon with missing height, spatially match it
      to the best overlapping Overture building.
    - Return one representative fallback height in meters.

Suggested fallback priority (after spatial match):
    When has_parts or parts are linked: max(part heights, parent height,
        parent num_floors * floor_height) so low parent OR low parts alone
        do not win (fixes both podium-on-parent and height-on-parent cases).
    Otherwise: parent height, then parts, then num_floors * floor_height.
    None if nothing usable.
"""

import warnings

import duckdb

# Overture Maps public release bucket (see https://docs.overturemaps.org)
OVERTURE_S3_RELEASE_BASE = "s3://overturemaps-us-west-2/release"
import geopandas as gpd
import pandas as pd
from shapely import wkb


class OvertureHeightLookup:
    def __init__(
        self,
        bbox,
        floor_height=3.5,
        min_overlap_ratio=0.30,
        release_url=None,
        verbose=True,
    ):
        """
        Parameters
        ----------
        bbox : tuple
            Bounding box as (min_lon, min_lat, max_lon, max_lat).

        floor_height : float
            Height in meters used when only num_floors is available.

        min_overlap_ratio : float
            Minimum fraction of the OSM building polygon area that must overlap
            the selected Overture building polygon.

        release_url : str or None
            Optional Overture release URL. If None, this class tries to query
            Overture's STAC catalog for the latest release.

        verbose : bool
            If True, print basic loading/debug information.
        """

        self.bbox = bbox
        self.floor_height = floor_height
        self.min_overlap_ratio = min_overlap_ratio
        self.release_url = release_url
        self.verbose = verbose

        self.buildings = gpd.GeoDataFrame()
        self.parts = gpd.GeoDataFrame()
        self.parts_by_building_id = {}

        self._load_overture_data()

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def get_height(self, osm_building_polygon):
        """
        Return Overture fallback height for one OSM building polygon.

        Parameters
        ----------
        osm_building_polygon : shapely Polygon or MultiPolygon
            OSM building footprint geometry.

        Returns
        -------
        float or None
            Fallback height in meters, or None if no suitable Overture height
            can be found.
        """

        if osm_building_polygon is None or osm_building_polygon.is_empty:
            return None

        if self.buildings.empty:
            return None

        # Make sure lookup polygon is valid enough for intersection checks.
        osm_poly = self._clean_geometry(osm_building_polygon)

        if osm_poly is None or osm_poly.is_empty:
            return None

        osm_area = osm_poly.area

        if osm_area <= 0:
            return None

        # Spatial index prefilter.
        possible_idx = list(self.buildings.sindex.intersection(osm_poly.bounds))

        if len(possible_idx) == 0:
            return None

        candidates = self.buildings.iloc[possible_idx].copy()
        candidates = candidates[candidates.geometry.intersects(osm_poly)]

        if candidates.empty:
            return None

        # Use overlap area to choose best Overture building.
        candidates["overlap_area"] = candidates.geometry.intersection(osm_poly).area
        candidates = candidates[candidates["overlap_area"] > 0]

        if candidates.empty:
            return None

        best = candidates.sort_values("overlap_area", ascending=False).iloc[0]
        overlap_ratio = float(best["overlap_area"] / osm_area)

        if overlap_ratio < self.min_overlap_ratio:
            return None

        building_id = best.get("id")
        part_height, parent_height, floors_height = self._heights_from_match(
            best, building_id
        )

        # Complex buildings: take the best of parts + parent (not parts alone).
        if self._prefer_parts_height(best, building_id):
            combined = self._max_valid_heights(
                part_height, parent_height, floors_height
            )
            if combined is not None:
                return combined

        # Simple buildings: parent, then parts, then floors.
        if self._valid_height(parent_height):
            return parent_height

        if self._valid_height(part_height):
            return part_height

        if self._valid_height(floors_height):
            return floors_height

        return None

    def summarize(self):
        """
        Print a short summary of loaded Overture data.
        """

        n_buildings = len(self.buildings)
        n_parts = len(self.parts)

        n_building_heights = 0
        n_building_floors = 0
        n_part_heights = 0

        if not self.buildings.empty:
            n_building_heights = self.buildings["height"].notna().sum()
            n_building_floors = self.buildings["num_floors"].notna().sum()

        if not self.parts.empty:
            n_part_heights = self.parts["height"].notna().sum()

        print("Overture height lookup summary")
        print("--------------------------------")
        print(f"bbox: {self.bbox}")
        print(f"buildings loaded: {n_buildings}")
        print(f"building_parts loaded: {n_parts}")
        print(f"buildings with height: {n_building_heights}")
        print(f"buildings with num_floors: {n_building_floors}")
        print(f"building_parts with height: {n_part_heights}")

    # -------------------------------------------------------------------------
    # Loading Overture data
    # -------------------------------------------------------------------------

    def _load_overture_data(self):
        """
        Load Overture building and building_part data inside bbox.
        """

        con = duckdb.connect()

        con.execute("INSTALL spatial;")
        con.execute("LOAD spatial;")
        con.execute("INSTALL httpfs;")
        con.execute("LOAD httpfs;")
        con.execute("SET s3_region='us-west-2';")

        if self.release_url is None:
            self.release_url = self._get_latest_release_url(con)
        else:
            self.release_url = self._normalize_release_url(self.release_url)

        if self.verbose:
            print(f"Using Overture release: {self.release_url}")

        self.buildings = self._query_overture_type(con, feature_type="building")
        self.parts = self._query_overture_type(con, feature_type="building_part")

        if not self.parts.empty and "building_id" in self.parts.columns:
            self.parts_by_building_id = {
                building_id: group
                for building_id, group in self.parts.groupby("building_id")
                if pd.notna(building_id)
            }

        if self.verbose:
            self.summarize()

    def _normalize_release_url(self, release):
        """
        Turn a STAC version string into a full S3 release base path.

        STAC returns e.g. "2026-05-20.0"; DuckDB needs:
        s3://overturemaps-us-west-2/release/2026-05-20.0
        """
        if release is None:
            return None

        release = str(release).strip()

        if release.startswith("s3://") or release.startswith("https://"):
            return release.rstrip("/")

        return f"{OVERTURE_S3_RELEASE_BASE}/{release}"

    def _get_latest_release_url(self, con):
        """
        Query Overture STAC catalog for latest release URL.

        Note:
            Overture's docs recommend using the STAC catalog to find the
            latest release instead of hardcoding a release path.
        """

        latest = con.execute(
            "SELECT latest FROM 'https://stac.overturemaps.org/catalog.json'"
        ).fetchone()[0]

        return self._normalize_release_url(latest)

    def _query_overture_type(self, con, feature_type):
        """
        Query Overture buildings or building_parts inside self.bbox.

        Parameters
        ----------
        con : duckdb connection
        feature_type : str
            Either "building" or "building_part".

        Returns
        -------
        geopandas.GeoDataFrame
        """

        min_lon, min_lat, max_lon, max_lat = self.bbox

        if feature_type == "building":
            cols = """
                id,
                height,
                num_floors,
                has_parts,
                geometry
            """
        elif feature_type == "building_part":
            cols = """
                id,
                building_id,
                height,
                min_height,
                num_floors,
                geometry
            """
        else:
            raise ValueError(f"Unknown Overture feature_type: {feature_type}")

        parquet_path = (
            f"{self.release_url}/theme=buildings/type={feature_type}/*"
        )

        query = f"""
            SELECT
                {cols}
            FROM read_parquet('{parquet_path}', hive_partitioning=1)
            WHERE bbox.xmin <= {max_lon}
              AND bbox.xmax >= {min_lon}
              AND bbox.ymin <= {max_lat}
              AND bbox.ymax >= {min_lat}
        """

        try:
            df = con.execute(query).df()
        except Exception as e:
            warnings.warn(
                f"Could not query Overture feature_type={feature_type}. "
                f"Returning empty GeoDataFrame. Error: {e}"
            )
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

        if df.empty:
            return gpd.GeoDataFrame(df, geometry=[], crs="EPSG:4326")

        # DuckDB may return geometry as WKB bytes from GeoParquet.
        df["geometry"] = df["geometry"].apply(self._geometry_from_overture)

        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
        gdf = gdf[gdf.geometry.notna()]
        gdf = gdf[~gdf.geometry.is_empty]

        # Clean invalid polygons if needed.
        gdf["geometry"] = gdf.geometry.apply(self._clean_geometry)
        gdf = gdf[gdf.geometry.notna()]
        gdf = gdf[~gdf.geometry.is_empty]

        return gdf

    # -------------------------------------------------------------------------
    # Height extraction helpers
    # -------------------------------------------------------------------------

    def _heights_from_match(self, building_row, building_id):
        """
        Collect part, parent, and floor-based height candidates for one match.
        """
        part_height = self._height_from_parts(building_id)
        parent_height = self._safe_float(building_row.get("height"))
        num_floors = self._safe_float(building_row.get("num_floors"))
        floors_height = None

        if self._valid_floor_count(num_floors):
            floors_height = num_floors * self.floor_height

        return part_height, parent_height, floors_height

    def _max_valid_heights(self, *values):
        """
        Return the maximum of all values that pass _valid_height, or None.
        """
        valid = [float(v) for v in values if self._valid_height(v)]

        if len(valid) == 0:
            return None

        return max(valid)

    def _prefer_parts_height(self, building_row, building_id):
        """
        Return True when building_part data should win over parent height.
        """
        has_parts = building_row.get("has_parts")
        if has_parts is True:
            return True
        if isinstance(has_parts, str) and has_parts.strip().lower() == "true":
            return True
        return self._has_linked_parts(building_id)

    def _has_linked_parts(self, building_id):
        parts, _note = self._get_parts_for_building_id(building_id)
        return parts is not None and not parts.empty

    def _get_parts_for_building_id(self, building_id):
        """
        Return parts GeoDataFrame for a building id, trying common key variants.
        """
        if building_id is None or pd.isna(building_id):
            return None, "missing building_id"

        if building_id in self.parts_by_building_id:
            return self.parts_by_building_id[building_id], "exact id key"

        str_id = str(building_id)
        if str_id in self.parts_by_building_id:
            return self.parts_by_building_id[str_id], "str(id) key"

        return None, "no parts for building_id"

    def _height_from_parts(self, building_id):
        """
        Collapse Overture building_parts into one representative height.

        Uses the maximum part ``height`` (meters above ground). Stacked parts
        with ``min_height`` are not summed into min_height+height here, because
        that can over-estimate a single extruded footprint (e.g. spire segments).
        Per-part fallback: num_floors * floor_height when height is missing.
        """

        if building_id is None or pd.isna(building_id):
            return None

        parts, _note = self._get_parts_for_building_id(building_id)
        if parts is None or parts.empty:
            return None

        tops = []

        for _, part in parts.iterrows():
            height = self._safe_float(part.get("height"))

            if not self._valid_height(height):
                num_floors = self._safe_float(part.get("num_floors"))

                if self._valid_floor_count(num_floors):
                    height = num_floors * self.floor_height

            if self._valid_height(height):
                tops.append(height)

        if len(tops) == 0:
            return None

        return max(tops)

    def _valid_height(self, value):
        """
        Return True if value looks like a usable building height in meters.
        """

        if value is None or pd.isna(value):
            return False

        # Avoid zero/negative heights and obvious crazy values.
        return 1.0 <= float(value) <= 1000.0

    def _valid_floor_count(self, value):
        """
        Return True if value looks like a usable number of floors.
        """

        if value is None or pd.isna(value):
            return False

        return 1.0 <= float(value) <= 200.0

    def _safe_float(self, value):
        """
        Convert value to float if possible; otherwise return None.
        """

        if value is None or pd.isna(value):
            return None

        try:
            return float(value)
        except Exception:
            return None

    # -------------------------------------------------------------------------
    # Geometry helpers
    # -------------------------------------------------------------------------

    def _geometry_from_overture(self, geom):
        """
        Convert Overture geometry into a shapely geometry.

        Depending on package versions, DuckDB may return geometry as WKB bytes,
        memoryview, bytearray, or already as a shapely geometry.
        """

        if geom is None or pd.isna(geom):
            return None

        if hasattr(geom, "geom_type"):
            return geom

        try:
            if isinstance(geom, memoryview):
                geom = geom.tobytes()

            if isinstance(geom, bytearray):
                geom = bytes(geom)

            if isinstance(geom, bytes):
                return wkb.loads(geom)
        except Exception:
            return None

        return None

    def _clean_geometry(self, geom):
        """
        Try to repair minor polygon validity issues.
        """

        if geom is None:
            return None

        try:
            if geom.is_empty:
                return None

            if not geom.is_valid:
                geom = geom.buffer(0)

            if geom.is_empty:
                return None

            return geom

        except Exception:
            return None