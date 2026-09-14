from typing import List, Optional
import geopandas as gpd
from shapely.geometry.base import BaseGeometry
import requests, os, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from requests.adapters import HTTPAdapter, Retry

class USGS1mLocator:
    """
    Locates USGS 1m DEM tile download URLs covering an area of interest,
    resolving intersecting grid cells in parallel.
    """

    def __init__(self, fesm_path: str, grid_path: str, debug: bool = True, max_workers: Optional[int] = None):
        self.debug = debug
        self.max_workers = max_workers or max(8, (os.cpu_count() or 8) * 5)  # I/O bound ⇒ lots of threads ok

        # Robust HTTP session (connection pooling + retries)
        self._session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504))
        self._session.mount("https://", HTTPAdapter(max_retries=retries, pool_connections=64, pool_maxsize=64))
        self._session.mount("http://", HTTPAdapter(max_retries=retries, pool_connections=64, pool_maxsize=64))

        # Load once
        self._log(f"[INIT] loading FESM: {fesm_path}")
        fesm = gpd.read_file(fesm_path)
        for req in ["project", "product_link"]:
            if req not in fesm.columns:
                raise KeyError(f"FESM missing required column: '{req}'")
        
        self._log(f"[INIT] loading GRID: {grid_path}")
        grid = gpd.read_file(grid_path)
        for req in ["utm_zone", "name"]:
            if req not in grid.columns:
                raise KeyError(f"GRID missing required column: '{req}'")

        # Normalize CRS to 4326 once
        self.fesm = self._to_4326(fesm, name="FESM")
        self.grid = self._to_4326(grid, name="GRID")

        # Pre-build spatial indexes ONCE to avoid race/lazy-build in threads
        _ = self.fesm.sindex
        _ = self.grid.sindex
        self._log("[INIT] spatial indexes ready")

        # Small lock for printing (optional) and any shared mutable state
        self._print_lock = threading.Lock()

    # ---------- public API ----------

    def set_debug(self, debug_flag):
        self.debug = debug_flag

    def get_links(self, aoi_4326: BaseGeometry) -> List[str]:
        """
        Resolve DEM tile download URLs for an AOI, parallelizing URL
        resolution when many grid cells intersect.
        """
        self._log(f"[Q] AOI bounds (4326): {aoi_4326.bounds}")

        # FESM: require containment
        fesm_hits = self._sindex_contains(self.fesm, aoi_4326, label="FESM")
        if fesm_hits.empty:
            self._log("[Q] No FESM row fully contains AOI — returning []")
            return []
        if len(fesm_hits) > 1:
            self._log(f"[WARN] Multiple FESM rows contain AOI ({len(fesm_hits)}). Using the first.")
        project = str(fesm_hits.iloc[0]["project"]).strip()
        product_link = str(fesm_hits.iloc[0]["product_link"]).strip()
        self._log(f"[Q] chosen project: {project}")

        # GRID: check contains first (fast path, one cell)
        grid_contains = self._sindex_contains(self.grid, aoi_4326, label="GRID")
        if len(grid_contains) >= 1:
            if len(grid_contains) > 1:
                self._log(f"[WARN] Multiple GRID cells contain AOI ({len(grid_contains)}). Using the first.")
            r = grid_contains.iloc[0]
            url = self._build_url(project, r, product_link)
            return [u for u in [url] if u]


        self._log("[Q] No GRID cells intersect AOI — returning []")

        
        # GRID: fallback to intersects (possibly many cells)
        self._log("[Q] No GRID cell contains AOI; returning all intersecting cells")
        grid_inter = self._sindex_intersects(self.grid, aoi_4326, label="GRID")
        if grid_inter.empty:
            self._log("[Q] No GRID cells intersect AOI — returning []")
            return []

        # PARALLEL: resolve URLs for each intersecting row
        urls = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = [ex.submit(self._build_url, project, row, product_link) for _, row in grid_inter.iterrows()]
            for fut in as_completed(futures):
                try:
                    u = fut.result()
                    if u:
                        urls.append(u)
                except Exception as e:
                    self._log(f"[ERR] worker failed: {e}")

        # Deduplicate
        uniq = list(dict.fromkeys(urls))
        self._log(f"[Q] final URLs: {len(uniq)}")
        return uniq

    # ---------- internals ----------

    # --- Manifest fetching & search ---

    @lru_cache(maxsize=5000)
    def _fetch_manifest(self, clean_baselink: str) -> Optional[str]:
        """Download once per baselink; cached across calls/threads."""
        manifest_url = clean_baselink.rstrip("/") + "/0_file_download_links.txt"
        self._log(f"[DBG] fetching manifest: {manifest_url}")
        try:
            r = self._session.get(manifest_url, timeout=20)
            r.raise_for_status()
            return r.text
        except Exception as e:
            self._log(f"[ERR] cannot fetch {manifest_url}: {e}")
            return None

    def _find_in_manifest(self, manifest_text: str, name: str) -> Optional[str]:
        # Simple substring scan; could be tightened with exact filename regex if needed
        for line in manifest_text.splitlines():
            line = line.strip()
            if not line:
                continue
            if name in line:
                self._log(f"[DBG] match for name={name}: {line}")
                return line
        return None

    def _build_url(self, project: str, grid_row, product_link) -> Optional[str]:
        """
        - Download product_link + "0_file_download_links.txt" (cached)
        - Search for grid_row['name']
        - Return first matching line as URL (or None)
        """
        name = str(grid_row["name"]).strip()
        baselink = str(product_link).strip()
        self._log(f"[product_link] {baselink}")
        clean_baselink = baselink.replace("index.html?prefix=", "")

        manifest = self._fetch_manifest(clean_baselink)
        if not manifest:
            return None
        url = self._find_in_manifest(manifest, name)
        if not url:
            self._log(f"[WARN] no match found for name={name} in {clean_baselink}/0_file_download_links.txt")
        return url

    # --- Geo helpers ---

    def _to_4326(self, gdf: gpd.GeoDataFrame, name: str) -> gpd.GeoDataFrame:
        if gdf.crs is None:
            self._log(f"[CRS] {name} has no CRS; assuming EPSG:4269")
            gdf = gdf.set_crs(4269)
        if gdf.crs.to_epsg() != 4326:
            self._log(f"[CRS] reprojecting {name} from {gdf.crs} to EPSG:4326")
            gdf = gdf.to_crs(4326)
        return gdf

    def _sindex_contains(self, gdf_4326: gpd.GeoDataFrame, aoi_4326: BaseGeometry, label: str):
        idx = gdf_4326.sindex.query(aoi_4326, predicate="covered_by")
        hits = gdf_4326.iloc[idx]
        hits = hits[hits.contains(aoi_4326)]
        self._log(f"[SI {label}] contains: {len(hits)}")
        self._log(f"{hits}")
        return hits

        

    def _sindex_intersects(self, gdf_4326: gpd.GeoDataFrame, aoi_4326: BaseGeometry, label: str):
        idx = gdf_4326.sindex.query(aoi_4326, predicate="intersects")
        hits = gdf_4326.iloc[idx]
        hits = hits[hits.intersects(aoi_4326)]
        self._log(f"[SI {label}] intersects: {len(hits)}")
        self._log(f"{hits}")
        return hits

    def _log(self, msg: str):
        if self.debug:
            # keep prints readable if many threads log
            with getattr(self, "_print_lock", threading.Lock()):
                print(msg)
