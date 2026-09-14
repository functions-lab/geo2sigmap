# deps:
# pip install rasterio shapely pyproj numpy

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import os
import math
import uuid

import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.mask import mask as rio_mask
from rasterio.merge import merge as rio_merge
from shapely.geometry.base import BaseGeometry
from shapely.geometry import mapping, shape
from shapely.ops import unary_union
from pyproj import CRS, Transformer


def clip_reproject_dem_to_wgs84_utm(
    aoi_4326: BaseGeometry,
    urls: List[str],
    mode: str = "single_zone",            # 'single_zone' or 'multi_zone'
    target_zone: Optional[int] = None,    # used only when mode='single_zone'
    dst_res_m: float | Tuple[float, float] = 1.0,
    mosaic: bool = True,
    out_prefix: str = "dem_wgs84utm",
    overwrite: bool = False,
    debug: bool = False
) -> Dict[str, Any]:
    """
    Debug-friendly version: streams USGS 1 m COGs, clips to AOI (EPSG:4326),
    reprojects to WGS84/UTM, writes GeoTIFF(s), and returns metadata.
    _logs [DBG]/[WARN]/[ERR] at each key step.
    """

    # ---------- helpers ----------

    def _log(msg: str):
        if debug:
            print(msg)
    def _utm_zone_from_lon(lon: float) -> int:
        return int(math.floor((lon + 180.0) / 6.0) + 1)

    def _utm_epsg_from_lonlat(lon: float, lat: float) -> int:
        zone = _utm_zone_from_lon(lon)
        epsg = 32600 + zone if lat >= 0 else 32700 + zone
        _log(f"[DBG] lon={lon:.6f}, lat={lat:.6f} -> UTM zone {zone}, EPSG {epsg}")
        return epsg

    def _geom_transform(geom, src: CRS, dst: CRS):
        _log(f"[DBG] reproject geom from {src.to_string()} -> {dst.to_string()}")
        t = Transformer.from_crs(src, dst, always_xy=True)
        coords_geojson = mapping(geom)

        def _walk_coords(obj):
            if isinstance(obj, (list, tuple)):
                if len(obj) == 0:
                    return obj
                if isinstance(obj[0], (list, tuple)):
                    return [_walk_coords(c) for c in obj]
                else:
                    x, y = obj
                    X, Y = t.transform(x, y)
                    return [X, Y]
            return obj

        new_geom = dict(coords_geojson)
        new_geom["coordinates"] = _walk_coords(coords_geojson["coordinates"])
        new_geom = shape(new_geom)
        _log(f"[DBG] transformed geom bounds: {new_geom.bounds}")
        return new_geom

    def _ensure_list_res(res) -> Tuple[float, float]:
        out = (float(res), float(res)) if not isinstance(res, (list, tuple)) else (float(res[0]), float(res[1]))
        _log(f"[DBG] target resolution (m): {out}")
        return out

    def _open_env():
        _log("[DBG] setting rasterio.Env for COG streaming")
        return rasterio.Env(
            AWS_NO_SIGN_REQUEST="YES",
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            CPL_VSIL_CURL_USE_HEAD="NO",
            GDAL_HTTP_MAX_RETRY="4",
            GDAL_HTTP_RETRY_DELAY="0.5",
            GDAL_HTTP_CONNECTTIMEOUT="10",
            GDAL_HTTP_TIMEOUT="60",
            CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff,.ovr,.xml,.aux.xml"
        )

    def _clip_reproject_single(url: str, target_epsg: int, out_path: str) -> Optional[Dict[str, Any]]:
        _log(f"[DBG] processing URL: {url}")
        with _open_env():
            try:
                with rasterio.open(url) as src:
                    _log(f"[DBG] opened src: CRS={src.crs}, bounds={src.bounds}, res={src.res}, nodata={src.nodata}")
                    src_crs = src.crs or CRS.from_epsg(4326)

                    # AOI in source CRS (for efficient windowing if needed)
                    aoi_src = _geom_transform(aoi_geom, crs_4326, src_crs)
                    # (We rely on rio_mask on the VRT, so we don't manually window here;
                    # but _loging helps you see alignment)
                    _log(f"[DBG] AOI in src CRS bounds={aoi_src.bounds}")

                    resx, resy = _ensure_list_res(dst_res_m)
                    dst_crs = CRS.from_epsg(target_epsg)
                    vrt_opts = dict(
                        crs=dst_crs,
                        resampling=Resampling.bilinear,
                        res=(resx, resy),
                        nodata=src.nodata if src.nodata is not None else -999999.0,
                    )
                    _log(f"[DBG] building WarpedVRT -> dst={dst_crs.to_string()}, res={resx}m x {resy}m, nodata={vrt_opts['nodata']}")

                    with WarpedVRT(src, **vrt_opts) as vrt:
                        aoi_dst = _geom_transform(aoi_geom, crs_4326, dst_crs)
                        _log(f"[DBG] masking with AOI in dst CRS; vrt.bounds={vrt.bounds}")
                        out_arr, out_transform = rio_mask(
                            vrt, [mapping(aoi_dst)], crop=True, filled=True, nodata=vrt.nodata
                        )
                        _log(f"[DBG] clip result: shape={out_arr.shape}, dtype={out_arr.dtype}, nodata={vrt.nodata}")
                        if out_arr.size == 0 or np.all(out_arr[0] == vrt.nodata):
                            _log(f"[DBG] no data overlap for {url}, skipping")
                            return None

                        meta = vrt.meta.copy()
                        meta.update(
                            driver="GTiff",
                            height=out_arr.shape[1],
                            width=out_arr.shape[2],
                            transform=out_transform,
                            compress="LZW",
                            tiled=True,
                            BIGTIFF="IF_SAFER"
                        )
                        if (not overwrite) and os.path.exists(out_path):
                            stem, ext = os.path.splitext(out_path)
                            new_out = f"{stem}_{uuid.uuid4().hex[:8]}{ext}"
                            _log(f"[DBG] output exists, renaming to {new_out}")
                            out_path = new_out

                        _log(f"[DBG] writing {out_path}")
                        with rasterio.open(out_path, "w", **meta) as dst:
                            dst.write(out_arr)

                        left, bottom, right, top = rasterio.transform.array_bounds(
                            meta["height"], meta["width"], meta["transform"]
                        )
                        _log(f"[DBG] wrote {out_path}, bounds={left, bottom, right, top}")
                        return dict(
                            path=out_path,
                            bounds=(left, bottom, right, top),
                            crs=str(dst_crs),
                            shape=(meta["height"], meta["width"]),
                            res=(resx, resy),
                            source_url=url,
                        )
            except Exception as e:
                _log(f"[ERR] failed on {url}: {e}")
                return None
        return None

    def _merge_zone_items(items: List[Dict[str, Any]], out_path: str) -> Optional[Dict[str, Any]]:
        """Mosaic multiple clips (already in the same EPSG) and write one GTiff."""
        _log(f"[DBG] mosaicking {len(items)} item(s) -> {out_path}")
        if len(items) == 0:
            _log("[DBG] nothing to mosaic")
            return None
        if len(items) == 1:
            # rename to expected out_path if needed
            if items[0]["path"] != out_path:
                _log(f"[DBG] single item: moving {items[0]['path']} -> {out_path}")
                if overwrite and os.path.exists(out_path):
                    os.remove(out_path)
                os.replace(items[0]["path"], out_path)
                items[0]["path"] = out_path
            return items[0]
        # read each for merging
        srcs = []
        try:
            for it in items:
                _log(f"[DBG] open for merge: {it['path']}")
                srcs.append(rasterio.open(it["path"]))
            mosaic_arr, mosaic_transform = rio_merge(srcs, nodata=srcs[0].nodata)
            meta = srcs[0].meta.copy()
            meta.update(
                height=mosaic_arr.shape[1],
                width=mosaic_arr.shape[2],
                transform=mosaic_transform,
                compress="LZW",
                tiled=True,
                BIGTIFF="IF_SAFER"
            )
            if (not overwrite) and os.path.exists(out_path):
                stem, ext = os.path.splitext(out_path)
                new_out = f"{stem}_{uuid.uuid4().hex[:8]}{ext}"
                _log(f"[DBG] mosaic output exists, renaming to {new_out}")
                out_path = new_out
            _log(f"[DBG] writing mosaic {out_path}")
            with rasterio.open(out_path, "w", **meta) as dst:
                dst.write(mosaic_arr)
            left, bottom, right, top = rasterio.transform.array_bounds(
                meta["height"], meta["width"], meta["transform"]
            )
            _log(f"[DBG] wrote mosaic {out_path}, bounds={left, bottom, right, top}")
            return dict(
                path=out_path,
                bounds=(left, bottom, right, top),
                crs=str(meta["crs"]),
                shape=(meta["height"], meta["width"]),
                res=(abs(meta["transform"].a), abs(meta["transform"].e)),
                sources=[it["source_url"] for it in items],
                skipped_sources=[it["source_url"] for it in items if it.get("_skipped")],
            )
        finally:
            for s in srcs:
                s.close()

    # ---------- normalize AOI ----------
    _log("[DBG] validating AOI")
    if aoi_4326.is_empty:
        _log("[ERR] empty AOI")
        return {"aoi_bounds_4326": aoi_4326.bounds, "mode": mode, "items": [], "error": "Empty AOI."}
    aoi_geom = unary_union([aoi_4326])  # dissolve multiparts if any
    crs_4326 = CRS.from_epsg(4326)
    _log(f"[DBG] AOI bounds (4326): {aoi_geom.bounds}")

    result: Dict[str, Any] = {"aoi_bounds_4326": aoi_geom.bounds, "mode": mode}
    if mode not in ("single_zone", "multi_zone"):
        _log(f"[WARN] unknown mode '{mode}', defaulting to 'single_zone'")
        mode = "single_zone"

    # ---------- SINGLE-ZONE ----------
    if mode == "single_zone":
        lon_c, lat_c = aoi_geom.centroid.x, aoi_geom.centroid.y
        zone = target_zone if target_zone else _utm_zone_from_lon(lon_c)
        target_epsg = (32600 + zone) if lat_c >= 0 else (32700 + zone)
        _log(f"[DBG] single_zone target: zone={zone}, EPSG={target_epsg} (from centroid lon={lon_c:.6f}, lat={lat_c:.6f})")

        per_tile: List[Dict[str, Any]] = []
        skipped: List[str] = []
        for i, url in enumerate(urls):
            out_path = f"{out_prefix}_z{target_epsg}_{i:02d}.tif" if not mosaic else f"{out_prefix}_z{target_epsg}_{i:02d}.clip.tif"
            _log(f"[DBG] -> tile {i}: {url} -> {out_path}")
            info = _clip_reproject_single(url, target_epsg, out_path)
            if info is None:
                _log(f"[DBG] skipped (no overlap or error): {url}")
                skipped.append(url)
            else:
                per_tile.append(info)

        if mosaic and per_tile:
            _log("[DBG] mosaicking single_zone outputs")
            merged = _merge_zone_items(per_tile, f"{out_prefix}.tif")
            if merged is None:
                _log("[WARN] mosaic failed or empty; returning partial info")
                result.update({"target_epsg": target_epsg, "path": None, "sources": urls, "skipped_sources": skipped})
            else:
                merged["target_epsg"] = target_epsg
                merged["skipped_sources"] = skipped
                result.update(merged)
        else:
            result.update({
                "target_epsg": target_epsg,
                "items": per_tile,
                "skipped_sources": skipped
            })
        _log("[DBG] done (single_zone)")
        return result

    # ---------- MULTI-ZONE ----------
    _log("[DBG] multi_zone: determining EPSG per tile")
    zones: Dict[int, Dict[str, Any]] = {}
    per_zone_items: Dict[int, List[Dict[str, Any]]] = {}
    per_zone_skips: Dict[int, List[str]] = {}

    for i, url in enumerate(urls):
        _log(f"[DBG] inspect tile {i}: {url}")
        with _open_env():
            try:
                with rasterio.open(url) as src:
                    src_crs = src.crs or CRS.from_epsg(4326)
                    cx = (src.bounds.left + src.bounds.right) / 2
                    cy = (src.bounds.top + src.bounds.bottom) / 2
                    t = Transformer.from_crs(src_crs, crs_4326, always_xy=True)
                    lon, lat = t.transform(cx, cy)
                    epsg = _utm_epsg_from_lonlat(lon, lat)
            except Exception as e:
                _log(f"[WARN] could not read tile center for {url}: {e}; fallback to AOI centroid")
                lon, lat = aoi_geom.centroid.x, aoi_geom.centroid.y
                epsg = _utm_epsg_from_lonlat(lon, lat)

        out_path = f"{out_prefix}_z{epsg}_{i:02d}.tif" if not mosaic else f"{out_prefix}_z{epsg}_{i:02d}.clip.tif"
        _log(f"[DBG] -> tile {i} target EPSG {epsg} -> {out_path}")
        info = _clip_reproject_single(url, epsg, out_path)
        if info is None:
            per_zone_skips.setdefault(epsg, []).append(url)
            continue
        per_zone_items.setdefault(epsg, []).append(info)

    # write mosaics per zone (optional)
    _log(f"[DBG] writing outputs per zone; zones found: {list(per_zone_items.keys())}")
    for epsg, items in per_zone_items.items():
        if mosaic and items:
            merged = _merge_zone_items(items, f"{out_prefix}.tif")
            if merged:
                zones[epsg] = dict(
                    path=merged["path"],
                    bounds=merged["bounds"],
                    crs=merged["crs"],
                    shape=merged["shape"],
                    res=merged["res"],
                    sources=merged.get("sources", [it["source_url"] for it in items]),
                    skipped_sources=per_zone_skips.get(epsg, []),
                )
            else:
                _log(f"[WARN] mosaic failed for zone {epsg}")
        else:
            zones[epsg] = dict(
                items=items,
                skipped_sources=per_zone_skips.get(epsg, []),
            )

    result["zones"] = zones
    _log("[DBG] done (multi_zone)")
    return result


import trimesh

def dem_to_ply(
    in_tif,
    out_ply,
    stride=1,
    z_scale=1.0,
    drop_nodata=True
):
    """
    Convert a DEM GeoTIFF to a triangle mesh and save as .ply

    Parameters
    ----------
    in_tif : str
        Path to DEM GeoTIFF.
    out_ply : str
        Output PLY file path.
    stride : int
        Subsample factor along rows/cols (e.g., 2 keeps every other cell).
    z_scale : float
        Scale factor to apply to Z values.
    drop_nodata : bool
        If True, skips faces touching nodata cells.
    """
    with rasterio.open(in_tif) as ds:
        dem = ds.read(1)
        transform = ds.transform
        crs = ds.crs
        nodata = ds.nodata

    # Optional downsampling
    dem = dem[::stride, ::stride]
    height, width = dem.shape

    # Create XY from affine transform
    # For a grid, x = x0 + col*ax + row*bx, y = y0 + col*ay + row*by
    rows = np.arange(height)
    cols = np.arange(width)
    cols2d, rows2d = np.meshgrid(cols, rows)
    xs = transform.c + cols2d * transform.a + rows2d * transform.b
    ys = transform.f + cols2d * transform.d + rows2d * transform.e

    zs = dem.astype(np.float64) * z_scale

    # Flatten vertices
    verts = np.column_stack([xs.ravel(), ys.ravel(), zs.ravel()])

    # --- Center mesh at (0,0,0) ---
    center = verts.mean(axis=0)   # centroid
    verts -= center

    # Build faces for a regular grid (2 triangles per cell)
    # Vertex index helper
    def vid(r, c): return r * width + c

    r = np.arange(height - 1)
    c = np.arange(width - 1)
    c2d, r2d = np.meshgrid(c, r)

    v00 = vid(r2d,     c2d)
    v10 = vid(r2d + 1, c2d)
    v01 = vid(r2d,     c2d + 1)
    v11 = vid(r2d + 1, c2d + 1)

    # Two tris per quad: (v00, v10, v11) and (v00, v11, v01)
    faces = np.column_stack([
        np.concatenate([v00.ravel(), v00.ravel()]),
        np.concatenate([v10.ravel(), v11.ravel()]),
        np.concatenate([v11.ravel(), v01.ravel()])
    ])

    if drop_nodata and nodata is not None:
        # Mask out faces that touch any nodata vertex
        mask = (zs != nodata)
        m00 = mask[:-1, :-1]
        m10 = mask[1:,  :-1]
        m01 = mask[:-1, 1:]
        m11 = mask[1:,  1:]
        good_quad = (m00 & m10 & m11 & m00)  # for first tri
        good_quad2 = (m00 & m11 & m01 & m00) # for second tri

        # Interleave the two boolean masks to match faces stacking order
        g1 = good_quad.ravel()
        g2 = good_quad2.ravel()
        good = np.concatenate([g1, g2])
        faces = faces[good]

    # Build and export with trimesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.rezero()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    mesh.vertices -= mesh.centroid

    mesh.export(out_ply)

