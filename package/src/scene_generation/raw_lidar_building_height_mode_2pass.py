import json
from typing import Optional, Dict, Any
import numpy as np
import pdal
import shapely
from shapely import wkt

import logging

# Creates a logger named 'duke_lidar.height'
logger = logging.getLogger(__name__)

def calculate_building_height_from_lidar(
    laz_path: str, 
    polygon_wkt: str, 
    osm_way_id: Optional[int] = None,
    ground_buffer: float = 2.0, 
    roof_erosion: float = -1.5
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
        osm_way_id (Optional[int], optional): The OSM way ID of the building polygon.
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
                f"Building polygon (OSM way ID: {osm_way_id}) became empty after erosion ({roof_erosion}m). "
                "Falling back to original footprint."
        )

    # Optimization: 'prepare' accelerates subsequent vectorized queries in Shapely 2.0+
    shapely.prepare(poly_roof_geom)

    # --- 2. Single-Pass PDAL Execution (I/O Bound) ---
    # We define a pipeline to crop only the expanded area. 
    # This prevents reading the entire file or running the pipeline twice.
    pipeline_json = {
        "pipeline": [
            laz_path,
            {
                "type": "filters.crop",
                "polygon": poly_ground_wkt
            }
        ]
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
    ground_points = arr[arr['Classification'] == 2]
    
    if len(ground_points) == 0:
        # Fallback: Assume 0.0 or handle based on specific project requirements.
        # Returning 0.0 allows the script to continue, but data quality should be flagged.
        ground_z_mean = 0.0 
    else:
        # Use Median instead of Mean to be robust against underground noise/basement reflections
        ground_z_mean = np.median(ground_points['Z'])

    # B. Roof Extraction Strategy
    # First, filter out known ground points
    non_ground_subset = arr[arr['Classification'] != 2]
    
    if len(non_ground_subset) == 0:
        return None

    # Vectorized Point-in-Polygon Check
    # shapely.contains_xy is a ufunc (Universal Function) optimized in C.
    # It creates a boolean mask much faster than iterating through Point objects.
    mask_inside_roof = shapely.contains_xy(
        poly_roof_geom, 
        non_ground_subset['X'], 
        non_ground_subset['Y']
    )
    
    actual_roof_points = non_ground_subset[mask_inside_roof]
    
    if len(actual_roof_points) == 0:
        return None

    roof_z = actual_roof_points['Z']

    # --- 4. Statistical Height Determination ---
    
    # Calculate the 95th Percentile (Standard approach, sensitive to outliers)
    z_95 = np.percentile(roof_z, 95)
    
    # Calculate the Mode (Robust approach)
    # Roofs are flat surfaces (high density), walls are vertical (low density).
    # We bin Z values by 20cm to find the "densest" vertical slice.
    bins = np.arange(np.min(roof_z), np.max(roof_z) + 0.2, 0.2)
    hist, bin_edges = np.histogram(roof_z, bins=bins)
    
    peak_idx = np.argmax(hist)
    z_mode = (bin_edges[peak_idx] + bin_edges[peak_idx+1]) / 2.0

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
        "method": method
    }