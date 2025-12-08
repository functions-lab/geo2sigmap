import pdal
import numpy as np
import shapely
from shapely import wkt
import json

class LiDARContext:
    def __init__(self, laz_path):
        """
        Loads the ENTIRE LiDAR file into RAM and optimizes it for querying.
        """
        print(f"Loading {laz_path} into RAM... (This may take a moment)")
        
        # 1. Read ONLY essential dimensions to save RAM
        # We don't need Intensity, Red, Green, Blue, UserData, etc.
        pipeline_json = {
            "pipeline": [
                {
                    "type": "readers.copc",
                    "filename": laz_path,
                    #"dimensions": "X,Y,Z,Classification" 
                }
            ]
        }
        pipeline = pdal.Pipeline(json.dumps(pipeline_json))
        pipeline.execute()

        # 2. Optimization: Keep only essential columns to save RAM
        # The pipeline returns ALL dimensions (Intensity, GPS time, etc.)
        # We slice the numpy array to keep only what we need.
        # This creates a copy, allowing the original big array to be garbage collected.
        raw_data = pipeline.arrays[0]
        self.data = raw_data[['X', 'Y', 'Z', 'Classification']]
        
        # Explicitly delete the raw data to free memory immediately
        del raw_data

        
        
        # 2. Store as structured NumPy array
        # self.data = pipeline.arrays[0]
        print(f"Loaded {len(self.data):,} points.")

        # 3. THE OPTIMIZATION TRICK: Sort by X-Coordinate
        # This allows us to use Binary Search (O(log n)) later
        print("Sorting data for fast slicing...")
        sort_idx = np.argsort(self.data['X'])
        self.data = self.data[sort_idx]
        
        # Pre-calculate X column for fast searchsorted calls
        self.xs = self.data['X']
        
    def get_height(self, polygon_wkt, osm_id=None, ground_buffer=2.0, roof_erosion=-1.5):
        """
        Calculates height using In-Memory slicing (No Disk I/O).
        """
        # --- 1. Geometry Prep ---
        try:
            poly_geom = wkt.loads(polygon_wkt)
        except:
            return None

        poly_ground = poly_geom.buffer(ground_buffer)
        minx, miny, maxx, maxy = poly_ground.bounds
        
        # --- 2. FAST CROP (NumPy Binary Search) ---
        # Find the start and end indices of the X-range instantly
        start = np.searchsorted(self.xs, minx, side='left')
        end = np.searchsorted(self.xs, maxx, side='right')
        
        # Slice the array (Zero Copy - very fast)
        # We now have a strip of the world containing the building
        x_strip = self.data[start:end]
        
        if len(x_strip) == 0:
            return None

        # Filter by Y-range (Vectorized boolean mask)
        # This reduces the strip to a box
        mask_box = (x_strip['Y'] >= miny) & (x_strip['Y'] <= maxy)
        points_box = x_strip[mask_box]
        
        if len(points_box) == 0:
            return None

        # --- 3. Geometric Filtering (Shapely) ---
        # Now we refine the box to the exact polygon shapes
        
        # A. Ground (Class 2) inside Expanded Polygon
        # Note: shapely.contains_xy requires Shapely 2.0+
        mask_ground_poly = shapely.contains_xy(poly_ground, points_box['X'], points_box['Y'])
        ground_subset = points_box[mask_ground_poly & (points_box['Classification'] == 2)]
        
        if len(ground_subset) == 0:
            return None
        else:
            ground_z_mean = np.median(ground_subset['Z'])

        # B. Roof (Non-Ground) inside Eroded Polygon
        poly_roof = poly_geom.buffer(roof_erosion)
        if poly_roof.is_empty: poly_roof = poly_geom # Fallback
        
        mask_roof_poly = shapely.contains_xy(poly_roof, points_box['X'], points_box['Y'])
        roof_subset = points_box[mask_roof_poly & (points_box['Classification'] != 2)]
        
        if len(roof_subset) == 0:
            return None

        roof_z = roof_subset['Z']

        # --- 4. Stats (Same as before) ---
        z_95 = np.percentile(roof_z, 95)
        
        # Mode Calculation
        bins = np.arange(np.min(roof_z), np.max(roof_z) + 0.2, 0.2)
        hist, bin_edges = np.histogram(roof_z, bins=bins)
        peak_idx = np.argmax(hist)
        z_mode = (bin_edges[peak_idx] + bin_edges[peak_idx+1]) / 2.0

        final_roof_z = z_mode if (z_95 - z_mode) > 3.0 else z_95

        return {
            "building_height": float(final_roof_z - ground_z_mean),
            "method": "mode" if (z_95 - z_mode) > 3.0 else "95th"
        }