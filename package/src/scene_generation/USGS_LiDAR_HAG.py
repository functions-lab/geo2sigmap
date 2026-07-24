import copy
import geopandas as gpd
import json
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pdal
import pyproj
import requests
from shapely.geometry import shape, Point, Polygon
from shapely.ops import transform

def proj_to_3857(poly, orig_crs):
    """
    Function for reprojecting a polygon from a shapefile of any CRS to Web Mercator (EPSG: 3857).
    The original polygon must have a CRS assigned.

    Parameters:
        poly (shapely polygon): User area of interest (AOI)
        orig_crs (str): the original CRS (EPSG) for the shapefile. It is stripped out during import_shapefile_to_shapely() method

    Returns:
        user_poly_proj4326 (shapely polygon): User AOI in EPSG 4326
        user_poly_proj3857 (shapely polygon): User AOI in EPSG 3857
    """
    wgs84 = pyproj.CRS("EPSG:4326")
    web_mercator = pyproj.CRS("EPSG:3857")
    project_gcs = pyproj.Transformer.from_crs(orig_crs, wgs84, always_xy=True).transform
    project_wm = pyproj.Transformer.from_crs(orig_crs, web_mercator, always_xy=True).transform
    user_poly_proj4326 = transform(project_gcs, poly)
    user_poly_proj3857 = transform(project_wm, poly)
    return (user_poly_proj4326, user_poly_proj3857)


def gcs_to_proj(poly):
    """
    Function for reprojecting polygon shapely object from geographic coordinates (EPSG:4326)
    to Web Mercator (EPSG: 3857)).

    Parameters:
        poly (shapely polygon): User area of interest (AOI)

    Returns:
        user_poly_proj3857 (shapely polygon): User AOI in EPSG 3857
    """
    wgs84 = pyproj.CRS("EPSG:4326")
    web_mercator = pyproj.CRS("EPSG:3857")
    project = pyproj.Transformer.from_crs(wgs84, web_mercator, always_xy=True).transform
    user_poly_proj3857 = transform(project, poly)
    return (user_poly_proj3857)


def build_pdal_pipeline(extent_epsg3857, usgs_3dep_dataset_names, pc_resolution, filterNoise=False,
                        reclassify=False, savePointCloud=True, outCRS=3857, pc_outName='filter_test',
                        pc_outType='laz'):
    """
    Build pdal pipeline for requesting, processing, and saving point cloud data. Each processing step is a 'stage'
    in the final pdal pipeline. Each stages is appended to the 'pointcloud_pipeline' object to produce the final pipeline.

    Parameters:
    extent_epsg3857 (shapely polygon): Polygon for user-defined AOI in Web Mercator projection (EPS:3857)Polygon is generated
                            either through the 'handle_draw' methor or by inputing their own shapefile.
    usgs_3dep_dataset_names (str): List of name of the 3DEP dataset(s) that the data will be obtained. This parameter is set
                                determined through intersecttino of the 3DEP and AOI polys.
    pc_resolution (float): The desired resolution of the pointcloud based on the following definition:

                        Source: https://pdal.io/stages/readers.ept.html#readers-ept
                            A point resolution limit to select, expressed as a grid cell edge length.
                            Units correspond to resource coordinate system units. For example,
                            for a coordinate system expressed in meters, a resolution value of 0.1
                            will select points up to a ground resolution of 100 points per square meter.
                            The resulting resolution may not be exactly this value: the minimum possible
                            resolution that is at least as precise as the requested resolution will be selected.
                            Therefore the result may be a bit more precise than requested.

    filterNoise (bool): Option to remove points from USGS Class 7 (Low Noise) and Class 18 (High Noise).
    reclassify (bool): Option to remove USGS classes and run SMRF to classify ground points only. Default == False.
    savePointCloud (bool): Option to save (or not) the point cloud data. If savePointCloud == False,
           the pc_outName and pc_outType parameters are not used and can be any value.
    outCRS (int): Output coordinate reference systemt (CRS), specified by ESPG code (e.g., 3857 - Web Mercator)
    pc_outName (str): Desired name of file on user's local file system. If savePointcloud = False,
                  pc_outName can be in value.
    pc_outType (str):  Desired file extension. Input must be either 'las' or 'laz'. If savePointcloud = False,
                  pc_outName can be in value. If a different file type is requested,the user will get error.

    Returns:
        pointcloud_pipeline (dict): Dictionary of processing stages in sequential order that define PDAL pipeline.

    Raises:
        Exception: If user passes in argument that is not 'las' or 'laz'.
    """

    # this is the basic pipeline which only accesses the 3DEP data
    readers = []
    for name in usgs_3dep_dataset_names:
        url = "https://s3-us-west-2.amazonaws.com/usgs-lidar-public/{}/ept.json".format(name)
        reader = {
            "type": "readers.ept",
            "filename": str(url),
            "polygon": str(extent_epsg3857),
            "requests": 3,
            "resolution": pc_resolution
        }
        readers.append(reader)

    NOAA_FMP_url = "https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com/entwine/geoid18/6209/ept.json"
    NOAA_FMP_reader = {
            "type": "readers.ept",
            "filename": str(NOAA_FMP_url),
            "polygon": str(extent_epsg3857),
            "requests": 3,
            "resolution": pc_resolution
    }
    readers.append(NOAA_FMP_reader)
    pointcloud_pipeline = {
        "pipeline":
            readers
    }

    if filterNoise == True:
        filter_stage = {
            "type": "filters.range",
            "limits": "Classification![7:7], Classification![18:18]"
        }

        pointcloud_pipeline['pipeline'].append(filter_stage)

    if reclassify == True:
        remove_classes_stage = {
            "type": "filters.assign",
            "value": "Classification = 0"
        }

        classify_ground_stage = {
            "type": "filters.smrf"
        }

        reclass_stage = {
            "type": "filters.range",
            "limits": "Classification[2:2]"
        }

        pointcloud_pipeline['pipeline'].append(remove_classes_stage)
        pointcloud_pipeline['pipeline'].append(classify_ground_stage)
        pointcloud_pipeline['pipeline'].append(reclass_stage)

    reprojection_stage = {
        "type": "filters.reprojection",
        "out_srs": outCRS
    }

    pointcloud_pipeline['pipeline'].append(reprojection_stage)
    
    remove_negative_stage = {
        "type": "filters.assign",
        "value": [
            "Z = 0 WHERE Z < 0"
        ]
    }
    
    pointcloud_pipeline['pipeline'].append(remove_negative_stage)
        
        

    if savePointCloud == True:

        if pc_outType == 'las':
            savePC_stage = {
                "type": "writers.las",
                "filename": str(pc_outName) + '.' + str(pc_outType),
            }
        elif pc_outType == 'laz':
            savePC_stage = {
                "type": "writers.las",
                "compression": "laszip",
                "filename": str(pc_outName) + '.' + str(pc_outType),
            }
        else:
            raise Exception("pc_outType must be 'las' or 'laz'.")

        pointcloud_pipeline['pipeline'].append(savePC_stage)

    return pointcloud_pipeline


def make_DEM_pipeline(extent_epsg3857, usgs_3dep_dataset_name, pc_resolution, dem_resolution,
                      filterNoise=True, reclassify=False, savePointCloud=False, outCRS=3857,
                      pc_outName='filter_test', pc_outType='laz', demType='dtm', gridMethod='idw',
                      dem_outName='dem_test', dem_outExt='tif', driver="GTiff"):
    """
    Build pdal pipeline for creating a digital elevation model (DEM) product from the requested point cloud data. The
    user must specify whether a digital terrain (bare earth) model (DTM) or digital surface model (DSM) will be created,
    the output DTM/DSM resolution, and the gridding method desired.

    The `build_pdal_pipeline() method is used to request the data from the Amazon Web Services ept bucket, and the
    user may define any processing steps (filtering, reclassifying, reprojecting). The user must also specify whether
    the point cloud should be saved or not. Saving the point cloud is not necessary for the generation of the DEM.

    Parameters:
        extent_epsg3857 (shapely polygon): User-defined AOI in Web Mercator projection (EPS:3857). Polygon is generated
                                           either through the 'handle_draw' methor or by inputing their own shapefile.
                                           This parameter is set automatically when the user-defined AOI is chosen.
        usgs_3dep_dataset_names (list): List of name of the 3DEP dataset(s) that the data will be obtained. This parameter is set
                                        determined through intersecttino of the 3DEP and AOI polys.
        pc_resolution (float): The desired resolution of the pointcloud based on the following definition:

                        Source: https://pdal.io/stages/readers.ept.html#readers-ept
                            A point resolution limit to select, expressed as a grid cell edge length.
                            Units correspond to resource coordinate system units. For example,
                            for a coordinate system expressed in meters, a resolution value of 0.1
                            will select points up to a ground resolution of 100 points per square meter.
                            The resulting resolution may not be exactly this value: the minimum possible
                            resolution that is at least as precise as the requested resolution will be selected.
                            Therefore the result may be a bit more precise than requested.

        pc_outName (str): Desired name of file on user's local file system. If savePointcloud = False,
                          pc_outName can be in value.
        pc_outType (str): Desired file extension. Input must be either 'las' or 'laz'. If savePointcloud = False,
                          pc_outName can be in value. If a different file type is requested,the user will get error.

        dem_resolution (float): Desired grid size (in meters) for output raster DEM
        filterNoise (bool): Option to remove points from USGS Class 7 (Low Noise) and Class 18 (High Noise).
        reclassify (bool): Option to remove USGS classes and run SMRF to classify ground points only. Default == False.
        savePointCloud (bool): Option to save (or not) the point cloud data. If savePointCloud == False, the pc_outName
                               and pc_outType parameters are not used and can be any value.

        outCRS (int): Output coordinate reference systemt (CRS), specified by ESPG code (e.g., 3857 - Web Mercator)
        pc_outName (str): Desired name of file on user's local file system. If savePointcloud = False,
                          pc_outName can be in value.
        pc_outType (str): Desired file extension. Input must be either 'las' or 'laz'. If a different file type is requested,
                    the user will get error stating "Extension must be 'las' or 'laz'". If savePointcloud = False,
                    pc_outName can be in value.
        demType (str): Type of DEM produced. Input must 'dtm' (digital terrain model) or 'dsm' (digital surface model).
        gridMethod (str): Method used. Options are 'min', 'mean', 'max', 'idw'.
        dem_outName (str): Desired name of DEM file on user's local file system.
        dem_outExt (str): DEM file extension. Default is TIF.
        driver (str): File format. Default is GTIFF

    Returns:
        dem_pipeline (dict): Dictionary of processing stages in sequential order that define PDAL pipeline.
    Raises:
        Exception: If user passes in argument that is not 'las' or 'laz'.
        Exception: If user passes in argument that is not 'dtm' or 'dsm'

    """

    dem_pipeline = build_pdal_pipeline(extent_epsg3857, usgs_3dep_dataset_name, pc_resolution,
                                       filterNoise, reclassify, savePointCloud, outCRS, dem_outName, pc_outType)

    if demType == 'dsm':
        dem_stage = {
            "type": "writers.gdal",
            "filename": str(dem_outName) + '.' + str(dem_outExt),
            "gdaldriver": driver,
            "nodata": -9999,
            "output_type": gridMethod,
            "resolution": float(dem_resolution),
            "gdalopts": "COMPRESS=LZW,TILED=YES,blockxsize=256,blockysize=256,COPY_SRC_OVERVIEWS=YES"
        }

    elif demType == 'dtm':
        groundfilter_stage = {
            "type": "filters.range",
            "limits": "Classification[2:2]"
        }

        dem_pipeline['pipeline'].append(groundfilter_stage)

        dem_stage = {
            "type": "writers.gdal",
            "filename": str(dem_outName) + '.' + str(dem_outExt),
            "gdaldriver": driver,
            "nodata": -9999,
            "output_type": gridMethod,
            "resolution": float(dem_resolution),
            "gdalopts": "COMPRESS=LZW,TILED=YES,blockxsize=256,blockysize=256,COPY_SRC_OVERVIEWS=YES"
        }

    elif demType == "hag":
        hagfilter_stage = {
            "type": "filters.hag_nn",
            "count": 10
        }
        dem_pipeline['pipeline'].append(hagfilter_stage)

        dem_stage = {
            "type": "writers.gdal",
            "filename": str(dem_outName) + '.' + str(dem_outExt),
            "gdaldriver": driver,
            "nodata": -9999,
            "output_type": gridMethod,
            "dimension": "HeightAboveGround",
            "resolution": float(dem_resolution),
            "gdalopts": "COMPRESS=LZW,TILED=YES,blockxsize=256,blockysize=256,COPY_SRC_OVERVIEWS=YES"
        }
    else:
        raise Exception("demType must be 'dsm' or 'dtm' or 'hag'.")

    dem_pipeline['pipeline'].append(dem_stage)

    return dem_pipeline

class CoverageChecker:
    def __init__(self, file_path: str="resources.geojson"):
        print("Loading 3DEP resources.geojson...")
        if not os.path.exists("resources.geojson"):
            #print("Requesting, loading, and projecting 3DEP dataset polygons...")
            url = 'https://raw.githubusercontent.com/hobuinc/usgs-lidar/master/boundaries/resources.geojson'
            r = requests.get(url)
            with open('resources.geojson', 'w') as f:
                f.write(r.content.decode("utf-8"))
        with open('resources.geojson', 'r') as f:
            df =  gpd.read_file(f) 
            self.names = df['name']
            self.urls = df['url']
            self.num_points = df['count']
            self.df =df
            projected_geoms = []
            for geometry in self.df['geometry']:
                projected_geoms.append(gcs_to_proj(geometry))

            self.geometries_GCS = self.df['geometry']
            self.geometries_EPSG3857 = gpd.GeoSeries(projected_geoms)

    def is_polygon_inside_3DEP(self, polygon: Polygon, polygon_CRS="EPSG:3857") -> bool:
        """
        Checks if a given polygon is inside LiDAR dataset geometries.

        Parameters:
            polygon (Polygon): The Shapely polygon to check.

        Returns:
            bool: True if condition is met, otherwise False.
        """
        

        AOI_EPSG3857 = proj_to_3857(polygon, "EPSG:4326")[1]

        intersecting_polys = []
        for i, geom in enumerate(self.geometries_EPSG3857):
            #if geom.intersects(AOI_EPSG3857):
            if AOI_EPSG3857.within(geom):
                intersecting_polys.append((self.names[i], self.geometries_GCS[i], self.geometries_EPSG3857[i], self.urls[i], self.num_points[i]))

        if len(intersecting_polys) ==0:
            return False
        return True

    

def generate_hag(polygon, data_dir, CRS="EPSG:3857"):
    """
    Generate Height Above Ground (HAG) data for a given polygon area.
    
    Parameters:
        polygon (shapely.geometry.Polygon): The area of interest polygon in EPSG:4326
        data_dir (str): Directory where output files will be saved
        
    Returns:
        None: Saves HAG data as GeoTIFF file in the specified data directory
    """
    if not os.path.exists("resources.geojson"):
        print("Requesting, loading, and projecting 3DEP dataset polygons...")
        url = 'https://raw.githubusercontent.com/hobuinc/usgs-lidar/master/boundaries/resources.geojson'
        r = requests.get(url)
        with open('resources.geojson', 'w') as f:
            f.write(r.content.decode("utf-8"))
    else:
        print("Loading local 3DEP dataset polygons...")

    with open('resources.geojson', 'r') as f:
        df = gpd.read_file(f)
        names = df['name']
        urls = df['url']
        num_points = df['count']

    projected_geoms = []
    for geometry in df['geometry']:
        projected_geoms.append(gcs_to_proj(geometry))

    geometries_GCS = df['geometry']
    geometries_EPSG3857 = gpd.GeoSeries(projected_geoms)

    print('Done. 3DEP polygons downloaded and projected to ', CRS.to_string())

    AOI_EPSG3857 = proj_to_3857(polygon, "EPSG:4326")[1]
    print("Area of Interest:", AOI_EPSG3857)

    intersecting_polys = []
    for i, geom in enumerate(geometries_EPSG3857):
        # if geom.intersects(AOI_EPSG3857):
        if AOI_EPSG3857.within(geom):
            intersecting_polys.append((names[i], geometries_GCS[i], geometries_EPSG3857[i], urls[i], num_points[i]))
            # print information about dataset
            print(names[i])
            print(urls[i])

    print(f"Found {len(intersecting_polys)} intersecting datasets")
    if len(intersecting_polys) ==0:
        raise ValueError("No LiDAR data available for the selected region.")

    usgs_3dep_datasets = [poly[0] for poly in intersecting_polys]
    
    pointcloud_resolution = 5.0
    dsm_resolution = 5.0
    
    try:
        dsm_pipeline = make_DEM_pipeline(
            AOI_EPSG3857.wkt, 
            usgs_3dep_datasets, 
            pointcloud_resolution, 
            dsm_resolution,
            filterNoise=True, 
            reclassify=False, 
            savePointCloud=True, 
            outCRS=CRS.to_string(),
            pc_outName=os.path.join(data_dir, 'pointcloud_test'), 
            pc_outType='laz', 
            demType='hag',
            gridMethod='idw', 
            dem_outName=os.path.join(data_dir, 'test_hag'), 
            dem_outExt='tif', 
            driver="GTiff"
        )
        dsm_pipeline = pdal.Pipeline(json.dumps(dsm_pipeline))
        dsm_pipeline.execute()
        print("Successfully generated HAG data")
        
    except Exception as e:
        print(f"Error generating HAG data: {e}")

def main():
    """
    Main function to test the HAG generation functionality.
    Creates a test polygon and generates HAG data for it.
    """ 
                                                                                                                                                                                                                        
    # Create a test polygon (example: a small area in Colorado)
    test_polygon = Polygon([
        # (-80.853247558587682, 35.218940598812395),
        # (-80.848318480301486, 35.234570940294113),
        # (-80.827192375413446, 35.230082242372461),
        # (-80.832124690884896, 35.214454483322911),
        # (-80.853247558587682, 35.218940598812395),
        (-71.0602, 42.3512),
        (-71.0602, 42.3591),
        (-71.0484, 42.3591),
        (-71.0484, 42.3512),
        (-71.0602, 42.3512)
    ])
    
    # Create output directory if it doesn't exist
    data_dir = "output"
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate HAG data
    print("Starting HAG generation...")
    generate_hag(test_polygon, data_dir)
    print("HAG generation complete")

if __name__ == "__main__":
    main()



'''
Prerequisites:
1. Install PDAL
    conda install -c conda-forge pdal
    conda install -c conda-forge pdal gdal sqlite
    ln /home/test/miniconda3/envs/geo2sigmap/lib/libsqlite3.so.3.50.1 /home/test/miniconda3/envs/geo2sigmap/lib/libsqlite3.so

'''