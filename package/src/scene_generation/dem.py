from pathlib import Path
from .USGS1mLocator_parallel import *
from .dem_utils import *
from importlib_resources import files
from .dem_gpkg_data import get_fesm_paths
def generate_terrain_mesh_dem(aoi_poly, ply_save_path,):
    ply_save_path = Path(ply_save_path)


    gpkg_files = get_fesm_paths()
    
    locator = USGS1mLocator(
        gpkg_files[1],
        gpkg_files[0],
        debug=True
    )
    locator.set_debug(False)
    links = locator.get_links(aoi_poly)
    if len(links) < 1:
        raise Exception("No DEM found!")
    mode = "single_zone"
    if len(links)>1: 
        mode = "multi_zone"     
    clip_reproject_dem_to_wgs84_utm(aoi_poly, links, mode=mode, mosaic=True, out_prefix=ply_save_path.parent / "dem")
    dem_to_ply(ply_save_path.parent / "dem.tif",ply_save_path, stride=1, z_scale=1.0)
    del locator
    
    
    

