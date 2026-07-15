import logging
import math
import os
import numpy as np
import open3d as o3d
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import osmnx as ox
import datetime
import pyvista as pv
import shapely
import pandas as pd

from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely import affinity
from tqdm import tqdm
from triangle import triangulate
from PIL import Image, ImageDraw, ImageChops
from pyproj import Transformer
import open3d.core as o3c
from pathlib import Path

from .dem import generate_terrain_mesh_dem
from .utils import *
from .itu_materials import ITU_MATERIALS

from .overture_buildings import (
    # HEIGHT_MODE_LIDAR_OSM,
    # HEIGHT_MODE_OVERTURE,
    load_overture_building_parts_for_aoi,
    load_overture_buildings_for_aoi,
    # normalize_building_height_mode,
    resolve_building_base_height,
    resolve_building_height,
    should_include_overture_parent_footprint,
)
from .overture_roads import (
    build_road_polygon,
    load_overture_roads_for_aoi,
    resolve_road_width,
)

# Create a module-level logger
logger = logging.getLogger(__name__)

# Vertical offset applied to road meshes so they don't sit exactly coplanar
# with the ground/terrain mesh (which causes z-fighting in the renderer).
ROAD_Z_OFFSET_M = 0.1

class Scene:
    """
    A class that encapsulates the logic for creating the ditital twins for a given
    bounding box with building information querying from OpenStreetMap server and
    ground mesh from lidar.


    Usage:
        scene_instance = Scene()
        scene_instance(
            points=[(lon1, lat1), (lon2, lat2), ...],
            data_dir="path/to/output",
            hag_tiff_path="path/to/hag.tiff",
            lidar_height_calibration=True,
            generate_building_map=True
        )
    """

    def __call__(
        self,
        points,
        data_dir,
        hag_tiff_path,
        osm_server_addr: str = "https://overpass-api.de/api/interpreter",
        lidar_height_calibration: bool = True,
        generate_building_map: bool = True,
        write_ply_ascii: bool = False,
        ground_scale: float = 1.5,
        ground_material_type="mat-itu_wet_ground",
        rooftop_material_type="mat-itu_metal",
        wall_material_type="mat-itu_concrete",
        lidar_terrain:bool = False,
        dem_terrain:bool = False,
        gen_lidar_terrain_only:bool = False,
        building_data_source: str = "overture",
        generate_roads: bool = True,
        road_material_type: str = None,
    ):
        """
        Generate a ground mesh from the given polygon (defined by `points`), extrude them into 3D meshes,
        and optionally produce a 2D building-height map.

        Parameters
        ----------
        points : list of (float, float)
            Coordinates defining the polygon (in WGS84 lon/lat).
        data_dir : str
            Directory where output files (XML, meshes, etc.) will be saved.
        lidar_height_calibration : bool
            LiDAR data is used as a step in the height prioritization hierarchy before random fallback.
        generate_building_map : bool, optional
            If True, generate a 2D building map image (and save as a NumPy file).
        write_ply_ascii : bool, optional
            If True, write the ply file in ascii format, otherwise binary format will be used.
        ground_scale : float, optional
            The ratio to scale the ground polygon. TODO:Add examples to show why need scale. OSMNX query intersection.
        generate_roads : bool, optional
            If True (default) and ``building_data_source == "overture"``, query Overture
            transportation segments (``subtype == "road"``) for the AOI, buffer each
            centerline into a flat footprint using a per-class default width, and add
            them to the scene as flat meshes.
        road_material_type : str, optional
            ITU material id (a key in ``ITU_MATERIALS``) used for road meshes.
            Defaults to ``None``, which reuses ``ground_material_type`` so roads are
            physically indistinguishable from the ground surface. Pass a different
            material (e.g. ``"mat-itu_chipboard"``) to give roads a distinct color
            in the Mitsuba/Sionna preview.

        Returns
        -------
        np.ndarray
            If generate_building_map is True, returns a 2D NumPy array of building heights.
            Otherwise, returns None.
        """
        # overture data source
        if building_data_source == "overture":
            if ground_material_type not in ITU_MATERIALS:
                raise ValueError(f"Invalid ground material type: {ground_material_type}")
            if rooftop_material_type not in ITU_MATERIALS:
                raise ValueError(f"Invalid rooftop material type: {rooftop_material_type}")
            if wall_material_type not in ITU_MATERIALS:
                raise ValueError(f"Invalid wall material type: {wall_material_type}")
            if road_material_type is not None and road_material_type not in ITU_MATERIALS:
                raise ValueError(f"Invalid road material type: {road_material_type}")
            resolved_road_material_type = road_material_type or ground_material_type

            # Determine the UTM projection from the first point
            projection_UTM_EPSG_code = get_utm_epsg_code_from_gps(
                points[0][0], points[0][1]
            )
            logger.info(f"Using UTM Zone: {projection_UTM_EPSG_code}")

            # Create transformations between WGS84 (EPSG:4326) and UTM
            to_projection = Transformer.from_crs(
                "EPSG:4326", projection_UTM_EPSG_code, always_xy=True
            )
            to_4326 = Transformer.from_crs(
                projection_UTM_EPSG_code, "EPSG:4326", always_xy=True
            )

            # ---------------------------------------------------------------------
            # 2) Prepare output directories and camera / material settings
            # ---------------------------------------------------------------------
            mesh_data_dir = os.path.join(data_dir, "mesh")
            os.makedirs(os.path.join(mesh_data_dir), exist_ok=True)

            def print_material_info(surface_name, material_type):
                if isinstance(ITU_MATERIALS[material_type]["lower_freq_limit"], list):
                    logger.info(
                        "{:<35}{:<20} | Frequency Range: {:^5} - {:^5} (GHz) | {:^5} - {:^5} (GHz)".format(
                            "{} Material Type:".format(surface_name),
                            ITU_MATERIALS[material_type]["name"],
                            print_if_int(
                                ITU_MATERIALS[material_type]["lower_freq_limit"][0] / 1e9
                            ),
                            print_if_int(
                                ITU_MATERIALS[material_type]["upper_freq_limit"][0] / 1e9
                            ),
                            print_if_int(
                                ITU_MATERIALS[material_type]["lower_freq_limit"][1] / 1e9
                            ),
                            print_if_int(
                                ITU_MATERIALS[material_type]["upper_freq_limit"][1] / 1e9
                            ),
                        )
                    )
                else:
                    logger.info(
                        "{:<35}{:<20} | Frequency Range: {:^5} - {:^5} (GHz)".format(
                            "{} Material Type:".format(surface_name),
                            ITU_MATERIALS[material_type]["name"],
                            print_if_int(
                                ITU_MATERIALS[material_type]["lower_freq_limit"] / 1e9
                            ),
                            print_if_int(
                                ITU_MATERIALS[material_type]["upper_freq_limit"] / 1e9
                            ),
                        )
                    )

            logger.info("")
            print_material_info("Ground", ground_material_type)
            print_material_info("Building Rooftop", rooftop_material_type)
            print_material_info("Building Wall", wall_material_type)
            logger.info("")



            camera_settings = {
                "rotation": (0, 0, -90),  # Assuming Z-up orientation
                "fov": 42.854885,
            }

            # ---------------------------------------------------------------------
            # 3) Build the XML scene root
            # ---------------------------------------------------------------------


            # Default Mitsuba rendering parameters
            spp_default = 4096
            resx_default = 1024
            resy_default = 1024

            scene = ET.Element("scene", version="2.1.0")
            # Default integrator / film settings
            ET.SubElement(scene, "default", name="spp", value=str(spp_default))
            ET.SubElement(scene, "default", name="resx", value=str(resx_default))
            ET.SubElement(scene, "default", name="resy", value=str(resy_default))

            ET.SubElement(scene, "default", name="scenegen_version", value=str(get_package_version()))
            ET.SubElement(scene, "default", name="scenegen_create_time", value=str(datetime.datetime.now()))

            ET.SubElement(scene, "default", name="scenegen_min_lat", value=str(points[0][1]))
            ET.SubElement(scene, "default", name="scenegen_max_lat", value=str(points[1][1]))
            ET.SubElement(scene, "default", name="scenegen_min_lon", value=str(points[0][0]))
            ET.SubElement(scene, "default", name="scenegen_max_lon", value=str(points[2][0]))
            
            ET.SubElement(scene, "default", name="scenegen_ground_material", value=str(ground_material_type))
            ET.SubElement(scene, "default", name="scenegen_rooftop_material", value=str(rooftop_material_type))
            ET.SubElement(scene, "default", name="scenegen_wall_material", value=str(wall_material_type))

            ET.SubElement(scene, "default", name="scenegen_UTM_zone", value=str(projection_UTM_EPSG_code))
            
            integrator = ET.SubElement(scene, "integrator", type="path")
            ET.SubElement(integrator, "integer", name="max_depth", value="12")

            # Define materials
            for material_id, material_content in ITU_MATERIALS.items():
                
                # Temporary workaround for Sionna v1.1 : Skip vacuum and P.527 materials.
                if "vacuum" in material_id in material_id:
                    continue

                if "P.527" not in material_id:
                    bsdf_twosided = ET.SubElement(
                        scene, "bsdf", type="twosided", id=material_id
                    )
                    bsdf_diffuse = ET.SubElement(bsdf_twosided, "bsdf", type="diffuse")
                    rgb = material_content["mitsuba_color"]
                    ET.SubElement(
                        bsdf_diffuse,
                        "rgb",
                        value=f"{rgb[0]} {rgb[1]} {rgb[2]}",
                        name="reflectance",
                    )
                else:
                    bsdf_twosided = ET.SubElement(
                        scene, "bsdf", type="radio-material", id=material_id
                    )
                    

            # Add emitter (constant environment light)
            emitter = ET.SubElement(scene, "emitter", type="constant", id="World")
            ET.SubElement(
                emitter, "rgb", value="1.000000 1.000000 1.000000", name="radiance"
            )

            # Add camera (sensor)
            sensor = ET.SubElement(scene, "sensor", type="perspective", id="Camera")
            ET.SubElement(sensor, "string", name="fov_axis", value="x")
            ET.SubElement(sensor, "float", name="fov", value=str(camera_settings["fov"]))
            ET.SubElement(
                sensor, "float", name="principal_point_offset_x", value="0.000000"
            )
            ET.SubElement(
                sensor, "float", name="principal_point_offset_y", value="-0.000000"
            )
            ET.SubElement(sensor, "float", name="near_clip", value="0.100000")
            ET.SubElement(sensor, "float", name="far_clip", value="10000.000000")
            sionna_transform = ET.SubElement(sensor, "transform", name="to_world")
            ET.SubElement(
                sionna_transform, "rotate", x="1", angle=str(camera_settings["rotation"][0])
            )
            ET.SubElement(
                sionna_transform, "rotate", y="1", angle=str(camera_settings["rotation"][1])
            )
            ET.SubElement(
                sionna_transform, "rotate", z="1", angle=str(camera_settings["rotation"][2])
            )
            camera_position = np.array([0, 0, 100])  # Adjust camera height
            ET.SubElement(
                sionna_transform, "translate", value=" ".join(map(str, camera_position))
            )
            sampler = ET.SubElement(sensor, "sampler", type="independent")
            ET.SubElement(sampler, "integer", name="sample_count", value="$spp")
            film = ET.SubElement(sensor, "film", type="hdrfilm")
            ET.SubElement(film, "integer", name="width", value="$resx")
            ET.SubElement(film, "integer", name="height", value="$resy")

            # ---------------------------------------------------------------------
            # 4) Create ground polygon (in UTM) and ground mesh
            # ---------------------------------------------------------------------

            # # Define the points in counter-clockwise order to create the polygon
            # points = [top_left, top_right, bottom_right, bottom_left]
            ground_polygon_4326 = shapely.geometry.Polygon(points)
            ground_polygon_4326_bbox = ground_polygon_4326.bounds

            # Transform each WGS84 coordinate into UTM
            coords = [to_projection.transform(x, y) for x, y in points]
            ground_polygon = shapely.geometry.Polygon(coords)
            ground_polygon_bbox = ground_polygon.bounds

            self._ground_polygon_envelope_UTM = ground_polygon.envelope

            center_x = ground_polygon.envelope.centroid.x
            center_y = ground_polygon.envelope.centroid.y

            ET.SubElement(scene, "default", name="scenegen_center_lat", value=f"{ground_polygon_4326.envelope.centroid.y:.6f}")
            ET.SubElement(scene, "default", name="scenegen_center_lon", value=f"{ground_polygon_4326.envelope.centroid.x:.6f}")


            # ---------------------------------------------------------------------
            # 0) Query USGS 3DEP LiDAR data and generate GEOTIFF file for building height calibration
            # ---------------------------------------------------------------------
            try:
                laz_file_path = Path(os.path.join(data_dir, "test_hag.laz"))
                tif_file_path = Path(os.path.join(data_dir, "test_hag.tif"))
                if lidar_terrain or lidar_height_calibration:
                    if not laz_file_path.exists() or not tif_file_path.exists():
                        
                        from .USGS_LiDAR_HAG import generate_hag
                        
                        generate_hag(affinity.scale(ground_polygon_4326, xfact=ground_scale, yfact=ground_scale, origin='centroid'), data_dir, projection_UTM_EPSG_code)
                    if lidar_height_calibration and hag_tiff_path is None and tif_file_path.exists():
                        hag_tiff_path = str(tif_file_path)
                    
                if lidar_terrain:
                    from .lidar_terrain_mesh import generate_terrain_mesh
        
                    assert laz_file_path.exists(), f"LAZ file does not exist: {laz_file_path}"
        
                    assert tif_file_path.exists(), f"TIF file does not exist: {tif_file_path}"
                    print("Checking lidar_terrain.ply")
                    if not Path(os.path.join(data_dir,"mesh" ,"lidar_terrain.ply")).exists():

                        if dem_terrain:
                            generate_terrain_mesh_dem(
                                affinity.scale(ground_polygon_4326, xfact=ground_scale, yfact=ground_scale, origin='centroid'),
                                os.path.join(mesh_data_dir, f"lidar_terrain.ply")
                            )
                        else:
                            
                            generate_terrain_mesh(os.path.join(data_dir, "test_hag.laz"),
                                os.path.join(mesh_data_dir, f"lidar_terrain.ply"), src_crs=projection_UTM_EPSG_code, dest_crs=projection_UTM_EPSG_code,
                                plot_figures=False, center_x=center_x, center_y=center_y
                            )
                if gen_lidar_terrain_only:
                    print("gen_lidar_terrain_only: True")
                    return
            except Exception as e:
                print(e)
            if lidar_terrain:
                lidar_terrain_ply_path = Path(os.path.join(data_dir,"mesh" ,"lidar_terrain.ply"))
                if not lidar_terrain_ply_path.exists():
                    return 1
                surface_mesh = pv.read(lidar_terrain_ply_path)
            #######Open3D#######
            outer_xy = unique_coords(
                reorder_localize_coords(ground_polygon.exterior, center_x, center_y)
            )
            holes_xy = []

            def edge_idxs(nv):
                i = np.append(np.arange(nv), 0)
                return np.stack([i[:-1], i[1:]], axis=1)

            nv = 0
            verts, edges = [], []
            for loop in (outer_xy, *holes_xy):
                logger.debug(f"Loop: {loop}")
                verts.append(loop)
                edges.append(nv + edge_idxs(len(loop)))
                nv += len(loop)

            verts, edges = np.concatenate(verts), np.concatenate(edges)

            logger.debug(f"Verts: {verts}, Edges: {edges}")

            # Triangulate needs to know a single interior point for each hole
            # Using the centroid works here, but for very non-convex holes may need a more sophisticated method,
            # e.g. shapely's `polylabel`
            holes = np.array([np.mean(h, axis=0) for h in holes_xy])

            # Because triangulate is a wrapper around a C library the syntax is a little weird, 'p' here means planar straight line graph
            d = triangulate(dict(vertices=verts, segments=edges), opts="p")

            # Convert back to pyvista
            v, f = d["vertices"], d["triangles"]
            nv, nf = len(v), len(f)
            points = np.concatenate([v, np.zeros((nv, 1))], axis=1)

            logger.debug(f"points from triangulate: {points}")
            # print("faces from triangulate", faces)

            # Build Open3D TriangleMesh
            mesh_o3d = o3d.t.geometry.TriangleMesh()
            mesh_o3d.vertex.positions = o3d.core.Tensor(points)
            mesh_o3d.triangle.indices = o3d.core.Tensor(f)

            # logger.debug(f"mesh_o3d.get_center():{mesh_o3d.scale(1.2, mesh_o3d.get_center())}" )

            mesh_o3d.scale(ground_scale, mesh_o3d.get_center())
            o3d.t.io.write_triangle_mesh(
                os.path.join(mesh_data_dir, f"ground.ply"),
                mesh_o3d,
                write_ascii=write_ply_ascii,
            )

            sionna_shape = ET.SubElement(scene, "shape", type="ply", id=f"mesh-ground")
            if lidar_terrain:
                ET.SubElement(sionna_shape, "string", name="filename", value=f"mesh/lidar_terrain.ply")
            else:
                ET.SubElement(sionna_shape, "string", name="filename", value=f"mesh/ground.ply")
            bsdf_ref = ET.SubElement(
                sionna_shape, "ref", id=ground_material_type, name="bsdf"
            )
            ET.SubElement(sionna_shape, "boolean", name="face_normals", value="true")

            # ---------------------------------------------------------------------
            # 5) Query buildings within the bounding box
            # ---------------------------------------------------------------------

            # ground_polygon_4326_bbox => (west, south, east, north)
            west = ground_polygon_4326_bbox[0]  # minx
            south = ground_polygon_4326_bbox[1]  # miny
            east = ground_polygon_4326_bbox[2]  # maxx
            north = ground_polygon_4326_bbox[3]  # maxy
            # Calculate width/height in UTM
            width = math.ceil(ground_polygon_bbox[2] - ground_polygon_bbox[0])
            height = math.ceil(ground_polygon_bbox[3] - ground_polygon_bbox[1])
            logger.info(f"Estimated ground polygon size: width={width}m, height={height}m")

            ET.SubElement(scene, "default", name="scenegen_bbox_width", value=str(width))
            ET.SubElement(scene, "default", name="scenegen_bbox_length", value=str(height))

            # if width > 5000 or height > 5000:
            #     logger.warning(f"Too large!")
            #     exit(-1)

            try:
                buildings = load_overture_buildings_for_aoi(
                    ground_polygon_4326_bbox,
                    projection_UTM_EPSG_code,
                )
                filtered_buildings = buildings[buildings.intersects(ground_polygon)]
            except Exception as exc:
                logger.warning(
                    "Unable to load Overture building footprints; skipping buildings: %s",
                    exc,
                )
                buildings_list = []
            try:
                building_parts = load_overture_building_parts_for_aoi(
                    ground_polygon_4326_bbox,
                    projection_UTM_EPSG_code,
                )
                filtered_parts = building_parts[
                    building_parts.intersects(ground_polygon)
                ]
            except Exception as exc:
                logger.warning(
                    "Unable to load Overture building parts; using whole building footprints only: %s",
                    exc,
                )
                filtered_parts = None

            part_records = (
                filtered_parts.to_dict("records")
                if filtered_parts is not None
                else []
            )
            # parts where height and num_floors are both None
            parts_without_height_or_num_floors = []
            # parts where height is None but num_floors is not None
            parts_with_num_floors = []
            # parts where height is not None
            parts_with_height = []
            for part in part_records:
                if part.get("height") is not None and not math.isnan(part.get("height")):
                    parts_with_height.append(part)
                    continue
                if part.get("num_floors") is not None and not math.isnan(part.get("num_floors")):
                    parts_with_num_floors.append(part)
                    continue
                parts_without_height_or_num_floors.append(part)

            all_building_records = filtered_buildings.to_dict("records")
            parent_building_by_id = {
                str(building.get("id")): building                    
                for building in all_building_records
            }
            parts_by_parent_id = {}
            for part in part_records:
                if part.get("building_id") is None:
                    continue
                parts_by_parent_id.setdefault(str(part["building_id"]), []).append(part)

            building_records = [
                building
                for building in all_building_records
                if should_include_overture_parent_footprint(
                    building,
                    parts_by_parent_id.get(str(building.get("id")),[]),
                )
            ]
            parent_buildings_to_be_mapped = {building.get("id") for building in building_records}
            # buildings where height and num_floors are both None
            buildings_without_height_or_num_floors = []
            # buildings where height is None but num_floors is not None
            buildings_with_num_floors = []
            # buildings where height is not None
            buildings_with_height = []
            for building in building_records:
                if building.get("height") is not None and not math.isnan(building.get("height")):
                    buildings_with_height.append(building)
                    continue
                if building.get("num_floors") is not None and not math.isnan(building.get("num_floors")):
                    buildings_with_num_floors.append(building)
                    continue
                buildings_without_height_or_num_floors.append(building)

            buildings_list = [*buildings_without_height_or_num_floors,
                            *buildings_with_num_floors,
                            *buildings_with_height,
                            *parts_without_height_or_num_floors,
                            *parts_with_num_floors,
                            *parts_with_height
                            ]
            
            # for b in buildings_list:
            #     print(b.get("id"), b.get("overture_feature_type"), type(b.get("height")),b.get("height"), b.get("num_floors"))
            # print("end of printing out building list")
            # buildings_list.sort(key=resolve_building_base_height)
            logger.info(
                "Using %d Overture building footprints and %d building parts",
                len(building_records),
                len(part_records),
            )
            # Deprecated: Using OSM
            # else:
            #     # OSMnx features API uses bounding box in the form (north, south, east, west)
            #     logger.debug(
            #         f"OSM bounding box: (north={north}, south={south}, east={east}, west={west})"
            #     )
            #     # buildings are identified from OSM (look for bounding box and "building" tag)
            #     buildings = ox.features.features_from_bbox(
            #         bbox=ground_polygon_4326_bbox, tags={"building": True}
            #     )
            #     buildings = buildings.to_crs(projection_UTM_EPSG_code)

            #     # Filter out the building which outside the bounding box since
            #     # OSM will return some extra buildings.
            #     filtered_buildings = buildings[buildings.intersects(ground_polygon)]
            #     buildings_list = filtered_buildings.to_dict("records")

            # ---------------------------------------------------------------------
            # 6) If generating building map, prepare an empty grayscale image
            # ---------------------------------------------------------------------
            # Create a new empty Image, mode 'I' means 32bit integer grayscale image.
            self._building_map = Image.new("I", (width, height), 0)

            # ---------------------------------------------------------------------
            # 7) Init the building height handler. (osm or lidar)
            # ---------------------------------------------------------------------
            if lidar_height_calibration:
                try:
                    hag_handler = GeoTIFFHandler(hag_tiff_path)
                except Exception as e:
                    hag_handler = None
            else:
                hag_handler = None

            # ---------------------------------------------------------------------
            # 8) Process each building to create a 3D mesh (extrude by building height)
            # ---------------------------------------------------------------------

            # store centroid and building height information in a dataframe
            centroids_and_heights = []

            # convert each record to a Shapely footprint
            for idx, building in tqdm(
                enumerate(buildings_list),
                total=len(buildings_list),
                desc="Parsing buildings",
            ):
                # Debug the inner hole buildings
                # if building['type'] != "multipolygon":
                #     continue
                # Convert building geometry to a shapely polygon
                geometry = building["geometry"]
                if isinstance(geometry, BaseGeometry):
                    building_polygon = geometry
                else:
                    building_polygon = shape(geometry)

                if building_polygon.geom_type != "Polygon":
                    logger.debug(
                        f"building_polygon.geom_type: {building_polygon.geom_type}"
                    )
                    continue

                # Height prioritization hierarchy
                '''
                # First, try to get building height from LiDAR
                if hag_handler:
                    # sample 30 random points within the building footprint
                    random_points = generate_random_points(building_polygon, 30)
                    abs_height = []
                    for point in random_points:
                        res = hag_handler.query(to_4326.transform(point.x, point.y), False)
                        abs_height.append(res)

                    # plt.scatter([point.x for point in random_points ],[point.y for point in random_points ], c=abs_height, cmap='viridis')
                    # plt.colorbar(label='Height above ground (DSM - DEM) meters')

                    # plt.title('Random Points within a Building Polygon')
                    # plt.xlabel('Longitude EPSG:6933')
                    # plt.ylabel('Latitude EPSG:6933')
                    # plt.show()
                    print("Building height list: ", abs_height)
                    print()
                    filtered_list = [
                        x for x in abs_height if x.size > 0 and x != -9999 and x > 2
                    ]
                    print("Building height list: ", abs_height)
                    print()
                    try:
                        # use the mean of valid HAG samples
                        building_height = np.mean(filtered_list)
                        print("Avg Building Height: ", building_height)
                        if math.isnan(building_height):
                            raise ValueError("The value is NaN")
                    except Exception as e:
                        # fallback: random_building_height()
                        print("Random Building Height: ", building_height)
                        building_height = random_building_height(building, building_polygon)
                else:
                    # if no LiDAR, directly use random_building_height()
                    building_height = random_building_height(building, building_polygon)
                '''
                building_height = resolve_building_height(
                    building,
                    building_polygon,
                    hag_handler=hag_handler,
                    to_4326=to_4326,
                    lidar_height_calibration=lidar_height_calibration
                )
                building_base_height = (
                    resolve_building_base_height(building)
                )
                if (
                    building.get("overture_feature_type") == "building_part"
                    and building.get("building_id") is not None
                ):
                    parent_building = parent_building_by_id.get(
                        str(building["building_id"])
                    )
                    if parent_building is not None:
                        parent_height = resolve_building_height(
                            parent_building,
                            parent_building["geometry"],
                            hag_handler=hag_handler,
                            to_4326=to_4326,
                            lidar_height_calibration=lidar_height_calibration
                        )
                        parent_base_height = resolve_building_base_height(parent_building)
                        parent_top = parent_height + parent_base_height
                        part_top = building_base_height + building_height
                        if part_top > parent_top:
                            '''
                            logger.warning(
                                "Building part %s top height %.2f exceeds parent building %s top height %.2f; "
                                "treating min_height/min_floor as 0 for the part",
                                building.get("id"),
                                part_top,
                                parent_building.get("id"),
                                parent_top,
                            )
                            '''
                            building_base_height = 0.0

                # Skip buildings with height <= 0 m
                if building_height <= 0:
                    continue
                # building_height = NYC_LiDAR_building_height(building, building_polygon)

                outer_xy = unique_coords(
                    reorder_localize_coords(building_polygon.exterior, center_x, center_y)
                )
                
                # USE THE BELOW SECTION FOR BUILDING CENTROID AND HEIGHT INFO
                # WITH LIDAR TERRAIN TURNED OFF
                centroid = building_polygon.centroid
                centroid_lon, centroid_lat = to_4326.transform(
                    centroid.x, centroid.y
                )
                print(
                    f"building centroid lon={centroid_lon:.6f} lat={centroid_lat:.6f} height (from bottom to top)={building_height:.2f} base height={building_base_height:.2f} height above ground={building_height + building_base_height:.2f}"
                )
                centroids_and_heights.append(
                    {
                        "overture_building_or_part_id": building.get("id"),
                        "overture_parent_building_id" : building.get("building_id") if building.get("overture_feature_type") == "building_part" else "N/A",
                        "centroid_lon": centroid_lon,
                        "centroid_lat": centroid_lat,
                        "height": building_height,
                        "base_height": building_base_height,
                        "height_above_ground": building_height + building_base_height,
                    }
                )
                # USE THE ABOVE SECTION FOR BUILDING CENTROID AND HEIGHT INFO
                # WITH LIDAR TERRAIN TURNED OFF
                
                if lidar_terrain:
                    mesh = surface_mesh
                    # Z bounds of the mesh
                    bottom, top = mesh.bounds[-2:]
                    # buffer = 1.0
                    # res_z = []
                    # for points in outer_xy:
                    #     # Define two points that form a line that interesects the mesh
                    #     x = points[0]
                    #     y = points[1]
                    #     start = [x, y, bottom - buffer]
                    #     stop = [x, y, top + buffer]
                        
                    #     # Perform ray trace
                    #     points, ind = mesh.ray_trace(start, stop)
                        
                    #     # Create geometry to represent ray trace
                    #     ray = pv.Line(start, stop)
                    #     intersection = pv.PolyData(points)
                    #     res_z.append(intersection.bounds[-1])

                    # res_z = np.array(res_z)
                    # res_z[res_z == -1e+299] = 1e+299
                    # #print(res_z)
                    # building_z_value = int(np.floor(np.min(res_z)))

                    
                    # if building_z_value > 1e+20:
                    #     building_z_value = 0

                    from concurrent.futures import ThreadPoolExecutor

                    def ray_trace_z(x, y):
                        start = [x, y, bottom - 1.0]
                        stop = [x, y, top + 1.0]
                        points, ind = mesh.ray_trace(start, stop)
                        if points.shape[0] == 0:
                            return 1e+299
                        return np.max(points[:, 2])

                    with ThreadPoolExecutor() as executor:
                        res_z_parallel = list(executor.map(lambda pt: ray_trace_z(pt[0], pt[1]), outer_xy))
                    building_z_value = int(np.floor(np.min(res_z_parallel)))
                    #building_z_value = int(np.floor(np.min(res_z)))

                    #assert building_z_value_parallel == building_z_value, f"differ value for parallel and single-thread:\n\t parallel {building_z_value_parallel} single-threaded {building_z_value}"
                    if building_z_value > 1e+20:
                        building_z_value = 0
                else:
                    building_z_value = 0
                building_base_z_value = building_z_value + building_base_height
                    
            
                #print("Building's Z-value: ", building_z_value)
                
                

                holes_xy = []
                if len(list(building_polygon.interiors)) != 0:
                    for inner_hole in list(building_polygon.interiors):
                        valid_coords = reorder_localize_coords(
                            inner_hole, center_x, center_y
                        )
                        holes_xy.append(unique_coords(valid_coords))

                def edge_idxs(nv):
                    i = np.append(np.arange(nv), 0)
                    return np.stack([i[:-1], i[1:]], axis=1)

                nv = 0
                verts, edges = [], []
                for loop in (outer_xy, *holes_xy):
                    verts.append(loop)
                    edges.append(nv + edge_idxs(len(loop)))
                    nv += len(loop)

                verts, edges = np.concatenate(verts), np.concatenate(edges)

                # Triangulate needs to know a single interior point for each hole
                # Using the centroid works here, but for very non-convex holes may need a more sophisticated method,
                # e.g. shapely's `polylabel`
                holes = np.array([np.mean(h, axis=0) for h in holes_xy])

                # Because triangulate is a wrapper around a C library the syntax is a little weird, 'p' here means planar straight line graph
                if len(holes) != 0:
                    d = triangulate(
                        dict(vertices=verts, segments=edges, holes=holes), opts="p"
                    )
                else:
                    d = triangulate(dict(vertices=verts, segments=edges), opts="p")

                # Convert back to pyvista
                v, f = d["vertices"], d["triangles"]
                nv, nf = len(v), len(f)

                # print(v)
                # print(f)

                #points = np.concatenate([v, np.zeros((nv, 1))], axis=1)
                points = np.concatenate([v, np.full((nv, 1), fill_value=building_base_z_value)], axis=1)
        
                mesh_o3d = o3d.t.geometry.TriangleMesh()
                mesh_o3d.vertex.positions = o3d.core.Tensor(points)
                mesh_o3d.triangle.indices = o3d.core.Tensor(f)

                wedge_t = mesh_o3d.extrude_linear([0, 0, building_height])
                # Get vertices and faces
                vertices_tensor = wedge_t.vertex["positions"]
                faces_tensor = wedge_t.triangle["indices"]

                # Convert to NumPy for calculations
                vertices_np = vertices_tensor.numpy()
                faces_np = faces_tensor.numpy()

                # Compute face centroids
                face_centroids = np.mean(vertices_np[faces_np], axis=1)

                z_values = vertices_np[:, 2]
                building_top_z_value = building_height + building_base_z_value
                top_vertex_indices = np.where(np.isclose(z_values, building_top_z_value))[
                    0
                ].tolist()  # Indices of top vertices
                #print("top vertex indices: ", top_vertex_indices)
                



                # Extract the top surface
                top_surface = wedge_t.select_by_index(top_vertex_indices)

                other_faces_np = faces_np[face_centroids[:, 2] < building_top_z_value]
                if len(other_faces_np) == 0:
                    print("All vertices: ", vertices_np)
                    print("top vertex indices: ", top_vertex_indices)
                    print("max height of meshes: ", np.max(z_values))
                    print("min height of meshes: ", np.min(z_values))
                    print("building height: ", building_height)
                    print("building z value: ", building_z_value)
                    print("building base height: ", building_base_height)
                    print("building top z value: ", building_top_z_value)
                    
                
                    print("other faces np: ", other_faces_np)
                    print("max height of meshes: ", np.max(z_values))
                    print("building height: ", building_height)
                    print("building z value: ", building_z_value)
                    print("building base height: ", building_base_height)
                    print("building top z value: ", building_top_z_value)
                # Convert to Open3D Tensor API
                other_faces_o3c = o3c.Tensor(other_faces_np, dtype=o3c.int32)

                wall_mesh = o3d.t.geometry.TriangleMesh()
                wall_mesh.vertex["positions"] = vertices_tensor  # Same vertices
                wall_mesh.triangle["indices"] = other_faces_o3c

                wall_mesh.remove_unreferenced_vertices()

                o3d.t.io.write_triangle_mesh(
                    os.path.join(mesh_data_dir, f"building_{idx}_rooftop.ply"),
                    top_surface,
                    write_ascii=write_ply_ascii,
                )
                o3d.t.io.write_triangle_mesh(
                    os.path.join(mesh_data_dir, f"building_{idx}_wall.ply"),
                    wall_mesh,
                    write_ascii=write_ply_ascii,
                )

                # o3d.t.io.write_triangle_mesh(os.path.join(mesh_data_dir, f"building_{idx}.ply"), wedge, write_ascii=write_ply_ascii)

                # Add shape elements for PLY files in the folder
                sionna_shape = ET.SubElement(
                    scene, "shape", type="ply", id=f"mesh-building_{idx}_rooftop"
                )
                ET.SubElement(
                    sionna_shape,
                    "string",
                    name="filename",
                    value=f"mesh/building_{idx}_rooftop.ply",
                )
                ET.SubElement(sionna_shape, "ref", id=rooftop_material_type, name="bsdf")
                ET.SubElement(sionna_shape, "boolean", name="face_normals", value="true")

                sionna_shape = ET.SubElement(
                    scene, "shape", type="ply", id=f"mesh-building_{idx}_wall"
                )
                ET.SubElement(
                    sionna_shape,
                    "string",
                    name="filename",
                    value=f"mesh/building_{idx}_wall.ply",
                )
                ET.SubElement(sionna_shape, "ref", id=wall_material_type, name="bsdf")
                ET.SubElement(sionna_shape, "boolean", name="face_normals", value="true")

                if generate_building_map:
                    self._draw_building(
                        building_polygon,
                        building_height + building_base_height,
                    )

            del hag_handler

            # ---------------------------------------------------------------------
            # 9) Query Overture roads within the bounding box and mesh them flat
            # ---------------------------------------------------------------------
            if generate_roads:
                try:
                    roads = load_overture_roads_for_aoi(
                        ground_polygon_4326_bbox,
                        projection_UTM_EPSG_code,
                    )
                    filtered_roads = roads[roads.intersects(ground_polygon)]
                    road_records = filtered_roads.to_dict("records")
                except Exception as exc:
                    logger.warning(
                        "Unable to load Overture road segments; skipping roads: %s",
                        exc,
                    )
                    road_records = []

                logger.info("Using %d Overture road segments", len(road_records))

                for idx, road in tqdm(
                    enumerate(road_records),
                    total=len(road_records),
                    desc="Parsing roads",
                ):
                    centerline = road["geometry"]
                    if not isinstance(centerline, BaseGeometry):
                        centerline = shape(centerline)
                    if centerline.geom_type != "LineString":
                        continue

                    road_width = resolve_road_width(road)
                    road_polygon = build_road_polygon(centerline, road_width)
                    if road_polygon is None or road_polygon.is_empty:
                        continue

                    # Clip to the AOI so buffered segment ends near the boundary
                    # don't extend past it.
                    road_polygon = road_polygon.intersection(ground_polygon)
                    if road_polygon.geom_type == "MultiPolygon":
                        road_polygon = max(road_polygon.geoms, key=lambda geom: geom.area)
                    if road_polygon.is_empty or road_polygon.geom_type != "Polygon":
                        continue

                    outer_xy = unique_coords(
                        reorder_localize_coords(road_polygon.exterior, center_x, center_y)
                    )
                    if len(outer_xy) < 3:
                        continue

                    holes_xy = []
                    for inner_hole in list(road_polygon.interiors):
                        valid_coords = reorder_localize_coords(
                            inner_hole, center_x, center_y
                        )
                        holes_xy.append(unique_coords(valid_coords))

                    def edge_idxs(nv):
                        i = np.append(np.arange(nv), 0)
                        return np.stack([i[:-1], i[1:]], axis=1)

                    nv = 0
                    verts, edges = [], []
                    for loop in (outer_xy, *holes_xy):
                        verts.append(loop)
                        edges.append(nv + edge_idxs(len(loop)))
                        nv += len(loop)
                    verts, edges = np.concatenate(verts), np.concatenate(edges)

                    holes = np.array([np.mean(h, axis=0) for h in holes_xy])
                    if len(holes) != 0:
                        d = triangulate(
                            dict(vertices=verts, segments=edges, holes=holes), opts="p"
                        )
                    else:
                        d = triangulate(dict(vertices=verts, segments=edges), opts="p")

                    v, f = d["vertices"], d["triangles"]
                    if len(f) == 0:
                        continue
                    nv = len(v)

                    if lidar_terrain:
                        mesh = surface_mesh
                        bottom, top = mesh.bounds[-2:]

                        def ray_trace_z(x, y):
                            start = [x, y, bottom - 1.0]
                            stop = [x, y, top + 1.0]
                            trace_points, _ = mesh.ray_trace(start, stop)
                            if trace_points.shape[0] == 0:
                                return 1e299
                            return np.max(trace_points[:, 2])

                        from concurrent.futures import ThreadPoolExecutor

                        with ThreadPoolExecutor() as executor:
                            res_z_parallel = list(
                                executor.map(lambda pt: ray_trace_z(pt[0], pt[1]), outer_xy)
                            )
                        road_z_value = int(np.floor(np.min(res_z_parallel)))
                        if road_z_value > 1e20:
                            road_z_value = 0
                    else:
                        road_z_value = 0

                    # Roads are perfectly flat and coplanar with the ground/terrain
                    # mesh; lift them slightly to avoid z-fighting in the renderer.
                    road_z_value += ROAD_Z_OFFSET_M

                    points = np.concatenate(
                        [v, np.full((nv, 1), fill_value=road_z_value)], axis=1
                    )

                    mesh_o3d = o3d.t.geometry.TriangleMesh()
                    mesh_o3d.vertex.positions = o3d.core.Tensor(points)
                    mesh_o3d.triangle.indices = o3d.core.Tensor(f)

                    o3d.t.io.write_triangle_mesh(
                        os.path.join(mesh_data_dir, f"road_{idx}.ply"),
                        mesh_o3d,
                        write_ascii=write_ply_ascii,
                    )

                    sionna_shape = ET.SubElement(
                        scene, "shape", type="ply", id=f"mesh-road_{idx}"
                    )
                    ET.SubElement(
                        sionna_shape,
                        "string",
                        name="filename",
                        value=f"mesh/road_{idx}.ply",
                    )
                    ET.SubElement(sionna_shape, "ref", id=resolved_road_material_type, name="bsdf")
                    ET.SubElement(sionna_shape, "boolean", name="face_normals", value="true")

            # Save the centroids and heights to a CSV file
            centroids_and_heights_df = pd.DataFrame(centroids_and_heights)
            centroids_and_heights_df.to_csv(
                os.path.join(data_dir, "building_centroids_and_heights.csv"), index=False
            )
            xml_string = ET.tostring(scene, encoding="utf-8")
            xml_pretty = minidom.parseString(xml_string).toprettyxml(
                indent="    "
            )  # Adjust the infdent as needed

            with open(
                os.path.join(data_dir, "scene.xml"), "w", encoding="utf-8"
            ) as xml_file:
                xml_file.write(xml_pretty)

            if generate_building_map:
                np.save(
                    os.path.join(data_dir, "2D_Building_Height_Map.npy"),
                    np.array(self._building_map),
                )

            return np.array(self._building_map)
        # osm data source
        else:
            if ground_material_type not in ITU_MATERIALS:
                raise ValueError(f"Invalid ground material type: {ground_material_type}")
            if rooftop_material_type not in ITU_MATERIALS:
                raise ValueError(f"Invalid rooftop material type: {rooftop_material_type}")
            if wall_material_type not in ITU_MATERIALS:
                raise ValueError(f"Invalid wall material type: {wall_material_type}")
            
            # ---------------------------------------------------------------------
            # 1) Setup OSM server and transforms
            # ---------------------------------------------------------------------
            if osm_server_addr:
                ox.settings.overpass_url = osm_server_addr
                ox.settings.overpass_rate_limit = False
            ox.settings.use_cache = False
            # Determine the UTM projection from the first point
            projection_UTM_EPSG_code = get_utm_epsg_code_from_gps(
                points[0][0], points[0][1]
            )
            logger.info(f"Using UTM Zone: {projection_UTM_EPSG_code}")

            # Create transformations between WGS84 (EPSG:4326) and UTM
            to_projection = Transformer.from_crs(
                "EPSG:4326", projection_UTM_EPSG_code, always_xy=True
            )
            to_4326 = Transformer.from_crs(
                projection_UTM_EPSG_code, "EPSG:4326", always_xy=True
            )

            # ---------------------------------------------------------------------
            # 2) Prepare output directories and camera / material settings
            # ---------------------------------------------------------------------
            mesh_data_dir = os.path.join(data_dir, "mesh")
            os.makedirs(os.path.join(mesh_data_dir), exist_ok=True)

            def print_material_info(surface_name, material_type):
                if isinstance(ITU_MATERIALS[material_type]["lower_freq_limit"], list):
                    logger.info(
                        "{:<35}{:<20} | Frequency Range: {:^5} - {:^5} (GHz) | {:^5} - {:^5} (GHz)".format(
                            "{} Material Type:".format(surface_name),
                            ITU_MATERIALS[material_type]["name"],
                            print_if_int(
                                ITU_MATERIALS[material_type]["lower_freq_limit"][0] / 1e9
                            ),
                            print_if_int(
                                ITU_MATERIALS[material_type]["upper_freq_limit"][0] / 1e9
                            ),
                            print_if_int(
                                ITU_MATERIALS[material_type]["lower_freq_limit"][1] / 1e9
                            ),
                            print_if_int(
                                ITU_MATERIALS[material_type]["upper_freq_limit"][1] / 1e9
                            ),
                        )
                    )
                else:
                    logger.info(
                        "{:<35}{:<20} | Frequency Range: {:^5} - {:^5} (GHz)".format(
                            "{} Material Type:".format(surface_name),
                            ITU_MATERIALS[material_type]["name"],
                            print_if_int(
                                ITU_MATERIALS[material_type]["lower_freq_limit"] / 1e9
                            ),
                            print_if_int(
                                ITU_MATERIALS[material_type]["upper_freq_limit"] / 1e9
                            ),
                        )
                    )

            logger.info("")
            print_material_info("Ground", ground_material_type)
            print_material_info("Building Rooftop", rooftop_material_type)
            print_material_info("Building Wall", wall_material_type)
            logger.info("")



            camera_settings = {
                "rotation": (0, 0, -90),  # Assuming Z-up orientation
                "fov": 42.854885,
            }

            # ---------------------------------------------------------------------
            # 3) Build the XML scene root
            # ---------------------------------------------------------------------


            # Default Mitsuba rendering parameters
            spp_default = 4096
            resx_default = 1024
            resy_default = 1024

            scene = ET.Element("scene", version="2.1.0")
            # Default integrator / film settings
            ET.SubElement(scene, "default", name="spp", value=str(spp_default))
            ET.SubElement(scene, "default", name="resx", value=str(resx_default))
            ET.SubElement(scene, "default", name="resy", value=str(resy_default))

            ET.SubElement(scene, "default", name="scenegen_version", value=str(get_package_version()))
            ET.SubElement(scene, "default", name="scenegen_create_time", value=str(datetime.datetime.now()))

            ET.SubElement(scene, "default", name="scenegen_min_lat", value=str(points[0][1]))
            ET.SubElement(scene, "default", name="scenegen_max_lat", value=str(points[1][1]))
            ET.SubElement(scene, "default", name="scenegen_min_lon", value=str(points[0][0]))
            ET.SubElement(scene, "default", name="scenegen_max_lon", value=str(points[2][0]))
            


            ET.SubElement(scene, "default", name="scenegen_ground_material", value=str(ground_material_type))
            ET.SubElement(scene, "default", name="scenegen_rooftop_material", value=str(rooftop_material_type))
            ET.SubElement(scene, "default", name="scenegen_wall_material", value=str(wall_material_type))

            ET.SubElement(scene, "default", name="scenegen_UTM_zone", value=str(projection_UTM_EPSG_code))
            
        

            integrator = ET.SubElement(scene, "integrator", type="path")
            ET.SubElement(integrator, "integer", name="max_depth", value="12")

            # Define materials
            for material_id, material_content in ITU_MATERIALS.items():
                
                # Temporary workaround for Sionna v1.1 : Skip vacuum and P.527 materials.
                if "vacuum" in material_id in material_id:
                    continue

                if "P.527" not in material_id:
                    bsdf_twosided = ET.SubElement(
                        scene, "bsdf", type="twosided", id=material_id
                    )
                    bsdf_diffuse = ET.SubElement(bsdf_twosided, "bsdf", type="diffuse")
                    rgb = material_content["mitsuba_color"]
                    ET.SubElement(
                        bsdf_diffuse,
                        "rgb",
                        value=f"{rgb[0]} {rgb[1]} {rgb[2]}",
                        name="reflectance",
                    )
                else:
                    bsdf_twosided = ET.SubElement(
                        scene, "bsdf", type="radio-material", id=material_id
                    )
                    

            # Add emitter (constant environment light)
            emitter = ET.SubElement(scene, "emitter", type="constant", id="World")
            ET.SubElement(
                emitter, "rgb", value="1.000000 1.000000 1.000000", name="radiance"
            )

            # Add camera (sensor)
            sensor = ET.SubElement(scene, "sensor", type="perspective", id="Camera")
            ET.SubElement(sensor, "string", name="fov_axis", value="x")
            ET.SubElement(sensor, "float", name="fov", value=str(camera_settings["fov"]))
            ET.SubElement(
                sensor, "float", name="principal_point_offset_x", value="0.000000"
            )
            ET.SubElement(
                sensor, "float", name="principal_point_offset_y", value="-0.000000"
            )
            ET.SubElement(sensor, "float", name="near_clip", value="0.100000")
            ET.SubElement(sensor, "float", name="far_clip", value="10000.000000")
            sionna_transform = ET.SubElement(sensor, "transform", name="to_world")
            ET.SubElement(
                sionna_transform, "rotate", x="1", angle=str(camera_settings["rotation"][0])
            )
            ET.SubElement(
                sionna_transform, "rotate", y="1", angle=str(camera_settings["rotation"][1])
            )
            ET.SubElement(
                sionna_transform, "rotate", z="1", angle=str(camera_settings["rotation"][2])
            )
            camera_position = np.array([0, 0, 100])  # Adjust camera height
            ET.SubElement(
                sionna_transform, "translate", value=" ".join(map(str, camera_position))
            )
            sampler = ET.SubElement(sensor, "sampler", type="independent")
            ET.SubElement(sampler, "integer", name="sample_count", value="$spp")
            film = ET.SubElement(sensor, "film", type="hdrfilm")
            ET.SubElement(film, "integer", name="width", value="$resx")
            ET.SubElement(film, "integer", name="height", value="$resy")

            # ---------------------------------------------------------------------
            # 4) Create ground polygon (in UTM) and ground mesh
            # ---------------------------------------------------------------------

            # # Define the points in counter-clockwise order to create the polygon
            # points = [top_left, top_right, bottom_right, bottom_left]
            ground_polygon_4326 = shapely.geometry.Polygon(points)
            ground_polygon_4326_bbox = ground_polygon_4326.bounds

            # Transform each WGS84 coordinate into UTM
            coords = [to_projection.transform(x, y) for x, y in points]
            ground_polygon = shapely.geometry.Polygon(coords)
            ground_polygon_bbox = ground_polygon.bounds

            self._ground_polygon_envelope_UTM = ground_polygon.envelope

            center_x = ground_polygon.envelope.centroid.x
            center_y = ground_polygon.envelope.centroid.y

            ET.SubElement(scene, "default", name="scenegen_center_lat", value=f"{ground_polygon_4326.envelope.centroid.y:.6f}")
            ET.SubElement(scene, "default", name="scenegen_center_lon", value=f"{ground_polygon_4326.envelope.centroid.x:.6f}")


            # ---------------------------------------------------------------------
            # 0) Query USGS 3DEP LiDAR data and generate GEOTIFF file for building height calibration
            # ---------------------------------------------------------------------
            try:
                laz_file_path = Path(os.path.join(data_dir, "test_hag.laz"))
                tif_file_path = Path(os.path.join(data_dir, "test_hag.tif"))
                if lidar_terrain or lidar_height_calibration:
                    if not laz_file_path.exists() or not tif_file_path.exists():
                        
                        from .USGS_LiDAR_HAG import generate_hag
                        
                        generate_hag(affinity.scale(ground_polygon_4326, xfact=ground_scale, yfact=ground_scale, origin='centroid'), data_dir, projection_UTM_EPSG_code)
                    
        
                

        
                if lidar_terrain:
                    from .lidar_terrain_mesh import generate_terrain_mesh
        
        
        
                    assert laz_file_path.exists(), f"LAZ file does not exist: {laz_file_path}"
        
                    assert tif_file_path.exists(), f"TIF file does not exist: {tif_file_path}"
                    print("Skip the lidar_terrain.ply")
                    if not Path(os.path.join(data_dir,"mesh" ,"lidar_terrain.ply")).exists():

                        if dem_terrain:
                            generate_terrain_mesh_dem(
                                affinity.scale(ground_polygon_4326, xfact=ground_scale, yfact=ground_scale, origin='centroid'),
                                os.path.join(mesh_data_dir, f"lidar_terrain.ply")
                            )
                        else:
                            
                            generate_terrain_mesh(os.path.join(data_dir, "test_hag.laz"),
                                os.path.join(mesh_data_dir, f"lidar_terrain.ply"), src_crs=projection_UTM_EPSG_code, dest_crs=projection_UTM_EPSG_code,
                                plot_figures=False, center_x=center_x, center_y=center_y
                            )
                if gen_lidar_terrain_only:
                    print("gen_lidar_terrain_only: True")
                    return
            except Exception as e:
                print(e)
            if lidar_terrain:
                lidar_terrain_ply_path = Path(os.path.join(data_dir,"mesh" ,"lidar_terrain.ply"))
                if not lidar_terrain_ply_path.exists():
                    return 1
                surface_mesh = pv.read(lidar_terrain_ply_path)
            #######Open3D#######
            outer_xy = unique_coords(
                reorder_localize_coords(ground_polygon.exterior, center_x, center_y)
            )
            holes_xy = []

            def edge_idxs(nv):
                i = np.append(np.arange(nv), 0)
                return np.stack([i[:-1], i[1:]], axis=1)

            nv = 0
            verts, edges = [], []
            for loop in (outer_xy, *holes_xy):
                logger.debug(f"Loop: {loop}")
                verts.append(loop)
                edges.append(nv + edge_idxs(len(loop)))
                nv += len(loop)

            verts, edges = np.concatenate(verts), np.concatenate(edges)

            logger.debug(f"Verts: {verts}, Edges: {edges}")

            # Triangulate needs to know a single interior point for each hole
            # Using the centroid works here, but for very non-convex holes may need a more sophisticated method,
            # e.g. shapely's `polylabel`
            holes = np.array([np.mean(h, axis=0) for h in holes_xy])

            # Because triangulate is a wrapper around a C library the syntax is a little weird, 'p' here means planar straight line graph
            d = triangulate(dict(vertices=verts, segments=edges), opts="p")

            # Convert back to pyvista
            v, f = d["vertices"], d["triangles"]
            nv, nf = len(v), len(f)
            points = np.concatenate([v, np.zeros((nv, 1))], axis=1)

            logger.debug(f"points from triangulate: {points}")
            # print("faces from triangulate", faces)

            # Build Open3D TriangleMesh
            mesh_o3d = o3d.t.geometry.TriangleMesh()
            mesh_o3d.vertex.positions = o3d.core.Tensor(points)
            mesh_o3d.triangle.indices = o3d.core.Tensor(f)

            # logger.debug(f"mesh_o3d.get_center():{mesh_o3d.scale(1.2, mesh_o3d.get_center())}" )

            mesh_o3d.scale(ground_scale, mesh_o3d.get_center())
            o3d.t.io.write_triangle_mesh(
                os.path.join(mesh_data_dir, f"ground.ply"),
                mesh_o3d,
                write_ascii=write_ply_ascii,
            )

            sionna_shape = ET.SubElement(scene, "shape", type="ply", id=f"mesh-ground")
            if lidar_terrain:
                ET.SubElement(sionna_shape, "string", name="filename", value=f"mesh/lidar_terrain.ply")
            else:
                ET.SubElement(sionna_shape, "string", name="filename", value=f"mesh/ground.ply")
            bsdf_ref = ET.SubElement(
                sionna_shape, "ref", id=ground_material_type, name="bsdf"
            )
            ET.SubElement(sionna_shape, "boolean", name="face_normals", value="true")

            # ---------------------------------------------------------------------
            # 5) Query OSM for buildings within the bounding box
            # ---------------------------------------------------------------------

            # ground_polygon_4326_bbox => (west, south, east, north)
            west = ground_polygon_4326_bbox[0]  # minx
            south = ground_polygon_4326_bbox[1]  # miny
            east = ground_polygon_4326_bbox[2]  # maxx
            north = ground_polygon_4326_bbox[3]  # maxy
            # Calculate width/height in UTM
            width = math.ceil(ground_polygon_bbox[2] - ground_polygon_bbox[0])
            height = math.ceil(ground_polygon_bbox[3] - ground_polygon_bbox[1])
            logger.info(f"Estimated ground polygon size: width={width}m, height={height}m")

            ET.SubElement(scene, "default", name="scenegen_bbox_width", value=str(width))
            ET.SubElement(scene, "default", name="scenegen_bbox_length", value=str(height))

            # if width > 5000 or height > 5000:
            #     logger.warning(f"Too large!")
            #     exit(-1)

            # OSMnx features API uses bounding box in the form (north, south, east, west)
            logger.debug(
                f"OSM bounding box: (north={north}, south={south}, east={east}, west={west})"
            )
            buildings = ox.features.features_from_bbox(
                bbox=ground_polygon_4326_bbox, tags={"building": True}
            )
            buildings = buildings.to_crs(projection_UTM_EPSG_code)

            # Filter out the building which outside the bounding box since
            # OSM will return some extra buildings.
            filtered_buildings = buildings[buildings.intersects(ground_polygon)]
            buildings_list = filtered_buildings.to_dict("records")

            # ---------------------------------------------------------------------
            # 6) If generating building map, prepare an empty grayscale image
            # ---------------------------------------------------------------------
            # Create a new empty Image, mode 'I' means 32bit grayscale image.
            self._building_map = Image.new("I", (width, height), 0)

            # ---------------------------------------------------------------------
            # 7) Init the building height handler. (osm or lidar)
            # ---------------------------------------------------------------------
            if lidar_height_calibration:
                try:
                    hag_handler = GeoTIFFHandler(hag_tiff_path)
                except Exception as e:
                    hag_handler = None
            else:
                hag_handler = None

            # ---------------------------------------------------------------------
            # 8) Process each building to create a 3D mesh (extrude by building height)
            # ---------------------------------------------------------------------

            centroids_and_heights = []

            for idx, building in tqdm(
                enumerate(buildings_list),
                total=len(buildings_list),
                desc="Parsing buildings",
            ):
                # Debug the inner hole buildings
                # if building['type'] != "multipolygon":
                #     continue
                # Convert building geometry to a shapely polygon
                building_polygon = shape(building["geometry"])

                if building_polygon.geom_type != "Polygon":
                    logger.debug(
                        f"building_polygon.geom_type: {building_polygon.geom_type}"
                    )
                    continue

                # First try to get building height from LiDAR
                if hag_handler:
                    random_points = generate_random_points(building_polygon, 30)
                    abs_height = []
                    for point in random_points:
                        res = hag_handler.query(to_4326.transform(point.x, point.y), False)
                        abs_height.append(res)

                    # plt.scatter([point.x for point in random_points ],[point.y for point in random_points ], c=abs_height, cmap='viridis')
                    # plt.colorbar(label='Height above ground (DSM - DEM) meters')

                    # plt.title('Random Points within a Building Polygon')
                    # plt.xlabel('Longitude EPSG:6933')
                    # plt.ylabel('Latitude EPSG:6933')
                    # plt.show()
                    print("Building height list: ", abs_height)
                    print()
                    filtered_list = [
                        x for x in abs_height if x.size > 0 and x != -9999 and x > 2
                    ]
                    print("Building height list: ", abs_height)
                    print()
                    try:
                        building_height = np.mean(filtered_list)
                        print("Avg Building Height: ", building_height)
                        if math.isnan(building_height):
                            raise ValueError("The value is NaN")
                    except Exception as e:
                        print("Random Building Height: ", building_height)
                        building_height = random_building_height(building, building_polygon)
                else:
                    building_height = random_building_height(building, building_polygon)

                # USE THE BELOW SECTION FOR BUILDING CENTROID AND HEIGHT INFO
                # WITH LIDAR TERRAIN TURNED OFF

                centroid = building_polygon.centroid
                centroid_lon, centroid_lat = to_4326.transform(
                    centroid.x, centroid.y
                )
                print(
                    f"Building centroid lon={centroid_lon:.6f} lat={centroid_lat:.6f} height={building_height:.2f} base height={0.0} height above ground={building_height:.2f}"
                )
                centroids_and_heights.append(
                    {
                        "centroid_lon": centroid_lon,
                        "centroid_lat": centroid_lat,
                        "height": building_height,
                        "base_height": 0.0,
                        "height_above_ground": building_height,
                    }
                )
                # USE THE ABOVE SECTION FOR BUILDING CENTROID AND HEIGHT INFO
                # WITH LIDAR TERRAIN TURNED OFF
                
                # Skip buildings with height <= 0
                if building_height <=0:
                    continue
                # building_height = NYC_LiDAR_building_height(building, building_polygon)

                outer_xy = unique_coords(
                    reorder_localize_coords(building_polygon.exterior, center_x, center_y)
                )
                
                
                if lidar_terrain:
                    mesh = surface_mesh
                    # Z bounds of the mesh
                    bottom, top = mesh.bounds[-2:]
                    # buffer = 1.0
                    # res_z = []
                    # for points in outer_xy:
                    #     # Define two points that form a line that interesects the mesh
                    #     x = points[0]
                    #     y = points[1]
                    #     start = [x, y, bottom - buffer]
                    #     stop = [x, y, top + buffer]
                        
                    #     # Perform ray trace
                    #     points, ind = mesh.ray_trace(start, stop)
                        
                    #     # Create geometry to represent ray trace
                    #     ray = pv.Line(start, stop)
                    #     intersection = pv.PolyData(points)
                    #     res_z.append(intersection.bounds[-1])

                    # res_z = np.array(res_z)
                    # res_z[res_z == -1e+299] = 1e+299
                    # #print(res_z)
                    # building_z_value = int(np.floor(np.min(res_z)))

                    
                    # if building_z_value > 1e+20:
                    #     building_z_value = 0

                    from concurrent.futures import ThreadPoolExecutor

                    def ray_trace_z(x, y):
                        start = [x, y, bottom - 1.0]
                        stop = [x, y, top + 1.0]
                        points, ind = mesh.ray_trace(start, stop)
                        if points.shape[0] == 0:
                            return 1e+299
                        return np.max(points[:, 2])

                    with ThreadPoolExecutor() as executor:
                        res_z_parallel = list(executor.map(lambda pt: ray_trace_z(pt[0], pt[1]), outer_xy))
                    building_z_value = int(np.floor(np.min(res_z_parallel)))
                    #building_z_value = int(np.floor(np.min(res_z)))

                    #assert building_z_value_parallel == building_z_value, f"differ value for parallel and single-thread:\n\t parallel {building_z_value_parallel} single-threaded {building_z_value}"
                    if building_z_value > 1e+20:
                        building_z_value = 0
                else:
                    building_z_value = 0
                    
            
                #print("Building's Z-value: ", building_z_value)
                
                

                holes_xy = []
                if len(list(building_polygon.interiors)) != 0:
                    for inner_hole in list(building_polygon.interiors):
                        valid_coords = reorder_localize_coords(
                            inner_hole, center_x, center_y
                        )
                        holes_xy.append(unique_coords(valid_coords))

                def edge_idxs(nv):
                    i = np.append(np.arange(nv), 0)
                    return np.stack([i[:-1], i[1:]], axis=1)

                nv = 0
                verts, edges = [], []
                for loop in (outer_xy, *holes_xy):
                    verts.append(loop)
                    edges.append(nv + edge_idxs(len(loop)))
                    nv += len(loop)

                verts, edges = np.concatenate(verts), np.concatenate(edges)

                # Triangulate needs to know a single interior point for each hole
                # Using the centroid works here, but for very non-convex holes may need a more sophisticated method,
                # e.g. shapely's `polylabel`
                holes = np.array([np.mean(h, axis=0) for h in holes_xy])

                # Because triangulate is a wrapper around a C library the syntax is a little weird, 'p' here means planar straight line graph
                if len(holes) != 0:
                    d = triangulate(
                        dict(vertices=verts, segments=edges, holes=holes), opts="p"
                    )
                else:
                    d = triangulate(dict(vertices=verts, segments=edges), opts="p")

                # Convert back to pyvista
                v, f = d["vertices"], d["triangles"]
                nv, nf = len(v), len(f)

                # print(v)
                # print(f)

                #points = np.concatenate([v, np.zeros((nv, 1))], axis=1)
                points = np.concatenate([v, np.full((nv, 1), fill_value=building_z_value)], axis=1)
        
                mesh_o3d = o3d.t.geometry.TriangleMesh()
                mesh_o3d.vertex.positions = o3d.core.Tensor(points)
                mesh_o3d.triangle.indices = o3d.core.Tensor(f)

                wedge_t = mesh_o3d.extrude_linear([0, 0, building_height])
                # Get vertices and faces
                vertices_tensor = wedge_t.vertex["positions"]
                faces_tensor = wedge_t.triangle["indices"]

                # Convert to NumPy for calculations
                vertices_np = vertices_tensor.numpy()
                faces_np = faces_tensor.numpy()

                # Compute face centroids
                face_centroids = np.mean(vertices_np[faces_np], axis=1)

                z_values = vertices_np[:, 2]
                top_vertex_indices = np.where(z_values == building_height + building_z_value)[
                    0
                ].tolist()  # Indices of top vertices
                #print("top vertex indices: ", top_vertex_indices)
                



                # Extract the top surface
                top_surface = wedge_t.select_by_index(top_vertex_indices)

                other_faces_np = faces_np[face_centroids[:, 2] < building_height+building_z_value]
                if len(other_faces_np) == 0:
                    print("All vertices: ", vertices_np)
                    print("top vertex indices: ", top_vertex_indices)
                    print("max height of meshes: ", np.max(z_values))
                    print("min height of meshes: ", np.min(z_values))
                    print("building height: ", building_height)
                    print("building z value: ", building_z_value)
                    print("building height + building z value: ", building_height + building_z_value)
                    
                
                    print("other faces np: ", other_faces_np)
                    print("max height of meshes: ", np.max(z_values))
                    print("building height: ", building_height)
                    print("building z value: ", building_z_value)
                    print("building height + building z value: ", building_height + building_z_value)
                # Convert to Open3D Tensor API
                other_faces_o3c = o3c.Tensor(other_faces_np, dtype=o3c.int32)

                wall_mesh = o3d.t.geometry.TriangleMesh()
                wall_mesh.vertex["positions"] = vertices_tensor  # Same vertices
                wall_mesh.triangle["indices"] = other_faces_o3c

                wall_mesh.remove_unreferenced_vertices()

                o3d.t.io.write_triangle_mesh(
                    os.path.join(mesh_data_dir, f"building_{idx}_rooftop.ply"),
                    top_surface,
                    write_ascii=write_ply_ascii,
                )
                o3d.t.io.write_triangle_mesh(
                    os.path.join(mesh_data_dir, f"building_{idx}_wall.ply"),
                    wall_mesh,
                    write_ascii=write_ply_ascii,
                )

                # o3d.t.io.write_triangle_mesh(os.path.join(mesh_data_dir, f"building_{idx}.ply"), wedge, write_ascii=write_ply_ascii)

                # Add shape elements for PLY files in the folder
                sionna_shape = ET.SubElement(
                    scene, "shape", type="ply", id=f"mesh-building_{idx}_rooftop"
                )
                ET.SubElement(
                    sionna_shape,
                    "string",
                    name="filename",
                    value=f"mesh/building_{idx}_rooftop.ply",
                )
                ET.SubElement(sionna_shape, "ref", id=rooftop_material_type, name="bsdf")
                ET.SubElement(sionna_shape, "boolean", name="face_normals", value="true")

                sionna_shape = ET.SubElement(
                    scene, "shape", type="ply", id=f"mesh-building_{idx}_wall"
                )
                ET.SubElement(
                    sionna_shape,
                    "string",
                    name="filename",
                    value=f"mesh/building_{idx}_wall.ply",
                )
                ET.SubElement(sionna_shape, "ref", id=wall_material_type, name="bsdf")
                ET.SubElement(sionna_shape, "boolean", name="face_normals", value="true")

                if generate_building_map:
                    self._draw_building(building_polygon, building_height)

            del hag_handler
            # save centroids_and_heights to a CSV file
            centroids_and_heights_df = pd.DataFrame(centroids_and_heights)
            centroids_and_heights_df.to_csv(
                os.path.join(data_dir, "building_centroids_and_heights.csv"), index=False
            )
            xml_string = ET.tostring(scene, encoding="utf-8")
            xml_pretty = minidom.parseString(xml_string).toprettyxml(
                indent="    "
            )  # Adjust the infdent as needed

            with open(
                os.path.join(data_dir, "scene.xml"), "w", encoding="utf-8"
            ) as xml_file:
                xml_file.write(xml_pretty)

            if generate_building_map:
                np.save(
                    os.path.join(data_dir, "2D_Building_Height_Map.npy"),
                    np.array(self._building_map),
                )

            return np.array(self._building_map)

    def _draw_building(self, building_polygon, building_height):
        # Draw onto a temporary image
        temp = Image.new("I", self._building_map.size, 0)
        draw = ImageDraw.Draw(temp)

        local_exterior = reorder_localize_coords(
            building_polygon.exterior,
            self._ground_polygon_envelope_UTM.bounds[0],
            self._ground_polygon_envelope_UTM.bounds[3],
        )

        draw.polygon(
            [(x, -y) for x, y in local_exterior],
            outline=int(building_height),
            fill=int(building_height),
        )

        # Remove holes
        for interior in building_polygon.interiors:
            local_interior = reorder_localize_coords(
                interior,
                self._ground_polygon_envelope_UTM.bounds[0],
                self._ground_polygon_envelope_UTM.bounds[3],
            )

            draw.polygon(
                [(x, -y) for x, y in local_interior],
                outline=0,
                fill=0,
            )

        current = np.asarray(self._building_map)
        new = np.asarray(temp)

        merged = np.maximum(current, new)
        self._building_map = Image.fromarray(merged.astype(np.uint32))