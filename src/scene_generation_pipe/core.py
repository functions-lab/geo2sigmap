
import logging
import math
import os

import numpy as np
import shapely
from shapely.geometry import shape, Polygon
from shapely import affinity
import open3d as o3d
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import osmnx as ox


from tqdm import tqdm
from triangle import triangulate
from PIL import Image, ImageDraw
from pyproj import Transformer

from .utils import *



# Create a module-level logger
logger = logging.getLogger(__name__)

class Scene:
    """
    A class that encapsulates the logic for creating the ditital twins for a given
    bouding box with building information querying from OpenStreetMap server and 
    ground mesh from lidar.


    Usage:
        scene_instance = Scene()
        scene_instance(
            points=[(lon1, lat1), (lon2, lat2), ...],
            data_dir="path/to/output",
            hag_tiff_path="path/to/hag.tiff",
            osm_server_addr="https://overpass-api.de/api/interpreter",
            lidar_calibration=True,
            generate_building_map=True
        )
    """
    def __call__(
        self,
        points,
        data_dir,
        hag_tiff_path,
        osm_server_addr=None,
        lidar_calibration: bool = True,
        generate_building_map: bool = True
    ):
        """
        Generate a ground mesh from the given polygon (defined by `points`),
        query OSM for building footprints, extrude them into 3D meshes,
        and optionally produce a 2D building-height map.

        Parameters
        ----------
        points : list of (float, float)
            Coordinates defining the polygon (in WGS84 lon/lat).
        data_dir : str
            Directory where output files (XML, meshes, etc.) will be saved.
        osm_server_addr : str, optional
            Custom Overpass API endpoint. If None, osmnx's default is used.
        lidar_calibration : bool, optional
            If True, attempt to derive building heights from the HAG file; else use random fallback.
        generate_building_map : bool, optional
            If True, generate a 2D building map image (and save as a NumPy file).

        Returns
        -------
        np.ndarray
            If generate_building_map is True, returns a 2D NumPy array of building heights.
            Otherwise, returns None.
        """
        
        # ---------------------------------------------------------------------
        # 1) Setup OSM server and transforms
        # ---------------------------------------------------------------------
        if osm_server_addr:
            ox.settings.overpass_url = osm_server_addr
            ox.settings.overpass_rate_limit = False
        

        # Determine the UTM projection from the first point
        projection_UTM_EPSG_code = get_utm_epsg_code_from_gps(points[0][0], points[0][1])
        logger.info(f"For the given bbox, using UTM area: {projection_UTM_EPSG_code}")

        # Create transformations between WGS84 (EPSG:4326) and UTM
        to_projection = Transformer.from_crs("EPSG:4326", projection_UTM_EPSG_code, always_xy=True)
        to_4326 = Transformer.from_crs(projection_UTM_EPSG_code, "EPSG:4326", always_xy=True)

        # ---------------------------------------------------------------------
        # 2) Prepare output directories and camera / material settings
        # ---------------------------------------------------------------------
        mesh_data_dir = os.path.join(data_dir, "mesh")
        os.makedirs(os.path.join(mesh_data_dir), exist_ok=True)

        # top_left_6933 = to_projection.transform(top_left[1], top_left[0])

        # bottom_right = to_4326.transform(top_left_6933[0] + width, top_left_6933[1] - height)

        # bottom_right = bottom_right[1], bottom_right[0]

        # print(bottom_right)
        # Calculate the other two corners based on the provided corners
        # top_right = (top_left[0], bottom_right[1])
        # bottom_left = (bottom_right[0], top_left[1])

        # Default Mitsuba rendering parameters
        spp_default = 4096
        resx_default = 1024
        resy_default = 768

        camera_settings = {
            "rotation": (0, 0, -90),  # Assuming Z-up orientation
            "fov": 42.854885
        }

        # Define material colors. This is RGB 0-1 formar https://rgbcolorpicker.com/0-1
        material_colors = {
            "mat-itu_concrete": (0.539479, 0.539479, 0.539480),
            "mat-itu_marble": (0.701101, 0.644479, 0.485150),
            "mat-itu_metal": (0.219526, 0.219526, 0.254152),
            "mat-itu_wood": (0.043, 0.58, 0.184),
            "mat-itu_wet_ground": (0.91, 0.569, 0.055),
        }

        # ---------------------------------------------------------------------
        # 3) Build the XML scene root
        # ---------------------------------------------------------------------

        scene = ET.Element("scene", version="2.1.0")
        # Default integrator / film settings
        ET.SubElement(scene, "default", name="spp", value=str(spp_default))
        ET.SubElement(scene, "default", name="resx", value=str(resx_default))
        ET.SubElement(scene, "default", name="resy", value=str(resy_default))

        integrator = ET.SubElement(scene, "integrator", type="path")
        ET.SubElement(integrator, "integer", name="max_depth", value="12")

        # Define materials
        for material_id, rgb in material_colors.items():
            bsdf_twosided = ET.SubElement(scene, "bsdf", type="twosided", id=material_id)
            bsdf_diffuse = ET.SubElement(bsdf_twosided, "bsdf", type="diffuse")
            ET.SubElement(bsdf_diffuse, "rgb", value=f"{rgb[0]} {rgb[1]} {rgb[2]}", name="reflectance")

        # Add emitter (constant environment light)
        emitter = ET.SubElement(scene, "emitter", type="constant", id="World")
        ET.SubElement(emitter, "rgb", value="1.000000 1.000000 1.000000", name="radiance")

        # Add camera (sensor)
        sensor = ET.SubElement(scene, "sensor", type="perspective", id="Camera")
        ET.SubElement(sensor, "string", name="fov_axis", value="x")
        ET.SubElement(sensor, "float", name="fov", value=str(camera_settings["fov"]))
        ET.SubElement(sensor, "float", name="principal_point_offset_x", value="0.000000")
        ET.SubElement(sensor, "float", name="principal_point_offset_y", value="-0.000000")
        ET.SubElement(sensor, "float", name="near_clip", value="0.100000")
        ET.SubElement(sensor, "float", name="far_clip", value="10000.000000")
        sionna_transform = ET.SubElement(sensor, "transform", name="to_world")
        ET.SubElement(sionna_transform, "rotate", x="1", angle=str(camera_settings["rotation"][0]))
        ET.SubElement(sionna_transform, "rotate", y="1", angle=str(camera_settings["rotation"][1]))
        ET.SubElement(sionna_transform, "rotate", z="1", angle=str(camera_settings["rotation"][2]))
        camera_position = np.array([0, 0, 100])  # Adjust camera height
        ET.SubElement(sionna_transform, "translate", value=" ".join(map(str, camera_position)))
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

        center_x_polygon = ground_polygon.centroid.x
        center_y_polygon = ground_polygon.centroid.y

        center_x = ground_polygon.envelope.centroid.x
        center_y = ground_polygon.envelope.centroid.y
        
        print("center_x_polygon,center_y_polygon",center_x_polygon,center_y_polygon )
        print("center_x, center_y",center_x, center_y)

        print("ground_polygon_bbox",ground_polygon_bbox)

        #######Open3D#######
        outer_xy = unique_coords(reorder_localize_coords(ground_polygon.exterior, center_x, center_y))
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
        d = triangulate(dict(vertices=verts, segments=edges), opts='p')

        # Convert back to pyvista
        v, f = d['vertices'], d['triangles']
        nv, nf = len(v), len(f)
        points = np.concatenate([v, np.zeros((nv, 1))], axis=1)

        logger.debug(f"points from triangulate: {points}" )
        # print("faces from triangulate", faces)


        # Build Open3D TriangleMesh
        mesh_o3d = o3d.t.geometry.TriangleMesh()
        mesh_o3d.vertex.positions = o3d.core.Tensor(points)
        mesh_o3d.triangle.indices = o3d.core.Tensor(f)
        
        #logger.debug(f"mesh_o3d.get_center():{mesh_o3d.scale(1.2, mesh_o3d.get_center())}" )

        o3d.t.io.write_triangle_mesh(os.path.join(mesh_data_dir, f"ground.ply"), mesh_o3d)

        material_type = "mat-itu_wet_ground"
        sionna_shape = ET.SubElement(scene, "shape", type="ply", id=f"mesh-ground")
        ET.SubElement(sionna_shape, "string", name="filename", value=f"mesh/ground.ply")
        bsdf_ref = ET.SubElement(sionna_shape, "ref", id=material_type, name="bsdf")
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
        logger.info(f"Estimated ground coverage: width={width}m, height={height}m")

        top_left = (west, north)

        # OSMnx features API uses bounding box in the form (north, south, east, west)
        logger.debug(f"OSM bounding box: (north={north}, south={south}, east={east}, west={west})")
        buildings = ox.features.features_from_bbox(bbox=ground_polygon_4326_bbox,
                                                tags={'building': True})
        buildings = buildings.to_crs(projection_UTM_EPSG_code)

        #Filter out the building which outside the bounding box since 
        #OSM will return some extra buildings.
        filtered_buildings = buildings[buildings.intersects(ground_polygon)]
        buildings_list = filtered_buildings.to_dict('records')


        # ---------------------------------------------------------------------
        # 6) If generating building map, prepare an empty grayscale image
        # ---------------------------------------------------------------------

        # Create a new empty Image, mode 'L' means 8bit grayscale image.
        img = Image.new('L', (width, height), 0)

        tmpres = to_projection.transform(top_left[0], top_left[1])
        tmpres = (ground_polygon_bbox[0], ground_polygon_bbox[3])
        print("tmpres",tmpres)
        # ---------------------------------------------------------------------
        # 7) Init the building height handler. (osm or lidar)
        # ---------------------------------------------------------------------
        if lidar_calibration:
            try:
                hag_handler = GeoTIFFHandler(hag_tiff_path)
            except Exception as e:
                hag_handler = None
        else:
            hag_handler = None

        # ---------------------------------------------------------------------
        # 8) Process each building to create a 3D mesh (extrude by building height)
        # ---------------------------------------------------------------------
        for idx, building in tqdm(enumerate(buildings_list), total=len(buildings_list),desc="Parsing buildings"):

            # Debug the inner hole buildings
            # if building['type'] != "multipolygon":
            #     continue
            # Convert building geometry to a shapely polygon
            building_polygon = shape(building['geometry'])

            if building_polygon.geom_type != 'Polygon':
                logger.debug(f"building_polygon.geom_type: {building_polygon.geom_type}")
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
                filtered_list = [x for x in abs_height if x.size > 0 and x != -9999 and x > 2]
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

            # building_height = NYC_LiDAR_building_height(building, building_polygon)

            outer_xy = unique_coords(reorder_localize_coords(building_polygon.exterior, center_x, center_y))

            holes_xy = []
            if len(list(building_polygon.interiors)) != 0:
                for inner_hole in list(building_polygon.interiors):
                    valid_coords = reorder_localize_coords(inner_hole, center_x, center_y)
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
                d = triangulate(dict(vertices=verts, segments=edges, holes=holes), opts='p')
            else:
                d = triangulate(dict(vertices=verts, segments=edges), opts='p')

            # Convert back to pyvista
            v, f = d['vertices'], d['triangles']
            nv, nf = len(v), len(f)

            # print(v)
            # print(f)

            points = np.concatenate([v, np.zeros((nv, 1))], axis=1)
            mesh_o3d = o3d.t.geometry.TriangleMesh()
            mesh_o3d.vertex.positions = o3d.core.Tensor(points)
            mesh_o3d.triangle.indices = o3d.core.Tensor(f)

            wedge = mesh_o3d.extrude_linear([0, 0, building_height])
            o3d.t.io.write_triangle_mesh(os.path.join(mesh_data_dir, f"building_{idx}.ply"), wedge)

            material_type = "mat-itu_marble"
            # Add shape elements for PLY files in the folder
            sionna_shape = ET.SubElement(scene, "shape", type="ply", id=f"mesh-building_{idx}")
            ET.SubElement(sionna_shape, "string", name="filename", value=f"mesh/building_{idx}.ply")
            bsdf_ref = ET.SubElement(sionna_shape, "ref", id=material_type, name="bsdf")
            ET.SubElement(sionna_shape, "boolean", name="face_normals", value="true")

            if generate_building_map:

                local_exterior = reorder_localize_coords(building_polygon.exterior, tmpres[0], tmpres[1])
                ImageDraw.Draw(img).polygon([(x, -y) for x, y in list(local_exterior)],
                                            outline=int(building_height), fill=int(building_height))
                # local_coor_building_polygon = affinity.translate(building_polygon, xoff=-1 * tmpres[0], yoff=-1 * tmpres[1])
                # # print("local_coor_building_polygon:",local_coor_building_polygon,'\n\n\n\n')

                # local_coor_building_polygon = round_polygon_coordinates(local_coor_building_polygon)

                # ImageDraw.Draw(img).polygon([(x, -y) for x, y in list(local_coor_building_polygon.exterior.coords)],
                #                             outline=int(building_height), fill=int(building_height))

                # # Handle the "holes" inside the polygon
                # if len(list(local_coor_building_polygon.interiors)) != 0:
                #     for inner_hole in list(local_coor_building_polygon.interiors):
                #         ImageDraw.Draw(img).polygon([(x, -y) for x, y in list(inner_hole.coords)], outline=int(0),
                #                                     fill=int(0))
            # Create and write the XML file
        del hag_handler
        xml_string = ET.tostring(scene, encoding="utf-8")
        xml_pretty = minidom.parseString(xml_string).toprettyxml(indent="    ")  # Adjust the indent as needed

        with open(os.path.join(data_dir, "scene.xml"), "w", encoding="utf-8") as xml_file:
            xml_file.write(xml_pretty)

        if generate_building_map:
            np.save(os.path.join(data_dir, '2D_Building_Height_Map.npy'), np.array(img))
        # plt.figure(figsize=(width / 96, height / 96), dpi=96)
        # plt.imshow(img, interpolation='none', interpolation_stage="rgba")
        # plt.colorbar()
        return np.array(img)
        # plt.show()