import bpy

# these are necessary for install_package(package_name)
import subprocess
import sys
import os
import time
import uuid
import argparse

import numpy as np
import traceback




if '--' in sys.argv:
    argv = sys.argv[sys.argv.index('--') + 1:]
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--idx', type=int, required=True)
parser.add_argument('-l', '--minLon', type=float, required=True)
parser.add_argument('-t', '--maxLat', type=float, required=True)
parser.add_argument('-r', '--maxLon', type=float, required=True)
parser.add_argument('-b', '--minLat', type=float, required=True)
parser.add_argument('-d', '--decimate_factor', type=int, required=False, default=1)
parser.add_argument('-p', '--BASE_PATH', type=str, required=True)
parser.add_argument('-u', '--BLENDER_OSM_DOWNLOAD_PATH', type=str, required=True)
parser.add_argument('-n', '--idx_uuid', type=str, required=True)
parser.add_argument('--terrain_or_plane', type=str, required=True)
parser.add_argument('--export_buildings', type=str, required=True)
parser.add_argument('--xml_to_building_map', type=str, required=True)  # =='y' when using existing xml to produce building map

parser.add_argument('-o', '--building_to_area_ratio', type=float, required=True)
parser.add_argument('--peoject_base_path', type=str, required=True)

args = parser.parse_known_args(argv)[0]

# this is the path where the results are stored (terrain_npy, building_npy, height files, Mitsuba_export)
BASE_PATH = args.BASE_PATH  # '/Users/zeyuli/Desktop/Duke/0. Su23_Research/Blender_stuff/res/'
CAMERA_ORTHO_SCALE = 1920



def install_Blender_add_on():
    bpy.ops.preferences.addon_install(filepath=os.path.join(args.peoject_base_path,'blosm.zip'))
    bpy.ops.preferences.addon_enable(module='blosm')
    bpy.ops.preferences.addon_install(filepath=os.path.join(args.peoject_base_path,'mitsuba-blender.zip'))
    bpy.ops.preferences.addon_enable(module='mitsuba-blender')
    bpy.context.preferences.addons['blosm'].preferences.dataDir = "/tmp"
    print(bpy.context.preferences.addons['blosm'].preferences.dataDir)
    # bpy.ops.wm.save_userpref()
def install_package(package_name):
    print("Start pip install process")
    try:
        # path to python.exe
        python_exe = os.path.join(sys.prefix, 'bin', 'python3.10')
        print(python_exe)
        # upgrade pip
        #subprocess.call([python_exe, "-m", "ensurepip"])
        subprocess.call([python_exe, "-m", "pip", "install", "--upgrade", "pip"])

        # install required packages
        subprocess.call([python_exe, "-m", "pip", "install", package_name])

        print("DONE")
        return
    except Exception as e:
        print("got error in install pip")
        raise e



# install_package("pillow")
install_package("mitsuba==3.0.1")
install_Blender_add_on()
# from PIL import Image
def delete_terrain_and_osm_files(PATH_download=BASE_PATH + 'Blender_download_files'):
    try:
        folder_path_osm = PATH_download + '/osm'  # enter path here
        delete_files_from_directory(folder_path_osm)
        folder_path_terrain = PATH_download + '/terrain'
        delete_files_from_directory(folder_path_terrain)
        return
    except Exception as e:
        raise e


def delete_files_from_directory(folder_path):
    try:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
        print("Deletion done")
        return
    except Exception as e:
        raise e


def delete_all_in_collection():
    try:
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)

        # clears all collections
        collections_list = list(bpy.data.collections)
        for col in collections_list:
            bpy.data.collections.remove(col)

        for block in bpy.data.meshes:
            bpy.data.meshes.remove(block)

        for block in bpy.data.materials:
            bpy.data.materials.remove(block)

        for block in bpy.data.textures:
            bpy.data.textures.remove(block)

        for block in bpy.data.images:
            bpy.data.images.remove(block)
        bpy.ops.outliner.orphans_purge(do_local_ids=True)
        return
    except Exception as e:
        raise e


def normalise_to_png(arr_to_norm, maxVal):
    try:
        return maxVal * (arr_to_norm - np.min(arr_to_norm)) / (np.max(arr_to_norm) - np.min(arr_to_norm))
    except Exception as e:
        raise e


def add_osm(maxLon, minLon, maxLat, minLat, from_file='n', osmFilepath=None):
    try:
        bpy.data.scenes['Scene'].blosm.osmSource = 'server'
        bpy.data.scenes['Scene'].blosm.dataType = 'osm'
        bpy.data.scenes['Scene'].blosm.ignoreGeoreferencing = True

        bpy.data.scenes['Scene'].blosm.maxLon = maxLon
        bpy.data.scenes['Scene'].blosm.minLon = minLon

        bpy.data.scenes['Scene'].blosm.maxLat = maxLat
        bpy.data.scenes['Scene'].blosm.minLat = minLat

        # ensure correct settings:
        # does not import as single object
        bpy.data.scenes["Scene"].blosm.singleObject = False

        # set osm import mode to 3Dsimple
        bpy.data.scenes["Scene"].blosm.mode = '3Dsimple'

        # only import buildings from osm
        bpy.data.scenes["Scene"].blosm.buildings = True
        bpy.data.scenes["Scene"].blosm.water = False
        bpy.data.scenes["Scene"].blosm.forests = False
        bpy.data.scenes["Scene"].blosm.vegetation = False
        bpy.data.scenes["Scene"].blosm.highways = False
        bpy.data.scenes["Scene"].blosm.railways = False
        bpy.data.scenes[0].render.engine = "BLENDER_EEVEE"
        # the following lines are for running on the server:
        # bpy.data.scenes[0].render.engine = "CYCLES"
        #
        # # Set the device_type
        # bpy.context.preferences.addons[
        #      "cycles"
        # ].preferences.compute_device_type = "CUDA" # or "OPENCL"
        #
        # # Set the device and feature set
        # bpy.context.scene.cycles.device = "GPU"
        # import from server
        if from_file == 'n':
            start = time.time()
            bpy.ops.blosm.import_data()
            print('\n\nosm download: ', str(time.time() - start) + ' seconds\n\n')
        if from_file == 'y' and osmFilepath is not None:
            bpy.data.scenes['Scene'].blosm.osmSource = 'file'
            bpy.data.scenes['Scene'].blosm.osmFilepath = osmFilepath
            bpy.ops.blosm.import_data()
        return
    except Exception as e:
        raise e


def add_terrain(material_name, maxLon, minLon, maxLat, minLat):
    try:
        # toggle to 'terrain' import mode
        bpy.data.scenes['Scene'].blosm.dataType = 'terrain'

        # ensure correct settings
        bpy.data.scenes['Scene'].blosm.ignoreGeoreferencing = False
        bpy.data.scenes['Scene'].blosm.ignoreGeoreferencing = True

        # set bounds of import
        bpy.data.scenes['Scene'].blosm.maxLon = maxLon
        bpy.data.scenes['Scene'].blosm.minLon = minLon

        bpy.data.scenes['Scene'].blosm.maxLat = maxLat
        bpy.data.scenes['Scene'].blosm.minLat = minLat

        # import
        start = time.time()
        bpy.ops.blosm.import_data()
        print('\n\nTerrain download: ', str(time.time() - start) + ' seconds\n\n')

        # set properties
        terrain_obj = bpy.data.objects['Terrain']
        mat = bpy.data.materials.get(material_name)

        if mat is None:
            mat = bpy.data.materials.new(name=material_name)
        terrain_obj.data.materials.append(mat)
        return
    except Exception as e:
        raise e


def add_plane(material_name, size=1100):
    try:
        bpy.ops.mesh.primitive_plane_add(size=size)

        plane_obj = bpy.data.objects['Plane']
        mat = bpy.data.materials.get(material_name)

        if mat is None:
            # create material
            mat = bpy.data.materials.new(name=material_name)
        plane_obj.data.materials.append(mat)
        print(plane_obj.data.materials[0])
        return
    except Exception as e:
        raise e


def replace_material(obj_data, mat_src, mat_dest):
    try:
        for iii in range(len(obj_data.materials)):
            # test if the old material is in the slot and replace it
            if obj_data.materials[iii] == mat_src:
                obj_data.materials[iii] = mat_dest
        return
    except Exception as e:
        raise e


def change_material_names_and_export(wall_name, roof_name, f_path, export='y'):
    print('\n\n\nstarting change material names and export\n\n\n')
    try:
        if len([obj.name for obj in list(bpy.data.objects)]) <= 2:
            return 1

        obj_names = []
        for obj in list(bpy.data.objects):
            if 'Camera' in obj.name or 'Terrain' in obj.name or 'osm_buildings' in obj.name or 'Plane' == obj.name:
                continue
            obj_names.append(obj.name)
        # check that there's more than one object
        if len(list(bpy.data.objects)) < 1:
            raise Exception("There are no OSM building objects in this Scene Collection")
        else:
            mat_wall = bpy.data.materials.get(wall_name)
            if mat_wall is None:
                bpy.data.materials.new(name=wall_name)
                bpy.data.materials[wall_name].diffuse_color[0] = 0.666

            mat_roof = bpy.data.materials.get(roof_name)
            if mat_roof is None:
                bpy.data.materials.new(name=roof_name)
            bpy.ops.object.select_all(action='DESELECT')
            for obj_name in obj_names:
                obj_data = bpy.data.objects[obj_name].data
                # bpy.ops.object.select_all(action='DESELECT')
                try:
                    if len(list(obj_data.materials)) <= 1:
                        obj_to_delete = bpy.data.objects[obj_name]
                        bpy.data.objects.remove(obj_to_delete, do_unlink=True)
                    for mat_idx in range(len(list(obj_data.materials))):

                        bpy.data.objects[obj_name].active_material_index = mat_idx
                        if bpy.data.objects[obj_name].active_material.name == 'wall' or bpy.data.objects[
                            obj_name].active_material.name == 'roof':
                            if bpy.data.objects[obj_name].active_material.name == 'wall':
                                mat_destination = bpy.data.materials[wall_name]
                            else:
                                mat_destination = bpy.data.materials[roof_name]
                            mat_source = bpy.data.materials[bpy.data.objects[obj_name].active_material.name]
                            replace_material(obj_data, mat_source, mat_destination)
                            # replace mat_source with mat_destination
                        else:
                            print('not wall or roof', bpy.data.objects[obj_name],
                                  bpy.data.objects[obj_name].active_material)
                            print('num mat:', len(list(obj_data.materials)), list(obj_data.materials))
                            if len(list(obj_data.materials)) == 2:
                                if mat_idx == 0:
                                    mat_destination = bpy.data.materials[wall_name]
                                if mat_idx == 1:
                                    mat_destination = bpy.data.materials[roof_name]
                                mat_source = bpy.data.materials[bpy.data.objects[obj_name].active_material.name]
                                replace_material(obj_data, mat_source, mat_destination)
                            if len(list(obj_data.materials)) > 2:
                                bpy.data.objects[obj_name].data.materials.pop(index=mat_idx)
                            print('after replacement:', len(list(obj_data.materials)), list(obj_data.materials))
                except AttributeError:
                    print('AttributeError', bpy.data.objects[obj_name], bpy.data.objects[obj_name].active_material)
                    # un-linking object.
                    obj_to_delete = bpy.data.objects[obj_name]
                    bpy.data.objects.remove(obj_to_delete, do_unlink=True)
                except RuntimeError as err:
                    print('RuntimeError', bpy.data.objects[obj_name], bpy.data.objects[obj_name].active_material)
                    print(err)
                    obj_to_delete = bpy.data.objects[obj_name]
                    bpy.data.objects.remove(obj_to_delete, do_unlink=True)
                    # raise err
                except KeyError:
                    pass
        bpy.ops.object.select_all(action='DESELECT')
        if export == 'y' and f_path is not None:
            bpy.ops.export_scene.mitsuba(filepath=f_path, axis_forward='Y', axis_up='Z')
        return 0
    except Exception as e:
        raise e


def squarify_photo(arr, trim=0):
    try:
        arr = np.asarray(arr)
        rrr, ccc = arr.shape
        min_rc = min([rrr, ccc]) - trim
        return arr[int((rrr - min_rc) / 2):int((rrr + min_rc) / 2), int((ccc - min_rc) / 2):int((ccc + min_rc) / 2)]
    except Exception as e:
        raise e


def terrain_to_npy(outer_idx, save='n', camera_height=2000, camera_orthoScale=CAMERA_ORTHO_SCALE, decimate='n',
                   decimate_factor=8):
    """
    The important part: returns terrain_height as np array.
    """
    try:
        bpy.data.objects["Terrain"].select_set(True)

        # compute mesh and vertices from Terrain object
        terrain_dg = bpy.context.evaluated_depsgraph_get()
        terrain_obj = bpy.context.object.evaluated_get(terrain_dg)
        mesh = terrain_obj.to_mesh(depsgraph=terrain_dg)

        # compute the boundaries of the terrain
        # min_x: top-left corner; min_y: bottom-right corner
        # max_x: bottom-right corner; max_y: top-left corner
        min_x, min_y = mesh.vertices[0].co.x, mesh.vertices[-1].co.y
        max_x, max_y = mesh.vertices[-1].co.x, mesh.vertices[0].co.y

        print('x min and max of terrain mesh: ', min_x, max_x)
        print('y min and max of terrain mesh: ', min_y, max_y)
        add_camera_and_set(camera_height, camera_orthoScale)

        if save == 'y' and outer_idx != -1:
            terrain_depth = take_picture_and_get_depth()

            # make terrain_depth square
            terrain_depth_square = squarify_photo(terrain_depth)
            # transform into height above xy plane in meters
            terrain_height = camera_height - terrain_depth_square
            print(terrain_height[0, :])

            # decimate (I know that scipy can also do this) by a factor of 8
            # it is also proper to do it with scipy since the frequency
            # components are considered, but I can't be bothered
            if decimate == 'y':
                terrain_height = terrain_height[::decimate_factor, ::decimate_factor]
            if outer_idx != -1:
                return min_x, max_x, min_y, max_y, terrain_height

        bpy.data.objects["Terrain"].select_set(False)
        return min_x, max_x, min_y, max_y
    except Exception as e:
        raise e


def get_depth():
    try:
        print("take_picture",flush=True)
        """Obtains depth map from Blender render.
        :return: The depth map of the rendered camera view as a numpy array of size (H,W).
        """
        z = bpy.data.images['Viewer Node']
        w, h = z.size
        dmap = np.array(z.pixels[:], dtype=np.float32)  # convert to numpy array
        dmap = np.reshape(dmap, (h, w, 4))[:, :, 0]
        dmap = np.rot90(dmap, k=2)
        dmap = np.fliplr(dmap)
        return dmap
    except Exception as e:
        print("ERROR get_depth",flush=True)
        raise e


def clear_compositing_nodes():
    try:
        bpy.data.scenes['Scene'].use_nodes = True
        tree = bpy.data.scenes['Scene'].node_tree
        for node in tree.nodes:
            tree.nodes.remove(node)
        return
    except Exception as e:
        raise e


def add_camera_and_set(camera_height, camera_orthoScale):
    try:
        # note: the camera is slightly "thinner" than the terrain.
        # Increase  camera_orthoScale  to increase the area captured.
        camera_data = bpy.data.cameras.new(name='Camera')
        camera_object = bpy.data.objects.new('Camera', camera_data)
        bpy.context.scene.collection.objects.link(camera_object)
        curr_camera = bpy.data.cameras["Camera"]

        camera_object.location[2] = camera_height  # setting camera height
        curr_camera.type = 'ORTHO'  # setting camera type
        curr_camera.clip_start = 0.1  # setting clipping
        curr_camera.clip_end = camera_height * 5
        curr_camera.ortho_scale = camera_orthoScale  # setting camera scale
        curr_camera.dof.use_dof = False  # do not use, this makes the photo misty
        #    curr_camera.track_axis = '-Z'
        #    curr_camera.up_axis = 'Y'
        return
    except Exception as e:
        raise e


def take_picture_and_get_depth():
    try:
        bpy.context.scene.camera = bpy.data.objects['Camera']
        # enable z data to be passed and use nodes for compositing
        bpy.data.scenes['Scene'].view_layers["ViewLayer"].use_pass_z = True
        bpy.data.scenes['Scene'].use_nodes = True

        tree = bpy.data.scenes['Scene'].node_tree

        # clear nodes
        clear_compositing_nodes()
        image_node = tree.nodes.new(type='CompositorNodeRLayers')
        viewer_node = tree.nodes.new(type='CompositorNodeViewer')
        viewer_node.location = 400, 0
        tree.links.new(image_node.outputs["Depth"], viewer_node.inputs['Image'])
        tree.nodes['Render Layers'].layer = 'ViewLayer'
        print("take_picture_1",flush=True)
        bpy.ops.render.render(layer='ViewLayer')
        print("take_picture_2",flush=True)
        return get_depth()
    except Exception as e:
        print("Error in create building height map",flush=True)
        print(e,flush=True)
        raise e


def get_height_at_origin(camera_height=2000, normalise_to_256='n', decimate='n', decimate_factor=8):
    """
    terrain_height must be of type(np.array) and values measure height from the xy plane
    returns building_height numpy array
    """
    try:
        if args.terrain_or_plane == 'plane':
            add_camera_and_set(camera_height, camera_orthoScale=CAMERA_ORTHO_SCALE)
        depth = take_picture_and_get_depth()  # values that are greater than 65500 have inf depth
        depth[depth > 65500] = camera_height

        # get height at origin using a simple reflection
        roww, coll = depth.shape
        height_at_origin = camera_height - depth[int(round(roww / 2)), int(round(coll / 2))]

        if normalise_to_256 == 'y':
            depth = normalise_to_png(depth, 256)
        if decimate == 'y':
            depth = depth[::decimate_factor, ::decimate_factor]

        height_arr = camera_height - depth
        height_square = squarify_photo(height_arr)
        return height_at_origin
    except Exception as e:
        raise e


def get_height_buildings_b4_terrain(loc_args_dict, from_file='n', osm_filepath=None, cam_ht=2000,
                                    cam_ortho_scale=CAMERA_ORTHO_SCALE, decimate='y', decimate_factor=8):
    # this should be called before most other functions in run()
    add_osm(**loc_args_dict, from_file=from_file, osmFilepath=osm_filepath)
    add_camera_and_set(camera_height=cam_ht, camera_orthoScale=cam_ortho_scale)
    # the main use of change_material_names_and_export is that it removes buildings that can't be exported
    # so that the building height map and the eventual raytrace model are fine.
    change_material_names_and_export(wall_name='itu_brick', roof_name='itu_plasterboard', f_path=None, export='n')
    building_depth = take_picture_and_get_depth()
    building_depth[building_depth > 65500] = cam_ht
    building_depth_sqr = squarify_photo(building_depth)
    building_ht_arr = cam_ht - building_depth_sqr
    if decimate == 'y':
        building_ht_arr = building_ht_arr[::decimate_factor, ::decimate_factor]
    return building_ht_arr


def get_height_buildings_after_import(cam_ht=2000, cam_ortho_scale=CAMERA_ORTHO_SCALE, decimate='y', decimate_factor=8):
    bpy.data.scenes["Scene"].render.engine = 'BLENDER_EEVEE'
    add_camera_and_set(camera_height=cam_ht, camera_orthoScale=cam_ortho_scale)
    # assumes camera has already been placed
    print("get_height", flush=True)
    building_depth = take_picture_and_get_depth()
    # print(squarify_photo(building_depth, trim=118))
    building_depth[building_depth > 65500] = cam_ht
    building_depth_sqr = squarify_photo(building_depth, trim=80)
    building_ht_arr = cam_ht - building_depth_sqr
    if decimate == 'y':
        building_ht_arr = building_ht_arr[::decimate_factor, ::decimate_factor]
    return building_ht_arr


def calc_totalOSM_above_thresh(perc, text_lines):
    try:
        c = 0
        for text_line in text_lines:
            text_line = text_line.replace('(', '')
            text_line = text_line.replace(')', '')
            text_line = text_line.replace('\n', '')
            text_line = text_line.split(',')
            _, _, _, _, temp_percent, _ = [lll for lll in text_line]
            if float(temp_percent) > perc:
                c += 1
        return c
    except Exception as e:
        raise e


def get_floats_from_coordsFile(line):
    try:
        line = line.replace('(', '')
        line = line.replace(')', '')
        line = line.replace('\n', '')
        line = line.split(',')
        # file format: (minLon, maxLat, maxLon, minLat), building_to_area ratio
        minLon, maxLat, maxLon, minLat, per = [float(l) for l in line]
        return minLon, maxLat, maxLon, minLat, per
    except Exception as e:
        raise e


def run_with_plane(maxLon, minLon, maxLat, minLat, run_idx, buildingToAreaRatio, decimate_factor,
                   use_path_osm):
    try:
        print("run_with_plane_start",flush=True)
        decimate = 'n'
        if decimate_factor != 1:
            decimate = 'y'
        osm_f_path = os.path.join(args.BLENDER_OSM_DOWNLOAD_PATH , args.idx_uuid + '.osm')
        delete_all_in_collection()

        # path of xml file which would be exported in change_material_names_and_export
        save_name = args.idx_uuid
        building_img_path = BASE_PATH + 'Bl_building_npy/' + save_name + '.npy'

        # if args.terrain_or_plane == 'plane':
        add_plane(material_name='itu_concrete', size=1000)

        loc_args_dict = {'maxLon': maxLon, 'minLon': minLon, 'maxLat': maxLat, 'minLat': minLat}
        # use already-downloaded osm is only selected when both from_file=='y' and osmFilepath is not None
        add_osm(**loc_args_dict, from_file=use_path_osm, osmFilepath=osm_f_path)

        # change_material_names_and_export WRITES the XML file as well as the Meshes
        export_path = BASE_PATH + 'Bl_xml_files/' + save_name + '/' + save_name + '.xml'
        ret = change_material_names_and_export(wall_name='itu_brick', roof_name='itu_plasterboard',
                                               f_path=export_path, export=args.export_buildings)
        if ret != 0:
            raise Exception('Not enough objects in scene for change_material_names_and_export.')
        # building height is an image containing the building height and nothing else
        building_ht_arr = get_height_buildings_after_import(cam_ht=2000, cam_ortho_scale=CAMERA_ORTHO_SCALE,
                                                            decimate=decimate, decimate_factor=decimate_factor)
        print("after export in run_with_plane",flush=True)
        print('BLENDER render engine:', str(bpy.data.scenes["Scene"].render.engine),flush=True)
        print("npy ",flush=True)
        if run_idx != -1:
            print("npy executed!")
            building_height_arr_int16 = building_ht_arr.astype(np.int16)
            np.save(building_img_path, building_height_arr_int16)
    except Exception as e:
        print("error in run_with_plane",flush=True)
        raise e
    return


def run_with_terrain(maxLon, minLon, maxLat, minLat, run_idx, buildingToAreaRatio, decimate_factor,
                     use_path_osm):
    try:
        decimate = 'n'
        if decimate_factor != 1:
            decimate = 'y'
        # bpy.ops.wm.read_userpref()
        delete_all_in_collection()
        loc_args_dict = {'maxLon': maxLon, 'minLon': minLon, 'maxLat': maxLat, 'minLat': minLat}
        osm_f_path = args.BLENDER_OSM_DOWNLOAD_PATH + args.idx_uuid + '.osm'
        building_ht_fixed = \
            get_height_buildings_b4_terrain(loc_args_dict, from_file=use_path_osm, osm_filepath=osm_f_path,
                                            cam_ht=2000, cam_ortho_scale=CAMERA_ORTHO_SCALE, decimate=decimate,
                                            decimate_factor=decimate_factor)
        delete_all_in_collection()

        # path of xml file which would be exported in change_material_names_and_export
        save_name = args.idx_uuid

        height_at_origin_path = BASE_PATH + 'height_at_origin/' + save_name + '.txt'  # 'HeightAtOrigin.txt'

        # should follow maxLon, minLon, maxLat, minLat
        diff = 0.0015
        loc_args_dict = {'maxLon': maxLon + diff, 'minLon': minLon - diff, 'maxLat': maxLat + diff,
                         'minLat': minLat - diff}

        terrain_img_path = BASE_PATH + 'Bl_terrain_npy/' + save_name + '.npy'
        building_img_path = BASE_PATH + 'Bl_building_npy/' + save_name + '.npy'

        add_terrain(material_name='itu_concrete', **loc_args_dict)
        # terrain_limits WRITES the terrain height information as png
        # increase camera_orthoScale to increase image size.
        terrain_save = 'y'

        # terrain_limits contains min_x, max_x, min_y, max_y, terrain_height
        terrain_limits = terrain_to_npy(save=terrain_save, outer_idx=run_idx, camera_height=2000,
                                        camera_orthoScale=CAMERA_ORTHO_SCALE, decimate=decimate,
                                        decimate_factor=decimate_factor)
        # if terrain_save=='n' then terrainImg is not returned.
        print(len(terrain_limits))
        terrain_arr = terrain_limits[-1]

        loc_args_dict = {'maxLon': maxLon, 'minLon': minLon, 'maxLat': maxLat, 'minLat': minLat}
        # use already-downloaded osm is only selected when both from_file=='y' and osmFilepath is not None
        add_osm(**loc_args_dict, from_file=use_path_osm, osmFilepath=osm_f_path)

        # change_material_names_and_export WRITES the XML file as well as the Meshes
        export_path = BASE_PATH + 'Bl_xml_files/' + save_name + '/' + save_name + '.xml'
        ret = change_material_names_and_export(wall_name='itu_brick', roof_name='itu_plasterboard',
                                               f_path=export_path,
                                               export=args.export_buildings)
        if ret != 0:
            raise Exception('Not enough objects in scene for change_material_names_and_export.')

        # building height is an image containing the building height and nothing else
        # note: this function outputs wavy buildings, since the building height is constant even on top of terrain
        height_at_origin = get_height_at_origin(camera_height=2000, decimate=decimate,
                                                decimate_factor=decimate_factor)

        if run_idx != -1 or True:
            f_ptr_HeightAtOrigin = open(height_at_origin_path, 'w')
            f_ptr_HeightAtOrigin.write(
                '({:f},{:f},{:f},{:f}),{:f},{:.1f},'.format(minLon, maxLat, maxLon, minLat, buildingToAreaRatio,
                                                            height_at_origin) + save_name + '\n')
            f_ptr_HeightAtOrigin.close()
            terrain_arr_int16 = terrain_arr.astype(np.int16)
            building_height_arr_int16 = building_ht_fixed.astype(np.int16)
            np.save(terrain_img_path, terrain_arr_int16)
            np.save(building_img_path, building_height_arr_int16)
        # delete_terrain_and_osm_files()
    except Exception as e:
        raise e
    return


def run_xml_to_building_map(maxLon, minLon, maxLat, minLat, run_idx, buildingToAreaRatio, decimate_factor,
                            use_path_osm):
    try:
        import_path = BASE_PATH + 'Bl_xml_files/' + args.idx_uuid + '/' + args.idx_uuid + '.xml'
        if not os.path.exists(import_path):
            return
        decimate = 'n'
        if decimate_factor != 1:
            decimate = 'y'
        delete_all_in_collection()
        bpy.ops.import_scene.mitsuba(filepath=import_path, axis_forward='Y', axis_up='Z')
        'BLENDER_EEVEE'
        building_ht_arr = get_height_buildings_after_import(cam_ht=2000, cam_ortho_scale=CAMERA_ORTHO_SCALE,
                                                            decimate=decimate, decimate_factor=decimate_factor)
        print('BLENDER render engine:', str(bpy.data.scenes["Scene"].render.engine))
        building_height_arr_int16 = building_ht_arr.astype(np.int16)
        building_img_path = BASE_PATH + 'Bl_building_npy/' + args.idx_uuid + '.npy'
        np.save(building_img_path, building_height_arr_int16)
    except Exception as e:
        traceback.print_exc()
        raise e
    return


def run_routine(maxLon, minLon, maxLat, minLat, run_idx, buildingToAreaRatio, decimate_factor,
                use_path_osm):
    if args.xml_to_building_map != 'y':
        if args.terrain_or_plane == 'plane':
            print("run_plane",flush=True)
            run_with_plane(maxLon, minLon, maxLat, minLat, run_idx, buildingToAreaRatio, decimate_factor, use_path_osm)

        if args.terrain_or_plane == 'terrain':
            run_with_terrain(maxLon, minLon, maxLat, minLat, run_idx, buildingToAreaRatio, decimate_factor, use_path_osm)
        if args.terrain_or_plane != 'plane' and args.terrain_or_plane != 'terrain':
            raise Exception('terrain or plane?')
    else:
        print("run_else")
        run_xml_to_building_map(maxLon, minLon, maxLat, minLat, run_idx, buildingToAreaRatio, decimate_factor,
                                use_path_osm)


try:
    print("run start",flush=True)
    run_routine(args.maxLon, args.minLon, args.maxLat, args.minLat, args.idx, args.building_to_area_ratio,
                decimate_factor=args.decimate_factor, use_path_osm='y')
    print('DONE with 1 rep')
    # raise KeyboardInterrupt
except Exception as e:
    raise e
