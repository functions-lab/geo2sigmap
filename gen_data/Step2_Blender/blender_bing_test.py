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
        

def add_osm(from_file, osmFilepath):
    try:
        bpy.data.scenes['Scene'].blosm.dataType = 'osm'
        bpy.data.scenes['Scene'].blosm.ignoreGeoreferencing = True

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
        # ].preferences.compute_device_type = "METAL" # or "OPENCL"
        #
        # # Set the device and feature set
        # bpy.context.scene.cycles.device = "GPU"
        # import from server
        if from_file == 'n':
            raise NotImplementedError
        if from_file == 'y' and osmFilepath is not None:
            bpy.data.scenes['Scene'].blosm.osmSource = 'file'
            bpy.data.scenes['Scene'].blosm.osmFilepath = osmFilepath
            bpy.ops.blosm.import_data()
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


CAMERA_ORTHO_SCALE = 1920
        
        
def squarify_photo(arr, trim=0):
    try:
        arr = np.asarray(arr)
        rrr, ccc = arr.shape
        min_rc = min([rrr, ccc]) - trim
        return arr[int((rrr - min_rc) / 2):int((rrr + min_rc) / 2), int((ccc - min_rc) / 2):int((ccc + min_rc) / 2)]
    except Exception as e:
        raise e


def get_depth():
    try:
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

        bpy.ops.render.render(layer='ViewLayer')

        return get_depth()
    except Exception as e:
        raise e



def get_height_buildings_after_import(cam_ht=2000, cam_ortho_scale=CAMERA_ORTHO_SCALE, decimate='y', decimate_factor=8):
    bpy.data.scenes["Scene"].render.engine = 'CYCLES'
    add_camera_and_set(camera_height=cam_ht, camera_orthoScale=cam_ortho_scale)
    # assumes camera has already been placed
    building_depth = take_picture_and_get_depth()
    # print(squarify_photo(building_depth, trim=118))
    building_depth[building_depth > 65500] = cam_ht
    building_depth_sqr = squarify_photo(building_depth, trim=80)
    building_ht_arr = cam_ht - building_depth_sqr
    if decimate == 'y':
        building_ht_arr = building_ht_arr[::decimate_factor, ::decimate_factor]
    return building_ht_arr



delete_all_in_collection()
add_plane("itu_concrete")
add_osm('y', '/Users/zeyuli/Desktop/Duke/0. Su23_Research/geo2sigmap/gen_data/Step2_Blender/osm/pci_EF_new.osm')

building_ht_arr = get_height_buildings_after_import(cam_ht=2000, cam_ortho_scale=CAMERA_ORTHO_SCALE,
                                                            decimate='n', decimate_factor=1)
building_height_arr_int16 = building_ht_arr.astype(np.int16)
np.save('/Users/zeyuli/Desktop/Duke/0. Su23_Research/geo2sigmap/gen_data/Step2_Blender/npy/pci_EF_new', building_height_arr_int16)