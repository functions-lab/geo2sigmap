
import os # Configure which GPU

import time
import argparse
import traceback
from datetime import datetime

from PIL import Image
import tensorflow as tf
import numpy as np

# Import Sionna RT components
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera

"""
    Base path for the Blender files. The height files, pngs, and xml folders should be in BASE_PATH. 
    Organisation: BASE_PATH/Bl_terrain_img/, BASE_PATH/Bl_xml_files/, BASE_PATH/height_at_origin/, 
"""
"""
    Parse arguments
"""
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--file_name_wo_type', type=str, required=True)
# extra_height is added to the height above the xy-plane of the terrain/building at the origin when
# placing Tx and when calculating signal strength in the coverage_map function
parser.add_argument('-e', '--extra_height', type=float, required=True)
# parser.add_argument('end', nargs='?', type=int, required=True)
parser.add_argument('-c', '--cm_cell_size', type=float, required=False, default=10)
parser.add_argument('-b', '--BASE_PATH_BLENDER', type=str, required=False, default='/res/')
parser.add_argument('-s', '--BASE_PATH_SIONNA', type=str, required=True)
parser.add_argument('-n', '--outer_idx', type=int, required=True)
parser.add_argument('-m', '--cm_num_samples', type=int, required=True)
parser.add_argument('-a', '--antenna_pattern', type=str, required=True)
args = parser.parse_args()
# print('height file name: ', args.height_file)
# print('extra height: ', args.extra_height)
BASE_PATH = args.BASE_PATH_BLENDER
BASE_PATH_SIONNA = args.BASE_PATH_SIONNA



"""
    Setting up the environment, including the GPUs
"""
# gpu_num = 0 # Use "" to use the CPU


gpus = tf.config.list_physical_devices('GPU')
print(gpus)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), 'Physical GPUs, ', len(logical_gpus), 'Logical GPUs')
    except RuntimeError as e:
        print(e)  # Avoid warnings from TensorFlow
        
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.random.set_seed(1) # Set global random seed for reproducibility


def generate_coverage_map_config_combination(file_path):

    # 1000 * 1000
    building_npy = np.load(file_path)
    """
    ----------
    ----------
    --TX----TX--
    ----------
    ----------
    ----TX-----
    ----------
    ----------
    --TX----TX--
    ----------
    """
    tx_xy_position = [[-244,244], [-244,-244], [244,244], [244,-244]]
    tx_height = [2]

    cm_conf_set = []
    for cm_conf in tx_xy_position:
        x = cm_conf[0] + 500
        y = cm_conf[1] + 500

        x = int(x)
        y = int(y)

        # Obtain the max height of the area
        building_height_at_xy_position = np.max(building_npy[x-256:x+256, y-256:y+256])

        for height in tx_height:
            for i in range(4):
                cm_conf_set.append([*cm_conf, height + building_height_at_xy_position, 
                                    np.random.randint(359)])

    print(cm_conf_set)
    return cm_conf_set
    

def cm_routine(extra_height):
    try:
        file_name = args.file_name_wo_type
        print("file name", file_name)
        cm_conf_list = generate_coverage_map_config_combination(
            BASE_PATH + 'Bl_building_npy/' + file_name+'.npy')
        exist = True
        for cm_conf in cm_conf_list:
            image_path = BASE_PATH_SIONNA + file_name
            for p in cm_conf:
                image_path = image_path + "_" + str(p)
            if not os.path.isfile(image_path + ".npy"):
                exist = False
                break
        if exist:
            print("all files for %s exists" % file_name)
            return
                
        start_loc = time.time()
        scene = load_scene(BASE_PATH + 'Bl_xml_files/' + file_name + '/' + file_name + '.xml')
        print('load scene time: ', str(time.time() - start_loc))
        
        scene.tx_array = PlanarArray(num_rows=1,
                                  num_cols=1,
                                  vertical_spacing=0.5,
                                  horizontal_spacing=0.5,
                                  pattern=args.antenna_pattern,
                                  polarization="VH")

        # Configure antenna array for all receivers
        scene.rx_array = PlanarArray(num_rows=1,
                                  num_cols=1,
                                  vertical_spacing=0.5,
                                  horizontal_spacing=0.5,
                                  pattern="iso",
                                  polarization="V")
        print("V")
        # loop to compute all cm
        for cm_conf in cm_conf_list:
            image_path = BASE_PATH_SIONNA + file_name
            for p in cm_conf:
                image_path = image_path + "_" + str(p)
            if os.path.isfile(image_path + ".npy"):
                print("Skipping existing file")
                continue
            start_loc = time.time()
            try:
                scene.remove("tx")
            except Exception as e:
                print(e)
                pass
            
            # Add a transmitter
            tx = Transmitter(name="tx",
                             position=cm_conf[0:-1],  # cm_conf = [x, y, z, angle]
                             orientation=[cm_conf[-1] / 180 * np.pi, 0, 0]) # deg to rad
            scene.add(tx)

            scene.frequency = 3.66e9 # in Hz; implicitly updates RadioMaterials
            scene.synthetic_array = True 
            # If set to False, ray tracing will be done per antenna element (slower for large arrays)

            # Compute coverage map
            
            cm_only_start = time.time()
            cm = scene.coverage_map(max_depth=8, cm_center=[cm_conf[0], cm_conf[1], extra_height], 
                                    cm_orientation=[0, 0, 0],
                                    cm_cell_size=[args.cm_cell_size, args.cm_cell_size], 
                                    cm_size=[512, 512], los=True, reflection=True, diffraction=True, 
                                    num_samples=args.cm_num_samples, check_scene=False)
            # does not check the scene. 
            
            print('compute cm only time: ', str(time.time() - cm_only_start))
                
            print("images path",image_path)
            
            cm_tensor = cm.as_tensor()
            cm_2D = cm_tensor.numpy()[0, :, :]

            # saving as power and do a flip with axis 0
            np.save(image_path, np.flip(cm_2D,0))
            
            print('compute cm whole time: ', str(time.time() - start_loc))
    except Exception as e:
        raise e


try:
   
    cm_routine(extra_height=args.extra_height)
    
    print('index ' + str(args.outer_idx), args.file_name_wo_type + ' DONE\n')
except Exception as e:
    print(e)
    raise e