import subprocess
import concurrent
from concurrent.futures import wait, as_completed
import os
import time
import multiprocessing
from PIL import Image
#import tensorflow as tf
import numpy as np
from tqdm import tqdm
from datetime import datetime
import logging
import argparse
# Import Sionna RT components
#from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera, Paths2CIR





def initializer_func(gpu_seq_queue: multiprocessing.Queue, log_level: int) -> None:
    """
    This is a initializer function run after the creation of each process in ProcessPoolExecutor, 
    to set the os env variable to limit the visiablity of GPU for each process inorder to achieve 
    the load balance bewteen diff GPU
    :gpu_seq_queue This is a queue storing the GPU ID as a token, each process will only get 
    """
    import os
    
    gpu_id = gpu_seq_queue.get()
    print("Initlizing the process: %d with GPU: %d"%(os.getpid(),gpu_id))
    
    # Configure visible GPU 
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"
    return


def count_frequency(my_list):
    freq = {}
    for item in my_list:
        if (item in freq):
            freq[item] += 1
        else:
            freq[item] = 1
    return freq


if __name__ == '__main__':




    # Create a logger
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG)  # Set the logging level

    console_handler = logging.StreamHandler()  # By default, this directs logs to stdout
    console_handler.setLevel(logging.INFO)  # Set the handler's logging level

    # Create a formatter and set it on the handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    # # Add the file handler to the logger
    logger.addHandler(console_handler)

    parser = argparse.ArgumentParser(description='Process latitude and longitude ranges.')

    # Adding arguments
    parser.add_argument('--extra-height', default=2, type=int, help='Extra height for Rx above the ground.')
    parser.add_argument('--cm-resolution', default=4, type=int, help='Coverage map cell size(meter).')
    parser.add_argument('--cm-sample-num', default='7e6', type=str, help='Number of samples when computing the coverage map.')
    parser.add_argument('--tx-antenna-pattern', default='tr38901', type=str, help='Antenna pattern for TX.')
    parser.add_argument('--data-dir', type=str, help='Data folder', required=True)
    
    parser.add_argument('--rx-antenna-pol',choices=['V','H','cross','VH'], default='cross', help='RX antenna polarization.')
    parser.add_argument('--tx-antenna-pol',choices=['V','H','cross','VH'], default='cross', help='TX antenna polarization.')

    parser.add_argument('--start-idx', default=0, type=int, help='The index of area which you prefer to start the process.')
    parser.add_argument('--stop-idx', type=int, default=80,help='The index of area which you prefer to stop the process. -1 means no stop.')
    parser.add_argument('--max-process', type=int, default=5, help='Maximum process used for generate data.')
    parser.add_argument('--num-gpu', type=int, default=1, help='Number of GPU used by Sionna.')

    
    #TODO
    #add a argument to specified the keep working dir 

    # Parse the arguments
    args = parser.parse_args()


    EXTRA_HEIGHT = args.extra_height  # height of Rx above ground. 
    CM_RESOLUTION = args.cm_resolution  # size in meters of each pixel in the coverage map
    ANTENNA_PATTERN = args.tx_antenna_pattern  # 'tr38901' also possible


    # testing new Blender_command_line function written on 23. Jun 2023
    BASE_PATH_BLENDER = args.data_dir

    CM_NUM_SAMPLES = args.cm_sample_num  # input for the num_samples in scene.coverage_map (sionna method), default 2e6
    print()

    BASE_PATH_SIONNA = os.path.join(BASE_PATH_BLENDER,"{}_RX{}_TX{}-{}_SampleNum{}".format(datetime.now().strftime("%b%d"),args.rx_antenna_pol,args.tx_antenna_pattern,args.tx_antenna_pol, args.cm_sample_num) ) 
    
    
    # '/home/yl826/3DPathLoss/nc_raytracing/cm_512_Oct9_RXV_' + \
    #                 str(int(CM_NUM_SAMPLES / (1e6))) + 'e6_' + ANTENNA_PATTERN + 'Tx_randAngle/'

    os.makedirs(BASE_PATH_SIONNA, exist_ok=True)

    # START_FROM_IDX = 512


    STOP_AT_IDX = args.start_idx
    STOP_AT_IDX = args.stop_idx

    NUM_OF_PROCESS = args.max_process
    NUMBER_OF_GPU = args.num_gpu






    # this gets the idx_uuid
    f_names_xml = [f for f in os.listdir(os.path.join(BASE_PATH_BLENDER,'Bl_xml_files/'))
                   if os.path.isdir(os.path.join(BASE_PATH_BLENDER,'Bl_xml_files/', f))]
    print('Number of xml files:', len(f_names_xml))
    
    # f[0:-4] to remove the ".npy" from file name, 
    # but we now have _xCoord_yCoord_heightVal_angleDegrees append after the idx_uuid, 
    # so we do this instead:
    f_names_sig_map_idx_uuid = list(set([f.split('_')[0] + '_' + f.split('_')[1]
                                          for f in os.listdir(BASE_PATH_SIONNA)
                                          if os.path.isfile(os.path.join(BASE_PATH_SIONNA, f))]))
    
    freq_count_sig_map_idx_uuid = count_frequency([f.split('_')[0] + '_' + f.split('_')[1] 
                                         for f in os.listdir(BASE_PATH_SIONNA)
                                         if os.path.isfile(os.path.join(BASE_PATH_SIONNA, f))])
    # maps idx_uuid to number of appearances of idx_uuid in BASE_PATH_SIONNA
    
    completed_idx_uuid = set([f for f in freq_count_sig_map_idx_uuid 
                              if freq_count_sig_map_idx_uuid[f] == 16])
    # idx_uuid for which there exists all 16 coverage maps. 
    
    for idx_uuid_temp in f_names_xml:  # adding key-value pairs for non-existent files
        if idx_uuid_temp not in freq_count_sig_map_idx_uuid.keys():
            freq_count_sig_map_idx_uuid[idx_uuid_temp] = 0

    futures = []
    
    # Create a GPU ID token Queue
    gpu_seq_queue = multiprocessing.Queue()

    for i in range(NUM_OF_PROCESS):
        gpu_seq_queue.put(i%NUMBER_OF_GPU)
    
    # Init pbar
    pbar = tqdm(total=len(set(f_names_xml) ^ completed_idx_uuid), desc='Sionna RT')
    count = 0
    # Init process pool executor
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=NUM_OF_PROCESS, initializer=initializer_func, initargs=(gpu_seq_queue,1)) as executor:
        for idx, f_name_xml in enumerate(f_names_xml):
            if idx > STOP_AT_IDX: 
                break
            if f_name_xml not in completed_idx_uuid: 
                # skip cmaps that have already been generated by considering 
                # only the idx_uuid part of generated maps' names. Avoids unnecessary 
                # and costly subprocesses for coverage maps that have already been computed. 
                print(f_name_xml, 'count: ' + str(freq_count_sig_map_idx_uuid[f_name_xml]))
                """
                Creating a subprocess for each job running in process pool
                This is the simples way I can find to free up the GPU memory 
                and do a load balance betwenn each GPU
                """
                # NOTE: CHANGE THE NAME OF THE FILES SO THAT THE random angle file
                # has the correct name. Check this file, xml_to_heatmap_wrapper512_tr38901, 
                # xml_to_heatmap_one_run512_tr38901_randAngle, and 
                # xml_to_heatmap_one_run512_tr38901_randAngle
                futures.append(executor.submit(subprocess.run,
                       ['python3', os.path.join(os.path.dirname(__file__),'xml_to_heatmap_one_run512_tr38901.py'),
                        '--file_name_wo_type', str(f_name_xml),
                        '--extra_height', str(EXTRA_HEIGHT),
                        '--cm_cell_size', str(CM_RESOLUTION),
                        '--BASE_PATH_BLENDER', str(BASE_PATH_BLENDER),
                        '--BASE_PATH_SIONNA', str(BASE_PATH_SIONNA),
                        '--outer_idx', str(idx), 
                        '--cm_num_samples', str(CM_NUM_SAMPLES), 
                        '--antenna_pattern', str(ANTENNA_PATTERN)],
                         capture_output=True, text=True))
                # print(' '.join(
                #             ['python', os.path.join(os.path.dirname(__file__),'xml_to_heatmap_one_run512_tr38901.py'),
                #             '--file_name_wo_type', str(f_name_xml),
                #             '--extra_height', str(EXTRA_HEIGHT),
                #             '--cm_cell_size', str(CM_RESOLUTION),
                #             '--BASE_PATH_BLENDER', str(BASE_PATH_BLENDER).replace(' ', '\ '),
                #             '--BASE_PATH_SIONNA', str(BASE_PATH_SIONNA).replace(' ', '\ '),
                #             '--outer_idx', str(idx), 
                #             '--cm_num_samples', str(CM_NUM_SAMPLES), 
                #             '--antenna_pattern', str(ANTENNA_PATTERN)]
                #         ))
        for idx, future in enumerate(as_completed(futures)):
            pbar.update(n=1)  
            try:
                data = str(future.result()).replace('\\n','\n')
                print('\n\n\n\n\n' + str(idx) + '\n' + data + '\n\n\n\n\n')
            except Exception as err:
                print(err)
    print('DONE')
    
