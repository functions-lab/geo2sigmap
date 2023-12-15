import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm





from scipy.constants import speed_of_light
import numpy as np
import matplotlib.pyplot as plt


import math
import os

from scipy import ndimage




import subprocess
import concurrent
from concurrent.futures import wait
import os
from os.path import join, dirname
import random





NUM_OF_POINTS = 200
AREA_SIZE = 512
RESOLUTION = 4
IMAGE_SIZE = int(AREA_SIZE / RESOLUTION)


class RTDataset_data_aug(Dataset):
    def __init__(self, subset, data_aug:bool = False):
        self.subset = subset
        self.data_aug = data_aug
        
    def __getitem__(self, index):
        
        combined_input, ground_truth_arr, name, sparse_ss_arr = self.subset[index]
        
 

        # if self.data_aug:
        #     if random.random() < 0.5:
        #         combined_input = np.fliplr(combined_input)
        #         ground_truth_arr = np.fliplr(ground_truth_arr)
        #         #print("flip! ", ground_truth_arr.shape[-1] - 1)
        #         for point in sparse_ss_arr:
        #             point[1] = ground_truth_arr.shape[-1] - 1 - point[1]
        
        
        
        return {
            'combined_input': torch.as_tensor(combined_input.copy()).float().contiguous(),
            'ground_truth': torch.as_tensor(ground_truth_arr.copy()).long().contiguous(),
            'file_name': name,
            'sparse_ss': torch.as_tensor(sparse_ss_arr.copy()).float().contiguous()
        }
    
        
    def __len__(self):
        return len(self.subset)

class RTDataset(Dataset):
    def __init__(self, building_height_map_dir: str, terrain_height_map_dir: str,
                 ground_truth_signal_strength_map_dir: str,sparse_ss_dir: str, pathloss: bool = False, median_filter_size: int = 0, transform = None):
        
        np.seterr(divide = 'ignore') 
        self.building_height_map_dir = Path(building_height_map_dir)
        self.terrain_height_map_dir = Path(terrain_height_map_dir)
        self.ground_truth_signal_strength_map_dir = Path(ground_truth_signal_strength_map_dir)
        
        self.sparse_ss_dir = Path(sparse_ss_dir)
        
        self.pathloss = pathloss
        
        self.median_filter_size = median_filter_size
        
        
        





        ids_gt = [splitext(file)[0] for file in listdir(self.ground_truth_signal_strength_map_dir) if
                  isfile(join(self.ground_truth_signal_strength_map_dir, file)) and not file.startswith(
                      '.')]
        ids_building = [splitext(file)[0].split("_")[0] for file in listdir(self.building_height_map_dir) if
                        isfile(join(self.building_height_map_dir, file)) and not file.startswith(
                            '.')]
        # ids_terrain = [splitext(file)[0].split("_")[0] for file in listdir(self.terrain_height_map_dir) if
        #                isfile(join(self.terrain_height_map_dir, file)) and not file.startswith(
        #                    '.')]
        
        self.ids = ids_gt
        
        
        # Here is a work around of tf variable length length in a single bath problem
        filtered_ids = []
        for file in tqdm(ids_gt, desc="Checking sparse points size"):
            tmp = np.load(os.path.join(self.sparse_ss_dir,file.split("\\")[-1]+".npy"))
            #print(tmp.shape)

            if len(tmp) >= NUM_OF_POINTS:
                filtered_ids.append(file)

        self.ids = filtered_ids  
        if not self.ids:
            raise RuntimeError(
                f'No input file found in {self.ground_truth_signal_strength_map_dir}, make sure you put your images there')

    def __len__(self):
        return len(self.ids)

    def get_ids(self):
        return self.ids

    @staticmethod
    def uma_los(d3d, d2d, dbp, fc, h_b, h_t):
        # 38.901 UMa LOS
        PL1 = 28+22*np.log10(d3d)+20*np.log10(fc)
        PL2 = 28+40*np.log10(d3d)+20*np.log10(fc) - 9*np.log10(dbp**2+(h_b - h_t)**2)
        PL = np.zeros((d3d.shape))
        PL = PL2 # Default pathloss
        PL[(np.greater_equal(d2d,10) & np.less_equal(d2d,dbp))] = PL1[(np.greater_equal(d2d,10) & np.less_equal(d2d,dbp))] # Overwrite if distance is greater than 10 meters or smaller than dbp
        return PL
    
    
    @staticmethod
    def uma_nlos(d3d, d2d, dbp, fc, h_b, h_t):
        # 38901 UMa NLOS
        PL_nlos = 13.54+39.08*np.log10(d3d)+20*np.log10(fc)-0.6*(h_t-1.5)
        PL = np.zeros((d3d.shape))
        PL = np.maximum(RTDataset.uma_los(d3d, d2d, dbp, fc, h_b, h_t), PL_nlos)
        return PL
    
    @staticmethod
    def pathloss_38901(distance, frequency, h_bs=30, h_ut=1.5):
        #print(distance)
        """
            Simple path loss model for computing RSRP based on distance.

            fc: frequency in GHz
            h_b: height of basestation
            h_t: height of UT
        """
        # Constants
        fc = frequency
        h_b =  h_bs # 30 meters
        h_t =  h_ut # 1.5

        # 2D distance 
        d2d = distance

        # 3D distance
        h_e = h_b - h_t # effective height
        d3d = np.sqrt(d2d**2+h_e**2)

        # Breakpoint distance
        dbp =  4*h_b*h_t*fc*10e8/speed_of_light

        loss = RTDataset.uma_nlos(d3d, d2d, dbp, fc, h_b, h_t)
        return loss

    def __getitem__(self, idx):
        
        name = self.ids[idx]
        

        name_splited = name.split("_")
        file_name_id_part = name_splited[0]
        tx_height = name_splited[-1]
        tx_x = int(name_splited[-3])+500
        tx_y = (-1 * int(name_splited[-2]))+500
        tx_position = [tx_x // 10, tx_y // 10]
        distance = np.arange(0, 1450,1)
        path_loss_res =  RTDataset.pathloss_38901(distance, 3.60, h_bs=int(tx_height), h_ut=2)


      
    

        # Ori image size 1040 * 1040, crop to 1000 * 1000
        # building_height_arr = np.load(os.path.join(self.building_height_map_dir, name_splited[0]+"_"+name_splited[1]+".npy"))[4:104,4:104]
        
        
        # Ori image size 1000*1000
        building_height_arr = np.load(os.path.join(self.building_height_map_dir, name_splited[0]+"_"+name_splited[1]+".npy"))
        
        # Crop to 512*512 based on location information coded in the file name.
        building_height_arr = building_height_arr[tx_y-256:tx_y+256,tx_x-256:tx_x+256]
        
        # Resize to 128*128 based on the resolution of 4m.
        building_height_arr = building_height_arr[::4, ::4]
        
        
        ground_truth_arr = np.load(os.path.join(self.ground_truth_signal_strength_map_dir , name+".npy"))
        


        
        if self.median_filter_size != 0:
            ground_truth_arr = ndimage.median_filter(ground_truth_arr, size = self.median_filter_size)

        sparse_ss_arr = np.load(os.path.join(self.sparse_ss_dir, name+".npy"))
        
        
#         # Generate random x, y coordinates within the range [0, 128)
#         x_coordinates = np.random.randint(0, 128, size=(200, 1))
#         y_coordinates = np.random.randint(0, 128, size=(200, 1))

#         # Generate random floats in the range [-160, -80]
#         float_values = np.random.uniform(-160, -80, size=(200, 1))

#         # Combine the arrays to create the final dataset
#         sparse_ss_arr = np.hstack((x_coordinates, y_coordinates, float_values))
        sparse_ss_arr = sparse_ss_arr[~np.any(np.isnan(sparse_ss_arr), axis=1)]
        sparse_ss_arr = sparse_ss_arr[~np.any(np.isinf(sparse_ss_arr), axis=1)]
        choice = np.random.choice(len(sparse_ss_arr), NUM_OF_POINTS, replace=True)
        sparse_ss_arr = sparse_ss_arr[choice, :]
        
        if (sparse_ss_arr[:,2] < -160).any():
            print("Fucked Inputs EXIT!")
            print( np.where(sparse_ss_arr[:,2]<-160))
            for idx in np.where(sparse_ss_arr[:,2]<-160):
                print(idx, sparse_ss_arr[idx,2])

        #sparse_ss_arr[:,2] = 10 * np.log10(sparse_ss_arr[:,2])
        


        #Convert the linear power to dB scale
        ground_truth_arr = 10 * np.log10(ground_truth_arr)


        # ground_truth_arr[ground_truth_arr == np.nan] = -160
        ground_truth_arr[ground_truth_arr == -np.inf] = -160
        ground_truth_arr = np.nan_to_num(ground_truth_arr, nan=0)
        ground_truth_arr[ground_truth_arr >= 0] = 0
        ground_truth_arr[ground_truth_arr <= -160] = -160

         # Construct the TX position channel
        tx_position_channel = np.full((IMAGE_SIZE, IMAGE_SIZE), 0, dtype=int)
        
        # Base station position channel
        #tx_position_channel[tx_position[1]][tx_position[0]] = tx_height
        
        tx_position_channel[63,63] = tx_height





        # Construct the Path Loss Model (3GPP TR 308.91 nLos UMa)
        path_loss_heat_map = np.full((IMAGE_SIZE, IMAGE_SIZE), 0, dtype=float)



        for row in range(path_loss_heat_map.shape[0]):
            for col in range(path_loss_heat_map.shape[1]):
                # Compute the distance between pixel and tx
                dist = math.sqrt((63*RESOLUTION - row*RESOLUTION)**2 + (63*RESOLUTION - col*RESOLUTION)**2)
                tmp = -1 * path_loss_res[int(dist)]
                if np.isinf(tmp):
                    tmp = -50.0
                    #print("Got inf in PL with index: ",row, col)
                path_loss_heat_map[row][col] =  tmp
                

        
            



        # Since right now GT.size is 100*100 and other two size is 1000 * 1000, just check the input.
        # assert building_height_arr.shape == terrain_height_aimport loggin
        combined_input = np.zeros((3, IMAGE_SIZE, IMAGE_SIZE), dtype=float)
        
        
    

            
        
        
        # Combine all the channels together
        combined_input[0,:, :] = building_height_arr 
        combined_input[1,:, :] = tx_position_channel
        combined_input[2,:, :] = path_loss_heat_map 
#         if np.isinf(combined_input).any() or np.isnan(combined_input).any():
            
#             print("this is 63,63 ", combined_input[2,63,63])
            
#             #print(combined_input)
#             print("Fucked! inputs problem!")
#             exit()
         
        
        
        
            
            
        return combined_input, ground_truth_arr, name, sparse_ss_arr

    
    
    


if __name__ == '__main__':
    building_height_map_dir = Path('../../res/Bl_building_npy')
    terrain_height_map_dir = Path('../../res/Bl_terrain_npy')
    ground_truth_signal_strength_map_dir = Path('./coverage_maps')
    sparse_ss_dir = Path('/home/yl826/3DPathLoss/nc_raytracing/jul18_sparse')
    dataset = RTDataset(building_height_map_dir, terrain_height_map_dir, ground_truth_signal_strength_map_dir)
