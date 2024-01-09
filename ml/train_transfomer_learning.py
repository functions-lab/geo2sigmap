import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from evaluate_rt import evaluate
from unet.unet_model_rt import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from RMSLELoss import RMSLELoss

from utils.data_loading_rt import RTDataset, RTDataset_data_aug
import matplotlib.pyplot as plt
# dir_img = Path('../res')
# dir_mask = Path('./coverage_maps')

from torch.masked import masked_tensor, as_masked_tensor

from sklearn.model_selection import train_test_split


from torch.utils.tensorboard import SummaryWriter

from datetime import datetime

import numpy as np


dir_checkpoint = Path('./checkpoints_transfer_tmp_RXV/')

#building_height_map_dir = os.path.abspath('/dev/shm/res_plane/Bl_building_npy')
terrain_height_map_dir = os.path.abspath('/dev/shm/res_plane/Bl_terrain_npy')
#ground_truth_signal_strength_map_dir = os.path.abspath('/dev/shm/coverage_maps_data_aug_Jul18/')
sparse_ss_dir = Path('/home/yl826/3DPathLoss/nc_raytracing/cm_512_Aug10_7e6_isoTx_PointCloud')

#building_height_map_dir = os.path.abspath('/home/yl826/res_plane/Bl_building_npy')
#ground_truth_signal_strength_map_dir = os.path.abspath('/home/yl826/3DPathLoss/nc_raytracing/cm_512_Oct5_7e6_tr38901Tx_randAngle')
#ground_truth_signal_strength_map_dir = os.path.abspath('/home/yl826/3DPathLoss/nc_raytracing/cm_512_Oct9_RXV_7e6_tr38901Tx_randAngle/')
def linear2dB(x):
    
    res = 10 * np.log10(x)
    res[res == np.nan] = -160
    return res




def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        pathloss = False,
        pathloss_multi_modality = False,
        start_epoch = 0,
        median_filter_size = 0,
        loss_alpha = 0.9,
        ss_num = 0
):
    
    # 0. Ignore the numpy warning
    np.seterr(divide = 'ignore') 
    
    
    # 1. Create dataset

    dataset = RTDataset(building_height_map_dir, terrain_height_map_dir,ground_truth_signal_strength_map_dir,
                        sparse_ss_dir, pathloss = pathloss, median_filter_size = median_filter_size, 
                        ss_num=ss_num, transfer_learning_input = transfer_learning_map_dir)
    

    
        

    # 2. Split into train / validation partitions
    
    
    # n_val = int(len(dataset) * val_percent)
    # n_train = len(dataset) - n_val
    # train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    
    # This is a custom implementation of random split in order to avoid the data leak when using 
    # our data augmentation 
    ids = dataset.get_ids()
    print("# of Total Ids: ",len(ids))
    file_id_set = set([ int(ids_file_name.split("_")[0]) for ids_file_name in ids])
    print("# of Total Ori Index: ",len(file_id_set))
    train_set_ori_idx, val_set_ori_idx = train_test_split(list(file_id_set), test_size=0.2, random_state=42)
    
    print("# of Train Ori Index: ",len(train_set_ori_idx))
    print("# of Val Ori Index: ",len(val_set_ori_idx))
    
    train_set_idx = []
    val_set_idx = []
    for cur_idx, cur_ids in enumerate(ids):
        if int(cur_ids.split("_")[0]) in train_set_ori_idx:
            train_set_idx.append(cur_idx)
        else:
            val_set_idx.append(cur_idx)
            
    n_train = len(train_set_idx)
    n_val = len(val_set_idx)
    print("# of Train Index: ",len(train_set_idx))
    print("# of Val Index: ",len(val_set_idx))
    
    #Do a sanity check
    tmp_train_idx_set = set([ int(ids[tmp_idx].split("_")[0]) for tmp_idx in train_set_idx])
    tmp_val_idx_set = set([ int(ids[tmp_idx].split("_")[0]) for tmp_idx in val_set_idx])
    #print(tmp_val_idx_set)
    print("Check this",set(tmp_val_idx_set).intersection(tmp_train_idx_set))
    
    train_set = torch.utils.data.Subset(dataset, train_set_idx)
    val_set = torch.utils.data.Subset(dataset, val_set_idx)
    
    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(RTDataset_data_aug(train_set, True,True), shuffle=True, **loader_args)
    val_loader = DataLoader(RTDataset_data_aug(val_set, False,False), shuffle=False, drop_last=True, **loader_args)
   
    # (Initialize logging)
    experiment = wandb.init(project='U-Net_RT', resume='allow', anonymous='must', sync_tensorboard=True)
    experiment.config.update(
        {"epochs":epochs, 
             "batch_size":batch_size, 
             "learning_rate":learning_rate,
             "val_percent":val_percent, 
             "save_checkpoint":save_checkpoint, 
             "img_scale":img_scale, 
             "amp":amp, 
             "samples_size":len(dataset),
             "Pre-processing wiht Path Loss Model":pathloss_multi_modality,
             "Multi-modality with Path Loss Model":pathloss,
             "tensorboard_log_dir":os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S")),
             "median_filter_size":median_filter_size,
             "loss_alpha": loss_alpha
             
        }
    )
    

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Samples size:    {len(dataset)}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
        Pre-processing wiht Path Loss Model: {pathloss}
        Multi-modality with Path Loss Model: {pathloss_multi_modality}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.Adam(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, foreach=True)
    
    # optimizer = optim.RMSprop(model.parameters(),
    #                           lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5,eps=1e-9)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.MSELoss()
    global_step = 0

    
    writer = SummaryWriter(experiment.config.tensorboard_log_dir)
    
    log_model_arch = True
    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img', leave=False) as pbar:
            for batch in train_loader:
                images, true_masks_cpu = batch['combined_input'], batch['ground_truth']
                sparse_ss = batch['sparse_ss']
                
                namee = batch['file_name']
                
                
                if (sparse_ss[:,:,0]<0).any() or (sparse_ss[:,:,1]<0).any():
                    print("Here is the dataset", namee)
            
                
                # assert images.shape[1] == model.n_channels, \
                #     f'Network has been defined with {model.n_channels} input channels, ' \
                #     f'but loaded images have {images.shape[1]} channels. Please check that ' \
                #     'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks_cpu.to(device=device, dtype=torch.long)
                sparse_ss = sparse_ss.to(device=device, dtype=torch.float32)
                if log_model_arch:
                    log_model_arch = False
                    writer.add_graph(model, [images, sparse_ss])

                    writer.close()
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    coverage_map_pred = model(images, sparse_ss )
 
                    # RMSE instead of MSE since the data is spread
                    # Only compute the loss on out door points
                    
                    # compute the mask of building area
                    building_mask = images.clone().detach().squeeze(1)
                    
                    
                    building_mask = building_mask[:,0,:,:].squeeze(1)
                    #print(building_mask.size())
                    building_mask[building_mask != 0] = 1
                    building_mask = building_mask.bool()
                    #print(building_mask)
                    #print(true_masks.size())
                    
                    
#                     coverage_map_pred_cpu = coverage_map_pred.clone().detach().cpu().squeeze(1).numpy()
                    
                    
#                     pred_sparese_ss = []
#                     #print(coverage_map_pred_cpu.shape)
#                     for idx, sparse_ss_one in enumerate(sparse_ss):
                        
#                         ttmp_pred_sparese_ss = []
#                         for poitn in sparse_ss_one:
#                             point_x = int(poitn[0])
#                             if point_x == 100:
#                                 point_x = 99
#                             point_y = int(poitn[1])
#                             if point_y == 100:
#                                 point_y = 99
#                             ttmp_pred_sparese_ss.append(coverage_map_pred_cpu[idx,point_x,point_y])
#                         pred_sparese_ss.append(ttmp_pred_sparese_ss)
#                     pred_sparese_ss_slow = torch.from_numpy(np.array(pred_sparese_ss))

                    
                    batch_indices = torch.arange(coverage_map_pred.shape[0]).unsqueeze(1)
                    pred_sparese_ss = coverage_map_pred[batch_indices, :, torch.clamp(sparse_ss[:, :, 0], min=0, max=127.999).int(), torch.clamp(sparse_ss[:, :, 1], min=0, max=127.999).int()]
                    pred_sparese_ss = pred_sparese_ss.squeeze()
                    #print(sparse_ss[:,:,2])
                    # if not torch.equal(pred_sparese_ss, pred_sparese_ss_slow.to(device)):
                    #     print(torch.equal(pred_sparese_ss, pred_sparese_ss_slow.to(device)))
                    
#                     for batch in range(64):
#                         for idxx in range(200):
#                             if pred_sparese_ss[batch,idxx] != pred_sparese_ss_slow[batch,idxx]:
#                                 print(batch, idxx)
#                                 print(sparse_ss[batch,idxx])
#                                 print(sparse_ss[batch,idxx][0].int(),sparse_ss[batch,idxx][1].int())
                                
#                                 print(coverage_map_pred[idx, :,sparse_ss[batch,idxx][0].int(),sparse_ss[batch,idxx][1].int()])
#                                 print(pred_sparese_ss[batch,idxx])
#                                 print(pred_sparese_ss_slow[batch,idxx])
#                                 print(namee[batch])
                    
                    
                    # print("pred_sparese_ss ", pred_sparese_ss.shape)
                    # print("sparse_ss ", sparse_ss[:,:,2].shape)
                        
#                     pred_sparese_ss = np.array(pred_sparese_ss)
                    #print("pred_sparese_ss Size: ",pred_sparese_ss.shape)
                    #print("sparese_ss Size: ",sparse_ss[:,:,2].shape)
                    
                    
                    
                    bl_mask = images[:,0,:,:].clone()
                    #print(building_mask.size())
                    bl_mask[bl_mask != 0] = 1
                    bl_mask = bl_mask.bool()
                    cm_loss_2 = (criterion(coverage_map_pred.squeeze(1), true_masks.float()))
                    # print(cm_loss_2)
                    # print(bl_mask.size())
                    # print(coverage_map_pred.size())
                    # print(torch.sum(bl_mask))

                    cm_loss =  (torch.sum(((coverage_map_pred.squeeze(1)-true_masks.float())* bl_mask)**2.0)  / torch.sum(bl_mask))
                    
            
                    
                    sparse_point_loss = torch.sqrt(criterion(pred_sparese_ss, sparse_ss[:,:,2]))
                    
                    #loss =  (1 - loss_alpha) *  cm_loss  + sparse_point_loss * loss_alpha
                    loss = 0.7 * cm_loss_2 + 0.3 * cm_loss
                

                    # print(true_masks.type())
                    # print(coverage_map_pred.type())
                    # print(bl_mask.type())
                    # print(coverage_map_pred.squeeze(1).size())
                    # print(coverage_map_pred.squeeze(1))
                    # print(torch.sum(torch.abs(coverage_map_pred.squeeze(1))))
                    # print(torch.abs(coverage_map_pred.squeeze(1)).type())
                    # print(torch.sum(torch.abs(coverage_map_pred.squeeze(1))).type())
                    # print('++++++++++++++++++++++++++++')
                    # print(true_masks.float().size())
                    # print(true_masks.float())
                    # print(torch.sum(torch.abs(coverage_map_pred.squeeze(1)-true_masks.float())))
                    # print('-----------------------------------')
                    # print(bl_mask.size())
                    # print(bl_mask)
                    # print(torch.sum(torch.abs(torch.multiply((coverage_map_pred.squeeze(1)-true_masks.float()), bl_mask))))
                    # print(torch.sum(torch.abs(((coverage_map_pred.squeeze(1)-true_masks.float())* bl_mask)**2.0)))
                    # print('...............................')
                    # exit()


                    #print("loss: ",cm_loss)
 
                    
                    #print(sparse_point_loss)
                    



                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'train cm(U-Net) loss': cm_loss.item(),
                    'train sparse_point_loss':sparse_point_loss.item(),
                    'step': global_step,
                    'epoch': epoch + start_epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (1 * batch_size))+1
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            none_res = (value.grad is None)
                            for tmp_value in value:
                                
                                if tmp_value.grad is None:
                                    none_res = True
                                    #print(tag)
                                    #print(tmp_value)
                                    #print(tmp_value.grad)
                                    #print()
                                    break
                            if none_res:
                                continue
                                
                                
                            if not(torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())
                                            # compute the mask of building area

                        val_score, outdoor_score = evaluate(model, val_loader, device, amp)
                        #scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        logging.info('Outdoor score: {}'.format(outdoor_score))

                        coverage_map_pred_cpu = coverage_map_pred.clone().detach().squeeze(1).cpu().numpy()
                        fig = plt.figure()
                        plt.imshow((coverage_map_pred_cpu[0]),vmin=-110,vmax=0)
                        plt.colorbar()
                        #print(coverage_map_pred_cpu[0])
                        #print((coverage_map_pred_cpu[0]))
                        
                        fig2 = plt.figure()
                        plt.imshow((batch['ground_truth'][0]))
                        plt.colorbar()
                        
                        fig3 = plt.figure()
                        plt.imshow((coverage_map_pred_cpu[0]))
                        plt.colorbar()
                        
                        
                        # Apply the mask on coverage_map_pred
                    
                        masked_coverage_map_pred = coverage_map_pred.clone().squeeze(1)
                        #print(masked_coverage_map_pred.size())
                        #print(building_mask.size())
                        masked_coverage_map_pred[masked_coverage_map_pred < -160] = -160
                        masked_coverage_map_pred[building_mask] = true_masks.float()[building_mask]
                        
                        masked_coverage_map_pred_cpu = masked_coverage_map_pred.clone().detach().squeeze(1).cpu().numpy()
    
                        fig4 = plt.figure()
                        plt.imshow((masked_coverage_map_pred_cpu[0]),vmin=-110,vmax=0)
                        plt.colorbar()
                
                        
                        fig5 = plt.figure()
                        plt.imshow(images[0,1].cpu())
                        plt.colorbar()
                        plt.title(batch['file_name'][0])
                        
                        
                        fig6 = plt.figure()
                        tttmp = images[0,2].cpu().numpy()
                        plt.imshow((tttmp))
                        
                        #print(np.sum(tttmp[tttmp != 0.0]))
                        plt.colorbar()


                        fig7 = plt.figure()
                        ss_map = images[0,1].cpu().numpy()
                        plt.imshow((tttmp))
                        #plt.title(batch['file_name'][0])
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Loss': val_score,
                                'validation Outdoor Loss': outdoor_score,
                                'Inputs':{
                                    'Building Height Map': wandb.Image(images[0,0].cpu()),
                                    'Path Loss Model Pred': wandb.Image(fig6),
                                    'TX Position Map': wandb.Image(fig5),
                                    'Coverage Map(Ground Truth)': wandb.Image(fig2),
                                },
                                'Outputs': {
                                    'Pred': wandb.Image(fig),
                                    'Pred_auto_range': wandb.Image(fig3),
                                    'Masked_pred': wandb.Image(fig4),
                                    #'Building_mask': wandb.Image(building_mask[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch + start_epoch,
                                **histograms
                            })
                            plt.close('all')
                        except Exception as e:
                            print(e)
                            pass


        Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
        state_dict = model.state_dict()
        #state_dict['mask_values'] = dataset.mask_values
        torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
    #writer.close()

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=64, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--pathloss',action='store_true', default=False , help='Using the Path Loss Model(3GPP TR 38.901) to pre process the input TX information')
    parser.add_argument('--pathloss_multi_modality',action='store_true', default=False , help='Apply Path Loss Model(3GPP TR 38.901) on the end of model forward.')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='Logging wandb with the start of epoch x', dest='start_epoch')
    parser.add_argument('--median-filter-size', type=int, default=0,
                        help='Applying median filter on the ground truth with filter size', dest='median_filter_size')
    parser.add_argument('--loss-alpha', type=float, default=0,
                        help='Loss = (1 - loss_alpha) * cm_loss + loss_alpha * sparse_loss', dest='loss_alpha')
    parser.add_argument('--sparse-point', type=int, default=0,
                        help='Using sparse point data to refine the result (PointNet). ', dest='ss_num')
    parser.add_argument('--building-height-map-dir', type=str, required=True,help="The 2D builing maps dir.")

    parser.add_argument('--ground-truth-dir', type=str, required=True,help="The ground truth signal coverage map dir.")
    parser.add_argument('--transfer-learning-map-dir', type=str, required=True,help="The Path Gain map dir generate by the first U-Net.") 
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    building_height_map_dir = args.building_height_map_dir
    ground_truth_signal_strength_map_dir = args.ground_truth_dir
    #"/home/yl826/3DPathLoss/nc_raytracing/Pytorch-UNet-master/prediction_result"
    transfer_learning_map_dir = args.transfer_learning_map_dir
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=3, n_classes=1, bilinear=args.bilinear,pathloss=args.pathloss_multi_modality)
    model = model.to(memory_format=torch.channels_last)



    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:

        
        state_dict = torch.load(args.load, map_location=device)
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            pathloss=args.pathloss,
            pathloss_multi_modality=args.pathloss_multi_modality,
            start_epoch=args.start_epoch,
            median_filter_size = args.median_filter_size,
            loss_alpha = args.loss_alpha,
            ss_num = args.ss_num
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
