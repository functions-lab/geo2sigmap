import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
from utils.dice_score import multiclass_dice_coeff, dice_coeff

from RMSLELoss import RMSLELoss
@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    loss = 0
    
    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['combined_input'], batch['ground_truth']
            sparse_ss = batch['sparse_ss']
            
            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)
            sparse_ss = sparse_ss.to(device=device, dtype=torch.float32)
            
            # predict the mask
            coverage_map_pred = net(image, sparse_ss)
            criterion = nn.MSELoss()
            
            building_mask = image.clone().squeeze(1)
            building_mask = building_mask[:,0,:,:].squeeze(1)
            building_mask[building_mask != 0] = 1
            building_mask = building_mask.bool()
            #print(building_mask)
            #print(true_masks.size())

            # Apply the mask on coverage_map_pred
            # masked_coverage_map_pred = coverage_map_pred.squeeze(1)
            # #print(masked_coverage_map_pred.size())
            # #print(building_mask.size())
            # masked_coverage_map_pred[masked_coverage_map_pred < -160] = -160
            # masked_coverage_map_pred[building_mask] = mask_true.float()[building_mask]


            bl_mask = image[:,0,:,:].clone()
            #print(building_mask.size())
            bl_mask[bl_mask != 0] = 1
            
            cm_loss_2 = torch.sqrt(criterion(coverage_map_pred.squeeze(1), mask_true.float()))
            # print(cm_loss_2)
            # print(bl_mask.size())
            # print(coverage_map_pred.size())
            # print(torch.sum(bl_mask))


            # print(mask_true.type())
            # print(coverage_map_pred.type())
            # print(bl_mask.type())
            # print(coverage_map_pred.squeeze(1).size())
            # print(coverage_map_pred.squeeze(1))
            # print(torch.sum(torch.abs(coverage_map_pred.squeeze(1))))
            # print(torch.abs(coverage_map_pred.squeeze(1)).type())
            # print(torch.sum(torch.abs(coverage_map_pred.squeeze(1))).type())
            # print('++++++++++++++++++++++++++++')
            # print(mask_true.float().size())
            # print(mask_true.float())
            # print(torch.sum(torch.abs(coverage_map_pred.squeeze(1)-mask_true.float())))
            # print('-----------------------------------')
            # print(bl_mask.size())
            # print(bl_mask)
            # print(torch.sum(torch.abs(torch.multiply((coverage_map_pred.squeeze(1)-mask_true.float()), bl_mask))))
            # print(torch.sum(torch.abs(((coverage_map_pred.squeeze(1)-mask_true.float())* bl_mask)**2.0)))
            # print('...............................')
            
            # print(mask_true.size())
            # print(bl_mask.size())
            

            ttmp = ((coverage_map_pred.squeeze(1)-mask_true.float())* bl_mask)**2.0
            # print(((coverage_map_pred.squeeze(1)-mask_true.float())* bl_mask)**2.0)
            # print(ttmp.size())
            # print("total:",(torch.sum(((coverage_map_pred.squeeze(1)-mask_true.float())* bl_mask)**2.0)  / torch.sum(bl_mask)))
            # print("first half:",             torch.sum(((coverage_map_pred.squeeze(1)-mask_true.float())* bl_mask)**2.0)    )
            # print("first half wo/*bl_mask:", torch.sum(((coverage_map_pred.squeeze(1)-mask_true.float()))**2.0) )
            # print("sec half:", torch.sum(bl_mask))
        
            # print(11111111111111111111)
            loss +=  torch.sum(((coverage_map_pred.squeeze(1)-mask_true.float())* bl_mask)**2.0)  / torch.sum(bl_mask)
            # print(torch.sum(((coverage_map_pred.squeeze(1)-mask_true.float())* bl_mask)**2.0)  / torch.sum(bl_mask))



            
            dice_score += torch.sqrt(criterion(coverage_map_pred.squeeze(1), mask_true.float()))

            # if net.n_classes == 1:
            #     assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
            #     mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
            #     # compute the Dice score
            #     dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            # else:
            #     assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
            #     # convert to one-hot format
            #     mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
            #     mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
            #     # compute the Dice score, ignoring background
            #     dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
    net.train()
    return dice_score / max(num_val_batches, 1), loss / max(num_val_batches, 1)
