""" Full assembly of the parts to form the complete network """

from .unet_parts import *
import torch.nn as nn
from .pointnet_model import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, pathloss=False):
        super(UNet, self).__init__()

        #self.pointnet = PointNetfeat(global_feat=True, feature_transform=False)
        
        #self.pointnet = PointNetDenseCls(feature_transform=True)
        

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.pathloss = pathloss


        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear, pointnet_in_channels=-1))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))


        #VAE
        # self.fc_mu = nn.Linear(1024, 10)
        # self.fc_var = nn.Linear(1024, 10)
        # self.decoder_input = nn.Linear(10, 1024)
        
    def forward(self, x, sparse_ss):

        back_x = x.clone().detach()
        sparse_ss = sparse_ss.transpose(2, 1)
        #sparse_ss, _, _ = self.pointnet(sparse_ss)




        #rint(sparse_ss.shape)
        x = x[:,0:self.n_channels:,:]  
        # change 2 or 3 for iso U-Net (should be 2) or directional U-Net (should be 3)
        #print(x.size())
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
    
        sparse_ss = torch.flatten(sparse_ss, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        # mu = self.fc_mu(sparse_ss)
        # log_var = self.fc_var(sparse_ss)
        # """
        # Reparameterization trick to sample from N(mu, var) from N(0,1).
        # :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        # :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        # :return: (Tensor) [B x D]
        # """
        # std = torch.exp(0.5 * log_var)
        # eps = torch.randn_like(std)
        # z = eps * std + mu
        # sparse_ss = self.decoder_input(z)





        #print(x5.shape)
        # pad_sparse_ss = torch.zeros(sparse_ss.shape[0], 8*8*16).to(x)
        # pad_sparse_ss[:,:1024] = sparse_ss
        # pad_sparse_ss = torch.reshape(pad_sparse_ss, (sparse_ss.shape[0],16,8,8))
        # ## #pad_sparse_ss = torch.reshape(sparse_ss[:,:28*6*6], (sparse_ss.shape[0],28,6,6))
        # x5 = torch.cat((x5, pad_sparse_ss), 1)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        # print(logits.size())
        # print(back_x.size())
        # print(back_x[:,1,:,:].reshape(logits.size()).size())
        # #exit()
        #print("outputsss: ",logits)

        if self.pathloss:
            return logits + back_x[:,2,:,:].reshape(logits.size())
        else:
            return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
