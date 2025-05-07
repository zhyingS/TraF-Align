
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from models.modules.utils.unet_utils import Up,Down,DoubleConv,OutConv

class Navigator(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg
        mapp_out = cfg['model']['backbone']['downsample']

        field_dim=cfg['model']['deform']['field_dim']
        self.in_layer = DoubleConv(mapp_out,16)
        self.inc = (DoubleConv(16, 16))
        self.down1 = (Down(16, 32))
        self.down2 = (Down(32, 64))
        self.up1 = (Up(64, 32))
        self.up2 = (Up(32, 16))

        stride = cfg['model']['deform']['input_stride']
        heads = []
        for i in range(len(stride)):
            head_layer = nn.Sequential(
                DoubleConv(16, 16, stride=2**i, mid_channels=16),
                OutConv(16, field_dim),
                nn.Sigmoid())
            heads.append(head_layer)
        self.heads = nn.ModuleList(heads)
        # self.init_weight()
        
    def init_weight(self):
        for m in self.modules():
            if isinstance(m,(nn.Conv2d,nn.Linear)):
                self.init_weight_(m)
            if isinstance(m,(nn.ModuleList,nn.Sequential)):
                m.apply(self.init_weight_)

    def init_weight_(self,m):
        if isinstance(m,(nn.Conv2d,nn.Linear)):
            nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
                
        return self.to_field_(x)

    def to_field_(self,x):
        
        x = self.in_layer(x)
        
        x_traj = self.unet(x)
        
        x_trajs = []
        for layer in self.heads:
            x_trajs.append(layer(x_traj))
        
        return x_trajs
    
    def unet(self,x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        # x=self.outc(x)
        return x 
