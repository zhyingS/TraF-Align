
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from models.modules.utils.unet_utils import Up,Down,DoubleConv,OutConv

class OffsetGenerator(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg

        layer=nn.ModuleList()
        cfg_ = cfg['model']['deform']
        heads = cfg_['heads']
        self.offset_head = heads
    
        offset_kernel=cfg_['offset']['kernel']
        in_ch=cfg['model']['deform']['field_dim']

        for i in range(len(offset_kernel)):
            out_ch = 2*offset_kernel[i]*offset_kernel[i]*heads
            layer.append(DoubleConv(in_ch,out_ch,mid_channels=out_ch,stride=1))
        self.offset_layer=layer

        self.init_weight()
        
    def init_weight(self):
        for m in self.modules():
            if isinstance(m,(nn.Conv2d,nn.Linear)):
                self.init_weight_(m)
            if isinstance(m,(nn.ModuleList,nn.Sequential)):
                m.apply(self.init_weight_)

    def init_weight_(self,m):
        if isinstance(m,(nn.Conv2d,nn.Linear)):
            nn.init.kaiming_normal_(m.weight)
                
    def forward(self, x_traj):
        
        return self.to_offset_(x_traj)

    def to_offset_(self,x):
        
        out=[]
        for i,layer in enumerate(self.offset_layer):
            x_offset=x[i].clone()
            x_offset=layer(x_offset) 
            # scale = self.cfg['model']['deform']['input_stride'][i]           
            out.append(x_offset)

        return out
    
