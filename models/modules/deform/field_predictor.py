import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from models.modules.utils.unet_utils import Up, Down, DoubleConv, OutConv


class FieldPredictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        mapp_out = cfg["model"]["backbone"]["downsample"]

        t_cls_dim = cfg["model"]["deform"].get("t_cls_dim", 0)
        field_dim = cfg["model"]["deform"]["field_dim"] + t_cls_dim

        self.in_layer = DoubleConv(mapp_out, 16)
        self.inc = DoubleConv(16, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.up1 = Up(64, 32)
        self.up2 = Up(32, 16)
        self.outc = OutConv(16, field_dim)
        self.norm = nn.Sigmoid()

    def forward(self, x):
        if not self.cfg["model"]["deform"].get("ablation", None) == "wofg":
            return self.to_field_(x)
        else:
            return x

    def to_field_(self, x):
        x = self.in_layer(x)
        x_traj = self.unet(x)
        x_traj = self.norm(x_traj)
        return x_traj

    def unet(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.outc(x)
        return x
