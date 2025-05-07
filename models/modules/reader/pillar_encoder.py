"""
PointPillars with hard/dynamic voxelization
Licensed under MIT License [see LICENSE].
"""

import torch
from torch import nn
from torch.nn import functional as F

import numpy as np
from functools import reduce
from numba import jit


class PFNLayer(nn.Module):
    """
    Pillar Feature Net Layer.
    The Pillar Feature Net could be composed of a series of these layers, but the PointPillars paper results only
    used a single PFNLayer. This layer performs a similar role as second.pytorch.voxelnet.VFELayer.
    :param in_channels: <int>. Number of input channels.
    :param out_channels: <int>. Number of output channels.
    :param last_layer: <bool>. If last_layer, there is no concatenation of features.
    """

    def __init__(self, in_channels, out_channels, norm_cfg=None, last_layer=False):
        super().__init__()
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels

        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)

        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, inputs):
        torch.backends.cudnn.enabled = False
        x = self.linear(inputs)
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.relu(x)
        torch.backends.cudnn.enabled = True

        # max pooling
        x_max = torch.max(x, dim=1).values
        if self.last_vfe:
            return x_max
        else:
            x_max = x_max.unsqueeze(1).repeat(1, x.shape[1], 1)
            x_concatenated = torch.cat([x, x_max], dim=-1)
            return x_concatenated


class TFNLayer(nn.Module):
    """
    Pillar Feature Net Layer.
    The Pillar Feature Net could be composed of a series of these layers, but the PointPillars paper results only
    used a single PFNLayer. This layer performs a similar role as second.pytorch.voxelnet.VFELayer.
    :param in_channels: <int>. Number of input channels.
    :param out_channels: <int>. Number of output channels.
    :param last_layer: <bool>. If last_layer, there is no concatenation of features.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.units = out_channels

        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)

        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, inputs):
        x = self.linear(inputs)
        x = self.norm(x)
        x = F.relu(x)

        return x


class PillarNet(nn.Module):
    """
    PillarNet.
    The network performs dynamic pillar scatter that convert point cloud into pillar representation
    and extract pillar features

    Reference:
    PointPillars: Fast Encoders for Object Detection from Point Clouds (https://arxiv.org/abs/1812.05784)
    End-to-End Multi-View Fusion for 3D Object Detection in LiDAR Point Clouds (https://arxiv.org/abs/1910.06528)

    Args:
        num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
        pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
    """

    def __init__(self, num_input_features, voxel_size, pc_range, args):
        super().__init__()
        self.voxel_size = torch.tensor(voxel_size)
        self.pc_range = torch.tensor(pc_range)
        self.args = args
        grid_size = (
            self.pc_range[3:] - self.pc_range[:3]
        ) / self.voxel_size  # x,  y, z

        self.grid_size = torch.ceil(grid_size).long()
        self.f_center_z = (
            True if args["model"]["reader"]["num_input_features"] == 9 else False
        )

    def forward(self, dict):
        """
        Args:
            points: torch.Tensor of size (N, d), format: batch_id, x, y, z, feat1, ...
        """
        import time

        features_ = dict["voxel_features"]  # x,y,z,t,one_hot_cav_id
        concat_t = self.args["model"]["reader"]["timestamp"]

        if concat_t:
            features = features_[:, :, [0, 1, 2, 3]]
        else:
            features = features_[:, :, [0, 1, 2]]

        coords = dict["voxel_coords"][:, [2, 3]]  # b,_,y,x
        nums = dict["voxel_num_points"]

        self.voxel_size = self.voxel_size.type_as(features)
        self.pc_range = self.pc_range.type_as(features)
        self.grid_size = self.grid_size.type_as(features)

        points_mean_scatter = torch.sum(features[:, :, :3], dim=1) / nums[:, None]

        f_cluster = features[:, :, :3] - points_mean_scatter.unsqueeze(1)

        # Find distance of x, y, and z from pillar center
        dim_z = 3 if self.f_center_z else 2
        f_center = torch.zeros_like(features[:, :, :dim_z])
        f_center[:, :, :2] = features[:, :, [0, 1]] - (
            (coords[:, [1, 0]] + 0.5) * self.voxel_size[:2].unsqueeze(0)
            + self.pc_range[:2].unsqueeze(0)
        ).unsqueeze(1)
        if self.f_center_z:
            f_center[:, :, 2] = features[:, :, 2] - (
                0.5 * self.voxel_size[2].unsqueeze(0) + self.pc_range[2].unsqueeze(0)
            ).unsqueeze(1)

        mask = features[:, :, :3] != 0
        f_cluster = f_cluster * mask
        f_center = f_center * mask[:, :, :dim_z]

        features = torch.cat([features, f_center, f_cluster], dim=-1)
        # features = features[:,:,[0,1,2,4,5,6,7,8,3]]

        return features, dict["voxel_coords"][:, [0, 2, 3]], self.grid_size[[1, 0]]


class PillarFeatureNet(nn.Module):
    def __init__(self, cfg):
        """
        Pillar Feature Net.
        The network prepares the pillar features and performs forward pass through PFNLayers. This net performs a
        similar role to SECOND's second.pytorch.voxelnet.VoxelFeatureExtractor.
        :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        :param voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
        :param pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
        """
        super().__init__()
        self.cfg = cfg
        kwargs = cfg["model"]["reader"]
        num_input_features = kwargs["num_input_features"]
        num_filters = kwargs["num_filters"]
        voxel_size = cfg["voxelization"]["voxel_size"]
        pc_range = cfg["voxelization"]["lidar_range"]
        norm_cfg = (None,)

        assert len(num_filters) > 0
        if cfg["model"]["reader"]["timestamp"]:
            num_input_features += 1

        # Create PillarFeatureNet layers
        num_filters = [num_input_features] + list(num_filters)
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(
                PFNLayer(
                    in_filters, out_filters, norm_cfg=norm_cfg, last_layer=last_layer
                )
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.feature_output_dim = num_filters[-1]
        self.voxel_size = np.array(voxel_size)
        self.pc_range = np.array(pc_range)
        self.voxelization = PillarNet(
            num_input_features, voxel_size, pc_range, self.cfg
        )

    def forward(self, points):
        # points: batch, x,y,z,[t],[one_hot_cav_id]
        # coords:batch,t,y,x grid_size:y,x
        with torch.no_grad():
            features, coords, grid_size = self.voxelization(points)
        # Forward pass through PFNLayers
        for pfn in self.pfn_layers:
            features = pfn(features)  # num_points, dim_feat

        if features.shape[0] == 0:
            features = torch.zeros((3, features.shape[1])).type_as(features)
            coords = torch.zeros((3, 3)).type_as(coords)

        return features, coords, grid_size
