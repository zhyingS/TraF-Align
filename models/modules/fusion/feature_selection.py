# -*- coding: utf-8 -*-
# Implementation of Select2Col.
# Author: Qian Huang <huangq@zhejianglab.com>, Yuntao Liu <liuyt@zhejianglab.com>
# License: TDG-Attribution-NonCommercial-NoDistrib

import numpy as np
import random
import torch
import torch.nn as nn

# selection the most informative semantic features for communication
class MostInformativeFeaSelection(nn.Module):
    def __init__(self, comm_args):
        super(MostInformativeFeaSelection, self).__init__()
        # Threshold of objectiveness
        self.threshold = comm_args['threshold'] if 'threshold' in comm_args.keys() else None
        if 'gaussian_smooth' in comm_args:
            # Gaussian Smooth
            self.smooth = True
            kernel_size = comm_args['gaussian_smooth']['k_size']
            c_sigma = comm_args['gaussian_smooth']['c_sigma']
            self.gaussian_filter = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)
            self.init_gaussian_filter(kernel_size, c_sigma)
            self.gaussian_filter.requires_grad = False
        else:
            self.smooth = False
            
    def init_gaussian_filter(self, k_size=5, sigma=1.0):
        center = k_size // 2
        x, y = np.mgrid[0 - center: k_size - center, 0 - center: k_size - center]
        gaussian_kernel = 1 / (2 * np.pi * sigma) * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))

        self.gaussian_filter.weight.data = torch.Tensor(gaussian_kernel).to(
            self.gaussian_filter.weight.device).unsqueeze(0).unsqueeze(0)
        self.gaussian_filter.bias.data.zero_()
    

    def forward(self, flatten_features):
        """
        Args:
            batch_confidence_maps: [(B, C, H1, W1), (B, C, H2, W2), ...]
        """
        target_flatten_max_map = torch.max(flatten_features, dim=2, keepdim=True)[0]
        target_flatten_avg_map = torch.mean(flatten_features, dim=2, keepdim=True)
        target_flatten_map = torch.sigmoid(target_flatten_max_map+target_flatten_avg_map)
        batch_size, spatial_lens, channels = target_flatten_map.shape
        if self.training:
            # Official training proxy objective
            K = int(spatial_lens * random.uniform(0, 1))
            target_flatten_map = target_flatten_map.reshape(batch_size, spatial_lens)
            _, indices = torch.topk(target_flatten_map, k=K, sorted=False)
            target_sparse_mask = torch.zeros_like(target_flatten_map).to(target_flatten_map.device)
            ones_fill = torch.ones(batch_size, K, dtype=target_flatten_map.dtype, device=target_flatten_map.device)
            target_sparse_mask = torch.scatter(target_sparse_mask, -1, indices, ones_fill).reshape(batch_size, spatial_lens, 1)
        elif self.threshold:
            ones_mask = torch.ones_like(target_flatten_map).to(target_flatten_map.device)
            zeros_mask = torch.zeros_like(target_flatten_map).to(target_flatten_map.device)
            target_sparse_mask = torch.where(target_flatten_map > self.threshold, ones_mask, zeros_mask)
        # elif self.threshold:
        #     K = self.threshold
        #     target_flatten_map = target_flatten_map.reshape(batch_size, spatial_lens)
        #     _, indices = torch.topk(target_flatten_map, k=K, sorted=False)
        #     target_sparse_mask = torch.zeros_like(target_flatten_map).to(target_flatten_map.device)
        #     ones_fill = torch.ones(batch_size, K, dtype=target_flatten_map.dtype, device=target_flatten_map.device)
        #     target_sparse_mask = torch.scatter(target_sparse_mask, -1, indices, ones_fill).reshape(batch_size, spatial_lens, 1)           
        key_spatial_flatten = target_sparse_mask * flatten_features
        # calculate the communication volumn
        agent_comm_volume = torch.sum(target_sparse_mask, dim=[1, 2], keepdim=True).reshape(batch_size)
        return key_spatial_flatten, agent_comm_volume
    

