import torch
from torch import nn
import torch.nn.functional as F
import time

import spconv
import spconv.pytorch
from spconv.pytorch import SparseSequential, SparseConv2d, SparseConvTranspose2d
from torchvision.ops import DeformConv2d
from collections import OrderedDict

from models.utils.sparse_conv import SparseConvBlock, SparseBasicBlock


class SparseResNet(spconv.pytorch.SparseModule):
    def __init__(self, cfg, kernel_size=[3, 3, 3, 3, 3]):

        super(SparseResNet, self).__init__()
        self.cfg = cfg
        kwargs = cfg["model"]["backbone"]["resnet"]
        max_cav = cfg["train_params"]["max_cav"]
        self.one_batch = cfg["dataset"]["frame_his"] + cfg["dataset"][
            "cav_frame_his"
        ] * (max_cav - 1)
        if cfg["dataset"]["merge_pcd_first"]:
            self.one_batch = max_cav
        layer_nums = kwargs["layer_nums"]
        ds_layer_strides = kwargs["ds_layer_strides"]
        ds_num_filters = kwargs["ds_num_filters"]
        num_input_features = kwargs["num_input_features"]
        if "kernel_size" in kwargs:
            kernel_size = kwargs["kernel_size"]
        if "out_channels" in kwargs:
            out_channels = kwargs["out_channels"]
        self._layer_strides = ds_layer_strides
        self._num_filters = ds_num_filters
        self._layer_nums = layer_nums
        self._num_input_features = num_input_features

        assert len(self._layer_strides) == len(self._layer_nums)
        assert len(self._num_filters) == len(self._layer_nums)

        in_filters = [self._num_input_features, *self._num_filters[:-1]]
        blocks = []

        for i, layer_num in enumerate(self._layer_nums):
            block = self._make_layer(
                in_filters[i],
                self._num_filters[i],
                kernel_size[i],
                self._layer_strides[i],
                layer_num,
            )
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)

        num_levels = len(layer_nums)
        upsample_strides = cfg["model"]["backbone"]["upsample_strides"]
        num_upsample_filters = cfg["model"]["backbone"]["num_upsample_filters"]

        in_dim = sum(num_upsample_filters)
        if "downsample" in cfg["model"]["backbone"]:
            out_dim = cfg["model"]["backbone"]["downsample"]
            self.downsample = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(),
            )
            in_dim = out_dim
        else:
            self.downsample = nn.Identity()

        self.deblocks = nn.ModuleList()

        for idx in range(num_levels):
            stride = upsample_strides[idx]
            if stride >= 1:
                self.deblocks.append(
                    SparseSequential(
                        SparseConvTranspose2d(
                            ds_num_filters[idx],
                            num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx],
                            bias=False,
                        ),
                        nn.BatchNorm1d(
                            num_upsample_filters[idx], eps=1e-3, momentum=0.01
                        ),
                        nn.ReLU(),
                    )
                )
            else:
                stride = int(1 / stride)
                self.deblocks.append(
                    SparseSequential(
                        SparseConv2d(
                            ds_num_filters[idx],
                            num_upsample_filters[idx],
                            stride,
                            stride=stride,
                            bias=False,
                        ),
                        nn.BatchNorm1d(
                            num_upsample_filters[idx], eps=1e-3, momentum=0.01
                        ),
                        nn.ReLU(),
                    )
                )
        self.features_save = OrderedDict()

    def forward(self, x, data_dict):
        pillar_features, coors, input_shape = x
        batch_size = torch.max(coors[:, 0]) + 1

        x = spconv.pytorch.SparseConvTensor(
            pillar_features, coors, input_shape, batch_size
        )

        ups = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            y = self.deblocks[i](x)
            ups.append(y.dense())

        # ups = torch.cat((ups[0]+ups[-1],ups[1]+ups[2]),dim=1)
        ups = torch.cat(ups, dim=1)
        ups = self.downsample(ups)

        return ups

    def _make_layer(self, inplanes, planes, kernel_size, stride, num_blocks):

        layers = []
        layers.append(
            SparseConvBlock(
                inplanes, planes, kernel_size=kernel_size, stride=stride, use_subm=False
            )
        )

        for j in range(num_blocks):
            layers.append(SparseBasicBlock(planes, kernel_size=kernel_size))

        return spconv.pytorch.SparseSequential(*layers)
