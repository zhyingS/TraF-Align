import torch.nn as nn
import torch

from models.modules.reader.pillar_encoder import PillarFeatureNet as reader
from models.modules.backbone.base_bev_backbone import BaseBEVBackbone as backbone
from models.modules.compressor.naive_compress import NaiveCompressor as compressor
from models.modules.utils.downsample_conv import DownsampleConv
from models.modules.fusion.where2comm_fuse import Where2comm as where2comm
from models.modules.head.anchor_head_single import AnchorHeadSingle as head
from models.modules.postprocessor.anchor_processor import AnchorProcessor as processor


class Where2comm(nn.Module):
    def __init__(self, cfg):
        super(Where2comm, self).__init__()

        self.cfg = cfg
        self.reader = reader(cfg)
        self.backbone = backbone(cfg)
        self.head = head(cfg)
        self.processor = processor(cfg)

        self.compression = False
        if "compression" in cfg["model"] and cfg["model"]["compression"] > 0:
            self.compression = True
            self.naive_compressor = compressor(cfg)

        # Used to down-sample the feature map for efficient computation
        if "shrink_header" in cfg["model"]:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(cfg["model"]["shrink_header"])
        else:
            self.shrink_flag = False

        self.fusion_net = where2comm(cfg["model"]["where2comm_fusion"])
        self.multi_scale = cfg["model"]["where2comm_fusion"]["multi_scale"]

        # self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                self.init_weight_(m)
            if isinstance(m, (nn.ModuleList, nn.Sequential)):
                m.apply(self.init_weight_)

    def init_weight_(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight)

    def forward(self, data_dict):

        voxel_features = data_dict["processed_lidar"]["voxel_features"]
        voxel_coords = data_dict["processed_lidar"]["voxel_coords"]
        voxel_num_points = data_dict["processed_lidar"]["voxel_num_points"]
        record_len = data_dict["record_len"]
        pairwise_t_matrix = data_dict["pairwise_t_matrix"]

        batch_dict = {
            "voxel_features": voxel_features,
            "voxel_coords": voxel_coords,
            "voxel_num_points": voxel_num_points,
        }

        x = self.reader(batch_dict)

        batch_dict = self.backbone(batch_dict, x)

        # N, C, H', W': [N, 256, 48, 176]
        # Down-sample feature to reduce memory
        if self.shrink_flag:
            batch_dict["spatial_features_2d"] = self.shrink_conv(
                batch_dict["spatial_features_2d"]
            )

        psm_single = self.head(batch_dict["spatial_features_2d"], only_cls=True)

        # Compressor
        if self.compression:
            # The ego feature is also compressed
            batch_dict["spatial_features_2d"] = self.naive_compressor(
                batch_dict["spatial_features_2d"]
            )

        if self.multi_scale:
            # Bypass communication cost, communicate at high resolution, neither shrink nor compress
            fused_feature, _ = self.fusion_net(
                batch_dict["spatial_features"],
                psm_single,
                record_len,
                pairwise_t_matrix,
                self.backbone,
            )
            if self.shrink_flag:
                fused_feature = self.shrink_conv(fused_feature)
        else:
            fused_feature, _ = self.fusion_net(
                batch_dict["spatial_features_2d"],
                psm_single,
                record_len,
                pairwise_t_matrix,
            )
        batch_dict["spatial_features_2d"] = fused_feature

        batch_dict = self.head(batch_dict["spatial_features_2d"])
        batch_dict = self.processor(data_dict, batch_dict)
        if self.training:
            return batch_dict
        else:
            dets = self.processor.post_processing(batch_dict)
            return batch_dict, dets
