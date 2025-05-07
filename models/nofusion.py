import torch.nn as nn
import torch

from models.modules.reader.pillar_encoder import PillarFeatureNet as reader
from models.modules.backbone.base_bev_backbone import BaseBEVBackbone as backbone
from models.modules.head.center_head import predict
from models.modules.compressor.naive_compress import NaiveCompressor as compressor
from models.modules.utils.downsample_conv import DownsampleConv


class NoFusion(nn.Module):
    def __init__(self, cfg):
        super(NoFusion, self).__init__()

        self.cfg = cfg
        self.reader = reader(cfg)
        self.backbone = backbone(cfg)

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

        headtype = cfg["model"]["head"]["core_method"]
        if "anchor" in headtype:
            self.headtype = "anchor"
            from models.modules.head.anchor_head_single import AnchorHeadSingle as head
            from models.modules.postprocessor.anchor_processor import (
                AnchorProcessor as processor,
            )

            self.processor = processor(cfg)
        elif "center" in headtype:
            self.headtype = "center"
            from models.modules.head.center_head import CenterHead as head
        else:
            raise RuntimeError("Unsupported detection head.")
        self.head = head(cfg)

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

        batch_dict = {
            "voxel_features": voxel_features,
            "voxel_coords": voxel_coords,
            "voxel_num_points": voxel_num_points,
            "record_len": record_len,
        }

        x = self.reader(batch_dict)

        batch_dict = self.backbone(batch_dict, x)

        batch_dict["spatial_features_2d"] = self.regroup(
            batch_dict["spatial_features_2d"], record_len
        )

        # N, C, H', W': [N, 256, 48, 176]
        # Down-sample feature to reduce memory
        if self.shrink_flag:
            batch_dict["spatial_features_2d"] = self.shrink_conv(
                batch_dict["spatial_features_2d"]
            )

        # Compressor
        if self.compression:
            # The ego feature is also compressed
            batch_dict["spatial_features_2d"] = self.naive_compressor(
                batch_dict["spatial_features_2d"]
            )

        if self.headtype == "anchor":
            batch_dict = self.head(batch_dict["spatial_features_2d"])
            batch_dict = self.processor(data_dict, batch_dict)
            if self.training:
                return batch_dict
            else:
                dets = self.processor.post_processing(batch_dict)
                return batch_dict, dets
        else:

            x = self.head(batch_dict["spatial_features_2d"])
            if self.training:
                return [x]

            else:
                dets = self.map2det(x)
                return [x], dets

    def map2det(self, preds):

        detections = []
        outputs = predict(preds, self.cfg)

        for _, output in enumerate(outputs):

            for k, v in output.items():
                if k != "token":
                    output[k] = v
            detections.append(output)
        return detections

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        x_list = []
        for i, x in enumerate(split_x):
            B, C, H, W = x.shape
            if x.shape[0] == 0:
                x = torch.zeros((1, C, H, W)).type_as(x)
            x_list.append(x)
        split_x = torch.cat(x_list, dim=0)
        return split_x
