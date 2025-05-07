import torch.nn as nn
import torch

from models.modules.reader.pillar_encoder import PillarFeatureNet as reader
from models.modules.backbone.base_bev_backbone import BaseBEVBackbone as backbone
from models.modules.head.center_head import predict
from models.modules.compressor.naive_compress import NaiveCompressor as compressor
from models.modules.utils.downsample_conv import DownsampleConv
from einops import rearrange, repeat
import numpy as np
from models.modules.fusion.swap_fusion_modules import SwapFusionEncoder


class COBEVT(nn.Module):
    def __init__(self, cfg):
        super(COBEVT, self).__init__()

        self.cfg = cfg
        self.reader = reader(cfg)
        self.backbone = backbone(cfg)

        self.compression = False
        if "compression" in cfg["model"] and cfg["model"]["compression"] > 0:
            self.compression = True
            self.naive_compressor = compressor(cfg)
        self.max_cav = cfg["train_params"]["max_cav"]

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
        self.fusion_net = SwapFusionEncoder(cfg["model"]["fax_fusion"])

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

        # N, C, H, W -> B,  L, C, H, W
        regroup_feature, mask = regroup(
            batch_dict["spatial_features_2d"], record_len, self.max_cav
        )
        com_mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        com_mask = repeat(
            com_mask,
            "b h w c l -> b (h new_h) (w new_w) c l",
            new_h=regroup_feature.shape[3],
            new_w=regroup_feature.shape[4],
        )

        fused_feature = self.fusion_net(regroup_feature, com_mask)
        batch_dict["spatial_features_2d"] = fused_feature

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


def regroup(dense_feature, record_len, max_len):
    """
    Regroup the data based on the record_len.

    Parameters
    ----------
    dense_feature : torch.Tensor
        N, C, H, W
    record_len : list
        [sample1_len, sample2_len, ...]
    max_len : int
        Maximum cav number

    Returns
    -------
    regroup_feature : torch.Tensor
        B, L, C, H, W
    """
    cum_sum_len = list(np.cumsum(torch_tensor_to_numpy(record_len)))
    split_features = torch.tensor_split(dense_feature, cum_sum_len[:-1])
    regroup_features = []
    mask = []

    for split_feature in split_features:
        B, C, H, W = split_feature.shape
        if B == 0:
            split_feature = torch.zeros((1, C, H, W)).type_as(split_feature)
        # M, C, H, W
        feature_shape = split_feature.shape
        # the maximum M is 5 as most 5 cavs
        padding_len = max_len - feature_shape[0]
        mask.append([1] * feature_shape[0] + [0] * padding_len)

        padding_tensor = torch.zeros(
            padding_len, feature_shape[1], feature_shape[2], feature_shape[3]
        )
        padding_tensor = padding_tensor.to(split_feature.device)

        split_feature = torch.cat([split_feature, padding_tensor], dim=0)

        # 1, 5C, H, W
        split_feature = split_feature.view(
            -1, feature_shape[2], feature_shape[3]
        ).unsqueeze(0)
        regroup_features.append(split_feature)

    # B, 5C, H, W
    regroup_features = torch.cat(regroup_features, dim=0)
    # B, L, C, H, W
    regroup_features = rearrange(
        regroup_features, "b (l c) h w -> b l c h w", l=max_len
    )
    mask = torch.from_numpy(np.array(mask)).to(regroup_features.device)

    return regroup_features, mask


def torch_tensor_to_numpy(torch_tensor):
    """
    Convert a torch tensor to numpy.

    Parameters
    ----------
    torch_tensor : torch.Tensor

    Returns
    -------
    A numpy array.
    """
    return (
        torch_tensor.numpy()
        if not torch_tensor.is_cuda
        else torch_tensor.cpu().detach().numpy()
    )
