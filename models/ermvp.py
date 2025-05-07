import torch.nn as nn
import torch

from models.modules.reader.pillar_encoder import PillarFeatureNet as reader
from models.modules.backbone.base_bev_backbone import BaseBEVBackbone as backbone
from models.modules.head.center_head import predict
from models.modules.compressor.naive_compress import NaiveCompressor as compressor
from models.modules.utils.downsample_conv import DownsampleConv
from einops import rearrange, repeat
import numpy as np
import math
from models.modules.utils.sampler import SortSampler
from models.modules.utils.cluster import merge_tokens, cluster_dpc_knn, index_points
from models.modules.head.anchor_head_single import AnchorHeadSingle as head
from models.modules.postprocessor.anchor_processor import AnchorProcessor as processor
from models.modules.fusion.ermvp_fusion_modules import *


class ERMVP(nn.Module):
    def __init__(self, cfg):
        super(ERMVP, self).__init__()

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

        self.processor = processor(cfg)
        self.head = head(cfg)

        self.topk_ratio = cfg["model"]["comm"]["topk_ratio"]
        self.cluster_sample_ratio = cfg["model"]["comm"]["cluster_sample_ratio"]

        self.sampler = SortSampler(
            topk_ratio=self.topk_ratio, input_dim=256, score_pred_net="2layer-fc-256"
        )

        self.fusion_net = ERMVPFusionEncoder(cfg["model"]["ermvp_fusion"])

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
        spatial_features_2d = batch_dict["spatial_features_2d"]

        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)

        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)

        N, C, H, W = spatial_features_2d.shape

        dis_priority = torch.ones([N, H, W]).type_as(spatial_features_2d)
        idx = (
            torch.arange(H * W)
            .repeat(N, 1, 1)
            .permute(2, 0, 1)
            .type_as(spatial_features_2d)
        )
        # src:N,B,C
        src, sample_reg_loss, sort_confidence_topk, pos_embed = self.sampler(
            spatial_features_2d, idx, None, dis_priority
        )
        # # # B N C
        src = src.permute(1, 0, 2)
        _, s_len, _ = src.shape

        cluster_num = max(math.ceil(s_len * self.cluster_sample_ratio), 1)

        idx_cluster, cluster_num = cluster_dpc_knn(src, cluster_num, 10)
        down_dict, idx = merge_tokens(
            src, idx_cluster, cluster_num, sort_confidence_topk.unsqueeze(2)
        )
        idxxs = []
        for b in range(N):
            i = torch.arange(s_len)
            idxxs.append(idx[b][i])
        idxxs = torch.vstack(idxxs)

        src = index_points(down_dict, idxxs)
        src = src.permute(0, 2, 1)

        pos_embed = pos_embed.permute(1, 2, 0)

        batch_spatial_features = []
        for cav_idx in range(N):
            index = cav_idx
            for i, ele in enumerate(record_len):
                if index < ele:
                    break
                index = index - ele
            spatial_feature = torch.zeros(C, H * W, dtype=src.dtype, device=src.device)
            spatial_feature[:, pos_embed[cav_idx][0].long()] = src[cav_idx]

            if index == 0:
                spatial_feature = spatial_features_2d[cav_idx].flatten(1)
            # print(timestamp_index)
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(N, C, H, W)

        ego_features = get_ego_feature(spatial_features_2d, record_len)

        regroup_feature, mask = regroup(spatial_features_2d, record_len, self.max_cav)
        # [1,1,1,1,2]
        com_mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        # [1,60,180,1,2]
        com_mask = repeat(
            com_mask,
            "b h w c l -> b (h new_h) (w new_w) c l",
            new_h=regroup_feature.shape[3],
            new_w=regroup_feature.shape[4],
        )

        fused_feature = self.fusion_net(regroup_feature, com_mask)

        batch_dict = self.head(fused_feature)
        batch_dict_ego = self.head(ego_features, prefix="_ego")

        batch_dict = self.processor(data_dict, batch_dict)
        batch_dict_ego = self.processor(data_dict, batch_dict_ego, prefix="_ego")

        if self.training:
            return batch_dict, batch_dict_ego
        else:
            dets = self.processor.post_processing(batch_dict)
            return batch_dict, dets

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


def get_ego_feature(x, record_len):
    cum_sum_len = torch.cumsum(record_len, dim=0)
    split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
    out = []
    for xx in split_x:
        if xx.shape[0] > 0:
            xx = xx[0].unsqueeze(0)
        else:
            B, C, H, W = xx.shape
            xx = torch.zeros((1, C, H, W)).type_as(xx)
        out.append(xx)
    return torch.cat(out, dim=0)
