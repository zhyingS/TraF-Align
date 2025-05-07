import torch.nn as nn
import torch

from models.modules.reader.pillar_encoder import PillarFeatureNet as reader
from models.modules.backbone.mrcnet_bev_backbone import MRCNetBEVBackbone as backbone
from models.modules.compressor.naive_compress import NaiveCompressor as compressor
from models.modules.utils.downsample_conv import DownsampleConv
from models.modules.fusion.where2comm_fuse import Where2comm as where2comm
from models.modules.head.anchor_head_single import AnchorHeadSingle as head
from models.modules.postprocessor.anchor_processor import AnchorProcessor as processor
from models.modules.fusion.mrcnet_fusion import MSRobustFusion
from models.modules.fusion.MEMmodule import MotionEnhancedMech


class MRCNet(nn.Module):
    def __init__(self, cfg):
        super(MRCNet, self).__init__()

        self.cfg = cfg
        self.reader = reader(cfg)
        self.backbone = backbone(cfg, 64)
        self.head = head(cfg, input_channels=128)
        self.processor = processor(cfg)

        self.compression = False
        if "compression" in cfg["model"] and cfg["model"]["compression"] > 0:
            self.compression = True
            self.naive_compressor = compressor(cfg)

        # use temporal
        self.use_temporal = True if "temporal_model" in cfg["model"] else False
        if self.use_temporal:
            # MEM module for motion context fusion
            in_channel = cfg["model"]["temporal_model"]["in_channel"]
            # shape_size = cfg['model']['temporal_model']['shape_size']
            self.temporal_fusion = MotionEnhancedMech(
                input_size=in_channel, hidden_size=in_channel
            )
            self.down_sampling = nn.Sequential(
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    in_channel,
                    in_channel,
                    kernel_size=3,
                    stride=2,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(in_channel, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            )
            self.adap_fusion = nn.Conv2d(128 * 6, 128 * 3, kernel_size=1)

        self.max_cav = cfg["train_params"]["max_cav"]
        self.fusion_net = MSRobustFusion(self.max_cav, cfg["model"]["mrcfusion"])
        self.down_conv = nn.Sequential(
            nn.Conv2d(128 * 6, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

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
            "ego_frames": data_dict["ego_frames"],
        }

        x = self.reader(batch_dict)

        batch_dict = self.backbone(batch_dict, x)

        batch_dict_list = batch_dict["align_features"]
        bev_features_2d = batch_dict["bev_features_2d"]
        if self.use_temporal:
            record_len = record_len.cpu().tolist()
            # bev_features_2d : ([7, 256, 100, 352])
            his_frames_lens = [data_dict["ego_frames"] - 1] * len(record_len)
            # torch.Size([2, 384, 100, 352])
            temporal_fusion_output = self.temporal_fusion(
                bev_features_2d, record_len, his_frames_lens
            )
            temporal_fusion_output_down = self.down_sampling(temporal_fusion_output)
            temporal_fusion_output_down = temporal_fusion_output_down.transpose(-1, -2)

        fusion_output, _ = self.fusion_net(batch_dict_list, record_len)
        # v2v_fusion_feature : ([2, 384, 176, 50])
        v2v_fusion_feature = fusion_output["aggregated_spatial_features_2d"]
        if self.use_temporal:
            target_feature = torch.cat(
                [temporal_fusion_output_down, v2v_fusion_feature], dim=1
            )
            if self.down_sampling:
                target_feature = self.down_conv(target_feature)
        else:
            target_feature = v2v_fusion_feature
        batch_dict["spatial_features_2d"] = target_feature

        batch_dict = self.head(batch_dict["spatial_features_2d"])
        batch_dict = self.processor(data_dict, batch_dict)
        if self.training:
            return batch_dict
        else:
            dets = self.processor.post_processing(batch_dict)
            return batch_dict, dets
