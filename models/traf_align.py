import torch
import torch.nn as nn

from models.modules.reader.pillar_encoder import PillarFeatureNet as reader
from models.modules.backbone.sparseresnet import SparseResNet as backbone
from models.modules.neck.aspp import ASPPNeck as aspp
import datasets.data_utils.data_check.data_check_utils as check_utils
from models.modules.head.center_head import predict
from models.modules.utils.downsample_conv import DownsampleConv
from models.modules.deform.traf_align_fusion import TrafAlign_


class TraFAlign(nn.Module):
    def __init__(self, cfg):
        super(TraFAlign, self).__init__()

        self.cfg = cfg
        self.reader = reader(cfg)
        self.backbone = backbone(cfg)
        self.fusion_net = TrafAlign_(cfg)
        if "neck" in cfg["model"]:
            self.aspp = aspp(cfg["model"]["neck"])
        else:
            self.aspp = nn.Identity()

        try:
            headtype = cfg["model"]["head"]["core_method"]
        except:
            headtype = "centerhead"

        if "anchor" in headtype:
            self.headtype = "anchor"
            from models.modules.head.anchor_head_single import AnchorHeadSingle as head
            from models.modules.postprocessor.anchor_processor import (
                AnchorProcessor as processor,
            )

            self.processor = processor(cfg)
            self.head = head(cfg, input_channels=cfg["model"]["head"]["in_channels"])

        elif "center" in headtype:
            self.headtype = "center"
            from models.modules.head.center_head import CenterHead as head

            self.head = head(cfg)

        else:
            raise RuntimeError("Unsupported detection head.")

        if "shrink_header" in cfg["model"]:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(cfg["model"]["shrink_header"])
        else:
            self.shrink_flag = False

        # self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight)
            if isinstance(m, (nn.ModuleList, nn.Sequential)):
                m.apply(self.init_weight_)

    def init_weight_(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_uniform_(m.weight)

    def forward(self, data_dict, infer=False):
        """
        data_dict['label_dict] Nx11 batch,x,y,z,h,w,l,phi,type,id,timestamp
        data_dict['processed_lidar]['voxel_coords] Nx4 batch,t,y,x
        data_dict['processed_lidar]['voxel_features] Nx6 x,y,z,t,one_hot_cav_id
        """

        lidar = data_dict["processed_lidar"]
        x = self.reader(lidar)
        x = self.backbone(x, data_dict)

        x, x_traj, offset_list = self.fusion_net(x, data_dict, lidar)

        if self.shrink_flag:
            x = self.shrink_conv(x)

        x = self.aspp(x)

        if self.headtype == "anchor":
            batch_dict = self.head(x)
            batch_dict = self.processor(data_dict, batch_dict)
            batch_dict.update({"x_traj": x_traj, "x_offset": offset_list})

            if self.training:
                return batch_dict
            else:
                dets = self.processor.post_processing(batch_dict)
                return batch_dict, dets
        else:

            x = self.head(x)
            if self.training:
                return [x, x_traj, offset_list]

            else:
                dets = self.map2det(x)
                return [x, x_traj, offset_list], dets

    def map2det(self, preds):

        detections = []
        outputs = predict(preds, self.cfg)

        for _, output in enumerate(outputs):

            for k, v in output.items():
                if k != "token":
                    output[k] = v
            detections.append(output)
        return detections
