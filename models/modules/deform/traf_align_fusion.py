import torch
from torch import nn

from models.modules.deform.field_predictor import FieldPredictor
from models.modules.deform.offset_generator import OffsetGenerator
from einops import rearrange
from models.modules.compressor.naive_compress import NaiveCompressor as compressor
from models.modules.deform.trafalign_transformer import (
    TrafalignTransformer as attention,
)
import datasets.data_utils.data_check.data_check_utils as check_utils


class TrafAlign_(nn.Module):
    def __init__(self, cfg):
        super(TrafAlign_, self).__init__()
        self.cfg = cfg

        self.ablation_mode = self.cfg["model"]["deform"].get("ablation", None)

        ## for temporal encoding
        self.ego_frame = cfg["dataset"]["frame_his"]
        self.cav_frame = cfg["dataset"]["cav_frame_his"]
        max_cav = cfg["train_params"]["max_cav"]
        self.one_batch = self.ego_frame + self.cav_frame * (max_cav - 1)
        if cfg["dataset"]["merge_pcd_first"]:
            self.one_batch = max_cav
            self.ego_frame = 1
        self.max_cav = max_cav

        if "downsample" in cfg["model"]["backbone"]:
            self.deform_dim = cfg["model"]["backbone"]["downsample"]
        else:
            self.deform_dim = sum(cfg["model"]["backbone"]["num_upsample_filters"])
        if not self.cfg["model"]["reader"]["timestamp"]:
            self.tfn_layer = nn.Sequential(
                nn.Conv2d(self.deform_dim * 2, self.deform_dim, kernel_size=1),
                nn.BatchNorm2d(self.deform_dim),
                nn.ReLU(),
                nn.Conv2d(self.deform_dim, self.deform_dim, kernel_size=1),
                nn.BatchNorm2d(self.deform_dim),
                nn.ReLU(),
            )

        ## feature communication
        self.compression = False
        if "compression" in cfg["model"] and cfg["model"]["compression"] > 0:
            self.compression = True
            input_dim = cfg["model"]["backbone"]["resnet"]["ds_num_filters"][-1]
            self.naive_compressor = compressor(input_dim, cfg)

        ## field prediction
        self.field_predictor = FieldPredictor(cfg)

        ## offset generation
        self.off_kernel = cfg["model"]["deform"]["offset"]["kernel"][0]
        offset_in_dim = (
            self.deform_dim
            if self.ablation_mode == "wofg"
            else cfg["model"]["deform"]["field_dim"]
        )
        self.offset_head = cfg["model"]["deform"]["offset"]["heads"]
        self.offset_generator = OffsetGenerator(cfg, in_ch=offset_in_dim)

        ## attention
        if cfg["model"]["deform"].get("attention", None) is not None:
            cfg_att = cfg["model"]["deform"]["attention"]
            layers = cfg_att.get("layer_num", 1)
            num_heads = cfg_att.get("num_heads", 1)
            att_dim = cfg_att.get("dim", 64)
            self.att_block = nn.ModuleList()
            for _ in range(layers):
                self.att_block.append(
                    attention(
                        cfg,
                        channels=self.deform_dim,
                        attention_dim=att_dim,
                        num_heads=num_heads,
                        num_random_points=18,
                    )
                )
        else:
            self.att_block = None

        ## cross-agent fusion
        self.core_method = cfg["model"]["deform"].get("fusion", "conv")
        if self.core_method == "conv":
            self.fusion_net = nn.Sequential(
                nn.Conv2d(
                    self.deform_dim * self.max_cav,
                    self.deform_dim,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(self.deform_dim),
                nn.ReLU(),
                nn.Conv2d(
                    self.deform_dim,
                    self.deform_dim,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(self.deform_dim),
                nn.ReLU(),
            )

        ## output mapping
        map_dim = cfg["model"]["deform"]["mapping_dim"]
        self.mapping = nn.Sequential(
            nn.Conv2d(
                self.deform_dim,
                map_dim,
                1,
                stride=cfg["model"]["deform"].get("mapping_stride", 1),
                bias=False,
            ),
            nn.BatchNorm2d(map_dim, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

    def forward(self, x, data_dict, lidar):
        # segment features of ego and agent i, and add temporal embeddings
        x = self.regroup(x, data_dict, lidar["voxel_coords"])
        x = self.communicate(x)

        batch = x.shape[0]
        x = rearrange(x, "b n c h w -> (b n) c h w")

        # generate field and offset
        x_traj = self.field_predictor(x)
        x_offset = self.offset_generator(x_traj)
        if self.ablation_mode in ["wofgog"]:
            x_offset = x_offset * 0
        offset = rearrange(x_offset, "b (c d) h w -> b c d h w", c=self.offset_head)

        atts = []
        if self.att_block is not None and self.ablation_mode != "direct":
            x = rearrange(x, "b c h w -> b h w c")
            # find the pixel indices at attention positions
            with torch.no_grad():
                selected_indices = self.offset_to_att_indice(offset.detach())

            # plot feature map
            # check_utils.vis_backbone_map_paper(x.permute(0,3,1,2),self.cfg)
            
            for att_block in self.att_block:
                # feature interaction along attention positions
                x, attention_weights = att_block(x, selected_indices)
                atts.append(attention_weights)

            # check_utils.vis_offset_on_pcd(lidar,x_offset,self.cfg,selected_indices,atts)
            x = rearrange(x, "b h w c -> b c h w")

        # plot attention matrix
        # check_utils.vis_attention(offset,selected_indices,atts)

        x = rearrange(x, "(b n) c h w -> b n c h w", b=batch)
        x = self.fusion(x)
        x = self.mapping(x)

        return x, x_traj, x_offset

    def offset_to_att_indice(self, offset):
        h, w = offset.shape[-2:]
        kernel = self.off_kernel

        anchor_h, anchor_w = torch.meshgrid(torch.arange(kernel), torch.arange(kernel))
        anchor_h, anchor_w = anchor_h - (kernel - 1) / 2, anchor_w - (kernel - 1) / 2
        anchor = torch.cat((anchor_h[None, ...], anchor_w[None, ...]), dim=0).type_as(
            offset
        )
        anchor = rearrange(anchor, "xy k v -> (xy k v)")
        offset = offset + anchor[None, None, :, None, None]

        offset = rearrange(
            offset, "b hd (xy k) h w -> b hd xy k h w", k=kernel * kernel
        )

        anchor_h, anchor_w = torch.meshgrid(torch.arange(h), torch.arange(w))
        anchor = torch.cat((anchor_h[None, ...], anchor_w[None, ...]), dim=0).type_as(
            offset
        )
        offset = offset + anchor[None, None, :, None, :, :]

        offset = rearrange(offset, "b hd xy kv h w -> b xy (hd kv) h w")
        offset = offset[:, 0, ...] * w + offset[:, 1, ...]
        offset = rearrange(offset, "b c h w -> b (h w) c")
        offset = torch.clamp_(offset.round().long(), 0, h * w - 1)

        return offset

    def fusion(self, x):
        if not self.core_method == "max":
            b, n, c, h, w = x.shape
            padding_len = self.max_cav - n
            if padding_len > 0:
                padding_tensor = torch.zeros(b, padding_len, c, h, w).type_as(x)
                x = torch.cat([x, padding_tensor], dim=1)
            x = rearrange(x, "b n c h w -> b (n c) h w")
        if self.core_method == "conv":
            x = self.fusion_net(x)
        elif self.core_method == "max":
            x = torch.max(x, dim=1).values
        return x

    def _regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def regroup(self, features, data_dict, coords=None):
        delay = data_dict["time_delays"]
        record_len = data_dict["record_len"]

        # if adding frame-wise time to features
        if not self.cfg["model"]["reader"]["timestamp"]:
            features = self.get_temporal_features(features, delay, record_len, coords)
            features = self.tfn_layer(features)

        # regroup ego features and agent i features
        features_ = self._regroup(features, record_len)
        ego_features, cav_features = [], []
        for feature in features_:
            ego_features.append(
                (feature[: self.ego_frame, :, :, :]).sum(0).unsqueeze(0)
            )
            cav_features.append(
                (feature[self.ego_frame :, :, :, :]).sum(0).unsqueeze(0)
            )
        ego_features = torch.cat(ego_features)
        cav_features = torch.cat(cav_features)

        return torch.cat((ego_features, cav_features), dim=0)

    def get_temporal_features(self, features, delay, record_len, coords):
        
        with torch.no_grad():
            one_batch = self.one_batch
            if self.training:
                raw_batch_size = self.cfg["train_params"]["train_batch_size"]
            else:
                raw_batch_size = self.cfg["train_params"]["val_batch_size"]
            assert len(delay[0]) == self.max_cav

            a_ = torch.linspace(
                start=0, end=self.ego_frame - 1, steps=self.ego_frame
            ).type_as(features)
            b_ = torch.linspace(
                start=0, end=self.cav_frame - 1, steps=self.cav_frame
            ).type_as(features)
            c_ = torch.zeros((raw_batch_size, one_batch)).type_as(features)

            for batch in range(raw_batch_size):
                c_[batch][: self.ego_frame] = a_ + delay[batch][0]
                for i in range(1, len(delay[batch])):
                    c_[batch][
                        self.ego_frame
                        + (i - 1) * self.cav_frame : self.ego_frame
                        + i * self.cav_frame
                    ] = (b_ + delay[batch][i])
            c_ = c_.reshape(-1, 1).repeat(1, self.deform_dim)
            i = torch.linspace(0, self.deform_dim - 1, steps=self.deform_dim).type_as(
                features
            )
            emb = c_ / (8 ** (2 * i / self.deform_dim))  # *100
            semb, cemb = torch.sin(emb), torch.cos(emb)
            for i in range(self.deform_dim):
                c_[:, i] = semb[:, i] if i % 2 == 0 else cemb[:, i]
            c_ = c_.reshape(raw_batch_size, one_batch, -1, 1, 1).repeat(
                1, 1, 1, features.shape[-2], features.shape[-1]
            )

            c__ = []
            for batch in range(len(record_len)):
                c__.append(c_[batch][: record_len[batch]])
            c_ = torch.cat(c__, dim=0)
            # filter coords that has points
            if coords is not None:
                b, c, h, w = c_.shape
                mask = torch.zeros((b, h, w)).type_as(features)
                stride = self.cfg["model"]["deform"]["input_stride"][0]
                coords[:, 2:] = coords[:, 2:] // stride
                mask[coords[:, 0], coords[:, 2], coords[:, 3]] = 1
                mask = mask.reshape(-1, 1, h, w)
                c_ = c_ * mask

        return torch.cat((features, c_), dim=1)

    def communicate(self, x):
        if self.compression:
            x = self.naive_compressor(x)
        _, c, h, w = x.shape
        x = x.reshape(self.max_cav, -1, c, h, w).transpose(0, 1)
        return x
