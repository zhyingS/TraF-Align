import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.ops import DeformConv2d
# input size: (bs, 256, 100, 352)
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # avg pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # max pooling
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # channel-wise attention
        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False) #kernel_size=1
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # add
        out = avg_out + max_out
        return self.sigmoid(out)

# spatial-wise attention 
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        # same padding
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # avg pool
        max_out, _ = torch.max(x, dim=1, keepdim=True) # max pool
        # concatenation
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
    
class MotionEnhancedMech(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        bottleneck_size = [384, 128]
        self.reduce_dim_z = BasicConv2d(input_size*2, bottleneck_size[0], kernel_size=1, padding=0)
        self.s_atten_z = SpatialAttention()
        self.c_atten_z = ChannelAttention(bottleneck_size[0])
        deform_groups = 32
        kernel_size = 3
        self.motion_offset = nn.Conv2d(input_size, 2*kernel_size*kernel_size*deform_groups, kernel_size=3, padding=1)
        self.modion_dconv = DeformConv2d(input_size, input_size, kernel_size=kernel_size, padding=1, deform_groups=deform_groups)

        self.fusion_offset = nn.Conv2d(input_size, 2*kernel_size*kernel_size*deform_groups, kernel_size=3, padding=1)
        self.fusion_dconv = DeformConv2d(input_size, input_size, kernel_size=kernel_size, padding=1, deform_groups=deform_groups)
        self.relu = nn.ReLU()
        
    def generate_attention_z(self, x):
        z = self.reduce_dim_z(x) #[1, 384, 100, 352]
        atten_s = self.s_atten_z(z.mean(dim=1, keepdim=True)).view(z.size(0), -1, z.size(2), z.size(3))
        atten_c = self.c_atten_z(z)
        z = F.sigmoid(atten_s * atten_c)
        return z, 1 - z
    
    def regroup(self, x, record_len):
        cum_sum_len = np.cumsum(record_len).tolist()
        split_x = torch.tensor_split(x, cum_sum_len[:-1])
        return split_x

    def forward(self, bev_features_2d, record_len, his_frames_lens):
        bev_features_2d_ = self.regroup(bev_features_2d, record_len)

        batch_temporal_fusion = []
        for i, batch_bev in enumerate(bev_features_2d_):
            if his_frames_lens[i] == 0:
                if batch_bev.shape[0]>0:
                    batch_temporal_fusion.append(batch_bev[0].unsqueeze(0))
                else:
                    _, C_, H_, W_ = batch_bev.shape
                    batch_temporal_fusion.append(torch.zeros((1,C_,H_,W_)).type_as(batch_bev))
                continue
            if batch_bev.shape[0] <=1:
                _, C_, H_, W_ = batch_bev.shape
                batch_temporal_fusion.append(torch.zeros((1,C_,H_,W_)).type_as(batch_bev))
                continue
            
            x = batch_bev[1:his_frames_lens[i]+1]  # (num_frames, c, h, w)
            depth, num_channels, h, w = x.shape # depth: num_frames

            res = torch.cat((x[0].unsqueeze(0), x), dim=0)  #torch.Size([num_frames+1, 384, 100, 352])
            # res：[t1, t1, t2, t3, tn-1]
            pre = res[:-1]
            # motion:  [t1-t1, t2-t1, t3-t2, ..., tn-tn-1]
            res = x - pre
            motion_offsets = self.motion_offset(res)
            motion_features = self.relu(self.modion_dconv(pre, motion_offsets))
            h = x[0] # refined the feature of the first frame
            for t in range(depth):
                # h(refined feature):torch.Size([1, 512, 100, 352])
                con_fea = torch.cat((h, motion_features[t]), dim=0).unsqueeze(0)  #initialize t=0 -----> 0
                z_p, z_r = self.generate_attention_z(con_fea)
                h = z_r * h + z_p * motion_features[t]

                fusion_offset = self.fusion_offset(h)
                fusion_dconv = self.relu(self.fusion_dconv(h, fusion_offset))
                h = fusion_dconv.squeeze(0)
            batch_temporal_fusion.append(h.unsqueeze(0))
        fea_t = torch.cat(batch_temporal_fusion, dim=0)  #channel层concat

        return fea_t
    
    def forward_(self, bev_features_2d, record_len, his_frames_lens):
        start_idx = [0] + record_len[:-1]
        curren_bev_features = bev_features_2d[start_idx]
        batch_size, C, H, W = curren_bev_features.shape
        # historical BEV feature
        his_bev_start_idx = sum(record_len)
        his_bev_end_idx = sum(record_len+his_frames_lens)
        his_bev = bev_features_2d[his_bev_start_idx: his_bev_end_idx]
        # [(frames1, c, h, w), (frames2, c, h, w)]
        batch_his_bev = self.regroup(his_bev, his_frames_lens)
        # input_size: (bs, num_frames, C, H, W)

        batch_temporal_fusion = []
        for i, his_bev in enumerate(batch_his_bev):
            if his_frames_lens[i] == 0:
                batch_temporal_fusion.append(curren_bev_features[i].unsqueeze(0))
                continue
            x = his_bev  # (num_frames, c, h, w)
            depth, num_channels, h, w = x.shape # depth: num_frames

            res = torch.cat((x[0].unsqueeze(0), x), dim=0)  #torch.Size([num_frames+1, 384, 100, 352])
            # res：[t1, t1, t2, t3, tn-1]
            pre = res[:-1]
            # motion:  [t1-t1, t2-t1, t3-t2, ..., tn-tn-1]
            res = x - pre
            motion_offsets = self.motion_offset(res)
            motion_features = self.relu(self.modion_dconv(pre, motion_offsets))
            h = x[0] # refined the feature of the first frame
            for t in range(depth):
                # h(refined feature):torch.Size([1, 512, 100, 352])
                con_fea = torch.cat((h, motion_features[t]), dim=0).unsqueeze(0)  #initialize t=0 -----> 0
                z_p, z_r = self.generate_attention_z(con_fea)
                h = z_r * h + z_p * motion_features[t]

                fusion_offset = self.fusion_offset(h)
                fusion_dconv = self.relu(self.fusion_dconv(h, fusion_offset))
                h = fusion_dconv.squeeze(0)
            batch_temporal_fusion.append(h.unsqueeze(0))
        fea_t = torch.cat(batch_temporal_fusion, dim=0)  #channel层concat

        return fea_t


if __name__ == "__main__":
    bev_features_2d = torch.ones(8, 384, 100, 352)
    record_len = [2, 3]
    his_frames_lens = [0, 3]
    model = MotionEnhancedMech(input_size=384, hidden_size=384)
    model(bev_features_2d, record_len, his_frames_lens)