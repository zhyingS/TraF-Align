import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
import spconv
import spconv.pytorch

def regroup(data_dict, record_len, max_len, feature_name):
    #ego his信息是放在data_dict['spatial_features']最后
    cum_sum_len = torch.cumsum(record_len, dim=0)
    dense_feature = data_dict[feature_name][:cum_sum_len[-1]]
    split_features = torch.tensor_split(dense_feature,
                                        cum_sum_len[:-1].cpu())

    his_lens = [data_dict['ego_frames']-1] * len(record_len)
    mask = []
    regroup_features = []
    for i_,split_feature in enumerate(split_features):
        B, C, H, W = split_feature.shape
        if B == 0:
            split_feature = torch.zeros((max_len,C,H,W)).type_as(split_feature)
        else:
            cav_feature = split_feature[his_lens[i_]+1:]
            if cav_feature.shape[0]>0:
                split_feature = torch.cat((split_feature[0][None,...], split_feature[his_lens[i_]+1:]),dim=0)
            else:
                split_feature = split_feature[0][None,...]
        # M, C, H, W
        feature_shape = split_feature.shape
        # the maximum M is 5 as most 5 cavs
        padding_len = max_len - feature_shape[0]
        mask.append([1] * feature_shape[0] + [0] * padding_len)
        ego_feature = split_feature[0].unsqueeze(0)
        padding_tensor = ego_feature.expand(padding_len,*feature_shape[1:]) 
        padding_tensor = padding_tensor.to(split_feature.device)

        split_feature = torch.cat([split_feature, padding_tensor],
                                  dim=0)

        # 1, 5C, H, W  5*64
        split_feature = split_feature.view(-1,
                                           feature_shape[2],
                                           feature_shape[3]).unsqueeze(0)
        regroup_features.append(split_feature)

    # B, 5C, H, W
    regroup_features = torch.cat(regroup_features, dim=0)
    # B, L, C, H, W========>([2, 5, 64, 100, 352])
    regroup_features = rearrange(regroup_features,
                                 'b (l c) h w -> b l c h w',
                                 l=max_len)
    for i in range(regroup_features.shape[1]):
        data_dict['align_features'][i]['src_features_for_align_'+feature_name[-2:]] = regroup_features[:,i,:,:,:]
    
    mask = torch.from_numpy(np.array(mask)).to(regroup_features.device)

    return data_dict, mask


class MRCNetBEVBackbone(nn.Module):
    def __init__(self, cfg, input_channels):
        super().__init__()
        self.cfg = cfg
        self.model_cfg = cfg['model']['backbone']
        self.max_cav = cfg['train_params']['max_cav']
        self.one_batch = cfg['dataset']['frame_his'] + cfg['dataset']['cav_frame_his'] * (self.max_cav - 1)

        if 'layer_nums' in self.model_cfg:

            assert len(self.model_cfg['layer_nums']) == \
                   len(self.model_cfg['layer_strides']) == \
                   len(self.model_cfg['num_filters'])

            layer_nums = self.model_cfg['layer_nums']
            layer_strides = self.model_cfg['layer_strides']
            num_filters = self.model_cfg['num_filters']
        else:
            layer_nums = layer_strides = num_filters = []
        if 'upsample_strides' in self.model_cfg:
            assert len(self.model_cfg['upsample_strides']) \
                   == len(self.model_cfg['num_upsample_filters'])
            num_upsample_filters = self.model_cfg['num_upsample_filters'] #[128, 128, 128]
            upsample_strides = self.model_cfg['upsample_strides'] #[1, 2, 4]
        else:
            upsample_strides = num_upsample_filters = []  
                  
        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]

        self.blocks = nn.ModuleList()
        # for temporal mode
        self.deblocks = nn.ModuleList()

        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx],
                              kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])

            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx],
                                       eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3,
                                       momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters) #128*3
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1],
                                   stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in
 #====================================================       

    def forward(self, data_dict, input):
        pillar_features, coors, input_shape = input
        # if self.training:
        #     batch_size=self.cfg['train_params']['train_batch_size']
        # else:
        #     batch_size=self.cfg['train_params']['val_batch_size']
        # batch_size *= self.one_batch
        batch_size = torch.max(coors[:,0]) + 1

        spatial_features = spconv.pytorch.SparseConvTensor(
            pillar_features, coors, input_shape, batch_size)
        spatial_features=spatial_features.dense()
        data_dict['spatial_features'] = spatial_features

        x = spatial_features
        ups = []

        for i in range(len(self.blocks)):
            x = self.blocks[i](x)  #([18, 64, 80, 80])

            stride = int(spatial_features.shape[2] / x.shape[2])
            data_dict['spatial_features_%dx' % stride] = x
            
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)
        if len(ups) > 1:
            x = torch.cat(ups, dim=1)  #[18, 384, 80, 80]
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)
        data_dict['bev_features_2d'] = x  #([num_cavs, 384, 100, 352])  # add temporal info
                
        #torch.Size([7, 64, 352, 100]),torch.Size([7, 128, 176, 50]),torch.Size([7, 128, 88, 25])
        record_len = data_dict['record_len']
        max_len = self.max_cav
 
        # padding
        data_dict['align_features'] = [{} for i in range(max_len)]
        for feature_name in data_dict:
            if 'spatial_features_' in  feature_name:
                data_dict, mask = regroup(data_dict,record_len,max_len, feature_name)

        return data_dict



if __name__ == "__main__":
    model_cfg = {'base_bev_backbone':True,
                 'layer_nums': [3, 5, 8], 
                 'layer_strides':[2, 2, 2],
                 'num_filters':[64, 128, 256]}
    model = MRCNetBEVBackbone(model_cfg, 64)
    data_dict = {'spatial_features':torch.ones(7,64,704,400)}
    output = model(data_dict)
    record_len = torch.tensor([3,4])
    max_len = 5
    data_dict['align_features'] = [{} for i in range(max_len)]
    for feature_name in data_dict:
        if 'spatial_features_' in  feature_name:
            data_dict, mask = regroup(data_dict,record_len,max_len, feature_name)
    data_dict 
    
            
    