from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
# MRF module for V2V fusion
import warnings
import math
from ast import Tuple
import copy
from tracemalloc import start
import numpy as np
from typing import List
from numpy.core.fromnumeric import shape
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_
from models.modules.fusion.feature_selection import MostInformativeFeaSelection

# document :gate mechanism -----------> adapive fusion
#           fusion extraction-----------------> avg (all cavs)
#           history deformable attention--------> avg (all cavs) deformable 
#           pyramid features aggression add multi scale detail
#           adap fusion add multi scale detail
#           multi scale flatten feature communication

class IPSA(nn.Module):
    def __init__(self, in_channels:int, num_levels:int, scale_list:list, use_pyramid_conv:bool):
        super(IPSA, self).__init__()
        self.n_levels       = num_levels #3
        self.in_channels    = in_channels #128
        func_unflatten_dict = {}
        for lvl, scale_size in enumerate(scale_list):
            scale = 2**(lvl+1)
            func_unflatten_dict[f'unflatten_{scale}x'] = nn.Unflatten(dim=-1, unflattened_size=scale_size)
        self.func_unflatten_dict = nn.ModuleDict(func_unflatten_dict)
        self.use_pyramid_conv = use_pyramid_conv
        if self.use_pyramid_conv:
            if self.n_levels == 3:
                encode_channel      = int(self.in_channels//3)
                self.fusion_encoder = nn.Conv2d(self.in_channels, encode_channel, kernel_size=3, stride=1, padding=1)

                self.block1_dec2x = nn.MaxPool2d(kernel_size=2)   ### C=64
                self.block1_dec4x = nn.MaxPool2d(kernel_size=4)   ### C=64

                self.block2_dec2x = nn.MaxPool2d(kernel_size=2)  ### C=128
                self.block2_inc2x = nn.ConvTranspose2d(encode_channel, encode_channel, kernel_size=2, stride=2)

                self.block3_inc2x = nn.ConvTranspose2d(encode_channel, encode_channel, kernel_size=2, stride=2)
                self.block3_inc4x = nn.ConvTranspose2d(encode_channel, encode_channel, kernel_size=4, stride=4)
            
            else:
                raise Exception()

            ms_fea_agg_list = [nn.Conv2d(encode_channel*self.n_levels, self.in_channels, kernel_size=3, stride=1, padding=1) for i in range(self.n_levels)]
            self.ms_fea_agg = torch.nn.ModuleList(ms_fea_agg_list)
        # Weight initialization
        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    constant_(m.bias.data, 0)
    
    # the ego agnet's multi-scale features, the collaborator's multi-scale features
    def forward(self, ego_agent_fea:dict, collaborator_fea:dict):
        #torch_size (B, H*W, C) --> (B, C, H*W) --> (B, C, H, W)
        collaborator_MS_2d    = [self.func_unflatten_dict[f'unflatten_{2**(lvl+1)}x'](torch.transpose(collaborator_fea[f'temporalAttn_result_{2**(lvl+1)}x'], 1,2)) for lvl in range(self.n_levels)]

        sampling_feature_list = []
        agents_max_feature_list = []

        for lvl in range(self.n_levels):
            fusion_feature = torch.stack([ego_agent_fea[lvl], collaborator_MS_2d[lvl]], dim=1)
            # initial fusion
            max_fusion_feature = torch.max(fusion_feature, dim=1)[0]
            agents_max_feature_list.append(max_fusion_feature)  
            if self.use_pyramid_conv:
                pyramid_sampling_feature = self.fusion_encoder(max_fusion_feature)
                sampling_feature_list.append(pyramid_sampling_feature)

        if self.use_pyramid_conv:
            fused_feature_list = []
            if self.n_levels == 3:
                for lvl in range(self.n_levels):
                    if lvl == 0:
                        fused_feature = self.ms_fea_agg[lvl](
                                                        torch.cat([sampling_feature_list[lvl], self.block2_inc2x(sampling_feature_list[lvl+1]), self.block3_inc4x(sampling_feature_list[lvl+2])],dim=1))
                        # Improment: add initial BEV feature of the collaborator
                        fused_feature = fused_feature + agents_max_feature_list[lvl]
                        fused_feature_list.append(fused_feature)       
                    elif lvl == 1:
                        fused_feature = self.ms_fea_agg[lvl](
                                                        torch.cat([self.block1_dec2x(sampling_feature_list[lvl-1]), sampling_feature_list[lvl], self.block3_inc2x(sampling_feature_list[lvl+1])],dim=1))
                        fused_feature = fused_feature + agents_max_feature_list[lvl]
                        fused_feature_list.append(fused_feature)
                    elif lvl == 2:
                        fused_feature = self.ms_fea_agg[lvl](
                                                        torch.cat([self.block1_dec4x(sampling_feature_list[lvl-2]), self.block2_dec2x(sampling_feature_list[lvl-1]), sampling_feature_list[lvl]],dim=1))
                        fused_feature = fused_feature + agents_max_feature_list[lvl]
                        fused_feature_list.append(fused_feature)
            # fused_motion_feature_list:[(B,C,352,200), (B,C,176,100),(B,C,88,50)] M_tilda
            else:
                raise Exception()
            assert len(fused_feature_list) == self.n_levels
            return fused_feature_list
        else:
            return agents_max_feature_list


class MSRobustFusion(nn.Module):
    def __init__(self, num_agents, mfgs):
        super(MSRobustFusion, self).__init__()
        self.mfgs = mfgs
        self.num_agents  = num_agents
        self.d_model = mfgs['configs_com']['d_model']
        self.rounds_com   = mfgs['configs_com']['rounds_com']
        # 352*100
        self.target_resolution  = mfgs['config_feature']['target_resolution']
        self.scale_level   = mfgs['config_feature']['scale_level']
        self.target_channel   = mfgs['config_feature']['target_channel']
        
        self.channel_list = [int(self.target_channel * ratio) for ratio in [0.5, 1, 2]]
        self.BEV_feature_resolution_list = [[int(i * ratio) for i in self.target_resolution] for ratio in [2, 1, 0.5]] 
        #[[352, 100], [176, 50], [88, 25]]
        
        self.dropout_prob = mfgs["config_fusion_block"]["dropout"]
        self.num_head = mfgs["config_fusion_block"]["num_head"]
        self.num_sampling_points = mfgs["config_fusion_block"]["num_sampling_points"]
        self.dim_feedforward = mfgs["config_fusion_block"]["dim_feedforward"]
        
        BEV_spatial_shape_tuple_list = []
        for _, (size_h, size_w) in enumerate(self.BEV_feature_resolution_list):
            BEV_spatial_shape_tuple_list.append([size_h, size_w])    
        self.BEV_spatial_shape_tuple_list = BEV_spatial_shape_tuple_list
        
        # initialization for channel compression
        multiscale_src_proj_layer_list = []
        for lvl, input_channel in enumerate(self.channel_list):
            multiscale_src_proj_layer_list.append(nn.Conv2d(input_channel, self.d_model, kernel_size=1, stride=1,padding=0))
        self.multiscale_src_proj_layers = nn.ModuleList(multiscale_src_proj_layer_list)
        # communication setting
        self.use_com = mfgs['use_com'] if 'use_com' in mfgs else False
        self.comm_args = mfgs['comm_args'] if self.use_com else None

        # initialization for multiple rounds communication
        # (self, num_agents,d_model,num_head,num_sampling_points,dropout_prob,dim_feedforward):
        self.use_pyramid_conv = mfgs['use_pyramid_conv'] if 'use_pyramid_conv' in mfgs else False
        self.use_msda = mfgs['use_msda'] if 'use_msda' in mfgs else False

        sinlg_round_com = SingleRoundCommunication(num_agents=self.num_agents,
                                           d_model=self.d_model,num_head=self.num_head,num_sampling_points=self.num_sampling_points,
                                           dropout_prob=self.dropout_prob,dim_feedforward=self.dim_feedforward, spatial_size=self.BEV_feature_resolution_list,
                                           comm_args = self.comm_args, use_pyramid_conv=self.use_pyramid_conv, use_msda=self.use_msda)
        self.rounds_communication  = RoundsCommunication(sinlg_round_com=sinlg_round_com, num_rounds=self.rounds_com, num_agents=self.num_agents, scale_level=self.scale_level)

        ## Initialization for upsampling layers
        self.deblocks = nn.ModuleList()
        upsample_strides = mfgs['config_feature']['upsample_strides']
        num_upsample_filters =  mfgs['config_feature']['num_upsample_filters']
        for lvl in range(self.scale_level):
            stride = upsample_strides[lvl]
            if stride >= 1:
                self.deblocks.append(nn.Sequential(
                    nn.ConvTranspose2d(in_channels=self.d_model, out_channels=num_upsample_filters[lvl],
                                        kernel_size=upsample_strides[lvl], stride=upsample_strides[lvl], bias=False),
                    nn.BatchNorm2d(num_upsample_filters[lvl], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                    )
                )
            else:
                stride = np.round(1/stride).astype(np.int)
                self.deblocks.append(nn.Sequential(
                    nn.Conv2d(in_channels=self.d_model, out_channels=num_upsample_filters[lvl],
                                kernel_size=stride, stride=stride, bias=False),
                    nn.BatchNorm2d(num_upsample_filters[lvl], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                    )
                )

        self.func_unflatten_dict = nn.ModuleDict()
        self.func_unflatten_dict['unflatten_2x'] = nn.Unflatten(dim=-1, unflattened_size=self.BEV_feature_resolution_list[0])
        self.func_unflatten_dict['unflatten_4x'] = nn.Unflatten(dim=-1, unflattened_size=self.BEV_feature_resolution_list[1])
        self.func_unflatten_dict['unflatten_8x'] = nn.Unflatten(dim=-1, unflattened_size=self.BEV_feature_resolution_list[2])

    @staticmethod
    def get_reference_points_DUCA(batch_size, spatial_shape_temporalAttn, n_level_attention, device):
        valid_ratios = torch.ones(batch_size,len(spatial_shape_temporalAttn),2).to(device)

        ref_points_deformAttn_list    = []
        for lvl, (H_, W_) in enumerate(spatial_shape_temporalAttn):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                        torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), dim=-1)
            ref_points_deformAttn_list.append(ref)
        
        ref_points_deformAttn = torch.cat(ref_points_deformAttn_list, dim=1)
        ref_points_deformAttn = ref_points_deformAttn[:,:,None] * valid_ratios[:, None]

        return ref_points_deformAttn
    

    def forward(self, batch_dict_list, record_len):
        target_batch_dict = batch_dict_list[0]  #ego_cav
        cur_device = target_batch_dict['src_features_for_align_2x'].device
        #batch_size:torch.size([Batch, self.num_bev_features*self.nz , self.ny , self.nx])
        batch_size = target_batch_dict['src_features_for_align_2x'].shape[0]

        ## Set up for dual-query
        feature_dict_list = [{} for i in range(self.num_agents)]
        #batch_dict_list: [{frame1}, {frame2}]
        # frame1:{spatial_features, src_features_for_align_2x:[torch_size b 64 h w ], src_features_for_align_4x, src_features_for_align_8x,}
        for t, batch_dict in enumerate(batch_dict_list):
            for lvl in range(self.scale_level):
                scale = 2**(lvl+1)
                #project to channels=128
                feature_dict_list[t][f'proj_align_src_{scale}x'] = self.multiscale_src_proj_layers[lvl](batch_dict[f'src_features_for_align_{scale}x'])
        #feature_dict_list:[{frame1}, {frame2}, {target}]
        #frame1[proj_align_src_2x] torch_size(b, c, h, w)
        
        ## Set up for reference points
        #ref_points_deformAttn:torch.Size([4, 46200, 3, 2])
        ref_points_deformAttn = self.get_reference_points_DUCA(batch_size=batch_size, spatial_shape_temporalAttn=self.BEV_spatial_shape_tuple_list,n_level_attention=self.scale_level, device=cur_device)
        ## Set up for spatial shape
        # BEV_spatial_shape_tensor: [(88,25), (176, 50), (352, 100)]
        BEV_spatial_shape_tensor = torch.as_tensor(self.BEV_spatial_shape_tuple_list, dtype=torch.long, device=cur_device)
        
        ## Set up for level start index
        temporal_level_start_idx = [0]
        for lvl, (size_h, size_w) in enumerate(self.BEV_feature_resolution_list, start=1):
            if lvl != self.scale_level:
                temporal_level_start_idx.append(size_h*size_w)
            else:
                pass
        assert len(temporal_level_start_idx) == self.scale_level
        temporal_level_start_idx = torch.as_tensor(temporal_level_start_idx, dtype=torch.long, device=cur_device).cumsum(dim=0)

        ## Dual-query based co-attention
        encoded_feature, total_comm_volumes = self.rounds_communication(   feature_dict_list   = feature_dict_list,
                                                spatial_shapes      = BEV_spatial_shape_tensor,
                                                level_start_idx     = temporal_level_start_idx,
                                                ref_points          = ref_points_deformAttn,
                                                record_len          = record_len)

        ups = []
        for lvl in range(self.scale_level):
            scale = 2**(lvl+1)
            ups.append(self.deblocks[lvl](self.func_unflatten_dict[f'unflatten_{scale}x'](torch.transpose(encoded_feature[0][f'spatialAttn_result_{scale}x'], 1,2))))
        # (b,128*3,176,50)
        target_batch_dict['aggregated_spatial_features_2d'] = torch.cat(ups, dim=1)
        total_comm_volume = sum(total_comm_volumes)  # set the communication rounds
        return target_batch_dict, total_comm_volume

class MSEnhancedAggre(nn.Module):
    def __init__(self, num_levels, d_model, num_head, num_sampling_points, dropout_prob, dim_feedforward, spatial_list, use_pyramid_conv, use_msda):
        super(MSEnhancedAggre, self).__init__()
        self.n_levels = num_levels
        # ablation experiments
        self.use_multi_scale = True
        self.use_msda = use_msda
        if self.use_multi_scale:
            self.use_pyramid_conv = use_pyramid_conv
            self.ipsa_module = IPSA(in_channels=d_model, num_levels=self.n_levels, scale_list=spatial_list, use_pyramid_conv= self.use_pyramid_conv)
        else:
            pass
        
        self.deform_attn    = MSDeformAttn(d_model, self.n_levels, num_head, num_sampling_points)
        self.dropout1       = nn.Dropout(dropout_prob)
        self.norm1          = nn.LayerNorm(d_model)
        self.linear1        = nn.Linear(d_model, dim_feedforward)
        self.activation     = nn.ReLU()
        self.dropout2       = nn.Dropout(dropout_prob)
        self.linear2        = nn.Linear(dim_feedforward, d_model)
        self.dropout3       = nn.Dropout(dropout_prob)
        self.norm2          = nn.LayerNorm(d_model)
        

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src     
       
    # communicate with one agent
    def forward(self, ego_feature_dict, collaborator_feature_dict, ref_points, spatial_shapes, level_start_idx):
        if self.use_multi_scale:
            # enhanced_feature_list:M_tilda [(B,C,352, 100), (B,C,176, 50),(B,C,88,25)]
            enhanced_feature_list = self.ipsa_module(ego_feature_dict, collaborator_feature_dict)
            deform_query_list_flatten   = [enhanced_feature.flatten(2).transpose(1,2) for enhanced_feature in enhanced_feature_list]
        else:
            deform_query_list_flatten   = [ego_feature_dict[f'spatialAttn_result_{2**(idx+1)}x'] for idx in range(self.n_levels)]
        if self.use_msda:
            deform_query_set = torch.cat(deform_query_list_flatten, dim=1)  # to multi-scale
            
            key_flatten = deform_query_set
            # multi-scale deformable attention module
            output          = self.deform_attn(deform_query_set, ref_points, key_flatten, spatial_shapes, level_start_idx)
            attn_result     = self.norm1(key_flatten + self.dropout1(output))
            attn_result = self.norm1(key_flatten)
            #(b,(352*200)+(176*100)+(88*50),128)
            return self.forward_ffn(attn_result)
        else:
            return torch.cat(deform_query_list_flatten, dim=1)
        
class CollaFeaAggregate(nn.Module):
    def __init__(self, d_model, num_agents, size):
        super(CollaFeaAggregate, self).__init__()
        self.in_channel = d_model
        self.num_agents = num_agents
        self.gating_map_channel = 1
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.conv_gating = nn.ModuleList()
        self.conv_fusion = nn.ModuleList()
        for t in range(self.num_agents-1):
            self.conv_gating.append(nn.Conv2d(self.in_channel*2, self.gating_map_channel, kernel_size=3, padding=1))
            self.conv_fusion.append(nn.Conv2d(self.in_channel, self.in_channel, kernel_size=3, padding=1))
        self.agg_all_time = nn.Conv2d(self.in_channel * (self.num_agents-1), self.in_channel, kernel_size=3, padding=1)

        self.func_unflatten = nn.Unflatten(dim=-1, unflattened_size=size)
        
        # spatial adaptive fusion module
        self.conv3d = nn.Sequential(
            nn.Conv3d(2, 1, 3, stride=1, padding=1),
            # nn.BatchNorm3d(1, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True)
        )

    # fusion all the agents' features but different scale
    def forward(self, feature_dict_list, scale):

        # store the feature of the scale
        features_total_list = []
        # feature of the ego agent
        features_total_list.append(self.func_unflatten(torch.transpose(feature_dict_list[0][f'spatialAttn_result_{scale}x'],1,2)))
        # feature of the comm agent
        features_total_list = features_total_list[:] + [self.func_unflatten(torch.transpose(feature_dict_list[t][f'temporalAttn_result_{scale}x'],1,2)) for t in range(1, self.num_agents)]
        features_total = torch.cat([feature.unsqueeze(1) for feature in features_total_list], dim=1)
        features_fusion = []
        # spatial-wise adaptive fusion
        for batch_index in range(features_total.shape[0]):
            features_max = torch.max(features_total[batch_index], dim=0, keepdim=True)[0]
            features_mean = torch.mean(features_total[batch_index], dim=0, keepdim=True)
            features_fusion.append(torch.cat((features_max, features_mean), dim=0).unsqueeze(0))
        features_fusion = torch.cat(features_fusion, dim=0)
        features_fusion = self.conv3d(features_fusion).squeeze(1)  #tensor_size[2, 128, 352, 100]
        # lateral connection
        features_fusion = features_fusion + features_total_list[0] 
        # features_fusion:(B,H*W,C)
        return features_fusion.flatten(2).transpose(1,2)

class SingleRoundCommunication(nn.Module):
    def __init__(self, num_agents, d_model,num_head,num_sampling_points,dropout_prob,dim_feedforward, spatial_size, comm_args, use_pyramid_conv, use_msda):
        super(SingleRoundCommunication, self).__init__()
        self.num_agents = num_agents #5
        self.n_spatial_levels    = 3
        # add comm module
        func_unflatten_dict = {}
        for lvl, scale_size in enumerate(spatial_size):
            scale = 2**(lvl+1)
            func_unflatten_dict[f'unflatten_{scale}x'] = nn.Unflatten(dim=-1, unflattened_size=scale_size)
        self.func_unflatten_dict = nn.ModuleDict(func_unflatten_dict)
        # confidence map generator
        self.comm_args = comm_args
        if self.comm_args:
            self.flatten_feature_map_generator = MostInformativeFeaSelection(comm_args)
        # initialization for MEA module 
        num_collaborators = self.num_agents - 1  # 4
        MEA_modules = [MSEnhancedAggre(num_levels=self.n_spatial_levels,
                            d_model=d_model, num_head=num_head, 
                            num_sampling_points=num_sampling_points, 
                            dropout_prob = dropout_prob, dim_feedforward=dim_feedforward, 
                            spatial_list=spatial_size, 
                            use_pyramid_conv=use_pyramid_conv, use_msda=use_msda) for _ in range(num_collaborators)]
        self.MEA_modules = nn.ModuleList(MEA_modules)
        
        # initialization for CFA module
        spatial_size = spatial_size
        IGANet_module_list = [CollaFeaAggregate(d_model, num_agents=self.num_agents, size=spatial_size[lvl]) for lvl in range(self.n_spatial_levels)]
        self.IGANet_module_list = nn.ModuleList(IGANet_module_list)
        
    def forward(self, feature_dict_list, ref_points, spatial_shapes, level_start_idx, record_len):
        ego_feature_dict = feature_dict_list[0]  # the ego_cav's feature
        # calculate the communication volumn in single round
        single_round_comm_volume = []
        ego_flatten_features = torch.cat([ego_feature_dict[f'spatialAttn_result_{2**(lvl+1)}x'] for lvl in range(self.n_spatial_levels)], dim=1)
        # feature selection tech of the ego agent

        if self.comm_args:
            ego_key_spatial_flatten, ego_comm_volume = self.flatten_feature_map_generator(ego_flatten_features)
            single_round_comm_volume.append(ego_comm_volume)
        else:
            ego_key_spatial_flatten = ego_flatten_features
            single_round_comm_volume.append(0)
        # represent in multi-scale
        ego_key_spatial_list = []
        for lvl in range(self.n_spatial_levels):
            if lvl != (self.n_spatial_levels-1):
                # from big size to small
                ego_key_spatial_list.append(ego_key_spatial_flatten[:, level_start_idx[lvl]:level_start_idx[lvl+1], :])
            else:
                ego_key_spatial_list.append(ego_key_spatial_flatten[:, level_start_idx[lvl]:, :])
        ego_key_spatial_MS = [self.func_unflatten_dict[f'unflatten_{2**(lvl+1)}x'](torch.transpose(ego_key_spatial_list[lvl], 1,2)) for lvl in range(self.n_spatial_levels)]
        # Multi-scale Enhanced Aggregation
        for agent_idx, feature_dict in enumerate(feature_dict_list):
            if (agent_idx == 0):  # skip the ego agent
                continue
            else:
                #deform_query_set:(B, (352*200)+(176*100)+(88*50), C)
                output = self.MEA_modules[agent_idx-1](ego_key_spatial_MS, feature_dict, ref_points, spatial_shapes, level_start_idx)
                #IDANet结束后， 通信智能体进行通信， 然后进行IGANet
                if self.comm_args:
                    agent_key_spatial_flatten, agent_comm_volume = self.flatten_feature_map_generator(output)
                    single_round_comm_volume.append(agent_comm_volume)
                else:
                    agent_key_spatial_flatten = output
                    single_round_comm_volume.append(0)
                for lvl in range(self.n_spatial_levels):
                    scale = 2**(lvl+1)
                    if lvl != (self.n_spatial_levels-1):
                        #from big size to small
                        feature_dict[f'temporalAttn_result_{scale}x'] = agent_key_spatial_flatten[:, level_start_idx[lvl]:level_start_idx[lvl+1], :]
                    else:
                        feature_dict[f'temporalAttn_result_{scale}x'] = agent_key_spatial_flatten[:, level_start_idx[lvl]:, :]
           
        # Collaborative Features Aggregation
        for lvl in range(self.n_spatial_levels):
            scale = 2**(lvl+1)
            # agents_comm_volume:[bs, 5, H, W]
            fusion_output = self.IGANet_module_list[lvl](feature_dict_list, scale)
            ego_feature_dict[f'spatialAttn_result_{scale}x'] = fusion_output
        # calculate the communication volumn
        if self.comm_args:
            single_round_comm_volume = torch.stack(single_round_comm_volume, dim=1)
            single_round_comm_volume = [torch.sum(single_round_comm_volume[index,:value]).item() for index, value in enumerate(record_len)]
        single_round_comm = sum(single_round_comm_volume) / len(single_round_comm_volume)  # get communication volumn in a batchs
        return feature_dict_list, single_round_comm

class RoundsCommunication(nn.Module):
    def __init__(self, sinlg_round_com, num_rounds, num_agents, scale_level):
        super(RoundsCommunication, self).__init__()
        self.communications     =  _get_clones(sinlg_round_com, num_rounds) # clone single layer
        self.num_agents = num_agents # 5
        self.scale_level = scale_level # 3

    def forward(self, feature_dict_list, spatial_shapes, level_start_idx, ref_points, record_len):
        # feature_dict_list: the features of the agents
        # init:feature_dict_list[t][f'proj_align_src_{scale}x']  (b,c,h,w)--->(b,h*w,C)
        for agent_idx, feature_dict in enumerate(feature_dict_list, start=1):
            for lvl in range(self.scale_level):
                scale = 2**(lvl+1)
                # the ego cav
                if agent_idx == 1:
                    if feature_dict.get(f'spatialAttn_result_{scale}x') == None: 
                        feature_dict[f'spatialAttn_result_{scale}x'] = feature_dict[f'proj_align_src_{scale}x'].flatten(2).transpose(1,2)
                        # copy to different name
                        feature_dict.pop(f'proj_align_src_{scale}x')
                else:
                    if feature_dict.get(f'temporalAttn_result_{scale}x') == None: 
                        feature_dict[f'temporalAttn_result_{scale}x'] = feature_dict[f'proj_align_src_{scale}x'].flatten(2).transpose(1,2)
                        feature_dict.pop(f'proj_align_src_{scale}x')                    
        total_comm_volumes = []
        for _, communication in enumerate(self.communications):
            feature_dict_list, single_round_comm_volume = communication(feature_dict_list, ref_points, spatial_shapes, level_start_idx, record_len)
            total_comm_volumes.append(single_round_comm_volume)
        return feature_dict_list, total_comm_volumes

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------



def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    def __init__(self, d_model, n_levels, n_heads, n_points):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2) #192
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)
    @staticmethod    
    def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
        # for debug and test only,
        # need to use cuda version instead
        N_, S_, M_, D_ = value.shape
        _, Lq_, M_, L_, P_, _ = sampling_locations.shape
        value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
        sampling_grids = 2 * sampling_locations - 1
        sampling_value_list = []
        for lid_, (H_, W_) in enumerate(value_spatial_shapes):
            # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
            value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, H_, W_)
            # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
            sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
            # N_*M_, D_, Lq_, P_
            sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                            mode='bilinear', padding_mode='zeros', align_corners=False)
            sampling_value_list.append(sampling_value_l_)
        # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
        attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, L_*P_)
        output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_*D_, Lq_)
        return output.transpose(1, 2).contiguous()
    
    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements
        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        #(value, value_spatial_shapes, sampling_locations, attention_weights):
        output = self.ms_deform_attn_core_pytorch(
            value, input_spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)
        return output

