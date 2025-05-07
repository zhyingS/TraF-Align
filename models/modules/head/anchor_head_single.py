import numpy as np
import torch.nn as nn

class AnchorHeadSingle(nn.Module):
    def __init__(self,cfg,input_channels=None,**kwargs):
        super(AnchorHeadSingle, self).__init__()
        if input_channels is None:
            if 'shrink_header' in cfg['model']:
                input_channels = cfg['model']['shrink_header']['dim'][-1]
            else:   
                input_channels = sum(cfg['model']['backbone']['num_upsample_filters'])

        num_class = len(cfg['dataset']['cls_group'])

        self.cfg = cfg
        code_size = 7
        if 'num_anchors_per_location' in cfg['model']['head']:
            num_anchors_per_location = cfg['model']['head']['num_anchors_per_location']
        else:
            num_anchors_per_location = 6
        
        self.conv_cls = nn.Conv2d(
            input_channels, num_anchors_per_location * num_class,
            kernel_size=1
        )
        
        self.conv_box = nn.Conv2d(
            input_channels, num_anchors_per_location * code_size,
            kernel_size=1
        )

        self.init_weights()


    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, spatial_features_2d, only_cls = False, prefix = ''):
        # spatial_features_2d = data_dict['spatial_features_2d']

        cls_preds = self.conv_cls(spatial_features_2d)
        if only_cls:
            return cls_preds
        else:
            box_preds = self.conv_box(spatial_features_2d)

            cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
            box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
            
            pred_dict={}
            pred_dict[f'cls_preds{prefix}'] = cls_preds
            pred_dict[f'box_preds{prefix}'] = box_preds

            return pred_dict
       
       
