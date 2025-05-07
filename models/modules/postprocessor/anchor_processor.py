
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from utils import box_coder_utils, common_utils
from models.modules.head.target_assigner.anchor_generator import AnchorGenerator
from models.modules.head.target_assigner.axis_aligned_target_assigner import AxisAlignedTargetAssigner
from models.ops.iou3d_nms import iou3d_nms_cuda

class AnchorProcessor(nn.Module):

    def __init__(self, cfg):
        super(AnchorProcessor, self).__init__()
        
        self.cfg = cfg
        
        grid_size = cfg['voxelization']['grid_size']
        point_cloud_range=cfg['voxelization']['lidar_range']
        self.num_class = len(cfg['dataset']['cls_group'])
        self.class_names = [values for values in cfg['dataset']['cls_group']]
        
        model_cfg = cfg['model']['head']
        self.model_cfg = model_cfg
        self.use_multihead = model_cfg.get('use_multihead', False)
        
        anchor_target_cfg = model_cfg['target_assigner_config']
        self.box_coder = getattr(box_coder_utils, anchor_target_cfg['box_coder'])(
            num_dir_bins=anchor_target_cfg.get('num_dir_bins', 6),
            **anchor_target_cfg.get('box_coder_config', {})
        )
        anchor_generator_cfg = model_cfg['anchor_generator_config']
        anchors, self.num_anchors_per_location = self.generate_anchors(
            anchor_generator_cfg, grid_size=grid_size[[1,0,-1]], point_cloud_range=point_cloud_range,
            anchor_ndim=self.box_coder.code_size
        )
        self.num_anchors_per_location = sum(self.num_anchors_per_location)
        
        self.anchors = [x.cuda() for x in anchors]
        self.target_assigner = self.get_target_assigner(anchor_target_cfg)
        
        
    def forward(self, data_dict, preds_dicts, prefix = ''):
        # preds_dicts={'cls_preds':xxx,'box_preds':xxx}

        if self.training:
            batch_size=self.cfg['train_params']['train_batch_size']
        else:
            batch_size=self.cfg['train_params']['val_batch_size']
        self.batch_size=batch_size
        self.cfg['batch_size'] = batch_size
        
        if data_dict[f'label_dict{prefix}'].shape[0]>0 and data_dict[f'label_dict{prefix}'].shape[1]>0:
            # assign gt box to anchors 
            gt_box=data_dict[f'label_dict{prefix}'][:,0:9].clone() # (N, 9), b,x,y,z,h,w,l,phi,cls
            try:
                targets_dict = self.target_assigner.assign_targets(
                    self.anchors, gt_box, self.cfg
                )
            except:
                pass
            box_cls_labels = targets_dict['box_cls_labels']
            box_reg_targets = targets_dict['box_reg_targets']
        else:   
            anchors = torch.cat(self.anchors,dim=0)
            anchors = anchors.view(-1,anchors.shape[-1])
            N, M = anchors.shape
            box_cls_labels = torch.zeros((batch_size,N)).type_as(anchors)
            box_reg_targets = torch.zeros((batch_size,N,M)).type_as(anchors)
            
        forward_ret_dict = {
            'box_cls_labels': box_cls_labels,
            'cls_preds':preds_dicts[f'cls_preds{prefix}'],
            'box_preds':preds_dicts[f'box_preds{prefix}'],
            'dir_cls_preds':None,
            'box_reg_targets': box_reg_targets,
        }

        batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
            batch_size=self.batch_size,
            cls_preds=preds_dicts[f'cls_preds{prefix}'], box_preds=preds_dicts[f'box_preds{prefix}'], dir_cls_preds=None
        )
        forward_ret_dict['batch_cls_preds'] = batch_cls_preds
        forward_ret_dict['batch_box_preds'] = batch_box_preds
        forward_ret_dict['cls_preds_normalized'] = False
        
        return forward_ret_dict

    @staticmethod
    def generate_anchors(anchor_generator_cfg, grid_size, point_cloud_range, anchor_ndim=7):
        anchor_generator = AnchorGenerator(
            anchor_range=point_cloud_range,
            anchor_generator_config=anchor_generator_cfg
        )
        feature_map_size = [grid_size[:2] // config['feature_map_stride'] for config in anchor_generator_cfg]
        anchors_list, num_anchors_per_location_list = anchor_generator.generate_anchors(feature_map_size)

        if anchor_ndim != 7:
            for idx, anchors in enumerate(anchors_list):
                pad_zeros = anchors.new_zeros([*anchors.shape[0:-1], anchor_ndim - 7])
                new_anchors = torch.cat((anchors, pad_zeros), dim=-1)
                anchors_list[idx] = new_anchors

        return anchors_list, num_anchors_per_location_list


    def get_target_assigner(self, anchor_target_cfg):

        target_assigner = AxisAlignedTargetAssigner(
            model_cfg=self.model_cfg,
            class_names=self.class_names,
            box_coder=self.box_coder,
            match_height=anchor_target_cfg['match_height']
        )
        
        return target_assigner

    def generate_predicted_boxes(self, batch_size, cls_preds, box_preds, dir_cls_preds=None):
        """
        Args:
            batch_size:
            cls_preds: (N, H, W, C1)
            box_preds: (N, H, W, C2)
            dir_cls_preds: (N, H, W, C3)

        Returns:
            batch_cls_preds: (B, num_boxes, num_classes)
            batch_box_preds: (B, num_boxes, 7+C)

        """
        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat([anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1])
                                     for anchor in self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        num_anchors = anchors.view(-1, anchors.shape[-1]).shape[0]
        batch_anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        batch_cls_preds = cls_preds.view(batch_size, num_anchors, -1).float() \
            if not isinstance(cls_preds, list) else cls_preds
        batch_box_preds = box_preds.view(batch_size, num_anchors, -1) if not isinstance(box_preds, list) \
            else torch.cat(box_preds, dim=1).view(batch_size, num_anchors, -1)
        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, batch_anchors)

        if dir_cls_preds is not None:
            dir_offset = self.model_cfg['dir_offset']
            dir_limit_offset = self.model_cfg['dir_limit_offset']
            dir_cls_preds = dir_cls_preds.view(batch_size, num_anchors, -1) if not isinstance(dir_cls_preds, list) \
                else torch.cat(dir_cls_preds, dim=1).view(batch_size, num_anchors, -1)
            dir_labels = torch.max(dir_cls_preds, dim=-1)[1]

            period = (2 * np.pi / self.model_cfg['num_dir_bins'])
            dir_rot = common_utils.limit_period(
                batch_box_preds[..., 6] - dir_offset, dir_limit_offset, period
            )
            batch_box_preds[..., 6] = dir_rot + dir_offset + period * dir_labels.to(batch_box_preds.dtype)

        if isinstance(self.box_coder, box_coder_utils.PreviousResidualDecoder):
            batch_box_preds[..., 6] = common_utils.limit_period(
                -(batch_box_preds[..., 6] + np.pi / 2), offset=0.5, period=np.pi * 2
            )

        return batch_cls_preds, batch_box_preds

    def post_processing(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                                or [(B, num_boxes, num_class1), (B, num_boxes, num_class2) ...]
                multihead_label_mapping: [(num_class1), (num_class2), ...]
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                has_class_labels: True/False
                roi_labels: (B, num_rois)  1 .. num_classes
                batch_pred_labels: (B, num_boxes, 1)
        Returns:

        """
        post_process_cfg = self.cfg['post_processing']
        batch_size = self.batch_size

        pred_dicts = []
        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_dict['batch_box_preds'].shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_box_preds'].shape.__len__() == 3
                batch_mask = index

            box_preds = batch_dict['batch_box_preds'][batch_mask]
            
            if not isinstance(batch_dict['batch_cls_preds'], list):
                cls_preds = batch_dict['batch_cls_preds'][batch_mask]

                assert cls_preds.shape[1] in [1, self.num_class]

                if not batch_dict['cls_preds_normalized']:
                    cls_preds = torch.sigmoid(cls_preds)
            else:
                cls_preds = [x[batch_mask] for x in batch_dict['batch_cls_preds']]
                if not batch_dict['cls_preds_normalized']:
                    cls_preds = [torch.sigmoid(x) for x in cls_preds]

            cls_preds, label_preds = torch.max(cls_preds, dim=-1)
            label_preds = label_preds # while calculating loss, but not for evaluation

            selected = self.rotate_nms_pcdet(box_preds, cls_preds,
                                                        thresh=post_process_cfg['score_thresh'],
                                                        pre_maxsize=post_process_cfg['nms_config']['nms_pre_maxsize'],
                                                        post_max_size=post_process_cfg['nms_config']['nms_post_maxsize'])

            selected_scores=cls_preds[selected]
            final_scores = selected_scores
            final_labels = label_preds[selected]
            final_boxes = box_preds[selected]

            score_mask = final_scores>post_process_cfg['score_thresh']
            final_scores = final_scores[score_mask]
            final_labels = final_labels[score_mask]
            final_boxes = final_boxes[score_mask]
            # from lwh -> hwl
            final_boxes = final_boxes[:,[0,1,2,5,4,3,6]]
            record_dict = {
                'box3d_lidar': final_boxes,
                'scores': final_scores,
                'label_preds': final_labels
            }
            pred_dicts.append(record_dict)

        return pred_dicts

    def rotate_nms_pcdet(self,boxes, scores, thresh, pre_maxsize=None, post_max_size=None):
        """
        :param boxes: (N, 7) [x, y, z, size_x, size_y, size_z, theta]
        :param scores: (N)
        :param thresh:
        :return:
        """
        # transform back to pcdet's coordinate
        order = scores.sort(0, descending=True)[1]
        if pre_maxsize is not None:
            order = order[:pre_maxsize]

        boxes = boxes[order].contiguous()

        keep = torch.LongTensor(boxes.size(0))

        if len(boxes) == 0:
            num_out = 0
        else:
            num_out = iou3d_nms_cuda.nms_gpu(boxes, keep, thresh)

        selected = order[keep[:num_out].cuda()].contiguous()

        if post_max_size is not None:
            selected = selected[:post_max_size]

        return selected

        
        
        