import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from utils import box_coder_utils, common_utils, loss_utils
from models.modules.head.target_assigner.anchor_generator import AnchorGenerator
from models.modules.head.target_assigner.axis_aligned_target_assigner import (
    AxisAlignedTargetAssigner,
)

from loss.loss_utils import OffSetLoss, FieldLoss, TclsLoss


class DTFPointPillarsLoss(nn.Module):

    def __init__(self, cfg):
        super(DTFPointPillarsLoss, self).__init__()

        if self.training:
            batch_size = cfg["train_params"]["train_batch_size"]
        else:
            batch_size = cfg["train_params"]["val_batch_size"]
        cfg["batch_size"] = batch_size
        self.cfg = cfg

        self.num_class = len(cfg["dataset"]["cls_group"])

        model_cfg = cfg["model"]["head"]
        self.model_cfg = model_cfg

        self.use_multihead = model_cfg.get("use_multihead", False)
        self.num_anchors_per_location = cfg["model"]["head"]["num_anchors_per_location"]
        self.forward_ret_dict = {}

        self.build_losses(model_cfg["loss_config"])

        self.offset_loss = OffSetLoss(cfg)
        self.field_loss = FieldLoss()
        self.tcls_loss = TclsLoss()
        self.bbox3d_iou_func = bbox3d_overlaps_diou

    def forward(self, data_dict, preds_dicts, train=False):
        # preds_dicts={'cls_preds':xxx,'box_preds':xxx}
        x_traj = preds_dicts["x_traj"]
        x_offset = preds_dicts["x_offset"]
        head_dicts = data_dict["head_dicts"]
        field_dim = self.cfg["model"]["deform"]["field_dim"]
        alpha, beta, cta, dta = self.cfg["loss"]["weight"]
        eta = self.cfg["loss"].get("t_cls_weight", 0)
        if not train:
            alpha, beta, cta, dta, eta = 1, 1, 0, 0, 0
            weight_iou = 0

        loss_offset, loss_field, loss_t_cls = 0, 0, 0

        if eta != 0:
            b = self.tcls_loss(
                x_traj[:, field_dim:, :, :],
                head_dicts["field"][0][:, :, field_dim + 1 :, :, :],
                head_dicts["offset_mask"][0],
            )
            loss_t_cls = loss_t_cls + b

        if dta != 0:
            b = self.field_loss(
                x_traj[:, :field_dim, :, :],
                head_dicts["field"][0][:, :, : field_dim + 1, :, :],
                head_dicts["offset_mask"][0],
            )
            if self.cfg["model"]["deform"]["offset"]["supervise_num"][0] != 0:
                loss_field = loss_field + b

        # if cta!=0:
        a = self.offset_loss(
            x_offset, head_dicts["offset_gt"][0], head_dicts["offset_mask"][0], level=0
        )
        loss_offset = loss_offset + a

        cls_loss = self.get_cls_layer_loss(preds_dicts)
        box_loss = self.get_box_reg_layer_loss(preds_dicts)

        try:
            reg_iou = self.cfg["loss"]["reg_iou"]
            weight_iou = self.cfg["loss"]["iou_weight"]
        except:
            reg_iou = False
            weight_iou = 0

        if reg_iou:
            iou_loss = self.get_iou_reg_loss(preds_dicts)
        else:
            iou_loss = 0

        loss = (
            alpha * cls_loss
            + beta * box_loss
            + cta * loss_offset
            + dta * loss_field
            + weight_iou * iou_loss
            + eta * loss_t_cls
        )

        return loss

    def build_losses(self, losses_cfg):
        self.add_module(
            "cls_loss_func",
            loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0),
        )
        reg_loss_name = (
            "WeightedSmoothL1Loss"
            if losses_cfg.get("reg_loss_type", None) is None
            else losses_cfg["reg_loss_type"]
        )
        self.add_module(
            "reg_loss_func",
            getattr(loss_utils, reg_loss_name)(
                code_weights=losses_cfg["loss_weights"]["code_weights"]
            ),
        )
        self.add_module("dir_loss_func", loss_utils.WeightedCrossEntropyLoss())

    def get_cls_layer_loss(self, pred_dicts):
        cls_preds = pred_dicts["cls_preds"]
        box_cls_labels = pred_dicts["box_cls_labels"]
        batch_size = int(cls_preds.shape[0])
        cared = box_cls_labels >= 0  # [N, num_anchors]
        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        reg_weights = positives.float()
        if self.num_class == 1:
            # class agnostic
            box_cls_labels[positives] = 1

        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
        cls_targets = cls_targets.unsqueeze(dim=-1)

        cls_targets = cls_targets.squeeze(dim=-1)
        one_hot_targets = torch.zeros(
            *list(cls_targets.shape),
            self.num_class + 1,
            dtype=cls_preds.dtype,
            device=cls_targets.device
        )
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        cls_preds = cls_preds.view(batch_size, -1, self.num_class)
        one_hot_targets = one_hot_targets[..., 1:]
        cls_loss_src = self.cls_loss_func(
            cls_preds, one_hot_targets, weights=cls_weights
        )  # [N, M]
        cls_loss = cls_loss_src.sum() / batch_size

        cls_loss = (
            cls_loss * self.model_cfg["loss_config"]["loss_weights"]["cls_weight"]
        )

        return cls_loss

    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim : dim + 1]) * torch.cos(
            boxes2[..., dim : dim + 1]
        )
        rad_tg_encoding = torch.cos(boxes1[..., dim : dim + 1]) * torch.sin(
            boxes2[..., dim : dim + 1]
        )
        boxes1 = torch.cat(
            [boxes1[..., :dim], rad_pred_encoding, boxes1[..., dim + 1 :]], dim=-1
        )
        boxes2 = torch.cat(
            [boxes2[..., :dim], rad_tg_encoding, boxes2[..., dim + 1 :]], dim=-1
        )
        return boxes1, boxes2

    def get_box_reg_layer_loss(self, pred_dicts):
        box_preds = pred_dicts["box_preds"]

        box_reg_targets = pred_dicts["box_reg_targets"]
        box_cls_labels = pred_dicts["box_cls_labels"]

        batch_size = int(box_preds.shape[0])

        positives = box_cls_labels > 0
        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        box_preds = box_preds.view(
            batch_size,
            -1,
            (
                box_preds.shape[-1] // self.num_anchors_per_location
                if not self.use_multihead
                else box_preds.shape[-1]
            ),
        )
        # sin(a - b) = sinacosb-cosasinb
        box_preds_sin, reg_targets_sin = self.add_sin_difference(
            box_preds, box_reg_targets
        )
        loc_loss_src = self.reg_loss_func(
            box_preds_sin, reg_targets_sin, weights=reg_weights
        )  # [N, M]
        loc_loss = loc_loss_src.sum() / batch_size

        loc_loss = (
            loc_loss * self.model_cfg["loss_config"]["loss_weights"]["loc_weight"]
        )
        box_loss = loc_loss

        return box_loss

    def get_iou_reg_loss(self, pred_dicts):
        box_preds = pred_dicts["box_preds"]

        box_reg_targets = pred_dicts["box_reg_targets"]
        box_cls_labels = pred_dicts["box_cls_labels"]
        batch_size = int(box_preds.shape[0])

        positives = box_cls_labels > 0
        if positives.sum() == 0:
            return box_preds.sum() * 0

        box_preds = box_preds.view(
            batch_size,
            -1,
            (
                box_preds.shape[-1] // self.num_anchors_per_location
                if not self.use_multihead
                else box_preds.shape[-1]
            ),
        )

        box_preds_pos = box_preds[positives]
        box_reg_targets_pos = box_reg_targets[positives]
        if box_preds_pos.shape[0] > 0 and box_reg_targets_pos.shape[0] > 1:
            iou = self.bbox3d_iou_func(box_preds_pos, box_reg_targets_pos)
            iou_loss = (1.0 - iou).sum() / (positives.sum() + 1e-4)
        else:
            iou_loss = box_preds.sum() * 0

        # if np.isnan(iou_loss.detach().cpu()):
        #     a=1
        #     pass

        return iou_loss

    def logging(self, epoch, batch_id, batch_len, total_loss, lr, pbar=None):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        # total_loss = self.loss_dict['total_loss']

        if pbar is None:
            print(
                "[epoch %d][%d/%d], || Loss: %.4f || precision: %.2f, || recall:%.2f"
                % (epoch, batch_id + 1, batch_len, total_loss)
            )
        else:
            pbar.set_description(
                "[epoch %d][%d/%d], || Loss: %.4f || Learning rate: %.4f"
                % (epoch, batch_id + 1, batch_len, (total_loss.item()), lr)
            )


def bbox3d_overlaps_diou(pred_boxes, gt_boxes):
    assert pred_boxes.shape[0] == gt_boxes.shape[0]

    qcorners = center_to_corner2d(pred_boxes[:, :2], pred_boxes[:, 3:5])
    gcorners = center_to_corner2d(gt_boxes[:, :2], gt_boxes[:, 3:5])

    inter_max_xy = torch.minimum(qcorners[:, 2], gcorners[:, 2])
    inter_min_xy = torch.maximum(qcorners[:, 0], gcorners[:, 0])
    out_max_xy = torch.maximum(qcorners[:, 2], gcorners[:, 2])
    out_min_xy = torch.minimum(qcorners[:, 0], gcorners[:, 0])

    # calculate area
    volume_pred_boxes = pred_boxes[:, 3] * pred_boxes[:, 4] * pred_boxes[:, 5]
    volume_gt_boxes = gt_boxes[:, 3] * gt_boxes[:, 4] * gt_boxes[:, 5]

    inter_h = torch.minimum(
        pred_boxes[:, 2] + 0.5 * pred_boxes[:, 5], gt_boxes[:, 2] + 0.5 * gt_boxes[:, 5]
    ) - torch.maximum(
        pred_boxes[:, 2] - 0.5 * pred_boxes[:, 5], gt_boxes[:, 2] - 0.5 * gt_boxes[:, 5]
    )
    inter_h = torch.clamp(inter_h, min=0)

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    volume_inter = inter[:, 0] * inter[:, 1] * inter_h
    volume_union = volume_gt_boxes + volume_pred_boxes - volume_inter

    # boxes_iou3d_gpu(pred_boxes, gt_boxes)
    inter_diag = torch.pow(gt_boxes[:, 0:3] - pred_boxes[:, 0:3], 2).sum(-1)

    outer_h = torch.maximum(
        gt_boxes[:, 2] + 0.5 * gt_boxes[:, 5], pred_boxes[:, 2] + 0.5 * pred_boxes[:, 5]
    ) - torch.minimum(
        gt_boxes[:, 2] - 0.5 * gt_boxes[:, 5], pred_boxes[:, 2] - 0.5 * pred_boxes[:, 5]
    )
    outer_h = torch.clamp(outer_h, min=0)
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = outer[:, 0] ** 2 + outer[:, 1] ** 2 + outer_h**2

    dious = volume_inter / volume_union - inter_diag / outer_diag
    dious = torch.clamp(dious, min=-1.0, max=1.0)

    return dious


def center_to_corner2d(center, dim):
    corners_norm = torch.tensor(
        [[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]],
        dtype=torch.float32,
        device=dim.device,
    )
    corners = dim.view([-1, 1, 2]) * corners_norm.view([1, 4, 2])
    corners = corners + center.view(-1, 1, 2)
    return corners
