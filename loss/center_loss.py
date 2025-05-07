import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from loss.loss_utils import FastFocalLoss, RegLoss, IouRegLoss, IouLoss


class CenterLoss(nn.Module):

    def __init__(
        self,
        cfg,
        with_reg_iou=True,
        voxel_size=None,
        pc_range=None,
        out_size_factor=None,
    ):

        super(CenterLoss, self).__init__()
        self.cfg = cfg
        kwargs = cfg["model"]["head"]
        weight = kwargs["weight"]

        with_reg_iou = kwargs["with_reg_iou"]
        voxel_size = cfg["voxelization"]["voxel_size"]
        pc_range = cfg["voxelization"]["lidar_range"]
        out_size_factor = cfg["model"]["backbone"]["out_size_factor"]

        self.weight = weight  # weight between hm loss and loc loss

        self.crit = FastFocalLoss()
        self.crit_reg = RegLoss()

        self.with_reg_iou = with_reg_iou
        if self.with_reg_iou:
            self.crit_iou_reg = IouRegLoss()

        self.with_iou = "iou" in kwargs["common_heads"]
        if self.with_iou:
            self.crit_iou = IouLoss()

        if self.with_iou or with_reg_iou:
            self.voxel_size = voxel_size
            self.pc_range = pc_range
            self.out_size_factor = out_size_factor

    def forward(self, data_dict, preds_dicts_list, train=False):

        preds_dicts = preds_dicts_list[0]

        head_dicts = data_dict["head_dicts"]

        for task_id, preds_dict in enumerate(preds_dicts):

            # heatmap focal loss
            preds_dict["hm"] = self._sigmoid(preds_dict["hm"])

            hm_loss = self.crit(
                preds_dict["hm"],
                head_dicts["hm"][task_id],
                head_dicts["ind"][task_id],
                head_dicts["mask"][task_id],
                head_dicts["cat"][task_id],
            )

            target_box = head_dicts["anno_box"][task_id]

            # reconstruct the anno_box from multiple reg heads
            preds_dict["anno_box"] = torch.cat(
                (
                    preds_dict["reg"],
                    preds_dict["height"],
                    preds_dict["dim"],
                    preds_dict["rot"],
                ),
                dim=1,
            )

            # Regression loss for dimension, offset, height, rotation
            box_loss = self.crit_reg(
                preds_dict["anno_box"],
                head_dicts["mask"][task_id],
                head_dicts["ind"][task_id],
                target_box,
            )

            loc_loss = box_loss.sum()

            loss = hm_loss + self.weight * loc_loss

            if self.with_iou or self.with_reg_iou:
                batch_dim = torch.exp(torch.clamp(preds_dict["dim"], min=-5, max=5))
                batch_dim = batch_dim.permute(0, 2, 3, 1).contiguous()
                batch_rot = preds_dict["rot"].clone()
                batch_rot = batch_rot.permute(0, 2, 3, 1).contiguous()
                batch_rots = batch_rot[..., 0:1]
                batch_rotc = batch_rot[..., 1:2]
                batch_rot = torch.atan2(batch_rots, batch_rotc)

                batch_reg = preds_dict["reg"].clone().permute(0, 2, 3, 1).contiguous()
                batch_hei = (
                    preds_dict["height"].clone().permute(0, 2, 3, 1).contiguous()
                )

                batch, H, W, _ = batch_dim.size()

                batch_reg = batch_reg.reshape(batch, H * W, 2)
                batch_hei = batch_hei.reshape(batch, H * W, 1)

                batch_rot = batch_rot.reshape(batch, H * W, 1)
                batch_dim = batch_dim.reshape(batch, H * W, 3)

                ys, xs = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
                ys = ys.view(1, H, W).repeat(batch, 1, 1).to(batch_dim)
                xs = xs.view(1, H, W).repeat(batch, 1, 1).to(batch_dim)

                xs = xs.view(batch, -1, 1) + batch_reg[:, :, 0:1]
                ys = ys.view(batch, -1, 1) + batch_reg[:, :, 1:2]

                xs = (
                    xs * self.out_size_factor[task_id] * self.voxel_size[0]
                    + self.pc_range[0]
                )
                ys = (
                    ys * self.out_size_factor[task_id] * self.voxel_size[1]
                    + self.pc_range[1]
                )

                batch_box_preds = torch.cat(
                    [xs, ys, batch_hei, batch_dim, batch_rot], dim=2
                )
                batch_box_preds = (
                    batch_box_preds.permute(0, 2, 1)
                    .contiguous()
                    .reshape(batch, -1, H, W)
                )

                if self.with_iou:
                    pred_boxes_for_iou = batch_box_preds.detach()
                    iou_loss = self.crit_iou(
                        preds_dict["iou"],
                        head_dicts["mask"][task_id],
                        head_dicts["ind"][task_id],
                        pred_boxes_for_iou,
                        head_dicts["gt_boxes"][task_id],
                    )
                    loss = loss + iou_loss

                if self.with_reg_iou:
                    iou_reg_loss = self.crit_iou_reg(
                        batch_box_preds,
                        head_dicts["mask"][task_id],
                        head_dicts["ind"][task_id],
                        head_dicts["gt_boxes"][task_id],
                    )
                    loss = loss + self.weight * iou_reg_loss

            if task_id == 0:
                total_loss = loss
            else:
                total_loss = total_loss + loss

        return total_loss

    def _sigmoid(self, x):
        y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
        return y

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
