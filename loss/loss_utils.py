import torch
import torch.nn as nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from einops import rearrange


class FastFocalLoss(nn.Module):
    """
    Reimplemented focal loss, exactly the same as the CornerNet version.
    Faster and costs much less memory.
    """

    def __init__(self):
        super(FastFocalLoss, self).__init__()

    def forward(self, out, target, ind, mask, cat):
        """
        Arguments:
        out, target: B x C x H x W
        ind, mask: B x M
        cat (category id for peaks): B x M
        """
        mask = mask.float()
        alpha, beta = 2, 4
        gt = torch.pow(1 - target, beta)
        neg_loss = torch.pow(out, alpha) * gt * torch.log(1 - out + 1e-7)
        neg_loss = neg_loss.sum()

        pos_pred_pix = _transpose_and_gather_feat(out, ind)  # B x M x C
        pos_pred = pos_pred_pix.gather(2, cat.unsqueeze(2))  # B x M

        num_pos = mask.sum()
        pos_loss = (
            torch.log(pos_pred + 1e-7)
            * torch.pow(1 - pos_pred, alpha)
            * mask.unsqueeze(2)
        )
        pos_loss = pos_loss.sum()
        if num_pos == 0:
            return -neg_loss
        return -(pos_loss + neg_loss) / (num_pos + 1e-4)


class RegLoss(nn.Module):
    """
    Regression loss for an output tensor
        Arguments:
        output (batch x dim x h x w)
        mask (batch x max_objects)
        ind (batch x max_objects)
        target (batch x max_objects x dim)
    """

    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, output, mask, ind, target):
        target = target.type_as(output)
        mask = mask.type_as(output)

        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.float().unsqueeze(2)

        target[torch.isnan(target)] = pred[torch.isnan(target)].detach().clone()
        loss = F.l1_loss(pred * mask, target * mask, reduction="none")

        loss = loss / (mask.sum() + 1e-4)
        loss = loss.transpose(2, 0).sum(dim=2).sum(dim=1)

        return loss


class OffSetLoss(nn.Module):
    """
    Reimplemented focal loss, exactly the same as the CornerNet version.
    Faster and costs much less memory.
    """

    def __init__(self, args):
        super(OffSetLoss, self).__init__()
        self.cfg = args

    def forward(self, out, target, mask, level=0):
        head = self.cfg["model"]["deform"]["offset"]["heads"]
        self.head = head
        self.level = level
        loss = self.forward_(out, target, mask)
        return loss

    def forward_(self, out, target, mask):
        """
        Arguments:
        out, [B x 36 x H x W] x layers
        target, Bx4xHxW. [mini_dis,direc_x,direc_y,mask]

        """
        try:
            target = rearrange(target, "b n c h w -> (b n) c h w")
            mask = rearrange(mask, "b n c h w -> (b n) c h w")
        except:
            pass
        b, c, h, w = target.shape
        supervise_num = self.cfg["model"]["deform"]["offset"]["supervise_num"][
            self.level
        ]
        # supervise_num=18
        kernel = self.cfg["model"]["deform"]["offset"]["kernel"][self.level]
        if supervise_num == 0:
            return 0

        out_ = out.reshape(b, self.head, -1, h, w)  # 2 is head, not yx
        out_ = out_.reshape(b, -1, h, w)
        mask__ = torch.sum(mask, dim=1, keepdim=True)
        mask__[mask__ >= 1] = 1

        # position without trajectory supervision
        mask_neg = torch.where(mask__ <= 0)
        out_zero = out_[mask_neg[0], :, mask_neg[2], mask_neg[3]]
        if out_zero.shape[0] > 0:
            loss_neg = torch.norm(out_zero) / out_zero.shape[0]
        else:
            loss_neg = 0

        if mask__.sum() > 0:
            # position with trajectory supervision
            mask_ = torch.where(mask__ == 1)
            mask_supervise = mask[mask_[0], :, mask_[2], mask_[3]]
            target = target[mask_[0], :, mask_[2], mask_[3]]
            out_ = out_[mask_[0], :, mask_[2], mask_[3]]

            out2 = out_.reshape(
                out_.shape[0], self.head, 2, kernel, kernel
            )  # .transpose(1,2)
            anchor_h, anchor_w = torch.meshgrid(
                torch.arange(kernel), torch.arange(kernel)
            )
            anchor_h, anchor_w = (
                anchor_h - (kernel - 1) / 2,
                anchor_w - (kernel - 1) / 2,
            )
            anchor = torch.cat(
                (anchor_h[None, ...], anchor_w[None, ...]), dim=0
            ).type_as(out_)

            out2 = out2 + anchor[None, None, ...]  # head,yx,3x3

            h_ = mask_[2].reshape(-1, 1, 1).repeat(1, kernel, kernel)
            w_ = mask_[3].reshape(-1, 1, 1).repeat(1, kernel, kernel)
            for i in range(out2.shape[1]):
                out2[:, i, 0, :, :] = out2[:, i, 0, :, :] + h_
                out2[:, i, 1, :, :] = out2[:, i, 1, :, :] + w_

            out2 = out2.reshape(-1, self.head * 2, kernel, kernel)
            target2 = target.reshape(-1, 2, supervise_num).permute(
                0, 2, 1
            )  # 2 is hw, not head
            a, b, c, d = out2.shape

            out2 = (
                out2.reshape(-1, self.head, 2, kernel, kernel)
                .transpose(1, 2)
                .reshape(a, 2, -1)
            )
            out2_neg = out2[:, :, supervise_num:]
            loss_pos_neg = torch.norm(out2_neg) / out2_neg.shape[0]

            out2 = out2[:, :, :supervise_num]
            out2 = out2.permute(0, 2, 1)

            ## assignment using sinkhorn
            cost = torch.cdist(target2.float(), out2.float(), p=1)
            cost = cost  # + (1-mask_supervise.unsqueeze(-1))*1e10

            assignment = self.log_optimal_transport(-cost, iters=100)
            ## plot assignment matrix
            # assignment_mask = self.retrieve_assignment_mask(assignment,mask_supervise)
            # vis_assignment_matrix(cost,assignment,i=0,cmap='RdPu',random_name=False)
            loss_pos = (assignment * cost * mask_supervise.unsqueeze(-1)).sum()

        else:
            loss_pos = 0
            loss_pos_neg = 0

        loss_neg = loss_neg + loss_pos_neg

        a = 100
        if mask__.sum() > 0:
            loss = loss_neg * a + loss_pos / (mask__.sum() + 1e-4)
        else:
            loss = loss_neg * a + loss_pos

        return loss

    def arange_like(self, x, dim: int):
        return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1

    def log_sinkhorn_iterations(self, Z, log_mu, log_nu, iters: int):
        #  Originating Authors: Paul-Edouard Sarlin
        """Perform Sinkhorn Normalization in Log-space for stability"""
        u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
        for _ in range(iters):
            u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
            v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
        return Z + u.unsqueeze(2) + v.unsqueeze(1)

    def log_optimal_transport(self, scores, iters: int):
        #  Originating Authors: Paul-Edouard Sarlin
        """Perform Differentiable Optimal Transport in Log-space for stability"""
        b, m, n = scores.shape
        one = scores.new_tensor(1)
        ms, ns = (m * one).to(scores), (n * one).to(scores)

        norm = -(ms + ns).log()
        log_mu = norm.expand(m)
        log_nu = norm.expand(n)
        # log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
        # log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
        log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

        Z = self.log_sinkhorn_iterations(scores, log_mu, log_nu, iters)
        Z = Z - norm  # multiply probabilities by M+N

        return Z.exp()

    def retrieve_assignment_mask(self, assignment):
        torch.max(assignment, dim=-1).indices
        pass


class FieldLoss(nn.Module):
    """
    Reimplemented focal loss, exactly the same as the CornerNet version.
    Faster and costs much less memory.
    """

    def __init__(self):
        super(FieldLoss, self).__init__()

    def forward(self, out, target, static_mask):
        """
        Arguments:
        out, B x 3 x H x W
        target, Bx4xHxW. [mini_dis,direc_x,direc_y,mask]

        """
        try:
            target = rearrange(target, "b n c h w -> (b n) c h w")
        except:
            pass
        pos_mask = target[:, 3, :, :][:, None, ...]
        target_hm = target[:, 0, :, :]
        target_angle = target[:, 1:3, :, :]

        out_hm = out[:, 0, :, :]
        out_hm = torch.clamp(out_hm, min=1e-4, max=1 - 1e-4)

        out_angle = out[:, 1:3, :, :]

        mask = pos_mask.float()
        alpha, beta = 2, 4
        gt = torch.pow(1 - target_hm, beta)

        neg_loss = (
            torch.pow(out_hm, alpha) * gt * torch.log(1 - out_hm + 1e-7)
        )  # * mask_static
        neg_loss = neg_loss.sum()

        num_pos = mask.sum()
        # pos_loss = F.l1_loss(out_hm * mask, target_hm * mask, reduction='none')#torch.pow(target_hm - out_hm, alpha) * mask

        pos_loss = torch.log(out_hm + 1e-7) * torch.pow(1 - out_hm, alpha) * mask

        pos_loss = pos_loss.sum()
        if num_pos == 0:
            hm_loss = -neg_loss
        else:
            hm_loss = -(pos_loss + neg_loss) / num_pos

        angle_loss = F.l1_loss(out_angle * mask, target_angle * mask, reduction="none")
        angle_loss_ = torch.sum(angle_loss)

        angle_loss_neg = F.l1_loss(
            out_angle * (1 - mask), target_angle * (1 - mask), reduction="none"
        )
        angle_loss_neg_ = torch.sum(angle_loss_neg)

        if num_pos == 0:
            angle_loss2 = 1 / 400 * angle_loss_neg_
        else:
            angle_loss2 = angle_loss_ / (num_pos + 1e-4) + 1 / 400 * angle_loss_neg_ / (
                num_pos + 1e-4
            )

        return angle_loss2 + hm_loss


class TclsLoss(nn.Module):
    """
    Reimplemented focal loss, exactly the same as the CornerNet version.
    Faster and costs much less memory.
    """

    def __init__(self):
        super(TclsLoss, self).__init__()
        self.CrossEntropyLoss = nn.CrossEntropyLoss()

    def forward(self, out, target, mask):
        """
        Arguments:
        out, B x 3 x H x W
        target, Bx4xHxW. [mini_dis,direc_x,direc_y,mask]

        """
        try:
            target = rearrange(target, "b n c h w -> (b n) c h w")
        except:
            pass
        target = target.squeeze(1)
        target[target >= out.shape[1]] = out.shape[1] - 1

        mask = rearrange(mask, "b n c h w -> (b n) c h w")
        mask__ = torch.sum(mask, dim=1)
        mask__[mask__ >= 1] = 1
        b, h, w = torch.where(mask__ > 0)
        pos_loss = self.CrossEntropyLoss(out[b, :, h, w], target[b, h, w].long())
        # b,h,w = torch.where(mask__ == 0)
        # neg_loss = torch.norm(out[b,:,h,w])

        return pos_loss  # + neg_loss


class IouRegLoss(nn.Module):
    """Distance IoU loss for output boxes
    Arguments:
      output (batch x dim x h x w)
      mask (batch x max_objects)
      ind (batch x max_objects)
      target (batch x max_objects x dim)
    """

    def __init__(self):
        super(IouRegLoss, self).__init__()
        self.bbox3d_iou_func = bbox3d_overlaps_diou

    def forward(self, box_pred, mask, ind, box_gt):
        if mask.sum() == 0:
            box_pred = box_pred.float()
            return box_pred.sum() * 0
        mask = mask.bool()
        box_gt = box_gt.permute(0, 2, 1)
        pred_box = _transpose_and_gather_feat(box_pred, ind)
        iou = self.bbox3d_iou_func(pred_box[mask], box_gt[mask])
        loss = (1.0 - iou).sum() / (mask.sum() + 1e-4)
        return loss


class IouLoss(nn.Module):
    """IouLoss loss for an output tensor
    Arguments:
    output (batch x dim x h x w)
    mask (batch x max_objects)
    ind (batch x max_objects)
    target (batch x max_objects x dim)
    """

    def __init__(self):
        super(IouLoss, self).__init__()

    def forward(self, iou_pred, mask, ind, box_pred, box_gt):
        if mask.sum() == 0:
            return iou_pred.sum() * 0
        mask = mask.bool()
        box_gt = box_gt.permute(0, 2, 1)
        pred = _transpose_and_gather_feat(iou_pred, ind)[mask]
        pred_box = _transpose_and_gather_feat(box_pred, ind)
        target = boxes_aligned_iou3d_gpu(pred_box[mask], box_gt[mask])
        target = 2 * target - 1
        loss = F.l1_loss(pred, target, reduction="sum")
        loss = loss / (mask.sum() + 1e-4)
        return loss


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def center_to_corner2d(center, dim):
    corners_norm = torch.tensor(
        [[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]],
        dtype=torch.float32,
        device=dim.device,
    )
    corners = dim.view([-1, 1, 2]) * corners_norm.view([1, 4, 2])
    corners = corners + center.view(-1, 1, 2)
    return corners


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


"""
3D IoU Calculation and Rotated NMS
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
"""
import torch

from models.ops.iou3d_nms import iou3d_nms_cuda


def boxes_iou3d_gpu(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (N, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        ans_iou: (N, M)
    """
    assert boxes_a.shape[1] == boxes_b.shape[1] == 7

    # height overlap
    boxes_a_height_max = (boxes_a[:, 2] + boxes_a[:, 5] / 2).view(-1, 1)
    boxes_a_height_min = (boxes_a[:, 2] - boxes_a[:, 5] / 2).view(-1, 1)
    boxes_b_height_max = (boxes_b[:, 2] + boxes_b[:, 5] / 2).view(1, -1)
    boxes_b_height_min = (boxes_b[:, 2] - boxes_b[:, 5] / 2).view(1, -1)

    # bev overlap
    overlaps_bev = torch.cuda.FloatTensor(
        torch.Size((boxes_a.shape[0], boxes_b.shape[0]))
    ).zero_()  # (N, M)
    iou3d_nms_cuda.boxes_overlap_bev_gpu(
        boxes_a.contiguous(), boxes_b.contiguous(), overlaps_bev
    )

    max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)
    min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)
    overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)

    # 3d iou
    overlaps_3d = overlaps_bev * overlaps_h

    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(1, -1)

    iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-6)

    return iou3d


def boxes_aligned_iou3d_gpu(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (N, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        ans_iou: (N,)
    """
    assert boxes_a.shape[0] == boxes_b.shape[0]
    assert boxes_a.shape[1] == boxes_b.shape[1] == 7

    # transform back to pcdet's coordinate
    # boxes_a = to_pcdet(boxes_a)
    # boxes_b = to_pcdet(boxes_b)

    # height overlap
    boxes_a_height_max = (boxes_a[:, 2] + boxes_a[:, 5] / 2).view(-1, 1)
    boxes_a_height_min = (boxes_a[:, 2] - boxes_a[:, 5] / 2).view(-1, 1)
    boxes_b_height_max = (boxes_b[:, 2] + boxes_b[:, 5] / 2).view(-1, 1)
    boxes_b_height_min = (boxes_b[:, 2] - boxes_b[:, 5] / 2).view(-1, 1)

    # bev overlap
    overlaps_bev = torch.cuda.FloatTensor(
        torch.Size((boxes_a.shape[0], 1))
    ).zero_()  # (N, M)
    iou3d_nms_cuda.boxes_aligned_overlap_bev_gpu(
        boxes_a.contiguous(), boxes_b.contiguous(), overlaps_bev
    )

    max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)
    min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)
    overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)

    # 3d iou
    overlaps_3d = overlaps_bev * overlaps_h

    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(-1, 1)

    iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-6)

    return iou3d


def nms_gpu(boxes, scores, thresh, pre_maxsize=None, **kwargs):
    """
    :param boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
    :param scores: (N)
    :param thresh:
    :return:
    """
    assert boxes.shape[1] == 7
    order = scores.sort(0, descending=True)[1]
    if pre_maxsize is not None:
        order = order[:pre_maxsize]

    boxes = boxes[order].contiguous()
    keep = torch.LongTensor(boxes.size(0))
    num_out = iou3d_nms_cuda.nms_gpu(boxes, keep, thresh)
    return order[keep[:num_out].cuda()].contiguous(), None


def vis_assignment_matrix(cost, assignment, i=0, cmap="bwr", random_name=False):
    # cmap='RdPu'
    # random_name = False
    # i = 0
    import matplotlib.pyplot as plt
    import random
    import string

    work_dir = "DEFU_paper/supplementary_assign_matrix/"
    fig, ax = plt.subplots(1, 2)
    im1 = ax[0].imshow(cost[i].cpu().numpy(), cmap=cmap)
    im2 = ax[1].imshow(assignment[i].cpu().numpy(), cmap=cmap)
    cb1 = fig.colorbar(im1, ax=ax[0], fraction=0.046, pad=0.04)
    cb2 = fig.colorbar(im2, ax=ax[1], fraction=0.046, pad=0.04)
    cb1.set_label("L1 distance (m)", rotation=90, labelpad=5, fontsize=10)  # 设置标题
    cb2.set_label(
        "Matching probability", rotation=90, labelpad=5, fontsize=10
    )  # 设置标题

    ax[0].axis("off")
    ax[1].axis("off")

    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域

    characters = string.ascii_letters + string.digits
    # 使用random.choice随机选择字符，并连接成字符串
    if random_name:
        random_string = "".join(random.choice(characters) for _ in range(6))
    else:
        random_string = "test"

    plt.savefig(f"{work_dir}{random_string}.png", dpi=600, bbox_inches="tight")
    plt.close()
