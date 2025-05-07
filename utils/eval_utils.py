# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib


import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import OrderedDict
import copy

from utils import common_utils
from hypes_yaml import yaml_utils


def voc_ap(rec, prec):
    """
    VOC 2010 Average Precision.
    """
    rec.insert(0, 0.0)
    rec.append(1.0)
    mrec = rec[:]

    prec.insert(0, 0.0)
    prec.append(0.0)
    mpre = prec[:]

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)

    ap = 0.0
    for i in i_list:
        ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap, mrec, mpre


def caluclate_tp_fp(
    det_boxes,
    det_score,
    det_cls,
    gt_boxes,
    gt_cls,
    result_stat,
    iou_threshs_,
    eval_cls=["overall"],
):
    """
    Calculate the true positive and false positive numbers of the current
    frames.

    Parameters
    ----------
    det_boxes : torch.Tensor
        The detection bounding box, shape (N, 8, 3) or (N, 4, 2).
    det_score :torch.Tensor
        The confidence score for each preditect bounding box.
    gt_boxes : torch.Tensor
        The groundtruth bounding box.
    result_stat: dict
        A dictionary contains fp, tp and gt number.
    iou_thresh : float
        The iou thresh.
    """

    eval_name_to_cls = {"overall": -1, "vehicle": 0, "cyclist": 1, "pedestrian": 2}

    if det_boxes is not None:
        # convert bounding boxes to numpy array
        det_boxes = common_utils.torch_tensor_to_numpy(det_boxes)
        det_score = common_utils.torch_tensor_to_numpy(det_score)
        gt_boxes = common_utils.torch_tensor_to_numpy(gt_boxes)
        det_cls = common_utils.torch_tensor_to_numpy(det_cls)
        gt_cls = common_utils.torch_tensor_to_numpy(gt_cls)

    for eval_cls_ in eval_cls:
        cls_num = eval_name_to_cls[eval_cls_]
        if not eval_cls_ == "overall":
            det_mask = det_cls == cls_num
            det_boxes_ = det_boxes[det_mask]
            det_score_ = det_score[det_mask]

            gt_mask = gt_cls == cls_num
            gt_boxes_ = gt_boxes[gt_mask]
        else:
            det_boxes_, det_score_, gt_boxes_ = (
                det_boxes.copy(),
                det_score.copy(),
                gt_boxes.copy(),
            )
        if "overall" in eval_cls:
            iou_threshs = iou_threshs_[cls_num + 1, :]
        else:
            iou_threshs = iou_threshs_[cls_num, :]

        gt = gt_boxes_.shape[0]

        for iou_thresh in iou_threshs:
            # fp, tp and gt in the current frame
            fp, tp, frames, iou = [], [], [], []
            if len(result_stat[eval_cls_][iou_thresh]["frame"]) > 0:
                fr = result_stat[eval_cls_][iou_thresh]["frame"][-1] + 1
            else:
                fr = 0

            if det_boxes_ is not None:

                # sort the prediction bounding box by score
                score_order_descend = np.argsort(-det_score_)
                det_score_ = det_score_[score_order_descend]  # from high to low
                det_polygon_list = list(common_utils.convert_format(det_boxes_))
                gt_polygon_list = list(common_utils.convert_format(gt_boxes_))

                # match prediction and gt bounding box
                for i in range(score_order_descend.shape[0]):
                    det_polygon = det_polygon_list[score_order_descend[i]]
                    ious = common_utils.compute_iou(det_polygon, gt_polygon_list)
                    # print(np.max(ious))
                    # iou.append(np.max(ious))
                    if len(gt_polygon_list) == 0 or np.max(ious) < iou_thresh:
                        fp.append(1)
                        tp.append(0)
                        frames.append(fr)
                        continue

                    fp.append(0)
                    tp.append(1)
                    frames.append(fr)

                    gt_index = np.argmax(ious)
                    gt_polygon_list.pop(gt_index)

                result_stat[eval_cls_][iou_thresh]["score"] += det_score_.tolist()
                result_stat[eval_cls_][iou_thresh]["frame"] += frames

            result_stat[eval_cls_][iou_thresh]["fp"] += fp
            result_stat[eval_cls_][iou_thresh]["tp"] += tp
            result_stat[eval_cls_][iou_thresh]["gt"] += gt
            # result_stat[eval_cls_][iou_thresh]['iou'] += iou

    return result_stat


def calculate_ap(result_stat, iou, global_sort_detections):
    """
    Calculate the average precision and recall, and save them into a txt.

    Parameters
    ----------
    result_stat : dict
        A dictionary contains fp, tp and gt number.

    iou : float
        The threshold of iou.

    global_sort_detections : bool
        Whether to sort the detection results globally.
    """
    iou_5 = copy.deepcopy(result_stat[iou])

    if global_sort_detections:
        fp = np.array(iou_5["fp"])
        tp = np.array(iou_5["tp"])
        score = np.array(iou_5["score"])

        assert len(fp) == len(tp) and len(tp) == len(score)
        sorted_index = np.argsort(-score)
        fp = fp[sorted_index].tolist()
        tp = tp[sorted_index].tolist()

    else:
        fp = iou_5["fp"]
        tp = iou_5["tp"]
        assert len(fp) == len(tp)

    gt_total = iou_5["gt"]

    cumsum = 0
    for idx, val in enumerate(fp):
        fp[idx] += cumsum
        cumsum += val

    cumsum = 0
    for idx, val in enumerate(tp):
        tp[idx] += cumsum
        cumsum += val

    rec = tp[:]
    for idx, val in enumerate(tp):
        rec[idx] = float(tp[idx]) / gt_total

    prec = tp[:]
    for idx, val in enumerate(tp):
        prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

    ap, mrec, mprec = voc_ap(rec[:], prec[:])

    return ap, mrec, mprec


def eval_final_results(
    result_stat, save_path, eval_cls, ious_, global_sort_detections=True
):
    dump_dict = {}
    gsd = global_sort_detections
    result_stat = copy.deepcopy(result_stat)

    for i, cls in enumerate(eval_cls):
        dump_dict_ = {}
        ious = ious_[i, :]
        for iou_ in ious:
            ap, _, _ = calculate_ap(result_stat[cls], iou_, gsd)
            dump_dict_.update(
                {
                    f"ap@{iou_}": ap,
                }
            )
        dump_dict.update({cls: dump_dict_})

    if save_path is not None:
        yaml_utils.save_yaml(dump_dict, save_path)

    # print(dump_dict)
    for cls, ap_list in dump_dict.items():
        for i, ap in ap_list.items():
            ap_list[i] = round(ap, 4) * 100
        print(f"{cls} ", ap_list)
