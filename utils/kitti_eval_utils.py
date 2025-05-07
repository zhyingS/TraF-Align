import torch
import io as sysio
import numpy as np
import json
import numba
from functools import reduce

from utils.box_utils import boxes_to_corners_3d

from utils.rotate_iou import rotate_iou_gpu_eval  # might cause rank 0 out of memory


def read_json(path):
    with open(path, "r") as f:
        json_ = json.load(f)
        return json_


def get_cam_calib_intrinsic(calib_path):
    json_ = read_json(calib_path)
    cam_K = json_["cam_K"]
    calib = np.array(cam_K).reshape([3, 3], order="C")
    return calib


def get_lidar2cam(calib_path):
    json_ = read_json(calib_path)
    r_velo2cam = np.array(json_["rotation"])
    t_velo2cam = np.array(json_["translation"])
    Tr_velo_to_cam = np.eye(4)
    Tr_velo_to_cam[:3, :3] = r_velo2cam
    Tr_velo_to_cam[:3, 3] = t_velo2cam.flatten()
    return Tr_velo_to_cam, r_velo2cam, t_velo2cam


def result2kitti(cfg, dets, token_dict):
    # dets:{
    #     'box3d_lidar': torch.tensor (N,8),
    #     'scores': torch.tensor (N,),
    #     'label_preds': torch.tensor (N,)
    # }

    # kitti label (all in camera coordinate):
    # [type,truncated,occluded,alpha,xmin(pixel),ymin,xmax,ymax,h,w,l,x,y,z,phi]
    for key, value in dets.items():
        if isinstance(value, torch.Tensor):
            dets[key] = np.asarray(value.cpu())
    boxes = dets["box3d_lidar"]
    scores = np.round(dets["scores"], 4)
    labels = dets["label_preds"]

    # # filtering low score boxes
    # score_threshold = np.asarray(cfg['post_processing']['score_threshold'])
    # score_th_ = score_threshold[labels]
    # mask = scores > score_th_
    # boxes = boxes[mask]
    # scores = scores[mask]
    # labels = labels[mask]

    boxes[:, 2] = boxes[:, 2] - boxes[:, 3] / 2

    # transform boxes from lidar to camera coordinate
    camera_intrinsic_file = token_dict["calib_camera_intrinsic_path"]

    if "calib_virtuallidar_to_camera_path" in token_dict:
        virtuallidar_to_camera_file = token_dict["calib_virtuallidar_to_camera_path"]
    elif "calib_lidar_to_camera_path" in token_dict:
        virtuallidar_to_camera_file = token_dict["calib_lidar_to_camera_path"]
    else:
        raise ("There's no calibration files for lidar to camera.")

    camera_intrinsic = get_cam_calib_intrinsic(camera_intrinsic_file)
    _, r_velo2cam, t_velo2cam = get_lidar2cam(virtuallidar_to_camera_file)

    alpha, box_corner3d = get_camera_3d_8points(boxes, r_velo2cam, t_velo2cam)
    yaw = -0.5 * np.pi - boxes[:, 6]

    # the code is to be corrected
    box2d, mask = get_camera_intri_box(box_corner3d, camera_intrinsic)

    # filtering the boxes out of image view
    labels = labels[mask]
    alpha = alpha[mask]
    yaw = yaw[mask]
    boxes = boxes[mask]
    scores = scores[mask]

    if "truncated" in dets:
        truncated = dets["truncated"][mask]
    else:
        truncated = np.zeros((labels.shape[0]))

    if "occluded" in dets:
        occluded = dets["occluded"][mask]
    else:
        occluded = np.zeros((labels.shape[0]))

    i1 = labels
    i2 = truncated
    i3 = occluded
    i4 = np.round(alpha, 4)
    i5, i6, i7, i8 = (
        np.round(box2d[0], 4),
        np.round(box2d[1], 4),
        np.round(box2d[2], 4),
        np.round(box2d[3], 4),
    )
    h, w, l = boxes.T[3:6]
    i9, i10, i11 = np.round(h, 4), np.round(w, 4), np.round(l, 4)
    cam_x, cam_y, cam_z = np.matmul(r_velo2cam, boxes[:, :3].T) + t_velo2cam

    i12, i13, i14 = np.round(cam_x, 4), np.round(cam_y, 4), np.round(cam_z, 4)
    i15 = np.round(yaw, 4)
    pred_lines = np.stack(
        (i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, scores)
    )

    return boxes, pred_lines


def get_camera_intri_box(boxes, camera_intrinsic_, img_size=[1920, 1080]):
    camera_intrinsic = camera_intrinsic_[None, ...].repeat(boxes.shape[0], 0)

    corners_2d = np.matmul(camera_intrinsic, boxes)
    corners_2d = corners_2d[:, :2] / corners_2d[:, 2][:, None, :]
    box2d = np.array(
        [
            np.min(corners_2d[:, 0], axis=1),
            np.min(corners_2d[:, 1], axis=1),
            np.max(corners_2d[:, 0], axis=1),
            np.max(corners_2d[:, 1], axis=1),
        ]
    )

    # [xmin, ymin, xmax, ymax]
    mask = reduce(
        np.logical_and,
        (
            box2d[0] <= img_size[0],
            box2d[2] >= 0,
            box2d[1] <= img_size[1],
            box2d[3] >= 0,
        ),
    )
    # mask = reduce(np.logical_and,(
    #     box2d[0]>=0,
    #     box2d[1]>=0,
    #     box2d[2]<=img_size[0],
    #     box2d[3]<=img_size[1]
    # ))
    box2d = np.maximum(box2d, 0.0)
    box2d[2] = np.minimum(box2d[2], img_size[0])
    box2d[3] = np.minimum(box2d[3], img_size[1])

    return box2d[:, mask], mask


def get_camera_3d_8points(boxes, r_velo2cam, t_velo2cam):
    # box_center_in_cam = np.dot(r_velo2cam, boxes[:,:3].T) + t_velo2cam
    box_corner = boxes_to_corners_3d(boxes, order="hwl")
    box_corner_cam = np.matmul(
        r_velo2cam[None, ...].repeat(box_corner.shape[0], 0),
        box_corner.transpose(0, 2, 1),
    ) + t_velo2cam[None, ...].repeat(box_corner.shape[0], 0)

    x0, z0 = box_corner_cam[:, 0, 0], box_corner_cam[:, 2, 0]
    x3, z3 = box_corner_cam[:, 0, 3], box_corner_cam[:, 2, 3]
    dx, dz = x0 - x3, z0 - z3
    yaw = np.arctan2(-dz, dx)
    alpha = yaw - np.arctan2(boxes[:, 0], boxes[:, 2])
    alpha[alpha > np.pi] = alpha[alpha > np.pi] - 2.0 * np.pi
    alpha[alpha <= -np.pi] = alpha[alpha <= -np.pi] + 2.0 * np.pi
    alpha_arctan = normalize_angle(alpha)

    return alpha_arctan, box_corner_cam


def normalize_angle(angle):
    alpha_tan = np.tan(angle)
    alpha_arctan = np.arctan(alpha_tan)
    mask = np.where(np.cos(angle) < 0)
    alpha_arctan[mask] = alpha_arctan[mask] + np.pi
    return alpha_arctan


def get_label_anno(content):
    # content: 16 x N
    # kitti label (all in camera coordinate):
    # [type,truncated,occluded,alpha,xmin(pixel),ymin,xmax,ymax,h,w,l,x,y,z,phi]
    annotations = {}
    annotations.update(
        {
            "name": [],
            "truncated": [],
            "occluded": [],
            "alpha": [],
            "bbox": [],
            "dimensions": [],
            "location": [],
            "rotation_y": [],
        }
    )

    annotations["name"] = content[0]
    annotations["truncated"] = content[1]
    annotations["occluded"] = content[2]
    annotations["alpha"] = content[3]
    annotations["bbox"] = content[4:8].T

    # dimensions will convert hwl format to standard lhw(camera) format.
    annotations["dimensions"] = content[8:11].T[:, [2, 0, 1]]
    annotations["location"] = content[11:14].T
    annotations["rotation_y"] = content[14]
    if content.shape[0] == 16:  # have score
        annotations["score"] = content[15]
    else:
        annotations["score"] = np.zeros([len(annotations["name"])])
    return annotations


def get_gt_label_anno(hypes, token_dict):

    # kitti label (all in camera coordinate):
    # [type,truncated,occluded,alpha,xmin(pixel),ymin,xmax,ymax,h,w,l,x,y,z,phi]
    frame = token_dict["frame_id"]
    kitti_label_path = (
        hypes["root_dir"] + hypes["dataset"]["label"][:-1] + "_kitti/" + frame + ".txt"
    )
    with open(kitti_label_path, "r") as f:
        lines = f.readlines()

    if len(lines) == 0 or len(lines[0]) < 15:
        content = []
    else:
        content = [line.strip().split(" ") for line in lines]
    annotations = {}
    annotations.update(
        {
            "name": [],
            "truncated": [],
            "occluded": [],
            "alpha": [],
            "bbox": [],
            "dimensions": [],
            "location": [],
            "rotation_y": [],
        }
    )

    name = np.array([x[0] for x in content])
    _cls_group = np.asarray(hypes["dataset"]["cls"])
    index = [int(np.where(i == _cls_group)[0]) for i in name]
    cls_map = np.asarray(hypes["dataset"]["cls_map"])
    name = cls_map[np.asarray(index)]

    annotations["name"] = name.astype(np.int)
    annotations["truncated"] = np.array([float(x[1]) for x in content])
    annotations["occluded"] = np.array([int(x[2]) for x in content])
    annotations["alpha"] = np.array([float(x[3]) for x in content])
    annotations["bbox"] = np.array(
        [[float(info) for info in x[4:8]] for x in content]
    ).reshape(-1, 4)
    # dimensions will convert hwl format to standard lhw(camera) format.
    annotations["dimensions"] = np.array(
        [[float(info) for info in x[8:11]] for x in content]
    ).reshape(-1, 3)[:, [2, 0, 1]]
    annotations["location"] = np.array(
        [[float(info) for info in x[11:14]] for x in content]
    ).reshape(-1, 3)
    annotations["rotation_y"] = np.array([float(x[14]) for x in content]).reshape(-1)
    if len(content) != 0 and len(content[0]) == 16:  # have score
        annotations["score"] = np.array([float(x[15]) for x in content])
    else:
        annotations["score"] = np.zeros([len(annotations["bbox"])])

    return annotations


def bev_box_overlap(boxes, qboxes, criterion=-1):
    riou = rotate_iou_gpu_eval(boxes, qboxes, criterion)
    return riou


def get_split_parts(num, num_part):
    same_part = num // num_part
    remain_num = num % num_part
    if same_part == 0:
        return [num]

    if remain_num == 0:
        return [same_part] * num_part
    else:
        return [same_part] * num_part + [remain_num]


def do_eval(
    gt_annos,
    dt_annos,
    current_classes,
    min_overlaps,
    compute_aos=False,
    metric=1,
    PR_detail_dict=None,
):

    # min_overlaps: [num_minoverlap, metric, num_class]
    difficultys = [0, 1, 2]
    (
        mAP_bbox,
        mAP_bev,
        mAP_3d,
        mAP_aos,
        mAP_bbox_R40,
        mAP_bev_R40,
        mAP_3d_R40,
        mAP_aos_R40,
    ) = (0, 0, 0, 0, 0, 0, 0, 0)

    if "bbox" in metric:
        ret = eval_class(
            gt_annos,
            dt_annos,
            current_classes,
            difficultys,
            0,
            min_overlaps,
            compute_aos,
        )
        # ret: [num_class, num_diff, num_minoverlap, num_sample_points]
        mAP_bbox = get_mAP(ret["precision"])
        mAP_bbox_R40 = get_mAP_R40(ret["precision"])

        if PR_detail_dict is not None:
            PR_detail_dict["bbox"] = ret["precision"]

        mAP_aos = mAP_aos_R40 = None
        if compute_aos:
            mAP_aos = get_mAP(ret["orientation"])
            mAP_aos_R40 = get_mAP_R40(ret["orientation"])

            if PR_detail_dict is not None:
                PR_detail_dict["aos"] = ret["orientation"]
    if "bev" in metric:

        ret = eval_class(
            gt_annos, dt_annos, current_classes, difficultys, 1, min_overlaps
        )
        mAP_bev = get_mAP(ret["precision"])
        mAP_bev_R40 = get_mAP_R40(ret["precision"])

        if PR_detail_dict is not None:
            PR_detail_dict["bev"] = ret["precision"]

    if "3d" in metric:
        ret = eval_class(
            gt_annos, dt_annos, current_classes, difficultys, 2, min_overlaps
        )
        mAP_3d = get_mAP(ret["precision"])
        mAP_3d_R40 = get_mAP_R40(ret["precision"])
        if PR_detail_dict is not None:
            PR_detail_dict["3d"] = ret["precision"]

    return (
        mAP_bbox,
        mAP_bev,
        mAP_3d,
        mAP_aos,
        mAP_bbox_R40,
        mAP_bev_R40,
        mAP_3d_R40,
        mAP_aos_R40,
    )


def get_official_eval_result(
    hypes, gt_annos, dt_annos, current_classes, metric=["bev"], PR_detail_dict=None
):
    # overlap: [difficulty, class]
    overlap = np.array([[0.5, 0.25, 0.1], [0.5, 0.25, 0.1], [0.5, 0.25, 0.1]])
    overlap_0_5 = np.array([[0.7, 0.5, 0.25], [0.7, 0.5, 0.25], [0.7, 0.5, 0.25]])
    min_overlaps = np.stack([overlap, overlap_0_5], axis=0)  # [2, 3, 5]

    # min_overlaps = np.stack([overlap], axis=0)  # [1, 3, 5]

    # class_to_name={}
    # for i,i_ in enumerate(hypes['dataset']['cls_group']):
    #     class_to_name.update({i:i_[0]})
    class_to_name = {0: "Car", 1: "Cyclist", 2: "Pedestrian"}
    name_to_class = {v: n for n, v in class_to_name.items()}
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    current_classes = current_classes_int
    min_overlaps = min_overlaps[:, :, current_classes]
    result = ""
    # check whether alpha is valid
    compute_aos = False
    # for anno in dt_annos:
    #     if anno['alpha'].shape[0] != 0:
    #         if anno['alpha'][0] != -10:
    #             compute_aos = True
    #         break
    mAPbbox, mAPbev, mAP3d, mAPaos, mAPbbox_R40, mAPbev_R40, mAP3d_R40, mAPaos_R40 = (
        do_eval(
            gt_annos,
            dt_annos,
            current_classes,
            min_overlaps,
            compute_aos,
            metric,
            PR_detail_dict=PR_detail_dict,
        )
    )

    ret_dict = {}
    AP11 = False
    for j, curcls in enumerate(current_classes):
        # mAP threshold array: [num_minoverlap, metric, class]
        # mAP result: [num_class, num_diff, num_minoverlap]
        for i in range(min_overlaps.shape[0]):
            if AP11:
                result += print_str(
                    (
                        f"{class_to_name[curcls]} "
                        "AP@{:.2f}, {:.2f}, {:.2f}:".format(*min_overlaps[i, :, j])
                    )
                )
                if "bbox" in metric:
                    result += print_str(
                        (
                            f"bbox AP:{mAPbbox[j, 0, i]:.4f}, "
                            f"{mAPbbox[j, 1, i]:.4f}, "
                            f"{mAPbbox[j, 2, i]:.4f}"
                        )
                    )
                if "bev" in metric:
                    result += print_str(
                        (
                            f"bev  AP:{mAPbev[j, 0, i]:.4f}, "
                            f"{mAPbev[j, 1, i]:.4f}, "
                            f"{mAPbev[j, 2, i]:.4f}"
                        )
                    )
                if "3d" in metric:
                    result += print_str(
                        (
                            f"3d   AP:{mAP3d[j, 0, i]:.4f}, "
                            f"{mAP3d[j, 1, i]:.4f}, "
                            f"{mAP3d[j, 2, i]:.4f}"
                        )
                    )

                if compute_aos:
                    result += print_str(
                        (
                            f"aos  AP:{mAPaos[j, 0, i]:.2f}, "
                            f"{mAPaos[j, 1, i]:.2f}, "
                            f"{mAPaos[j, 2, i]:.2f}"
                        )
                    )

            result += print_str(
                (
                    f"{class_to_name[curcls]} "
                    "AP_R40@{:.2f}, {:.2f}, {:.2f}:".format(*min_overlaps[i, :, j])
                )
            )
            if "bbox" in metric:
                result += print_str(
                    (
                        f"bbox AP:{mAPbbox_R40[j, 0, i]:.4f}, "
                        f"{mAPbbox_R40[j, 1, i]:.4f}, "
                        f"{mAPbbox_R40[j, 2, i]:.4f}"
                    )
                )
            if "bev" in metric:
                result += print_str(
                    (
                        f"bev  AP:{mAPbev_R40[j, 0, i]:.4f}, "
                        f"{mAPbev_R40[j, 1, i]:.4f}, "
                        f"{mAPbev_R40[j, 2, i]:.4f}"
                    )
                )
            if "3d" in metric:
                result += print_str(
                    (
                        f"3d   AP:{mAP3d_R40[j, 0, i]:.4f}, "
                        f"{mAP3d_R40[j, 1, i]:.4f}, "
                        f"{mAP3d_R40[j, 2, i]:.4f}"
                    )
                )
            if compute_aos:
                result += print_str(
                    (
                        f"aos  AP:{mAPaos_R40[j, 0, i]:.2f}, "
                        f"{mAPaos_R40[j, 1, i]:.2f}, "
                        f"{mAPaos_R40[j, 2, i]:.2f}"
                    )
                )
                if i == 0:
                    ret_dict["%s_aos/easy_R40" % class_to_name[curcls]] = mAPaos_R40[
                        j, 0, 0
                    ]
                    ret_dict["%s_aos/moderate_R40" % class_to_name[curcls]] = (
                        mAPaos_R40[j, 1, 0]
                    )
                    ret_dict["%s_aos/hard_R40" % class_to_name[curcls]] = mAPaos_R40[
                        j, 2, 0
                    ]

    return result, ret_dict


def eval_class(
    gt_annos,
    dt_annos,
    current_classes,
    difficultys,
    metric,
    min_overlaps,
    compute_aos=False,
    num_parts=100,
):
    """Kitti eval. support 2d/bev/3d/aos eval. support 0.5:0.05:0.95 coco AP.
    Args:
        gt_annos: dict, must from get_label_annos() in kitti_common.py
        dt_annos: dict, must from get_label_annos() in kitti_common.py
        current_classes: list of int, 0: car, 2: pedestrian, 1: cyclist
        difficultys: list of int. eval difficulty, 0: easy, 1: normal, 2: hard
        metric: eval type. 0: bbox, 1: bev, 2: 3d
        min_overlaps: float, min overlap. format: [num_overlap, metric, class].
        num_parts: int. a parameter for fast calculate algorithm

    Returns:
        dict of recall, precision and aos
    """
    assert len(gt_annos) == len(dt_annos)
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)

    rets = calculate_iou_partly(dt_annos, gt_annos, metric, num_parts)
    overlaps, parted_overlaps, total_dt_num, total_gt_num = rets
    N_SAMPLE_PTS = 41
    num_minoverlap = len(min_overlaps)
    num_class = len(current_classes)
    num_difficulty = len(difficultys)
    precision = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    recall = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    aos = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    for m, current_class in enumerate(current_classes):
        for l, difficulty in enumerate(difficultys):
            rets = _prepare_data(gt_annos, dt_annos, current_class, difficulty)

            (
                gt_datas_list,
                dt_datas_list,
                ignored_gts,
                ignored_dets,
                dontcares,
                total_dc_num,
                total_num_valid_gt,
            ) = rets
            print(total_num_valid_gt)
            for k, min_overlap in enumerate(min_overlaps[:, metric, m]):
                thresholdss = []
                for i in range(len(gt_annos)):
                    rets = compute_statistics_jit(
                        overlaps[i],
                        gt_datas_list[i],
                        dt_datas_list[i],
                        ignored_gts[i],
                        ignored_dets[i],
                        dontcares[i],
                        metric,
                        min_overlap=min_overlap,
                        thresh=0.0,
                        compute_fp=False,
                    )
                    tp, fp, fn, similarity, thresholds = rets
                    thresholdss += thresholds.tolist()
                    # find out all the det scores of true positive boxes

                thresholdss = np.array(thresholdss)
                thresholds = get_thresholds(thresholdss, total_num_valid_gt)
                thresholds = np.array(thresholds)
                pr = np.zeros([len(thresholds), 4])
                idx = 0
                for j, num_part in enumerate(split_parts):
                    gt_datas_part = np.concatenate(
                        gt_datas_list[idx : idx + num_part], 0
                    )
                    dt_datas_part = np.concatenate(
                        dt_datas_list[idx : idx + num_part], 0
                    )
                    dc_datas_part = np.concatenate(dontcares[idx : idx + num_part], 0)
                    ignored_dets_part = np.concatenate(
                        ignored_dets[idx : idx + num_part], 0
                    )
                    ignored_gts_part = np.concatenate(
                        ignored_gts[idx : idx + num_part], 0
                    )
                    fused_compute_statistics(
                        parted_overlaps[j],
                        pr,
                        total_gt_num[idx : idx + num_part],
                        total_dt_num[idx : idx + num_part],
                        total_dc_num[idx : idx + num_part],
                        gt_datas_part,
                        dt_datas_part,
                        dc_datas_part,
                        ignored_gts_part,
                        ignored_dets_part,
                        metric,
                        min_overlap=min_overlap,
                        thresholds=thresholds,
                        compute_aos=compute_aos,
                    )
                    idx += num_part
                for i in range(len(thresholds)):
                    recall[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 2])
                    precision[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 1])
                    if compute_aos:
                        aos[m, l, k, i] = pr[i, 3] / (pr[i, 0] + pr[i, 1])
                for i in range(len(thresholds)):
                    precision[m, l, k, i] = np.max(precision[m, l, k, i:], axis=-1)
                    recall[m, l, k, i] = np.max(recall[m, l, k, i:], axis=-1)
                    if compute_aos:
                        aos[m, l, k, i] = np.max(aos[m, l, k, i:], axis=-1)
    # print('precision of vehicle easy iou 0.5:',precision[0][0][0])
    # print('recall of vehicle easy iou 0.5:',recall[0][0][0])
    # print('precision of vehicle mid iou 0.5:',precision[0][1][0])
    # print('recall of vehicle mid iou 0.5:',recall[0][1][0])
    # print('precision of vehicle hard iou 0.5:',precision[0][2][0])
    # print('recall of vehicle hard iou 0.5:',recall[0][2][0])
    ret_dict = {
        "recall": recall,
        "precision": precision,
        "orientation": aos,
    }
    return ret_dict


def calculate_iou_partly(gt_annos, dt_annos, metric, num_parts=50):
    """fast iou algorithm. this function can be used independently to
    do result analysis. Must be used in CAMERA coordinate system.
    Args:
        gt_annos: dict, must from get_label_annos() in kitti_common.py
        dt_annos: dict, must from get_label_annos() in kitti_common.py
        metric: eval type. 0: bbox, 1: bev, 2: 3d
        num_parts: int. a parameter for fast calculate algorithm
    """
    assert len(gt_annos) == len(dt_annos)
    total_dt_num = np.stack([len(a["name"]) for a in dt_annos], 0)
    total_gt_num = np.stack([len(a["name"]) for a in gt_annos], 0)
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)
    parted_overlaps = []
    example_idx = 0

    for num_part in split_parts:
        gt_annos_part = gt_annos[example_idx : example_idx + num_part]
        dt_annos_part = dt_annos[example_idx : example_idx + num_part]
        if metric == 0:
            gt_boxes = np.concatenate([a["bbox"] for a in gt_annos_part], 0)
            dt_boxes = np.concatenate([a["bbox"] for a in dt_annos_part], 0)
            overlap_part = image_box_overlap(gt_boxes, dt_boxes)
        elif metric == 1:
            loc = np.concatenate([a["location"][:, [0, 2]] for a in gt_annos_part], 0)
            dims = np.concatenate(
                [a["dimensions"][:, [0, 2]] for a in gt_annos_part], 0
            )
            rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
            loc = np.concatenate([a["location"][:, [0, 2]] for a in dt_annos_part], 0)
            dims = np.concatenate(
                [a["dimensions"][:, [0, 2]] for a in dt_annos_part], 0
            )
            rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
            overlap_part = bev_box_overlap(gt_boxes, dt_boxes).astype(np.float64)
        elif metric == 2:
            loc = np.concatenate([a["location"] for a in gt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in gt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
            loc = np.concatenate([a["location"] for a in dt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in dt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
            overlap_part = d3_box_overlap(gt_boxes, dt_boxes).astype(np.float64)
        else:
            raise ValueError("unknown metric")
        parted_overlaps.append(overlap_part)
        example_idx += num_part
    overlaps = []
    example_idx = 0
    for j, num_part in enumerate(split_parts):
        gt_annos_part = gt_annos[example_idx : example_idx + num_part]
        dt_annos_part = dt_annos[example_idx : example_idx + num_part]
        gt_num_idx, dt_num_idx = 0, 0
        for i in range(num_part):
            gt_box_num = total_gt_num[example_idx + i]
            dt_box_num = total_dt_num[example_idx + i]
            overlaps.append(
                parted_overlaps[j][
                    gt_num_idx : gt_num_idx + gt_box_num,
                    dt_num_idx : dt_num_idx + dt_box_num,
                ]
            )
            gt_num_idx += gt_box_num
            dt_num_idx += dt_box_num
        example_idx += num_part

    return overlaps, parted_overlaps, total_gt_num, total_dt_num


def get_mAP(prec):
    sums = 0
    for i in range(0, prec.shape[-1], 4):
        sums = sums + prec[..., i]
    return sums / 11 * 100


def get_mAP_R40(prec):
    sums = 0
    for i in range(1, prec.shape[-1]):
        sums = sums + prec[..., i]
    return sums / 40 * 100


def print_str(value, *arg, sstream=None):
    if sstream is None:
        sstream = sysio.StringIO()
    sstream.truncate(0)
    sstream.seek(0)
    print(value, *arg, file=sstream)
    return sstream.getvalue()


def clean_data(gt_anno, dt_anno, current_cls_name, difficulty):

    MIN_HEIGHT = [40, 25, 25]
    MAX_OCCLUSION = [0, 1, 2]
    MAX_TRUNCATION = [0.15, 0.3, 0.5]
    dc_bboxes, ignored_gt, ignored_dt = [], [], []

    num_gt = len(gt_anno["name"])
    num_dt = len(dt_anno["name"])
    num_valid_gt = 0
    for i in range(num_gt):
        bbox = gt_anno["bbox"][i]
        gt_name = gt_anno["name"][i]
        height = bbox[3] - bbox[1]
        valid_class = -1
        if gt_name == current_cls_name:
            valid_class = 1
        # elif (current_cls_name == 'Pedestrian'.lower()
        #       and 'Person_sitting'.lower() == gt_name):
        #     valid_class = 0
        # elif (current_cls_name == 'Car'.lower() and 'Van'.lower() == gt_name):
        #     valid_class = 0
        else:
            valid_class = -1
        ignore = False
        # if ((gt_anno["occluded"][i] > MAX_OCCLUSION[difficulty])
        #         or (gt_anno["truncated"][i] > MAX_TRUNCATION[difficulty])
        #         or (height <= MIN_HEIGHT[difficulty])):
        #     ignore = True
        if valid_class == 1 and not ignore:
            ignored_gt.append(0)
            num_valid_gt += 1
        elif valid_class == 0 or (ignore and (valid_class == 1)):
            ignored_gt.append(1)
        else:
            ignored_gt.append(-1)

        if gt_anno["name"][i] == "DontCare":
            dc_bboxes.append(gt_anno["bbox"][i])
    for i in range(num_dt):
        if dt_anno["name"][i] == current_cls_name:
            valid_class = 1
        else:
            valid_class = -1
        height = abs(dt_anno["bbox"][i, 3] - dt_anno["bbox"][i, 1])
        if height < MIN_HEIGHT[difficulty]:
            ignored_dt.append(1)
        elif valid_class == 1:
            ignored_dt.append(0)
        else:
            ignored_dt.append(-1)

    return num_valid_gt, ignored_gt, ignored_dt, dc_bboxes


def _prepare_data(gt_annos, dt_annos, current_class, difficulty):
    gt_datas_list = []
    dt_datas_list = []
    total_dc_num = []
    ignored_gts, ignored_dets, dontcares = [], [], []
    total_num_valid_gt = 0
    for i in range(len(gt_annos)):
        rets = clean_data(gt_annos[i], dt_annos[i], current_class, difficulty)
        num_valid_gt, ignored_gt, ignored_det, dc_bboxes = rets
        ignored_gts.append(np.array(ignored_gt, dtype=np.int64))
        ignored_dets.append(np.array(ignored_det, dtype=np.int64))
        if len(dc_bboxes) == 0:
            dc_bboxes = np.zeros((0, 4)).astype(np.float64)
        else:
            dc_bboxes = np.stack(dc_bboxes, 0).astype(np.float64)
        total_dc_num.append(dc_bboxes.shape[0])
        dontcares.append(dc_bboxes)
        total_num_valid_gt += num_valid_gt
        gt_datas = np.concatenate(
            [gt_annos[i]["bbox"], gt_annos[i]["alpha"][..., np.newaxis]], 1
        )
        dt_datas = np.concatenate(
            [
                dt_annos[i]["bbox"],
                dt_annos[i]["alpha"][..., np.newaxis],
                dt_annos[i]["score"][..., np.newaxis],
            ],
            1,
        )
        gt_datas_list.append(gt_datas)
        dt_datas_list.append(dt_datas)
    total_dc_num = np.stack(total_dc_num, axis=0)
    return (
        gt_datas_list,
        dt_datas_list,
        ignored_gts,
        ignored_dets,
        dontcares,
        total_dc_num,
        total_num_valid_gt,
    )


@numba.jit(nopython=True, parallel=True)
def d3_box_overlap_kernel(boxes, qboxes, rinc, criterion=-1):
    # ONLY support overlap in CAMERA, not lider.
    N, K = boxes.shape[0], qboxes.shape[0]
    for i in range(N):
        for j in range(K):
            if rinc[i, j] > 0:
                # iw = (min(boxes[i, 1] + boxes[i, 4], qboxes[j, 1] +
                #         qboxes[j, 4]) - max(boxes[i, 1], qboxes[j, 1]))
                iw = min(boxes[i, 1], qboxes[j, 1]) - max(
                    boxes[i, 1] - boxes[i, 4], qboxes[j, 1] - qboxes[j, 4]
                )

                if iw > 0:
                    area1 = boxes[i, 3] * boxes[i, 4] * boxes[i, 5]
                    area2 = qboxes[j, 3] * qboxes[j, 4] * qboxes[j, 5]
                    inc = iw * rinc[i, j]
                    if criterion == -1:
                        ua = area1 + area2 - inc
                    elif criterion == 0:
                        ua = area1
                    elif criterion == 1:
                        ua = area2
                    else:
                        ua = inc
                    rinc[i, j] = inc / ua
                else:
                    rinc[i, j] = 0.0


def d3_box_overlap(boxes, qboxes, criterion=-1):
    rinc = rotate_iou_gpu_eval(boxes[:, [0, 2, 3, 5, 6]], qboxes[:, [0, 2, 3, 5, 6]], 2)
    d3_box_overlap_kernel(boxes, qboxes, rinc, criterion)
    return rinc


@numba.jit(nopython=True)
def compute_statistics_jit(
    overlaps,
    gt_datas,
    dt_datas,
    ignored_gt,
    ignored_det,
    dc_bboxes,
    metric,
    min_overlap,
    thresh=0,
    compute_fp=False,
    compute_aos=False,
):

    det_size = dt_datas.shape[0]
    gt_size = gt_datas.shape[0]
    dt_scores = dt_datas[:, -1]  # todo: really?
    dt_alphas = dt_datas[:, 4]
    gt_alphas = gt_datas[:, 4]
    dt_bboxes = dt_datas[:, :4]
    gt_bboxes = gt_datas[:, :4]

    assigned_detection = [False] * det_size
    ignored_threshold = [False] * det_size
    if compute_fp:
        for i in range(det_size):
            if dt_scores[i] < thresh:
                ignored_threshold[i] = True
    NO_DETECTION = -10000000
    tp, fp, fn, similarity = 0, 0, 0, 0
    # thresholds = [0.0]
    # delta = [0.0]
    thresholds = np.zeros((gt_size,))
    thresh_idx = 0
    delta = np.zeros((gt_size,))
    delta_idx = 0
    for i in range(gt_size):
        if ignored_gt[i] == -1:
            continue
        det_idx = -1
        valid_detection = NO_DETECTION
        max_overlap = 0
        assigned_ignored_det = False

        for j in range(det_size):
            if ignored_det[j] == -1:
                continue
            if assigned_detection[j]:
                continue
            if ignored_threshold[j]:
                continue
            overlap = overlaps[j, i]
            dt_score = dt_scores[j]
            if (
                not compute_fp
                and (overlap > min_overlap)
                and dt_score > valid_detection
            ):
                det_idx = j
                valid_detection = dt_score
            elif (
                compute_fp
                and (overlap > min_overlap)
                and (overlap > max_overlap or assigned_ignored_det)
                and ignored_det[j] == 0
            ):
                max_overlap = overlap
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = False
            elif (
                compute_fp
                and (overlap > min_overlap)
                and (valid_detection == NO_DETECTION)
                and ignored_det[j] == 1
            ):
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = True

        if (valid_detection == NO_DETECTION) and ignored_gt[i] == 0:
            fn += 1
        elif (valid_detection != NO_DETECTION) and (
            ignored_gt[i] == 1 or ignored_det[det_idx] == 1
        ):
            assigned_detection[det_idx] = True
        elif valid_detection != NO_DETECTION:
            tp += 1
            # thresholds.append(dt_scores[det_idx])
            thresholds[thresh_idx] = dt_scores[det_idx]
            thresh_idx += 1
            if compute_aos:
                # delta.append(gt_alphas[i] - dt_alphas[det_idx])
                delta[delta_idx] = gt_alphas[i] - dt_alphas[det_idx]
                delta_idx += 1

            assigned_detection[det_idx] = True
    if compute_fp:
        for i in range(det_size):
            if not (
                assigned_detection[i]
                or ignored_det[i] == -1
                or ignored_det[i] == 1
                or ignored_threshold[i]
            ):
                fp += 1
        nstuff = 0
        if metric == 0:
            overlaps_dt_dc = image_box_overlap(dt_bboxes, dc_bboxes, 0)
            for i in range(dc_bboxes.shape[0]):
                for j in range(det_size):
                    if assigned_detection[j]:
                        continue
                    if ignored_det[j] == -1 or ignored_det[j] == 1:
                        continue
                    if ignored_threshold[j]:
                        continue
                    if overlaps_dt_dc[j, i] > min_overlap:
                        assigned_detection[j] = True
                        nstuff += 1
        fp -= nstuff
        if compute_aos:
            tmp = np.zeros((fp + delta_idx,))
            # tmp = [0] * fp
            for i in range(delta_idx):
                tmp[i + fp] = (1.0 + np.cos(delta[i])) / 2.0
                # tmp.append((1.0 + np.cos(delta[i])) / 2.0)
            # assert len(tmp) == fp + tp
            # assert len(delta) == tp
            if tp > 0 or fp > 0:
                similarity = np.sum(tmp)
            else:
                similarity = -1
    return tp, fp, fn, similarity, thresholds[:thresh_idx]


@numba.jit(nopython=True)
def image_box_overlap(boxes, query_boxes, criterion=-1):
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    for k in range(K):
        qbox_area = (query_boxes[k, 2] - query_boxes[k, 0]) * (
            query_boxes[k, 3] - query_boxes[k, 1]
        )
        for n in range(N):
            iw = min(boxes[n, 2], query_boxes[k, 2]) - max(
                boxes[n, 0], query_boxes[k, 0]
            )
            if iw > 0:
                ih = min(boxes[n, 3], query_boxes[k, 3]) - max(
                    boxes[n, 1], query_boxes[k, 1]
                )
                if ih > 0:
                    if criterion == -1:
                        ua = (
                            (boxes[n, 2] - boxes[n, 0]) * (boxes[n, 3] - boxes[n, 1])
                            + qbox_area
                            - iw * ih
                        )
                    elif criterion == 0:
                        ua = (boxes[n, 2] - boxes[n, 0]) * (boxes[n, 3] - boxes[n, 1])
                    elif criterion == 1:
                        ua = qbox_area
                    else:
                        ua = 1.0
                    overlaps[n, k] = iw * ih / ua
    return overlaps


@numba.jit(nopython=True)
def get_thresholds(scores: np.ndarray, num_gt, num_sample_pts=41):
    # scores: all the scores of true positive boxes

    scores.sort()
    scores = scores[::-1]
    current_recall = 0
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt  #
        if i < (len(scores) - 1):
            r_recall = (i + 2) / num_gt
        else:
            r_recall = l_recall
        if ((r_recall - current_recall) < (current_recall - l_recall)) and (
            i < (len(scores) - 1)
        ):
            continue
        # recall = l_recall
        thresholds.append(score)
        current_recall += 1 / (num_sample_pts - 1.0)
    return thresholds


@numba.jit(nopython=True)
def fused_compute_statistics(
    overlaps,
    pr,
    gt_nums,
    dt_nums,
    dc_nums,
    gt_datas,
    dt_datas,
    dontcares,
    ignored_gts,
    ignored_dets,
    metric,
    min_overlap,
    thresholds,
    compute_aos=False,
):
    gt_num = 0
    dt_num = 0
    dc_num = 0
    for i in range(gt_nums.shape[0]):
        for t, thresh in enumerate(thresholds):
            overlap = overlaps[
                dt_num : dt_num + dt_nums[i], gt_num : gt_num + gt_nums[i]
            ]

            gt_data = gt_datas[gt_num : gt_num + gt_nums[i]]
            dt_data = dt_datas[dt_num : dt_num + dt_nums[i]]
            ignored_gt = ignored_gts[gt_num : gt_num + gt_nums[i]]
            ignored_det = ignored_dets[dt_num : dt_num + dt_nums[i]]
            dontcare = dontcares[dc_num : dc_num + dc_nums[i]]
            tp, fp, fn, similarity, _ = compute_statistics_jit(
                overlap,
                gt_data,
                dt_data,
                ignored_gt,
                ignored_det,
                dontcare,
                metric,
                min_overlap=min_overlap,
                thresh=thresh,
                compute_fp=True,
                compute_aos=compute_aos,
            )
            # print(tp,fp,fn)
            pr[t, 0] += tp
            pr[t, 1] += fp
            pr[t, 2] += fn
            if similarity != -1:
                pr[t, 3] += similarity
        gt_num += gt_nums[i]
        dt_num += dt_nums[i]
        dc_num += dc_nums[i]
