import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys

sys.path.append("/".join(sys.argv[0].split("/")[:-2]))
import random
import torch.backends.cudnn as cudnn
import time
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from utils.box_utils import *
from utils.vis_utils import *
from utils import train_utils, eval_utils, kitti_eval_utils
import datasets.data_utils.data_check.data_check_utils as check_utils
import hypes_yaml.yaml_utils as yaml_utils
from datasets import build_dataset


def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument(
        "--model_dir",
        default="checkpoints/dair_v2x_seq_Trafalign_2024_12_28_04_42_36",
        help="Checkpoint path",
    )
    parser.add_argument(
        "--model_epoch", default=None, help="inference on model of specific epoch"
    )
    parser.add_argument("--batch_size", default=4)
    parser.add_argument(
        "--delay_ego", default=[0], help="delay of Ego vehicle, iterate in the list"
    )
    parser.add_argument(
        "--delay",
        default=[0],
        help="delay of Cooperative agent, iterate in the list",
    )

    parser.add_argument(
        "--for_pr_curve",
        default=False,
        help="Whether save the results for pr curve plotting, will keep all the bbx, with confidence score from 0 to 1.",
    )
    parser.add_argument("--dataset", default="test")
    parser.add_argument("--selected_frame", default=None)
    parser.add_argument("--metric", default="bev", help="AP bev or 3d")
    parser.add_argument(
        "--visualize", default=False, help="whether to visualize point cloud and offset"
    )
    parser.add_argument("--debug_per_scenario", default=None)
    parser.add_argument("--global_sort_detections", default=False)

    cfg = parser.parse_args()
    return cfg


def main(cfg, delay, delay_ego):
    hypes_yaml = cfg.model_dir + "config.yaml"
    hypes = yaml_utils.load_yaml(hypes_yaml, cfg)

    hypes["dataset"]["infer_range"] = True
    hypes.update({"delay_ego": delay_ego})

    if cfg.selected_frame is not None:
        hypes.update({"selected_frame": cfg.selected_frame})

    hypes = yaml_utils.check_pillar_params(hypes)
    try:
        hypes["post_processing"]["score_threshold"] = hypes["post_processing"][
            "score_threshold_inference"
        ]
    except:
        pass

    # setting batch size and communication delay
    if cfg.batch_size is not None:
        hypes["train_params"]["train_batch_size"] = cfg.batch_size
        hypes["train_params"]["val_batch_size"] = cfg.batch_size
        print(f"batch size is {cfg.batch_size}")
    if delay is not None:
        if hypes["wild_setting"]["async"]:
            hypes["wild_setting"]["agent_i_delay"] = delay
        print(f"delay is {delay}")

    print("Dataset Building")
    dataset = build_dataset(hypes, set=cfg.dataset)
    print(f"{len(dataset)} samples found.")
    shuf = 1 if cfg.visualize and cfg.selected_frame is None else 0

    data_loader = DataLoader(
        dataset,
        batch_size=hypes["train_params"]["val_batch_size"],
        num_workers=12,
        collate_fn=dataset.collate_batch,
        shuffle=shuf,
        pin_memory=False,
        drop_last=True,
    )

    print("Creating Model")
    model = train_utils.create_model(hypes)
    model = model.float()

    if torch.cuda.is_available():
        model.cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading Model from checkpoint")
    saved_path = cfg.model_dir
    epoch, model = train_utils.load_saved_model(saved_path, model, cfg.model_epoch)
    pbar = tqdm(total=len(data_loader), leave=True)
    if cfg.for_pr_curve:
        saved_path = cfg.model_dir + "/for_pr_curve"
        hypes["post_processing"]["score_thresh"] = 0
        if not os.path.exists(saved_path):
            os.mkdir(saved_path)

    # define the loss
    criterion = train_utils.create_loss(hypes)

    try:
        eval_cls = hypes["dataset"]["eval_cls"]
        ious = np.array(hypes["dataset"]["eval_iou_threshold"])
    except:
        eval_cls = ["overall", "vehicle", "cyclist", "pedestrian"]
        ious = np.array(
            [[0.3, 0.5, 0.7], [0.3, 0.5, 0.7], [0.1, 0.25, 0.5], [0.1, 0.25, 0.5]]
        )

    result_stat = ini_result_stat(eval_cls, ious)
    scenario = ""

    for i, batch_data in tqdm(enumerate(data_loader)):
        with torch.no_grad():
            if "v2v4real" in hypes["fusion"]["dataset"]:
                scenario_name = batch_data["lidar_path"][0][-54:-13]
            elif "v2xseq" in hypes["fusion"]["dataset"]:
                scenario_name = batch_data["token_dicts"][0]["seq"]

            model.eval()
            batch_data = train_utils.to_device(batch_data, device)
            infer = False
            assert (
                infer == False
            ), "Only support false infer now, try to reduce impact of timestamp to get freedom."

            preds, dets = model(batch_data)
            loss = criterion(batch_data, preds)

            gt_box = batch_data["label_dict"]
            # b,x,y,z,h,w,l,phi,type,id,time,truncated,occluded

            # visualization
            if cfg.visualize == True:
                batch = 0
                if gt_box.shape[0] == 0:
                    gt_box = torch.zeros((2, 12))
                mask = gt_box[:, 0] == batch

                ## plot point cloud and bounding boxes, save to 'visualization/lidar_box.png'
                check_utils.check_detection_boxes(
                    batch_data["processed_lidar"],
                    [gt_box[mask][:, 1:8], dets[batch]["box3d_lidar"]],
                    cfg=hypes,
                    detc="_box",
                )

                deform = 1  # different for trafalign and baselines
                if deform:
                    ## plot trajectory field, save to 'visualization/field_veh{i}.png'
                    check_utils.vis_field(
                        batch_data["head_dicts"]["field"][0], preds["x_traj"]
                    )

                    ## plot offset, need h_query and w_query, save to 'visualization/offset_val.png'
                    gt_offset = batch_data["head_dicts"]["offset_gt"][0]
                    gt_offset_mask = batch_data["head_dicts"]["offset_mask"][0]
                    check_utils.vis_offset_val(
                        gt_offset,
                        gt_offset_mask,
                        preds["x_offset"],
                        hypes,
                        hm=batch_data["head_dicts"]["field"][0],
                        masks=preds["x_traj"],
                    )

            for j, dets_ in enumerate(dets):  # per batch
                # map ground truth class to detection group
                dets_ = mask_box_out_of_range(dets_, hypes)
                if gt_box.shape[0] > 0:
                    gt_box_i = gt_box[gt_box[:, 0] == j, 1:]
                    gt_box_i = map_gt_cls(hypes, gt_box_i)

                    if cfg.metric == "bev":
                        det_box42 = box_utils.boxes_to_corners2d(
                            dets_["box3d_lidar"], order="hwl"
                        )[:, :, :2]
                        gt_box42 = box_utils.boxes_to_corners2d(
                            gt_box_i[:, :7], order="hwl"
                        )[:, :, :2]
                    elif cfg.metric == "3d":
                        det_box42 = box_utils.boxes_to_corners_3d(
                            dets_["box3d_lidar"], order="hwl"
                        )
                        gt_box42 = box_utils.boxes_to_corners_3d(
                            gt_box_i[:, :7], order="hwl"
                        )

                    # update overall tp and fp
                    # eval_cls -1: overall, 0: vehicle, 1: cyclist, 2: pedestrian
                    result_stat = eval_utils.caluclate_tp_fp(
                        det_box42,
                        dets_["scores"],
                        dets_["label_preds"],
                        gt_box42,
                        gt_box_i[:, 7],
                        result_stat,
                        ious,
                        eval_cls,
                    )
                else:
                    if dets_["scores"].shape[0] > 0:
                        eval_name_to_cls = {
                            "overall": -1,
                            "vehicle": 0,
                            "cyclist": 1,
                            "pedestrian": 2,
                        }
                        for eval_cls_ in eval_cls:
                            cls_num = eval_name_to_cls[eval_cls_]
                            if not eval_cls_ == "overall":
                                det_mask = dets_["label_preds"] == cls_num
                            if "overall" in eval_cls:
                                iou_threshs = ious[cls_num + 1, :]
                            else:
                                iou_threshs = ious[cls_num, :]
                            for iou_thresh in iou_threshs:
                                result_stat[eval_cls_][iou_thresh]["fp"] += [1] * dets_[
                                    "box3d_lidar"
                                ][det_mask].shape[0]
                                result_stat[eval_cls_][iou_thresh]["tp"] += [0] * dets_[
                                    "box3d_lidar"
                                ][det_mask].shape[0]
                                result_stat[eval_cls_][iou_thresh]["score"] += dets_[
                                    "scores"
                                ][det_mask].tolist()

            if scenario == "" and cfg.debug_per_scenario is not None:
                print(scenario_name)
            if (
                scenario != ""
                and scenario_name != scenario
                and cfg.debug_per_scenario is not None
                and i != 0
            ):
                eval_utils.eval_final_results(
                    result_stat, None, eval_cls, ious, global_sort_detections=0
                )
                result_stat = ini_result_stat(eval_cls, ious)
                print(scenario_name)
            scenario = scenario_name

            pbar.update(1)

    save = hypes["wild_setting"]["agent_i_delay"]
    if not cfg.global_sort_detections:
        pre = "eval_results"
    else:
        pre = "eval_results_global_sort_detections_"
    if delay_ego > 0:
        pre = pre + f"_ego_{delay_ego}ms"
    path = saved_path + f"/{pre}_epoch_{epoch}_{cfg.dataset}_{save}ms.yaml"
    stat_path = (
        saved_path + f"/{pre}_epoch_{epoch}_{cfg.dataset}_{save}ms_result_stat.npy"
    )
    eval_utils.eval_final_results(
        result_stat,
        path,
        eval_cls,
        ious,
        global_sort_detections=cfg.global_sort_detections,
    )

    np.save(stat_path, result_stat)

    print(f"===============Evaluation done! Results saved to {path}")


def mask_box_out_of_range(dets, params):
    mask_range = params["dataset"]["eval_range"]
    boxes = dets["box3d_lidar"]
    if boxes.shape[0] > 0:
        mask = (
            (boxes[:, 0] >= mask_range[0])
            & (boxes[:, 0] <= mask_range[3])
            & (boxes[:, 1] >= mask_range[1])
            & (boxes[:, 1] <= mask_range[4])
        )
        for key, value in dets.items():
            dets[key] = value[mask]

    return dets


def mask_ego_box(dets_):
    try:
        points = dets_["box3d_lidar"]
    except:
        points = dets_
    mask = (
        (points[:, 0] >= -1.95)
        & (points[:, 0] <= 2.95)
        & (points[:, 1] >= -1.1)
        & (points[:, 1] <= 1.1)
    )
    if mask.sum() >= 1:
        #     print(points[mask][0,-2:])
        mask = torch.logical_not(mask)
        if isinstance(dets_, dict):
            for key, value in dets_.items():
                try:
                    dets_[key] = value[mask]
                except:
                    pass
        else:
            dets_ = dets_[mask]
    return dets_


def init_seeds(seed=2023, cuda_deterministic=True):
    seed = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def map_gt_cls(hypes, gt_box_i):
    # gt_box_i: x,y,z,h,w,l,phi,type,id,time,truncated,occluded

    gt_cls = gt_box_i[:, 7]
    hypes = torch.Tensor(hypes["dataset"]["cls_map"]).cuda()
    gt_cls_ = hypes[gt_cls.long()]

    gt_box_i[:, 7] = gt_cls_

    return gt_box_i


def cls_map(hypes):
    hypes = hypes["dataset"]
    _cls_group = hypes["cls_group"]
    threshold = hypes["eval_iou_threshold"]

    eval_cls, thr = [], []

    for i, _ in enumerate(_cls_group):
        eval_cls.append(i)
        thr.append(threshold[i])

    return eval_cls, thr


def ini_result_stat(
    eval_cls=["overall", "vehicle", "cyclist", "pedestrian"], ious=[0.3, 0.5, 0.7]
):

    result_stat = {}
    for i, cls in enumerate(eval_cls):
        result_stat_ = {}
        iou = ious[i]
        for iou_ in iou:
            result_stat_.update(
                {
                    iou_: {
                        "tp": [],
                        "fp": [],
                        "gt": 0,
                        "score": [],
                        "frame": [],
                        "iou": [],
                    }
                }
            )
        result_stat.update({cls: result_stat_})

    return result_stat


if __name__ == "__main__":
    cfg = test_parser()
    for i, delay in enumerate(cfg.delay):
        main(cfg, delay, delay_ego=cfg.delay_ego[i])
