# -*- coding: utf-8 -*-
# Author: Zhiying Song
# Modified from OpenCOOD Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
Dataset class for intermediate fusion
"""
from collections import OrderedDict

import numpy as np
import torch
from functools import reduce
import copy
from scipy.spatial.distance import cdist

# from utils.pcd_utils import downsample_lidar_minimum
from datasets.utils import preprocess, pre_collate_batch, collate_batch_label
from datasets.data_utils.augmentor.augment_utils import (
    Flip,
    Scaling,
    Rotation,
    Translation,
)
from datasets.center_utils import draw_gaussian, gaussian_radius
import datasets.data_utils.data_check.data_check_utils as check_utils


def getNoFusionDataset(x):

    class NoFusionDataset(x):
        """
        This class is for intermediate fusion where each vehicle transmit the
        deep features to ego.
        """

        def __init__(self, params, set, single_agent=True):
            super(NoFusionDataset, self).__init__(params, set, single_agent=True)

            # if project first, cav's lidar will first be projected to
            # the ego's coordinate frame. otherwise, the feature will be
            # projected instead.
            self.proj_first = True
            if (
                "proj_first" in params["fusion"]["args"]
                and not params["fusion"]["args"]["proj_first"]
            ):
                self.proj_first = False

            # whether there is a time delay between the time that cav project
            # lidar to ego and the ego receive the delivered feature
            self.cur_ego_pose_flag = (
                True
                if "cur_ego_pose_flag" not in params["fusion"]["args"]
                else params["fusion"]["args"]["cur_ego_pose_flag"]
            )

            self.aug_ = params["dataset"]["augment"]
            aug = ["Flip", "Scaling", "Rotation", "Translation"]
            self.aug = aug
            aug_func = [Flip, Scaling, Rotation, Translation]
            for key, values in enumerate(aug):
                self.__setattr__(
                    values, aug_func[key](params["dataset"]["augmentation"][values])
                )
            self.ego_agent = "vehicle"
            self.his_frames = params["dataset"]["frame_his"]
            self.params = params

            self.pre_processor = preprocess
            self.pre_collate_batch = pre_collate_batch
            # self.post_processor = post_processor.build_postprocessor(
            # params['postprocess'],train)

        def __getitem__(self, idx):
            base_data_dict = self.retrieve_base_data(
                idx, cur_ego_pose_flag=self.cur_ego_pose_flag
            )
            processed_data_dict = OrderedDict()
            processed_data_dict["ego"] = {}

            ego_id = -1
            ego_lidar_pose = []

            # first find the ego vehicle's lidar pose
            for cav_id, cav_content in base_data_dict.items():
                if cav_content["ego"]:
                    ego_id = cav_id
                    ego_lidar_pose = cav_content["params"]["lidar_pose"]
                    break
            assert (
                cav_id == list(base_data_dict.keys())[0]
            ), "The first element in the OrderedDict must be ego"
            assert ego_id != -1
            assert len(ego_lidar_pose) > 0

            pairwise_t_matrix = self.get_pairwise_transformation(
                base_data_dict, self.max_cav
            )

            # processed_features = []
            lidars = []

            # prior knowledge for time delay correction and indicating data type
            # (V2V vs V2i)
            velocity = []
            time_delay = []
            infra = []
            spatial_correction_matrix = []
            object_12d = []
            lidar_paths = []
            vehicle_labels = []

            # loop over all CAVs to process information
            cav_num = 0
            for cav_id, selected_cav_base in base_data_dict.items():

                if not isinstance(cav_id, int):
                    continue

                # in single agent mode, only load data from ego vehicle.
                if not selected_cav_base["ego"] and self.single_agent:
                    continue

                selected_cav_processed, void_lidar = self.get_item_single_car(
                    selected_cav_base
                )

                vehicle_labels.append(selected_cav_base["params"]["labels"])
                object_12d.append(selected_cav_processed["object_12d"])
                # processed_features.append(
                #     selected_cav_processed['processed_features'])
                time_delay.append(float(selected_cav_base["time_delay"]))
                velocity.append(selected_cav_processed["velocity"])
                infra.append(1 if int(cav_id) < 0 else 0)
                spatial_correction_matrix.append(
                    selected_cav_base["params"]["spatial_correction_matrix"]
                )

                lidar_paths.append(selected_cav_base["params"]["lidar_path"])
                lidars.append(selected_cav_processed["projected_lidar"])

                frame_threshold = self.his_frames
                if len(selected_cav_base["params"]["lidar_path"]) < frame_threshold:
                    frame_lack = frame_threshold - len(
                        selected_cav_base["params"]["lidar_path"]
                    )
                    for _ in range(frame_lack):
                        lidar_paths[cav_id].append("")
                        lidars[cav_id].append(np.array([]))

                void_lidar_frames = 0
                # self.lidars = lidars
                for li in lidars[cav_id]:
                    void_lidar_frames += 1 if li.shape[0] < 1 else 0

                cav_num += (
                    self.his_frames if selected_cav_base["ego"] else self.cav_his_frames
                )
                cav_num -= void_lidar_frames

            gt_boxes = base_data_dict["gt_boxes"]

            if self.params["dataset"]["merge_pcd_first"]:
                # merge multi frame point cloud first to reduce training burden
                for i in range(len(lidars)):
                    lidars[i] = [np.concatenate(lidars[i])]

            lidars_ = []
            for lidar in lidars:
                lidars_ = lidars_ + lidar
            lidars = lidars_

            object_12d = self.iou_coop_VI_labels(lidars, object_12d, vehicle_labels)

            object_12d = [object_12d[0]]

            # data augmentation
            if self.train and self.aug_:
                augs = self.params["dataset"]["augmentation"]
                boxes = object_12d + [gt_boxes]
                for key, _ in augs.items():
                    if key in self.aug:
                        a = self.__getattribute__(key)
                        try:
                            lidars, boxes = a(lidars, boxes)
                        except:
                            pass
                object_12d = boxes[: len(object_12d)]
                gt_boxes = boxes[-1]

            gt_boxes = self.mask_gt_box_out_of_range(gt_boxes)

            processed_features = []

            extra_feature = 0
            if self.params["wild_setting"]["one_hot_cav_id"]:
                extra_feature += self.max_cav
            if self.params["model"]["reader"]["timestamp"]:
                extra_feature += 1

            for lidar in lidars:
                if lidar.shape[0] > 1:
                    x = self.pre_processor(
                        self.params["voxelization"],
                        lidar,
                        extra_feature=extra_feature,
                        train=self.train,
                    )
                    processed_features.append(x)

            # merge preprocessed features from different cavs into the same dict
            merged_feature_dict = self.merge_features_to_dict(processed_features)

            # generate attention field labels using object 12d
            if "deform" in self.params["model"]:
                field_, offset_gt_, offset_mask_ = self.retrieve_fields(object_12d[0])
            else:
                field_, offset_gt_, offset_mask_ = [], [], []
            # pad dv, dt, infra to max_cav
            time_delay = time_delay + (self.max_cav - len(time_delay)) * [0.0]
            infra = infra + (self.max_cav - len(infra)) * [0.0]
            velocity = velocity + (self.max_cav - len(velocity)) * [0.0]
            spatial_correction_matrix = np.stack(spatial_correction_matrix)
            padding_eye = np.tile(
                np.eye(4)[None], (self.max_cav - len(spatial_correction_matrix), 1, 1)
            )
            spatial_correction_matrix = np.concatenate(
                [spatial_correction_matrix, padding_eye], axis=0
            )

            processed_data_dict["ego"].update(
                {
                    "raw_lidar_points": lidars,
                    "processed_lidar": merged_feature_dict,
                    "label_dict": gt_boxes,
                    "cav_num": cav_num,
                    "time_delay": time_delay,
                    "velocity": velocity,
                    "infra": infra,
                    "pairwise_t_matrix": pairwise_t_matrix,
                    "spatial_correction_matrix": spatial_correction_matrix,
                    "field": field_,
                    "offset_gt": offset_gt_,
                    "offset_mask": offset_mask_,
                    "token_dict": base_data_dict["token_dict"],
                    "lidar_path": lidar_paths,
                }
            )
            return processed_data_dict

        def retrieve_fields(self, boxes):
            # try:
            #     upstride = self.params['model']['head']['upstride']
            # except:
            upstride = 1
            stride = self.params["model"]["deform"]["input_stride"]

            if isinstance(stride, int):
                stride = [stride * upstride]
            else:
                stride = [i * upstride for i in stride]

            field_, offset_gt_, offset_mask_ = [], [], []
            for i in stride:
                self.deform_stride = i

                supervise_num = self.params["model"]["deform"]["offset"][
                    "supervise_num"
                ]
                self.supervise_num = supervise_num
                if supervise_num == 17:
                    offset_gt, offset_mask = self.get_offset_gt_17(boxes)
                if supervise_num == 7:
                    offset_gt, offset_mask = self.get_offset_gt_7(boxes)
                if supervise_num == 9:
                    offset_gt, offset_mask = self.get_offset_gt_9(boxes)
                if supervise_num == 18:
                    offset_gt, offset_mask, boxes = self.get_offset_gt_18(boxes)

                field = self.get_field(boxes)

                # check_utils.vis_fieldvsoffset(field[0],offset_mask[0])
                field_.append(field)
                offset_gt_.append(offset_gt)
                offset_mask_.append(offset_mask)

            return field_, offset_gt_, offset_mask_

        def mask_gt_box_out_of_range(self, boxes):
            try:
                infer_range = self.params["dataset"]["infer_range"]
            except:
                infer_range = False

            try:
                mask_range = (
                    self.params["dataset"]["eval_range"]
                    if infer_range
                    else self.lidar_range
                )
            except:
                mask_range = self.lidar_range

            if boxes.shape[0] > 0 and boxes.shape[1] > 0:
                mask = (
                    (boxes[:, 0] >= mask_range[0])
                    & (boxes[:, 0] <= mask_range[3])
                    & (boxes[:, 1] >= mask_range[1])
                    & (boxes[:, 1] <= mask_range[4])
                )
                boxes = boxes[mask]

            return boxes

        @staticmethod
        def merge_features_to_dict(processed_feature_list):
            """
            Merge the preprocessed features from different cavs to the same
            dictionary.

            Parameters
            ----------
            processed_feature_list : list
                A list of dictionary containing all processed features from different cavs.

            Returns
            -------
            merged_feature_dict: dict
                key: feature names, value: list of features.
            """

            merged_feature_dict = OrderedDict()

            for i in range(len(processed_feature_list)):

                for feature_name, feature in processed_feature_list[i].items():
                    if feature_name not in merged_feature_dict:
                        merged_feature_dict[feature_name] = []
                    if isinstance(feature, list):
                        merged_feature_dict[feature_name] += feature
                    else:
                        merged_feature_dict[feature_name].append(feature)

            return merged_feature_dict

        def collate_batch(self, batch):
            output_dict = {}

            self.batch_size = len(batch)
            head_dicts, processed_lidar_torch_dict, label_torch_dict = (
                self.collate_batch_base(batch)
            )
            token_dicts, raw_lidars, lidar_paths, time_delays = [], [], [], []
            for b in batch:
                token_dicts.append(b["ego"]["token_dict"])
                raw_lidars.append(b["ego"]["raw_lidar_points"])
                time_delays.append(b["ego"]["time_delay"])
                lidar_paths.append(
                    [item for sublist in b["ego"]["lidar_path"] for item in sublist]
                )
            lidar_paths = [item for sublist in lidar_paths for item in sublist]
            ## prepare labels for centerhead
            task = self.params["dataset"]["cls_group"]

            if self.params["model"]["head"]["sep_head"]:
                for task_id in range(len(task)):
                    head_dict = self.get_head_labels(label_torch_dict, task_id)

                    for key, value in head_dict.items():
                        if key in head_dicts:
                            head_dicts[key].append(value)
                        else:
                            head_dicts.update({key: []})
                            head_dicts[key].append(value)

            output_dict = self.collate_batch_opencood(batch, output_dict)

            output_dict.update(
                {
                    "processed_lidar": processed_lidar_torch_dict,
                    "label_dict": label_torch_dict,
                    "head_dicts": head_dicts,
                    "token_dicts": token_dicts,
                    "ego_frames": self.his_frames,
                    "lidar_path": lidar_paths,
                    "time_delays": time_delays,
                }
            )

            # check if the data is correctly loaded.
            # check_utils.check_lidar_alignment(raw_lidars[7],output_dict['label_dict'][label_torch_dict[:,0]==7,:],self.params)

            return output_dict

        def collate_batch_opencood(self, batch, output_dict):

            # used to record different scenario
            record_len = []

            # used for PriorEncoding for models
            velocity = []
            time_delay = []
            infra = []

            # pairwise transformation matrix
            pairwise_t_matrix_list = []

            # used for correcting the spatial transformation between delayed timestamp
            # and current timestamp
            spatial_correction_matrix_list = []

            for i in range(len(batch)):
                ego_dict = batch[i]["ego"]
                record_len.append(ego_dict["cav_num"])
                pairwise_t_matrix_list.append(ego_dict["pairwise_t_matrix"])

                velocity.append(ego_dict["velocity"])
                time_delay.append(ego_dict["time_delay"])
                infra.append(ego_dict["infra"])
                spatial_correction_matrix_list.append(
                    ego_dict["spatial_correction_matrix"]
                )

            # [2, 3, 4, ..., M], M <= max_cav
            record_len = torch.from_numpy(np.array(record_len, dtype=int))

            # (B, max_cav)
            velocity = torch.from_numpy(np.array(velocity))
            time_delay = torch.from_numpy(np.array(time_delay))
            infra = torch.from_numpy(np.array(infra))
            spatial_correction_matrix_list = torch.from_numpy(
                np.array(spatial_correction_matrix_list)
            )

            # # (B, max_cav, 3)
            prior_encoding = torch.stack([velocity, time_delay, infra], dim=-1).float()
            # (B, max_cav)
            pairwise_t_matrix = torch.from_numpy(np.array(pairwise_t_matrix_list))

            # object id is only used during inference, where batch size is 1.
            # so here we only get the first element.
            output_dict.update(
                {
                    "record_len": record_len,
                    "prior_encoding": prior_encoding,
                    "spatial_correction_matrix": spatial_correction_matrix_list,
                    "pairwise_t_matrix": pairwise_t_matrix,
                }
            )
            return output_dict

        def post_process(self, data_dict, output_dict):
            """
            Process the outputs of the model to 2D/3D bounding box.

            Parameters
            ----------
            data_dict : dict
                The dictionary containing the origin input data of model.

            output_dict :dict
                The dictionary containing the output of the model.

            Returns
            -------
            pred_box_tensor : torch.Tensor
                The tensor of prediction bounding box after NMS.
            gt_box_tensor : torch.Tensor
                The tensor of gt bounding box.
            """
            pred_box_tensor, pred_score = self.post_processor.post_process(
                data_dict, output_dict
            )
            gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

            return pred_box_tensor, pred_score, gt_box_tensor

        def get_pairwise_transformation(self, base_data_dict, max_cav):
            """
            Get pair-wise transformation matrix accross different agents.

            Parameters
            ----------
            base_data_dict : dict
                Key : cav id, item: transformation matrix to ego, lidar points.

            max_cav : int
                The maximum number of cav, default 5

            Return
            ------
            pairwise_t_matrix : np.array
                The pairwise transformation matrix across each cav.
                shape: (L, L, 4, 4)
            """
            pairwise_t_matrix = np.zeros((max_cav, max_cav, 4, 4))

            if self.proj_first:
                # if lidar projected to ego first, then the pairwise matrix
                # becomes identity
                pairwise_t_matrix[:, :] = np.identity(4)
            else:
                t_list = []

                # save all transformation matrix in a list in order first.
                for cav_id, cav_content in base_data_dict.items():
                    t_list.append(cav_content["params"]["transformation_matrix"])

                for i in range(len(t_list)):
                    for j in range(len(t_list)):
                        # identity matrix to self
                        if i == j:
                            t_matrix = np.eye(4)
                            pairwise_t_matrix[i, j] = t_matrix
                            continue
                        # i->j: TiPi=TjPj, Tj^(-1)TiPi = Pj
                        t_matrix = np.dot(np.linalg.inv(t_list[j]), t_list[i])
                        pairwise_t_matrix[i, j] = t_matrix

            return pairwise_t_matrix

        def collate_batch_base(self, batch):
            """
            Basic batch collation.

            Parameters
            ----------
            batch : dict
                The dictionary containing all information from '__getitem__'.

            Returns
            -------
            head_dicts: for offset field generation of DTF
            processed_lidar_torch_dict: batch of voxelized features
            label_torch_dict: cooperative box labels
            """

            processed_lidar_list = []
            label_dict_list = []
            field_torch = []
            offset_gt_list = []
            offset_mask_list = []
            head_dicts = {}

            for i in range(len(batch)):
                dict = batch[i]["ego"]
                processed_lidar_list.append(dict["processed_lidar"])
                label_dict_list.append(dict["label_dict"])
                for j in range(len(dict["field"])):
                    dict["field"][j] = torch.from_numpy(dict["field"][j])[None, ...]
                    dict["offset_gt"][j] = torch.from_numpy(dict["offset_gt"][j])[
                        None, ...
                    ]
                    dict["offset_mask"][j] = torch.from_numpy(dict["offset_mask"][j])[
                        None, ...
                    ]

                field_torch.append(dict["field"])
                offset_gt_list.append(dict["offset_gt"])
                offset_mask_list.append(dict["offset_mask"])

            merged_feature_dict = self.merge_features_to_dict(processed_lidar_list)

            processed_lidar_torch_dict = pre_collate_batch(merged_feature_dict)

            # [b,x,y,z,...]
            label_torch_dict = collate_batch_label(label_dict_list)

            x = [field_torch, offset_gt_list, offset_mask_list]
            y = ["field", "offset_gt", "offset_mask"]
            for i, x_ in enumerate(x):
                field_torch2 = []
                for layer in range(len(x_[0])):
                    field_torch2.append([])
                    for batch in range(len(x_)):
                        field_torch2[layer].append(x_[batch][layer])
                    field_torch2[layer] = torch.cat(field_torch2[layer], dim=0)
                head_dicts.update({y[i]: field_torch2})

            return head_dicts, processed_lidar_torch_dict, label_torch_dict

        def get_field(self, boxes):
            """
            Retrieve gt trajectory field

            Parameters
            ----------
            boxes : np.array, Nx10
                [x, y, z, h, w, l, phi, type, id, timestamp]

            """

            heat, pos, box_, time_field = self.get_existence_field(boxes)

            angle_field = self.get_angle_field(box_)

            if self.params["model"]["deform"]["field_dim"] == 4:
                return np.concatenate((heat, angle_field, time_field, pos), axis=0)
            else:
                return np.concatenate((heat, angle_field, pos), axis=0)

        def get_existence_field(self, boxes):
            """
            Retrieve gt angular trajectory field

            Parameters
            ----------
            boxes : np.array, Nx10
                [x, y, z, h, w, l, phi, type, id, timestamp]

            Returns
            -------
            heat: np.array
                heatmap of the trajectories
            pos: np.array, {0,1}
                indicates the existence of trajectories
            box_[index]: np.array
                interpolated boxes on the trajectories
            """

            voxel_size = self.voxel_size * self.deform_stride
            lidar_range = self.lidar_range
            grid_size = self.grid_size / self.deform_stride
            grid_size = np.ceil(grid_size).astype(np.int)

            heat = np.zeros((1, grid_size[0], grid_size[1]))
            time_mask = np.zeros((grid_size[0], grid_size[1])) - 1
            pos = np.zeros((1, grid_size[0], grid_size[1]))
            box2 = []

            if boxes.shape[0] < 1:
                return heat, pos, boxes, time_mask[None, :, :]

            # interpolation on the trajectory
            for alpha in [0.2, 0.4, 0.6, 0.8, 1]:
                x_h = boxes[:, 0] + np.cos(boxes[:, 6]) * boxes[:, 5] / 2 * alpha
                y_h = boxes[:, 1] + np.sin(boxes[:, 6]) * boxes[:, 4] / 2 * alpha
                x_e = boxes[:, 0] - np.cos(boxes[:, 6]) * boxes[:, 5] / 2 * alpha
                y_e = boxes[:, 1] - np.sin(boxes[:, 6]) * boxes[:, 4] / 2 * alpha

                box_h = copy.deepcopy(boxes)
                box_h[:, 0], box_h[:, 1] = x_h, y_h
                box_e = copy.deepcopy(boxes)
                box_e[:, 0], box_e[:, 1] = x_e, y_e
                box2.append(np.vstack((box_h, box_e)))
            box_ = np.vstack((boxes, np.concatenate(box2)))

            xy = box_[:, :2] - lidar_range[:2]
            xy = (xy / voxel_size[:2]).astype(np.int)

            mask_ = reduce(
                np.logical_and,
                [
                    xy[:, 0] < grid_size[1],
                    xy[:, 1] < grid_size[0],
                    xy[:, 0] >= 0,
                    xy[:, 1] >= 0,
                ],
            )
            xy_ = xy[mask_]

            type = box_[:, 7][mask_]

            time = box_[:, 9][mask_]

            radius = (
                np.zeros_like(type) + self.params["dataset"]["centermap"]["traj_radius"]
            )
            radius[type > 3] = radius[type > 3] // 2

            pos[0][(xy_[:, 1]).astype(np.int), (xy_[:, 0]).astype(np.int)] = 1
            index = []
            for i, xyy in enumerate(xy_):
                x_, y_ = xyy
                if time_mask[np.int(y_), np.int(x_)] != -1:
                    if time[i] < time_mask[np.int(y_), np.int(x_)]:
                        heat[0] = draw_gaussian(
                            heat[0], [np.int(x_), np.int(y_)], radius=int(radius[i])
                        )
                        time_mask[np.int(y_), np.int(x_)] = time[i]
                        index.append(i)
                else:
                    heat[0] = draw_gaussian(
                        heat[0], [np.int(x_), np.int(y_)], radius=int(radius[i])
                    )
                    time_mask[np.int(y_), np.int(x_)] = time[i]
                    index.append(i)
            index = np.array(index).astype(np.int)

            time_mask[time_mask == -1] = 0
            time_mask = time_mask / self.params["wild_setting"]["lidar_frequency"]
            # heat[0] = heat[0] * time_mask

            return heat, pos, box_[index], time_mask[None, :, :]

        def get_angle_field(self, boxes):
            """
            Retrieve gt angular trajectory field

            Parameters
            ----------
            boxes : np.array, Nx10
                [x, y, z, h, w, l, phi, type, id, timestamp]

            Returns
            -------
            angular field: np.array
                x and y motion direction of per-pixel trajectory
            """
            # boxes: [x,y,z,h,w,l,phi,type,id,timestamp]
            y_size, x_size = self.grid_size[[1, 0]]
            stride = self.deform_stride
            y_size, x_size = y_size / stride, x_size / stride

            if boxes.shape[0] < 1:
                return np.zeros((2, int(x_size), int(y_size)))

            x, y, phi = boxes[:, 0], boxes[:, 1], boxes[:, 6]

            x_grid, y_grid = np.meshgrid(np.arange(y_size), np.arange(x_size))
            voxel_size = self.voxel_size[0]

            x_ = ((x - self.lidar_range[0]) / voxel_size / stride).astype(np.int)
            y_ = ((y - self.lidar_range[1]) / voxel_size / stride).astype(np.int)

            mask_ = reduce(np.logical_and, [x_ < y_size, y_ < x_size])
            x_, y_ = x_[mask_], y_[mask_]

            if x_.shape[0] > 0:
                img = np.hstack((x_grid.reshape(-1, 1), y_grid.reshape(-1, 1)))
                bx = np.hstack((x_.reshape(-1, 1), y_.reshape(-1, 1)))

                dis = cdist(img, bx, metric="euclidean")

                mini_index = np.argmin(dis, axis=1)
                direc = phi[mini_index].reshape(int(x_size), int(y_size))
                direc = direc - np.pi

                direc_x = np.cos(direc)
                direc_y = np.sin(direc)

                return np.stack([direc_x, direc_y])

            else:
                return np.zeros((2, int(x_size), int(y_size)))

        def get_offset_gt_18(self, boxes):
            """
            Retrieve ground truth deformable offset for two-head 3x3 conv kernels,
            17 pixels in a kernel have supervised offsets, other 1 free

            Parameters
            ----------
            boxes : np.array, Nx12
                [x, y, z, h, w, l, phi, type, id, timestamp, truncated, occluded]

            Returns
            -------
            offset_gt: np.array
                Per-pixel deformable offset
            mask: np.array, {0,1}
                mask == 1: this pixel has non-zero deformable offset
            """
            # check_utils.check_lidar_alignment([],[boxes[:,:7]],self.params,detc='_offset')
            supervise_num = self.params["model"]["deform"]["offset"]["supervise_num"]
            assert supervise_num == 18, "supervised number should be 18"
            voxel_size = self.voxel_size * self.deform_stride
            grid_size = self.grid_size / self.deform_stride
            grid_size = np.ceil(grid_size).astype(np.int)
            lidar_range = self.lidar_range

            offset_gt = np.zeros((supervise_num * 2, grid_size[0], grid_size[1]))
            mask = np.zeros((supervise_num, grid_size[0], grid_size[1]))

            if boxes.shape[0] >= 1:
                ids, indexs = np.unique(boxes[:, 8], return_inverse=True)
                static_large_mask = np.zeros((boxes.shape[0],), dtype=bool)
                for i in range(ids.shape[0]):
                    traj_mask = indexs == i
                    box_traj_i = boxes[traj_mask]
                    box_traj_i = box_traj_i[np.argsort(box_traj_i[:, 9])]  # [::-1]]
                    index_ = []
                    # check_utils.check_lidar_alignment([np.zeros((3,3))*10],[box_traj_i[:,:7]],self.params,detc='_label')

                    if box_traj_i.shape[0] > 0:
                        box_num = box_traj_i.shape[0]
                        x, y = box_traj_i[:, 0], box_traj_i[:, 1]
                        w, phi = box_traj_i[:, 4], box_traj_i[:, 6]

                        static_large = False
                        dist = cdist(box_traj_i[:, :2], box_traj_i[:, :2])
                        if (
                            np.max(dist) <= 0.5 and np.mean(box_traj_i[:, 5]) > 2
                        ):  # regard as static trajectory
                            static_large = True
                            # static_large_mask = np.logical_or(static_large_mask,traj_mask)

                        if not static_large:
                            x_h = (
                                box_traj_i[0, 0]
                                + np.cos(box_traj_i[0, 6]) * box_traj_i[0, 5] / 2
                            )
                            y_h = (
                                box_traj_i[0, 1]
                                + np.sin(box_traj_i[0, 6]) * box_traj_i[0, 5] / 2
                            )
                            w_h, phi_h = box_traj_i[0, 4], box_traj_i[0, 6]
                            x_e = (
                                box_traj_i[-1, 0]
                                - np.cos(box_traj_i[-1, 6]) * box_traj_i[-1, 5] / 2
                            )
                            y_e = (
                                box_traj_i[-1, 1]
                                - np.sin(box_traj_i[-1, 6]) * box_traj_i[-1, 5] / 2
                            )
                            w_e, phi_e = box_traj_i[-1, 4], box_traj_i[-1, 6]

                            x = np.concatenate(([x_h], x, [x_e]))
                            y = np.concatenate(([y_h], y, [y_e]))
                            t = np.concatenate(
                                (
                                    [
                                        [box_traj_i[0, 9]],
                                        box_traj_i[:, 9],
                                        [box_traj_i[-1, 9]],
                                        box_traj_i[:, 9],
                                        box_traj_i[:, 9],
                                    ]
                                )
                            )
                            w = np.concatenate(([w_h], w, [w_e]))
                            phi = np.concatenate(([phi_h], phi, [phi_e]))

                            x_l = x - w / 2 * np.sin(phi)
                            y_l = y + w / 2 * np.cos(phi)
                            x_r = x + w / 2 * np.sin(phi)
                            y_r = y - w / 2 * np.cos(phi)

                            x_tensor = np.vstack((x_l, x, x_r)).T
                            y_tensor = np.vstack((y_l, y, y_r)).T

                            # t_matrix = t.reshape(-1,1).repeat(box_num,1)
                            index = np.array(
                                [ind for ind in range(supervise_num // 3)]
                            ).reshape(1, -1)
                            for i_ in range(box_num):
                                try:
                                    index_ = np.vstack((index_, index + i_))
                                except:
                                    index_ = index.copy()
                            index_[1:, :] = (
                                index_[1:, :] + 1
                            )  # not to see the head of first vehicle

                            mask_matrix = (
                                index_ < box_num + 2
                            )  # min(box_num+2,1+6) # 2: head and tail

                            # if static_large:
                            #     # if the trajectory is static, not supervise its deform offset
                            #     mask_matrix = np.zeros_like(mask_matrix,dtype=bool)

                            index_ = index_ * mask_matrix

                            x_matrix = np.take(x_tensor, index_, axis=0)
                            y_matrix = np.take(y_tensor, index_, axis=0)

                            x_matrix_ = x_matrix.transpose(0, 2, 1).reshape(
                                x_matrix.shape[0], 1, -1
                            )
                            y_matrix_ = y_matrix.transpose(0, 2, 1).reshape(
                                y_matrix.shape[0], 1, -1
                            )
                            mask_matrix_ = (
                                mask_matrix[:, :, None]
                                .repeat(3, -1)
                                .transpose(0, 2, 1)
                                .reshape(mask_matrix.shape[0], -1)
                                .T
                            )

                            c = np.concatenate((y_matrix_, x_matrix_), axis=1)
                            b = c.reshape(c.shape[0], -1)

                            x, y, t_ = (
                                box_traj_i[:, 0],
                                box_traj_i[:, 1],
                                box_traj_i[:, 9].reshape(-1, 1),
                            )
                            x_, y_ = np.floor(
                                (x - lidar_range[0]) / voxel_size[0]
                            ), np.floor((y - lidar_range[1]) / voxel_size[1])
                            clip_ = np.logical_and(x_ < grid_size[1], y_ < grid_size[0])
                            clip_ = np.logical_and(
                                clip_, np.logical_and(x_ >= 0, y_ >= 0)
                            )

                            z_ = np.vstack((x_, y_))
                            keep_mask = np.ones((x_.shape[0],), dtype=bool)
                            for col_index in range(z_.shape[1]):
                                z_tmp = (
                                    z_[:, col_index].reshape(-1, 1) - z_[:, :col_index]
                                )
                                z_tmp = (z_tmp == 0).sum(0)
                                if (z_tmp == 2).sum() != 0:
                                    keep_mask[col_index] = False
                            clip_ = np.logical_and(clip_, keep_mask)
                            x_, y_ = x_[clip_], y_[clip_]
                            b = b[clip_].transpose(1, 0)
                            try:
                                offset_gt[:, y_.astype(np.int), x_.astype(np.int)] = b
                            except:
                                pass
                            mask[:, y_.astype(np.int), x_.astype(np.int)] = (
                                mask_matrix_[:, clip_]
                            )

                return offset_gt, mask, boxes[~static_large_mask]

            else:
                return offset_gt, mask, boxes

        def get_offset_gt_18_v2xseq(self, boxes):
            """
            Retrieve ground truth deformable offset for two-head 3x3 conv kernels,
            17 pixels in a kernel have supervised offsets, other 1 free

            Parameters
            ----------
            boxes : np.array, Nx12
                [x, y, z, h, w, l, phi, type, id, timestamp, truncated, occluded]

            Returns
            -------
            offset_gt: np.array
                Per-pixel deformable offset
            mask: np.array, {0,1}
                mask == 1: this pixel has non-zero deformable offset
            """
            # check_utils.check_lidar_alignment([],[boxes[:,:7]],self.params,detc='_offset')
            supervise_num = self.params["model"]["deform"]["offset"]["supervise_num"]
            assert supervise_num == 18, "supervised number should be 18"
            voxel_size = self.voxel_size * self.deform_stride
            grid_size = self.grid_size / self.deform_stride
            grid_size = np.ceil(grid_size).astype(np.int)
            lidar_range = self.lidar_range

            offset_gt = np.zeros((supervise_num * 2, grid_size[0], grid_size[1]))
            mask = np.zeros((supervise_num, grid_size[0], grid_size[1]))

            if boxes.shape[0] >= 1:
                ids, indexs = np.unique(boxes[:, 8], return_inverse=True)
                static_large_id_list = []
                for i in range(ids.shape[0]):
                    traj_mask = indexs == i
                    box_traj_i = boxes[traj_mask]
                    box_traj_i = box_traj_i[np.argsort(box_traj_i[:, 9])]  # [::-1]]
                    index_ = []
                    # check_utils.check_lidar_alignment([np.zeros((3,3))*10],[box_traj_i[:,:7]],self.params,detc='_label')

                    if box_traj_i.shape[0] > 0:
                        box_num = box_traj_i.shape[0]
                        x, y = box_traj_i[:, 0], box_traj_i[:, 1]
                        w, phi = box_traj_i[:, 4], box_traj_i[:, 6]

                        static_large = False
                        dist = cdist(box_traj_i[:, :2], box_traj_i[:, :2])
                        if (
                            np.max(dist) <= 0.5 and np.mean(box_traj_i[:, 5]) > 2
                        ):  # regard as static trajectory
                            static_large = True
                            static_large_id_list.append(ids[i])

                        x_h = (
                            box_traj_i[0, 0]
                            + np.cos(box_traj_i[0, 6]) * box_traj_i[0, 5] / 2
                        )
                        y_h = (
                            box_traj_i[0, 1]
                            + np.sin(box_traj_i[0, 6]) * box_traj_i[0, 5] / 2
                        )
                        w_h, phi_h = box_traj_i[0, 4], box_traj_i[0, 6]

                        x_e = (
                            box_traj_i[-1, 0]
                            - np.cos(box_traj_i[-1, 6]) * box_traj_i[-1, 5] / 2
                        )
                        y_e = (
                            box_traj_i[-1, 1]
                            - np.sin(box_traj_i[-1, 6]) * box_traj_i[-1, 5] / 2
                        )
                        w_e, phi_e = box_traj_i[-1, 4], box_traj_i[-1, 6]

                        if static_large:
                            # todo
                            box_traj_i = box_traj_i[0].reshape(1, -1)
                            box_num = 1
                            interp = np.linspace(0, 5, 6)
                            x = (x_e - x_h) / 5 * interp + x_h
                            y = (y_e - y_h) / 5 * interp + y_h
                            phi = (phi_e - phi_h) / 5 * interp + phi_h
                            w = (w_e - w_h) / 5 * interp + w_h

                            x_l = x - w / 2 * np.sin(phi)
                            y_l = y + w / 2 * np.cos(phi)
                            x_r = x + w / 2 * np.sin(phi)
                            y_r = y - w / 2 * np.cos(phi)

                            x_tensor = np.vstack((x_l, x, x_r)).T
                            y_tensor = np.vstack((y_l, y, y_r)).T

                            # t_matrix = t.reshape(-1,1).repeat(box_num,1)
                            index = np.array(
                                [ind for ind in range(supervise_num // 3)]
                            ).reshape(1, -1)
                            for i_ in range(box_num):
                                try:
                                    index_ = np.vstack((index_, index + i_))
                                except:
                                    index_ = index.copy()

                            mask_matrix = index_ >= 0

                            # if static_large:
                            #     # if the trajectory is static, not supervise its deform offset
                            #     mask_matrix = np.zeros_like(mask_matrix,dtype=bool)

                            index_ = index_ * mask_matrix

                        else:

                            x = np.concatenate(([x_h], x, [x_e]))
                            y = np.concatenate(([y_h], y, [y_e]))
                            t = np.concatenate(
                                (
                                    [
                                        [box_traj_i[0, 9]],
                                        box_traj_i[:, 9],
                                        [box_traj_i[-1, 9]],
                                        box_traj_i[:, 9],
                                        box_traj_i[:, 9],
                                    ]
                                )
                            )
                            w = np.concatenate(([w_h], w, [w_e]))
                            phi = np.concatenate(([phi_h], phi, [phi_e]))

                            x_l = x - w / 2 * np.sin(phi)
                            y_l = y + w / 2 * np.cos(phi)
                            x_r = x + w / 2 * np.sin(phi)
                            y_r = y - w / 2 * np.cos(phi)

                            x_tensor = np.vstack((x_l, x, x_r)).T
                            y_tensor = np.vstack((y_l, y, y_r)).T

                            # t_matrix = t.reshape(-1,1).repeat(box_num,1)
                            index = np.array(
                                [ind for ind in range(supervise_num // 3)]
                            ).reshape(1, -1)
                            for i_ in range(box_num):
                                try:
                                    index_ = np.vstack((index_, index + i_))
                                except:
                                    index_ = index.copy()
                            index_[1:, :] = (
                                index_[1:, :] + 1
                            )  # not to see the head of first vehicle

                            mask_matrix = (
                                index_ < box_num + 2
                            )  # min(box_num+2,1+6) # 2: head and tail

                            # if static_large:
                            #     # if the trajectory is static, not supervise its deform offset
                            #     mask_matrix = np.zeros_like(mask_matrix,dtype=bool)

                            index_ = index_ * mask_matrix

                        x_matrix = np.take(x_tensor, index_, axis=0)
                        y_matrix = np.take(y_tensor, index_, axis=0)

                        x_matrix_ = x_matrix.transpose(0, 2, 1).reshape(
                            x_matrix.shape[0], 1, -1
                        )
                        y_matrix_ = y_matrix.transpose(0, 2, 1).reshape(
                            y_matrix.shape[0], 1, -1
                        )
                        mask_matrix_ = (
                            mask_matrix[:, :, None]
                            .repeat(3, -1)
                            .transpose(0, 2, 1)
                            .reshape(mask_matrix.shape[0], -1)
                            .T
                        )

                        c = np.concatenate((y_matrix_, x_matrix_), axis=1)
                        b = c.reshape(c.shape[0], -1)

                        x, y, t_ = (
                            box_traj_i[:, 0],
                            box_traj_i[:, 1],
                            box_traj_i[:, 9].reshape(-1, 1),
                        )
                        x_, y_ = np.floor(
                            (x - lidar_range[0]) / voxel_size[0]
                        ), np.floor((y - lidar_range[1]) / voxel_size[1])
                        clip_ = np.logical_and(x_ < grid_size[1], y_ < grid_size[0])
                        clip_ = np.logical_and(clip_, np.logical_and(x_ >= 0, y_ >= 0))

                        z_ = np.vstack((x_, y_))
                        keep_mask = np.ones((x_.shape[0],), dtype=bool)
                        for col_index in range(z_.shape[1]):
                            z_tmp = z_[:, col_index].reshape(-1, 1) - z_[:, :col_index]
                            z_tmp = (z_tmp == 0).sum(0)
                            if (z_tmp == 2).sum() != 0:
                                keep_mask[col_index] = False
                        clip_ = np.logical_and(clip_, keep_mask)
                        x_, y_ = x_[clip_], y_[clip_]
                        b = b[clip_].transpose(1, 0)
                        try:
                            offset_gt[:, y_.astype(np.int), x_.astype(np.int)] = b
                        except:
                            pass
                        mask[:, y_.astype(np.int), x_.astype(np.int)] = mask_matrix_[
                            :, clip_
                        ]

            return offset_gt, mask, boxes

        def get_offset_gt_18_5046(self, boxes):
            """
            Retrieve ground truth deformable offset for two-head 3x3 conv kernels,
            17 pixels in a kernel have supervised offsets, other 1 free

            Parameters
            ----------
            boxes : np.array, Nx12
                [x, y, z, h, w, l, phi, type, id, timestamp, truncated, occluded]

            Returns
            -------
            offset_gt: np.array
                Per-pixel deformable offset
            mask: np.array, {0,1}
                mask == 1: this pixel has non-zero deformable offset
            """
            # check_utils.check_lidar_alignment([],[boxes[:,:7]],self.params,detc='_offset')
            supervise_num = self.params["model"]["deform"]["offset"]["supervise_num"]
            assert supervise_num == 18, "supervised number should be 18"
            voxel_size = self.voxel_size * self.deform_stride
            grid_size = self.grid_size / self.deform_stride
            grid_size = np.ceil(grid_size).astype(np.int)
            lidar_range = self.lidar_range

            ids, indexs = np.unique(boxes[:, 8], return_inverse=True)

            offset_gt = np.zeros((supervise_num * 2, grid_size[0], grid_size[1]))
            mask = np.zeros((supervise_num, grid_size[0], grid_size[1]))

            for i in range(ids.shape[0]):
                traj_mask = indexs == i
                box_traj_i = boxes[traj_mask]
                box_traj_i = box_traj_i[np.argsort(box_traj_i[:, 9])]  # [::-1]]
                index_ = []

                if box_traj_i.shape[0] > 0:
                    box_num = box_traj_i.shape[0]
                    x, y = box_traj_i[:, 0], box_traj_i[:, 1]
                    w, phi = box_traj_i[:, 4], box_traj_i[:, 6]

                    # if box_num <= 9-3-2:
                    x_h = (
                        box_traj_i[0, 0]
                        + np.cos(box_traj_i[0, 6]) * box_traj_i[0, 5] / 2
                    )
                    y_h = (
                        box_traj_i[0, 1]
                        + np.sin(box_traj_i[0, 6]) * box_traj_i[0, 5] / 2
                    )
                    w_h, phi_h = box_traj_i[0, 4], box_traj_i[0, 6]

                    x_e = (
                        box_traj_i[-1, 0]
                        - np.cos(box_traj_i[-1, 6]) * box_traj_i[-1, 5] / 2
                    )
                    y_e = (
                        box_traj_i[-1, 1]
                        - np.sin(box_traj_i[-1, 6]) * box_traj_i[-1, 5] / 2
                    )
                    w_e, phi_e = box_traj_i[-1, 4], box_traj_i[-1, 6]

                    x = np.concatenate(([x_h], x, [x_e]))
                    y = np.concatenate(([y_h], y, [y_e]))
                    t = np.concatenate(
                        (
                            [
                                [box_traj_i[0, 9]],
                                box_traj_i[:, 9],
                                [box_traj_i[-1, 9]],
                                box_traj_i[:, 9],
                                box_traj_i[:, 9],
                            ]
                        )
                    )
                    w = np.concatenate(([w_h], w, [w_e]))
                    phi = np.concatenate(([phi_h], phi, [phi_e]))

                    x_l = x - w / 2 * np.sin(phi)
                    y_l = y + w / 2 * np.cos(phi)
                    x_r = x + w / 2 * np.sin(phi)
                    y_r = y - w / 2 * np.cos(phi)

                    x_tensor = np.vstack((x_l, x, x_r)).T
                    y_tensor = np.vstack((y_l, y, y_r)).T

                    # t_matrix = t.reshape(-1,1).repeat(box_num,1)
                    index = np.array(
                        [ind for ind in range(supervise_num // 3)]
                    ).reshape(1, -1)
                    for i_ in range(box_num):
                        try:
                            index_ = np.vstack((index_, index + i_))
                        except:
                            index_ = index.copy()
                    index_[1:, :] = (
                        index_[1:, :] + 1
                    )  # not to see the head of first vehicle

                    mask_matrix = (
                        index_ < box_num + 2
                    )  # min(box_num+2,1+6) # 2: head and tail
                    index_ = index_ * mask_matrix

                    x_matrix = np.take(x_tensor, index_, axis=0)
                    y_matrix = np.take(y_tensor, index_, axis=0)

                    x_matrix_ = x_matrix.transpose(0, 2, 1).reshape(
                        x_matrix.shape[0], 1, -1
                    )
                    y_matrix_ = y_matrix.transpose(0, 2, 1).reshape(
                        y_matrix.shape[0], 1, -1
                    )
                    mask_matrix_ = (
                        mask_matrix[:, :, None]
                        .repeat(3, -1)
                        .transpose(0, 2, 1)
                        .reshape(mask_matrix.shape[0], -1)
                        .T
                    )

                    c = np.concatenate((y_matrix_, x_matrix_), axis=1)
                    b = c.reshape(c.shape[0], -1)

                    x, y, t_ = (
                        box_traj_i[:, 0],
                        box_traj_i[:, 1],
                        box_traj_i[:, 9].reshape(-1, 1),
                    )
                    x_, y_ = np.floor((x - lidar_range[0]) / voxel_size[0]), np.floor(
                        (y - lidar_range[1]) / voxel_size[1]
                    )
                    clip_ = np.logical_and(x_ < grid_size[1], y_ < grid_size[0])
                    clip_ = np.logical_and(clip_, np.logical_and(x_ >= 0, y_ >= 0))

                    z_ = np.vstack((x_, y_))
                    keep_mask = np.ones((x_.shape[0],), dtype=bool)
                    for col_index in range(z_.shape[1]):
                        z_tmp = z_[:, col_index].reshape(-1, 1) - z_[:, :col_index]
                        z_tmp = (z_tmp == 0).sum(0)
                        if (z_tmp == 2).sum() != 0:
                            keep_mask[col_index] = False
                    clip_ = np.logical_and(clip_, keep_mask)
                    x_, y_ = x_[clip_], y_[clip_]
                    b = b[clip_].transpose(1, 0)
                    try:
                        offset_gt[:, y_.astype(np.int), x_.astype(np.int)] = b
                    except:
                        pass
                    mask[:, y_.astype(np.int), x_.astype(np.int)] = mask_matrix_[
                        :, clip_
                    ]

            # from utils.vis_utils import vis_gt_offset
            # if int(input("illustrate offset: ")) == 1:
            #     vis_gt_offset(offset_gt,mask)

            return offset_gt, mask

        def get_offset_gt_18_experimental(self, boxes):
            """
            Retrieve ground truth deformable offset for two-head 3x3 conv kernels,
            17 pixels in a kernel have supervised offsets, other 1 free

            Parameters
            ----------
            boxes : np.array, Nx12
                [x, y, z, h, w, l, phi, type, id, timestamp, truncated, occluded]

            Returns
            -------
            offset_gt: np.array
                Per-pixel deformable offset
            mask: np.array, {0,1}
                mask == 1: this pixel has non-zero deformable offset
            """
            # check_utils.check_lidar_alignment([],[boxes[:,:7]],self.params,detc='_offset')
            supervise_num = self.params["model"]["deform"]["offset"]["supervise_num"]
            assert supervise_num == 18, "supervised number should be 18"
            voxel_size = self.voxel_size * self.deform_stride
            grid_size = self.grid_size / self.deform_stride
            grid_size = np.ceil(grid_size).astype(np.int)
            lidar_range = self.lidar_range

            ids, indexs = np.unique(boxes[:, 8], return_inverse=True)

            offset_gt = np.zeros((supervise_num * 2, grid_size[0], grid_size[1]))
            mask = np.zeros((supervise_num, grid_size[0], grid_size[1]))

            for i in range(ids.shape[0]):
                traj_mask = indexs == i
                box_traj_i = boxes[traj_mask]
                box_traj_i = box_traj_i[np.argsort(box_traj_i[:, 9])]  # [::-1]]
                index_ = []

                if box_traj_i.shape[0] > 0:
                    box_num = box_traj_i.shape[0]
                    x, y = box_traj_i[:, 0], box_traj_i[:, 1]
                    w, phi = box_traj_i[:, 4], box_traj_i[:, 6]

                    static = False
                    dist = cdist(box_traj_i[:, :2], box_traj_i[:, :2])
                    if np.max(dist) <= 0.5:  # regard as static trajectory
                        static = True

                        # x_h=box_traj_i[0,0]+np.cos(box_traj_i[0,6])*box_traj_i[0,5]/2
                        # y_h=box_traj_i[0,1]+np.sin(box_traj_i[0,6])*box_traj_i[0,5]/2
                        # w_h,phi_h = box_traj_i[0,4],box_traj_i[0,6]

                        # x_e=box_traj_i[-1,0]-np.cos(box_traj_i[-1,6])*box_traj_i[-1,5]/2
                        # y_e=box_traj_i[-1,1]-np.sin(box_traj_i[-1,6])*box_traj_i[-1,5]/2
                        # w_e,phi_e = box_traj_i[-1,4],box_traj_i[-1,6]

                        # interp = np.linspace(0,5,6)
                        # x = (x_e-x_h)/5*interp + x_h
                        # y = (y_e-y_h)/5*interp + y_h
                        # phi = (phi_e-phi_h)/5*interp + phi_h
                        # w = (w_e-w_h)/5*interp + w_h

                        # box_traj_i = box_traj_i[0,:].reshape(1,-1).repeat(6,0)
                        # box_traj_i[:,0], box_traj_i[:,1] = x, y
                        # box_traj_i[:,4], box_traj_i[:,6] = w, phi
                        # box_num=box_traj_i.shape[0]

                        # # x=np.concatenate(([x_h],x,[x_e]))
                        # # y=np.concatenate(([y_h],y,[y_e]))
                        # # t=np.concatenate(([[box_traj_i[0,9]],box_traj_i[:,9],[box_traj_i[-1,9]],box_traj_i[:,9],box_traj_i[:,9]]))
                        # # w=np.concatenate(([w_h],w,[w_e]))
                        # # phi=np.concatenate(([phi_h],phi,[phi_e]))

                    x_l = x - w / 2 * np.sin(phi)
                    y_l = y + w / 2 * np.cos(phi)
                    x_r = x + w / 2 * np.sin(phi)
                    y_r = y - w / 2 * np.cos(phi)

                    x_tensor = np.vstack((x_l, x, x_r)).T
                    y_tensor = np.vstack((y_l, y, y_r)).T

                    # t_matrix = t.reshape(-1,1).repeat(box_num,1)
                    index = np.array(
                        [ind for ind in range(supervise_num // 3)]
                    ).reshape(1, -1)

                    index_ = []
                    for i_ in range(box_num):
                        try:
                            index_ = np.vstack((index_, index + i_))
                        except:
                            index_ = index.copy()
                    # index_[1:,:] = index_[1:,:] + 1 # not to see the head of first vehicle

                    mask_matrix = (
                        index_ < box_num
                    )  # min(box_num+2,1+6) # 2: head and tail
                    if static:
                        # if the trajectory is static, not supervise its deform offset
                        mask_matrix = np.zeros_like(mask_matrix, dtype=bool)
                    index_ = index_ * mask_matrix

                    x_matrix = np.take(x_tensor, index_, axis=0)
                    y_matrix = np.take(y_tensor, index_, axis=0)

                    x_matrix_ = x_matrix.transpose(0, 2, 1).reshape(
                        x_matrix.shape[0], 1, -1
                    )
                    y_matrix_ = y_matrix.transpose(0, 2, 1).reshape(
                        y_matrix.shape[0], 1, -1
                    )
                    mask_matrix_ = (
                        mask_matrix[:, :, None]
                        .repeat(3, -1)
                        .transpose(0, 2, 1)
                        .reshape(mask_matrix.shape[0], -1)
                        .T
                    )

                    c = np.concatenate((y_matrix_, x_matrix_), axis=1)
                    b = c.reshape(c.shape[0], -1)

                    x, y = box_traj_i[:, 0], box_traj_i[:, 1]
                    x_, y_ = np.floor((x - lidar_range[0]) / voxel_size[0]), np.floor(
                        (y - lidar_range[1]) / voxel_size[1]
                    )
                    clip_ = np.logical_and(x_ < grid_size[1], y_ < grid_size[0])
                    clip_ = np.logical_and(clip_, np.logical_and(x_ >= 0, y_ >= 0))

                    z_ = np.vstack((x_, y_))
                    keep_mask = np.ones((x_.shape[0],), dtype=bool)
                    for col_index in range(z_.shape[1]):
                        z_tmp = z_[:, col_index].reshape(-1, 1) - z_[:, :col_index]
                        z_tmp = (z_tmp == 0).sum(0)
                        if (z_tmp == 2).sum() != 0:
                            keep_mask[col_index] = False
                    clip_ = np.logical_and(clip_, keep_mask)
                    x_, y_ = x_[clip_], y_[clip_]
                    b = b[clip_].transpose(1, 0)
                    try:
                        offset_gt[:, y_.astype(np.int), x_.astype(np.int)] = b
                    except:
                        pass
                    mask[:, y_.astype(np.int), x_.astype(np.int)] = mask_matrix_[
                        :, clip_
                    ]

            # from utils.vis_utils import vis_gt_offset
            # if int(input("illustrate offset: ")) == 1:
            #     vis_gt_offset(offset_gt,mask)

            return offset_gt, mask

        def get_offset_gt_9(self, boxes):
            """
            Retrieve ground truth deformable offset for one-head 3x3 conv kernels,
            9 pixels in a kernel have supervised offsets

            Parameters
            ----------
            boxes : np.array, Nx12
                [x, y, z, h, w, l, phi, type, id, timestamp, truncated, occluded]

            Returns
            -------
            offset_gt: np.array
                Per-pixel deformable offset
            mask: np.array, {0,1}
                mask == 1: this pixel has non-zero deformable offset
            """
            # check_utils.check_lidar_alignment([],[boxes[:,:7]],self.params,detc='_offset')
            supervise_num = self.params["model"]["deform"]["offset"]["supervise_num"]

            voxel_size = self.voxel_size * self.deform_stride
            grid_size = self.grid_size / self.deform_stride
            grid_size = np.ceil(grid_size).astype(np.int)
            lidar_range = self.lidar_range

            ids, indexs = np.unique(boxes[:, 8], return_inverse=True)

            offset_gt = np.zeros((supervise_num * 2, grid_size[0], grid_size[1]))
            mask = np.zeros((supervise_num, grid_size[0], grid_size[1]))

            for i in range(ids.shape[0]):
                traj_mask = indexs == i
                box_traj_i = boxes[traj_mask]
                box_traj_i = box_traj_i[np.argsort(box_traj_i[:, 9])]  # [::-1]]
                index_ = []
                if box_traj_i.shape[0] > 0:
                    box_num = box_traj_i.shape[0]
                    x, y = box_traj_i[:, 0], box_traj_i[:, 1]
                    super_mask = [1] * x.shape[0]

                    if box_num <= supervise_num - 2:
                        x_h = (
                            box_traj_i[0, 0]
                            + np.cos(box_traj_i[0, 6]) * box_traj_i[0, 5] / 2
                        )
                        y_h = (
                            box_traj_i[0, 1]
                            + np.sin(box_traj_i[0, 6]) * box_traj_i[0, 5] / 2
                        )
                        x_e = (
                            box_traj_i[-1, 0]
                            - np.cos(box_traj_i[-1, 6]) * box_traj_i[-1, 5] / 2
                        )
                        y_e = (
                            box_traj_i[-1, 1]
                            - np.sin(box_traj_i[-1, 6]) * box_traj_i[-1, 5] / 2
                        )

                        x = np.concatenate(([x_h], x, [x_e]))
                        y = np.concatenate(([y_h], y, [y_e]))
                        super_mask = super_mask + [1, 1]

                    zeros = np.zeros((supervise_num - 1,))
                    x = np.concatenate((x, zeros + x[-1]))
                    y = np.concatenate((y, zeros + y[-1]))
                    super_mask = super_mask + [0] * (supervise_num - 1)
                    super_mask = np.asarray(super_mask)

                    x_matrix = x.reshape(-1, 1).repeat(box_num, 1)
                    y_matrix = y.reshape(-1, 1).repeat(box_num, 1)
                    mask_matrix = super_mask.reshape(-1, 1).repeat(box_num, 1)

                    # t_matrix = t.reshape(-1,1).repeat(box_num,1)
                    index = np.array([ind for ind in range(box_num)]).reshape(1, -1)
                    for i_ in range(0, supervise_num):
                        try:
                            index_ = np.vstack((index_, index + i_))
                        except:
                            index_ = index.copy()
                    x_matrix_ = x_matrix[index_, np.arange(0, box_num)].T[:, None, :]
                    y_matrix_ = y_matrix[index_, np.arange(0, box_num)].T[:, None, :]
                    mask_matrix_ = mask_matrix[index_, np.arange(0, box_num)]
                    # t_matrix_ = t_matrix[index_,np.arange(0,box_num)].T[:,None,:]

                    c = np.concatenate((y_matrix_, x_matrix_), axis=1)

                    b = c.reshape(c.shape[0], -1)

                    x, y, t_ = (
                        box_traj_i[:, 0],
                        box_traj_i[:, 1],
                        box_traj_i[:, 9].reshape(-1, 1),
                    )
                    x_, y_ = np.floor((x - lidar_range[0]) / voxel_size[0]), np.floor(
                        (y - lidar_range[1]) / voxel_size[1]
                    )
                    clip_ = np.logical_and(x_ < grid_size[1], y_ < grid_size[0])
                    clip_ = np.logical_and(clip_, np.logical_and(x_ >= 0, y_ >= 0))

                    x_, y_ = x_[clip_], y_[clip_]

                    b = b[clip_].transpose(1, 0)
                    try:
                        offset_gt[:, y_.astype(np.int), x_.astype(np.int)] = b
                    except:
                        pass
                    mask[:, y_.astype(np.int), x_.astype(np.int)] = mask_matrix_[
                        :, clip_
                    ]

            # from utils.vis_utils import vis_gt_offset
            # if int(input("illustrate offset: ")) == 1:
            #     vis_gt_offset(offset_gt,mask)

            return offset_gt, mask

        def get_offset_gt_17(self, boxes):
            """
            Retrieve ground truth deformable offset for two-head 3x3 conv kernels,
            17 pixels in a kernel have supervised offsets, other 1 free

            Parameters
            ----------
            boxes : np.array, Nx12
                [x, y, z, h, w, l, phi, type, id, timestamp, truncated, occluded]

            Returns
            -------
            offset_gt: np.array
                Per-pixel deformable offset
            mask: np.array, {0,1}
                mask == 1: this pixel has non-zero deformable offset
            """
            # check_utils.check_lidar_alignment([],[boxes[:,:7]],self.params,detc='_offset')
            supervise_num = self.params["model"]["deform"]["offset"]["supervise_num"]
            his_frame = self.params["dataset"]["frame_his"]

            voxel_size = self.voxel_size * self.deform_stride
            grid_size = self.grid_size / self.deform_stride
            grid_size = np.ceil(grid_size).astype(np.int)
            lidar_range = self.lidar_range

            ids, indexs = np.unique(boxes[:, 8], return_inverse=True)

            offset_gt = np.zeros((supervise_num * 2, grid_size[0], grid_size[1]))
            mask = np.zeros((supervise_num, grid_size[0], grid_size[1]))

            for i in range(ids.shape[0]):
                traj_mask = indexs == i
                box_traj_i = boxes[traj_mask]
                box_traj_i = box_traj_i[np.argsort(box_traj_i[:, 9])][
                    :his_frame, :
                ]  # [::-1]]

                if box_traj_i.shape[0] > 0:
                    box_num = box_traj_i.shape[0]
                    x, y = box_traj_i[:, 0], box_traj_i[:, 1]
                    w, phi = box_traj_i[:, 4], box_traj_i[:, 6]

                    x_h = (
                        box_traj_i[0, 0]
                        + np.cos(box_traj_i[0, 6]) * box_traj_i[0, 5] / 2
                    )
                    y_h = (
                        box_traj_i[0, 1]
                        + np.sin(box_traj_i[0, 6]) * box_traj_i[0, 5] / 2
                    )
                    x_e = (
                        box_traj_i[-1, 0]
                        - np.cos(box_traj_i[-1, 6]) * box_traj_i[-1, 5] / 2
                    )
                    y_e = (
                        box_traj_i[-1, 1]
                        - np.sin(box_traj_i[-1, 6]) * box_traj_i[-1, 5] / 2
                    )

                    x_l = x - w / 2 * np.sin(phi)
                    y_l = y + w / 2 * np.cos(phi)
                    x_r = x + w / 2 * np.sin(phi)
                    y_r = y - w / 2 * np.cos(phi)

                    x = np.concatenate(([x_h], x, [x_e], x_l, x_r))
                    y = np.concatenate(([y_h], y, [y_e], y_l, y_r))
                    t = np.concatenate(
                        (
                            [
                                [box_traj_i[0, 9]],
                                box_traj_i[:, 9],
                                [box_traj_i[-1, 9]],
                                box_traj_i[:, 9],
                                box_traj_i[:, 9],
                            ]
                        )
                    )

                    assert (
                        supervise_num >= x.shape[0]
                    ), f"x shape is {x.shape[0]}, super_num is {supervise_num}, adjust supervised number (y) according to historic frame number (x), y>=3x+2"
                    zeros = np.zeros((supervise_num - x.shape[0],))
                    x = np.concatenate((x, zeros + x[1]))
                    y = np.concatenate((y, zeros + y[1]))
                    c = np.vstack((y, x))[None, :, :].repeat(box_num, axis=0)
                    t = t[None, :].repeat(box_num, axis=0)

                    b = c.reshape(c.shape[0], -1)

                    x, y, t_ = (
                        box_traj_i[:, 0],
                        box_traj_i[:, 1],
                        box_traj_i[:, 9].reshape(-1, 1),
                    )
                    x_, y_ = np.floor((x - lidar_range[0]) / voxel_size[0]), np.floor(
                        (y - lidar_range[1]) / voxel_size[1]
                    )
                    clip_ = np.logical_and(x_ < grid_size[1], y_ < grid_size[0])
                    clip_ = np.logical_and(clip_, np.logical_and(x_ >= 0, y_ >= 0))

                    x_, y_ = x_[clip_], y_[clip_]
                    b = b[clip_].transpose(1, 0)
                    offset_gt[:, y_.astype(np.int), x_.astype(np.int)] = b
                    # mask[0][y_.astype(np.int), x_.astype(np.int)] = 1
                    t_mask = (t >= t_)[clip_].T
                    mask[: t_mask.shape[0], y_.astype(np.int), x_.astype(np.int)] = (
                        t_mask
                    )

            # from utils.vis_utils import vis_gt_offset
            # if int(input("illustrate offset: ")) == 1:
            #     vis_gt_offset(offset_gt,mask)

            return offset_gt, mask

        def get_offset_gt_7(self, boxes):
            """
            Retrieve ground truth deformable offset for 3x3 conv kernel

            Parameters
            ----------
            boxes : np.array, Nx12
                [x, y, z, h, w, l, phi, type, id, timestamp, truncated, occluded]

            Returns
            -------
            offset_gt: np.array
                Per-pixel deformable offset
            mask: np.array
                mask == 1: this pixel has non-zero deformable offset
            """

            supervise_num = self.supervise_num

            lidar_range = self.lidar_range

            voxel_size = self.voxel_size * self.deform_stride
            grid_size = self.grid_size / self.deform_stride
            grid_size = np.ceil(grid_size).astype(np.int)

            ids, indexs = np.unique(boxes[:, 8], return_inverse=True)

            offset_gt = np.zeros((supervise_num * 2, grid_size[0], grid_size[1]))
            mask = np.zeros((1, grid_size[0], grid_size[1]))

            for i in range(ids.shape[0]):
                traj_mask = indexs == i
                box_traj_i = boxes[traj_mask]
                box_traj_i = box_traj_i[np.argsort(box_traj_i[:, 9])]

                if box_traj_i.shape[0] > 0:
                    box_num = box_traj_i.shape[0]
                    x, y = box_traj_i[:, 0], box_traj_i[:, 1]
                    x_h = (
                        box_traj_i[0, 0]
                        + np.cos(box_traj_i[0, 6]) * box_traj_i[0, 5] / 2
                    )
                    y_h = (
                        box_traj_i[0, 1]
                        + np.sin(box_traj_i[0, 6]) * box_traj_i[0, 4] / 2
                    )
                    x_e = (
                        box_traj_i[-1, 0]
                        - np.cos(box_traj_i[-1, 6]) * box_traj_i[-1, 5] / 2
                    )
                    y_e = (
                        box_traj_i[-1, 1]
                        - np.sin(box_traj_i[-1, 6]) * box_traj_i[-1, 4] / 2
                    )

                    x = np.concatenate(([x_h], x, [x_e]))
                    y = np.concatenate(([y_h], y, [y_e]))

                    if box_num == 1:
                        x = np.concatenate((x, x, [x[1]]))
                        y = np.concatenate((y, y, [y[1]]))
                    else:
                        zeros = np.zeros((supervise_num - x.shape[0],))
                        x = np.concatenate((x, zeros + x[1]))
                        y = np.concatenate((y, zeros + y[1]))

                    c = np.vstack((y, x))[None, :, :].repeat(box_num, axis=0)

                    b = c.reshape(c.shape[0], -1)

                    x, y = box_traj_i[:, 0], box_traj_i[:, 1]
                    x_, y_ = np.floor((x - lidar_range[0]) / voxel_size[0]), np.floor(
                        (y - lidar_range[1]) / voxel_size[1]
                    )
                    clip_ = np.logical_and(x_ < grid_size[1], y_ < grid_size[0])
                    clip_ = np.logical_and(clip_, np.logical_and(x_ >= 0, y_ >= 0))

                    x_, y_ = x_[clip_], y_[clip_]
                    b = np.flipud(b[clip_]).transpose(1, 0)
                    offset_gt[:, y_.astype(np.int), x_.astype(np.int)] = b
                    mask[0][y_.astype(np.int), x_.astype(np.int)] = 1

            return offset_gt, mask

        def get_head_labels(self, box_list_, task_id):
            """
            Retrieve annotation boxes for seperate heads

            Parameters
            ----------
            box_list_ : torch.Tensor, Nx13
                Annotation boxes,
                    0 , 1,2,3,4,5,6,7,   8,  9,  10,  11,      12
                [batch,x,y,z,h,w,l,phi,type,id,time,truncated,occluded]

            task_id : int
                Id of detection head

            Returns
            -------
            head_dict : dict
                Dictionary of annotation boxes for head [task_id]
            """

            head_dict = {}
            params = self.params
            box_list = box_list_.clone()
            cls_group = params["dataset"]["cls_group"]
            classes = cls_group[task_id]
            cls_map = torch.Tensor(params["dataset"]["cls_map"])

            # map the type id to the target task group
            box_list[:, 8] = cls_map[box_list[:, 8].long()]
            self.cats = len(params["dataset"]["cls_group"][task_id])

            # get subtype in tasks, for example, if cls_group = [["Car"],["Cyclist","Pedestrian"]],
            # subtype of "Car" and "Cyclist" is 0, "Pedestrian" is 1
            cls_dict = np.asarray(self.params["dataset"]["cls"])
            a = False
            zero = torch.zeros((box_list.shape[0], 1))
            box_list = torch.hstack((box_list, zero))
            for i, cls in enumerate(classes):
                b = box_list[:, 8] == cls_map[int(np.where(cls == cls_dict)[0])]
                box_list[torch.where(b == True)[0], -1] = i
                a = np.logical_or(b.cpu(), a)
            box = box_list[a.to(torch.bool)]

            try:
                upstride = self.params["model"]["head"]["upstride"]
            except:
                upstride = 1
            self.out_size_factor = (
                params["model"]["backbone"]["out_size_factor"][task_id] * upstride
            )

            head_dict = self.get_anno_box(head_dict, box)

            return head_dict

        def get_anno_box(self, head_dict, box):
            bs = self.batch_size
            voxel_size = torch.from_numpy(self.voxel_size) * self.out_size_factor
            lidar_range = torch.from_numpy(self.lidar_range)
            grid_size = (lidar_range[3:6] - lidar_range[:3]) / voxel_size
            grid_size = torch.ceil(grid_size).to(torch.int)

            heat_map_batch = torch.zeros((bs, 1, grid_size[1], grid_size[0]))
            reg_map_batch = torch.zeros((bs, 2, grid_size[1], grid_size[0]))
            height_batch = torch.zeros((bs, 1, grid_size[1], grid_size[0]))
            dim_batch = torch.zeros((bs, 3, grid_size[1], grid_size[0]))
            rot_batch = torch.zeros((bs, 2, grid_size[1], grid_size[0]))
            mask_batch = torch.zeros((bs, 1, grid_size[1], grid_size[0]))
            cat_batch = torch.zeros((bs, 1, grid_size[1], grid_size[0]))
            gt_box_batch = torch.zeros((bs, 7, grid_size[1], grid_size[0]))

            # batch,x,y,z,h,w,l,phi,type,id,t,subtype
            x, y = box[:, 1] - lidar_range[0], box[:, 2] - lidar_range[1]
            x_, y_ = torch.floor(x / voxel_size[0]), torch.floor(y / voxel_size[1])

            mask_ = reduce(
                torch.logical_and,
                [x_ < grid_size[0], y_ < grid_size[1], x_ >= 0, y_ >= 0],
            )
            x_, y_ = x_[mask_], y_[mask_]
            x, y = x[mask_], y[mask_]

            box = box[mask_]

            w, l = box[:, 5], box[:, 6]
            size_l, size_w = torch.ceil(l / voxel_size[0]), torch.ceil(
                w / voxel_size[1]
            )

            if box.shape[0] > 0:
                min_radius = self.params["dataset"]["centermap"]["min_radius"]
                batch = box[:, 0].long()
                for i, line in enumerate(box):
                    radius = gaussian_radius(
                        (size_w[i], size_l[i]),
                        min_overlap=self.params["dataset"]["centermap"][
                            "gaussian_overlap"
                        ],
                    )
                    radius = max(int(radius), min_radius)
                    heat_map_batch[batch[i]][0] = draw_gaussian(
                        heat_map_batch[batch[i]][0], [x_[i], y_[i]], radius=radius
                    )

                x_reconstruct, y_reconstruct = x_ * voxel_size[0], y_ * voxel_size[1]
                reg_map_batch[batch, 0, y_.long(), x_.long()] = x - x_reconstruct
                reg_map_batch[batch, 1, y_.long(), x_.long()] = y - y_reconstruct
                height_batch[batch, 0, y_.long(), x_.long()] = box[:, 3]  # z
                cat_batch[batch, 0, y_.long(), x_.long()] = box[:, -1]

                dim_batch[batch, 0, y_.long(), x_.long()] = torch.log(box[:, 4])  # h
                dim_batch[batch, 1, y_.long(), x_.long()] = torch.log(box[:, 5])  # w
                dim_batch[batch, 2, y_.long(), x_.long()] = torch.log(box[:, 6])  # l

                rot_batch[batch, 0, y_.long(), x_.long()] = torch.sin(
                    box[:, 7]
                )  # sin alpha
                rot_batch[batch, 1, y_.long(), x_.long()] = torch.cos(
                    box[:, 7]
                )  # cos alpha

                mask_batch[batch, 0, y_.long(), x_.long()] = 1
                gt_box_batch[batch, :, y_.long(), x_.long()] = box[:, 1:8]

            head_dict["hm"] = heat_map_batch

            b, c, h, w = gt_box_batch.shape
            gt_box_batch = gt_box_batch.reshape(b, c, h * w)
            head_dict["gt_boxes"] = gt_box_batch

            head_dict["mask"] = mask_batch.reshape(self.batch_size, -1)
            ind = torch.arange(head_dict["mask"].shape[1], dtype=torch.int64)
            head_dict["ind"] = (
                ind[None, :].repeat(self.batch_size, 1) * head_dict["mask"]
            ).to(torch.int64)
            head_dict["cat"] = cat_batch.reshape(self.batch_size, -1).to(torch.int64)

            anno_box = torch.cat(
                (reg_map_batch, height_batch, dim_batch, rot_batch), dim=1
            )
            b, c, h, w = anno_box.shape
            anno_box = anno_box.reshape(b, c, h * w)
            head_dict["anno_box"] = anno_box.permute(0, 2, 1)

            return head_dict

        def debug_by_visualize(self, output_dict):

            pass

    return NoFusionDataset
