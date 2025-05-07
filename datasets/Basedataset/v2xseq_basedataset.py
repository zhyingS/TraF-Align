# -*- coding: utf-8 -*-
# Author: Zhiying Song
# Modified from OpenCOOD Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
Basedataset class for all kinds of fusion.
"""

import os
from collections import OrderedDict
import yaml
import json
from functools import reduce
import numpy as np
from torch.utils.data import Dataset
from random import randint
import scipy

import utils.pcd_utils as pcd_utils
from utils.transformation_utils import x1_to_x2_matrix
from utils.pcd_utils import mask_points_by_range, mask_ego_points
from utils import box_utils
from shapely.geometry import Polygon


class V2XSEQBaseDataset(Dataset):
    """
    Mainly used to initialize the
    database and associate the __get_item__ index with the correct timestamp and scenario.

    Parameters
    __________
    params : dict
        The dictionary contains all parameters for training/testing.

    Attributes
    ----------
    scenario_database : OrderedDict
        A structured dictionary contains all file information.

    len_record : list
        The list to record each scenario's data length. This is used to
        retrieve the correct index during training.

    pre_processor :
        Used to preprocess the raw data.

    post_processor :
        Used to generate training labels and convert the model outputs to
        bbx formats.

    data_augmentor :
        Used to augment data.

    """

    def __init__(self, params, set, single_agent=False):
        self.params = params
        self.set = set
        self.single_agent = single_agent

        if set == "train":
            self.train = True
        else:
            self.train = False

        self.pre_processor = None
        self.post_processor = None

        # if the training/testing include noisy setting
        if "wild_setting" in params:
            wild_parms = params["wild_setting"]
            self.seed = wild_parms["seed"]  # random seed for localization noise
            # whether to add time delay
            self.async_flag = wild_parms["async"]
            self.async_mode = "sim"
            self.agent_i_delay = wild_parms["agent_i_delay"]
            self.async_ego = wild_parms["async_ego"]
            # localization error, not support yet
            self.loc_calib = wild_parms["loc_calib"]

            self.loc_err = wild_parms["loc_err"]
            if self.loc_err:
                raise ("Not support adding loc noise yet")
            self.xyz_noise_std = wild_parms["xyz_std"]
            self.ryp_noise_std = wild_parms["ryp_std"]

            self.ego_agent = wild_parms["ego"]
            self.lidar_freq = wild_parms.get("lidar_frequency", 10)
            self.agent_i_delay_train_aug = wild_parms.get(
                "agent_i_delay_train_aug", [0, 0]
            )

        else:
            self.async_flag = False
            self.agent_i_delay = 0  # ms
            self.async_mode = "sim"
            self.loc_calib = False
            self.xyz_noise_std = 0
            self.ryp_noise_std = 0
            self.ego_agent = None
            self.lidar_freq = 10
            self.async_ego = False

        self.root_dir = params["root_dir"]

        self.coop_label_dict = self.update_object_param(params, side="coop")
        self.veh_label_dict = self.update_object_param(params, side="veh")
        self.infra_label_dict = self.update_object_param(params, side="infra")

        # if the dataset type is nofusion, only load ego labels while training
        if self.single_agent and self.train:
            if self.ego_agent == "vehicle":
                self.coop_label_dict = self.veh_label_dict
            elif self.ego_agent == "infra":
                self.coop_label_dict = self.infra_label_dict
            else:
                raise ("Please assign the ego vehicle.")

        self.vehicle_infos = self.get_data_structure(params, agent="vehicle")
        self.infra_infos = self.get_data_structure(params, agent="infrastructure")

        self.lidar_range = params["voxelization"]["lidar_range"]
        self.voxel_size = params["voxelization"]["voxel_size"]
        self.grid_size = params["voxelization"]["grid_size"]

        if "train_params" not in params or "max_cav" not in params["train_params"]:
            self.max_cav = 2
        else:
            self.max_cav = params["train_params"]["max_cav"]

        # first load all paths of different scenarios
        self.scenario_folders = self.structure_scenario(params)
        # Structure: {scenario_id : {cav_1 : {timestamp1 : {ego: True/False,
        # data_path: path}}}}
        self.scenario_database = OrderedDict()
        self.len_record = []
        self.COM_RANGE = 10000

        self.reinitialize()

    def reinitialize(self):
        # loop over all scenarios
        for i, scenario_folder in enumerate(self.scenario_folders):
            self.scenario_database.update({i: OrderedDict()})

            cav_list = [0, 1]
            # loop over all CAV data
            for j, cav_id in enumerate(cav_list):
                if j > self.max_cav - 1:
                    print("too many cavs")
                    break
                self.scenario_database[i][cav_id] = OrderedDict()

                timestamps = list(
                    self.scenario_folders[scenario_folder][str(cav_id)].keys()
                )
                if cav_id == 0:
                    t_first = timestamps.copy()

                cav_path = self.scenario_folders[scenario_folder][str(cav_id)][
                    "data_path"
                ]
                agent = self.scenario_folders[scenario_folder][str(cav_id)]["agent"]
                eliminate_list = ["ego", "agent", "data_path"]
                for k, timestamp in enumerate(timestamps):
                    if timestamp in eliminate_list:
                        continue

                    self.scenario_database[i][cav_id][t_first[k]] = OrderedDict()
                    loc_err = self.scenario_folders[scenario_folder][str(cav_id)][
                        timestamp
                    ]["system_error_offset"]

                    yaml_file = os.path.join(cav_path + "data_info.json")
                    self.scenario_database[i][cav_id][t_first[k]]["yaml"] = yaml_file
                    self.scenario_database[i][cav_id][t_first[k]][
                        "timestamp"
                    ] = timestamp
                    self.scenario_database[i][cav_id][t_first[k]][
                        "sequence"
                    ] = scenario_folder
                    self.scenario_database[i][cav_id][t_first[k]][
                        "system_error_offset"
                    ] = loc_err

                self.scenario_database[i][cav_id]["agent"] = agent
                self.scenario_database[i][cav_id]["ego"] = self.scenario_folders[
                    scenario_folder
                ][str(cav_id)]["ego"]

                if j == 0:
                    if not self.len_record:
                        self.len_record.append(len(timestamps) - len(eliminate_list))
                    else:
                        prev_last = self.len_record[-1]
                        self.len_record.append(
                            prev_last + len(timestamps) - len(eliminate_list)
                        )

        self.len_record = np.array(self.len_record)

    def shuffle_each_epoch(self):
        raise ("Shuffling Ego is not supported on v2x-seq dataset.")

    def __len__(self):
        return self.len_record[-1]

    def __getitem__(self, idx):
        """
        Abstract method, needs to be define by the children class.
        """
        pass

    def retrieve_base_data(self, idx, cur_ego_pose_flag=False):
        """
        Given the index, return the corresponding data.

        Parameters
        ----------
        idx : int
            Index given by dataloader.

        cur_ego_pose_flag : bool
            Indicate whether to use current timestamp ego pose to calculate
            transformation matrix. If set to false, meaning when other cavs
            project their LiDAR point cloud to ego, they are projecting to
            past ego pose.

        Returns
        -------
        data : dict
            The dictionary contains loaded yaml params and lidar data for each cav.
        """
        # print(f"{self.set}, idx is {idx}, len_record is {self.len_record}, {idx<=self.len_record}")
        scenario_index = np.where((idx < self.len_record) == True)[0][0]
        scenario_database = self.scenario_database[scenario_index]

        if "selected_frame" in self.params:
            frame = self.params["selected_frame"]
            index_ = list(scenario_database[0]).index(frame)
            keys_to_remove = list(scenario_database[0].keys())[: index_ - 5]
            for key in keys_to_remove:
                del scenario_database[0][key]
                del scenario_database[1][key]

        # check the timestamp index
        timestamp_index = (
            idx
            if scenario_index == 0
            else idx - self.len_record[scenario_index - 1] - 1
        )
        # retrieve the corresponding timestamp key
        assert timestamp_index <= len(
            scenario_database[0]
        ), f"The timestamp index is {timestamp_index}, but the scenario has only {len(scenario_database[0])} frames."

        timestamp_key = self.return_timestamp_key(scenario_database, timestamp_index)

        ego_cav_content = self.get_ego_content(scenario_database)

        data = OrderedDict()
        max_delay, min_max_delay = 0, 0
        ego_his_frame = self.params["dataset"]["frame_his"]
        cav_his_frame = self.params["dataset"]["cav_frame_his"]  # 1
        for cav_id, cav_content in scenario_database.items():

            data[cav_id] = OrderedDict()
            data[cav_id]["ego"] = cav_content["ego"]

            # calculate delay for this vehicle
            data[cav_id]["time_delay"] = self.time_delay_calculation(cav_content["ego"])
            if not cav_content["ego"]:
                max_delay = max(max_delay, cav_his_frame + data[cav_id]["time_delay"])
            else:
                max_delay = max(max_delay, ego_his_frame + data[cav_id]["time_delay"])
        self.max_delay = max_delay

        for cav_id, cav_content in scenario_database.items():
            if self.single_agent and not cav_content["ego"]:
                continue
            timestamp_delay = data[cav_id]["time_delay"]

            if timestamp_index - timestamp_delay <= 0:
                timestamp_delay = timestamp_index
            timestamp_index_delay = max(0, timestamp_index - timestamp_delay)
            timestamp_key_delay = self.return_timestamp_key(
                scenario_database, timestamp_index_delay
            )

            data[cav_id]["params"] = self.reform_param(
                cav_id,
                cav_content,
                ego_cav_content,
                timestamp_key,
                timestamp_key_delay,
                cur_ego_pose_flag,
                timestamp_delay,
            )

            data[cav_id]["gt_his_label"] = []

            data[cav_id]["lidar_np"] = data[cav_id]["params"]["lidar"]

            seq = cav_content[timestamp_key]["sequence"]
            if max_delay > min_max_delay:
                selected_cav_id = cav_id
            min_max_delay = max_delay

            if cav_content["ego"]:
                ego_cav_id = cav_id
                ego_timestamp_key = timestamp_key
                if cav_content["agent"] == "vehicle":
                    self.ego_agent = "vehicle"
                    # for kitti evaluation
                    data["token_dict"] = self.vehicle_infos[seq][
                        cav_content[timestamp_key]["timestamp"]
                    ]
                    gt_boxes_ego = self.generate_object_center(
                        [self.veh_label_dict[cav_content[timestamp_key]["timestamp"]]]
                    )
                else:
                    self.ego_agent = "infra"
                    data["token_dict"] = self.infra_infos[seq][
                        cav_content[timestamp_key]["timestamp"]
                    ]
                    gt_boxes_ego = self.generate_object_center(
                        [self.infra_label_dict[data["timestamp"]]]
                    )

                data["gt_boxes_ego"] = mask_ego_points(gt_boxes_ego)

            if self.ego_agent == "infra" and not cav_content["ego"]:
                data["label_transform"] = data[cav_id]["params"][
                    "gt_transformation_matrix"
                ]
            if cav_content["agent"] == "vehicle":
                vehicle_timestamp_key = cav_content[timestamp_key]["timestamp"]
            else:
                infra_timestamp_key = cav_content[timestamp_key]["timestamp"]

        pose_cur = data[ego_cav_id]["params"]["his_lidar_pos"][0]
        for t in range(max_delay):
            # for t in range(ego_his_frame):
            time_ = str(int(ego_timestamp_key) - t).zfill(len(ego_timestamp_key))
            if time_ in ego_cav_content:
                if ego_cav_content[time_]["timestamp"] in self.coop_label_dict:
                    gt_his_label = self.coop_label_dict[
                        ego_cav_content[time_]["timestamp"]
                    ]
                    pose_cur_t = data[ego_cav_id]["params"]["his_lidar_pos"][t]
                    T = np.linalg.inv(pose_cur).dot(pose_cur_t)
                    object_12d = self.generate_object_center([gt_his_label], T, time=t)
                    data[ego_cav_id]["gt_his_label"].append(object_12d)

        data["timestamp"] = vehicle_timestamp_key
        data["label"] = self.coop_label_dict[vehicle_timestamp_key]

        # generate cooperative labels [x,y,z,h,w,l,phi,type,id,time]
        if "label_transform" in data:
            label_transform = data["label_transform"]
        else:
            label_transform = np.eye(4)

        gt_boxes = self.generate_object_center(
            [self.coop_label_dict[data["timestamp"]]], label_transform
        )
        gt_boxes = mask_ego_points(gt_boxes)

        data["gt_boxes"] = gt_boxes

        return data

    def get_item_single_car(self, selected_cav_base):
        """
        Project the lidar and bbx to ego space first, and then do clipping.

        Parameters
        ----------
        selected_cav_base : dict
            The dictionary contains a single CAV's raw information.

        Returns
        -------
        selected_cav_processed : dict
            The dictionary contains the cav's processed information.
        """
        selected_cav_processed = {}

        # calculate the transformation matrix
        transformation_matrix = selected_cav_base["params"]["transformation_matrix"]

        # filter lidar
        lidar_np = selected_cav_base["lidar_np"]

        # project the lidar to ego space
        if not selected_cav_base["ego"]:
            for i, lidar in enumerate(lidar_np):
                if "inf_lidar_range" in self.params["voxelization"]:
                    lidar = mask_points_by_range(
                        lidar, self.params["voxelization"]["inf_lidar_range"]
                    )
                    lidar_np[i] = lidar
                if self.proj_first:
                    lidar_np[i][:, :3] = box_utils.project_points_by_matrix_torch(
                        lidar[:, :3], transformation_matrix
                    )
        else:
            if self.train or self.params.get("delay_ego", 0) > 0:
                # ego may has delay while training
                for i, lidar in enumerate(lidar_np):
                    lidar_np[i][:, :3] = box_utils.project_points_by_matrix_torch(
                        lidar[:, :3], transformation_matrix
                    )
        lidar_np = mask_ego_points(lidar_np)

        void_lidar = True if lidar_np[0].shape[0] < 1 else False

        # velocity
        velocity = 0  # ego speed,
        if selected_cav_base["ego"]:
            object_12d = np.vstack((selected_cav_base["gt_his_label"]))
        else:
            object_12d = []

        selected_cav_processed.update(
            {
                "projected_lidar": lidar_np,
                "velocity": velocity,
                "object_12d": object_12d,
            }
        )

        return selected_cav_processed, void_lidar

    def iou_coop_VI_labels(self, lidar, object_12d, vehicle_labels):
        try:
            infer_range = self.params["dataset"]["infer_range"]
        except:
            infer_range = False

        try:
            filter_range = (
                self.params["dataset"]["eval_range"]
                if infer_range
                else self.params["voxelization"]["lidar_range"]
            )
        except:
            filter_range = self.params["voxelization"]["lidar_range"]

        xmin, ymin, zmin, xmax, ymax, zmax = filter_range[:6]

        for i in range(len(vehicle_labels)):
            try:
                vehicle_labels[i] = np.concatenate(vehicle_labels[i])
                # compute iou to filter object 12d boxes

                mask_ = reduce(
                    np.logical_and,
                    [
                        vehicle_labels[i][:, 0] >= xmin,
                        vehicle_labels[i][:, 0] <= xmax,
                        vehicle_labels[i][:, 1] >= ymin,
                        vehicle_labels[i][:, 1] <= ymax,
                        vehicle_labels[i][:, 2] >= zmin,
                        vehicle_labels[i][:, 2] <= zmax,
                    ],
                )
                vehicle_labels[i] = vehicle_labels[i][mask_]
            except:
                pass

        if object_12d[0].shape[0] > 0 and len(vehicle_labels) > 0:
            boxes_b = object_12d[0][:, :7]
            keep_mask = np.zeros((boxes_b.shape[0],), dtype=bool)
            dis_threshold = 0.5

            for i in range(len(vehicle_labels)):
                if vehicle_labels[i].shape[0] == 0:
                    continue
                boxes_a = vehicle_labels[i][:, :7]
                if boxes_a.shape[0] > 0 and boxes_b.shape[0] > 0:
                    dis_matirx = scipy.spatial.distance_matrix(
                        boxes_b[:, :2], boxes_a[:, :2]
                    )
                    keep_mask_i = np.min(dis_matirx, axis=1) <= dis_threshold
                else:
                    keep_mask_i = np.zeros((boxes_b.shape[0],), dtype=bool)

                keep_mask = np.logical_or(keep_mask, keep_mask_i)

            object_12d[0] = object_12d[0][keep_mask]

        return object_12d

    def box2polygon(self, boxes):
        # boxes: np.array Nx7
        boxes_bev = box_utils.boxes_to_corners2d(boxes, order="hwl")
        polygons = [
            Polygon([(box[i, 0], box[i, 1]) for i in range(4)]) for box in boxes_bev
        ]
        return np.array(polygons)

    def generate_object_center(self, objects, transform=np.eye(4), time=None):
        """
        Retrieve all objects in a format of (n, 12)

        Parameters
        ----------
        objects : list
            List of dictionary, save cavs' object information.

        transform : array
            Transformation from cav to ego.

        Returns
        -------
        object_np : np.ndarray
            Shape is (max_num, 12). [x,y,z,h,w,l,phi,type,id,time,truncated,occluded]
        """
        boxes = np.array([[]])

        for i, tmp_object_list in enumerate(objects):

            for object in tmp_object_list:
                loc = object["3d_location"]
                x, y, z = loc["x"], loc["y"], loc["z"]
                phi = object["rotation"]
                dim = object["3d_dimensions"]
                try:
                    order = self.params["post_processing"]["order"]
                except:
                    order = "hwl"

                h, w, l = dim["h"], dim["w"], dim["l"]
                object_id = int(object["track_id"])
                object_type = object["type"]
                if time is None:
                    timestamp = i
                else:
                    timestamp = time
                assert order in [
                    "hwl",
                    "lwh",
                ], "Only order 'hwl' and 'lwh' is supported."

                truncated = object["truncated_state"]
                occluded = object["occluded_state"]

                cls = np.asarray(self.params["dataset"]["cls"])
                type = int(np.where(object_type == cls)[0])

                if order == "hwl":
                    box = np.array(
                        [
                            [
                                x,
                                y,
                                z,
                                h,
                                w,
                                l,
                                phi,
                                type,
                                object_id,
                                timestamp,
                                truncated,
                                occluded,
                            ]
                        ]
                    )
                else:
                    box = np.array(
                        [
                            [
                                x,
                                y,
                                z,
                                l,
                                w,
                                h,
                                phi,
                                type,
                                object_id,
                                timestamp,
                                truncated,
                                occluded,
                            ]
                        ]
                    )

                if boxes.shape[1] == 0:
                    boxes = box
                else:
                    boxes = np.vstack((boxes, box))

        if boxes.shape[0] > 0 and boxes.shape[1] > 0:
            boxes[:, :7] = box_utils.project_box7d(boxes[:, :7], transform)

        return boxes

    # @staticmethod
    def return_timestamp_key(self, scenario_database, timestamp_index):
        """
        Given the timestamp index, return the correct timestamp key, e.g.
        2 --> '000078'.

        Parameters
        ----------
        scenario_database : OrderedDict
            The dictionary contains all contents in the current scenario.

        timestamp_index : int
            The index for timestamp.

        Returns
        -------
        timestamp_key : str
            The timestamp key saved in the cav dictionary.
        """
        # get all timestamp keys
        timestamp_keys = list(scenario_database.items())[0][1]
        key_to_remove = ["agent", "ego"]
        timestamp_keys = timestamp_keys.copy()
        for key in key_to_remove:
            timestamp_keys.pop(key)
        # retrieve the correct index
        timestamp_key = list(timestamp_keys.items())[timestamp_index][0]

        return timestamp_key

    def time_delay_calculation(self, ego_flag):
        """
        Calculate the time delay for a certain vehicle.

        Parameters
        ----------
        ego_flag : boolean
            Whether the current cav is ego.

        Return
        ------
        time_delay : int
            The time delay quantization.
        """
        interval = 1e3 // self.lidar_freq
        # if adding delay on ego while train or test
        if (
            (self.train == False or self.async_ego == False)
            and self.params.get("delay_ego", 0) == 0
            and ego_flag
        ):
            return 0
        if self.train == False and ego_flag and self.params.get("delay_ego", 0) != 0:
            return self.params.get("delay_ego", 0) // (interval)

        time_delay = np.abs(self.agent_i_delay)

        if self.train:
            rand_delay = randint(
                self.agent_i_delay_train_aug[0], self.agent_i_delay_train_aug[1]
            )
            time_delay = rand_delay // interval * interval

        time_delay = time_delay // interval
        self.delay = time_delay if self.async_flag else 0

        return int(self.delay)

    def add_loc_noise(self, pose, xyz_std, ryp_std):
        """
        Add localization noise to the pose.

        Parameters
        ----------
        pose : list
            x,y,z,roll,yaw,pitch

        xyz_std : float
            std of the gaussian noise on xyz

        ryp_std : float
            std of the gaussian noise
        """
        np.random.seed(self.seed)
        xyz_noise = np.random.normal(0, xyz_std, 3)
        ryp_std = np.random.normal(0, ryp_std, 3)
        noise_pose = [
            pose[0] + xyz_noise[0],
            pose[1] + xyz_noise[1],
            pose[2] + xyz_noise[2],
            pose[3],
            pose[4] + ryp_std[1],
            pose[5],
        ]
        return noise_pose

    def reform_param(
        self,
        cav_id,
        cav_content,
        ego_content,
        timestamp_cur,
        timestamp_delay,
        cur_ego_pose_flag,
        delay,
    ):
        """
        Reform the data params with current timestamp object groundtruth and
        delay timestamp LiDAR pose for other CAVs.

        Parameters
        ----------
        cav_content : dict
            Dictionary that contains all file paths in the current cav/rsu.

        timestamp_cur : str
            The current timestamp.

        timestamp_delay : str
            The delayed timestamp.

        cur_ego_pose_flag : bool
            Whether use current ego pose to calculate transformation matrix.

        Return
        ------
        The merged parameters.
        """
        ego_lidar_path_cur, ego_pose_list_cur, ego_lidar_pose_cur = (
            self.get_lidar_pose_44(ego_content, timestamp_cur)
        )
        ego_lidar_path_delay, ego_pose_list_delay, ego_lidar_pose_delay = (
            self.get_lidar_pose_44(ego_content, timestamp_delay)
        )

        cav_lidar_path_cur, cav_pose_list_cur, cav_lidar_pose_cur = (
            self.get_lidar_pose_44(cav_content, timestamp_cur)
        )
        cav_lidar_path_delay, cav_pose_list_delay, cav_lidar_pose_delay = (
            self.get_lidar_pose_44(cav_content, timestamp_delay)
        )
        if not cav_content["ego"]:
            correction = cav_content[timestamp_delay]["system_error_offset"]
            correction = np.array([correction["delta_x"], correction["delta_y"], 0])

            gt_correction = cav_content[timestamp_cur]["system_error_offset"]
            gt_correction = np.array(
                [gt_correction["delta_x"], gt_correction["delta_y"], 0]
            )
        else:
            correction = np.zeros((3, 1))
            gt_correction = np.zeros((3, 1))

        if not self.loc_calib:
            correction = np.zeros((3, 1))

        # cav_delay -> ego_delay -> ego_current
        transformation_matrix = x1_to_x2_matrix(
            cav_lidar_pose_delay, ego_lidar_pose_delay, correction
        )
        spatial_correction_matrix = x1_to_x2_matrix(
            ego_lidar_pose_delay, ego_lidar_pose_cur
        )
        transformation_matrix = spatial_correction_matrix @ transformation_matrix
        spatial_correction_matrix = np.eye(4)

        gt_transformation_matrix = x1_to_x2_matrix(
            cav_lidar_pose_cur, ego_lidar_pose_cur, gt_correction
        )

        delay_params = OrderedDict()
        delay_params["transformation_matrix"] = transformation_matrix
        delay_params["gt_transformation_matrix"] = gt_transformation_matrix
        delay_params["spatial_correction_matrix"] = spatial_correction_matrix
        # delay = float(timestamp_cur) - float(timestamp_delay)

        # delay = float(timestamp_cur) - float(timestamp_delay)
        if cav_content["ego"]:
            # retrieve lidars
            delay_params["lidar"] = self.get_his_lidar(
                ego_lidar_path_delay,
                ego_pose_list_delay,
                cav_one_hot=cav_id,
                delay=delay,
            )
            delay_params["lidar_path"] = ego_lidar_path_delay

            delay_params["lidar_pose"] = ego_lidar_pose_delay
            # timestamp_cur = str(int(timestamp_cur)-6).rjust(6,'0')
            # retrieve lables
            vehicle_labels = self.get_vehicle_label(
                cav_content, timestamp_cur, ego_pose_list_cur, delay
            )
            keep_list = []
            for i in range(len(vehicle_labels)):
                if vehicle_labels[i].shape[0] > 0:
                    try:
                        vehicle_labels[i][:, :7] = box_utils.project_box7d(
                            vehicle_labels[i][:, :7], gt_transformation_matrix
                        )
                        keep_list.append(i)
                    except:
                        pass
            if len(keep_list) > 0:
                delay_params["labels"] = [vehicle_labels[i] for i in keep_list]
            else:
                delay_params["labels"] = [np.array([])]
        else:
            delay_params["lidar"] = self.get_his_lidar(
                cav_lidar_path_delay,
                cav_pose_list_delay,
                cav_one_hot=cav_id,
                delay=delay,
            )
            delay_params["lidar_path"] = cav_lidar_path_delay

            delay_params["lidar_pose"] = cav_lidar_pose_delay

            vehicle_labels = self.get_vehicle_label(
                cav_content, timestamp_cur, cav_pose_list_cur, delay
            )

            keep_list = []
            for i in range(len(vehicle_labels)):
                if vehicle_labels[i].shape[0] > 0:
                    try:
                        vehicle_labels[i][:, :7] = box_utils.project_box7d(
                            vehicle_labels[i][:, :7], gt_transformation_matrix
                        )
                        keep_list.append(i)
                    except:
                        pass
            if len(keep_list) > 0:
                delay_params["labels"] = [vehicle_labels[i] for i in keep_list]
            else:
                delay_params["labels"] = [np.array([])]

        delay_params["his_lidar_pos"] = ego_pose_list_cur
        return delay_params

    def get_vehicle_label(self, cav_content, timestamp_key, pose_list, delay):

        data = []

        ego = cav_content["ego"]
        his_frame = self.params["dataset"]["frame_his"]
        if not ego:
            his_frame = self.params["dataset"]["cav_frame_his"]

        pose_cur = pose_list[0]
        his_frame = int(his_frame + delay)
        if cav_content["agent"] == "vehicle":
            for t in range(his_frame):
                time_ = str(int(timestamp_key) - t).zfill(len(timestamp_key))
                if time_ in cav_content:
                    if cav_content[time_]["timestamp"] in self.veh_label_dict:
                        label = self.veh_label_dict[cav_content[time_]["timestamp"]]
                        try:
                            pose_cur_t = pose_list[t]
                            T = np.linalg.inv(pose_cur).dot(pose_cur_t)
                            label_object = self.generate_object_center(
                                [label], T, time=t
                            )
                            data.append(label_object)
                        except:
                            pass
                    else:
                        data.append(np.array([]))
                else:
                    data.append(np.array([]))

        else:
            for t in range(his_frame):
                time_ = str(int(timestamp_key) - t).zfill(len(timestamp_key))
                if time_ in cav_content:
                    if cav_content[time_]["timestamp"] in self.infra_label_dict:
                        label = self.infra_label_dict[cav_content[time_]["timestamp"]]
                        try:
                            pose_cur_t = pose_list[t]
                            T = np.linalg.inv(pose_cur).dot(pose_cur_t)
                            label_object = self.generate_object_center(
                                [label], T, time=t
                            )
                            data.append(label_object)
                        except:
                            pass
                    else:
                        data.append(np.array([]))
                else:
                    data.append(np.array([]))

        return data

    def get_his_lidar(self, lidar_path_list, pose_list, cav_one_hot=0, delay=0):
        try:
            concat_t = self.params["model"]["reader"]["timestamp"]
        except:
            concat_t = False
        if concat_t:
            lidar_cur = pcd_utils.pcd_to_np(
                lidar_path_list[0], -delay / self.lidar_freq
            )
        else:
            lidar_cur = pcd_utils.pcd_to_np(lidar_path_list[0])

        pose_cur = pose_list[0]

        concat_one_hot_id = self.params["wild_setting"].get("one_hot_cav_id", False)

        if concat_one_hot_id:
            cav_one_hot = int(cav_one_hot)
            cav_one_hot_feature = np.zeros((lidar_cur.shape[0], self.max_cav))
            cav_one_hot_feature[:, cav_one_hot] = 1

            lidars = [np.hstack((lidar_cur, cav_one_hot_feature))]
        else:
            lidars = [lidar_cur]
        for t in range(1, len(lidar_path_list)):
            lidar_cur_t = lidar_path_list[t]
            pose_cur_t = pose_list[t]
            if concat_t:
                lidar_cur_t = pcd_utils.pcd_to_np(
                    lidar_cur_t, -(t + delay) / self.lidar_freq
                )
            else:
                lidar_cur_t = pcd_utils.pcd_to_np(lidar_cur_t)

            T = np.linalg.inv(pose_cur).dot(pose_cur_t)
            R, t = T[:-1, :-1], T[:-1, -1]
            lidar_cur_t_ = R.dot(lidar_cur_t[:, :3].T) + t.reshape(-1, 1)
            lidar_cur_t[:, :3] = lidar_cur_t_.T

            if concat_one_hot_id:
                # lidar_cur = np.vstack((lidar_cur,lidar_cur_t))
                cav_one_hot_feature = np.zeros((lidar_cur_t.shape[0], self.max_cav))
                cav_one_hot_feature[:, cav_one_hot] = 1

                lidar_cur_t = np.hstack((lidar_cur_t, cav_one_hot_feature))
            lidars.append(lidar_cur_t)

        return lidars

    @staticmethod
    def load_camera_files(cav_path, timestamp):
        """
        Retrieve the paths to all camera files.

        Parameters
        ----------
        cav_path : str
            The full file path of current cav.

        timestamp : str
            Current timestamp

        Returns
        -------
        camera_files : list
            The list containing all camera png file paths.
        """
        camera0_file = os.path.join(cav_path, timestamp + "_camera0.png")
        camera1_file = os.path.join(cav_path, timestamp + "_camera1.png")
        camera2_file = os.path.join(cav_path, timestamp + "_camera2.png")
        camera3_file = os.path.join(cav_path, timestamp + "_camera3.png")
        return [camera0_file, camera1_file, camera2_file, camera3_file]

    def project_points_to_bev_map(self, points, ratio=0.1):
        """
        Project points to BEV occupancy map with default ratio=0.1.

        Parameters
        ----------
        points : np.ndarray
            (N, 3) / (N, 4)

        ratio : float
            Discretization parameters. Default is 0.1.

        Returns
        -------
        bev_map : np.ndarray
            BEV occupancy map including projected points
            with shape (img_row, img_col).

        """
        return self.pre_processor.project_points_to_bev_map(points, ratio)

    def augment(self, lidar_np, object_bbx_center, object_bbx_mask):
        """
        Given the raw point cloud, augment by flipping and rotation.

        Parameters
        ----------
        lidar_np : np.ndarray
            (n, 4) shape

        object_bbx_center : np.ndarray
            (n, 7) shape to represent bbx's x, y, z, h, w, l, yaw

        object_bbx_mask : np.ndarray
            Indicate which elements in object_bbx_center are padded.
        """
        tmp_dict = {
            "lidar_np": lidar_np,
            "object_bbx_center": object_bbx_center,
            "object_bbx_mask": object_bbx_mask,
        }
        tmp_dict = self.data_augmentor.forward(tmp_dict)

        lidar_np = tmp_dict["lidar_np"]
        object_bbx_center = tmp_dict["object_bbx_center"]
        object_bbx_mask = tmp_dict["object_bbx_mask"]

        return lidar_np, object_bbx_center, object_bbx_mask

    def visualize_result(
        self, pred_box_tensor, gt_tensor, pcd, show_vis, save_path, dataset=None
    ):
        # visualize the model output
        self.post_processor.visualize(
            pred_box_tensor, gt_tensor, pcd, show_vis, save_path, dataset=dataset
        )

    def structure_scenario(self, params):

        with open(params["split_dir"]) as f:
            split = yaml.safe_load(f)[self.set]

        scenario_folders = OrderedDict()
        with open(self.root_dir + "cooperative/data_info.json") as f:
            data_info = json.load(f)

        for frame in data_info:
            veh_seq = frame["vehicle_sequence"]
            infra_seq = frame["infrastructure_sequence"]
            assert (
                veh_seq == infra_seq
            ), "The vehicle and infrastructure are not in the same scenario."
            if veh_seq in split:
                if not veh_seq in scenario_folders:
                    scenario_folders.update({veh_seq: {"0": {}, "1": {}}})

                assert self.ego_agent in ["vehicle", "infra"], "please set ego vehicle"

                if self.ego_agent == "vehicle":
                    scenario_folders[veh_seq]["0"].update(
                        {
                            "ego": True,
                            "agent": "vehicle",
                            "data_path": self.root_dir + "vehicle-side/",
                        }
                    )
                    scenario_folders[veh_seq]["0"].update(
                        {
                            frame["vehicle_frame"]: {
                                "system_error_offset": frame["system_error_offset"]
                            }
                        }
                    )

                    scenario_folders[veh_seq]["1"].update(
                        {
                            "ego": False,
                            "agent": "infrastructure",
                            "data_path": self.root_dir + "infrastructure-side/",
                        }
                    )
                    scenario_folders[veh_seq]["1"].update(
                        {
                            frame["infrastructure_frame"]: {
                                "system_error_offset": frame["system_error_offset"]
                            }
                        }
                    )
                elif self.ego_agent == "infra":
                    scenario_folders[veh_seq]["0"].update(
                        {
                            "ego": True,
                            "agent": "infrastructure",
                            "data_path": self.root_dir + "infrastructure-side/",
                        }
                    )
                    scenario_folders[veh_seq]["0"].update(
                        {
                            frame["infrastructure_frame"]: {
                                "system_error_offset": frame["system_error_offset"]
                            }
                        }
                    )

                    scenario_folders[veh_seq]["1"].update(
                        {
                            "ego": False,
                            "agent": "vehicle",
                            "data_path": self.root_dir + "vehicle-side/",
                        }
                    )
                    scenario_folders[veh_seq]["1"].update(
                        {
                            frame["vehicle_frame"]: {
                                "system_error_offset": frame["system_error_offset"]
                            }
                        }
                    )
        return scenario_folders

    def get_data_structure(self, cfg, agent="vehicle"):
        root_path = cfg["root_dir"]
        path = root_path + agent + "-side/"
        data_info_path = path + "data_info.json"
        with open(data_info_path) as file:
            data_info = json.load(file)
        data = OrderedDict()

        for i_, i in enumerate(data_info):
            seq = i["sequence_id"]
            frame = i["image_path"][-10:-4]
            if not seq in data:
                data[seq] = OrderedDict()
            if agent == "vehicle":
                data[seq][frame] = {
                    "seq": seq,
                    "frame_id": frame,
                    "lidar": path + i["pointcloud_path"],
                    "camera": path + i["image_path"],
                    "calib_lidar_to_novatel": path + i["calib_lidar_to_novatel_path"],
                    "calib_novatel_to_world": path + i["calib_novatel_to_world_path"],
                    "calib_camera_intrinsic_path": path
                    + i["calib_camera_intrinsic_path"],
                    "calib_lidar_to_camera_path": path
                    + i["calib_lidar_to_camera_path"],
                }
            else:
                data[seq][frame] = {
                    "seq": seq,
                    "frame_id": frame,
                    "lidar": path + i["pointcloud_path"],
                    "camera": path + i["image_path"],
                    "calib_lidar_to_world": path
                    + i["calib_virtuallidar_to_world_path"],
                    "intersection": i["intersection_loc"],
                    "calib_camera_intrinsic_path": path
                    + i["calib_camera_intrinsic_path"],
                    "calib_virtuallidar_to_camera_path": path
                    + i["calib_virtuallidar_to_camera_path"],
                }
        return data

    def get_lidar_pose_44(self, cav_content_, timestamp_cur):
        ego = cav_content_["ego"]
        agent = cav_content_["agent"]
        cav_content = cav_content_[timestamp_cur]
        time = cav_content["timestamp"]
        sequence = cav_content["sequence"]
        his_frame = self.params["dataset"]["frame_his"]
        if not ego:
            his_frame = self.params["dataset"]["cav_frame_his"]
        pose, lidar_path = [], []

        for t in range(self.max_delay):
            time_ = str(int(time) - t).zfill(len(time))
            if agent == "vehicle":
                if time_ in self.vehicle_infos[sequence]:
                    info = self.vehicle_infos[sequence][time_]
                    lidar_to_novatel = info["calib_lidar_to_novatel"]
                    novatel_to_world = info["calib_novatel_to_world"]
                    with open(lidar_to_novatel) as f:
                        lidar_to_novatel = json.load(f)["transform"]
                    T1 = np.eye(4)
                    T1[:-1, :-1] = lidar_to_novatel["rotation"]
                    T1[:-1, -1] = [i[0] for i in lidar_to_novatel["translation"]]
                    with open(novatel_to_world) as f:
                        novatel_to_world = json.load(f)
                    T2 = np.eye(4)
                    T2[:-1, :-1] = novatel_to_world["rotation"]
                    T2[:-1, -1] = [i[0] for i in novatel_to_world["translation"]]
                    lidar_pose = np.dot(T2, T1)
                else:
                    info = self.vehicle_infos[sequence][time]

            else:
                if time_ in self.infra_infos[sequence]:
                    info = self.infra_infos[sequence][time_]
                    lidar_to_world = info["calib_lidar_to_world"]
                    with open(lidar_to_world) as f:
                        lidar_to_world = json.load(f)
                    lidar_pose = np.eye(4)
                    lidar_pose[:-1, :-1] = lidar_to_world["rotation"]
                    lidar_pose[:-1, -1] = [i[0] for i in lidar_to_world["translation"]]
                else:
                    info = self.infra_infos[sequence][time]
            pose.append(lidar_pose)
            lidar_path.append(info["lidar"])

        if len(lidar_path) < his_frame:
            for i in range(his_frame - len(lidar_path)):
                lidar_path.append(lidar_path[-1])
                pose.append(pose[-1])
        else:
            lidar_path = lidar_path[:his_frame]

        return lidar_path, pose, pose[0]

    def get_ego_content(self, scenario_database):

        # Find ego pose first
        for cav_id, cav_content in scenario_database.items():
            if cav_content["ego"]:
                ego_cav_content = cav_content
                break
        return ego_cav_content

    def update_object_param(self, cfg, side="veh"):
        x = {
            "coop": self.params["dataset"]["label"],
            "veh": self.params["dataset"]["veh_label"],
            "infra": self.params["dataset"]["infra_label"],
        }
        label_dir = self.params["root_dir"] + x[side]
        fs = os.listdir(label_dir)
        ignore_cls = cfg["dataset"]["ignore_cls"]
        data = OrderedDict()
        for f in fs:
            with open(label_dir + f) as f_:
                data_ = json.load(f_)
                remain_ = []
                for object in data_:
                    if not object["type"] in ignore_cls:
                        remain_.append(object)
                data[f[:-5]] = remain_
        return data
