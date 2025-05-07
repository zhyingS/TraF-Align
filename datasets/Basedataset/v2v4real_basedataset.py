# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
Basedataset class for all kinds of fusion.
"""

import os
import math
from collections import OrderedDict
from random import randint
import time
import random

import torch
import numpy as np
from torch.utils.data import Dataset
from functools import reduce
import scipy
import tqdm
from scipy.spatial.transform import Rotation

import utils.pcd_utils as pcd_utils
from datasets.data_utils.augmentor.data_augmentor import DataAugmentor
from hypes_yaml.yaml_utils import load_yaml
from utils.pcd_utils import downsample_lidar_minimum
from utils.transformation_utils import (
    x1_to_x2,
    dist_two_pose,
    x1_to_x2_matrix,
    x_to_world,
)
from utils.pcd_utils import (
    mask_points_by_range,
    mask_ego_points,
    shuffle_points,
    downsample_lidar_minimum,
)
from utils import box_utils
from utils.icp import IterativeClosestPoint as ICP


class V2V4RealBaseDataset(Dataset):
    """
    Base dataset for all kinds of fusion. Mainly used to initialize the
    database and associate the __get_item__ index with the correct timestamp
    and scenario.

    Parameters
    __________
    params : dict
        The dictionary contains all parameters for training/testing.

    visualize : false
        If set to true, the raw point cloud will be saved in the memory
        for visualization.

    Attributes
    ----------
    scenario_database : OrderedDict
        A structured dictionary contains all file information.

    len_record : list
        The list to record each scenario's data length. This is used to
        retrieve the correct index during training.

    pre_processor : opencood.pre_processor
        Used to preprocess the raw data.

    post_processor : opencood.post_processor
        Used to generate training labels and convert the model outputs to
        bbx formats.

    data_augmentor : opencood.data_augmentor
        Used to augment data.

    """

    def __init__(self, params, set, single_agent=False):
        self.params = params
        self.set = set
        self.single_agent = single_agent
        self.pre_processor = None
        self.post_processor = None

        if set == "train":
            self.train = True
        else:
            self.train = False

        # if the training/testing include noisy setting
        if "wild_setting" in params:
            wild_parms = params["wild_setting"]
            self.seed = wild_parms["seed"]
            # whether to add time delay
            self.async_flag = wild_parms["async"]
            self.async_mode = "sim"
            self.agent_i_delay = wild_parms["agent_i_delay"]
            self.async_ego = wild_parms["async_ego"]
            # localization error
            self.loc_calib = wild_parms["loc_calib"]

            self.loc_err = wild_parms["loc_err"]
            if self.loc_err:
                raise ("Not support adding loc noise yet")
            self.xyz_noise_std = wild_parms["xyz_std"]
            self.ryp_noise_std = wild_parms["ryp_std"]

            self.ego_agent = wild_parms["ego"]
            if self.ego_agent != "vehicle":
                raise ("Ego agent must be vehicle.")
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

        if self.set == "train":
            root_dir = params["root_dir"]
        else:
            root_dir = params["validate_dir"]
        self.root_dir = root_dir

        if params["dataset"]["correct_pitch_and_roll"]:
            self.correct = "_corrected"
        else:
            self.correct = ""

        self.lidar_range = params["voxelization"]["lidar_range"]
        self.voxel_size = params["voxelization"]["voxel_size"]
        self.grid_size = params["voxelization"]["grid_size"]

        if "train_params" not in params or "max_cav" not in params["train_params"]:
            self.max_cav = 7
        else:
            self.max_cav = params["train_params"]["max_cav"]

        # first load all paths of different scenarios
        self.scenario_folders = sorted(
            [
                os.path.join(root_dir, x)
                for x in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, x))
            ]
        )

        try:
            self.COM_RANGE = params["dataset"]["COM_RANGE"]
        except:
            self.COM_RANGE = 1e4

        self.reinitialize()

    def __len__(self):
        return self.len_record[-1]

    def __getitem__(self, idx):
        """
        Abstract method, needs to be define by the children class.
        """
        pass

    def reinitialize(self):
        # Structure: {scenario_id : {cav_1 : {timestamp1 : {yaml: path,
        # lidar: path, cameras:list of path}}}}
        self.scenario_database = OrderedDict()
        self.len_record = []
        if os.path.exists(self.root_dir + f"/scenario_database{self.correct}.npy"):
            self.scenario_database = np.load(
                self.root_dir + f"/scenario_database{self.correct}.npy",
                allow_pickle=True,
            ).item()
            for i, scenario_folder in enumerate(self.scenario_folders):
                # at least 1 cav should show up
                cav_list = sorted(
                    [
                        x
                        for x in os.listdir(scenario_folder)
                        if os.path.isdir(os.path.join(scenario_folder, x))
                    ]
                )
                assert len(cav_list) > 0

                # roadside unit data's id is always negative, so here we want to
                # make sure they will be in the end of the list as they shouldn't
                # be ego vehicle.
                if int(cav_list[0]) < 0:
                    cav_list = cav_list[1:] + [cav_list[0]]

                # loop over all CAV data
                for j, cav_id in enumerate(cav_list):
                    if j > self.max_cav - 1:
                        print("too many cavs")
                        break
                    cav_path = os.path.join(scenario_folder, cav_id)
                    yaml_files = sorted(
                        [
                            os.path.join(cav_path, x)
                            for x in os.listdir(cav_path)
                            if x.endswith(".yaml") and "additional" not in x
                        ]
                    )
                    timestamps = self.extract_timestamps(yaml_files)
                    if j == 0:
                        # we regard the agent with the minimum id as the ego
                        if not self.len_record:
                            self.len_record.append(len(timestamps))
                        else:
                            prev_last = self.len_record[-1]
                            self.len_record.append(prev_last + len(timestamps))
        else:
            print("shuffling training scenario database and generating new data")
            pbar = tqdm.tqdm(total=len(self.scenario_folders), leave=True)
            # loop over all scenarios
            for i, scenario_folder in enumerate(self.scenario_folders):
                self.scenario_database.update({i: OrderedDict()})

                # # at least 1 cav should show up
                # if self.train:
                #     cav_list = sorted([x for x in os.listdir(scenario_folder)
                #                 if os.path.isdir(
                #             os.path.join(scenario_folder, x))])
                #     random.shuffle(cav_list)
                # else:
                cav_list = sorted(
                    [
                        x
                        for x in os.listdir(scenario_folder)
                        if os.path.isdir(os.path.join(scenario_folder, x))
                    ]
                )
                assert len(cav_list) > 0

                # roadside unit data's id is always negative, so here we want to
                # make sure they will be in the end of the list as they shouldn't
                # be ego vehicle.
                if int(cav_list[0]) < 0:
                    cav_list = cav_list[1:] + [cav_list[0]]

                # loop over all CAV data
                for j, cav_id in enumerate(cav_list):
                    if j > self.max_cav - 1:
                        print("too many cavs")
                        break
                    self.scenario_database[i][j] = OrderedDict()

                    # save all yaml files to the dictionary
                    cav_path = os.path.join(scenario_folder, cav_id)

                    # use the frame number as key, the full path as the values
                    yaml_files = sorted(
                        [
                            os.path.join(cav_path, x)
                            for x in os.listdir(cav_path)
                            if x.endswith(".yaml") and "additional" not in x
                        ]
                    )
                    timestamps = self.extract_timestamps(yaml_files)

                    for timestamp in timestamps:
                        self.scenario_database[i][j][timestamp] = OrderedDict()

                        yaml_file = os.path.join(cav_path, timestamp + ".yaml")
                        yaml_file_ = load_yaml(yaml_file)
                        lidar_file = os.path.join(cav_path, timestamp + ".pcd")

                        if j != 0 and self.correct != "":
                            yaml_file_ = self.correct_pitch_roll(
                                yaml_file_,
                                timestamp,
                                ego_yamls=self.scenario_database[i][0],
                            )

                        self.scenario_database[i][j][timestamp]["yaml"] = yaml_file_
                        self.scenario_database[i][j][timestamp]["lidar"] = lidar_file

                    # Assume all cavs will have the same timestamps length. Thus
                    # we only need to calculate for the first vehicle in the
                    # scene.
                    if j == 0:
                        self.scenario_database[i][j]["ego"] = True
                        if not self.len_record:
                            self.len_record.append(len(timestamps))
                        else:
                            prev_last = self.len_record[-1]
                            self.len_record.append(prev_last + len(timestamps))
                    else:
                        self.scenario_database[i][j]["ego"] = False
                pbar.update(1)
            np.save(
                self.root_dir + f"/scenario_database{self.correct}.npy",
                self.scenario_database,
            )

        self.coop_label_dict, self.coop_label_dict_ego = self.generate_coop_label()

    def shuffle_each_epoch(self):
        for i in range(len(self.scenario_database)):
            scenario = self.scenario_database[i]
            random.shuffle(scenario)
            scenario[0]["ego"] = True
            scenario[1]["ego"] = False
            self.scenario_database[i] = scenario

        self.coop_label_dict, self.coop_label_dict_ego = self.generate_coop_label()

    def generate_coop_label(self):

        a = self.scenario_database
        label_dict = OrderedDict()
        label_dict_ego = OrderedDict()

        len_ = 0
        for scenario, content in a.items():
            len_ += len(content[0])

        for scenario, content in a.items():
            assert len(content[0]) == len(
                content[1]
            ), "Warning: Ego frames are not equal to cav's."
            label_dict.update({scenario: OrderedDict()})
            label_dict_ego.update({scenario: OrderedDict()})

            for timestamp in content[0]:
                boxes = []
                if timestamp == "ego":
                    continue
                for cav_id, cav_content in content.items():
                    params = cav_content[timestamp]["yaml"]
                    if cav_content["ego"]:
                        boxes_ = self.generate_object_center(cav_id, [params], time=0)
                        ego_lidar_pose = params["lidar_pose"]
                        if boxes_.shape[0] > 0:
                            boxes.append(boxes_)
                            label_dict_ego[scenario][timestamp] = boxes_
                        else:
                            label_dict_ego[scenario][timestamp] = np.array([])

                    else:
                        # if the dataset type is nofusion, only load ego labels while training
                        if self.single_agent and self.train:
                            continue
                        cav_lidar_pose = params["lidar_pose"]
                        distance = dist_two_pose(cav_lidar_pose, ego_lidar_pose)
                        if distance > self.COM_RANGE:
                            continue
                        transformation_matrix = x1_to_x2(cav_lidar_pose, ego_lidar_pose)
                        boxes_ = self.generate_object_center(
                            cav_id, [params], transform=transformation_matrix, time=0
                        )
                        if boxes_.shape[0] > 0:
                            boxes.append(boxes_)
                try:
                    boxes = np.concatenate(boxes)
                    # Shape is (max_num, 12). [x,y,z,h,w,l,phi,type,id,time,truncated,occluded]
                    ids = boxes[:, 8]
                    _, index = np.unique(ids, return_index=True)
                    unique_boxes = boxes[index]
                    label_dict[scenario][timestamp] = unique_boxes
                except:
                    label_dict[scenario][timestamp] = np.array([])

        return label_dict, label_dict_ego

    def retrieve_base_data(self, idx, cur_ego_pose_flag=True):
        """
        Given the index, return the corresponding data.

        Parameters
        ----------
        idx : int
            Index given by dataloader.

        cur_ego_pose_flag : bool
            Indicate whether to use current timestamp ego pose to calculate
            transformation matrix.

        Returns
        -------
        data : dict
            The dictionary contains loaded yaml params and lidar data for
            each cav.
        """
        # we loop the accumulated length list to see get the scenario index
        scenario_index = 0
        for i, ele in enumerate(self.len_record):
            if idx < ele:
                scenario_index = i
                break
        scenario_database = self.scenario_database[scenario_index]

        # check the timestamp index
        timestamp_index = (
            idx if scenario_index == 0 else idx - self.len_record[scenario_index - 1]
        )
        # retrieve the corresponding timestamp key
        timestamp_key = self.return_timestamp_key(scenario_database, timestamp_index)
        # calculate distance to ego for each cav for time delay estimation
        ego_cav_content = self.calc_dist_to_ego(scenario_database, timestamp_key)

        data = OrderedDict()
        max_delay, min_max_delay = 0, 0
        ego_his_frame = self.params["dataset"]["frame_his"]
        cav_his_frame = self.params["dataset"]["cav_frame_his"]  # 1
        for cav_id, cav_content in scenario_database.items():
            cav_id = int(cav_id)
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
            cav_id = int(cav_id)
            if self.single_agent and not cav_content["ego"]:
                continue
            timestamp_delay = data[cav_id]["time_delay"]

            if timestamp_index - timestamp_delay <= 0:
                timestamp_delay = timestamp_index
            timestamp_index_delay = max(0, timestamp_index - timestamp_delay)
            timestamp_key_delay = self.return_timestamp_key(
                scenario_database, timestamp_index_delay
            )
            # load the corresponding data into the dictionary
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

            if cav_content["ego"]:
                ego_cav_id = cav_id
                ego_timestamp_key = timestamp_key
                # for kitti evaluation
                data["token_dict"] = {}

        pose_cur = data[ego_cav_id]["params"]["his_lidar_pos"][0]
        for t in range(max_delay):
            time_ = str(int(ego_timestamp_key) - t).zfill(len(ego_timestamp_key))
            if time_ in self.coop_label_dict[scenario_index]:
                object_12d = self.coop_label_dict[scenario_index][time_]
                try:
                    object_12d[:, 9] = t
                except:
                    pass
                pose_cur_t = data[ego_cav_id]["params"]["his_lidar_pos"][t]
                T = np.linalg.inv(pose_cur).dot(pose_cur_t)
                try:
                    object_12d[:, :7] = box_utils.project_box7d(object_12d[:, :7], T)
                except:
                    pass
                if object_12d.shape[0] >= 1:
                    data[ego_cav_id]["gt_his_label"].append(object_12d)

        data["timestamp"] = ego_timestamp_key
        gt_boxes = self.coop_label_dict[scenario_index][ego_timestamp_key]
        gt_boxes_ego = self.coop_label_dict_ego[scenario_index][ego_timestamp_key]

        data["gt_boxes"] = gt_boxes
        data["gt_boxes_ego"] = gt_boxes_ego

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
        # lidar_np = shuffle_points(lidar_np)
        # remove points that hit itself

        # project the lidar to ego space
        if not selected_cav_base["ego"]:
            for i, lidar in enumerate(lidar_np):
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

        void_lidar = False  # True if lidar_np.shape[0] < 1 else False

        # velocity
        velocity = selected_cav_base["params"]["ego_speed"]
        # normalize veloccity by average speed 30 km/h
        velocity = velocity / 30

        if selected_cav_base["ego"]:
            try:
                object_12d = np.vstack((selected_cav_base["gt_his_label"]))
            except:
                object_12d = np.array([])

        else:
            object_12d = np.array([])

        selected_cav_processed.update(
            {
                "projected_lidar": lidar_np,
                # 'processed_features': processed_lidar,
                "velocity": velocity,
                "object_12d": object_12d,
            }
        )

        return selected_cav_processed, void_lidar

    def iou_coop_VI_labels(self, lidar, object_12d, vehicle_labels):
        # filter road boxes out of range
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
            # b_polygon = list(self.box2polygon(boxes_b))

            # iou_threshold = 0.25
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

    @staticmethod
    def extract_timestamps(yaml_files):
        """
        Given the list of the yaml files, extract the mocked timestamps.

        Parameters
        ----------
        yaml_files : list
            The full path of all yaml files of ego vehicle

        Returns
        -------
        timestamps : list
            The list containing timestamps only.
        """
        timestamps = []

        for file in yaml_files:
            res = file.split("/")[-1]

            timestamp = res.replace(".yaml", "")
            timestamps.append(timestamp)

        return timestamps

    @staticmethod
    def return_timestamp_key(scenario_database, timestamp_index):
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
        # retrieve the correct index
        timestamp_key = list(timestamp_keys.items())[int(timestamp_index)][0]

        return timestamp_key

    def calc_dist_to_ego(self, scenario_database, timestamp_key):
        """
        Calculate the distance to ego for each cav.
        """
        ego_lidar_pose = None
        ego_cav_content = None
        # Find ego pose first
        for cav_id, cav_content in scenario_database.items():
            if cav_content["ego"]:
                ego_cav_content = cav_content
                ego_lidar_pose = cav_content[timestamp_key]["yaml"]["lidar_pose"]
                break

        assert ego_lidar_pose is not None

        # calculate the distance
        for cav_id, cav_content in scenario_database.items():
            cur_lidar_pose = cav_content[timestamp_key]["yaml"]["lidar_pose"]
            distance = dist_two_pose(cur_lidar_pose, ego_lidar_pose)
            cav_content["distance_to_ego"] = distance
            scenario_database.update({cav_id: cav_content})

        return ego_cav_content

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
            sample_strategy = 2
            if sample_strategy == 1:
                # sample strategy 1: rand in [0,100,200,300,400] might cause delay imbalance
                rand_delay = randint(
                    self.agent_i_delay_train_aug[0], self.agent_i_delay_train_aug[1]
                )
                time_delay = rand_delay // interval * interval
            if sample_strategy == 2:
                # sample strategy 2: rand in [0,0,0,0,100,200,300,400], balance between sync and async data
                rand_delay = np.arange(
                    self.agent_i_delay_train_aug[0] // interval,
                    self.agent_i_delay_train_aug[1] // interval + 1,
                )
                rand_delay = np.concatenate(
                    (np.array([0] * (len(rand_delay) - 1)), rand_delay)
                )
                rand_delay_ = np.random.choice(rand_delay)
                time_delay = rand_delay_ * interval

        time_delay = time_delay // (interval)
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
        if not self.train:
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

        correction = np.zeros(
            (3, 1)
        )  # manual correction for pose misalignment, for dair-v2x only
        gt_correction = np.zeros((3, 1))

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

        # delay_params = OrderedDict()
        delay_params = cav_content[timestamp_delay]["yaml"]
        delay_params["transformation_matrix"] = transformation_matrix
        delay_params["gt_transformation_matrix"] = gt_transformation_matrix
        delay_params["spatial_correction_matrix"] = spatial_correction_matrix

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
                cav_id, cav_content, timestamp_cur, ego_pose_list_cur, delay
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
                cav_id, cav_content, timestamp_cur, cav_pose_list_cur, delay
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

    def get_vehicle_label(self, cav_id, cav_content, timestamp_key, pose_list, delay):

        data = []

        ego = cav_content["ego"]
        his_frame = self.params["dataset"]["frame_his"]
        if not ego:
            his_frame = self.params["dataset"]["cav_frame_his"]

        pose_cur = pose_list[0]
        his_frame = int(his_frame + delay)
        for t in range(his_frame):
            time_ = str(int(timestamp_key) - t).zfill(len(timestamp_key))
            if time_ in cav_content:
                params = cav_content[time_]["yaml"]
                pose_cur_t = pose_list[t]
                T = np.linalg.inv(pose_cur).dot(pose_cur_t)
                label_object = self.generate_object_center(cav_id, [params], T, time=t)
                data.append(label_object)
            else:
                data.append(np.array([]))

        return data

    def generate_object_center(self, cav_id, objects, transform=np.eye(4), time=None):
        """
        Retrieve all objects in a format of (n, 12)

        Parameters
        ----------
        objects : list
            List of dictionary, save cavs' object information.

        transform : array
            Transformation from * to ego.

        Returns
        -------
        object_np : np.ndarray
            Shape is (max_num, 12). [x,y,z,h,w,l,phi,type,id,time,truncated,occluded]
        """
        boxes = np.array([[]])

        for i, tmp_object_list in enumerate(objects):

            for id, object in tmp_object_list["vehicles"].items():
                # object:{angle:roll, yaw, pitch [degree],
                #          center: the relative position from bounding box center to the frontal axis of this vehicle,
                #           extent: half (l,w,h),
                #           location: x,y,z of frontal axis center}

                if "ass_id" not in object:
                    ass_id = id
                else:
                    ass_id = object["ass_id"]
                if ass_id == -1:
                    ass_id = id + 100 * int(cav_id)

                if "obj_type" not in object:
                    obj_type = "Car"
                else:
                    obj_type = object["obj_type"]

                # todo: pedestrain is not consdered yet
                # todo: only single class now
                if obj_type == "Pedestrian":
                    continue

                center = object["center"]
                location = object["location"]
                x = location[0] + center[0]
                y = location[1] + center[1]
                z = location[2] + center[2]

                phi = np.radians(object["angle"][1])
                l, w, h = [2 * i for i in object["extent"]]
                try:
                    order = self.params["post_processing"]["order"]
                except:
                    order = "hwl"

                object_id = ass_id
                object_type = obj_type
                if time is None:
                    timestamp = i
                else:
                    timestamp = time
                assert order in [
                    "hwl",
                    "lwh",
                ], "Only order 'hwl' and 'lwh' is supported."

                truncated = -1
                occluded = -1

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

        try:
            boxes[:, :7] = box_utils.project_box7d(boxes[:, :7], transform)
        except:
            pass

        return boxes

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

    def get_lidar_pose_44(self, cav_content, time__):
        ego = cav_content["ego"]
        his_frame = self.params["dataset"]["frame_his"]
        if not ego:
            his_frame = self.params["dataset"]["cav_frame_his"]

        pose, lidar_path = [], []
        for t in range(self.max_delay):
            time_ = str(int(time__) - t).zfill(len(time__))
            if time_ in cav_content:
                params = cav_content[time_]["yaml"]
                lidar_pose = params["lidar_pose"]  # lidar to world trans
                if isinstance(lidar_pose, list):
                    lidar_pose = x_to_world(lidar_pose)
                assert lidar_pose.shape[0] == 4, "the shape of pose should be 4x4."
                assert lidar_pose.shape[1] == 4, "the shape of pose should be 4x4."

                pose.append(lidar_pose)
                lidar_path.append(cav_content[time_]["lidar"])

        if len(lidar_path) < his_frame:
            for i in range(his_frame - len(lidar_path)):
                lidar_path.append(lidar_path[-1])
                pose.append(pose[-1])
        else:
            lidar_path = lidar_path[:his_frame]

        return lidar_path, pose, pose[0]

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

    def augment(
        self,
        lidar_np,
        object_bbx_center,
        object_bbx_mask,
        flip=None,
        rotation=None,
        scale=None,
    ):
        """ """
        tmp_dict = {
            "lidar_np": lidar_np,
            "object_bbx_center": object_bbx_center,
            "object_bbx_mask": object_bbx_mask,
            "flip": flip,
            "noise_rotation": rotation,
            "noise_scale": scale,
        }
        tmp_dict = self.data_augmentor.forward(tmp_dict)

        lidar_np = tmp_dict["lidar_np"]
        object_bbx_center = tmp_dict["object_bbx_center"]
        object_bbx_mask = tmp_dict["object_bbx_mask"]

        return lidar_np, object_bbx_center, object_bbx_mask

    def collate_batch_train(self, batch):
        """
        Customized collate function for pytorch dataloader during training
        for late fusion dataset.

        Parameters
        ----------
        batch : dict

        Returns
        -------
        batch : dict
            Reformatted batch.
        """
        # during training, we only care about ego.
        output_dict = {"ego": {}}

        object_bbx_center = []
        object_bbx_mask = []
        processed_lidar_list = []
        label_dict_list = []

        if self.visualize:
            origin_lidar = []

        for i in range(len(batch)):
            ego_dict = batch[i]["ego"]
            object_bbx_center.append(ego_dict["object_bbx_center"])
            object_bbx_mask.append(ego_dict["object_bbx_mask"])
            processed_lidar_list.append(ego_dict["processed_lidar"])
            label_dict_list.append(ego_dict["label_dict"])

            if self.visualize:
                origin_lidar.append(ego_dict["origin_lidar"])

        # convert to numpy, (B, max_num, 7)
        object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
        object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

        processed_lidar_torch_dict = self.pre_processor.collate_batch(
            processed_lidar_list
        )
        label_torch_dict = self.post_processor.collate_batch(label_dict_list)
        output_dict["ego"].update(
            {
                "object_bbx_center": object_bbx_center,
                "object_bbx_mask": object_bbx_mask,
                "processed_lidar": processed_lidar_torch_dict,
                "label_dict": label_torch_dict,
            }
        )
        if self.visualize:
            origin_lidar = np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
            origin_lidar = torch.from_numpy(origin_lidar)
            output_dict["ego"].update({"origin_lidar": origin_lidar})

        return output_dict

    def visualize_result(
        self, pred_box_tensor, gt_tensor, pcd, show_vis, save_path, dataset=None
    ):
        self.post_processor.visualize(
            pred_box_tensor, gt_tensor, pcd, show_vis, save_path, dataset=dataset
        )

    def correct_pitch_roll(self, yaml, timestamp, ego_yamls):
        ego_yaml = ego_yamls[timestamp]["yaml"]
        cav_pose, ego_pose = yaml["lidar_pose"], ego_yaml["lidar_pose"]
        trans_ego_to_cav = x1_to_x2_matrix(ego_pose, cav_pose)
        ego_box_ = self.generate_object_center(0, [ego_yaml])[:, :7]
        cav_box_ = self.generate_object_center(1, [yaml])[:, :7]

        if ego_box_.shape[1] > 0 and cav_box_.shape[1] > 0:
            ego_box = box_utils.boxes_to_corners_3d(ego_box_, order="hwl")
            cav_box = box_utils.boxes_to_corners_3d(cav_box_, order="hwl")
            ego_ground_ = ego_box[:, :4, :].reshape(-1, 3)
            cav_ground = cav_box[:, :4, :].reshape(-1, 3)
            ego_ground = self.transform_R_t(ego_ground_, trans_ego_to_cav)
            T_pr = ICP(ego_ground.T, cav_ground.T)
            euler = Rotation.from_matrix(T_pr[:-1, :-1]).as_euler("xyz", degrees=True)
            euler[-1] = 0
            correct = True

            R = Rotation.from_euler("xyz", euler, degrees=True).as_matrix()
            if abs(euler[0]) > 90:
                Rx = Rotation.from_euler("xyz", [180, 0, 0], degrees=True).as_matrix()
                R = np.dot(Rx, R)
            if abs(euler[1]) > 90:
                Ry = Rotation.from_euler("xyz", [0, 180, 0], degrees=True).as_matrix()
                R = np.dot(Ry, R)
            if abs(euler[0]) > 10 or abs(euler[1]) > 10:
                correct = False
            u1 = np.mean(ego_ground.T, axis=1).reshape((3, 1))
            u2 = np.mean(cav_ground.T, axis=1).reshape((3, 1))
            t = u2 - np.dot(R, u1)

            T_pr[:-1, :-1] = R
            T_pr[:-1, -1] = t.reshape(
                -1,
            )
            T_pr[:-2, -1] = 0

            if correct:
                yaml["lidar_pose"] = np.dot(cav_pose, np.linalg.inv(T_pr))

        return yaml

    def transform_R_t(self, points, T):
        # points: N x 3
        points = np.dot(T[:-1, :-1], points.T) + T[:-1, -1].reshape(-1, 1)

        return points.T
