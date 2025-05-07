import numpy as np
from cumm import tensorview as tv
import torch
import sys
import json
import math

    
def collate_batch_label(label_list):
    a = torch.tensor([])
    for i, j in enumerate(label_list):
        b = np.zeros((j.shape[0], 1)) + i
        j = np.hstack((b, j))
        a = torch.cat((a, torch.from_numpy(j).to(torch.float32)), dim=0)

    return a

def pre_collate_batch(batch):
    """
    Customized pytorch data loader collate function.

    Parameters
    ----------
    batch : list or dict
        List or dictionary.

    Returns
    -------
    processed_batch : dict
        Updated lidar batch.
    """

    if isinstance(batch, list):
        return collate_batch_list(batch)
    elif isinstance(batch, dict):
        return collate_batch_dict(batch)
    else:
        sys.exit('Batch has too be a list or a dictionarn')


def collate_batch_list(batch):
    """
    Customized pytorch data loader collate function.

    Parameters
    ----------
    batch : list
        List of dictionary. Each dictionary represent a single frame.

    Returns
    -------
    processed_batch : dict
        Updated lidar batch.
    """
    voxel_features = []
    voxel_num_points = []
    voxel_coords = []

    for i in range(len(batch)):
        voxel_features.append(batch[i]['voxel_features'])
        voxel_num_points.append(batch[i]['voxel_num_points'])
        coords = batch[i]['voxel_coords']

        voxel_coords.append(
            np.pad(coords, ((0, 0), (1, 0)),
                   mode='constant', constant_values=i))

    voxel_num_points = torch.from_numpy(np.concatenate(voxel_num_points))
    voxel_features = torch.from_numpy(np.concatenate(voxel_features))
    voxel_coords = torch.from_numpy(np.concatenate(voxel_coords))

    return {
            'voxel_features': voxel_features,
            'voxel_coords': voxel_coords,
            'voxel_num_points': voxel_num_points}


def collate_batch_dict(batch: dict):
    """
    Collate batch if the batch is a dictionary,
    eg: {'voxel_features': [feature1, feature2...., feature n]}

    Parameters
    ----------
    batch : dict

    Returns
    -------
    processed_batch : dict
        Updated lidar batch.
    """
    voxel_features = \
        torch.from_numpy(np.concatenate(batch['voxel_features']))
    voxel_num_points = \
        torch.from_numpy(np.concatenate(batch['voxel_num_points']))
    coords = batch['voxel_coords']
    voxel_coords = []

    for i in range(len(coords)):
        voxel_coords.append(
                np.pad(coords[i], ((0, 0), (1, 0)),
                    mode='constant', constant_values=i))
    voxel_coords = torch.from_numpy(np.concatenate(voxel_coords))

    return {'voxel_features': voxel_features,
            'voxel_coords': voxel_coords,
            'voxel_num_points': voxel_num_points}

def collate_batch_label(label_list):
    a = torch.tensor([])
    for i, j in enumerate(label_list):
        if j.shape[0]==0 or j.shape[1]==0:
            continue
        b = np.zeros((j.shape[0], 1)) + i
        j = np.hstack((b, j))
        a = torch.cat((a, torch.from_numpy(j).to(torch.float32)), dim=0)
        
    return a
    
def preprocess(params, pcd_np, extra_feature = 2,only_coords=False,train=False):
    # if pcd_np.shape[0] == 0:
    #     pcd_np = np.zeros((3,3+extra_feature)) #- 1e4
        
    spconv = 1
    try:
        # spconv v1.x
        from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
    except:
        # spconv v2.x
        from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
        spconv = 2
    lidar_range = params['lidar_range']
    voxel_size = params['voxel_size']
    max_points_per_voxel = params['max_points_per_voxel']

    try:
        max_voxels = params['max_voxel']
    except:
        max_voxels = params['max_voxel_train'] if train else params['max_voxel_test']

    grid_size = (np.array(lidar_range[3:6]) -
                 np.array(lidar_range[0:3])) / np.array(voxel_size)
    grid_size = np.round(grid_size).astype(np.int64)

    # use sparse conv library to generate voxel
    if spconv == 1:
        voxel_generator = VoxelGenerator(
            voxel_size=voxel_size,
            point_cloud_range=lidar_range,
            max_num_points=max_points_per_voxel,
            max_voxels=max_voxels
        )
    else:
        voxel_generator = VoxelGenerator(
            vsize_xyz=voxel_size,
            coors_range_xyz=lidar_range,
            max_num_points_per_voxel=max_points_per_voxel,
            num_point_features=3+extra_feature,
            max_num_voxels=max_voxels
        )

    data_dict = {}
    if spconv == 1:
        voxel_output = voxel_generator.generate(pcd_np)
    else:
        pcd_np = pcd_np.astype(np.float)
        pcd_tv = tv.from_numpy(pcd_np)
        voxel_output = voxel_generator.point_to_voxel(pcd_tv)
    if isinstance(voxel_output, dict):
        voxels, coordinates, num_points = \
            voxel_output['voxels'], voxel_output['coordinates'], \
            voxel_output['num_points_per_voxel']
    else:
        voxels, coordinates, num_points = voxel_output

    if spconv == 2:
        voxels = voxels.numpy()
        coordinates = coordinates.numpy()
        num_points = num_points.numpy()
    data_dict['voxel_coords'] = coordinates

    if not only_coords:
        data_dict['voxel_features'] = voxels
        data_dict['voxel_num_points'] = num_points

    return data_dict

def read_json(path):
    with open(path, "r") as f:
        my_json = json.load(f)
        return my_json

def get_lidar2camera(path_lidar2camera):
    lidar2camera = read_json(path_lidar2camera)
    rotation = lidar2camera['rotation']
    translation = lidar2camera['translation']
    rotation = np.array(rotation).reshape(3, 3)
    translation = np.array(translation).reshape(3, 1)
    return rotation, translation


def trans_point(input_point, rotation, translation=None):
    if translation is None:
        translation = [0.0, 0.0, 0.0]
    input_point = np.array(input_point).reshape(3, 1)
    translation = np.array(translation).reshape(3, 1)
    rotation = np.array(rotation).reshape(3, 3)
    output_point = np.dot(rotation, input_point).reshape(3, 1) + np.array(translation).reshape(3, 1)
    output_point = output_point.reshape(1, 3).tolist()
    return output_point[0]


def get_lidar_3d_8points(label_3d_dimensions, lidar_3d_location, rotation_z):
    lidar_rotation = np.matrix(
        [
            [math.cos(rotation_z), -math.sin(rotation_z), 0],
            [math.sin(rotation_z), math.cos(rotation_z), 0],
            [0, 0, 1]
        ]
    )
    l, w, h = label_3d_dimensions
    corners_3d_lidar = np.matrix(
        [
            [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2],
        ]
    )
    lidar_3d_8points = lidar_rotation * corners_3d_lidar + np.matrix(lidar_3d_location).T
    return lidar_3d_8points.T.tolist()


def get_camera_3d_alpha_rotation(camera_3d_8_points, camera_3d_location):
    x0, z0 = camera_3d_8_points[0][0], camera_3d_8_points[0][2]
    x3, z3 = camera_3d_8_points[3][0], camera_3d_8_points[3][2]
    dx, dz = x0 - x3, z0 - z3
    rotation_y = -math.atan2(dz, dx)  # 相机坐标系xyz下的偏航角yaw绕y轴与x轴夹角，方向符合右手规则，所以用(-dz,dx)
    # alpha = rotation_y - math.atan2(center_in_cam[0], center_in_cam[2])
    alpha = rotation_y - (-math.atan2(-camera_3d_location[2], -camera_3d_location[0])) + math.pi / 2  # yzw
    # add transfer
    if alpha > math.pi:
        alpha = alpha - 2.0 * math.pi
    if alpha <= (-1 * math.pi):
        alpha = alpha + 2.0 * math.pi
    return alpha, rotation_y

