# Modified from OpenCOOD (https://github.com/DerrickXuNu/OpenCOOD)
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import re
import yaml

import os
import numpy as np


def load_yaml(file, cfg=None):
    """
    Load yaml file and return a dictionary.

    Parameters
    ----------
    file : string
        yaml file path.

    cfg : argparser
         Argparser.
    Returns
    -------
    cfg : dict
        A dictionary that contains defined parameters.
    """
    if cfg and cfg.model_dir:
        file = os.path.join(cfg.model_dir, 'config.yaml')

    stream = open(file, 'r')
    loader = yaml.Loader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    cfg = yaml.load(stream, Loader=loader)

    return cfg

def check_pillar_params(cfg):
    """
    Based on the lidar range and resolution of voxel, calcuate the resolution.

    Parameters
    ----------
    cfg : dict
        Original loaded parameter dictionary.

    Returns
    -------
    cfg : dict
        Modified parameter dictionary with new attribute.
    """
    args = cfg['voxelization']
    lidar_range = np.array(args['lidar_range'])
    voxel_size = np.array(args['voxel_size'])
    grid_size_ = np.array(args['grid_size'])

    grid_size = (lidar_range[3:6] - 
        lidar_range[0:3]) / voxel_size
    grid_size[-1] = 1
    grid_size = np.round(grid_size).astype(np.int64)
    grid_size = grid_size[[1,0,2]]
    # assert np.array_equal(grid_size, grid_size_), 'Grid size not matched, grid size is not equal to lidar_range/voxel_size.'

    cfg['voxelization']['lidar_range'] = lidar_range
    cfg['voxelization']['voxel_size'] = voxel_size
    cfg['voxelization']['grid_size'] = grid_size

    return cfg


def save_yaml(data, save_name):
    """
    Save the dictionary into a yaml file.

    Parameters
    ----------
    data : dict
        The dictionary contains all data.

    save_name : string
        Full path of the output yaml file.
    """

    with open(save_name, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)
