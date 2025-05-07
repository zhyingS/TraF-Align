# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

from datasets.data_utils.post_processor.voxel_postprocessor import VoxelPostprocessor


__all__ = {
    'VoxelPostprocessor': VoxelPostprocessor
    }


def build_postprocessor(anchor_cfg):
    process_method_name = anchor_cfg['core_method']
    assert process_method_name in ['VoxelPostprocessor']
    anchor_generator = __all__[process_method_name](
        anchor_params=anchor_cfg
    )

    return anchor_generator
