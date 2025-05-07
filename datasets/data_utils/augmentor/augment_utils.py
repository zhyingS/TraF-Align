# Modified from PillarNext (https://github.com/qcraftai/pillarnext)

import numpy as np
import datasets.data_utils.augmentor.box_np_ops as box_np_ops

# points [x,y,z,t]
class Flip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob #[0.5,0.5]
        assert 0 <= flip_prob[0] < 1
        assert 0 <= flip_prob[1] < 1

    def __call__(self, points, boxes):
        # x flip
        if self.flip_prob[0] > 0:
            enable = np.random.choice([False, True], replace=False,
                                      p=[1 - self.flip_prob[0], self.flip_prob[0]])
            if enable:
                for i,point in enumerate(points):
                    if point.shape[0]>0 and point.shape[1]>0:
                        points[i][:, 1] = -point[:, 1]
                for i,box in enumerate(boxes):
                    if box.shape[0]>0 and box.shape[1]>0:
                        mask = np.isnan(box)
                        box[mask] = 0
                        box = box_np_ops.flip(
                            box, axis='x')
                        box[mask] = np.nan
                        boxes[i] = box
        # y flip
        if self.flip_prob[1] > 0:
            enable = np.random.choice(
                [False, True], replace=False, p=[1 - self.flip_prob[1], self.flip_prob[1]])
            if enable:
                for i,point in enumerate(points):
                    if point.shape[0]>0 and point.shape[1]>0:
                        points[i][:, 0] = -point[:, 0]
                for i,box in enumerate(boxes):
                    if box.shape[0]>0 and box.shape[1]>0:
                        mask = np.isnan(box)
                        box[mask] = 0
                        box = box_np_ops.flip(
                            box, axis='y')
                        box[mask] = np.nan
                        boxes[i] = box
        return points, boxes


class Scaling(object):
    def __init__(self, scale):
        self.min_scale, self.max_scale = scale

    def __call__(self, points, boxes):
       
        noise_scale = np.random.uniform(self.min_scale, self.max_scale)
        for i, point in enumerate(points):
            if point.shape[0] > 0:
                point[:,:3] = point[:, :3] * noise_scale
                points[i] = point
        for i,box in enumerate(boxes):
            if box.shape[0]>0 and box.shape[1]>0:
                mask = np.isnan(box)
                box[mask] = 0
                box = box_np_ops.scaling(
                    box, noise_scale)
                box[mask] = np.nan
                boxes[i] = box
        return points, boxes


class Rotation(object):
    def __init__(self, rotation):
        self.rotation = rotation

    def __call__(self, points, boxes):
        noise_rotation = np.random.uniform(self.rotation[0], self.rotation[1])
        for i,point in enumerate(points):
            if point.shape[0]>0 and point.shape[1]>0:
                point[:, :3] = box_np_ops.yaw_rotation(
                    point[:, :3], noise_rotation)
                points[i] = point
        for i,box in enumerate(boxes):
            if box.shape[0]>0 and box.shape[1]>0:
                mask = np.isnan(box)
                box[mask] = 0
                box = box_np_ops.rotate(
                    box, noise_rotation)
                box[mask] = np.nan
                boxes[i] = box
            
        return points, boxes


class Translation(object):
    def __init__(self, noise):
        self.noise = noise

    def __call__(self, points, boxes):
        noise_translate = np.random.normal(0, self.noise, 1)
        for i,point in enumerate(points):
            if point.shape[0]>0 and point.shape[1]>0:
                point[:, :3] += noise_translate
                points[i] = point
        for i,box in enumerate(boxes):
            if box.shape[0]>0 and box.shape[1]>0:
                mask = np.isnan(box)
                box[mask] = 0
                box = box_np_ops.translate(
                    box, noise_translate)
                box[mask] = np.nan
                boxes[i] = box
            
        return points, boxes
