# type: ignore
import copy
import json
import logging
import math
import os
from collections import namedtuple
from typing import Any, Dict, List, NamedTuple, Optional, Type, Union

import cv2
import habitat
import magnum as mn
import numpy as np
import quaternion
import torch
from detectron2.config import get_cfg
from detectron2.structures import Boxes, Instances
from gym import spaces
from habitat.config import Config
from habitat.core.registry import registry
from habitat.core.simulator import AgentState, Sensor, SensorTypes, Simulator
from habitat.core.utils import try_cv2_import
from habitat.tasks.nav.nav import PointGoalSensor
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import (
    angle_between_quaternions,
    quaternion_from_coeff,
    quaternion_from_two_vectors,
    quaternion_rotate_vector,
)
from habitat.utils.visualizations import fog_of_war, maps
from habitat_sim.utils import common as sim_utils
from quaternion import numpy_quaternion

from experimenting_env.detector.model.multi_stage_models import (
    FocalMultiStageModel,
    MultiStageModel,
)
from experimenting_env.sensor_data import AgentPoseSense, BBSense
from experimenting_env.utils import predictor_utils
from experimenting_env.utils.predictor_utils import Predictor, setup_cfg
from experimenting_env.utils.roi_head_wrappers import (
    BoxPredictorWrapper,
    SoftHeadWrapper,
)

log = logging.getLogger(__name__)


@registry.register_sensor(name="object_detector_features")
class ObjectDetectorWithFeatures(habitat.Sensor):
    def __init__(self, sim, config, **kwargs: Any):
        super().__init__(config=config)
        self._sim = sim
        self.object_detector = MultiStageModel.load_from_checkpoint(config.LOAD_CKPT)
        self.object_detector.eval()

    # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "bbsf"

    # Defines the type of the sensor
    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return habitat.SensorTypes.MEASUREMENT

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(10, 1 + 4 + 128),
            dtype=np.float32,
        )

    # This is called whenever reset is called or an action is taken
    def get_observation(self, *args: Any, **kwargs: Any):
        rgb = kwargs['observations']['rgb']

        (instances, box_features, predictions_images) = self.object_detector.infer(rgb)
        feats = box_features.detach().numpy()

        # dim: (10 instances max, class + box + feature)
        results = np.zeros((10, 1 + 4 + 128), dtype=np.float32)
        results[:, 0] = -1
        for n in range(min(10, len(instances[0]['instances'].pred_classes))):
            cl = instances[0]['instances'].pred_classes[n]
            # sc = instances[0]['instances'].scores[n]
            bx = instances[0]['instances'].pred_boxes[n][0].tensor.numpy()[0]
            ft = feats[n]

            results[n, 0] = cl
            results[n, 1:5] = bx
            results[n, 5:] = ft

        return results


@registry.register_sensor(name="object_detector_gt")
class ObjectDetectorGT(habitat.Sensor):
    MATTERPORT_SIM_TO_COCO_MAPPING = {
        5:60,
        3: 56,  # chair
        10: 57,  # couch
        14: 58,  # plan
        11: 59,  # beed
        18: 61,  # toilet
        22: 62,  # tv

    }

    SIM_TO_COCO_MAPPING = {
        #"chair": 56,  # chair
        "couch": 57,  # couch
        "potted plant": 58,  # plan
        "bed": 59,  # bed
        "toilet": 61,  # toilet
        "tv": 62,  # tv
        "dining table": 60,  # dining table
    }

    DEPTH_THR = 5

    def __init__(self, sim, config, **kwargs: Any):
        super().__init__(config=config)
        self._sim = sim

        self._objects = self._sim.semantic_annotations().objects

        self.scene = ""

        # filter occluded objects in mp3d by checking avg depth inside instance mask wrt gt
        self.filter_occluded_instances = True

    # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "bbsgt"

    # Defines the type of the sensor
    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return habitat.SensorTypes.MEASUREMENT

    # Defines the size and range of the observations of the sensor
    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(3,),
            dtype=np.float32,
        )

    def _convert_matterport_to_coco_labels(self, label):
        switch = {
            'table': 'dining table',
            'plant': 'potted plant',
            'sofa': 'couch',
            'tv_monitor': 'tv',
        }
        if label in switch.keys():
            return switch[label]
        else:
            return label

    # This is called whenver reset is called or an action is taken
    def get_observation(self, *args: Any, **kwargs: Any):
        sense = kwargs['observations']['semantic']

        current_scene = self._sim.habitat_config.SCENE
        if current_scene != self.scene:

            self._objects = self._sim.semantic_annotations().objects
            self.mapping = {
                int(obj.id.split("_")[-1]): obj.category.name()
                for obj in self._sim.semantic_annotations().objects
                if obj is not None
            }
            self.scene = current_scene
            log.info("Updating mapping")

        bounding_boxes = []
        classes = []
        pred_masks = []
        infos = []
        mask_cleaned = sense.astype('uint8')

        objects_id = np.unique(mask_cleaned)
        for id_object in objects_id:

            bb = cv2.boundingRect((mask_cleaned == id_object).astype('uint8'))
            x, y, w, h = bb
            if id_object not in self.mapping:
                continue

            if (mask_cleaned == id_object).sum() < 1000:
                continue
            habitat_id = self._convert_matterport_to_coco_labels(
                self.mapping[id_object]
            )

            if habitat_id in self.SIM_TO_COCO_MAPPING:
                pred_mask = torch.zeros(sense.shape, dtype=torch.bool)
                pred_mask[(mask_cleaned == id_object).astype('bool')] = True

                coco_id = self.SIM_TO_COCO_MAPPING[habitat_id]
                bounding_boxes.append(torch.tensor([x, y, x + w, y + h]).unsqueeze(0))
                classes.append(torch.tensor(coco_id).unsqueeze(0))
                pred_masks.append(pred_mask.unsqueeze(0))
                infos.append(
                    {
                        'id_object': id_object,
                        'center': self._objects[id_object].aabb.center,
                    }
                )

        if len(bounding_boxes) > 0:
            results = Instances(
                image_size=sense.shape,
                pred_boxes=Boxes(torch.cat(bounding_boxes)),
                pred_classes=torch.cat(classes),
                scores=torch.ones(len(bounding_boxes)),
                pred_masks=torch.cat(pred_masks),
                infos=np.array(infos),
            )

        else:
            results = Instances(
                pred_boxes=Boxes(torch.Tensor()),
                image_size=sense.shape,
                pred_classes=torch.Tensor(),
                pred_masks=torch.Tensor(),
                scores=torch.Tensor(),
                infos=infos,
            )

        return {'instances': results}


@registry.register_sensor(name="object_detector_gt_discard_occlusions")
class ObjectDetectorGTDiscardOcclusions(habitat.Sensor):
    MATTERPORT_SIM_TO_COCO_MAPPING = {
        # 3: 56,  # chair
        10: 57,  # couch
        14: 58,  # plan
        11: 59,  # bed
        18: 61,  # toilet
        22: 62,  # tv
        5: 60,  # dining table
        # 15: 71,  # sink
    }

    SIM_TO_COCO_MAPPING = {
        # "chair": 56,  # chair
        "couch": 57,  # couch
        "potted plant": 58,  # plan
        "bed": 59,  # bed
        "toilet": 61,  # toilet
        "tv": 62,  # tv
        "dining table": 60,  # dining table
        # "sink": 71,  # sink
    }

    def __init__(self, sim, config, **kwargs: Any):
        super().__init__(config=config)
        self._sim = sim

        self._objects = self._sim.semantic_annotations().objects

        # filter occluded objects in mp3d by checking avg depth inside instance mask wrt gt
        self.filter_occluded_instances = True
        self.visualize = False

    # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "bbsgt"

    # Defines the type of the sensor
    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return habitat.SensorTypes.MEASUREMENT

    # Defines the size and range of the observations of the sensor
    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(3,),
            dtype=np.float32,
        )

    def _convert_matterport_to_coco_labels(self, label):
        switch = {
            'table': 'dining table',
            'plant': 'potted plant',
            'sofa': 'couch',
            'tv_monitor': 'tv',
        }
        if label in switch.keys():
            return switch[label]
        else:
            return label

    # This is called whenver reset is called or an action is taken
    def get_observation(self, *args: Any, **kwargs: Any):
        sense = kwargs['observations']['semantic']
        depth = kwargs['observations']['depth']
        self._objects = self._sim.semantic_annotations().objects
        self.mapping = {
            int(obj.id.split("_")[-1]): obj.category.name()
            for obj in self._sim.semantic_annotations().objects
            if obj is not None
        }

        objects_id = np.unique(sense)
        bounding_boxes = []
        classes = []
        pred_masks = []
        infos = []

        img = copy.deepcopy(kwargs['observations']['rgb'])

        for id_object in objects_id:
            bounding_box = cv2.boundingRect((sense == id_object).astype('uint8'))
            x, y, w, h = bounding_box
            if id_object not in self.mapping:
                continue
            habitat_id = self._convert_matterport_to_coco_labels(
                self.mapping[id_object]
            )

            if habitat_id in self.SIM_TO_COCO_MAPPING:
                pred_mask = torch.zeros(sense.shape, dtype=torch.bool)
                pred_mask[sense == id_object] = True
                coco_id = self.SIM_TO_COCO_MAPPING[habitat_id]

                if self.filter_occluded_instances:
                    object_rotation = np.quaternion(0, 0, 0, 1)  # not used now
                    object_position = [
                        self._objects[id_object].aabb.center[0],
                        self._objects[id_object].aabb.center[2],
                        self._objects[id_object].aabb.center[-1],
                    ]
                    agent_rotation = self._sim.get_agent_state().rotation.normalized()
                    agent_position = self._sim.get_agent_state().position

                    # get object pose in agent reference frame
                    object_rot_in_agent_ref = agent_rotation.inverse() * object_rotation

                    vq = np.quaternion(0, 0, 0, 1)
                    vq.imag = object_position - agent_position
                    object_pos_in_agent_ref = (
                        agent_rotation.inverse() * vq * agent_rotation
                    ).imag

                    avg_object_depth = np.mean(depth[pred_mask])

                    # if (depth - object distance) < thresh (1.0m) add objects to gt detections, else discard it
                    if (
                        np.linalg.norm(avg_object_depth + object_pos_in_agent_ref[2])
                        < 1.0
                    ):
                        bounding_boxes.append(
                            torch.tensor([x, y, x + w, y + h]).unsqueeze(0)
                        )
                        classes.append(torch.tensor(coco_id).unsqueeze(0))
                        pred_masks.append(pred_mask.unsqueeze(0))
                        infos.append(
                            {
                                'id_object': id_object,
                                'center': self._objects[id_object].aabb.center,
                            }
                        )
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                        cv2.putText(
                            img,
                            'z: %.2f' % -object_pos_in_agent_ref[2],
                            (x + 6 - 1, y + 8 - 1),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (0, 0, 0),
                            1,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            img,
                            'dep: %.2f' % avg_object_depth,
                            (x + 6 - 1, y + 18 - 1),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (0, 0, 0),
                            1,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            img,
                            'z: %.2f' % -object_pos_in_agent_ref[2],
                            (x + 6, y + 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            img,
                            'dep: %.2f' % avg_object_depth,
                            (x + 6, y + 18),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            img,
                            habitat_id,
                            (x + 6, y + 28),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            img,
                            habitat_id,
                            (x + 6, y + 28),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA,
                        )

                    else:
                        print("-- object", habitat_id, "is occluded --")
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
                        cv2.putText(
                            img,
                            'z: %.2f' % -object_pos_in_agent_ref[2],
                            (x + 6 - 1, y + 8 - 1),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (0, 0, 0),
                            1,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            img,
                            'dep: %.2f' % avg_object_depth,
                            (x + 6 - 1, y + 18 - 1),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (0, 0, 0),
                            1,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            img,
                            'z: %.2f' % -object_pos_in_agent_ref[2],
                            (x + 6, y + 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (0, 0, 255),
                            1,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            img,
                            'dep: %.2f' % avg_object_depth,
                            (x + 6, y + 18),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (0, 0, 255),
                            1,
                            cv2.LINE_AA,
                        )
                else:
                    bounding_boxes.append(
                        torch.tensor([x, y, x + w, y + h]).unsqueeze(0)
                    )
                    classes.append(torch.tensor(coco_id).unsqueeze(0))
                    pred_masks.append(pred_mask.unsqueeze(0))
                    infos.append(
                        {
                            'id_object': id_object,
                            'center': self._objects[id_object].aabb.center,
                        }
                    )

        if self.visualize:
            cv2.imshow("objects", img)
            cv2.waitKey(1)

        if len(bounding_boxes) > 0:
            results = Instances(
                image_size=sense.shape,
                pred_boxes=Boxes(torch.cat(bounding_boxes)),
                pred_classes=torch.cat(classes),
                scores=torch.ones(len(bounding_boxes)),
                pred_masks=torch.cat(pred_masks),
                infos=np.array(infos),
            )

        else:
            results = Instances(
                pred_boxes=Boxes(torch.Tensor()),
                image_size=sense.shape,
                pred_classes=torch.Tensor(),
                pred_masks=torch.Tensor(),
                scores=torch.Tensor(),
                infos=infos,
            )

        return {'instances': results}


@registry.register_sensor(name="position_sensor_origin")
class AgentPoseSensorStartingAtOrigin(Sensor):
    def __init__(self, sim: Simulator, config: Config, *args: Any, **kwargs: Any):
        self._sim = sim

        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "position"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.POSITION

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (3,)
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def _quat_to_xy_heading(self, quat):
        direction_vector = np.array([0, 0, -1])

        heading_vector = quaternion_rotate_vector(quat, direction_vector)

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return np.array(phi)

    def get_observation(self, *args: Any, observations, episode, **kwargs: Any):

        agent_state = self._sim.get_agent_state()

        origin_pose = AgentPoseSense(
            np.array(episode.start_position, dtype=np.float32),
            quaternion.quaternion(*episode.start_rotation),
        )
        current_pose = AgentPoseSense(agent_state.position, agent_state.rotation)

        current_pose_wrt_starting = current_pose.get_transformation_to_pose(origin_pose)

        return {
            'position': current_pose_wrt_starting[:3, 3],
            'orientation': quaternion.from_rotation_matrix(
                current_pose_wrt_starting[:3, :3]
            ),
        }


@registry.register_sensor(name="position_sensor")
class AgentPositionSensor(habitat.Sensor):
    def __init__(self, sim, config, **kwargs: Any):
        super().__init__(config=config)
        self._sim = sim

    # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "position"

    # Defines the type of the sensor
    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return habitat.SensorTypes.POSITION

    # Defines the size and range of the observations of the sensor
    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(3,),
            dtype=np.float32,
        )

    # This is called whenver reset is called or an action is taken
    def get_observation(self, *args: Any, **kwargs: Any):
        return {
            'position': self._sim.get_agent_state().position,
            'orientation': self._sim.get_agent_state().rotation,
        }


# @registry.register_sensor(name="noisy_position_sensor")
# class AgentNoisyPositionSensor(habitat.Sensor):
#     def __init__(self, sim, config, **kwargs: Any):
#         super().__init__(config=config)
#         self._sim = sim

#         self.position_covariance = 0.2
#         self.orientation_covariance = 0.05

#         if hasattr(config, "POSE_COV"):
#             self.position_covariance = config.POSE_COV
#         if hasattr(config, "ORIENTATION_COV"):
#             self.orientation_covariance = config.ORIENTATION_COV

#     # Defines the name of the sensor in the sensor suite dictionary
#     def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
#         return "position"

#     # Defines the type of the sensor
#     def _get_sensor_type(self, *args: Any, **kwargs: Any):
#         return habitat.SensorTypes.POSITION

#     # Defines the size and range of the observations of the sensor
#     def _get_observation_space(self, *args: Any, **kwargs: Any):
#         return spaces.Box(
#             low=np.finfo(np.float32).min,
#             high=np.finfo(np.float32).max,
#             shape=(3,),
#             dtype=np.float32,
#         )

#     # This is called whenver reset is called or an action is taken
#     def get_observation(self, *args: Any, **kwargs: Any):
#         noisy_position = self._sim.get_agent_state().position
#         noisy_position[0] += np.random.normal(0, self.position_covariance)
#         noisy_position[2] += np.random.normal(0, self.position_covariance)
#         noisy_rotation = self._sim.get_agent_state().rotation * quaternion.from_rotation_vector(np.array([0.0, 1.0, 0.0]) * np.random.normal(0, self.orientation_covariance))
#         #noisy_rotation = self._sim.get_agent_state().rotation + quaternion.from_rotation_vector(np.random.normal(0, self.orientation_covariance, size=(3,)))

#         return {
#             'position': noisy_position,
#             'orientation': noisy_rotation,
#         }


@registry.register_sensor(name="position_sensor_pixels")
class AgentPositionSensorPixels(habitat.Sensor):
    def __init__(self, sim, config, **kwargs: Any):
        super().__init__(config=config)
        self._sim = sim

    # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "position_pixels"

    # Defines the type of the sensor
    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return habitat.SensorTypes.MEASUREMENT

    # Defines the size and range of the observations of the sensor
    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(2,),
            dtype=np.float32,
        )

    # This is called whenver reset is called or an action is taken
    def get_observation(self, *args: Any, **kwargs: Any):
        top_down_map = maps.get_topdown_map_from_sim(
            self._sim,
            map_resolution=212,
            draw_border=True,
        )

        position = self._sim.get_agent_state().position
        position_pixels = maps.to_grid(
            position[2], position[0], top_down_map.shape[0:2], sim=self._sim
        )

        return position_pixels


@registry.register_sensor(name="gt_ego_map")
class GTEgoMap(Sensor):
    r"""Estimates the top-down occupancy based on current depth-map.
    Args:
        sim: reference to the simulator for calculating task observations.
        config: contains the MAP_SCALE, MAP_SIZE, HEIGHT_THRESH fields to
                decide grid-size, extents of the projection, and the thresholds
                for determining obstacles and explored space.
    """

    def __init__(self, sim: Simulator, config: Config, *args: Any, **kwargs: Any):
        self._sim = sim

        super().__init__(config=config)

        # Map statistics
        self.map_size = self.config.MAP_SIZE
        self.map_scale = self.config.MAP_SCALE
        if self.config.MAX_SENSOR_RANGE > 0:
            self.max_forward_range = self.config.MAX_SENSOR_RANGE
        else:
            self.max_forward_range = self.map_size * self.map_scale

        # Agent height for pointcloud tranforms
        self.camera_height = self._sim.habitat_config.DEPTH_SENSOR.POSITION[1]

        # Compute intrinsic matrix
        depth_H = self._sim.habitat_config.DEPTH_SENSOR.HEIGHT
        depth_W = self._sim.habitat_config.DEPTH_SENSOR.WIDTH
        hfov = float(self._sim.habitat_config.DEPTH_SENSOR.HFOV) * np.pi / 180
        vfov = 2 * np.arctan((depth_H / depth_W) * np.tan(hfov / 2.0))
        self.intrinsic_matrix = np.array(
            [
                [1 / np.tan(hfov / 2.0), 0.0, 0.0, 0.0],
                [0.0, 1 / np.tan(vfov / 2.0), 0.0, 0.0],
                [0.0, 0.0, 1, 0],
                [0.0, 0.0, 0, 1],
            ]
        )
        self.inverse_intrinsic_matrix = np.linalg.inv(self.intrinsic_matrix)

        # Height thresholds for obstacles
        self.height_thresh = self.config.HEIGHT_THRESH

        # Depth processing
        self.min_depth = float(self._sim.habitat_config.DEPTH_SENSOR.MIN_DEPTH)
        self.max_depth = float(self._sim.habitat_config.DEPTH_SENSOR.MAX_DEPTH)

        # Pre-compute a grid of locations for depth projection
        W = self._sim.habitat_config.DEPTH_SENSOR.WIDTH
        H = self._sim.habitat_config.DEPTH_SENSOR.HEIGHT
        self.proj_xs, self.proj_ys = np.meshgrid(
            np.linspace(-1, 1, W), np.linspace(1, -1, H)
        )

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "egomap"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return habitat.SensorTypes.TENSOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (self.config.MAP_SIZE, self.config.MAP_SIZE, 3)
        return spaces.Box(
            low=0,
            high=1,
            shape=sensor_shape,
            dtype=np.uint8,
        )

    def convert_to_pointcloud(self, depth):
        """
        Inputs:
            depth = (H, W, 1) numpy array
        Returns:
            xyz_camera = (N, 3) numpy array for (X, Y, Z) in egocentric world coordinates
        """

        depth_float = depth.astype(np.float32)[..., 0]

        # =========== Convert to camera coordinates ============

        xs = np.copy(self.proj_xs).reshape(-1)
        ys = np.copy(self.proj_ys).reshape(-1)
        depth_float = depth_float.reshape(-1)
        # Filter out invalid depths
        valid_depths = (depth_float != self.min_depth) & (
            depth_float <= self.max_forward_range
        )
        xs = xs[valid_depths]
        ys = ys[valid_depths]
        depth_float = depth_float[valid_depths]
        # Unproject
        # negate depth as the camera looks along -Z
        xys = np.vstack(
            (
                xs * depth_float,
                ys * depth_float,
                -depth_float,
                np.ones(depth_float.shape),
            )
        )
        inv_K = self.inverse_intrinsic_matrix
        xyz_camera = np.matmul(inv_K, xys).T  # XYZ in the camera coordinate system
        xyz_camera = xyz_camera[:, :3] / xyz_camera[:, 3][:, np.newaxis]

        return xyz_camera

    def safe_assign(self, im_map, x_idx, y_idx, value):
        try:
            im_map[x_idx, y_idx] = value
        except IndexError:
            valid_idx1 = np.logical_and(x_idx >= 0, x_idx < im_map.shape[0])
            valid_idx2 = np.logical_and(y_idx >= 0, y_idx < im_map.shape[1])
            valid_idx = np.logical_and(valid_idx1, valid_idx2)
            im_map[x_idx[valid_idx], y_idx[valid_idx]] = value

    def _get_depth_projection(self, sim_depth):
        """
        Project pixels visible in depth-map to ground-plane
        """

        if self._sim.habitat_config.DEPTH_SENSOR.NORMALIZE_DEPTH:
            depth = sim_depth * (self.max_depth - self.min_depth) + self.min_depth
        else:
            depth = sim_depth

        XYZ_ego = self.convert_to_pointcloud(depth)

        # Adding agent's height to the pointcloud
        XYZ_ego[:, 1] += self.camera_height

        # Convert to grid coordinate system
        V = self.map_size
        Vby2 = V // 2

        points = XYZ_ego

        grid_x = (points[:, 0] / self.map_scale) + Vby2
        grid_y = (points[:, 2] / self.map_scale) + V

        # Filter out invalid points
        valid_idx = (
            (grid_x >= 0) & (grid_x <= V - 1) & (grid_y >= 0) & (grid_y <= V - 1)
        )
        points = points[valid_idx, :]
        grid_x = grid_x[valid_idx].astype(int)
        grid_y = grid_y[valid_idx].astype(int)

        # Create empty maps for the two channels
        obstacle_mat = np.zeros((self.map_size, self.map_size), np.uint8)
        explore_mat = np.zeros((self.map_size, self.map_size), np.uint8)

        # Compute obstacle locations
        high_filter_idx = points[:, 1] < self.height_thresh[1]
        low_filter_idx = points[:, 1] > self.height_thresh[0]
        obstacle_idx = np.logical_and(low_filter_idx, high_filter_idx)

        self.safe_assign(obstacle_mat, grid_y[obstacle_idx], grid_x[obstacle_idx], 1)
        kernel = np.ones((3, 3), np.uint8)
        obstacle_mat = cv2.dilate(obstacle_mat, kernel, iterations=1)

        # Compute explored locations
        explored_idx = high_filter_idx
        self.safe_assign(explore_mat, grid_y[explored_idx], grid_x[explored_idx], 1)
        kernel = np.ones((3, 3), np.uint8)
        obstacle_mat = cv2.dilate(obstacle_mat, kernel, iterations=1)

        # Smoothen the maps
        kernel = np.ones((3, 3), np.uint8)

        obstacle_mat = cv2.morphologyEx(obstacle_mat, cv2.MORPH_CLOSE, kernel)
        explore_mat = cv2.morphologyEx(explore_mat, cv2.MORPH_CLOSE, kernel)

        # Ensure all expanded regions in obstacle_mat are accounted for in explored_mat
        explore_mat = np.logical_or(explore_mat, obstacle_mat)

        return np.stack([obstacle_mat, explore_mat], axis=2)

    def get_observation(self, *args: Any, **kwargs: Any):

        sim_depth = kwargs['observations']['depth']

        ego_map_gt = self._get_depth_projection(sim_depth)

        return ego_map_gt


@registry.register_sensor(name="agent_collision_sensor")
class AgentCollisionSensor(habitat.Sensor):
    r"""Estimates agent collision (when moving forward)
    Args:
        sim: reference to the simulator for calculating task observations.
        config: contains the MAP_SCALE, MAP_SIZE, HEIGHT_THRESH fields to
                decide grid-size, extents of the projection, and the thresholds
                for determining obstacles and explored space.
    """

    def __init__(self, sim, config, **kwargs):
        super().__init__(config=config)
        self._sim = sim
        self.prev_position = self._sim.get_agent_state().position

    def _get_uuid(self, *args, **kwargs):
        return "agent_collision_sensor"

    def _get_sensor_type(self, *args, **kwargs):
        return bool

    def _get_observation_space(self, *args, **kwargs):
        from gym import spaces

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(1,),
            dtype=np.bool,
        )

    def get_observation(self, observations, *args, episode, **kwargs):
        self.curr_position = self._sim.get_agent_state().position
        collision = False
        # print ("curr position:",self.curr_position)
        if np.linalg.norm(self.curr_position - self.prev_position) < 0.15:
            collision = True
            # print("collision!")
        self.prev_position = copy.deepcopy(self.curr_position)

        return collision


@registry.register_sensor(name="proximity_sensor")
class ProximitySensor(Sensor):
    r"""Sensor for observing the distance to the closest obstacle
    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor.
    """

    def __init__(self, sim, config):
        self._sim = sim
        self._max_detection_radius = 2.0
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "proximity"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TACTILE

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0.0, high=self._max_detection_radius, shape=(1,), dtype=np.float64
        )

    def get_observation(self, observations, episode):
        current_position = self._sim.get_agent_state().position

        return self._sim.distance_to_closest_obstacle(
            current_position, self._max_detection_radius
        )


@registry.register_sensor(name="movement_sensor")
class MovementSensor(habitat.Sensor):
    r"""Computes cartesian robot movement on the plane
    Args:
        sim: reference to the simulator for calculating task observations.
        config: contains the MAP_SCALE, MAP_SIZE, HEIGHT_THRESH fields to
                decide grid-size, extents of the projection, and the thresholds
                for determining obstacles and explored space.
    """

    def __init__(self, sim, config, **kwargs):
        super().__init__(config=config)
        self._sim = sim
        self.prev_position = self._sim.get_agent_state().position

    def _get_uuid(self, *args, **kwargs):
        return "movement"

    def _get_sensor_type(self, *args, **kwargs):
        return habitat.SensorTypes.MEASUREMENT

    def _get_observation_space(self, *args, **kwargs):
        from gym import spaces

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(1,),
            dtype=np.float32,
        )

    def get_observation(self, observations, *args, episode, **kwargs):
        self.curr_position = self._sim.get_agent_state().position
        moved_dist = np.linalg.norm(self.curr_position - self.prev_position)
        self.prev_position = copy.deepcopy(self.curr_position)
        return moved_dist


@registry.register_sensor(name="map_sensor")
class MapSensor(habitat.Sensor):
    r"""Gets environment map
    Args:
        sim: reference to the simulator for calculating task observations.
        config: contains the MAP_SCALE, MAP_SIZE, HEIGHT_THRESH fields to
                decide grid-size, extents of the projection, and the thresholds
                for determining obstacles and explored space.
    """

    def __init__(self, sim, config, **kwargs):
        super().__init__(config=config)
        self._sim = sim
        self.prev_position = self._sim.get_agent_state().position
        self.config = config

    def _get_uuid(self, *args, **kwargs):
        return "map_sensor"

    def _get_sensor_type(self, *args, **kwargs):
        return bool

    def _get_observation_space(self, *args, **kwargs):
        from gym import spaces

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(512, 512),
            dtype=np.int64,
        )

    def get_observation(self, observations, *args, episode, **kwargs):
        agent_position = self._sim.get_agent_state().position
        agent_rotation = self._sim.get_agent_state().rotation
        meters_per_pixel = 0.025
        scene_bb = self._sim.get_active_scene_graph().get_root_node().cumulative_bb
        height = scene_bb.y().min
        top_down_map = maps.get_topdown_map_from_sim(
            self._sim,
            map_resolution=0,
            draw_border=True,
            meters_per_pixel=meters_per_pixel,
        )

        recolor_map = np.array(
            [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
        )

        top_down_map = recolor_map[top_down_map]
        grid_dimensions = (top_down_map.shape[0], top_down_map.shape[1])
        # convert world trajectory points to maps module grid points
        agent_position_on_map = maps.to_grid(
            agent_position[2],
            agent_position[0],
            grid_dimensions,
            pathfinder=self._sim.pathfinder,
        )
        agent_forward = sim_utils.quat_to_magnum(agent_rotation).transform_vector(
            mn.Vector3(0, 0, -1.0)
        )
        agent_orientation = math.atan2(agent_forward[0], agent_forward[2])
        top_down_map = maps.draw_agent(
            top_down_map,
            agent_center_coord=agent_position_on_map,
            agent_rotation=agent_orientation,
            agent_radius_px=4,
        )

        return top_down_map


@registry.register_sensor(name="object_detector_detectron")
class ObjectDetectorDetectron(habitat.Sensor):
    def __init__(self, sim, config, **kwargs: Any):
        super().__init__(config=config)
        self._sim = sim

        self.object_detector = MultiStageModel(config)

        if os.path.exists(config.LOAD_PATH):
            self.object_detector.load_state_dict(torch.load(config.LOAD_PATH))
        if hasattr(config, "DEVICE"):
            self.object_detector.to(config.DEVICE)

        self.object_detector.eval()
        self.object_detector.model.roi_heads.box_predictor.box_predictor.test_score_thresh = (
            0.5
        )

    def load(self, load_path, device="cpu"):
        if os.path.exists(load_path):
            self.object_detector.load_state_dict(torch.load(load_path)).to(device)

    def move_to_device(self, device='cpu'):
        self.object_detector.to(device)

    # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "bbs"

    # Defines the type of the sensor
    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return habitat.SensorTypes.MEASUREMENT

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(3,),
            dtype=np.float32,
        )

    # This is called whenver reset is called or an action is taken
    def get_observation(self, *args: Any, **kwargs: Any):
        rgb = kwargs['observations']['rgb']

        out = self.object_detector.infer([rgb])
        instances[0][0]['instances'] = instances[0][0]['instances'].cpu()
        logits = instances[1][0]['instances']

        return instances[0][0], logits.to("cpu")


@registry.register_sensor(name="noisy_position_sensor")
class AgentNoisyPositionSensor(habitat.Sensor):
    def __init__(self, sim, config, **kwargs: Any):
        super().__init__(config=config)
        self._sim = sim

        self.position_covariance = 0.001
        self.orientation_covariance = 0.0002

        if hasattr(config, "POSE_COV"):
            self.position_covariance = config.POSE_COV
        if hasattr(config, "ORIENTATION_COV"):
            self.orientation_covariance = config.ORIENTATION_COV

    # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "position"

    # Defines the type of the sensor
    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return habitat.SensorTypes.POSITION

    # Defines the size and range of the observations of the sensor
    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(3,),
            dtype=np.float32,
        )

    def get_observation(self, *args: Any, **kwargs: Any):
        noisy_position = self._sim.get_agent_state().position
        noisy_position[0] += np.random.normal(0, self.position_covariance)
        noisy_position[2] += np.random.normal(0, self.position_covariance)

        # Check for out of bounds
        if noisy_position[0] < self._sim.pathfinder.get_bounds()[0][0]:
            noisy_position[0] = (
                self._sim.pathfinder.get_bounds()[0][0] + self.position_covariance
            )
        if noisy_position[2] < self._sim.pathfinder.get_bounds()[0][2]:
            noisy_position[2] = (
                self._sim.pathfinder.get_bounds()[0][2] + self.position_covariance
            )
        if noisy_position[0] > self._sim.pathfinder.get_bounds()[1][0]:
            noisy_position[0] = (
                self._sim.pathfinder.get_bounds()[1][0] - self.position_covariance
            )
        if noisy_position[2] > self._sim.pathfinder.get_bounds()[1][2]:
            noisy_position[2] = (
                self._sim.pathfinder.get_bounds()[1][2] - self.position_covariance
            )

        y = 2.0 * math.atan2(
            self._sim.get_agent_state().rotation.y,
            self._sim.get_agent_state().rotation.w,
        )
        y += np.random.normal(0, self.orientation_covariance)
        noisy_rotation = quaternion.from_rotation_vector([0.0, y, 0.0])

        return {
            'position': noisy_position,
            'orientation': noisy_rotation,
        }


@registry.register_sensor(name="noisy_position_sensor2")
class AgentNoisyPositionSensor(habitat.Sensor):
    def __init__(self, sim, config, **kwargs: Any):
        super().__init__(config=config)
        self._sim = sim

        self.position_covariance = 0.01
        self.orientation_covariance = 0.01

        if hasattr(config, "POSE_COV"):
            self.position_covariance = config.POSE_COV
        if hasattr(config, "ORIENTATION_COV"):
            self.orientation_covariance = config.ORIENTATION_COV

    # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "position"

    # Defines the type of the sensor
    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return habitat.SensorTypes.POSITION

    # Defines the size and range of the observations of the sensor
    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(3,),
            dtype=np.float32,
        )

    # This is called whenver reset is called or an action is taken
    def get_observation(self, *args: Any, **kwargs: Any):
        noisy_position = self._sim.get_agent_state().position
        noisy_position[0] += np.random.normal(0, self.position_covariance)
        noisy_position[2] += np.random.normal(0, self.position_covariance)

        # check for out of bounds
        if noisy_position[0] < self._sim.pathfinder.get_bounds()[0][0]:
            noisy_position[0] = (
                self._sim.pathfinder.get_bounds()[0][0] + self.position_covariance
            )
        if noisy_position[2] < self._sim.pathfinder.get_bounds()[0][2]:
            noisy_position[2] = (
                self._sim.pathfinder.get_bounds()[0][2] + self.position_covariance
            )
        if noisy_position[0] > self._sim.pathfinder.get_bounds()[1][0]:
            noisy_position[0] = (
                self._sim.pathfinder.get_bounds()[1][0] - self.position_covariance
            )
        if noisy_position[2] > self._sim.pathfinder.get_bounds()[1][2]:
            noisy_position[2] = (
                self._sim.pathfinder.get_bounds()[1][2] - self.position_covariance
            )

        noisy_rotation = (
            quaternion.from_rotation_vector(
                np.array([0.0, 1.0, 0.0])
                * np.random.normal(0, self.orientation_covariance)
            )
            * self._sim.get_agent_state().rotation
        )
        # renormalize
        d = np.sqrt(
            noisy_rotation.x ** 2
            + noisy_rotation.y ** 2
            + noisy_rotation.z ** 2
            + noisy_rotation.w ** 2
        )
        noisy_rotation.x /= d
        noisy_rotation.y /= d
        noisy_rotation.z /= d
        noisy_rotation.w /= d

        return {
            'position': noisy_position,
            'orientation': noisy_rotation,
        }


@registry.register_sensor(name="semantic_instances")
class SemanticInstances(habitat.Sensor):
    MATTERPORT_SIM_TO_COCO_MAPPING = {
        3: 56,  # chair
        10: 57,  # couch
        14: 58,  # plan
        11: 59,  # beed
        18: 61,  # toilet
        22: 62,  # tv
    }

    SIM_TO_COCO_MAPPING = {
        # "chair": 56,  # chair
        "couch": 57,  # couch
        "potted plant": 58,  # plan
        "bed": 59,  # bed
        "toilet": 61,  # toilet
        "tv": 62,  # tv
        "dining table": 60,  # dining table
    }

    DEPTH_THR = 5

    def __init__(self, sim, config, **kwargs: Any):
        super().__init__(config=config)
        self._sim = sim

        self._objects = self._sim.semantic_annotations().objects

        self.scene = ""

        # filter occluded objects in mp3d by checking avg depth inside instance mask wrt gt
        self.filter_occluded_instances = True

    # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "semanticinstances"

    # Defines the type of the sensor
    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return habitat.SensorTypes.MEASUREMENT

    # Defines the size and range of the observations of the sensor
    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(3,),
            dtype=np.float32,
        )

    def _convert_matterport_to_coco_labels(self, label):
        switch = {
            'table': 'dining table',
            'plant': 'potted plant',
            'sofa': 'couch',
            'tv_monitor': 'tv',
        }
        if label in switch.keys():
            return switch[label]
        else:
            return label

    # This is called whenver reset is called or an action is taken
    def get_observation(self, *args: Any, **kwargs: Any):
        sense = kwargs['observations']['semantic']

        self._objects = self._sim.semantic_annotations().objects

        self.mapping = {
            int(obj.id.split("_")[-1]): obj.category.name()
            for obj in self._sim.semantic_annotations().objects
            if obj is not None
        }

        room_categories = {
            k: self._sim.semantic_annotations().objects[k].region.category.name()
            for k in self.mapping.keys()
            if self._sim.semantic_annotations().objects[k].region is not None
        }

        rooms = {v: k for k, v in enumerate(set(self.mapping.values()))}
        idxs = {v: k for k, v in enumerate(set(self.mapping.values()))}
        result = np.zeros_like(sense)
        objects_id = np.unique(sense)
        for id_object in objects_id:
            if id_object not in self.mapping:
                continue
            id_class = idxs[self.mapping[id_object]]
            result[sense == id_object] = id_class

        return {
            'semantic_instances': result,
            "mapping": idxs,
            "room_categories": room_categories,
        }
