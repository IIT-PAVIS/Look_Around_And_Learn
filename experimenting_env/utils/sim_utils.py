# type: ignore
"""
From https://github.com/facebookresearch/habitat-sim/blob/master/habitat_sim/utils/data/data_extractor.py
"""

import collections
import copy
import json
import math
from collections import namedtuple
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Type,
    Union,
)

import attr
import cv2
import habitat
import habitat_sim
import habitat_sim.registry as registry
import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.structures import Boxes, Instances
from gym import spaces
from habitat.config import Config
from habitat.core.simulator import Sensor, SensorTypes, Simulator
from habitat.core.utils import try_cv2_import
from habitat.tasks.nav.nav import PointGoalSensor
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.visualizations import fog_of_war, maps
from habitat_sim import bindings as hsim
from habitat_sim.agent.agent import AgentConfiguration, AgentState
from habitat_sim.utils.common import (
    quat_from_angle_axis,
    quat_from_two_vectors,
)
from habitat_sim.utils.data import ImageExtractor, PoseExtractor
from habitat_sim.utils.data.pose_extractor import TopdownView
from numpy import bool_, float32, float64, ndarray
from quaternion import numpy_quaternion, quaternion

from experimenting_env.detector.model.multi_stage_models import MultiStageModel
from experimenting_env.envs.sensors import ObjectDetectorGT
from experimenting_env.sensor_data import BBSense
from experimenting_env.utils.predictor_utils import Predictor, setup_cfg


@registry.register_pose_extractor(name="closest_point_extractor")
class ClosestPointExtractor(PoseExtractor):
    def __init__(
        self,
        topdown_views: List[Tuple[TopdownView, str, Tuple[float32, float32, float32]]],
        meters_per_pixel: float = 0.1,
    ) -> None:
        super().__init__(topdown_views, meters_per_pixel)

    def extract_poses(
        self, view: ndarray, fp: str
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int], str]]:
        # Determine the physical spacing between each camera position
        height, width = view.shape
        dist = min(height, width) // 50  # We can modify this to be user-defined later

        # Create a grid of camera positions
        n_gridpoints_width, n_gridpoints_height = (
            width // dist - 1,
            height // dist - 1,
        )

        # Exclude camera positions at invalid positions
        gridpoints = []
        for h in range(n_gridpoints_height):
            for w in range(n_gridpoints_width):
                point = (dist + h * dist, dist + w * dist)
                if self._valid_point(*point, view):
                    gridpoints.append(point)

        # Find the closest point of the target class to each gridpoint
        poses = []
        for point in gridpoints:
            closest_point_of_interest, label = self._bfs(point, self.labels, view, 5)
            if closest_point_of_interest is None:
                continue

            poses.append((point, closest_point_of_interest, fp))

        # Returns poses in the coordinate system of the topdown view
        return poses

    def _bfs(
        self, point: Tuple[int, int], labels: List[float], view: ndarray, dist: int
    ) -> Union[Tuple[Tuple[int, int], float64], Tuple[None, None]]:
        step = 3  # making this larger really speeds up BFS

        def get_neighbors(p):
            r, c = p
            return [
                (r - step, c - step),
                (r - step, c),
                (r - step, c + step),
                (r, c - step),
                (r, c + step),
                (r + step, c - step),
                (r + step, c),
                (r + step, c + step),
            ]

        point_row, point_col = point
        bounding_box = [
            point_row - 2 * dist,
            point_row + 2 * dist,
            point_col - 2 * dist,
            point_col + 2 * dist,
        ]
        in_bounds = (
            lambda row, col: bounding_box[0] <= row <= bounding_box[1]
            and bounding_box[2] <= col <= bounding_box[3]
        )
        is_valid = lambda row, col: 0 <= row < len(view) and 0 <= col < len(view[0])
        visited = set()
        q = collections.deque([(point, 0)])
        while q:
            cur, layer = q.popleft()
            if not in_bounds(*cur):  # No point of interest found within bounding box
                return None, None

            visited.add(cur)
            is_point_of_interest, label = self._is_point_of_interest(cur, labels, view)
            if is_point_of_interest:
                if layer > dist / 2:
                    return cur, label
                else:
                    return None, None

            for n in get_neighbors(cur):
                if n not in visited and is_valid(*n):
                    q.append((n, layer + step))

        return None, None


@registry.register_pose_extractor(name="random_pose_extractor")
class RandomPoseExtractor(PoseExtractor):
    def extract_poses(self, view, fp):
        height, width = view.shape
        num_random_points = 5000
        points = []
        while len(points) < num_random_points:
            # Get the row and column of a random point on the topdown view
            row, col = np.random.randint(0, height), np.random.randint(0, width)

            # Convenient method in the PoseExtractor class to check if a point
            # is navigable
            if self._valid_point(row, col, view):
                points.append((row, col))

        poses = []

        # Now we need to define a "point of interest" which is the point the camera will
        # look at. These two points together define a camera position and angle
        for point in points:
            r, c = point
            point_of_interest = (r - 1, c)  # Just look forward
            pose = (point, point_of_interest, fp)
            poses.append(pose)

        return poses


class FirstPersonImageExtractor(ImageExtractor):
    def __init__(
        self,
        scene_filepath: Union[str, List[str]],
        labels: List[float] = None,
        img_size: tuple = (512, 512),
        output: List[str] = None,
        pose_extractor_name: str = "random_pose_extractor",
        sim=None,
        shuffle: bool = True,
        split: tuple = (100, 0),
        use_caching: bool = True,
        meters_per_pixel: float = 0.1,
    ):
        super().__init__(
            scene_filepath,
            labels,
            img_size,
            output,
            pose_extractor_name,
            sim,
            shuffle,
            split,
            use_caching,
            meters_per_pixel,
        )

        self.mode = "full"
        self._objects = self.sim.semantic_scene.objects
        self.mapping = self.instance_id_to_name

        self.filter_occluded_instances = True

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start, stop, step = idx.start, idx.stop, idx.step
            if start is None:
                start = 0
            if stop is None:
                stop = len(self.mode_to_data[self.mode])
            if step is None:
                step = 1

            return [
                self.__getitem__(i)
                for i in range(start, stop, step)
                if i < len(self.mode_to_data[self.mode])
            ]

        mymode = self.mode.lower()
        if self.use_caching:
            cache_entry = (idx, mymode)
            if cache_entry in self.cache:
                return self.cache[cache_entry]

        poses = self.mode_to_data[mymode]
        pos, rot, fp = poses[idx]

        # Only switch scene if it is different from the last one accessed
        if fp != self.cur_fp:
            self.sim.reconfigure(self._config_sim(fp, self.img_size))
            self.cur_fp = fp

        new_state = AgentState()
        new_state.position = pos
        new_state.rotation = rot
        self.sim.agents[0].set_state(new_state)

        obs = self.sim.get_sensor_observations()

        sample = {
            out_name: obs[self.out_name_to_sensor_name[out_name]]
            for out_name in self.output
        }

        obs = {}

        obs['semantic'] = sample['semantic']
        obs['rgb'] = sample['rgba']
        obs['depth'] = sample['depth']
        obs['position'] = {'position': pos, 'orientation': rot}
        detections = self.get_bbs(**obs)
        obs['bbsgt'] = detections

        if self.use_caching:
            self.cache.add(cache_entry, obs)

        return obs

    SIM_TO_COCO_MAPPING = {
        "couch": 57,  # couch
        "potted plant": 58,  # plan
        "bed": 59,  # bed
        "toilet": 61,  # toilet
        "tv": 62,  # tv
        "dining table": 60,  # dining table

    }
    # This is called whenver reset is called or an action is taken

    def get_bbs(self, *args, **kwargs):

        sense = kwargs['semantic']
        depth = kwargs['depth']

        objects_id = np.unique(sense)
        bounding_boxes = []
        classes = []
        pred_masks = []
        infos = []

        img = copy.deepcopy(kwargs['rgb'])

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

                    agent_rotation = (
                        self.sim.agents[0].get_state().rotation.normalized()
                    )
                    agent_position = self.sim.agents[0].get_state().position

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

    def _config_sim(self, scene_filepath, img_size):
        settings = {
            "width": img_size[1],  # Spatial resolution of the observations
            "height": img_size[0],
            "scene": scene_filepath,  # Scene path
            "default_agent": 0,
            "sensor_height": 0.88,  # Height of sensors in meters
            "color_sensor": True,  # RGBA sensor
            "semantic_sensor": True,  # Semantic sensor
            "depth_sensor": True,  # Depth sensor
            "silent": True,
        }

        sim_cfg = hsim.SimulatorConfiguration()
        sim_cfg.enable_physics = False
        sim_cfg.gpu_device_id = 0
        sim_cfg.scene_id = settings["scene"]

        # define default sensor parameters (see src/esp/Sensor/Sensor.h)
        sensor_specs = []
        if settings["color_sensor"]:
            color_sensor_spec = habitat_sim.CameraSensorSpec()
            color_sensor_spec.uuid = "color_sensor"
            color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
            color_sensor_spec.hfov = 79
            color_sensor_spec.resolution = [settings["height"], settings["width"]]
            color_sensor_spec.postition = [0.0, settings["sensor_height"], 0.0]
            color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            sensor_specs.append(color_sensor_spec)

        if settings["depth_sensor"]:
            depth_sensor_spec = habitat_sim.CameraSensorSpec()
            depth_sensor_spec.uuid = "depth_sensor"
            depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
            depth_sensor_spec.hfov = 79
            depth_sensor_spec.resolution = [settings["height"], settings["width"]]
            depth_sensor_spec.postition = [0.0, settings["sensor_height"], 0.0]
            depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            sensor_specs.append(depth_sensor_spec)

        if settings["semantic_sensor"]:
            semantic_sensor_spec = habitat_sim.CameraSensorSpec()
            semantic_sensor_spec.uuid = "semantic_sensor"
            semantic_sensor_spec.hfov = 79
            semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
            semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
            semantic_sensor_spec.postition = [0.0, settings["sensor_height"], 0.0]
            semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            sensor_specs.append(semantic_sensor_spec)

        # create agent specifications
        agent_cfg = AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_specs
        return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def draw_top_down_map(topdown_map):
    return maps.colorize_topdown_map(topdown_map)
