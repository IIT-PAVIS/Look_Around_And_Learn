import abc
import dataclasses
from dataclasses import dataclass
from pathlib import WindowsPath

import cv2
import numpy as np
import quaternion
import torch
from detectron2.data import MetadataCatalog
from detectron2.structures import BitMasks
from detectron2.structures.instances import Instances
from detectron2.utils.visualizer import ColorMode, Visualizer
from habitat.utils.visualizations import maps

from .utils import SenseInfo, get_sense_info


def get_class_from_modality_code(code: str):
    switch = {
        "rgb": RGBSense,
        "depth": DepthSense,
        "semantic": SemanticSense,
        "semanticinstances": SemanticInstancesSense,
        "bbs": BBSense,
        "bbsgt": BBSense,
        'position': AgentPoseSense,
        'egomap': EgomapSense,
    }
    return switch[code]


class Sense(abc.ABC):
    def __init__(self, path: str = None, sense_info: SenseInfo = None):
        if sense_info is None and path is not None:
            self.sense_info = get_sense_info(path)
        elif sense_info is not None:
            self.sense_info = sense_info
        else:
            self.sense_info = None

        if self.sense_info is not None:
            self.name = f"{self.sense_info.episode}-{self.sense_info.mod}-{self.sense_info.camera_id}"
        else:
            self.name = ""

    @staticmethod
    def load(path):
        return Sense(path)


class Pose(Sense):
    AGENT_TO_SENSOR_TRANSLATION = np.array([0, 0.88, 0])

    def __init__(
        self,
        position: np.ndarray,
        orientation,
        reference: str,
        path=None,
        sense_info=None,
    ):
        super().__init__(path, sense_info)
        self.position = position
        self.orientation = orientation
        self.reference = reference

    def get_T(self):
        """
        Get pose_world transformation matrix for pose
        """
        rotation_0 = quaternion.as_rotation_matrix(self.orientation)
        T = np.eye(4)
        T[0:3, 0:3] = rotation_0
        T[0:3, 3] = self.position
        return T

    def get_transformation_to_pose(self, pose2):
        """
        Transformation from current pose to given pose
        """

        T_world_pose1 = self.get_T()
        T_world_pose2 = pose2.get_T()

        T_pose2_world = np.linalg.inv(T_world_pose2)

        T_pose2_pose1 = np.matmul(T_pose2_world, T_world_pose1)
        return T_pose2_pose1


class AgentPoseSense(Pose):

    CODE = "position"

    def __init__(
        self, position: np.ndarray, orientation: quaternion, path=None, sense_info=None
    ):
        super().__init__(
            position, orientation, "agent", path=path, sense_info=sense_info
        )

    def get_T_world_agent(self):
        """
        Get pose_world transformation matrix for pose
        """
        rotation_0 = quaternion.as_rotation_matrix(self.orientation)
        T = np.eye(4)
        T[0:3, 0:3] = rotation_0
        T[0:3, 3] = self.position
        return T

    def get_cam_pose(self):
        """
        Get pose_world transformation matrix for pose
        """
        rot_mat = quaternion.as_rotation_matrix(self.orientation)
        translation = np.matmul(rot_mat, AgentPoseSense.AGENT_TO_SENSOR_TRANSLATION)
        position = self.position + translation
        return CamPoseSense(
            position=position, orientation=self.orientation, sense_info=self.sense_info
        )

    @staticmethod
    def load(path):

        location_data = np.load(path, allow_pickle=True)

        try:
            position = location_data.item()['position']
            orientation = location_data.item()['orientation']

        except Exception as ex:  # type: ignore[F841]
            position = location_data[0]
            orientation = location_data[1]

        return AgentPoseSense(position, orientation, path).get_cam_pose()


class CamPoseSense(Pose):
    def __init__(
        self, position: np.ndarray, orientation: quaternion, path=None, sense_info=None
    ):
        super().__init__(position, orientation, "cam", path=path, sense_info=sense_info)


@dataclass
class Intrinsics:
    xc: float
    yc: float
    focal_length: float
    width: int
    height: int

    def get_mat(self) -> np.ndarray:
        return np.array(
            [
                [self.focal_length, 0, self.xc],
                [0.0, self.focal_length, self.yc],
                [0.0, 0, 1],
            ]
        )


class VisualSense(Sense):
    HFOV_DEG = 90

    def get_camera_matrix(self, fov=HFOV_DEG):
        """
        From Object-Goal-Navigation
        Returns a camera matrix from image size and fov.
        """
        width = height = self.get_width()
        xc = (width - 1.0) / 2.0
        yc = (height - 1.0) / 2.0
        f = (width / 2.0) / np.tan(np.deg2rad(fov) / 2.0)

        return Intrinsics(xc, yc, f, width, height)

    def __init__(self, data: np.ndarray = None, path=None, sense_info=None):
        super().__init__(path, sense_info)

        self.data = data

    def get_width(self):
        return self.data.shape[0]

    def show(self):
        cv2.imshow(self.name, self.data)


class DepthSense(VisualSense):
    CODE = "depth"

    def __init__(self, data=None, path=None, sense_info=None):
        super().__init__(data, path, sense_info)

    @staticmethod
    def load(path):
        depth_image = np.load(path)

        if "neuralslam" in path:
            depth_image = depth_image * 10  # only for neuralslam

        return DepthSense(depth_image, path)


class RGBSense(VisualSense):
    CODE = "rgb"
    INPUT_FORM = "RGB"  # RGB

    def __init__(self, data: np.ndarray = None, path=None, sense_info=None):
        super().__init__(data, path, sense_info)

    @staticmethod
    def load(path):

        rgb_image = np.load(path)

        if (
            rgb_image.shape[0] == 3
            or rgb_image.shape[0] == 1
            or rgb_image.shape[0] == 4
        ):
            # channel-last
            rgb_image = rgb_image.transpose(1, 2, 0)

        if rgb_image.shape[-1] > 3:
            rgb_image = rgb_image[:, :, :-1]  # remove `a`` channel

        rgb_image = np.ascontiguousarray(
            rgb_image[:, :, ::-1]
        )  # RGB (from np loading) to BGR (cv2)

        return RGBSense(rgb_image, path)


class SemanticSense(VisualSense):
    CODE = "semantic"

    def __init__(self, data: np.ndarray, path=None, sensor_info=None):
        super().__init__(data, path, sensor_info)

    @staticmethod
    def load(path):

        semantic_image = np.load(path).astype("uint8")
        return SemanticSense(semantic_image, path)

    def show(self):
        heatmap = cv2.applyColorMap(self.data, cv2.COLORMAP_HSV)
        cv2.imshow(self.name, heatmap)


class SemanticInstancesSense(VisualSense):
    CODE = "semantic"

    def __init__(self, data: np.ndarray, mapping=None, path=None, sensor_info=None):
        super().__init__(data, path, sensor_info)
        self.mapping = mapping

    @staticmethod
    def load(path):
        data = np.load(path, allow_pickle=True).item()

        semantic_image = data['semantic_instances']
        mapping = data['mapping']
        return SemanticInstancesSense(semantic_image, mapping, path)

    def show(self):
        heatmap = cv2.applyColorMap(self.data, cv2.COLORMAP_HSV)
        cv2.imshow(self.name, heatmap)


class EgomapSense(VisualSense):
    CODE = "egomap"

    def __init__(self, data: np.ndarray, path=None, sensor_info=None):
        super().__init__(data, path, sensor_info)

    @staticmethod
    def load(path):
        egomap = np.load(path)
        return EgomapSense(egomap, path)

    def show(self):
        heatmap = maps.colorize_topdown_map(self.data[:, :, 1])
        cv2.imshow(self.name, heatmap)


class BBSense(VisualSense):
    CODE = "bbs"
    CLASSES = {
        57: "couch",
        58: "plant",
        59: "bed",
        61: "toilet",
        62: "tv",
        60: "table",
    }

    REMAP = {i: k for i, k in enumerate(CLASSES)}
    CLASSES_TO_IDX = {k: i for i, k in enumerate(CLASSES.keys())}

    def __init__(self, bbs: Instances, frame=None, path=None, sense_info=None):
        super().__init__(bbs, path, sense_info)
        self.bbs = bbs
        rgb_sense_info = dataclasses.replace(self.sense_info, mod=RGBSense.CODE)

        try:
            if frame is None and rgb_sense_info is not None:
                frame = RGBSense.load(rgb_sense_info.get_path())
            self.frame = frame
        except Exception as ex:
            self.frame = None

    @staticmethod
    def load(path_bb):
        bbs = BBSense._load_bbs(path_bb)

        if len(bbs) > 0:
            mask = [x.item() in BBSense.CLASSES.keys() for x in bbs.pred_classes]
            bbs = bbs[mask]

        return BBSense(path=path_bb, bbs=bbs)

    @staticmethod
    def _load_bbs(path):
        raw_prediction = np.load(path, allow_pickle=True)

        instances = raw_prediction.item()['instances']
        if len(instances) == 0:
            return instances
        mask = [
            instances[i].pred_classes.item() in BBSense.CLASSES.keys()
            for i in range(len(instances))
        ]

        return instances[mask]

    def get_bbs_as_gt(self):

        target = Instances(self.bbs.image_size)
        target.gt_boxes = self.bbs.pred_boxes
        target.gt_classes = self.bbs.pred_classes

        if hasattr(self.bbs, "pred_masks"):
            target.gt_masks = self.bbs.pred_masks

        if hasattr(self.bbs, "infos"):

            target.infos = self.bbs.infos
            for t in target.infos:
                t['episode'] = self.sense_info.episode

        return target

    def get_bounding_boxes(self):
        if 'pred_boxes' in self.bbs:
            return self.bbs.pred_boxes
        else:
            return []

    def show(self):
        metadata = MetadataCatalog.get('coco_2017_val')

        visualizer = Visualizer(
            self.frame.data, metadata, instance_mode=ColorMode.IMAGE
        )
        self.bbs.scores = torch.tensor([0.95 for _ in range(len(self.bbs))])
        frame = visualizer.draw_instance_predictions(predictions=self.bbs.to('cpu'))

        cv2.imshow(self.name, frame.get_image()[:, :, ::-1])

        cv2.imwrite(
            self.name + str(self.sense_info.step) + '.png',
            frame.get_image()[:, :, ::-1],
        )
