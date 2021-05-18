import importlib
import logging
import time
from dataclasses import dataclass
from functools import partial, reduce
from operator import add
from typing import Dict, Union

import cc3d
import cv2
import numpy as np
import pandas as pd
import torch
from detectron2.structures import Boxes, Instances
from pyntcloud import PyntCloud
from scipy.spatial.kdtree import KDTree
from torch.nn import functional as F

from experimenting_env.sensor_data import AgentPoseSense, BBSense, DepthSense
from experimenting_env.utils import inconsistencies as inc

log = logging.getLogger(__name__)
O3D_AVAILABLE = importlib.util.find_spec("open3d") is not None
if O3D_AVAILABLE:
    import open3d as o3d  # noqa: E402


def _project_predictions(
    instances: Instances,
    depth: DepthSense,
    cam1_to_cam0_transformation,
    filtering_depth_value=7.0,
):
    results = []
    W = depth.get_width()
    intrinsics_matrix = depth.get_camera_matrix()
    Fx = Fy = intrinsics_matrix.focal_length
    Cx = intrinsics_matrix.xc
    Cy = intrinsics_matrix.yc

    def _project_point(u, v):

        zl = depth.data[v, u]
        if zl > filtering_depth_value:
            return

        xyzl_cam1 = np.vstack(((u - Cx) / Fx * zl, -(v - Cy) / Fy * zl, -zl, 1))

        xyzl_wrt_cam0 = np.matmul(cam1_to_cam0_transformation, xyzl_cam1)

        xyz_wrt_cam0 = np.matmul(
            np.concatenate([intrinsics_matrix.get_mat(), np.zeros((3, 1))], axis=1),
            xyzl_wrt_cam0,
        )

        # Normalize by depth
        xy_wrt_cam0 = xyz_wrt_cam0[0:2, :] / xyz_wrt_cam0[2, :]
        xy_wrt_cam0[0] = W - xy_wrt_cam0[0]

        if all([x > -1 and x < W for x in xy_wrt_cam0]):
            return xy_wrt_cam0
        else:
            return

    projected_predictions: Dict = {
        'pred_boxes': [],
        'pred_classes': [],
        'scores': [],
    }
    projected_boxes = Boxes(torch.Tensor())
    projected_classes = torch.Tensor()
    projected_scores = torch.Tensor()

    for idx in range(len(instances)):
        prediction = instances[idx]
        pred_box = prediction.pred_boxes.tensor[0]

        u1 = min(int(pred_box[0]), W - 1)
        v1 = min(int(pred_box[1]), W - 1)
        p1 = _project_point(u1, v1)

        u2 = min(int(pred_box[2]), W - 1)
        v2 = min(int(pred_box[3]), W - 1)

        p2 = _project_point(u2, v2)

        if p1 is None or p2 is None:
            continue

        projected_predictions['pred_boxes'].append(
            torch.tensor([p1[0], p1[1], p2[0], p2[1]])
        )
        projected_predictions['pred_classes'].append(prediction.pred_classes)
        projected_predictions['scores'].append(prediction.scores)

    if len(projected_predictions['pred_boxes']) > 0:

        idx_sorted = sorted(
            range(len(projected_predictions['pred_boxes'])),
            key=lambda k: projected_predictions['pred_boxes'][k][0],
        )

        projected_boxes = Boxes(
            torch.cat(projected_predictions['pred_boxes'], 1).permute(1, 0)
        )[idx_sorted]
        projected_classes = torch.cat(projected_predictions['pred_classes'], 0)[
            idx_sorted
        ]
        projected_scores = torch.cat(projected_predictions['scores'], 0)[idx_sorted]

    results = Instances(
        image_size=instances.image_size,
        pred_boxes=projected_boxes,
        pred_classes=projected_classes,
        scores=projected_scores,
    )
    return results


def project_cam2_bb_onto_cam1(
    depth: DepthSense,
    pose: AgentPoseSense,
    predictions: Instances,
    new_pose,
    fov_deg=79,
) -> Instances:
    """
    Depth, pose, and predictions from one camera, and new pose
    """

    cam1_to_cam0_transformation = pose.get_transformation_to_pose(new_pose)
    projected_predictions = _project_predictions(
        predictions, depth, cam1_to_cam0_transformation
    )

    return projected_predictions


@dataclass
class Info:
    classes: torch.Tensor
    score: torch.Tensor
    pose: torch.Tensor


def project_semantic_masks_to_3d(
    depth: torch.Tensor,
    pose: Union[torch.Tensor, AgentPoseSense, Dict],
    instances: Instances,
    infos,
    update_logits=True,
) -> "SemanticPointCloud":
    """
    depth: WxH depth tensor
    pose: 4x1 position and location
    instances: pred or ground-truth detectron2 Instances
    """

    n_instances = len(instances)

    if isinstance(pose, dict) and "position" in pose and "orientation" in pose:
        pose = AgentPoseSense(**pose).get_cam_pose().get_T()
    elif isinstance(pose, AgentPoseSense):
        pose = pose.get_cam_pose().get_T()

    if len(infos):
        episode = infos[0]['episode']
    else:
        episode = 0
    if n_instances == 0:

        return SemanticPointCloud(-1, torch.Tensor(), torch.Tensor())
    observation_pcd = SemanticPointCloud(
        episode, torch.Tensor(), torch.Tensor()
    )

    for i in range(n_instances):
        mask = (
            instances[i].gt_masks
            if hasattr(instances, "gt_masks")
            else instances[i].pred_masks
        ).squeeze()

        cls = (
            instances[i].gt_classes
            if hasattr(instances, "gt_classes")
            else instances[i].pred_classes
        )

        kernel = np.ones((7, 7), 'uint8')

        mask = torch.tensor(
            cv2.erode(mask.cpu().numpy().astype('uint8'), kernel, iterations=1),
        )
        if mask.sum() == 0:
            continue

        points = project_instance_segmentation(depth, pose, mask.squeeze()).cpu()

        points_cls = np.ones((points.shape[0], 1)) * cls.cpu().numpy()

        points_obj_ids = np.ones((points.shape[0], 1)) * infos[i]['id_object']
        points_obj_episodes = np.ones((points.shape[0], 1)) * infos[i]['episode']
        points_infos = np.hstack((points_cls, points_obj_episodes, points_obj_ids))
        _pcd = SemanticPointCloud(infos[i]['episode'], points, points_infos)

        observation_pcd = observation_pcd + _pcd

    if update_logits:
        observation_pcd.update_logits(instances, infos)
    return observation_pcd


def project_instance_segmentation(
    depth, pose, mask=None, hfov=79, min_depth=0.5, max_depth=10.0
) -> torch.Tensor:
    # if O3D_AVAILABLE:
    #     return _project_using_open3d(depth, pose, mask, hfov, min_depth, max_depth)
    # else:
    return _project_batch_masks(depth, pose, mask, hfov, min_depth, max_depth)


def _project_using_open3d(
    depth, pose, mask=None, hfov=79, min_depth=0.5, max_depth=5.0
):
    if not isinstance(depth, torch.Tensor):

        depth = torch.tensor(depth)
    if not isinstance(pose, torch.Tensor):
        pose = torch.tensor(pose)
    H = depth.shape[-2]
    W = depth.shape[-1]
    device = pose.device
    fy = W / 2 * np.tan(np.deg2rad(hfov) / 2.0)
    fx = H / 2.0 * np.tan(np.deg2rad(hfov) / 2.0)
    fx = fy

    xc = (W - 1.0) / 2.0
    yc = (H - 1.0) / 2.0

    if mask is not None:
        mask[depth == min_depth] = False
        mask[depth == max_depth] = False

    depth[mask is False] = 0
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(mask.numpy().astype(np.float32)),
        o3d.geometry.Image(depth.numpy().astype(np.float32) * 1000),
    )
    o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, xc, yc)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd, o3d_intrinsics  # , np.linalg.inv(pose)
    )

    return torch.tensor(np.asarray(pcd.points)).to(device)


def _project_batch_masks(
    depth, pose, mask=None, hfov=79, min_depth=0.5, max_depth=15.0
):
    if not isinstance(depth, torch.Tensor):

        depth = torch.tensor(depth)
    if not isinstance(pose, torch.Tensor):
        pose = torch.tensor(pose)

    H = depth.shape[-2]
    W = depth.shape[-1]

    fx = W / 2.0 / np.tan(np.deg2rad(hfov) / 2.0)
    fy = H / 2.0 / np.tan(np.deg2rad(hfov) / 2.0)

    xc = (W - 1.0) / 2.0
    yc = (H - 1.0) / 2.0

    device = pose.device
    if mask is not None:
        mask[depth <= min_depth] = False
        mask[depth >= max_depth] = False
        mask = mask.bool()

    ys, xs = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))

    xs = xs[mask].reshape(1, -1).to(device)
    ys = ys[mask].reshape(1, -1).to(device)
    depth = depth.squeeze()[mask].reshape(1, -1).to(device)

    xyz = (
        torch.cat(
            [
                (xs - xc) / fx * depth,
                -(ys - yc) / fy * depth,
                -depth,
                torch.ones_like(depth),
            ]
        )
        .to(device)
        .double()
    )

    xyz_world = torch.matmul(pose, xyz).permute(1, 0)
    if mask is not None:
        xyz_world = xyz_world[_outlier_removal(depth)]
    return xyz_world[:, :3]


def _outlier_removal(points: torch.Tensor, max_deviations=1):

    mean = torch.mean(points)
    standard_deviation = torch.std(points)

    distance_from_mean = torch.norm(points - mean, dim=0)

    not_outlier = distance_from_mean < max_deviations * standard_deviation
    return not_outlier


class SemanticPointCloud:
    def __init__(self, episode: int = -1, points=None, infos=None, distance_f=None, solution="max"):
        """
        sem_pcd_points: Array of Nx6 elements. Each element is (x,y,z,class,episode,object_id)
        object_id_to_logits (dict): for each object id, list of logits
        """
        if points is None:
            points = torch.Tensor()
        if infos is None:
            infos = torch.Tensor()
        self.points = np.asarray(points).reshape(-1, 3)  #  N x 3
        if distance_f is None:
            distance_f = _cosine_distance
        self.distance_f = distance_f
        self.infos = np.asarray(infos).reshape(
            -1, 3
        )  # N x 3 (class, episode_id, object_id)

        self._init_pcd()
        self.thr = 0.05
        self._episode = episode
        self.scene = None
        self.object_id_to_logits: Dict = dict()

        switch = {
            "seal": inc._seal_impl,
            "bayesian": inc._ours_bayesian,
            "ours": inc._ours_impl,
            "avg": inc._ours_avg,
            "max": inc._ours_max
        }
        self.solve = switch[solution]

        self.current_connected_pcd = np.zeros((150, 150, 150))
        self.current_voxel_ids = np.empty(0)
        self.voxelize_pcd = None
        self.past_unique_voxels = np.empty(0)
        self.voxel_points = None
        self.results = {}

    def __len__(self):
        return len(self.points)

    def _init_pcd(self):
        self._kdtree = None
        self.invalid_index = len(self.points)
        self.classes = self.infos[:, 0].astype(int)
        self.episodes = self.infos[:, 1].astype(int)
        self.object_ids = self.infos[:, 2].astype(int)

    def __add__(self, x: "SemanticPointCloud"):
        assert self._episode == x._episode

        self.points = np.concatenate([self.points, x.points])
        self.infos = np.concatenate([self.infos, x.infos])

        if hasattr(self, "object_ids") and len(self.object_ids):
            # Objects logits are always concatenated. Call preprocess() for solving inconsistencies
            key = max(self.object_ids) + 1
            for old_key in x.object_id_to_logits.keys():
                self.object_id_to_logits[key] = x.object_id_to_logits[old_key]
                self.infos[self.infos[:, -1] == old_key, -1] = key
                key += 1
        elif len(x.object_ids):
            self.object_id_to_logits = x.object_id_to_logits
        self._init_pcd()

        return self

    def update_logits(self, instances: Instances, infos):
        """Receive instances prediction from the model and update the logits for each object_id"""

        for i in range(len(instances)):
            logits = instances[i].gt_logits
            info = infos[i]

            object_id = info['id_object']
            self.object_id_to_logits[
                object_id
            ] = logits  # self.object_id_to_logits.get(
            # object_id, []
            # ) + [logits]

    def save(self, name="semantic_pcd.ply"):

        _pcd = o3d.geometry.PointCloud()

        _pcd.points = o3d.utility.Vector3dVector(self.points)

        _pcd.colors = o3d.utility.Vector3dVector(self.infos)

        o3d.io.write_point_cloud(name, _pcd)

    @staticmethod
    def _get_mesh_from_pcd(_pcd):
        _pcd.estimate_normals()

        _mesh = o3d.geometry.TriangleMesh()

        _mesh.vertices = _pcd.points
        _mesh.vertex_normals = _pcd.normals

        # estimate radius for rolling ball
        distances = _pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = 3 * avg_dist

        _mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            _pcd, o3d.utility.DoubleVector([radius, radius * 2])
        )
        return _mesh

    def preprocess(self, lower_bound=None, upper_bound=None):
        vox_size = self.thr

        if len(self.points) == 0:
            return

        _pcd = PyntCloud(
            pd.DataFrame(
                self.points,
                columns=["x", "y", "z"],
            )
        )
        self._pcd = _pcd
        if lower_bound is None:
            lower_bound = np.asarray(_pcd.points.min(0))

        self.lower_bound = lower_bound

        if upper_bound is None:
            upper_bound = np.asarray(_pcd.points.max(0))

        self.upper_bound = upper_bound

        d_x = int((upper_bound[0] - lower_bound[0]) // vox_size)
        d_y = int((upper_bound[1] - lower_bound[1]) // vox_size)
        d_z = int((upper_bound[2] - lower_bound[2]) // vox_size)

        if (d_x * d_y * d_z) == 0:
            return

        voxelize_pcd = np.zeros((d_x * d_y * d_z))


        voxelgrid_id = _pcd.add_structure(
            "fixedvoxelgrid",
            n_x=d_x,
            n_y=d_y,
            n_z=d_z,
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
        )

        voxelgrid = _pcd.structures[voxelgrid_id]

        x_cords = voxelgrid.voxel_x
        y_cords = voxelgrid.voxel_y
        z_cords = voxelgrid.voxel_z
        voxel_ids = voxelgrid.voxel_n
        self.coords = np.stack([x_cords, y_cords, z_cords], 1)

        consistency_solution = partial(
            inc.solve_inconsistency,  # main inconsistency solution function
            solve_function=self.solve,  # `solve_function` is used in solve_inconsistency
            voxel_ids=voxel_ids,
            object_ids=self.object_ids,
            object_id_to_logits=self.object_id_to_logits,
        )

        # Use consistency_solution only for updated voxels, keep same class,logits tuple for the others
        t = time.time()
        update_voxels = np.setdiff1d(voxel_ids, self.current_voxel_ids)
        self.current_voxel_ids = np.unique(voxel_ids)
        cpu_counts = 20
        n_chunks = len(update_voxels) // cpu_counts
        inputs = [
            update_voxels[
                i * n_chunks : min(i * n_chunks + n_chunks, len(update_voxels))
            ]
            for i in range(cpu_counts)
        ]

        solutions = []

        for i in inputs:
            solutions.append(consistency_solution(i))

        solutions = reduce(add, solutions)

        for sol in solutions:
            self.results[sol[0]] = sol[1]

        for vox_id, result in self.results.items():
            voxelize_pcd[vox_id] = result[0] + 1


        voxelize_pcd = voxelize_pcd.reshape((d_x, d_y, d_z))

        connected_pcd = cc3d.connected_components(voxelize_pcd, connectivity=26)
        self.current_connected_pcd = connected_pcd

        unique_voxels = np.stack(np.where(connected_pcd > 0)).T
        # unique_voxels = np.unique(self.coords, axis=0)
        self.points = voxelgrid.voxel_centers[connected_pcd.flatten() != 0]

        classes = (
            voxelize_pcd[unique_voxels[:, 0], unique_voxels[:, 1], unique_voxels[:, 2]]
            - 1
        )  # taking classes for points after dust

        # taking unique object ids starting from 0 by using "unique(..., return_inverse=True)"
        objects_to_update = np.unique(
            connected_pcd[
                unique_voxels[:, 0], unique_voxels[:, 1], unique_voxels[:, 2]
            ],
        )
        objects = connected_pcd[np.where(connected_pcd > 0)]

        resolved_logits = dict()

        # concatenating logits for each object
        for k in objects_to_update:

            voxels_to_use = np.where(connected_pcd.flatten() == k)[0]
            # Compute voxel ids for object k u sing connected_pcd, from results the ids of the voxels
            if len(voxels_to_use):
                acc_logits = (
                    torch.cat([self.results[i][1] for i in voxels_to_use])
                    .squeeze()
                    .unique(dim=0)
                )  # keep all unique logits for each object

                resolved_logits[k] = acc_logits

        episodes = np.zeros_like(unique_voxels[:, 0])

        self.infos = np.vstack([classes, episodes, objects]).T
        assert len(self.infos) == len(self.points)

        self.object_id_to_logits = resolved_logits

        self.voxelize_pcd = voxelgrid
        self._init_pcd()

    def show(self):
        _pcd = o3d.geometry.PointCloud()

        _pcd.points = o3d.utility.Vector3dVector(self.points)

        _pcd.colors = o3d.utility.Vector3dVector(self.infos)

        o3d.visualization.draw_geometries(
            [_pcd],
            zoom=0.3412,
            front=[0.4257, -0.2125, -0.8795],
            lookat=[2.6172, 2.0475, 1.532],
            up=[-0.0694, -0.9768, 0.2024],
        )

    def save(self, idx="0"):
        _pcd = o3d.geometry.PointCloud()
        _pcd.points = o3d.utility.Vector3dVector(self.points)
        _pcd.colors = o3d.utility.Vector3dVector(self.infos)
        o3d.io.write_point_cloud("pcd_" + str(idx) + ".ply", _pcd)

    def _safe_assign(self, im_map, y_idx, x_idx, value):
        try:

            im_map[y_idx, x_idx] = value
        except IndexError:

            valid_idx1 = np.logical_and(x_idx >= 0, x_idx < im_map.shape[1])
            valid_idx2 = np.logical_and(y_idx >= 0, y_idx < im_map.shape[0])
            valid_idx = np.logical_and(valid_idx1, valid_idx2)

            if valid_idx.sum() == 0:
                return

            try:
                if isinstance(value, np.ndarray):
                    value = value[valid_idx]

                im_map[y_idx[valid_idx], x_idx[valid_idx]] = value
            except (IndexError, ValueError):
                return

    def get_topdown_curiosity(self, lower_bound, upper_bound, map_scale=0.025):
        # Convert to grid coordinate system

        grid_size = (
            int(abs(upper_bound[2] - lower_bound[2]) / map_scale),  # z resolution
            int(abs(upper_bound[0] - lower_bound[0]) / map_scale),  # x resolution
        )

        # Create empty maps for the two channels

        semantic_mat = np.zeros((len(BBSense.CLASSES), *grid_size), np.uint8)
        height_thresh = [2, 5]

        if len(self) and self.points is not None:

            points = self.points
            classes = self.classes
            high_filter_idx = points[:, 1] < height_thresh[1]
            low_filter_idx = points[:, 1] > height_thresh[0]
            obstacle_idx = high_filter_idx

            semantics = classes[obstacle_idx]
            grid_x = (points[:, 0] / map_scale).astype(int) - int(
                np.floor(lower_bound[0] / map_scale)
            )  # columns in numpy

            grid_y = (points[:, 2] / map_scale).astype(int) - int(
                np.ceil(lower_bound[2] / map_scale)  # corresponding to z values
            )  # rows in numpy

            y_idx = grid_y[obstacle_idx]
            x_idx = grid_x[obstacle_idx]
            
            try:
                semantic_mat[semantics, y_idx, x_idx] = 1

            except IndexError:
                
                valid_idx1 = np.logical_and(x_idx >= 0, x_idx < semantic_mat.shape[1])
                valid_idx2 = np.logical_and(y_idx >= 0, y_idx < semantic_mat.shape[0])
                valid_idx = np.logical_and(valid_idx1, valid_idx2)

                if valid_idx.sum() == 0:
                    return semantic_mat

                semantics = semantics[valid_idx]
                semantic_mat[semantics, y_idx[valid_idx], x_idx[valid_idx]] = 1

        return semantic_mat

    def get_topdown_semantic(self, lower_bound, upper_bound, map_scale=0.025):
        # Convert to grid coordinate system

        grid_size = (
            int(abs(upper_bound[2] - lower_bound[2]) / map_scale),  # z resolution
            int(abs(upper_bound[0] - lower_bound[0]) / map_scale),  # x resolution
        )

        # Create empty maps for the two channels
        obstacle_mat = np.zeros(grid_size, np.uint8)
        explore_mat = np.zeros(grid_size, np.uint8)
        disagreement_mat = np.zeros(grid_size, np.float32)
        semantic_mat = np.zeros(grid_size, np.float32)
        height_thresh = [2, 5]

        if len(self) and self.points is not None:

            points = self.points
            object_ids = self.object_ids
            classes = self.classes

            grid_x = (points[:, 0] / map_scale).astype(int) - int(
                np.floor(lower_bound[0] / map_scale)
            )  # columns in numpy
            grid_y = (points[:, 2] / map_scale).astype(int) - int(
                np.ceil(lower_bound[2] / map_scale)  # corresponding to z values
            )  # rows in numpy

            high_filter_idx = points[:, 1] < height_thresh[1]
            low_filter_idx = points[:, 1] > height_thresh[0]
            obstacle_idx = np.logical_and(low_filter_idx, high_filter_idx)

            self._safe_assign(
                obstacle_mat, grid_y[obstacle_idx], grid_x[obstacle_idx], 1
            )

            kernel = np.ones((3, 3), np.uint8)
            obstacle_mat = cv2.dilate(obstacle_mat, kernel, iterations=1)
            # Compute explored locations
            explored_idx = high_filter_idx
            self._safe_assign(
                explore_mat, grid_y[explored_idx], grid_x[explored_idx], 1
            )
            kernel = np.ones((3, 3), np.uint8)
            obstacle_mat = cv2.dilate(obstacle_mat, kernel, iterations=1)

            semantics = classes[explored_idx]
            self._safe_assign(
                semantic_mat, grid_y[explored_idx], grid_x[explored_idx], semantics
            )
            # Compute explored locations
            explored_idx = high_filter_idx
            self._safe_assign(
                explore_mat, grid_y[explored_idx], grid_x[explored_idx], 1
            )
            # Disagreement mat

            cosine_distances = {
                k: self.distance_f(self.object_id_to_logits[k])
                for k in self.object_id_to_logits.keys()
            }
            cosine_distance_per_point = [cosine_distances[k] for k in object_ids]

            if len(cosine_distance_per_point):
                cosine_distance_per_point = torch.stack(cosine_distance_per_point)

                valid_points_for_disagreement = cosine_distance_per_point > 0

                self._safe_assign(
                    disagreement_mat,
                    grid_y[valid_points_for_disagreement],
                    grid_x[valid_points_for_disagreement],
                    cosine_distance_per_point[valid_points_for_disagreement],
                )

                # inflate current cell (useful is self.thr > 0.025)
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if i != j:
                            self._safe_assign(
                                disagreement_mat,
                                grid_y[valid_points_for_disagreement] + i,
                                grid_x[valid_points_for_disagreement] + j,
                                cosine_distance_per_point[
                                    valid_points_for_disagreement
                                ],
                            )

            # Smoothen the maps
            kernel = np.ones((3, 3), np.uint8)

            obstacle_mat = cv2.morphologyEx(obstacle_mat, cv2.MORPH_CLOSE, kernel)
            explore_mat = cv2.morphologyEx(explore_mat, cv2.MORPH_CLOSE, kernel)

            # Ensure all expanded regions in obstacle_mat are accounted for in explored_mat
            explore_mat = np.logical_or(explore_mat, obstacle_mat)

        return np.stack(
            [obstacle_mat, explore_mat, semantic_mat, disagreement_mat], axis=2
        )

    def raytracing(self, depth, pose):
        # if O3D_AVAILABLE:

        #     return self._open3d_raytracing(depth, pose)
        # else:
        return self._depth_raytracing(depth, pose)

    def _get_kl(self,
                depth: torch.Tensor,
                pose: Union[torch.Tensor, AgentPoseSense, Dict],
                instances: Instances,
                infos):
        
        kdtree = KDTree(self.points)

        projected_points = project_semantic_masks_to_3d(
                depth,
                pose,
                instances,
                infos,
            )

        kdtree_thr = 2 * self.thr  # + self.thr // 2
        
        founds = kdtree.query(
            projected_points.points, distance_upper_bound=kdtree_thr, k=1, workers=-1
        )

        keep = torch.tensor(founds[1] != self.invalid_index)
        
        pred_ids = projected_points.object_ids[keep]
        target_ids = self.object_ids[founds[1][keep]]
        seen = set()
        kls = []
        for target_id, pred_id in zip(target_ids, pred_ids):
            if target_id in seen:
                continue
            seen.add(target_id)
            target_logits = self.object_id_to_logits[target_id]
            pred_logits = projected_points.object_id_to_logits[pred_id]

            kl = F.kl_div(pred_logits, target_logits)
            kls.append(kl)
        return np.array(kls)



    def _depth_raytracing(self, depth, pose):
        # def pnt2img_projection(pnts, cam_mat, resolution):
        if self._kdtree is None:
            self.kdtree = KDTree(self.points)
        if len(self.points) == 0:
            log.warning(
                "Semantic Point Cloud for episode {self._episode} has no points"
            )
            return (
                torch.Tensor(),
                torch.Tensor(),
                torch.Tensor(),
                torch.Tensor(),
                torch.Tensor(),
            )
        kdtree_thr = 2 * self.thr  # + self.thr // 2

        projected_points = _project_batch_masks(depth, pose).numpy()

        founds = self.kdtree.query(
            projected_points, distance_upper_bound=kdtree_thr, k=1, workers=-1
        )

        keep = torch.tensor(founds[1] != self.invalid_index)

        object_ids = self.object_ids[founds[1][keep]]

        outs = []

        resolved_ids = []
        resolved_classes = []
        resolved_logits = []

        for _, obj_id in enumerate(np.unique(object_ids)):

            segm = object_ids == obj_id
            mask = np.zeros_like(keep)


            mask[keep] = segm

            cls_id = int(self.classes[founds[1][mask]][0])
            if mask.sum() < 100:
                continue
            
            mask = mask.reshape(*depth.shape)  # .astype('uint8')
            if len(self.object_id_to_logits[obj_id]) == 0:
                continue
            try:
                logits = self.solve(self.object_id_to_logits[obj_id])[1]
            except:
                log.warning(self.object_id_to_logits[obj_id])

                continue

            outs.append(torch.tensor(mask, dtype=torch.bool))
            resolved_classes.append(cls_id)
            resolved_ids.append(obj_id)
            resolved_logits.append(logits.cpu())

        if len(outs) == 0:
            return (
                torch.Tensor(),
                torch.Tensor(),
                torch.Tensor(),
                torch.Tensor(),
                torch.Tensor(),
            )
        out = torch.stack(outs)

        return (
            out,
            torch.tensor(resolved_ids, dtype=torch.int),
            torch.tensor(resolved_classes, dtype=torch.int),
            torch.cat(resolved_logits),
            []
        )

 
def _cosine_distance(mat):
    """

    inputs: [x, y]
    mid:
    | d(x, x), d(x, y)|
    | d(y, y), d(y, x)|

    output: mid.mean()

    """
    if len(mat) == 1:
        return torch.tensor(0)
    norm = (mat * mat).sum(0, keepdims=True) ** 0.5
    return (1 - (mat.T @ mat) / norm / norm.T).mean()
