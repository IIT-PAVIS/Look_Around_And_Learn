from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torchmetrics
from detectron2.structures import pairwise_ioa, pairwise_iou
from detectron2.structures.instances import Instances
from scipy.spatial import KDTree
from sklearn import metrics
from sklearn.cluster import DBSCAN
from torch.nn import functional as f

from experimenting_env.utils.triplet import _pairwise_distances


def get_objects_ids(batch, predictions: List[Instances], overlap_thr=0.5):
    """
    batch: dictionary containing field "instances" for ground-truth instances and "episode" for current episode
    predictions: Instances object containing model prediction
    """

    gt_labels = []

    # compute y_gt with unique ids starting from 0

    for idx in range(len(predictions)):

        pred = predictions[idx]
        gt = batch[idx]['instances'].to('cpu')

        # Get instance ids for objects
        ids = _get_objects_unique_ids_impl(
            pred,
            gt,
            device='cpu',
            thr=overlap_thr,
            episode=batch[idx]['episode'],
        )

        gt_labels.append(ids)

    return gt_labels


def get_wasserstein_labels(centroids, covs, thr, device='cpu'):

    location_distance = _pairwise_distances(centroids, squared=True, device=device)
    n = len(centroids)
    size_distances = _pairwise_distances(
        covs.reshape(n, -1), squared=True, device=device
    )
    distances = location_distance + size_distances
    labels = DBSCAN(eps=thr, min_samples=2).fit(distances.cpu()).labels_

    return torch.tensor(labels)


def get_centroids_labels_grid(centroids: torch.Tensor, infos, thr=4.0, device='cpu'):
    if infos is not None:
        centroids = torch.hstack([centroids, infos.unsqueeze(1)])

    from torch_cluster import grid_cluster

    labels = grid_cluster(
        centroids.double(), torch.tensor([thr, thr, thr], dtype=torch.double).to(device)
    )

    return torch.tensor(labels).to(device)


def get_centroids_labels_dbscan(centroids: torch.Tensor, infos, thr=4.0, device='cpu'):
    if infos is not None:
        centroids = torch.hstack([centroids, infos.unsqueeze(1)])

    location_distance = _pairwise_distances(
        centroids, squared=False, device=device
    ).cpu()

    labels = DBSCAN(eps=thr, min_samples=2).fit(location_distance).labels_

    return torch.tensor(labels).to(device)


def get_objects_using_sm(predictions, gts, device="cuda", thr=0.3):
    """

    Given N predictions and M ground-truth, returns list of length N of unique instance ids given for each prediction. If no ground-truth / prediction matching occurs, id is -1

    """
    results = []

    for pred_idx in range(len(predictions)):
        ious = []
        pred = predictions[pred_idx]
        for idx in range(len(gts)):
            gt = gts[idx]
            ious.append(torchmetrics.IoU(2)(pred.pred_masks.int(), gt.gt_masks.int()))

        max_iou_id = np.argmax(np.array(ious))

        if ious[max_iou_id] > thr:
            id_obj = gts.infos[max_iou_id]
            results.append(id_obj)
        else:
            results.append({"id_object": -1, "episode": -1})

    return results


def _get_objects_ids_impl(predictions, gt, device="cuda", thr=0.3, episode=-1):
    """
    Given N predictions and M ground-truth, returns list of length N of unique instance ids given for each prediction. If no ground-truth / prediction matching occurs, id is -1
    """
    if not hasattr(get_objects_ids, "current_unique_id"):
        get_objects_ids.current_unique_id = 500
    results = []
    # Use masks IOU for matching gt and preds

    if len(gt) < 1:
        dummy_ids = []
        for _ in range(len(predictions)):
            dummy_ids.append(
                {"id_object": get_objects_ids.current_unique_id, "episode": episode}
            )
            get_objects_ids.current_unique_id += 1
        return dummy_ids

    for idx_pred in range(len(predictions)):

        gt_boxes = gt.gt_boxes if hasattr(gt, "gt_boxes") else gt.pred_boxes
        pred_boxes = (
            predictions[idx_pred].pred_boxes
            if hasattr(predictions[idx_pred], "pred_boxes")
            else predictions[idx_pred].gt_boxes
        )
        iou = pairwise_iou(gt_boxes.to(device), pred_boxes.to(device))

        max_iou = torch.argmax(iou)

        if iou[max_iou] > thr:
            id_obj = gt.infos[max_iou]
            if "episode" not in id_obj:
                id_obj['episode'] = episode
            results.append(id_obj)
        else:

            results.append(
                {"id_object": get_objects_ids.current_unique_id, "episode": episode}
            )
            get_objects_ids.current_unique_id += 1

    return results


def _get_objects_unique_ids_impl(predictions, gt, device="cuda", thr=0.3, episode=-1):
    """
    Given N predictions and M ground-truth, returns list of length N of unique instance ids given for each prediction. If no ground-truth / prediction matching occurs, id is -1
    """
    if not hasattr(get_objects_ids, "current_unique_id"):
        get_objects_ids.current_unique_id = 5000000
    results = []
    # Use masks IOU for matching gt and preds

    dummy_ids = []
    for _ in range(len(predictions)):
        dummy_ids.append(
            {"id_object": get_objects_ids.current_unique_id, "episode": episode}
        )
        get_objects_ids.current_unique_id += 1
    return dummy_ids


def get_objects_ids_and_centroids(predictions, gt, device="cuda", thr=0.3):
    """

    Given N predictions and M ground-truth, returns list of length N of unique instance ids given for each prediction. If no ground-truth / prediction matching occurs, id is -1

    """
    results = []

    for idx_pred in range(len(predictions)):
        iou = pairwise_ioa(
            gt.gt_boxes.to(device), predictions[idx_pred].pred_boxes.to(device)
        )

        max_iou = torch.argmax(iou)

        if iou[max_iou] > thr:

            results.append(gt.infos[max_iou])
        else:
            results.append(-1)

    return results
