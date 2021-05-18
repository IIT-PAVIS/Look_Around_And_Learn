import os
from typing import List, Tuple

import fvcore.nn.weight_init as weight_init
import hydra
import numpy as np
import pycocotools.mask as mask_util
import torch
import torchvision
from detectron2 import model_zoo
from detectron2.config import configurable, get_cfg
from detectron2.layers import (
    Conv2d,
    ConvTranspose2d,
    ShapeSpec,
    batched_nms,
    cat,
    get_norm,
)
from detectron2.modeling import StandardROIHeads, build_box_head
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import (
    ROI_HEADS_REGISTRY,
    FastRCNNOutputLayers,
)
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.structures.boxes import BoxMode
from detectron2.structures.masks import BitMasks
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from torch import nn
from torch.nn import functional as F


def enable_dropout(model):
    """Function to enable the dropout layers during test-time"""
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def get_mc_faster_rcnn_config(dataset_name, pretrained_weights_path=None):

    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.DATASETS.TRAIN = (dataset_name,)
    cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
    cfg.DATALOADER.NUM_WORKERS = 16
    if pretrained_weights_path is not None:
        cfg.MODEL.WEIGHTS = pretrained_weights_path
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.002
    cfg.SOLVER.MAX_ITER = 100000
    cfg.MODEL.ROI_HEADS.NAME = "MCStandardROIHeads"
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
        128  # faster, and good enough for this toy dataset
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80
    return cfg


def fast_rcnn_inference(
    boxes,
    scores,
    image_shapes,
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
    logits=None,
):
    """
    Call `fast_rcnn_inference_single_image` for all images.

    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`FastRCNNOutputLayers.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FastRCNNOutputLayers.predict_probs`.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.

    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections.
        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/scores index in [0, Ri) from the input, for image i.
    """
    result_per_image = [
        fast_rcnn_inference_single_image(
            boxes_per_image,
            scores_per_image,
            image_shape,
            score_thresh,
            nms_thresh,
            topk_per_image,
            logits_per_image,
        )
        for scores_per_image, boxes_per_image, image_shape, logits_per_image in zip(
            scores, boxes, image_shapes, logits
        )
    ]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]


def fast_rcnn_inference_single_image(
    boxes,
    scores,
    image_shape: Tuple[int, int],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
    logits: None,
):
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]

    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # save the scores for later
    scores_all = scores

    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]

    # 2. Apply NMS for each class independently.
    keep = torchvision.ops.nms(boxes, scores, nms_thresh)
    # keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)

    if topk_per_image >= 0:
        keep = keep[:topk_per_image]

    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]

    filter_mask = filter_inds[:, 0]
    scores_all = scores_all[filter_mask]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    # result.scores = scores # original code
    if logits is not None:
        result.gt_logits = logits[filter_mask]

    result.scores = scores
    result.pred_classes = scores_all.max(1).indices  # filter_inds[:, 1]
    return result, filter_inds[:, 0]


def get_coco_item_dict(labels):
    instances = []

    for index, y in enumerate(labels):
        class_labels = y.gt_classes

        annotations = [
            {
                'bbox': y[id_instance].gt_boxes.tensor[0].tolist(),
                'bbox_mode': BoxMode.XYXY_ABS,
                'category_id': class_labels[id_instance],
                'segmentation': mask_util.encode(
                    np.asfortranarray(y.gt_masks[id_instance])
                ),
                # TODO introduce 'uncertainties': y[id_instance].gt_uncertainty_masks[0],
                'iscrowd': 0,
                'infos': y[id_instance].infos[0],
                'gt_logits': y[id_instance].gt_logits[0],
            }
            for id_instance in range(len(y))
        ]

        instances.append(annotations)

    return instances


@torch.jit.unused
def weighted_mask_rcnn_loss(
    pred_mask_logits: torch.Tensor, instances: List[Instances], vis_period: int = 0
):
    """
    Compute the mask prediction loss defined in the Mask R-CNN paper.

    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1
            correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.
        vis_period (int): the period (in steps) to dump visualization.

    Returns:
        mask_loss (Tensor): A scalar tensor containing the loss.
    """

    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    total_num_masks = pred_mask_logits.size(0)
    mask_side_len = pred_mask_logits.size(2)
    assert pred_mask_logits.size(2) == pred_mask_logits.size(
        3
    ), "Mask prediction must be square!"

    gt_classes = []
    gt_masks = []
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        if not cls_agnostic_mask:
            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
            gt_classes.append(gt_classes_per_image)

        gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
            instances_per_image.proposal_boxes.tensor, mask_side_len
        ).to(device=pred_mask_logits.device)
        # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
        gt_masks.append(gt_masks_per_image)

    if len(gt_masks) == 0:
        return pred_mask_logits.sum() * 0

    gt_masks = cat(gt_masks, dim=0)

    if cls_agnostic_mask:
        pred_mask_logits = pred_mask_logits[:, 0]
    else:
        indices = torch.arange(total_num_masks)
        gt_classes = cat(gt_classes, dim=0)
        pred_mask_logits = pred_mask_logits[indices, gt_classes]

    if gt_masks.dtype == torch.bool:
        gt_masks_bool = gt_masks
    else:
        # Here we allow gt_masks to be float as well (depend on the implementation of rasterize())
        gt_masks_bool = gt_masks > 0.5
    gt_masks = gt_masks.to(dtype=torch.float32)

    # Log the training accuracy (using gt classes and 0.5 threshold)
    mask_incorrect = (pred_mask_logits > 0.0) != gt_masks_bool
    mask_accuracy = 1 - (mask_incorrect.sum().item() / max(mask_incorrect.numel(), 1.0))
    num_positive = gt_masks_bool.sum().item()
    false_positive = (mask_incorrect & ~gt_masks_bool).sum().item() / max(
        gt_masks_bool.numel() - num_positive, 1.0
    )
    false_negative = (mask_incorrect & gt_masks_bool).sum().item() / max(
        num_positive, 1.0
    )

    storage = get_event_storage()
    storage.put_scalar("mask_rcnn/accuracy", mask_accuracy)
    storage.put_scalar("mask_rcnn/false_positive", false_positive)
    storage.put_scalar("mask_rcnn/false_negative", false_negative)
    if vis_period > 0 and storage.iter % vis_period == 0:
        pred_masks = pred_mask_logits.sigmoid()
        vis_masks = torch.cat([pred_masks, gt_masks], axis=2)
        name = "Left: mask prediction;   Right: mask GT"
        for idx, vis_mask in enumerate(vis_masks):
            vis_mask = torch.stack([vis_mask] * 3, axis=0)
            storage.put_image(name + f" ({idx})", vis_mask)

    breakpoint()
    mask_loss = F.binary_cross_entropy_with_logits(
        pred_mask_logits, gt_masks, reduction="mean"
    )
    return mask_loss
