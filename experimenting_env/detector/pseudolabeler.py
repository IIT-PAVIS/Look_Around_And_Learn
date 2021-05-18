"""
Implemented consensus mechanism for solving pseudo-labels
"""
import functools
import gc
import itertools as it
import logging
import time
from copy import deepcopy
from multiprocessing.pool import Pool
from typing import List

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import tqdm
from detectron2 import model_zoo
from detectron2.structures import BitMasks, Boxes, Instances, pairwise_iou
from omegaconf import DictConfig
from torch.nn import functional as F
from torchmetrics.detection.map import MAP


from tqdm.contrib.concurrent import process_map

from experimenting_env.detector.model.multi_stage_models import MultiStageModel
from experimenting_env.sensor_data import BBSense
from experimenting_env.utils import projection_utils as pu
from experimenting_env.utils.matching import get_objects_ids
from experimenting_env.utils.predictor_utils import Predictor
from experimenting_env.utils.roi_head_wrappers import (
    BoxPredictorWrapper,
    SoftHeadWrapper,
)

log = logging.getLogger(__name__)


class ConsensusLabeler(pl.LightningModule):
    def __init__(self, model=None, thr=0.7, overlap_thr=0.5, *args, **kwargs):
        super().__init__()
        self.model = None
        self.thr = thr
        self.reinit(model)

        self.overlap_thr = overlap_thr

        
    def reinit(self, model=None):
        if model is None:
            model = Predictor()
            breakpoint()
            model.reinit_head(BBSense.CLASSES)
            model.set_head_wrapper(BoxPredictorWrapper)

        self.update_model(model)
        self.test_map_metric = MAP(class_metrics=True)

    def update_model(self, model):
        self.model = model
        self.model.model.roi_heads.box_predictor.box_predictor.test_score_thresh = (
            self.thr
        )
        self.model.eval()

    def save_model(self, path):

        torch.save(self.model.state_dict(), path)

    def validation_step(self, batch, batch_idx):
        losses, predictions = self.model._common_step(
            batch,
        )

        return losses, predictions

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        self.model.eval()
        instances = self(batch)

        return instances

    def forward(self, batch):
        self.model.eval()
        preds = [x['instances'].to('cpu') for x in self.model(batch)[0]]
        ids = get_objects_ids(batch, preds)

        return preds, ids

    def get_pseudo_labels(self, *args, **kwargs):
        pass


class VanillaConsensusLabeler(ConsensusLabeler):
    def __init__(self, temperature=1, model=None, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.temperature = temperature

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        self.model.eval()
        instances = self(batch)

        return instances

    def get_pseudo_labels(self, model_outs, *args, **kwargs):
        """
        Returns predictions as pseudo ground-truth
        """
        result = []
        for out in model_outs:
            for pred, infos in zip(out[0], out[1]):

                mask = pred.scores > 0.0001
                pred = pred[mask]
                target = Instances(len(pred))
                target.gt_classes = pred.pred_classes
                target.gt_boxes = pred.pred_boxes
                if len(pred) > 0:
                    target.gt_logits = pred.gt_logits
                else:
                    target.gt_logits = torch.Tensor()

                target.scores = pred.scores
                target.gt_masks = pred.pred_masks

                target.infos = [info for idx, info in enumerate(infos) if mask[idx]]
                result.append(target)
        return result


class SemanticMapConsensusLabeler(ConsensusLabeler):
    def __init__(
        self, thr=0.7, temperature=1.0, model=None, solution="ours", *args, **kwargs
    ):
        super().__init__(model=model, thr=thr)
        self.temperature = temperature
        self.solution = solution

        
    def reinit(self, model):
        super().reinit(model)
        self.global_pcds = {}

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        self.model.eval()
        instances, infos = self(batch)

        for b, prediction, info in zip(batch, instances, infos):
            episode = b['episode']
            if episode in self.global_pcds:

                episode_pcd = self.global_pcds[episode]
            else:
                episode_pcd = pu.SemanticPointCloud(
                    episode=episode, solution=self.solution
                )
                self.global_pcds[episode] = episode_pcd

            _pcd = pu.project_semantic_masks_to_3d(
                b['depth'].squeeze(0),
                b['location'],
                prediction.to(b['depth'].device),
                info,
                update_logits=False # do not update logits locally, only per episode
            )
            _pcd._episode = episode

            episode_pcd += _pcd

            if len(_pcd):
                episode_pcd.update_logits(prediction, info)
             #     # episode_pcd.preprocess()
        return instances, infos

    def get_pseudo_labels(self, model_outs, dataloader):
        """
        Returns predictions as pseudo ground-truth
        """

        labels = []

        for k in self.global_pcds.keys():
            self.global_pcds[k].preprocess()

        for batch in tqdm.tqdm(dataloader):

            for data in batch:
                pcd = self.global_pcds[data['episode']]

                # Compute the ray intersections.
                _time = time.time()
                
                (
                    semantic_masks,
                    object_ids,
                    classes,
                    r_logits,
                    _
                ) = pcd._depth_raytracing(data['depth'].squeeze(), data['location'])

                log.info(f"It took {time.time() - _time} for raytracing")
                t = Instances(image_size=data['depth'].squeeze().shape)

                # Get bbs from semantic
                resolved_masks = []
                bounding_boxes = []
                logits = []
                resolved_classes = []
                ids = []

                for mask, object_id, cls, l in zip(
                    semantic_masks, object_ids, classes, r_logits
                ):
                    bb = cv2.boundingRect(mask.numpy().astype('uint8'))

                    x, y, w, h = bb
                    if w == 0 or h == 0:
                        continue
                    if cls >= len(BBSense.CLASSES):
                        continue  # Background or overflowd class
                    logits.append(l)

                    resolved_class = cls

                    ids.append(object_id)
                    bounding_boxes.append(
                        torch.tensor([x, y, x + w, y + h]).unsqueeze(0)
                    )
                    resolved_classes.append(resolved_class)

                    resolved_masks.append(mask)

                t.gt_classes = (
                    torch.tensor(resolved_classes)
                    if len(resolved_classes) > 0
                    else torch.Tensor()
                )

                t.gt_masks = (
                    torch.stack(resolved_masks)
                    if len(resolved_masks)
                    else torch.Tensor()
                )
                # TODO uncertainty information per bbox

                t.gt_logits = torch.stack(logits) if len(logits) > 0 else torch.Tensor()

                t.infos = [{'id_object': id.item()} for id in ids]

                t.gt_boxes = (
                    Boxes(torch.cat(bounding_boxes))
                    if len(bounding_boxes)
                    else Boxes(torch.Tensor())
                )
                labels.append(t)
        gc.collect()
        return labels


class SoftConsensusLabeler(ConsensusLabeler):
    def __init__(self, temperature=1, model=None):
        super().__init__(model)
        self.temperature = temperature

    def get_pseudo_labels(self, model_outs, *args, **kwargs):
        """
        Returns predictions as pseudo ground-truth
        """

        predictions = list(it.chain(*[m[0] for m in model_outs]))

        y_ids = list(it.chain(*list(it.chain(*[m[1] for m in model_outs]))))

        max_id = max([m['id_object'] for m in y_ids])

        y_matching = torch.tensor(
            [
                m['id_object'] + m['episode'] * max_id if m['id_object'] > 0 else -1
                for m in y_ids
            ]
        )

        preds_logits = torch.cat([pred.gt_logits for pred in predictions])

        match_ids = torch.unique(y_matching)

        logits_per_instance = {
            m.item(): preds_logits[y_matching == m] for m in match_ids
        }

        gt_instances = []

        pred_counting = 0

        for idx in range(len(predictions)):

            preds_per_image = predictions[idx]

            if len(preds_per_image) == 0:
                target = Instances(preds_per_image.image_size)
                target.gt_boxes = preds_per_image.pred_boxes
                target.gt_classes = preds_per_image.pred_classes
                target.gt_logits = preds_per_image.gt_logits
                target.gt_masks = preds_per_image.pred_masks
                target.scores = preds_per_image.scores
                target.infos = []

                gt_instances.append(target)
                continue
            resolved_classes = []
            gt_logits = []
            gt_ids = []
            mask = torch.ones(len(preds_per_image), dtype=torch.bool)
            for pred_id in range(len(preds_per_image)):
                p = pred_id + pred_counting

                y = y_matching[p].item()

                logits = logits_per_instance[y]
                breakpoint()
                soft_softmax = F.softmax(logits / self.temperature, -1).mean(0)
                resolved_class = torch.argmax(soft_softmax[:-1])
                score = soft_softmax.max()
                if score < 0.001:
                    mask[pred_id] = False
                else:
                    resolved_classes.append(resolved_class)
                    gt_logits.append(soft_softmax)
                    gt_ids.append({'id_object': y})

            pred_counting += len(preds_per_image)
            target = Instances(preds_per_image[mask].image_size)
            target.gt_boxes = preds_per_image[mask].pred_boxes
            target.gt_classes = (
                torch.tensor(resolved_classes)
                if len(resolved_classes) > 0
                else torch.Tensor()
            )
            target.gt_masks = preds_per_image[mask].pred_masks
            target.gt_logits = (
                torch.stack(gt_logits) if len(gt_logits) > 0 else torch.Tensor()
            )

            target.infos = gt_ids

            gt_instances.append(target)

        return gt_instances


class MajorityConsensusLabeler(ConsensusLabeler):
    def __init__(self, temperature=1, model=None):
        super().__init__(model)
        self.temperature = temperature

    def get_pseudo_labels(self, model_outs, *args, **kwargs):
        """
        Returns predictions as pseudo ground-truth
        """

        predictions = list(it.chain(*[m[0] for m in model_outs]))

        y_ids = list(it.chain(*list(it.chain(*[m[1] for m in model_outs]))))

        max_id = max([m['id_object'] for m in y_ids])

        y_matching = torch.tensor(
            [
                m['id_object'] + m['episode'] * max_id if m['id_object'] > 0 else -1
                for m in y_ids
            ]
        )

        preds_logits = torch.cat([pred.gt_logits for pred in predictions])

        match_ids = torch.unique(y_matching)

        logits_per_instance = {
            m.item(): preds_logits[y_matching == m] for m in match_ids
        }

        gt_instances = []

        pred_counting = 0

        for idx in range(len(predictions)):

            preds_per_image = predictions[idx]

            if len(preds_per_image) == 0:
                target = Instances(preds_per_image.image_size)
                target.gt_boxes = preds_per_image.pred_boxes
                target.gt_classes = preds_per_image.pred_classes
                target.gt_logits = preds_per_image.gt_logits
                target.gt_masks = preds_per_image.pred_masks
                target.scores = preds_per_image.scores
                target.infos = []

                gt_instances.append(target)
                continue
            resolved_classes = []
            gt_logits = []
            gt_ids = []
            mask = torch.ones(len(preds_per_image), dtype=torch.bool)
            for pred_id in range(len(preds_per_image)):
                p = pred_id + pred_counting

                y = y_matching[p].item()

                logits = logits_per_instance[y]

                soft_softmax = F.softmax(logits / self.temperature, -1).mean(0)
                classes, counts = logits.argmax(1).unique(return_counts=True)
                resolved_class = classes[counts.argmax()]
                score = soft_softmax.max()
                if score < 0.001:
                    mask[pred_id] = False
                else:
                    resolved_classes.append(resolved_class)
                    gt_logits.append(soft_softmax)
                    gt_ids.append({'id_object': y})

            pred_counting += len(preds_per_image)
            target = Instances(preds_per_image[mask].image_size)
            target.gt_boxes = preds_per_image[mask].pred_boxes
            target.gt_classes = (
                torch.tensor(resolved_classes)
                if len(resolved_classes) > 0
                else torch.Tensor()
            )
            target.gt_masks = preds_per_image[mask].pred_masks
            target.gt_logits = (
                torch.stack(gt_logits) if len(gt_logits) > 0 else torch.Tensor()
            )

            target.infos = gt_ids

            gt_instances.append(target)

        return gt_instances


class MaxScoreConsensusLabeler(ConsensusLabeler):
    def __init__(self, temperature=1, model=None):
        super().__init__(model)
        self.temperature = temperature

    def get_pseudo_labels(self, model_outs, *args, **kwargs):
        """
        Returns predictions as pseudo ground-truth
        """

        predictions = list(it.chain(*[m[0] for m in model_outs]))

        y_ids = list(it.chain(*list(it.chain(*[m[1] for m in model_outs]))))

        max_id = max([m['id_object'] for m in y_ids])

        y_matching = torch.tensor(
            [
                m['id_object'] + m['episode'] * max_id if m['id_object'] > 0 else -1
                for m in y_ids
            ]
        )

        preds_logits = torch.cat([pred.gt_logits for pred in predictions])

        match_ids = torch.unique(y_matching)

        logits_per_instance = {
            m.item(): preds_logits[y_matching == m] for m in match_ids
        }

        gt_instances = []

        pred_counting = 0

        for idx in range(len(predictions)):

            preds_per_image = predictions[idx]

            if len(preds_per_image) == 0:
                target = Instances(preds_per_image.image_size)
                target.gt_boxes = preds_per_image.pred_boxes
                target.gt_classes = preds_per_image.pred_classes
                target.gt_logits = preds_per_image.gt_logits
                target.gt_masks = preds_per_image.pred_masks
                target.scores = preds_per_image.scores
                target.infos = []

                gt_instances.append(target)
                continue
            resolved_classes = []
            gt_logits = []
            gt_ids = []
            mask = torch.ones(len(preds_per_image), dtype=torch.bool)
            for pred_id in range(len(preds_per_image)):
                p = pred_id + pred_counting

                y = y_matching[p].item()

                logits = logits_per_instance[y]

                soft_softmax = F.softmax(logits / self.temperature, -1).mean(0)

                values, indexes = F.softmax(logits).max(1)
                resolved_class = indexes[values.argmax()]

                score = soft_softmax.max()
                if score < 0.001:
                    mask[pred_id] = False
                else:
                    resolved_classes.append(resolved_class)
                    gt_logits.append(soft_softmax)
                    gt_ids.append({'id_object': y})

            pred_counting += len(preds_per_image)
            target = Instances(preds_per_image[mask].image_size)
            target.gt_boxes = preds_per_image[mask].pred_boxes
            target.gt_classes = (
                torch.tensor(resolved_classes)
                if len(resolved_classes) > 0
                else torch.Tensor()
            )
            target.gt_masks = preds_per_image[mask].pred_masks
            target.gt_logits = (
                torch.stack(gt_logits) if len(gt_logits) > 0 else torch.Tensor()
            )

            target.infos = gt_ids

            gt_instances.append(target)

        return gt_instances


def closure(pcd):
    pcd[1].preprocess()
    return pcd
