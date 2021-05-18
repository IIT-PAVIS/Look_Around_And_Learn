import logging
from copy import copy, deepcopy
from typing import List

import torch
from detectron2.data import DatasetCatalog, MetadataCatalog
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
import wandb
from experimenting_env.sensor_data import BBSense
from experimenting_env.utils import triplet
from experimenting_env.utils.predictor_utils import Predictor, setup_cfg
from experimenting_env.utils.roi_head_wrappers import (
    BoxPredictorWrapper,
    DetachedPredictorWrapper,
    DropoutPredictorWrapper,
    FocalHeadWrapper,
    MSEFocalHeadWrapper,
    SoftDropoutPredictorWrapper,
    SoftFocalHeadWrapper,
    SoftHeadWrapper,
)

__ALL__ = ['MultiStageModel']
log = logging.getLogger(__name__)


class MultiStageModel(Predictor):
    def __init__(
        self,
        cfg=None,
        lr=0.01,
        loss_weights={},
        use_gt_matching=True,
        optimizer="SGD",
        optimizer_params={},
        prune=True,
        compute_loss=True,
        loss_margin=0.3,
        head_cls=BoxPredictorWrapper,
        mask_on=True,
        load_checkpoint=True,
        *args,
        **kwargs,
    ):

        cfg = setup_cfg(cfg).clone()

        if not prune:
            labels = BBSense.CLASSES
            cfg['MODEL']['ROI_HEADS']['NUM_CLASSES'] = len(labels)
            cfg['MODEL']['ROI_BOX_HEAD']['BBOX_REG_LOSS_TYPE'] = 'giou'

            cfg.DATASETS.TEST = ["habitat_dataset"]

            if "habitat_dataset" not in DatasetCatalog.data:
                DatasetCatalog.register("habitat_dataset", lambda x: x)
            MetadataCatalog.get("habitat_dataset").thing_classes = list(labels.values())
        super().__init__(setup_cfg(cfg), load_checkpoint=load_checkpoint)

        if prune:  # Prune classifier instead of initializing a new one
            self.reinit_head(BBSense.CLASSES)
        self.set_head_wrapper(head_cls)
        self.lr = lr

        self.loss_weights = loss_weights
        self.use_gt_matching = use_gt_matching
        self.loss_margin = loss_margin
        self.optimizer = optimizer
        self.opt_params = optimizer_params  # Params for optimizer

        self.feature_projector = triplet.tinyprojection_MLP(1024, out_dim=128)

        # Stages losses
        self.compute_head_loss = True
        self.compute_projector_loss = True
        self.compute_proposal_loss = True

        self.model.roi_heads.mask_on = mask_on  # Only box-prediction component

        self.save_hyperparameters()

    def featureprojector_training_mode(self):
        self.compute_head_loss = False
        self.compute_proposal_loss = False
        self.compute_projector_loss = True

    def proposal_training_mode(self):
        self.compute_proposal_loss = True
        self.compute_head_loss = False
        self.compute_projector_loss = False

    def classifier_finetune_mode(self):
        self.compute_proposal_loss = False
        self.compute_head_loss = True
        self.compute_projector_loss = False
        self.compute_projector_loss = True
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.roi_heads.parameters():
            param.requires_grad = True

        for param in self.feature_projector.parameters():
            param.requires_grad = True

    def configure_optimizers(self, *args, **kwargs):
        optimizer = getattr(torch.optim, self.optimizer)(
            params=self.parameters(),
            lr=self.lr,
            **self.opt_params,
        )

        return optimizer

    def _compute_contrastive_loss(self, features, y):
        if self.compute_projector_loss:

            y_mask = y != -1
            y = y[y_mask]

            if len(y) > 1:

                features = self.feature_projector(features[y_mask])
                return triplet.online_mine_hard(
                    y.to(self.device), features, self.loss_margin, device=self.device
                )[0]
            else:
                return features.sum() * 0.0  # connect the gradient
        else:
            return None

    def _common_step(self, batch):

        (
            predictions,
            pred_loss,
            box_features,
            y_matching,
        ) = self._compute(batch)

        contrastive_loss = None
        if self.loss_weights.get('contrastive_loss', 1.0) > 0:
            contrastive_loss = self._compute_contrastive_loss(box_features, y_matching)

        result = {}
        if contrastive_loss is not None:
            contrastive_loss = contrastive_loss * self.loss_weights.get(
                'contrastive_loss', 1.0
            )
            result['loss_contrastive'] = contrastive_loss

        if pred_loss is not None:

            for key, _ in pred_loss.items():
                pred_loss[key] *= self.loss_weights.get(key, 1.0)
            result = {**result, **pred_loss}

        return result, predictions

    def training_step(self, batch, batch_idx):

        losses, predictions = self._common_step(batch)

        return losses, predictions

    def validation_step(self, batch, batch_idx):
        losses, predictions = self._common_step(
            batch,
        )  # always use gt for validation

        return losses, predictions

    @torch.no_grad()
    def __call__(self, inputs):
        self.eval()
        height = inputs[0]['height']
        width = inputs[0]['width']

        images = self.preprocess_image(inputs)

        if "instances" in inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in inputs]
        else:
            gt_instances = None

        features = self.model.backbone(images.tensor)
        proposals, _ = self.model.proposal_generator(images, features, gt_instances)

        instances, _ = self.model.roi_heads(images, features, proposals, gt_instances)
        mask_features = [features[f] for f in self.model.roi_heads.in_features]
        predictions_images = []

        for i in range(len(instances)):
            predictions_images += [i] * len(instances[i])

        if gt_instances is not None:
            boxes = [gt_instances[i].gt_boxes for i in range(len(gt_instances))]
        else:
            boxes = [instances[i].pred_boxes for i in range(len(instances))]

        pooled_features = self.model.roi_heads.box_pooler(mask_features, boxes)
        box_features = self.feature_projector(
            self.model.roi_heads.box_head(pooled_features)
        )

        predictions = self.postprocess(height, width, instances)

        return predictions, box_features  # , predictions_images

    def _compute(self, batched_inputs):
        inputs = []
        for i in batched_inputs:
            if isinstance(i, List):
                inputs += i
            else:
                inputs.append(i)

        gt_instances = []

        for x in inputs:
            x = x["instances"].to(self.device)

            if hasattr(x, "infos"):

                x.gt_ids = torch.tensor(
                    [i['id_object'] for i in x.infos], dtype=torch.int16
                )
            else:
                x.gt_ids = torch.ones(len(x)) * -1

            gt_instances.append(x)

        images = self.preprocess_image(inputs)

        features = self.model.backbone(images.tensor)

        proposals, prop_loss = self.model.proposal_generator(
            images, features, gt_instances
        )

        # extract features
        mask_features = [features[f] for f in self.model.roi_heads.in_features]

        labeled_props = self.model.roi_heads.label_and_sample_proposals(
            proposals, gt_instances
        )
        prop_boxes = [p.proposal_boxes for p in labeled_props]
        prop_ids = torch.cat(
            [
                p.gt_ids if hasattr(p, "gt_ids") else torch.ones(len(p)) * -1
                for p in labeled_props
            ]
        )

        # Get features for the regions proposals
        prop_classes = torch.cat([p.gt_classes for p in labeled_props])

        box_features = self.model.roi_heads.box_pooler(mask_features, prop_boxes)
        box_features = self.model.roi_heads.box_head(box_features)
        y_mask = prop_classes != len(BBSense.CLASSES)  # Background
        box_features = box_features[y_mask]
        prop_ids = prop_ids[y_mask]

        height = inputs[0]['height']
        width = inputs[0]['width']

        prediction_loss = {**prop_loss}
        if self.compute_head_loss:
            self.model.roi_heads.train()

            _, head_loss = self.model.roi_heads(
                images, features, proposals, gt_instances
            )
            prediction_loss = {**prediction_loss, **head_loss}

        self.model.roi_heads.eval()
        instances = self.head_forward(images, features, proposals)
        self.model.roi_heads.train()
        outputs = self.postprocess(height, width, instances)

        return (
            outputs,
            prediction_loss,
            box_features,
            prop_ids,
        )


class SoftMultiStageModel(MultiStageModel):
    def __init__(self, *args, **kwargs):

        if "head_cls" not in kwargs:
            kwargs['head_cls'] = lambda x: SoftHeadWrapper(
                prediction_head=x,
                temperature=kwargs.get('temperature', 1.0),
                alpha=kwargs.get('alpha', 0.7),
            )
        super().__init__(*args, **kwargs)


class DetachedMultiStageModel(MultiStageModel):
    def __init__(self, *args, **kwargs):

        if "head_cls" not in kwargs:
            kwargs['head_cls'] = lambda x: DetachedPredictorWrapper(
                prediction_head=x,
                temperature=kwargs.get('temperature', 1.0),
                alpha=kwargs.get('alpha', 0.7),
            )
        super().__init__(*args, **kwargs)


class FocalMultiStageModel(MultiStageModel):
    def __init__(self, *args, **kwargs):
        if "head_cls" not in kwargs:
            kwargs['head_cls'] = lambda x: SoftFocalHeadWrapper(
                prediction_head=x,
                temperature=kwargs.get('temperature', 1.0),
                alpha=kwargs.get('alpha', 0.5),
            )
        super().__init__(*args, **kwargs)


class MSEFocalMultiStageModel(MultiStageModel):
    def __init__(self, *args, **kwargs):
        if "head_cls" not in kwargs:
            kwargs['head_cls'] = lambda x: MSEFocalHeadWrapper(
                prediction_head=x,
                temperature=kwargs.get('temperature', 1.0),
                alpha=kwargs.get('alpha', 0.5),
            )
        super().__init__(*args, **kwargs)


class FocalOnlyMultiStageModel(MultiStageModel):
    def __init__(self, *args, **kwargs):
        if "head_cls" not in kwargs:
            kwargs['head_cls'] = lambda x: FocalHeadWrapper(
                prediction_head=x,
            )
        super().__init__(*args, **kwargs)


class DropoutMultiStageModel(MultiStageModel):
    def __init__(self, *args, **kwargs):
        if "head_cls" not in kwargs:

            kwargs['head_cls'] = lambda x: DropoutPredictorWrapper(
                prediction_head=x, **kwargs
            )
        super().__init__(*args, **kwargs)


class SoftDropoutMultiStageModel(MultiStageModel):
    def __init__(self, *args, **kwargs):
        if "head_cls" not in kwargs:

            kwargs['head_cls'] = lambda x: SoftDropoutPredictorWrapper(
                prediction_head=x, **kwargs
            )
        super().__init__(*args, **kwargs)


class MSEFocalDropoutMultiStageModel(MultiStageModel):
    def __init__(self, *args, **kwargs):
        if "head_cls" not in kwargs:

            kwargs['head_cls'] = lambda x: SoftDropoutPredictorWrapper(
                prediction_head=x, **kwargs
            )
        super().__init__(*args, **kwargs)


class MultiStageModelLRScheduler(MultiStageModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def configure_optimizers(self, total_steps=1000, *args, **kwargs):
        optimizer = getattr(torch.optim, self.optimizer)(
            params=self.parameters(),
            lr=self.lr,
            **self.opt_params,
        )
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                max_lr=0.1,
                total_steps=total_steps
            ),
            "interval": "step",
        }
        return optimizer

class MultiStageModelCosineLRScheduler(MultiStageModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def configure_optimizers(self, total_steps=1000, *args, **kwargs):

        optimizer = getattr(torch.optim, self.optimizer)(
            params=self.parameters(),
            lr=self.lr,
            **self.opt_params,
        )
        scheduler_dict = {
            "scheduler": CosineAnnealingLR(
                optimizer,
                T_max=total_steps
            ),
            "interval": "step",
        }
        return optimizer

