import numpy as np
import torch
from detectron2.layers import cross_entropy
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.structures import ImageList
from fvcore.nn import sigmoid_focal_loss_jit
from kornia.losses.focal import focal_loss
from torch import nn
from torch.nn import functional as F

from experimenting_env.sensor_data import BBSense
from experimenting_env.utils.detectron_utils import fast_rcnn_inference


class MinimalPredictorWrapper(nn.Module):
    def __init__(self, prediction_head: FastRCNNOutputLayers):
        super().__init__()
        assert isinstance(
            prediction_head, FastRCNNOutputLayers
        ), "Trying to wrap a ROIHead different from FastRCNNOutputLayer"
        self.box_predictor = prediction_head

    def init_cls(self):
        nn.init.normal_(self.box_predictor.cls_score.weight, std=0.01)
        nn.init.normal_(self.box_predictor.bbox_pred.weight, std=0.001)
        for l in [self.box_predictor.cls_score, self.box_predictor.bbox_pred]:
            nn.init.constant_(l.bias, 0)

    def reinit_head(self, classes_idxs):
        classes_to_keep = np.array([*classes_idxs, 80])

        self.box_predictor.cls_score.bias = torch.nn.Parameter(
            self.box_predictor.cls_score.bias[classes_to_keep]
        )
        self.box_predictor.cls_score.weight = torch.nn.Parameter(
            self.box_predictor.cls_score.weight[classes_to_keep]
        )
        classes_to_keep = np.array([*classes_idxs])
        mask = np.repeat(classes_to_keep * 4, 4) + np.tile(
            np.arange(0, 4), len(classes_to_keep)
        )
        self.box_predictor.bbox_pred.weight = torch.nn.Parameter(
            self.box_predictor.bbox_pred.weight[mask]
        )
        self.box_predictor.bbox_pred.bias = torch.nn.Parameter(
            self.box_predictor.bbox_pred.bias[mask]
        )
        self.box_predictor.num_classes = len(classes_idxs)
        self.box_predictor.cls_score.out_features = len(classes_idxs)
        self.box_predictor.bbox_pred.out_features = len(classes_idxs) * 4

    def forward(self, x):
        return self.box_predictor(x)

    def losses(self, predictions, proposals):
        return self.box_predictor.losses(predictions, proposals)

    def inference(self, predictions, proposals):
        return self.box_predictor.inference(predictions, proposals)


class DetachedPredictorWrapper(MinimalPredictorWrapper):
    def __init__(
        self, prediction_head: FastRCNNOutputLayers, cls_loss=None, *args, **kwargs
    ):
        super().__init__(prediction_head)
        if cls_loss is None:
            cls_loss = lambda x, y, reduction: cross_entropy(  # noqa: E731
                x, y, reduce=False, reduction=reduction
            )
        self.cls_loss = cls_loss

    def forward(self, x):

        return self.box_predictor(x.detach())


class BoxPredictorWrapper(MinimalPredictorWrapper):
    def __init__(
        self, prediction_head: FastRCNNOutputLayers, cls_loss=None, *args, **kwargs
    ):
        super().__init__(prediction_head)
        if cls_loss is None:
            cls_loss = lambda x, y, reduction: cross_entropy(  # noqa: E731
                x, y, reduce=False, reduction=reduction
            )
        self.cls_loss = cls_loss

    def forward(self, x):
        return self.box_predictor(x)

    def losses(self, predictions, proposals):
        return self.box_predictor.losses(predictions, proposals)

    def train_losses(self, predictions, proposals, reduction='mean'):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` are expected.

        Returns:
            Dict[str, Tensor]: dict of losses
        """
        logits, proposal_deltas = predictions

        gt_classes = (
            torch.cat([p.gt_classes for p in proposals], dim=0)
            if len(proposals)
            else torch.empty(0)
        )

        # parse box regression outputs
        if len(proposals):
            proposal_boxes = torch.cat(
                [p.proposal_boxes.tensor for p in proposals], dim=0
            )  # Nx4
            assert (
                not proposal_boxes.requires_grad
            ), "Proposals should not require gradients!"
            # If "gt_boxes" does not exist, the proposals must be all negative and
            # should not be included in regression loss computation.
            # Here we just use proposal_boxes as an arbitrary placeholder because its
            # value won't be used in self.box_reg_loss().
            gt_boxes = torch.cat(
                [
                    (p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor
                    for p in proposals
                ],
                dim=0,
            )
        else:
            proposal_boxes = gt_boxes = torch.empty(
                (0, 4), device=proposal_deltas.device
            )

        if len(proposals) == 0:
            cross_entropy_loss = logits.sum() * 0.0  # connect the gradient

        else:
            # Calculate distillation loss

            cross_entropy_loss = self.cls_loss(logits, gt_classes, "mean")

        losses = {
            "loss_cls": cross_entropy_loss,
            "loss_box_reg": self.box_predictor.box_reg_loss(
                proposal_boxes, gt_boxes, proposal_deltas, gt_classes
            ),
        }

        return {
            k: v * self.box_predictor.loss_weight.get(k, 1.0) for k, v in losses.items()
        }

    def inference(self, predictions, proposals):
        logits, _ = predictions
        num_inst_per_image = [len(p) for p in proposals]

        boxes = self.box_predictor.predict_boxes(predictions, proposals)
        logits_per_image = F.softmax(logits, dim=-1).split(num_inst_per_image, dim=0)

        scores = self.box_predictor.predict_probs(predictions, proposals)

        shapes = [x.image_size for x in proposals]
        return fast_rcnn_inference(
            boxes,
            scores,
            shapes,
            self.box_predictor.test_score_thresh,
            self.box_predictor.test_nms_thresh,
            self.box_predictor.test_topk_per_image,
            logits=logits_per_image,
        )


class FinetuneBoxPredictorWrapper(MinimalPredictorWrapper):
    def __init__(self, *args):
        super().__init__(*args)
        nn.init.normal_(self.box_predictor.cls_score.weight, std=0.01)


class SoftHeadWrapper(BoxPredictorWrapper):
    def __init__(
        self,
        prediction_head,
        temperature=1,
        alpha=0.5,
        cls_loss=None,
        soft_loss=None,
        *args,
        **kwargs
    ):
        super().__init__(prediction_head=prediction_head, *args, **kwargs)
        self.temperature = temperature
        self.alpha = alpha
        if cls_loss is None:

            cls_loss = lambda x, y, reduction: cross_entropy(  # noqa: E731
                x, y, reduction=reduction
            )
        if soft_loss is None:
            soft_loss = lambda x, y: soft_cross_entropy(x, y)
        self.cls_loss = cls_loss
        self.soft_loss = soft_loss

    def _get_consistency_loss(self, predictions, proposals, reduction="mean"):
        logits, proposal_deltas = predictions
        gt_logits = []

        if len(proposals):
            logits_mask = []

            for p in proposals:
                if hasattr(p, 'gt_logits'):
                    gt_logits.append(p[p.gt_classes != len(BBSense.CLASSES)].gt_logits)
                    logits_mask.append(p.gt_classes != len(BBSense.CLASSES))
                else:
                    logits_mask.append(torch.zeros_like(p.gt_classes, dtype=torch.bool))
            logits_mask = torch.cat(logits_mask)
            gt_logits = torch.cat(gt_logits, dim=0)

        else:
            logits_mask = torch.empty()

        if gt_logits.numel() == 0 and reduction == "mean":
            distillation_loss = logits.sum() * 0.0  # connect the gradient

        else:
            # Calculate distillation loss

            soft_log_probs = F.softmax(logits[logits_mask] / self.temperature, dim=-1)

            distillation_loss = self.soft_loss(
                soft_log_probs, gt_logits
            )  # Note: gt_logits have been already normalized by temperature
        return distillation_loss * self.alpha

    def train_losses(self, predictions, proposals, reduction='mean'):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` are expected.

        Returns:
            Dict[str, Tensor]: dict of losses
        """
        logits, proposal_deltas = predictions

        gt_classes = (
            torch.cat([p.gt_classes for p in proposals], dim=0)
            if len(proposals)
            else torch.empty(0)
        )

        # parse box regression outputs
        if len(proposals):
            proposal_boxes = torch.cat(
                [p.proposal_boxes.tensor for p in proposals], dim=0
            )  # Nx4
            assert (
                not proposal_boxes.requires_grad
            ), "Proposals should not require gradients!"
            # If "gt_boxes" does not exist, the proposals must be all negative and
            # should not be included in regression loss computation.
            # Here we just use proposal_boxes as an arbitrary placeholder because its
            # value won't be used in self.box_reg_loss().
            gt_boxes = torch.cat(
                [
                    (p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor
                    for p in proposals
                ],
                dim=0,
            )
        else:
            proposal_boxes = gt_boxes = torch.empty(
                (0, 4), device=proposal_deltas.device
            )

        if len(proposals) == 0 and reduction == "mean":
            cross_entropy_loss = logits.sum() * 0.0  # connect the gradient

        else:
            # Calculate distillation loss

            cross_entropy_loss = self.cls_loss(logits, gt_classes, reduction='mean')

        losses = {
            "loss_cls": cross_entropy_loss,
            "loss_box_reg": self.box_predictor.box_reg_loss(
                proposal_boxes, gt_boxes, proposal_deltas, gt_classes
            ),
        }

        return {
            k: v * self.box_predictor.loss_weight.get(k, 1.0) for k, v in losses.items()
        }

    def losses(self, predictions, proposals):

        losses = self.train_losses(predictions, proposals)
        if len(proposals) == 0 or any([hasattr(p, 'gt_logits') for p in proposals]):
            cons_loss = self._get_consistency_loss(predictions, proposals)
            losses['soft_loss'] = cons_loss
        return losses


class SoftFocalHeadWrapper(SoftHeadWrapper):
    def __init__(self, focal_loss_alpha=0.25, focal_loss_gamma=2.0, *args, **kwargs):
        def cls_loss(x, y, reduction):
            return 10 * focal_loss(
                x,
                y,
                alpha=self.focal_loss_alpha,
                gamma=self.focal_loss_gamma,
                reduction=reduction,
            )

        super().__init__(cls_loss=cls_loss, *args, **kwargs)

        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma


class MSEFocalHeadWrapper(SoftHeadWrapper):
    def __init__(self, focal_loss_alpha=0.25, focal_loss_gamma=2.0, *args, **kwargs):
        def cls_loss(x, y, reduction):
            return 10 * focal_loss(
                x,
                y,
                alpha=self.focal_loss_alpha,
                gamma=self.focal_loss_gamma,
                reduction=reduction,
            )

        super().__init__(
            cls_loss=cls_loss,
            soft_loss=lambda x, y: F.mse_loss(x, F.log_softmax(y)),
            *args,
            **kwargs
        )

        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma


class FocalHeadWrapper(BoxPredictorWrapper):
    def __init__(self, focal_loss_alpha=0.25, focal_loss_gamma=2.0, *args, **kwargs):
        def cls_loss(x, y, reduction):
            return 10 * focal_loss(
                x,
                y,
                alpha=self.focal_loss_alpha,
                gamma=self.focal_loss_gamma,
                reduction="mean",
            )

        super().__init__(cls_loss=cls_loss, *args, **kwargs)

        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma


class DropoutPredictorWrapper(MinimalPredictorWrapper):
    def __init__(
        self,
        prediction_head: FastRCNNOutputLayers,
        dropout_p=0.4,
        cls_loss=None,
        *args,
        **kwargs
    ):

        prediction_head.cls_score = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(
                prediction_head.cls_score.in_features,
                prediction_head.cls_score.out_features,
            ),
        )
        prediction_head.bbox_pred = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(
                prediction_head.bbox_pred.in_features,
                prediction_head.bbox_pred.out_features,
            ),
        )

        nn.init.normal_(prediction_head.cls_score[1].weight, std=0.01)
        nn.init.normal_(prediction_head.bbox_pred[1].weight, std=0.001)
        for l in [prediction_head.cls_score[1], prediction_head.bbox_pred[1]]:
            nn.init.constant_(l.bias, 0)

        super().__init__(prediction_head)
        if cls_loss is None:
            cls_loss = lambda x, y, reduction: cross_entropy(  # noqa: E731
                x, y, reduce=False, reduction=reduction
            )
        self.cls_loss = cls_loss


class SoftDropoutPredictorWrapper(SoftHeadWrapper):
    def __init__(
        self, prediction_head: FastRCNNOutputLayers, dropout_p=0.8, *args, **kwargs
    ):

        prediction_head.cls_score = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(
                prediction_head.cls_score.in_features,
                prediction_head.cls_score.out_features,
            ),
        )
        prediction_head.bbox_pred = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(
                prediction_head.bbox_pred.in_features,
                prediction_head.bbox_pred.out_features,
            ),
        )

        nn.init.normal_(prediction_head.cls_score[1].weight, std=0.01)
        nn.init.normal_(prediction_head.bbox_pred[1].weight, std=0.001)
        for l in [prediction_head.cls_score[1], prediction_head.bbox_pred[1]]:
            nn.init.constant_(l.bias, 0)

        super().__init__(prediction_head, *args, **kwargs)


class SoftFocalDropoutHeadWrapper(SoftDropoutPredictorWrapper):
    def __init__(self, focal_loss_alpha=0.25, focal_loss_gamma=2.0, *args, **kwargs):
        def cls_loss(x, y, reduction):
            return 10 * focal_loss(
                x,
                y,
                alpha=self.focal_loss_alpha,
                gamma=self.focal_loss_gamma,
                reduction="mean",
            )

        super().__init__(cls_loss=cls_loss, soft_loss=F.mse_loss, *args, **kwargs)

        self.focal_loss_alpha = focal_loss_alpha

        self.focal_loss_gamma = focal_loss_gamma


def soft_cross_entropy(input, target):
    logprobs = F.log_softmax(input, dim=1)
    return -(target * logprobs).sum() / input.shape[0]
