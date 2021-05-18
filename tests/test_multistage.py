import random
from unittest.mock import patch

import numpy as np
import pytest
import torch
from detectron2.utils.events import EventStorage
from mock import MagicMock, Mock
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from experimenting_env.detector.dataset import EpisodeFullDataset
from experimenting_env.detector.model.multi_stage_models import (
    MultiStageModel,
    SoftMultiStageModel,
)
from experimenting_env.detector.pseudolabeler import (
    MajorityConsensusLabeler,
    SoftConsensusLabeler,
)
from experimenting_env.sensor_data import BBSense
from experimenting_env.utils.predictor_utils import Predictor, SoftHeadWrapper
from experimenting_env.utils.train_helpers import (
    DistributedWeightSampler,
    get_loader,
    get_training_params,
    list_helper_collate,
)


def test_collect_hard_pseudo():
    dataset = EpisodeFullDataset(exp_path="exps/hres_gibson_def/test")
    cfg = DictConfig(
        {
            'config_file': 'third_parties/detectron2/configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml',
            'confidence_threshold': 0.5,
            'opts': [
                'MODEL.ROI_HEADS.NMS_THRESH_TEST',
                0.9,
                'MODEL.WEIGHTS',
                'detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl',
            ],
        }
    )
    model = MultiStageModel(cfg)
    pseudo_labeler = MajorityConsensusLabeler(model)
    B = 4
    model_outs = [
        (
            [MagicMock(pred_classes=torch.randint(0, 8, (3, 1))) for _ in range(B)],
            [[random.randint(0, 40) for _ in range(3)] for _ in range(B)],
        )
        for _ in range(100)
    ]

    pseudo_labels = pseudo_labeler.get_pseudo_labels(model_outs)
    assert pseudo_labels is not None


def test_collect_soft_pseudo():
    dataset = EpisodeFullDataset(exp_path="exps/hres_gibson_def/test")
    cfg = DictConfig(
        {
            'config_file': 'third_parties/detectron2/configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml',
            'confidence_threshold': 0.5,
            'opts': [
                'MODEL.ROI_HEADS.NMS_THRESH_TEST',
                0.9,
                'MODEL.WEIGHTS',
                'detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl',
            ],
        }
    )
    model = MultiStageModel(cfg)
    pseudo_labeler = SoftConsensusLabeler(model)

    B = 4
    model_outs = [
        (
            [MagicMock(gt_logits=torch.rand(3, 9)) for _ in range(B)],
            [[random.randint(0, 40) for _ in range(3)] for _ in range(B)],
        )
        for _ in range(100)
    ]

    pseudo_labels = pseudo_labeler.get_pseudo_labels(model_outs)
    assert pseudo_labels is not None


def test_model_with_soft_pseudo():
    dataset = EpisodeFullDataset(exp_path="exps/hres_gibson_def/test")
    cfg = DictConfig(
        {
            'config_file': 'third_parties/detectron2/configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml',
            'confidence_threshold': 0.5,
            'opts': [
                'MODEL.ROI_HEADS.NMS_THRESH_TEST',
                0.9,
                'MODEL.WEIGHTS',
                'detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl',
            ],
        }
    )
    with patch.object(MultiStageModel, 'log', return_value=None) as mock_method:
        model = SoftMultiStageModel(cfg)

        model.train()

        val_loader = get_loader(
            dataset,
            shuffle=False,
            batch_size=4,
            num_workers=12,
            collate_fn=list_helper_collate,
        )
        inputs = next(iter(val_loader))

        with EventStorage():
            out, _ = model.forward_head_loss(inputs)
            assert out['loss_cls'] is not None
            for x in inputs:
                x['instances'].gt_logits = torch.rand(
                    len(x['instances']), len(BBSense.CLASSES) + 1
                )

            out, _ = model.forward_head_loss(inputs)

            assert out['loss_soft'] is not None


def test_model_with_hard_pseudo():
    dataset = EpisodeFullDataset(exp_path="exps/hres_gibson_def/test")
    cfg = DictConfig(
        {
            'config_file': 'third_parties/detectron2/configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml',
            'confidence_threshold': 0.5,
            'opts': [
                'MODEL.ROI_HEADS.NMS_THRESH_TEST',
                0.9,
                'MODEL.WEIGHTS',
                'detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl',
            ],
        }
    )
    with patch.object(MultiStageModel, 'log', return_value=None) as mock_method:
        model = MultiStageModel(cfg)

        model.set_pseudolabels({i: random.randint(0, 8) for i in range(1000)})
        val_loader = get_loader(
            dataset,
            shuffle=False,
            batch_size=4,
            num_workers=12,
            collate_fn=list_helper_collate,
        )
        inputs = next(iter(val_loader))
        with EventStorage():
            model.train()
            out = model.training_step(inputs, 0)
            assert out is not None
            out = model.validation_step(inputs, 0)
            assert out is not None


def test_soft_kl():
    box_mock = Mock()
    box_mock.box_reg_loss.return_value = 0
    box_mock.loss_weight = {}
    head = SoftHeadWrapper(box_mock)
    predictions = (torch.rand(10, 9), Mock(device='cpu'))
    predictions[0].requires_grad = True
    proposals = []

    out = head.train_losses(predictions, proposals)
    assert out['loss_soft'] == 0
    assert out['loss_soft'].requires_grad
