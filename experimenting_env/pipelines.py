"""
Pipeline implementation
"""
import logging
import math
import multiprocessing
import multiprocessing as mp
import os
from copy import copy, deepcopy
from itertools import chain
from typing import List, Sequence, Union
import pickle
import albumentations as A
import habitat
import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from albumentations.augmentations.crops.transforms import RandomCropNearBBox
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data import transforms as T
from detectron2.structures.masks import BitMasks
from detectron2.utils.events import EventStorage
from detectron2.utils.visualizer import ColorMode, Visualizer
from habitat.core.dataset import Dataset
from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from pl_bolts.callbacks.byol_updates import BYOLMAWeightUpdate
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import StochasticWeightAveraging
from torch import Tensor
from torch.nn import Module
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import SWALR, AveragedModel
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import Sampler
from torchmetrics.detection.map import MAP

import shutil
import wandb
from experimenting_env.detector.augmentations import get_transform
from experimenting_env.detector.dataset import (
    EpisodeFullDataset,
    PseudoFullDataset,
    SinglecamEpisodeDetectionHabitatObjectsDataset,
    SinglecamEpisodeFullDataset,
    _transform_batch_with_logits,
)
from experimenting_env.detector.model import multi_stage_models as models
from experimenting_env.detector.model.multi_stage_models import *
from experimenting_env.detector.pseudolabeler import (
    ConsensusLabeler,
    SemanticMapConsensusLabeler,
    SoftConsensusLabeler,
    VanillaConsensusLabeler,
)
from experimenting_env.detector.utils import coco_laoder
from experimenting_env.replay import SampleLoader
from experimenting_env.sensor_data import BBSense
from experimenting_env.utils.detectron_utils import get_coco_item_dict
from experimenting_env.utils.train_helpers import (
    DistributedWeightSampler,
    dict_helper_collate,
    evaluate_labeler,
    get_loader,
    get_training_params,
    list_helper_collate,
    mixup_batch,
)

log = logging.getLogger(__name__)

__ALL__ = ["Pipeline", "GTPipeline", "SWAPipeline"]




class HabitatDataModule(pl.LightningDataModule):
    def __init__(self, pseudo_labeler, policy, dataset_path, data_base_dir, test_set, transform_type='none', batch_size=8, consecutive_obs=1, use_coco=False,  *args, **kwargs ):
        super().__init__()
        self.pseudo_labeler = pseudo_labeler
        self.policy = policy
        self.dataset_path = dataset_path
        self.sampler = None
        self.labels = None
        self.batch_size=batch_size
        self.data_base_dir = data_base_dir
        self.test_set = test_set
        self.transform_type = transform_type
        self.num_workers = 0
        self.use_coco = use_coco

        self.consecutive_obs = int(consecutive_obs)

    def _get_labels(self, sampler):
        val_transform = A.Compose(
            get_transform("none"),
            bbox_params=A.BboxParams(
                format='pascal_voc',
                min_area=0,
                min_visibility=0,
                label_fields=['class_labels', 'infos'],
            ),
        )
        pseudolabel_dataset = SinglecamEpisodeFullDataset(
            None,
            sampler=sampler,
            transform=val_transform,
        )

        pseudolabel_loader = get_loader(
            pseudolabel_dataset,
            shuffle=False,
            batch_size=32,
            num_workers=self.num_workers,
            collate_fn=dict_helper_collate,
        )

        pseudolabel_trainer = pl.Trainer(gpus=1)
        self.pseudo_labeler.global_pcds = {}
        model_outs = pseudolabel_trainer.predict(
            self.pseudo_labeler, pseudolabel_loader
        )
        pseudo_labels = self.pseudo_labeler.get_pseudo_labels(
            model_outs, pseudolabel_loader
        )

        coco_pseudo_labels = get_coco_item_dict(pseudo_labels)

        return coco_pseudo_labels

    def prepare_data(self):
        if not os.path.exists(self.dataset_path):
            path = os.path.join(os.getcwd(), f"dataset")
            if os.path.exists(path):
                shutil.rmtree(path) # Empty dataset before generating new samples
            self.policy.exp_path = path
            self.policy.generate()
        sampler = self._get_sampler()
        labels = self._get_labels(sampler)
        with open("labels.pkl", "wb") as fp:
            pickle.dump(labels, fp)

    def _get_sampler(self):
        if os.path.exists(self.dataset_path):
            sampler = SampleLoader(self.dataset_path)
        else:
            path = os.path.join(os.getcwd(), f"dataset")
            sampler = SampleLoader(path)
        return sampler

    def _get_dataset(self, sampler: SampleLoader, coco_pseudo_labels):

        """
        Apply pseudo-labeler and return consistent pseudolabel dataset
        """

        train_transform = A.Compose(
            get_transform(self.transform_type),
            bbox_params=A.BboxParams(
                format='pascal_voc',
                min_area=0,
                label_fields=['class_labels', 'infos', 'gt_logits'],
            ),
        )
        assert len(coco_pseudo_labels) > 0, "No pseudo-labels provided"
        assert len(coco_pseudo_labels) == len(
            sampler
        ), f"Expected {len(sampler)} got {len(coco_pseudo_labels)}"

        train_dataset = PseudoFullDataset(
            exp_path=None,
            sampler=sampler,
            transform=train_transform,
            pseudo_labels=coco_pseudo_labels,
            consecutive_obs=self.consecutive_obs
        )
        return train_dataset

    def _get_validation(self):
        transform = A.Compose(
            [
                A.pytorch.ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['class_labels', 'infos'],
            ),
        )

        dataset = SinglecamEpisodeDetectionHabitatObjectsDataset(
            os.path.join(self.data_base_dir, self.test_set),
            transform=transform,
            remap_classes=True,
        )
        return dataset

    def setup(self, stage):
        sampler = self._get_sampler()

        with open('labels.pkl', 'rb') as handle:
            labels = pickle.load(handle)
    
        self.train_dataset = self._get_dataset(sampler, labels)
        self.test_dataset = self._get_validation()

    def train_dataloader(self):
        batch_size = self.batch_size
        train_loader = get_loader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=list_helper_collate,
        )
        if self.use_coco:
            batch_size = batch_size // 2
            coco_trainloader = coco_laoder(
                "./data/coco/annotations/instances_train2017.json",
                "./data/coco/train2017",
                batch_size,
                [
                    T.RandomBrightness(0.9, 1.1),
                    T.RandomFlip(prob=0.5),
                    T.RandomCrop("absolute", (640, 640)),
                ],
                self.num_workers
            )
            train_loader = [train_loader, coco_trainloader]

        return train_loader


    def val_dataloader(self):
        test_loader = get_loader(
                self.test_dataset,
                batch_size=4,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=dict_helper_collate,
            )

        return test_loader

    def test_dataloader(self):
        test_loader = get_loader(
                self.test_dataset,
                batch_size=4,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=dict_helper_collate,
            )

        return test_loader


class GTDataModule(HabitatDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prepare_data(self):
        if not os.path.exists(self.dataset_path):
            path = os.path.join(os.getcwd(), f"dataset")
            if os.path.exists(path):
                shutil.rmtree(path) # Empty dataset before generating new samples
            self.policy.exp_path = path
            self.policy.generate()

    def setup(self, stage):
        sampler = self._get_sampler()

        self.train_dataset = self._get_dataset(sampler)
        self.test_dataset = self._get_validation()

    def _get_dataset(self, sampler: SampleLoader):

        """
        Apply pseudo-labeler and return consistent pseudolabel dataset
        """
        log.info("Using ground-truth for detector training")

        train_transform = A.Compose(
            get_transform(self.transform_type),
            bbox_params=A.BboxParams(
                format='pascal_voc',
                min_area=0,
                label_fields=['class_labels', 'infos'],
            ),
        )

        inputs = sampler.get_episode_and_steps_dense_list()
        filter_empty_instances = []

        for ep, step in zip(inputs[0], inputs[1]):
            instances = sampler.get_sample(ep, 0, "bbsgt", step).get_bbs_as_gt()

            filter_empty_instances.append(len(instances) > 0)


        return EpisodeFullDataset(
            exp_path=None,
            sampler=sampler,
            index_mask=filter_empty_instances,
            transform=train_transform,
            remap_classes=True,
        )

    def train_dataloader(self):
        batch_size = self.batch_size
        train_loader = get_loader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=list_helper_collate,
        )

        return train_loader


class TeacherStudent(pl.LightningModule):
    def __init__(
        self,
        detectron_args,
        consensus="vanilla",
        temperature=1,
        student_model=None,
        thr=0.7,
        freeze_teacher=True,
        use_teacher=False,
        batch_size=1,
        mixup=False,
        solution="ours",
        *args,
        **kwargs,
    ):
        super().__init__()
        switch = {
            "logits": SoftConsensusLabeler,
            "vanilla": VanillaConsensusLabeler,
            "semantic_map": SemanticMapConsensusLabeler,
        }
        # Initialize student and teacher / pseudo-labeler

        if student_model is not None:
            self.student_model_class = getattr(models, student_model)
        else:
            self.student_model_class = models.FocalMultiStageModel

        self.kwargs = kwargs
        self.batch_size = batch_size
        self.mixup = mixup
        self.use_teacher = use_teacher
        self.detectron_args = detectron_args

        self.target_network: ConsensusLabeler = switch[consensus](
            model=models.MultiStageModel(detectron_args, prune=True),
            temperature=temperature,
            thr=thr,
            solution=solution,
        )
        self.max_steps = None
        if freeze_teacher:
            self.target_network.freeze()
        else:
            self.target_network.train()

        self.freeze_teacher = freeze_teacher


        self.online_val_map_metric = MAP(class_metrics=True)
        self.test_map_metric = MAP(class_metrics=True)
        self.reinit_online()
        self.save_hyperparameters()

    def reinit_online(self):
        self.online_network = self.student_model_class(self.detectron_args, **self.kwargs)
        self.online_network.model.roi_heads.box_predictor.box_predictor.test_score_thresh = (
            0.5
        )
    def training_step(self, batched_inputs, batch_idx):
        batch = []

        for i in batched_inputs:
            if isinstance(i, List):
                batch += i
            else:
                batch.append(i)
        if self.mixup:
            mixup_batch(batch)

        if batch_idx % 50 == 0:
            self.log_batch(batch, batch_idx)
        losses, _ = self.online_network.training_step(batch, batch_idx)
        if losses is None:
            return None
        loss = sum(losses.values())
        for k in losses.keys():

            self.log(
                f"train_{k}",
                losses[k],
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.batch_size,
            )
        self.log(
            'train_loss',
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        if self.use_teacher:
            self.target_validation_step(batch, batch_idx)
        self.online_validation_step(batch, batch_idx)

    def target_validation_step(self, batch, batch_idx):
        self.target_network.eval()
        losses, predictions = self.target_network.validation_step(batch, batch_idx)
        if losses is None:
            return None
        loss = sum(losses.values())
        for k in losses.keys():

            self.log(
                f"val_{k}_target",
                losses[k],
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.batch_size,
            )

        self.log(
            'val_loss_target',
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )

    def _val_map_segm(self, batch, predictions):

        gt = [
            {
                'masks': b['instances'].gt_masks.tensor.to("cpu"),
                'labels': b['instances'].gt_classes.int().to("cpu"),
            }
            for b in batch
        ]
        pred = [
            {
                'masks': b['instances'].pred_masks.to("cpu"),
                'labels': b['instances'].pred_classes.to("cpu"),
                'scores': b['instances'].scores.to("cpu"),
            }
            for b in predictions
        ]

        self.online_val_map_segm_metric.update(pred, gt)

    def _val_map(self, batch, predictions):
        device = self.device

        gt = [
            {
                'boxes': b['instances'].gt_boxes.tensor.to(device),
                'labels': b['instances'].gt_classes.int().to(device),
            }
            for b in batch
        ]
        pred = [
            {
                'boxes': b['instances'].pred_boxes.tensor.to(device),
                'labels': b['instances'].pred_classes.to(device),
                'scores': b['instances'].scores.to(device),
            }
            for b in predictions
        ]
        self.online_val_map_metric.update(pred, gt)

    def online_validation_step(self, batch, batch_idx):
        self.online_network.eval()
        losses, predictions = self.online_network.validation_step(batch, batch_idx)
        if losses is None:
            return None
        loss = sum(losses.values())
        for k in losses.keys():

            self.log(
                f"val_{k}_online",
                losses[k],
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.batch_size,
            )

        self.log(
            'val_loss_online',
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )

        self._val_map(batch, predictions)
        # self._val_map_segm(batch, predictions)

    def validation_epoch_end(self, outputs):

        results = self.online_val_map_metric.compute()
        for k in results.keys():
            self.log(
                f"val_{k}_online",
                results[k],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.batch_size,
            )
        self.online_val_map_metric = MAP(class_metrics=True)
        self.online_val_map_metric.to(self.device)

            #        self.reset_metric()
        # self.online_val_map_segm_metric.to("cpu")
        # results = self.online_val_map_segm_metric.compute()
        # for k in results.keys():
        #     self.log(
        #         f"val_segm_{k}_online",
        #         results[k],
        #         on_step=False,
        #         on_epoch=True,
        #         sync_dist=True,
        #         batch_size=self.batch_size,
        #     )


    def test_step(self, batch, batch_idx):
        self.online_network.eval()
        _, predictions = self.online_network.validation_step(batch, batch_idx)

        gt = [
            {
                'boxes': b['instances'].gt_boxes.tensor,
                'labels': b['instances'].gt_classes.int(),
            }
            for b in batch
        ]
        pred = [
            {
                'boxes': b['instances'].pred_boxes.tensor,
                'labels': b['instances'].pred_classes,
                'scores': b['instances'].scores,
            }
            for b in predictions
        ]

        self.test_map_metric.update(pred, gt)

    def on_test_epoch_end(self):

        results = self.test_map_metric.compute()
        for k in results.keys():
            self.log(
                f"test_{k}",
                results[k],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.batch_size,
            )

    def configure_optimizers(self):
        optimizer = self.online_network.configure_optimizers(max_steps=self.max_steps)

        return optimizer

    def reset_metric(self):
        self.online_val_map_metric = MAP(class_metrics=True)
        self.online_val_map_metric.to(self.device)

    def log_batch(self, batch, batch_idx):
        for idx, x in enumerate(batch):

            remap = BBSense.REMAP
            metadata = MetadataCatalog.get('coco_2017_val')
            visualizer = Visualizer(
                deepcopy(x['image'].permute(1, 2, 0).cpu()),
                metadata,
                instance_mode=ColorMode.IMAGE,
            )
            y = deepcopy(x['instances'])

            y.pred_classes = torch.tensor([remap[p.item()] for p in y.gt_classes])
            if hasattr(y, "gt_boxes"):
                y.pred_boxes = y.gt_boxes
            if hasattr(y, "gt_masks"):
                if isinstance(y.gt_masks, Tensor):
                    y.pred_masks = y.gt_masks
                else:
                    y.pred_masks = y.gt_masks.tensor
            frame = visualizer.draw_instance_predictions(
                predictions=y.to('cpu')
            ).get_image()
            img = wandb.Image(frame)

            self.trainer.logger[0].log_metrics(
                {
                    f"gt-batch-{batch_idx}-img-{idx}": img,
                    "trainer/global_step": self.trainer.global_step,
                }
            )


class OnlineTeacherStudent(TeacherStudent):
    transform = None

    def __init__(self, *args, **kwargs):
        super().__init__(use_teacher=True, *args, **kwargs)

    def training_step(self, batched_inputs, batch_idx):
        batch = []
        if isinstance(batched_inputs[0], List):  # assuming this is not COCO
            pseudo_batch = batched_inputs[0]
        else:
            pseudo_batch = batched_inputs
        outs = [self.target_network.forward(pseudo_batch)]
        pseudo_labels = self.target_network.get_pseudo_labels(outs)

        # Apply augmentation only for student training_step
        for b, pseudo in zip(pseudo_batch, pseudo_labels):
            device = b['image'].device
            x, y = _transform_batch_with_logits(
                self.transform, b['image'].permute(1, 2, 0).cpu().numpy(), pseudo
            )
            x.to(device)
            b['image'] = x
            b['instances'] = y

        for i in batched_inputs:
            if isinstance(i, List):
                batch += i
            else:
                batch.append(i)

        if not self.freeze_teacher:
            self.target_network.train()
        else:
            self.target_network.eval()

        return super().training_step(batch, batch_idx)


class Pipeline:
    teacher_student_model_class = TeacherStudent

    def __init__(self, cfg):
        self.use_coco = cfg.training.use_coco

        self.teacher_student = self.teacher_student_model_class(
            habitat=cfg.habitat_cfg,
            **cfg,
            **cfg.training,
        )

        self.pseudo_labeler = self.teacher_student.target_network

        self.teacher_path = "updated_teacher.ckpt"

        # Get config for policy
        config = habitat.get_config(
            os.path.join(cfg.habitat_base_cfg_dir, cfg.habitat_cfg)
        )
        trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)

        assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"

        self.epochs_per_iteration = cfg.epochs
        self.policy_trainer: BaseRLTrainer = trainer_init(config, **cfg)

        # Model training
        self.set_trainer_params(cfg)


        self.num_workers = 0  # multiprocessing.cpu_count()

        self.base_path = os.path.join(cfg.exp_base_dir, cfg.replay.exp_name)
        self.cfg = cfg

        # Update teacher path into config for policy trainer
        

    def set_trainer_params(self, cfg):
        self.trainer_config = get_training_params(cfg)
        if "ema" in cfg.training and cfg.training.ema:
            self.trainer_config['callbacks'].append(
                BYOLMAWeightUpdate(cfg.training.teacher_momentum)
            )

    def train_policy(self):
        self.policy_trainer.train()

    def collect_pseudolabeled_dataset(self):
        sampler = self._get_observations_sampler()
        labels = self._get_labels(sampler)
        loader = self._get_dataloader(sampler, labels)
        assert len(labels) == len(sampler), f"Expected {len(sampler)} got {len(labels)}"
        return loader

    def _get_observations_sampler(self) -> SampleLoader:
        """
        Use policy to explore environment
        Returns sampler of the collectedv observations
        """

        obs_path = self.policy_trainer.generate()

        paths = list(chain(*obs_path))
        sampler = SampleLoader("", samples_path=paths)

        return sampler

    def _get_labels(self, sampler):
        val_transform = A.Compose(
            get_transform("none"),
            bbox_params=A.BboxParams(
                format='pascal_voc',
                min_area=0,
                min_visibility=0,
                label_fields=['class_labels', 'infos'],
            ),
        )
        pseudolabel_dataset = SinglecamEpisodeFullDataset(
            os.path.join(self.base_path, "train"),
            sampler=sampler,
            transform=val_transform,
        )

        pseudolabel_loader = get_loader(
            pseudolabel_dataset,
            shuffle=False,
            batch_size=32,
            num_workers=self.num_workers,
            collate_fn=dict_helper_collate,
        )

        pseudolabel_trainer = pl.Trainer(gpus=1)
        self.pseudo_labeler.global_pcds = {}
        model_outs = pseudolabel_trainer.predict(
            self.pseudo_labeler, pseudolabel_loader
        )
        pseudo_labels = self.pseudo_labeler.get_pseudo_labels(
            model_outs, pseudolabel_loader
        )

        coco_pseudo_labels = get_coco_item_dict(pseudo_labels)

        # results = evaluate_labeler(pseudolabel_loader, labels=pseudo_labels)
        # log.info(f"Pseudo-labels accuracy: {results}")
        return coco_pseudo_labels

    def _get_dataloader(self, sampler: SampleLoader, coco_pseudo_labels) -> DataLoader:
        train_dataset = self._get_dataset(sampler, coco_pseudo_labels)
        batch_size = self.cfg.training['batch_size']
        if self.use_coco:
            batch_size = batch_size // 2
        if (
            "weight_sampler" in self.cfg.training
            and self.cfg.training['weight_sampler']
        ):
            weight_sampler = DistributedWeightSampler(
                dataset=train_dataset, replacement=True
            )

        else:
            weight_sampler = None

        train_loader = get_loader(
            train_dataset,
            batch_size=batch_size,
            shuffle=weight_sampler is None,
            sampler=weight_sampler,
            num_workers=self.num_workers,
            collate_fn=list_helper_collate,
        )


        return train_loader

    def _get_dataset(self, sampler: SampleLoader, coco_pseudo_labels):

        """
        Apply pseudo-labeler and return consistent pseudolabel dataset
        """

        train_transform = A.Compose(
            get_transform(self.cfg.training['transform_type']),
            bbox_params=A.BboxParams(
                format='pascal_voc',
                min_area=0,
                label_fields=['class_labels', 'infos', 'gt_logits'],
            ),
        )
        assert len(coco_pseudo_labels) > 0, "No pseudo-labels provided"
        assert len(coco_pseudo_labels) == len(
            sampler
        ), f"Expected {len(sampler)} got {len(coco_pseudo_labels)}"

        train_dataset = PseudoFullDataset(
            exp_path=os.path.join(self.base_path, "train"),
            sampler=sampler,
            transform=train_transform,
            pseudo_labels=coco_pseudo_labels,
            consecutive_obs=self.cfg.training.consecutive_obs
            if "consecutive_obs" in self.cfg.training
            else 1,
        )
        return train_dataset

    def fit_student_and_update_teacher(
        self, dataloader: DataLoader, checkpoint_path: str = None, trainer = None
    ):
        self.max_steps=1000
        # self.max_steps = (
        #     len(dataloader) * self.cfg.training.n_iterations * self.epochs_per_iteration
        # )  # Same at every iteration, as #samples doesn't change

        if "checkpoint" in self.cfg:
            ckpt_path = os.path.join(
                self.cfg.exp_base_dir, self.cfg.checkpoint, 'checkpoints/last.ckpt'
            )
            if os.path.exists(ckpt_path):
                self.teacher_student.load_from_checkpoint(ckpt_path)
            else:
                log.warning(f"Cannot load checkpoint {ckpt_path}")

        with EventStorage():
            if checkpoint_path:
                if "reinit_online" in self.cfg.training and self.cfg.training['reinit_online']:
                    trainer.checkpoint_connector.resume_start(checkpoint_path)
                    trainer.checkpoint_connector.restore_callbacks()
                    trainer.checkpoint_connector.restore_datamodule()
                    trainer.checkpoint_connector.restore_training_state()
                    teacher_student.reinit_online() 
                else:
                    trainer.fit(self.teacher_student, dataloader, ckpt_path=checkpoint_path)
            else:
                trainer.fit(self.teacher_student, dataloader)


    def save_teacher_and_update_configs(self):
        """
        Trained object-detector becomes new pseudo-labeler. It's attached to
        the policy and the process can start again with a new iteration

        """
        self.pseudo_labeler.to("cpu")
        self.trainer_config['max_epochs'] += self.epochs_per_iteration
        if "update_target" in self.cfg.training and self.cfg.training['update_target']:
            if not "ema" in self.cfg.training or not self.cfg.training.ema:
                self.pseudo_labeler.reinit(self.teacher_student.online_network)
        

        # if hasattr(self.trainer, "object_detector"):  # Trained with Object-detector
        #     self.pseudo_labeler.save_model(self.teacher_path)
        #     self.trainer.object_detector.load(self.teacher_path)


class OnlinePipeline(Pipeline):
    teacher_student_model_class = OnlineTeacherStudent

    def __init__(self, cfg):
        super().__init__(cfg)
        self.teacher_student.transform = A.Compose(
            get_transform(self.cfg.training['transform_type']),
            bbox_params=A.BboxParams(
                format='pascal_voc',
                min_area=0,
                label_fields=['class_labels', 'infos'],
            ),
        )

    def _get_labels(self, sampler: SampleLoader):
        return None

    def _get_dataloader(self, sampler: SampleLoader, coco_pseudo_labels) -> DataLoader:
        """
        Apply pseudo-labeler and return consistent pseudolabel dataset
        """
        log.info("Generating pseudo-labels online")

        inputs = sampler.get_episode_and_steps_dense_list()
        filter_empty_instances = []
        for ep, step in zip(inputs[0], inputs[1]):
            instances = sampler.get_sample(ep, 0, "bbsgt", step).get_bbs_as_gt()

            filter_empty_instances.append(len(instances) > 0)

        train_dataset = EpisodeFullDataset(
            exp_path=os.path.join(self.base_path, "train"),
            sampler=sampler,
            index_mask=filter_empty_instances,
            transform=None,
            remap_classes=True,
        )

        train_loader = get_loader(
            train_dataset,
            batch_size=self.cfg.training['batch_size'],
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=list_helper_collate,
        )
        return train_loader


class GTPipeline(Pipeline):
    def __init__(self, cfg):
        super().__init__(cfg)

    def _get_labels(self, sampler: SampleLoader):
        return None

    def _get_dataloader(self, sampler: SampleLoader, coco_pseudo_labels) -> DataLoader:
        """
        Apply pseudo-labeler and return consistent pseudolabel dataset
        """
        log.info("Using ground-truth for detector training")

        train_transform = A.Compose(
            get_transform(self.cfg.training['transform_type']),
            bbox_params=A.BboxParams(
                format='pascal_voc',
                min_area=0,
                label_fields=['class_labels', 'infos'],
            ),
        )

        inputs = sampler.get_episode_and_steps_dense_list()
        filter_empty_instances = []
        for ep, step in zip(inputs[0], inputs[1]):
            instances = sampler.get_sample(ep, 0, "bbsgt", step).get_bbs_as_gt()

            filter_empty_instances.append(len(instances) > 0)

        train_dataset = EpisodeFullDataset(
            exp_path=os.path.join(self.base_path, "train"),
            sampler=sampler,
            index_mask=filter_empty_instances,
            transform=train_transform,
            remap_classes=True,
        )

        train_loader = get_loader(
            train_dataset,
            batch_size=self.cfg.training['batch_size'],
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=list_helper_collate,
        )
        return train_loader


class SWAPipeline(Pipeline):
    def __init__(self, cfg):
        super().__init__(cfg)

    def set_trainer_params(self, cfg):
        super().set_trainer_params(cfg)

        self.trainer_config['callbacks'].append(
            StochasticWeightAveraging(
                swa_epoch_start=30, annealing_epochs=5, swa_lrs=cfg.training.swa_lr
            )
        )
