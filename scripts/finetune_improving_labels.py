"""
Train full pipeline
"""
import datetime
import gc
import logging
import multiprocessing as mp
import os

import albumentations as A
import hydra
import numba as nb
import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from detectron2.data import transforms as T
from detectron2.utils.events import EventStorage
from habitat_baselines.common.baseline_registry import baseline_registry
from torch.utils.data.dataloader import DataLoader

from experimenting_env import pipelines, replay
from experimenting_env.detector.dataset import (
    SinglecamEpisodeDetectionHabitatObjectsDataset,
    repeat_factors_from_category_frequency,
)
from experimenting_env.detector.utils import coco_laoder
from experimenting_env.replay import SampleLoader
from experimenting_env.utils.detectron_utils import get_coco_item_dict
from experimenting_env.utils.train_helpers import (
    dict_helper_collate,
    get_loader,
    get_training_params,
    list_helper_collate,
)

log = logging.getLogger(__name__)


@hydra.main(config_path='../confs/', config_name='train.yaml')
def main(cfg) -> None:

    data_path = os.path.join(os.getcwd(), "data")
    exp_path = os.path.join(os.getcwd(), "exps")
    if not (os.path.exists(data_path)):
        os.symlink(cfg.data_base_dir, data_path)
    if not (os.path.exists(exp_path)):
        os.symlink(cfg.exp_base_dir, exp_path)

    if "pipeline" in cfg.training and hasattr(pipelines, cfg.training['pipeline']):

        pipeline = getattr(pipelines, cfg.training['pipeline'])(cfg)
    else:
        pipeline = pipelines.Pipeline(cfg)
    dataset_path = os.path.join(os.getcwd(), "exps", cfg.replay.exp_name, "dataset")

    
    trainer = pl.Trainer(**pipeline.trainer_config)    

    checkpoint_path = None
    for id_iteration in range(cfg.training.n_iterations):

        if 'use_gt' in cfg.training and cfg.training['use_gt']:
            dm = pipelines.GTDataModule(pipeline.pseudo_labeler, pipeline.policy_trainer, dataset_path, **cfg, **cfg.training)
        else:
            dm = pipelines.HabitatDataModule(pipeline.pseudo_labeler, pipeline.policy_trainer, dataset_path, **cfg, **cfg.training)




        pipeline.fit_student_and_update_teacher(dm, checkpoint_path, trainer)

        checkpoint_path = f"iteration-{id_iteration}.ckpt"
        trainer.save_checkpoint(checkpoint_path)
        pipeline.save_teacher_and_update_configs()


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
            os.path.join(cfg.data_base_dir, "fix_test"),
            transform=transform,
            remap_classes=True,
        )

    test_loader = get_loader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=10,
        collate_fn=dict_helper_collate,
    )

    with EventStorage():
        with torch.no_grad():
            trainer.test(pipeline.teacher_student, test_loader)


if __name__ == "__main__":
    main(
)
