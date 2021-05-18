import copy
import logging
import multiprocessing
import os
import pickle
from copy import copy, deepcopy
from itertools import chain
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
from detectron2.structures.masks import BitMasks
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import DistributedSampler
from torch.utils.data.dataloader import default_collate

import experimenting_env
from experimenting_env.utils.ddp_long_timeout import (  # type: ignore  # noqa
    DDPPlugin,
)

logging.basicConfig(level=logging.INFO)


def get_training_params(cfg):
    """

    Parameters

    cfg: DictConfig :
        hydra configuration (examples in conf/train)

    -------

    """

    logger = [
        _get_wandb_logger(
            project_name=experimenting_env.project_name,
            exp_name=cfg.training.exp_base_name + "/" + cfg.exp_name,
        ),
    ]
    exp_path = os.getcwd()
    checkpoint_dir = os.path.join(exp_path, "checkpoints")
    log_profiler = os.path.join(exp_path, "profile.txt")
    os.makedirs(checkpoint_dir, exist_ok=True)

    ckpt_cb = ModelCheckpoint(
        monitor="val_map_50_online",
        mode='max',
        save_last=True,
        verbose=True,
        dirpath=checkpoint_dir,
        filename="{epoch:02d}",
        every_n_epochs=2,
    )

    gpus = cfg["gpus"]

    if "plugins" in cfg:
        plugins = cfg['plugins']
    else:
        plugins = None



    trainer_configuration = {
        "multiple_trainloader_mode": "min_size",
        "default_root_dir": checkpoint_dir,
        "gpus": gpus,
        "max_epochs": cfg["epochs"] if "epochs" in cfg else 100,
        "callbacks": [ckpt_cb],
        "enable_checkpointing": True,
        "weights_summary": "top",
        "logger": logger,
        "plugins": plugins,
        "num_sanity_val_steps": 0,
        "check_val_every_n_epoch": 2,

    }
    trainer_configuration['strategy'] = DDPPlugin(find_unused_parameters=True)

    if "debug" in cfg and cfg['debug']:
        torch.autograd.set_detect_anomaly(True)
        trainer_configuration["overfit_batches"] = 50
        trainer_configuration["log_gpu_memory"] = True

    if "early_stopping" in cfg and cfg['early_stopping'] > 0:
        early_stop_callback = EarlyStopping(
            monitor="train_loss_cls_epoch",
            min_delta=0.001,
            patience=cfg["early_stopping"],
            verbose=False,
            mode="min",
        )
        trainer_configuration["callbacks"].append(early_stop_callback)

    return trainer_configuration


def _safe_train_end():
    exp_path = Path(os.getcwd())
    exp_name = exp_path.name
    error_path = os.path.join(exp_path.parent, "with_errors", exp_name)
    logging.log(logging.ERROR, f"exp error {error_path}")


def _get_wandb_logger(exp_name: str, project_name: str):
    logger = WandbLogger(
        name=exp_name,
        project=project_name,
        entity='gianscarpe'
    )
    return logger


def collate_fn_helper(batch):
    if isinstance(batch, list):
        return list_helper_collate(batch)
    elif isinstance(batch, dict):
        return dict_helper_collate(batch)
    else:
        return default_collate(batch)


def list_helper_collate(batch):
    return list(chain(*[[elem for elem in elems_list] for elems_list in batch]))


def dict_helper_collate(batch):

    elem = batch[0]
    return [{key: d[key] for key in elem} for d in batch]


def mixup_batch(batch):
    # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
    device = batch[0]['image'].device
    indexes = np.random.permutation(len(batch))
    alpha = beta = 1.5
    r = np.random.beta(alpha, beta)  # mixup ratio, alpha=beta=32.0
    for b1, idx in zip(batch, indexes):
        b2_image = deepcopy(batch[idx]['image'])
        b2_instances = deepcopy(batch[idx]['instances'])
        b1['image'] = (b1['image'] * r + b2_image * (1 - r)).int()
        y = Instances(b1['image'].shape)
        y.gt_classes = torch.cat([b1['instances'].gt_classes, b2_instances.gt_classes])

        y.gt_logits = torch.cat([b1['instances'].gt_logits, b2_instances.gt_logits])

        y.infos = b1['instances'].infos + b2_instances.infos

        if hasattr(b1['instances'], "gt_masks"):
            y.gt_masks = BitMasks(
                torch.cat(
                    [b1['instances'].gt_masks.tensor, b2_instances.gt_masks.tensor]
                )
            )

        y.gt_boxes = Boxes(
            torch.cat([b1['instances'].gt_boxes.tensor, b2_instances.gt_boxes.tensor])
        )
        b1['instances'] = y
        del b2_image
        del b2_instances


def get_loader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=multiprocessing.cpu_count(),
    collate_fn=dict_helper_collate,
    sampler=None,
):
    return torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=False,
        persistent_workers=False,
    )


class DistributedWeightSampler(DistributedSampler):
    def __init__(self, replacement=False, *args, **kwargs):
        super().__init__(shuffle=False, *args, **kwargs)
        self.weights = torch.as_tensor(self.dataset.weights, dtype=torch.double)
        self.replacement = replacement

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size
        # subsample indices
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples
        # do the weighted sampling
        subsample_balanced_indices = torch.multinomial(
            self.weights, self.total_size, self.replacement
        )
        # subsample the balanced indices
        subsample_balanced_indices = subsample_balanced_indices[indices]

        return iter(subsample_balanced_indices)


def collect_results_gpu(world_size, result_part, size):

    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda'
    )
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)

    # padding result part tensor to max length
    shape_max = int(torch.tensor(shape_list).max())
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[: shape_tensor[0]] = part_tensor
    part_recv_list = [part_tensor.new_zeros(shape_max) for _ in range(world_size)]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    part_list = []
    for recv, shape in zip(part_recv_list, shape_list):
        part_list.append(pickle.loads(recv[: shape[0]].cpu().numpy().tobytes()))
        # sort the results
    ordered_results = []
    for res in zip(*part_list):
        ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
    return ordered_results


def collect_results_horovod(result_part):
    import horovod.torch as hvd

    out = hvd.allgather_object(result_part)

    results = list(chain(*out))

    return results


def evaluate_labeler(loader, outputs=None, labeler=None, labels=None, device="cpu"):
    if labels is None:
        labels = labeler.get_pseudo_labels(outputs, loader)
    for batch, batch_pseudo in zip(loader, labels):

        gt = [
            {
                'boxes': b['instances'].gt_boxes.tensor.to(device),
                'labels': b['instances'].gt_classes.int().to(device),
                'masks': b['instances'].gt_masks.tensor.to(device),
            }
            for b in batch
        ]

        pred = [
            {
                'boxes': pseudo.gt_boxes.tensor.to(device),
                'labels': pseudo.gt_classes.int().to(device),
                'scores': (
                    pseudo.gt_logits.max(1).values.to(device)
                    if len(pseudo) > 0
                    else torch.Tensor([]).float().to(device)
                ),
                'masks': pseudo.gt_masks.to(device),
            }
            for pseudo in [batch_pseudo]
        ]

        labeler.test_map_metric.update(pred, gt)

    results = labeler.test_map_metric.compute()
    return results
