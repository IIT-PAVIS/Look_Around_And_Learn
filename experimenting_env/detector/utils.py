# type: ignore
import contextlib
import copy
import glob
import io
import itertools
import json
import logging
import os
import pickle
from collections import OrderedDict
from typing import Union

import albumentations as A
import detectron2.utils.comm as comm
import habitat
import hydra
import numpy as np
import pycocotools.mask as mask_util
import pytorch_lightning as pl
import torch
from albumentations.augmentations.transforms import Lambda
from albumentations.core.utils import DataProcessor
from detectron2.config import CfgNode
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    build_detection_train_loader,
)
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.datasets.coco import convert_to_coco_json, load_coco_json
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.evaluation.coco_evaluation import _evaluate_predictions_on_coco
from detectron2.evaluation.fast_eval_api import COCOeval_opt
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import create_small_table
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate
from torch import functional as F
from torch import nn
from torch.utils.data.sampler import RandomSampler

import experimenting_env
from experimenting_env.detector import model as exp_models
from experimenting_env.detector.dataset import (
    SinglecamEpisodeDetectionHabitatObjectsDataset,
)
from experimenting_env.sensor_data import BBSense
from experimenting_env.utils import predictor_utils
from experimenting_env.utils.detectron_utils import fast_rcnn_inference
from experimenting_env.utils.predictor_utils import Predictor, setup_cfg
from experimenting_env.utils.train_helpers import (
    dict_helper_collate,
    get_loader,
)


def get_mapper():
    from detectron2.data import detection_utils as utils

    # Show how to implement a minimal mapper, similar to the default DatasetMapper
    def mapper(dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # can use other ways to read image
        image = utils.read_image(dataset_dict["file_name"], format="BGR")
        # See "Data Augmentation" tutorial for details usage
        auginput = T.AugInput(image)
        transform = T.Resize((800, 800))(auginput)
        image = torch.from_numpy(auginput.image.transpose(2, 0, 1))
        annos = [
            utils.transform_instance_annotations(
                annotation, [transform], image.shape[1:]
            )
            for annotation in dataset_dict.pop("annotations")
        ]
        return {
            # create the format that the model expects
            "image": image,
            "instances": utils.annotations_to_instances(annos, image.shape[1:]),
        }

    return mapper


def coco_laoder(json_file, image_root, batch_size, train_augm, num_workers):

    DatasetCatalog.register(
        "custom_coco", lambda: load_coco_json(json_file, image_root)
    )

    coco_mapping = {
        63: 0,  # "couch",
        64: 1,  # "potted",
        65: 2,  # "bed",
        70: 3,  # "toilet",
        72: 4,  # "tv",
        67: 5,  # "table",
    }
    data = DatasetCatalog.get('custom_coco')
    filtered_data = []
    for idx in range(len(data)):
        d = data[idx]
        d['annotations'] = list(
            filter(lambda x: x['category_id'] in coco_mapping.keys(), d['annotations'])
        )

        if len(d['annotations']) > 0:

            for i, _ in enumerate(d['annotations']):
                cat_id = d['annotations'][i]['category_id']
                d['annotations'][i]['category_id'] = coco_mapping[cat_id]
            filtered_data.append(d)

    train_dataset = DatasetFromList(filtered_data)
    sampler = RandomSampler(train_dataset)
    mapper = DatasetMapper(
        is_train=True,
        image_format="RGB",
        use_instance_mask=True,
        instance_mask_format="bitmask",
        augmentations=train_augm,
    )

    train_loader = build_detection_train_loader(
        train_dataset,
        mapper=mapper,
        sampler=sampler,
        total_batch_size=batch_size,
        aspect_ratio_grouping=False,
        num_workers=num_workers,
    )

    return train_loader


@torch.no_grad()
def coco_evaluate(
    model: Union[predictor_utils.Predictor, Lambda],
    dataset,
    dataset_name: str,
    dataset_json_dir=None,
    output_dir=os.getcwd(),
    batch_size=1,
    remap_classes=False,
):
    if dataset_json_dir is None:
        dataset_json_dir = os.path.join(
            os.getcwd(), "exps", "json_datasets", dataset_name
        )

    def get_dict():
        return [dataset.get_coco_item_dict(idx) for idx in range(len(dataset))]

    os.makedirs(dataset_json_dir, exist_ok=True)

    if isinstance(model, predictor_utils.Predictor):
        model.model.eval()

    DatasetCatalog.register(dataset_name, get_dict)
    MetadataCatalog.get(dataset_name).thing_classes = list(BBSense.CLASSES.values())

    json_file = os.path.join(dataset_json_dir, "coco_format.json")

    if not (os.path.exists(json_file)):
        convert_to_coco_json(dataset_name, output_file=json_file, allow_cached=False)
    MetadataCatalog.get(dataset_name).json_file = json_file
    loader = get_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=10,
        collate_fn=dict_helper_collate,
    )

    evaluator = COCOEvaluator(
        dataset_name, tasks=['bbox'], output_dir=output_dir, use_fast_impl=False
    )

    evaluator.reset()
    for inputs in loader:
        outputs = model(inputs)

        if remap_classes:

            for i in range(len(outputs)):
                instances = outputs[i]['instances']
                mask = torch.tensor(
                    [
                        instances.pred_classes[instance_id].item() in BBSense.CLASSES
                        for instance_id in range(len(instances))
                    ],
                    dtype=torch.bool,
                )

                outputs[i]['instances'] = instances[mask]
                outputs[i]['instances'].pred_classes = torch.tensor(
                    [
                        BBSense.CLASSES_TO_IDX[
                            outputs[i]['instances'].pred_classes[instance_id].item()
                        ]
                        for instance_id in range(len(outputs[i]['instances']))
                    ]
                )

        evaluator.process(inputs, outputs)

    eval_results = evaluator.evaluate()
    return eval_results
