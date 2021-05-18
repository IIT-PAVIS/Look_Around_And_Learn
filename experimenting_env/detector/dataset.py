import logging
import math
import os

import cv2
import numpy as np
import torch
import tqdm
from albumentations.pytorch import ToTensorV2
from detectron2.data import detection_utils as du
from detectron2.structures.boxes import Boxes, BoxMode
from detectron2.structures.instances import Instances
from detectron2.structures.masks import BitMasks
from torch.utils.data import Dataset

from experimenting_env import replay
from experimenting_env.sensor_data import BBSense
from experimenting_env.utils import predictor_utils as pu

# A logger for this file
log = logging.getLogger(__name__)


class SinglecamEpisodeDetectionHabitatObjectsDataset(Dataset):
    """
    Instead of full COCO, return only boxes for habitat with new label
    """

    def __init__(
        self,
        exp_path,
        inputs=None,
        sampler=None,
        remap_classes=True,
        transform=None,
        modalities=None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.exp_path = exp_path
        if modalities is None:
            modalities = ['rgb', 'bbsgt']
        if sampler is None:
            sampler = replay.SampleLoader(exp_path)
        self.sampler = sampler
        if inputs is None:
            (
                episode_list,
                steps_list,
            ) = self.sampler.get_episode_and_steps_dense_list(*args, **kwargs)
            inputs = np.array([x for x in zip(episode_list, steps_list)])

        self.inputs = inputs

        indices = np.arange(len(self.inputs))

        self.index = indices

        self.remap_classes = remap_classes
        if transform is None:
            transform = ToTensorV2()
        self.transform = transform

        self.camera_id = 0
        self.modalities = modalities
        self.exp_path = exp_path

    def __len__(self):
        return len(self.index)

    def _transform_batch(self, x, y):

        if isinstance(self.transform, ToTensorV2):
            out = self.transform(image=x)
            x = out['image']

        else:

            transformed = self.transform(
                image=x,
                bboxes=y.gt_boxes.tensor.numpy(),
                class_labels=y.gt_classes,
                masks=[m for m in y.gt_masks.numpy().astype(np.uint8)],
                infos=y.infos,
            )

            transformed_image = transformed['image']
            transformed_bboxes = transformed['bboxes']

            transformed_class_labels = transformed['class_labels']
            transformed_infos = transformed['infos']
            transformed_mask = transformed['masks']
            size = transformed_image.shape[1:]

            x = transformed_image

            if len(transformed_class_labels) > 0:

                y = Instances(
                    image_size=size,
                    gt_boxes=Boxes(transformed_bboxes),
                    gt_classes=torch.stack(transformed_class_labels),
                    gt_masks=BitMasks(
                        torch.stack([torch.tensor(x) for x in transformed_mask])
                    ),
                    infos=transformed_infos,
                )
            else:
                y = Instances(
                    gt_boxes=Boxes(torch.Tensor()),
                    image_size=size,
                    gt_masks=BitMasks(torch.Tensor(size=[0, *size])),
                    gt_classes=torch.Tensor(),
                    infos=[],
                )

        if self.remap_classes:
            y.gt_classes = torch.tensor(
                [BBSense.CLASSES_TO_IDX[x.item()] for x in y.gt_classes]
            )
        return x, y

    def __getitem__(self, idx):
        """
        Output RGBD image
        """
        episode, step = self.inputs[self.index[idx]]

        data = self.sampler.get_sample_multimodality(
            episode, self.camera_id, self.modalities, step
        )

        x = data['rgb'].data
        y = data['bbsgt'].get_bbs_as_gt()

        x, y = self._transform_batch(x, y)

        size = x.shape[1:]

        return {
            'episode': episode,
            'image': x,
            'episode': episode,
            'image_id': idx,
            'instances': y,
            'width': size[1],
            'height': size[0],
        }

    def get_coco_item_dict(self, idx):
        ind = self.index[idx].item()
        episode, step = self.inputs[ind]

        data = self.sampler.get_sample_multimodality(
            episode, self.camera_id, ['bbsgt'], step
        )

        gt = data['bbsgt']
        file_name = gt.frame.sense_info.get_path()
        y = data['bbsgt'].get_bbs_as_gt()
        class_labels = y.gt_classes
        if self.remap_classes:
            class_labels = torch.tensor(
                [BBSense.CLASSES_TO_IDX[x.item()] for x in class_labels]
            )
        annotations = [
            {
                'bbox': y[id_instance].gt_boxes.tensor[0].tolist(),
                'bbox_mode': BoxMode.XYXY_ABS,
                'category_id': class_labels[id_instance],
                # 'segmentation': y[id_instance].gt_masks,
                'iscrowd': 0,
            }
            for id_instance in range(len(y))
        ]

        instance_dict = {
            'file_name': file_name,
            'image_id': ind,
            'height': y.image_size[0],
            'width': y.image_size[1],
            'annotations': annotations,
            'episode': episode,
        }

        return instance_dict


class SinglecamEpisodeFullDataset(SinglecamEpisodeDetectionHabitatObjectsDataset):
    """
    All episodes information (depth, rgb, pose, gt)
    """

    def __init__(self, exp_path, *args, **kwargs):
        super().__init__(
            exp_path, modalities=["rgb", "depth", "position", "bbsgt"], *args, **kwargs
        )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        """
        Output RGBD image
        """
        episode, step = self.inputs[self.index[idx]]

        data = self.sampler.get_sample_multimodality(
            episode, self.camera_id, self.modalities, step
        )

        x = data['rgb'].data
        depth = data['depth'].data
        location = data['position'].get_T()
        y = data['bbsgt'].get_bbs_as_gt()

        rgbd = np.concatenate((x, depth), -1)
        transformed_rgbd, y = self._transform_batch(rgbd, y)
        x = transformed_rgbd[:3]
        depth = transformed_rgbd[-1].unsqueeze(0)
        size = x.shape[1:]

        return {
            'image': x,
            "depth": depth,
            "location": torch.tensor(location),
            "instances": y,
            'width': size[1],
            'height': size[0],
            'info': f"episode_{episode}_step_{step}",
            'episode': episode,
        }


class EpisodeSequentalObservationsDataset(
    SinglecamEpisodeDetectionHabitatObjectsDataset
):
    """
    Full dataset with sequential observations, ground-truths, location, and depth
    """

    def __init__(
        self, exp_path, consecutive_obs=4, subsample_factor=2, *args, **kwargs
    ):
        sampler = replay.SampleLoader(exp_path)
        (
            self.episode_list,
            self.steps_list,
        ) = sampler.get_episode_and_steps_dense_list()

        num_sample = len(self.steps_list) // consecutive_obs * subsample_factor
        episode_list = np.resize(self.episode_list, (num_sample, consecutive_obs))
        steps_list = np.resize(self.steps_list, (num_sample, consecutive_obs))

        inputs = np.array([x for x in zip(episode_list, steps_list)])

        self.window_size = consecutive_obs
        modalities = ['rgb', 'bbsgt']

        super().__init__(
            exp_path=exp_path,
            sampler=sampler,
            inputs=inputs,
            modalities=modalities,
            *args,
            **kwargs,
        )
        self.subsample_factor = subsample_factor

    def __getitem__(self, idx):
        """
        Output sequences of RGBD image with random window
        """

        result = []
        episodes, steps = self.inputs[self.index[idx]]
        length_mask = len(episodes)
        sub_randomized_mask = np.random.choice(
            range(length_mask), length_mask // self.subsample_factor
        )
        for index in sub_randomized_mask:
            episode = episodes[index]
            step = steps[index]

            data = self.sampler.get_sample_multimodality(
                episode, self.camera_id, self.modalities, step
            )

            x = data['rgb'].data
            y = data['bbsgt'].get_bbs_as_gt()

            x, y = self._transform_batch(x, y)
            if self.remap_classes:
                y.gt_classes = torch.tensor(
                    [BBSense.CLASSES_TO_IDX[x.item()] for x in y.gt_classes]
                )

            size = x.shape[1:]
            result.append(
                {
                    'image': x,
                    'instances': y,
                    'width': size[1],
                    'height': size[0],
                    'episode': episode,
                    'info': f"episode_{episode}_step_{step}",
                }
            )
        return result


class EpisodeFullDataset(SinglecamEpisodeDetectionHabitatObjectsDataset):
    """
    Full dataset with sequential observations, ground-truths, location, and depth
    """

    def __init__(
        self,
        exp_path,
        sampler=None,
        consecutive_obs=1,
        subsample_factor=1,
        index_mask=None,
        *args,
        **kwargs,
    ):
        if sampler is None:
            sampler = replay.SampleLoader(exp_path)
        (
            self.episode_list,
            self.steps_list,
        ) = sampler.get_episode_and_steps_dense_list(*args, **kwargs)

        if index_mask:
            self.episode_list = self.episode_list[index_mask]
            self.steps_list = self.steps_list[index_mask]
        num_sample = len(self.steps_list) // consecutive_obs * subsample_factor
        episode_list = np.resize(self.episode_list, (num_sample, consecutive_obs))
        steps_list = np.resize(self.steps_list, (num_sample, consecutive_obs))

        inputs = np.array([x for x in zip(episode_list, steps_list)])

        modalities = ['rgb', 'bbsgt']

        super().__init__(
            exp_path=exp_path,
            sampler=sampler,
            inputs=inputs,
            modalities=modalities,
            *args,
            **kwargs,
        )
        self.window_size = consecutive_obs

        self.subsample_factor = subsample_factor

    def __getitem__(self, idx):
        """
        Output sequences of RGBD image with random window
        """

        result = []
        episodes, steps = self.inputs[self.index[idx]]
        length_mask = len(episodes)
        sub_randomized_mask = np.random.choice(
            range(length_mask), length_mask // self.subsample_factor, replace=False
        )
        for index in sub_randomized_mask:

            episode = episodes[index]
            step = steps[index]

            data = self.sampler.get_sample_multimodality(
                episode, self.camera_id, self.modalities, step
            )

            x = data['rgb'].data
            y = data['bbsgt'].get_bbs_as_gt()

            x, y = self._transform_batch(x, y)

            size = x.shape[1:]

            result.append(
                {
                    'image_id': int(idx * self.window_size + index),
                    'image': x,
                    'instances': y,
                    'width': size[1],
                    'height': size[0],
                    'episode': torch.tensor(episode),
                    'step': torch.tensor(step),
                }
            )
        return result

    def get_coco_item_dict(self, index):

        idx = index // self.window_size
        sub_id = index % self.window_size

        episodes, steps = self.inputs[self.index[idx]]
        episode = episodes[sub_id]
        step = steps[sub_id]

        data = self.sampler.get_sample_multimodality(
            episode, self.camera_id, ['bbsgt'], step
        )

        gt = data['bbsgt']
        file_name = gt.frame.sense_info.get_path()
        y = data['bbsgt'].get_bbs_as_gt()
        class_labels = y.gt_classes
        if self.remap_classes:
            class_labels = torch.tensor(
                [BBSense.CLASSES_TO_IDX[x.item()] for x in class_labels]
            )
        annotations = [
            {
                'bbox': y[id_instance].gt_boxes.tensor[0].tolist(),
                'bbox_mode': BoxMode.XYXY_ABS,
                'category_id': class_labels[id_instance],
                # 'segmentation': y[id_instance].gt_masks,
                'iscrowd': 0,
            }
            for id_instance in range(len(y))
        ]

        instance_dict = {
            'file_name': file_name,
            'image_id': index,
            'height': y.image_size[0],
            'width': y.image_size[1],
            'annotations': annotations,
        }

        return instance_dict


def _get_dataset_labels(dataset):
    labels = []
    log.info("Computing dataset annotations for weighted upsampling")
    for items in tqdm.tqdm(dataset.inputs):
        window_labels = []
        for idx in range(len(items)):
            episode = items[0][idx]
            step = items[1][idx]
            data = dataset.sampler.get_sample_multimodality(
                episode, dataset.camera_id, ['bbsgt'], step
            )

            y = data['bbsgt'].get_bbs_as_gt()
            class_labels = y.gt_classes
            window_labels.append(class_labels)
        labels.append(window_labels)
    return labels


def repeat_factors_from_category_frequency(labels, repeat_thresh=0.5):
    """
    Compute (fractional) per-image repeat factors based on category frequency.
    The repeat factor for an image is a function of the frequency of the rarest
    category labeled in that image. The "frequency of category c" in [0, 1] is defined
    as the fraction of images in the training set (without repeats) in which category c
    appears.
    See :paper:`lvis` (>= v2) Appendix B.2.

    Args:
        dataset_dicts (list[dict]): annotations in Detectron2 dataset format.
        repeat_thresh (float): frequency threshold below which data is repeated.
            If the frequency is half of `repeat_thresh`, the image will be
            repeated twice.

    Returns:
        torch.Tensor:
            the i-th element is the repeat factor for the dataset image at index i.
    """
    # 1. For each category c, compute the fraction of images that contain it: f(c)
    category_freq = {k: 0.0 for k in range(len(BBSense.CLASSES))}

    for labels_per_step in labels:
        for l in labels_per_step:
            category_freq[l['category_id'].item()] += 1

    num_images = len(labels)
    for k, v in category_freq.items():
        category_freq[k] = v / num_images

        # 2. For each category c, compute the category-level repeat factor:
        #    r(c) = max(1, sqrt(t / f(c)))
    category_rep = {
        cat_id: (max(1.0, math.sqrt(repeat_thresh / cat_freq)) if cat_freq > 0 else 1.0)
        for cat_id, cat_freq in category_freq.items()
    }

    # 3. For each image I, compute the image-level repeat factor:
    #    r(I) = max_{c in I} r(c)
    rep_factors = []
    for labels_per_step in labels:
        rep_factor = max(
            {category_rep[l['category_id'].item()] for l in labels_per_step},
            default=1.0,
        )
        rep_factors.append(rep_factor)
    return torch.tensor(rep_factors, dtype=torch.float32)


class PartiallyAnnotatedEpisodeFullDataset(EpisodeFullDataset):
    """
    Full dataset with sequential observations, ground-truths, location, and depth
    """

    def __init__(self, randomly_annotated=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.provide_annotation_mask = np.random.choice(
            self.index, int(len(self.inputs) * randomly_annotated)
        )

    def __getitem__(self, idx):
        """
        Output sequences of RGBD image with random window
        """
        batch = super().__getitem__(idx)
        if self.index[idx] not in self.provide_annotation_mask:
            for idx in range(len(batch)):
                batch[idx].pop('instances')
        return batch


class SinglecamEpisodeDetectionOnlyDataset(
    SinglecamEpisodeDetectionHabitatObjectsDataset
):
    def __init__(self, remap_classes=False, *args, **kwargs):
        super().__init__(remap_classes=remap_classes, *args, **kwargs)


class PseudoFullDataset(SinglecamEpisodeDetectionHabitatObjectsDataset):
    """
    Full dataset with sequential observations, ground-truths, location, and depth
    """

    def __init__(
        self,
        exp_path,
        pseudo_labels,
        sampler=None,
        consecutive_obs=1,
        subsample_factor=1,
        *args,
        **kwargs,
    ):

        if sampler is None:
            sampler = replay.SampleLoader(exp_path)
        (
            self.episode_list,
            self.steps_list,
        ) = sampler.get_episode_and_steps_dense_list(*args, **kwargs)

        self.pseudo_labels = []
        mask = []
        for pseudo in pseudo_labels:
            if len(pseudo) > 0:
                self.pseudo_labels.append(pseudo)
                mask.append(True)
            else:
                mask.append(False)

        if len(self.episode_list) > len(pseudo_labels):
            mask += [False] * (len(self.episode_list) - len(pseudo_labels))

        self.steps_list = self.steps_list[mask]
        self.episode_list = self.episode_list[mask]
        # self.compute_weights()

        # if consecutive_obs > 1:
        #     self.weights = (self.weights.roll(consecutive_obs - 1) + self.weights)[
        #         torch.arange(1, len(self.weights), consecutive_obs)
        #     ]
        num_sample = len(self.steps_list) // consecutive_obs * subsample_factor
        episode_list = np.resize(self.episode_list, (num_sample, consecutive_obs))
        steps_list = np.resize(self.steps_list, (num_sample, consecutive_obs))

        pseudo_list = np.resize(
            np.array(self.pseudo_labels, dtype='object'), (num_sample, consecutive_obs)
        )

        inputs = np.array([x for x in zip(episode_list, steps_list, pseudo_list)])

        modalities = ['rgb', 'bbsgt']

        super().__init__(
            exp_path=exp_path,
            sampler=sampler,
            inputs=inputs,
            modalities=modalities,
            *args,
            **kwargs,
        )
        self.window_size = consecutive_obs

        self.subsample_factor = subsample_factor

        assert self.pseudo_labels is not None

    def compute_weights(self):
        self.weights = repeat_factors_from_category_frequency(self.pseudo_labels)

    def __getitem__(self, idx):
        """
        Output sequences of RGBD image with random window
        """

        result = []
        episodes, steps, pseudo_labels = self.inputs[self.index[idx]]
        length_mask = len(episodes)
        sub_randomized_mask = np.random.choice(
            range(length_mask), length_mask // self.subsample_factor, replace=False
        )

        for index in sub_randomized_mask:

            image_id = int(idx * self.window_size + index)

            episode = episodes[index]
            step = steps[index]

            data = self.sampler.get_sample_multimodality(
                episode, self.camera_id, self.modalities, step
            )

            x = data['rgb'].data

            coco_ann = pseudo_labels[index]
            y = du.annotations_to_instances(
                coco_ann, x.shape[1:], mask_format='bitmask'
            )

            y.infos = [x['infos'] for x in coco_ann]

            y.gt_logits = torch.stack([x['gt_logits'] for x in coco_ann])

            if isinstance(y.gt_masks, BitMasks):
                y.gt_masks = y.gt_masks.tensor

            # TODO y.gt_uncertainty_masks = [x['uncertainties'] for x in coco_ann]
            x, y = _transform_batch_with_logits(self.transform, x, y)

            gt = data['bbsgt'].get_bbs_as_gt()

            if self.remap_classes:
                gt.gt_classes = torch.tensor(
                    [BBSense.CLASSES_TO_IDX[x.item()] for x in gt.gt_classes]
                )

            size = x.shape[1:]

            result.append(
                {
                    'image_id': image_id,
                    'image': x,
                    'instances': y,
                    'gt': gt,
                    'width': size[1],
                    'height': size[0],
                    'episode': torch.tensor(episode),
                    'step': torch.tensor(step),
                }
            )
        return result


def _transform_batch_with_logits(transform, x, y, remap_classes=True):

    if isinstance(transform, ToTensorV2):
        out = transform(image=x)
        x = out['image']

    else:

        transformed = transform(
            image=x,
            bboxes=y.gt_boxes.tensor.numpy(),
            class_labels=y.gt_classes,
            masks=[m for m in y.gt_masks.numpy().astype(np.uint8)],
            infos=y.infos,
            gt_logits=[l for l in y.gt_logits],
        )

        transformed_image = transformed['image']
        transformed_bboxes = transformed['bboxes']

        transformed_class_labels = transformed['class_labels']
        transformed_infos = transformed['infos']
        transformed_mask = transformed['masks']

        transformed_gt_logits = transformed['gt_logits']
        size = transformed_image.shape[1:]

        x = transformed_image

        min_area = -1  # transform._to_dict()['bbox_params']['min_area']

        if len(transformed_class_labels) > 0:

            semantic_masks_boxes = [cv2.boundingRect(x.numpy() if not isinstance(x, np.ndarray) else x) for x in transformed_mask]
            semantic_masks_area = [x[-1] * x[-2] for x in semantic_masks_boxes]
            gt_masks = BitMasks(
                torch.stack(
                    [
                        torch.tensor(x)
                        for x, area in zip(transformed_mask, semantic_masks_area)
                        if area > min_area
                    ]
                )
            )

            y = Instances(
                image_size=size,
                gt_boxes=Boxes(transformed_bboxes),
                gt_classes=torch.stack(transformed_class_labels),
                gt_logits=torch.stack(transformed_gt_logits),
                gt_masks=gt_masks,
                infos=transformed_infos,
            )
        else:
            y = Instances(
                gt_boxes=Boxes(torch.Tensor()),
                image_size=size,
                gt_masks=BitMasks(torch.Tensor(size=[0, *size])),
                gt_classes=torch.Tensor(),
                gt_logits=torch.Tensor(),
                infos=[],
            )

    return x, y
