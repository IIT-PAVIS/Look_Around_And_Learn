import glob
import os
from typing import Dict, List

import cv2
import numba as nb
import numpy as np

from experimenting_env.sensor_data import (
    AgentPoseSense,
    BBSense,
    DepthSense,
    RGBSense,
    SemanticSense,
    Sense,
    VisualSense,
    get_class_from_modality_code,
)
from experimenting_env.utils.sensors_utils import (
    _get_info_from_string,
    get_sense_info,
)


def _mask_more_n(arr, n):
    mask = np.ones(arr.shape, np.bool_)

    current = arr[0]
    count = 0
    for idx, item in enumerate(arr):
        if item == current:
            count += 1
        else:
            current = item
            count = 1
        mask[idx] = count <= n
    return mask


class SampleLoader:
    paths: Dict
    episode_list: np.ndarray

    def __init__(self, exp_path, samples_path=None):
        self._load_paths(exp_path, samples_path)

    def __len__(self):
        return len(self.get_episode_and_steps_dense_list()[0])

    def _load_paths(self, load_path, samples_paths=None):
        if samples_paths is None:
            samples_paths = sorted(glob.glob(load_path + '/*.npy'))
        paths = {}

        episode_list = [int(_get_info_from_string(s, "episode")) for s in samples_paths]
        mod_list = [_get_info_from_string(s, "modality") for s in samples_paths]
        idx_list = [int(_get_info_from_string(s, "id")) for s in samples_paths]
        steps_list = [int(_get_info_from_string(s, "step")) for s in samples_paths]

        for sample_path, episode_id, input_id, mod, step in zip(
            samples_paths, episode_list, idx_list, mod_list, steps_list
        ):
            if episode_id not in paths:
                paths[episode_id] = {}
            if input_id not in paths[episode_id]:
                paths[episode_id][input_id] = {}
            if mod not in paths[episode_id][input_id]:
                paths[episode_id][input_id][mod] = {}

            paths[episode_id][input_id][mod][step] = sample_path

        self.paths = paths
        self.episode_list = np.array(episode_list)
        self.steps_list = np.array(steps_list)

    @staticmethod
    def _load_data(path: str) -> Sense:
        sense_info = get_sense_info(path)
        return get_class_from_modality_code(sense_info.mod).load(path)

    def get_episode(self, episode_id, modalities, cameras):
        for step in range(self.get_episode_length(episode_id)):
            for mod in modalities:
                for camera_id in cameras:
                    yield self.get_sample(episode_id, mod, camera_id, step)

    def get_episode_length(self, episode_id):
        return len(self.paths[episode_id][0][RGBSense.CODE])

    def get_sample(self, episode_id, input_id, mod, step):
        try:
            data_path = self.paths[episode_id][input_id][mod][step]
            return SampleLoader._load_data(data_path)
        except Exception as ex:
            raise Exception(f"{episode_id}, {input_id}, {mod}, {step}")

    def get_sample_multimodality(self, episode_id, id_camera, modalities, step):
        results = {}
        for mod in modalities:
            data = self.get_sample(episode_id, id_camera, mod, step)
            results[mod] = data
        return results

    def get_episode_and_steps_dense_list(self, filter_episodes=None, *args, **kwargs):
        """
        Get list of episodes and of steps
        """
        mask = _mask_more_n(self.steps_list, 1)

        if filter_episodes is not None:
            mask_episodes = np.array(
                [li in filter_episodes for li in self.episode_list]
            )
            mask *= mask_episodes

        return self.episode_list[mask], self.steps_list[mask]


def replay_experiment(
    exp_path, modalities, episode_id, cameras_id, start_step=1, end_step=10
):
    sampler = SampleLoader(exp_path)
    running = True

    if end_step < 0:
        end_step = sampler.get_episode_length(episode_id)
    if start_step < 1:
        start_step = 1

    n = start_step
    while n < end_step - start_step:
        ch = cv2.waitKey(500)
        if ch == ord("q") or ch == ord("Q"):
            break
        elif ch == ord("s") or ch == ord("S"):
            running = False
        elif ch == ord("r") or ch == ord("R"):
            running = True
        if running:
            for id_camera in cameras_id:
                for mod in modalities:
                    try:
                        data = sampler.get_sample(episode_id, id_camera, mod, n)
                    except Exception as ex:
                        print(ex)
                        data = None

                    if isinstance(data, VisualSense):
                        data.show()

            n += 1


def get_experiment_bbs_and_locations(
    exp_path, episode_id, cameras_id, start_step=1, additional_mods=None
):
    rgb_modality = RGBSense.CODE
    location_modality = AgentPoseSense.CODE
    semantic_modality = SemanticSense.CODE
    bbs_modality = BBSense.CODE
    depth_modality = DepthSense.CODE
    sampler = SampleLoader(exp_path)
    results = []

    for n in range(start_step, sampler.get_episode_length(episode_id)):
        step_results = {}
        for id_camera in cameras_id:
            location = sampler.get_sample(episode_id, id_camera, location_modality, n)
            bbs = sampler.get_sample(episode_id, id_camera, bbs_modality, n)
            depth = sampler.get_sample(episode_id, id_camera, depth_modality, n)
            rgb = sampler.get_sample(episode_id, id_camera, rgb_modality, n)
            semantic = sampler.get_sample(episode_id, id_camera, semantic_modality, n)

            step_results[id_camera] = {
                "location": location,
                "bbs": bbs,
                'depth': depth,
                'rgb': rgb,
                'semantic': semantic,
            }
            for mod in additional_mods:
                data = sampler.get_sample(episode_id, id_camera, mod, n)
                step_results[mod] = data
        results.append(step_results)
    return results
