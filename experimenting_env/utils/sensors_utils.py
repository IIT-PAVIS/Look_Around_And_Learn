import os
from dataclasses import dataclass

import numpy as np


def _get_info_from_string(path, info, split_symbol="_"):

    filename = os.path.split(os.path.splitext(path)[0])[1]
    return filename[filename.find(info) :].split(split_symbol)[1]


@dataclass
class SenseInfo:
    """Class for keeping track of an item in inventory."""

    base_path: str
    mod: str
    episode: int = 0
    camera_id: int = 0
    step: int = 0

    def get_path(self) -> str:
        return os.path.join(
            self.base_path,
            f"episode_{self.episode:06d}_step_{self.step:05d}_modality_{self.mod}_id_{self.camera_id}.npy",
        )


def get_sense_info(path):

    base_path = os.path.dirname(path)

    episode = int(_get_info_from_string(path, "episode"))
    mod = _get_info_from_string(path, "modality")
    idx = int(_get_info_from_string(path, "id"))
    step = int(_get_info_from_string(path, "step"))
    return SenseInfo(base_path, mod, episode, idx, step)


def save_obs(exp_path, episode_id, observations, timestamp):

    paths = []
    for camera_id, camera_obs in enumerate(observations):
        for modality, data in camera_obs.items():
            saved_path = _save_data(
                exp_path,
                int(episode_id),
                modality,
                int(camera_id),
                int(timestamp),
                data,
            )
            paths.append(saved_path)
    return paths


def _save_data(exp_path, episode_id, modality, camera_id, timestamp, data):

    path = f"{exp_path}/episode_{episode_id:06d}_step_{timestamp:05d}_modality_{modality}_id_{camera_id}.npy"

    np.save(
        path,
        data,
    )
    return path
