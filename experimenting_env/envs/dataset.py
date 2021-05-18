#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import json
import os
from typing import List, Optional

import attr
import magnum as mn
from habitat.config import Config
from habitat.core.dataset import ALL_SCENES_MASK, Dataset
from habitat.core.registry import registry
from habitat.tasks.nav.nav import (
    NavigationEpisode,
    NavigationGoal,
    ShortestPathPoint,
)

CONTENT_SCENES_PATH_FIELD = "content_scenes_path"
DEFAULT_SCENE_PATH_PREFIX = "data/scene_datasets/"


@attr.s(auto_attribs=True, kw_only=True)
class ObjectGoalNavEpisode(NavigationEpisode):
    r"""ObjectGoal Navigation Episode

    :param object_category: Category of the obect
    """
    object_category: Optional[str] = None
    object_id: Optional[str] = None
    floor_id: Optional[str] = None

    @property
    def goals_key(self) -> str:
        r"""The key to retrieve the goals"""
        return f"{os.path.basename(self.scene_id)}_{self.object_category}"


@registry.register_dataset(name="ExpNav-v1")
class ExpNavDatasetV1(Dataset):
    r"""Class inherited from Dataset that loads Point Navigation dataset."""

    episodes: List[NavigationEpisode]
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"

    @staticmethod
    def check_config_paths_exist(config: Config) -> bool:
        return os.path.exists(
            config.DATA_PATH.format(split=config.SPLIT)
        ) and os.path.exists(config.SCENES_DIR)

    @classmethod
    def get_scenes_to_load(cls, config: Config) -> List[str]:
        r"""Return list of scene ids for which dataset has separate files with
        episodes.
        """

        assert cls.check_config_paths_exist(config)
        dataset_dir = os.path.dirname(config.DATA_PATH.format(split=config.SPLIT))

        cfg = config.clone()
        cfg.defrost()
        cfg.CONTENT_SCENES = []
        dataset = cls(cfg)

        has_individual_scene_files = os.path.exists(
            dataset.content_scenes_path.split("{scene}")[0].format(
                data_path=dataset_dir
            )
        )
        if has_individual_scene_files:
            return cls._get_scenes_from_folder(
                content_scenes_path=dataset.content_scenes_path,
                dataset_dir=dataset_dir,
            )
        else:
            # Load the full dataset, things are not split into separate files
            cfg.CONTENT_SCENES = [ALL_SCENES_MASK]
            dataset = cls(cfg)
            return list(map(cls.scene_from_scene_path, dataset.scene_ids))

    @staticmethod
    def _get_scenes_from_folder(
        content_scenes_path: str, dataset_dir: str
    ) -> List[str]:
        scenes: List[str] = []

        content_dir = content_scenes_path.split("{scene}")[0]
        scene_dataset_ext = content_scenes_path.split("{scene}")[1]
        content_dir = content_dir.format(data_path=dataset_dir)
        if not os.path.exists(content_dir):
            return scenes

        for filename in os.listdir(content_dir):
            if filename.endswith(scene_dataset_ext):
                scene = filename[: -len(scene_dataset_ext)]
                scenes.append(scene)
        scenes.sort()
        return scenes

    def __init__(self, config: Optional[Config] = None) -> None:

        self.episodes = []

        if config is None:
            return

        datasetfile_path = config.DATA_PATH.format(split=config.SPLIT)

        with gzip.open(datasetfile_path, "rt") as f:
            self.from_json(f.read(), scenes_dir=config.SCENES_DIR)

        # Read separate file for each scene
        dataset_dir = os.path.dirname(datasetfile_path)
        all_scenes = os.listdir(os.path.join(dataset_dir, "content"))
        has_individual_scene_files = os.path.exists(
            self.content_scenes_path.split("{scene}")[0].format(data_path=dataset_dir)
        )

        repeat_factor = config.REPEAT if "REPEAT" in config else 1
        if has_individual_scene_files:
            scenes = config.CONTENT_SCENES
            if ALL_SCENES_MASK in scenes:
                scenes = self._get_scenes_from_folder(
                    content_scenes_path=self.content_scenes_path,
                    dataset_dir=dataset_dir,
                )

            for scene in scenes:
                scene_filename = self.content_scenes_path.format(
                    data_path=dataset_dir, scene=scene
                )
                with gzip.open(scene_filename, "rt") as f:
                    self.from_json(
                        f.read(), scenes_dir=config.SCENES_DIR, repeat=repeat_factor
                    )

        else:
            self.episodes = list(
                filter(self.build_content_scenes_filter(config), self.episodes)
            )
        # NOTE: objectnavv1.1 provides a single episode for each scene

        ids = {
            item.split(".")[0]: i * repeat_factor + 100
            for i, item in enumerate(all_scenes)
        }

        for idx in range(len(self.episodes)):

            episode_scene = os.path.split(self.episodes[idx].scene_id)[-1].split(".")[0]
            id_episode = ids[episode_scene]

            self.episodes[idx].start_position[0] = 0.0
            self.episodes[idx].start_position[2] = 0.0
            self.episodes[idx].episode_id = str(id_episode)
            ids[episode_scene] += 1

    def from_json(
        self,
        json_str: str,
        scenes_dir: Optional[str] = None,
        object_category=None,
        repeat=1,
    ) -> None:
        deserialized = json.loads(json_str)
        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

        for episode_cfg in deserialized["episodes"]:

            for _ in range(repeat):
                if "object_category" in episode_cfg:
                    continue
                episode = NavigationEpisode(**episode_cfg)

                if scenes_dir is not None:
                    if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                        episode.scene_id = episode.scene_id[
                            len(DEFAULT_SCENE_PATH_PREFIX) :
                        ]

                    episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

                for g_index, goal_cfg in enumerate(episode.goals):
                    if isinstance(goal_cfg, NavigationGoal):
                        episode.goals[g_index] = goal_cfg
                    else:
                        episode.goals[g_index] = NavigationGoal(**goal_cfg)
                if episode.shortest_paths is not None:
                    for path in episode.shortest_paths:
                        for p_index, point_cfg in enumerate(path):
                            path[p_index] = ShortestPathPoint(**point_cfg)
                episode.start_position[0] = 0.0
                episode.start_position[2] = 0.0
                self.episodes.append(episode)


@registry.register_dataset(name="ExpNav-v2")
class ExpNavDatasetV2(Dataset):
    r"""Class inherited from Dataset that loads Point Navigation dataset."""

    episodes: List[NavigationEpisode]
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"

    @staticmethod
    def check_config_paths_exist(config: Config) -> bool:
        return os.path.exists(
            config.DATA_PATH.format(split=config.SPLIT)
        ) and os.path.exists(config.SCENES_DIR)

    @classmethod
    def get_scenes_to_load(cls, config: Config) -> List[str]:
        r"""Return list of scene ids for which dataset has separate files with
        episodes.
        """

        assert cls.check_config_paths_exist(config)
        dataset_dir = os.path.dirname(config.DATA_PATH.format(split=config.SPLIT))

        cfg = config.clone()
        cfg.defrost()
        cfg.CONTENT_SCENES = []
        dataset = cls(cfg)

        has_individual_scene_files = os.path.exists(
            dataset.content_scenes_path.split("{scene}")[0].format(
                data_path=dataset_dir
            )
        )
        if has_individual_scene_files:
            return cls._get_scenes_from_folder(
                content_scenes_path=dataset.content_scenes_path,
                dataset_dir=dataset_dir,
            )
        else:
            # Load the full dataset, things are not split into separate files
            cfg.CONTENT_SCENES = [ALL_SCENES_MASK]
            dataset = cls(cfg)
            return list(map(cls.scene_from_scene_path, dataset.scene_ids))

    @staticmethod
    def _get_scenes_from_folder(
        content_scenes_path: str, dataset_dir: str
    ) -> List[str]:
        scenes: List[str] = []

        content_dir = content_scenes_path.split("{scene}")[0]
        scene_dataset_ext = content_scenes_path.split("{scene}")[1]
        content_dir = content_dir.format(data_path=dataset_dir)
        if not os.path.exists(content_dir):
            return scenes

        for filename in os.listdir(content_dir):
            if filename.endswith(scene_dataset_ext):
                scene = filename[: -len(scene_dataset_ext)]
                scenes.append(scene)
        scenes.sort()
        return scenes

    def __init__(self, config: Optional[Config] = None) -> None:

        self.episodes = []

        if config is None:
            return

        datasetfile_path = config.DATA_PATH.format(split=config.SPLIT)

        with gzip.open(datasetfile_path, "rt") as f:
            self.from_json(f.read(), scenes_dir=config.SCENES_DIR)

        # Read separate file for each scene
        dataset_dir = os.path.dirname(datasetfile_path)
        all_scenes = os.listdir(os.path.join(dataset_dir, "content"))
        has_individual_scene_files = os.path.exists(
            self.content_scenes_path.split("{scene}")[0].format(data_path=dataset_dir)
        )

        repeat_factor = config.REPEAT if "REPEAT" in config else 1
        if has_individual_scene_files:
            scenes = config.CONTENT_SCENES
            if ALL_SCENES_MASK in scenes:
                scenes = self._get_scenes_from_folder(
                    content_scenes_path=self.content_scenes_path,
                    dataset_dir=dataset_dir,
                )

            for scene in scenes:
                scene_filename = self.content_scenes_path.format(
                    data_path=dataset_dir, scene=scene
                )
                with gzip.open(scene_filename, "rt") as f:
                    self.from_json(
                        f.read(), scenes_dir=config.SCENES_DIR, repeat=repeat_factor
                    )

        else:
            self.episodes = list(
                filter(self.build_content_scenes_filter(config), self.episodes)
            )
        # NOTE: mp3d provides a single episode for each scene

        ids = {item.split(".")[0]: 0 for i, item in enumerate(all_scenes)}

        episodes_unique = []
        episode_id = 0

        for idx in range(len(self.episodes)):

            episode_scene = os.path.split(self.episodes[idx].scene_id)[-1].split(".")[0]
            id_episode = ids[episode_scene]
            if id_episode > 0:
                continue
            self.episodes[idx].episode_id = episode_id
            episode_id += 1
            episodes_unique.append(self.episodes[idx])
            ids[episode_scene] += 1
        self.episodes = episodes_unique

    def from_json(
        self,
        json_str: str,
        scenes_dir: Optional[str] = None,
        object_category=None,
        repeat=1,
    ) -> None:
        deserialized = json.loads(json_str)
        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

        for episode_cfg in deserialized["episodes"]:

            for _ in range(repeat):
                if "object_category" in episode_cfg:
                    continue
                episode = NavigationEpisode(**episode_cfg)

                if scenes_dir is not None:
                    if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                        episode.scene_id = episode.scene_id[
                            len(DEFAULT_SCENE_PATH_PREFIX) :
                        ]

                    episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

                for g_index, goal_cfg in enumerate(episode.goals):
                    if isinstance(goal_cfg, NavigationGoal):
                        episode.goals[g_index] = goal_cfg
                    else:
                        episode.goals[g_index] = NavigationGoal(**goal_cfg)
                if episode.shortest_paths is not None:
                    for path in episode.shortest_paths:
                        for p_index, point_cfg in enumerate(path):
                            path[p_index] = ShortestPathPoint(**point_cfg)

                self.episodes.append(episode)
