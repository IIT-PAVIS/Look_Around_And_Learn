import math
import multiprocessing
import os
import random
from itertools import compress
from typing import List

import numpy as np
import torch
from habitat import Config, RLEnv, ThreadedVectorEnv, VectorEnv, make_dataset


def make_env_fn(env_class, config, kwargs) -> RLEnv:
    r"""Creates an env of type env_class with specified config and rank.
    This is to be passed in as an argument when creating VectorEnv.
    Args:
        config: root exp config that has core env config node as well as
            env-specific config node.
        env_class: class type of the env to be created.
    Returns:
        env object created according to specification.
    """

    env = env_class(config=config, **kwargs)
    # env.seed(config.TASK_CONFIG.SEED)
    return env


def get_unique_scene_envs_generator(config, env_class, **kwargs):
    dataset = make_dataset(config.DATASET.TYPE, config=config.DATASET)

    scenes = config.DATASET.CONTENT_SCENES
    if "*" in config.DATASET.CONTENT_SCENES:
        scenes = dataset.get_scenes_to_load(config.DATASET)

    for i, scene_name in enumerate(scenes):
        task_config = config.clone()
        task_config.defrost()

        task_config.SEED = config.SEED + i
        task_config.DATASET.CONTENT_SCENES = [scene_name]

        task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = (
            task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID
        )
        task_config.freeze()
        yield env_class(config=task_config, **kwargs)


def construct_envs(
    config: Config,
    env_class: RLEnv,
    workers_ignore_signals: bool = False,
    mode="train",
    **kwargs
) -> VectorEnv:
    r"""Create VectorEnv object with specified config and env class type.
    To allow better performance, dataset are split into small ones for
    each individual env, grouped by scenes.
    :param config: configs that contain num_environments as well as information
    :param necessary to create individual environments.
    :param env_class: class type of the envs to be created.
    :param workers_ignore_signals: Passed to :ref:`habitat.VectorEnv`'s constructor
    :return: VectorEnv object created according to specification.
    """

    dataset = make_dataset(config.DATASET.TYPE, config=config.DATASET)

    scenes = config.DATASET.CONTENT_SCENES
    if "*" in config.DATASET.CONTENT_SCENES:
        scenes = dataset.get_scenes_to_load(config.DATASET)
        semantic_filter_scenes = [
            os.path.exists(
                dataset.scene_ids[i].replace("//", "/").replace(".glb", "_semantic.ply")
            )
            for i in range(len(dataset.scene_ids))
        ]

        scenes = list(compress(scenes, semantic_filter_scenes))

    (
        sim_gpu_id,
        num_processes,
        num_processes_on_first_gpu,
        num_processes_per_gpu,
    ) = get_multi_gpu_config(len(scenes))

    num_processes = min(num_processes, len(scenes))
    configs = []

    env_classes = [env_class for _ in range(num_processes)]

    kwargs_per_env = [kwargs for _ in range(num_processes)]

    random.shuffle(scenes)

    scene_splits: List[List[str]] = [[] for _ in range(num_processes)]

    for idx, scene in enumerate(scenes):
        scene_splits[idx % len(scene_splits)].append(scene)

    assert sum(map(len, scene_splits)) == len(scenes)
    for i in range(num_processes):
        task_config = config.clone()
        task_config.defrost()

        task_config.SEED = config.SEED + i
        if len(scenes) > 0:
            task_config.DATASET.CONTENT_SCENES = scene_splits[i]

        if i < num_processes_on_first_gpu:
            gpu_id = 0
        else:
            gpu_id = (i - num_processes_on_first_gpu) % (
                torch.cuda.device_count() - 1
            ) + sim_gpu_id

        task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = gpu_id
        task_config.SEED = random.randint(1, 10000)
        task_config.freeze()
        configs.append(task_config)

    envs = VectorEnv(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(zip(env_classes, configs, kwargs_per_env)),
        workers_ignore_signals=workers_ignore_signals,
     
    )

    return envs


def get_multi_gpu_config(num_scenes=25, x=10):
    # Automatically configure number of training threads based on
    # number of GPUs available and GPU memory size
    total_num_scenes = num_scenes
    gpu_memory = 100
    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        gpu_memory = min(
            gpu_memory,
            torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024,
        )
        if i == 0:
            assert (
                torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024
                > 10.0
            ), "Insufficient GPU memory"

    num_processes_per_gpu = int(gpu_memory / 1.4)

    num_processes_on_first_gpu = int((gpu_memory - x) / 1.4)

    sim_gpu_id = 0

    if num_gpus == 1:
        num_processes_on_first_gpu = num_processes_on_first_gpu
        num_processes_per_gpu = 0
        num_processes = num_processes_on_first_gpu
    else:
        total_threads = (
            num_processes_per_gpu * (num_gpus - 1) + num_processes_on_first_gpu
        )

        num_scenes_per_thread = math.ceil(total_num_scenes / total_threads)
        num_threads = math.ceil(total_num_scenes / num_scenes_per_thread)
        num_processes_per_gpu = min(
            num_processes_per_gpu, math.ceil(num_threads // (num_gpus - 1))
        )

        num_processes_on_first_gpu = max(
            0, num_threads - num_processes_per_gpu * (num_gpus - 1)
        )

        num_processes = num_processes_on_first_gpu + num_processes_per_gpu * (
            num_gpus - 1
        )  # num_threads

        sim_gpu_id = 1

    print("Auto GPU config:")
    print("Number of processes: {}".format(num_processes))
    print("Number of processes on GPU 0: {}".format(num_processes_on_first_gpu))
    print("Number of processes per GPU: {}".format(num_processes_per_gpu))
    return sim_gpu_id, num_processes, num_processes_on_first_gpu, num_processes_per_gpu
