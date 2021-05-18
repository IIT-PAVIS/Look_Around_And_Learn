import math
import multiprocessing
import os
from itertools import chain
import wandb
import cv2
import habitat
import hydra
import magnum as mn
import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations import fog_of_war, maps
from habitat_baselines.agents.simple_agents import RandomAgent
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_sim.utils import common as sim_utils

from experimenting_env.replay import SampleLoader
from experimenting_env.utils.sensors_utils import save_obs


def run_exp(config, run_type: str, **kwargs) -> None:
    r"""This function runs the specified config with the specified runtype
    Args:
    config: Habitat.config
    runtype: str {train or eval}
    """

    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config, **kwargs)

    if run_type == "train":
        trainer.train()
    elif run_type == "generate":
        trainer.generate()


@hydra.main(config_path='../confs/', config_name='train_policy.yaml')
def main(cfg) -> None:
    if not (os.path.exists(os.path.join(os.getcwd(), "data"))):
        os.symlink(cfg.data_base_dir, os.path.join(os.getcwd(), "data"))
    config = habitat.get_config(os.path.join(cfg.habitat_base_cfg_dir, cfg.habitat_cfg))
    wandb.init(
            name=cfg.exp_name,
            config=config,
            project='look_around',
            entity='')

    run_exp(config, cfg.MODE if 'MODE' in cfg else 'train', **cfg)


if __name__ == '__main__':
    main()
