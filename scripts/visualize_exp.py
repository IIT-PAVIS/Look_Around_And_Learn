import os

import hydra

from experimenting_env import replay_experiment


@hydra.main(config_path='../confs/', config_name='config.yaml')
def main(cfg) -> None:
    replay_experiment(
        os.path.join(cfg.exp_base_dir, cfg.replay.exp_name),
        cfg.replay.modalities,
        cfg.replay.episode_id,
        cfg.replay.cameras_id,
        cfg.replay.start_step,
        cfg.replay.end_step,
    )
    print("")


if __name__ == "__main__":
    main()
