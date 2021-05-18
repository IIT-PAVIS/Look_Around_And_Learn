import os
from itertools import compress

import cv2
import habitat
import hydra
import matplotlib.pyplot as plt
import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from habitat import make_dataset

from experimenting_env.sensor_data import BBSense
from experimenting_env.utils import sim_utils
from experimenting_env.utils.sensors_utils import save_obs


# For viewing the extractor outputw
def display_sample(sample):
    img = sample["rgba"]
    depth = sample["depth"]
    semantic = sample["semantic"]

    arr = [img, depth, semantic]
    titles = ["rgba", "depth", "semantic"]
    plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i + 1)
        ax.axis("off")
        ax.set_title(titles[i])
        plt.imshow(data)

    plt.show()


# import hydra
# from plyfile import PlyData, PlyElement


def count_interesting_classes(semantic_data, id_to_name):

    classes = list(
        filter(
            lambda x: (
                x in id_to_name and id_to_name[x] in BBSense.CLASSES.values()
            ),  # Count objects with "interesting" classes for our work
            np.unique(semantic_data),
        )
    )

    return len(classes)


@hydra.main(config_path='../confs/', config_name='config.yaml')
def main(cfg) -> None:
    os.symlink(cfg.data_base_dir, os.path.join(os.getcwd(), "data"))
    config = habitat.get_config(os.path.join(cfg.habitat_base_cfg_dir, cfg.habitat_cfg))
    dataset = make_dataset(config.DATASET.TYPE, config=config.DATASET)
    output_path = "/work/gscarpellini/training_scene_test"
    os.makedirs(output_path, exist_ok=True)
    N_TOT = 500 

    semantic_filter_scenes = [
        os.path.exists(
            dataset.scene_ids[i].replace("//", "/").replace(".glb", "_semantic.ply")
        )
        for i in range(len(dataset.scene_ids))
    ]
    scenes = dataset.scene_ids

    scenes = list(compress(scenes, semantic_filter_scenes))

    for scene_count, scene_id in enumerate(scenes):

        extractor = sim_utils.FirstPersonImageExtractor(
            scene_filepath=scene_id,
            img_size=(640, 640),
            output=["rgba", "depth", "semantic"],
        )

        count_samples = 0
        indexes = np.random.permutation(len(extractor))
        for idx in indexes:
            x = extractor[idx]

            if count_samples >= N_TOT:
                break

            if len(x['bbsgt']['instances']) == 0:
                continue
            save_obs(output_path, scene_count, [x], count_samples)
            count_samples += 1

        extractor.close()
        print(f"tot samples {count_samples}")
        print(f"{scene_id} completed")


if __name__ == '__main__':
    main()
