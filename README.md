# [Look around and learn](https://iit-pavis.github.io/Look_Around_And_Learn/)

## Setup
- Clone this repository
- Clone submodules: `git submodule update --init --recursive`
- Create conda env from `env.yml` file: `conda env create -f env.yml`
- Install dependencies `python -m pip install requirements.txt`
- Install pkg `python setup.py develop`
- Move into each subdir inside `third_parties` and execute `python setup.py
  develop --all`
- Specify `base_dir` at `confs/config.yaml` as the absolute path of this project
   
## Data
All the data are contained inside the `data` directory.

### Example: habitat test scenes
1. Download test-scenes `http://dl.fbaipublicfiles.com/habitat/habitat-test-scenes.zip`
2. Unzip inside project directory

- Suggestion: you can keep data in a separate folder and use soft links (`ln -s
  /path/to/dataset /path/to/project/data`)

## Launch an experiment
`python scripts/run_exp.py` run training or deployment of a policy. More information about "RL baselines" and our RL policy below.

## Replay experiment
To replay an experiment, use the following
`python scripts/visualize_exp.py replay.episode_id={ID episode} replay.exp_name={PATH TO EPISODE} replay.modalities="['rgb', 'depth','semantic']"`

## Run baselines
The following learned baselines are implemented:
- `neuralslam`: start from `confs/habitat/gibson_neuralslam.yaml`
- `seal-v0`: start from `confs/habitat/gibson_seal.yaml`
- `curiosity-v0`: start from `confs/habitat/gibson_semantic_curiosity.yaml`

The following classical baselines are implemented:
- `randomgoalsbaseline`
- `frontierbaseline-v1` (`frontierbaseline-v2`, `frontierbaseline-v3`) 
- `bouncebaseline`
- `rotatebaseline`
- `randombaseline`

## Train goalexploration policy
Start from `confs/habitat/gibson_goal_exploration.yaml`

- `CHECKPOINT_FOLDER` folder in which checkpoints are saved
- `TOTAL_NUM_STEPS` max number of training steps
- under `ppo`:
  - `replanning_steps` how often to run the policy
  - `num_global_steps` how often to train the policy
  - `save_periodic` how often to save a checkpoint
  - `load_checkpoint_path` full path to a checkpoint to load at start
  - `load_checkpoint` set True to load `load_checkpoint_path`
  - `visualize` if True, debug images are shown

Environments:
- `SemanticDisagreement-v0`  reward: sum(disagreement_t)

Environments for the RL baselines are also provided:
- `SemanticCuriosity-v0` (Semantic Curiosity)
- `sealenv-v0` (SEAL)
- `ExpSlam-v0` (NeuralSLAM)

Policies:
- `goalexplorationbaseline-v0`  State: disagreement_t, map_t, agent pose

Checkpoints:
- Ours {ADD LINK}

## Generate from goalexploration policy
Start from `confs/habitat/gibson_goal_exploration.yaml`

- `replanning_steps` how often to run the policy
- `load_checkpoint_path` full path to a checkpoint to load at start
- `load_checkpoint` set to True

## Datasets
### Scenes datasets
| Scenes models                 | Extract path                                   | Archive size |
| ---                           | ---                                            | ---          |
| [Gibson](#Gibson)             | `data/scene_datasets/gibson/{scene}.glb`       | 1.5 GB       |
| [MatterPort3D](#Matterport3D) | `data/scene_datasets/mp3d/{scene}/{scene}.glb` | 15 GB        |
|                               |                                                |              |

### Tasks 
#### Gibson object navigation (used for the main results of the paper)
You can download the task at the following link {ADD LINK}, unzip and put it in `data/datasets/objectnav/gibson/v1.1`

#### Habitat's pointnav
| Task | Scenes | Link | Extract path | Config to use | Archive size |
| --- | --- | --- | --- | --- | --- |
| [Point goal navigation](https://arxiv.org/abs/1807.06757) | Gibson | [pointnav_gibson_v1.zip](https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/gibson/v1/pointnav_gibson_v1.zip) | `data/datasets/pointnav/gibson/v1/` |  [`datasets/pointnav/gibson.yaml`](configs/datasets/pointnav/gibson.yaml) | 385 MB |
| [Point goal navigation corresponding to Sim2LoCoBot experiment configuration](https://arxiv.org/abs/1912.06321) | Gibson | [pointnav_gibson_v2.zip](https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/gibson/v2/pointnav_gibson_v2.zip) | `data/datasets/pointnav/gibson/v2/` |  [`datasets/pointnav/gibson_v2.yaml`](configs/datasets/pointnav/gibson_v2.yaml) | 274 MB |
| [Point goal navigation](https://arxiv.org/abs/1807.06757) | MatterPort3D | [pointnav_mp3d_v1.zip](https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/mp3d/v1/pointnav_mp3d_v1.zip) | `data/datasets/pointnav/mp3d/v1/` | [`datasets/pointnav/mp3d.yaml`](configs/datasets/pointnav/mp3d.yaml) | 400 MB |

### MatterPort
- Follow instruction in the main [Habitat-lab](https://github.com/facebookresearch/habitat-lab) repository 

### Gibson
- Ask for the license from Gibson website `https://stanfordvl.github.io/iGibson/dataset.html`
- Download gibson tiny with `wget https://storage.googleapis.com/gibson_scenes/gibson_tiny.tar.gz`
- Follow instructions at [Habitat-sim](https://github.com/facebookresearch/habitat-sim) to generate gibson semantic

### Dependencies
- detectron2 >= 0.5
- torch >= 1.9
- pytorch-lightning >= 1.5
- habitat-sim = 0.2
- habitat-lab
- torchmetrics >= 0.6

## Contributing
If you want to contribute to the project, I suggest to install `requirements-dev.txt` and abilitate pre-commit
```
python -m pip install -r requirements-dev.txt
pre-commit install
```
