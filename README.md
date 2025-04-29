# Traffic Control RL
## Results
<p align="center">
  <img src="resources/gifs/demo_mod.gif" alt="Gif of traffix sim across single intersection, 2x2 grid, and 3x3 grid">
</p>

## Overview
This project implements deep reinforcement learning to optimize traffic signal timing. Using SUMO with RL algorithms, the system learns traffic light patterns to minimize congestion and improve traffic flow. To exploit learning gains from single-intersection traffic tasks, the `DuelDQN` model is modified for easier weight transferring. This is done since the observation can be made that internal logic for `n x n` traffic control scenarios use similar internal logic, only distinguished by different observation spaces and action spaces. Using `base_policy_weights.pth`, derived from `2way-single-intersection-vhvh.sh` experiment, traffic control agent(s) are able to converge quickly and perform well under previously unseen traffic scenarios. 

For traffic scenarios with more than one intersection, multiple `TrafficAgent` models are instantiated to handle each intersection in a `n x n` grid.

## Installation
```bash
git clone https://github.com/yourusername/Traffic-Control-RL.git
cd Traffic-Control-RL
sudo bash setup.sh
```

## Usage
### Training
For Training, you can trigger the training process for a given `net` and `rou` file using the `--train` flag:
```bash
python src/main.py --config src/config.yaml --net path/to/net.xml --route path/to/rou.xml --path models/single-intersection --train
```

### Testing
For Testing, you can trigger the render process for a given `net` and `rou` file by providing `weight` args:
```bash
python src/main.py --config src/config.yaml --net path/to/net.xml --route path/to/rou.xml --path models/single-intersection --weights models/single-intersection
```
Alternatively, if you're running the rendering job on a cloud instance, you can simluate via `xvfb` with the following bash command:
```bash
xvfb-run -a -s "-screen 0 1400x900x24"     python main.py --config config.yaml --net single-intersection.net.xml --route single-intersection.rou.xml --path Outputs/
```
