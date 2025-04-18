# Traffic Control RL
## Overview
This project implements deep reinforcement learning to optimize traffic signal timing. Using SUMO with RL algorithms, the system learns traffic light patterns to minimize congestion and improve traffic flow.

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