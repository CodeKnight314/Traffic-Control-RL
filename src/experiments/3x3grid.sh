#!/bin/bash

python main.py --config config.yaml --net routes/3x3grid/3x3Grid2lanes.net.xml --route routes/3x3grid/routes14000.rou.xml --path 3x3grid/weights/ --train --weights ../resources/weights/base_policy_weights.pth
xvfb-run --server-args="-screen 0 1024x768x24" python main.py --config config.yaml --net routes/3x3grid/3x3Grid2lanes.net.xml --route routes/3x3grid/routes14000.rou.xml --path 3x3grid/ --weight 3x3grid/weights/