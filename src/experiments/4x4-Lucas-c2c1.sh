#!/bin/bash

python main.py --config config.yaml --net routes/4x4-Lucas/4x4.net.xml --route routes/4x4-Lucas/4x4c2c1.rou.xml --path 4x4_Lucas_c2c1/weights/ --train --weights ../resources/weights/base_policy_weights.pth
xvfb-run --server-args="-screen 0 1024x768x24" python main.py --config config.yaml --net routes/4x4-Lucas/4x4.net.xml --route routes/4x4-Lucas/4x4c2c1.rou.xml --path 4x4_Lucas_c2c1/ --weight 4x4_Lucas_c2c1/weights/