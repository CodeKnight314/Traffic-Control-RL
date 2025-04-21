#!/bin/bash

python main.py --config config.yaml --net routes/4x4-Lucas/4x4.net.xml --route routes/4x4-Lucas/4x4teste.rou.xml --path 4x4_Lucas_teste/weights/ --train --weights ../resources/weights/base_policy_weights.pth
xvfb-run --server-args="-screen 0 1024x768x24" python main.py --config config.yaml --net routes/4x4-Lucas/4x4.net.xml --route routes/4x4-Lucas/4x4teste.rou.xml --path 4x4_Lucas_teste/ --weight 4x4_Lucas_teste/weights/