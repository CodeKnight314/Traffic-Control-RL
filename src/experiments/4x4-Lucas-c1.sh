#!/bin/bash

python main.py --config config.yaml --net routes/4x4-Lucas/4x4.net.xml --route routes/4x4-Lucas/4x4c1.rou.xml --path 4x4_Lucas_c1/weights/ --train
xvfb-run --server-args="-screen 0 1024x768x24" python main.py --config config.yaml --net routes/4x4-Lucas/4x4.net.xml --route routes/4x4-Lucas/4x4c1.rou.xml --path 4x4_Lucas_c1/ --weight 4x4_Lucas_c1/weights/