#!/bin/bash

python main.py --config config.yaml --net routes/2x2grid/2x2.net.xml --route routes/2x2grid/2x2.rou.xml --path 2x2grid/weights/ --train
xvfb-run --server-args="-screen 0 1024x768x24" python main.py --config config.yaml --net routes/2x2grid/2x2.net.xml --route routes/2x2grid/2x2.rou.xml --path 2x2grid/ --weight 2x2grid/weights/