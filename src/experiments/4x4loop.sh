#!/bin/bash

python main.py --config config.yaml --net routes/4x4loop/4x4loop.net.xml --route routes/4x4loop/4x4loop.rou.xml --path 4x4loop/weights/ --train
xvfb-run --server-args="-screen 0 1024x768x24" python main.py --config config.yaml --net routes/4x4loop/4x4loop.net.xml --route routes/4x4loop/4x4loop.rou.xml --path 4x4loop/ --weight 4x4loop/weights/