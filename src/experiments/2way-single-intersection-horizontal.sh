#!/bin/bash

python main.py --config config.yaml --net routes/2way-single-intersection/single-intersection.net.xml --route routes/2way-single-intersection/single-intersection-horizontal.rou.xml --path two_way_single_intersection_horizontal/ --train
xvfb-run --server-args="-screen 0 1024x768x24" python main.py --config config.yaml --net routes/2way-single-intersection/single-intersection.net.xml --route routes/2way-single-intersection/single-intersection-horizontal.rou.xml --path two_way_single_intersection_horizontal/ --weight two_way_single_intersection_horizontal/weights/shared_policy.pth