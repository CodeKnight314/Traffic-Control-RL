#!/bin/bash

python main.py --config config.yaml --net routes/2way-single-intersection/single-intersection.net.xml --route routes/2way-single-intersection/single-intersection-vhvh.rou.xml --path two_way_single_intersection_vhvh/ --train
xvfb-run --server-args="-screen 0 1024x768x24" python main.py --config config.yaml --net routes/2way-single-intersection/single-intersection.net.xml --route routes/2way-single-intersection/single-intersection-vhvh.rou.xml --path two_way_single_intersection_vhvh/ --weight two_way_single_intersection_vhvh/weights/shared_policy.pth