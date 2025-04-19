#!/bin/bash

python main.py --config config.yaml --net routes/2way-single-intersection/single-intersection.net.xml --route routes/2way-single-intersection/single-intersection-gen.rou.xml --path two_way_single_intersection_gen/weights/ --train
xvfb-run --server-args="-screen 0 1024x768x24" python main.py --config config.yaml --net routes/2way-single-intersection/single-intersection.net.xml --route routes/2way-single-intersection/single-intersection-gen.rou.xml --path two_way_single_intersection_gen/ --weight two_way_single_intersection_gen/weights/shared_policy.pth