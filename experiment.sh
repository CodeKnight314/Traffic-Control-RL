#!/bin/bash
cd src

# 2way-single-intersection
python main.py --config config.yaml --net routes/2way-single-intersection/single-intersection.net.xml --route routes/2way-single-intersection/single-intersection-gen.rou.xml --output two_way_single_intersection_gen/ --train
python main.py --config config.yaml --net routes/2way-single-intersection/single-intersection.net.xml --route routes/2way-single-intersection/single-intersection-horizontal.rou.xml --output two_way_single_intersection_horizontal/ --train
python main.py --config config.yaml --net routes/2way-single-intersection/single-intersection.net.xml --route routes/2way-single-intersection/single-intersection-vertical.rou.xml --output two_way_single_intersection_vertical/ --train
python main.py --config config.yaml --net routes/2way-single-intersection/single-intersection.net.xml --route routes/2way-single-intersection/single-intersection-vhvh.rou.xml --output two_way_single_intersection_vhvh/ --train

# 2x2grid
python main.py --config config.yaml --net routes/2x2grid/2x2.net.xml --route routes/2x2grid/2x2.rou.xml --output 2x2grid/ --train

# 3x3grid
python main.py --config config.yaml --net routes/3x3grid/3x3Grid2lanes.net.xml --route routes/3x3grid/routes14000.rou.xml --output 3x3grid/ --train

# 4x4-Lucas
python main.py --config config.yaml --net routes/4x4-Lucas/4x4.net.xml --route routes/4x4-Lucas/4x4c1.rou.xml --output 4x4_Lucas_c1/ --train
python main.py --config config.yaml --net routes/4x4-Lucas/4x4.net.xml --route routes/4x4-Lucas/4x4c2.rou.xml --output 4x4_Lucas_c2/ --train
python main.py --config config.yaml --net routes/4x4-Lucas/4x4.net.xml --route routes/4x4-Lucas/4x4c1c2.rou.xml --output 4x4_Lucas_c1c2/ --train
python main.py --config config.yaml --net routes/4x4-Lucas/4x4.net.xml --route routes/4x4-Lucas/4x4c2c1.rou.xml --output 4x4_Lucas_c2c1/ --train
python main.py --config config.yaml --net routes/4x4-Lucas/4x4.net.xml --route routes/4x4-Lucas/4x4c1c2c1c2.rou.xml --output 4x4_Lucas_c1c2c1c2/ --train
python main.py --config config.yaml --net routes/4x4-Lucas/4x4.net.xml --route routes/4x4-Lucas/4x4teste.rou.xml --output 4x4_Lucas_teste/ --train

# 4x4loop
python main.py --config config.yaml --net routes/4x4loop/4x4loop.net.xml --route routes/4x4loop/4x4loop.rou.xml --output 4x4loop/ --train