#!/bin/bash

# Change to src directory
cd src

# 2way-single-intersection experiments
bash experiments/2way-single-intersection-gen.sh
bash experiments/2way-single-intersection-horizontal.sh
bash experiments/2way-single-intersection-vertical.sh
bash experiments/2way-single-intersection-vhvh.sh

# 2x2grid experiments
bash experiments/2x2grid.sh

# 3x3grid experiments
bash experiments/3x3grid.sh