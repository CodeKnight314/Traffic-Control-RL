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

# 4x4-Lucas experiments
bash experiments/4x4-Lucas-c1.sh
bash experiments/4x4-Lucas-c2.sh
bash experiments/4x4-Lucas-c1c2.sh
bash experiments/4x4-Lucas-c2c1.sh
bash experiments/4x4-Lucas-c1c2c1c2.sh
bash experiments/4x4-Lucas-teste.sh

# 4x4loop experiments
bash experiments/4x4loop.sh