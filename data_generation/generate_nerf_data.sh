#!/bin/bash
# data_generation/generate_nerf_data.sh
# Generate NeRF data - exact same as ManiGaussian

DATA_PATH=manigaussian_nerf_raw/

tasks=(
    open_drawer
    close_jar
    push_buttons
    stack_blocks
)

# Run using the RLBench folder from ManiGaussian (it has the nerf generator python script)
num_tasks=${#tasks[@]}
for ((i=0; i<$num_tasks; i++)); do
     xvfb-run -a python RLBench/tools/nerf_dataset_generator.py \  
          --save_path ${DATA_PATH} \
          --image_size 128,128 --renderer opengl \
          --episodes_per_task 20 \
          --tasks ${tasks[$i]} \
          --processes 1 \
          --all_variations True
done