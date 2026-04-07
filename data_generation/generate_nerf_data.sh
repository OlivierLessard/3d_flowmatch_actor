#!/bin/bash
# data_generation/generate_nerf_data.sh

# Base path for the final dataset
BASE_DATA_PATH="manigaussian_nerf_raw"

# Define tasks to generate
tasks=(
    open_drawer
    close_jar
    push_buttons
    stack_blocks
)

# Loop through each task
for task in "${tasks[@]}"; do
    echo "------------------------------------------------"
    echo "Processing Task: $task"
    echo "------------------------------------------------"

    # 1. Generate Training Data
    echo "Generating TRAIN data for $task..."
    xvfb-run -a python RLBench/tools/nerf_dataset_generator.py \
        --tasks=$task \
        --save_path="${BASE_DATA_PATH}/train" \
        --image_size=128,128 \
        --renderer=opengl \
        --episodes_per_task=20 \
        --processes=1 \
        --all_variations=True

    # 2. Generate Validation/Test Data
    echo "Generating TEST data for $task..."
    xvfb-run -a python RLBench/tools/nerf_dataset_generator.py \
        --tasks=$task \
        --save_path="${BASE_DATA_PATH}/test" \
        --image_size=128,128 \
        --renderer=opengl \
        --episodes_per_task=25 \
        --processes=1 \
        --all_variations=True

done

echo "Data generation complete!"