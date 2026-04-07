DATA_PATH=hiveformer_raw/
ZARR_PATH=zarr_datasets/hiveformer/

# First we generate and store raw demos
seed=0
variation=0
variation_count=1
# If you change this list of tasks here, then change
# data_processing/hiveformer_to_zarr.py, ln 221 to be consistent!
tasks=(
    close_door
)

# Train demos
num_tasks=${#tasks[@]}
for ((i=0; i<$num_tasks; i++)); do
     xvfb-run -a python data_generation/generate.py \
          --save_path ${DATA_PATH}/train \
          --image_size 256,256 --renderer opengl \
          --episodes_per_task 2 \
          --tasks ${tasks[$i]} --variations ${variation_count} --offset ${variation} \
          --processes 1 --seed 0
done
# Val demos (different seed!)
num_tasks=${#tasks[@]}
for ((i=0; i<$num_tasks; i++)); do
     xvfb-run -a python data_generation/generate.py \
          --save_path ${DATA_PATH}/val \
          --image_size 256,256 --renderer opengl \
          --episodes_per_task 2 \
          --tasks ${tasks[$i]} --variations ${variation_count} --offset ${variation} \
          --processes 1 --seed 1
done