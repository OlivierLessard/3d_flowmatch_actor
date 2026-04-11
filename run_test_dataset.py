from matplotlib.pylab import sample
import torch
from torch.utils.data import DataLoader
import numpy as np
# Import your updated dataset
# Adjust "your_dataset_module" to wherever you saved PerActNerfDataset
from datasets.rlbench import PerActNerfDataset 
import matplotlib.pyplot as plt
from datasets import fetch_dataset_class

def test_dataloader(zarr_root, split="train"):
    print(f"\n--- Testing DataLoader for {split} split ---")
    
    # 1. Instantiate the Dataset
    dataset_class = fetch_dataset_class("PerActNerf")
    dataset = dataset_class(
        root=zarr_root,
        instructions="instructions/peractnerf/instructions.json", 
        copies=1,        # Keep at 1 for quick testing
        chunk_size=1     # Match your chunk size
    )
    print(f"Successfully initialized dataset! Length: {len(dataset)}")
    
    # 2. Instantiate the DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=2,    # Test with a batch > 1 to verify collation
        shuffle=True, 
        num_workers=0 
    )
    
    # Then iterated in the main training loop
    for sample in dataloader:
        print(f"Sample keys: {list(sample.keys())}")
        
        # first task
        print(f"Task in first batch item: {sample['task'][0][0]}")
        
        # first instruction
        print(f"Instruction in first batch item: {sample['instr'][0][0]}")
        
        # first RGB image
        print(f"RGB shape in first batch item: {sample['rgb'][0, 0, 0].shape}")
        img_to_show = sample['rgb'][0, 0, 0].cpu().permute(1, 2, 0).numpy()
        plt.figure(figsize=(8, 5))
        plt.imshow(img_to_show)
        plt.title(f"Key: rgb | Shape: {img_to_show.shape}")
        plt.axis('off') # Hide the pixel coordinate axes
        # plt.waitforbuttonpress()
        plt.show()
        
        # first depth image
        print(f"Depth shape in first batch item: {sample['depth'][0, 0, 0].shape}")
        depth_to_show = sample['depth'][0, 0, 0].cpu().numpy()
        plt.figure(figsize=(8, 5))
        plt.imshow(depth_to_show, cmap='gray')
        plt.title(f"Key: depth | Shape: {depth_to_show.shape}")
        plt.axis('off') # Hide the pixel coordinate axes
        # plt.waitforbuttonpress()
        plt.show()
        
        # first proprioception
        print(f"Proprioception shape in first batch item: {sample['proprioception'][0].shape}")
        prop = sample["proprioception"][0, 0, 0, :][0]  # Shape: (8,) - single element
        print(f"\nProprioception values (first batch, first history, first hand):")
        labels = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'gripper_open']
        for label, val in zip(labels, prop):
            print(f"  {label}: {float(val):.6f}") 
            
        # first action
        print(f"Action shape in first batch item: {sample['action'][0].shape}")
        print(f"\nAction values (first batch, first history, first hand):")
        action = sample["action"][0, 0, 0][0]  # Shape: (8,) - single element
        action_labels = ['delta_x', 'delta_y', 'delta_z', 'delta_qx', 'delta_qy', 'delta_qz', 'delta_qw', 'gripper_action']
        for label, val in zip(action_labels, action):
            print(f"  {label}: {float(val):.6f}")   
        
        # first extrinsics
        print(f"Extrinsics shape in first batch item: {sample['extrinsics'][0, 0, 0].shape}")
        
        # fist intrinsics
        print(f"Intrinsics shape in first batch item: {sample['intrinsics'][0].shape}")
        
        # first nerf rgb
        print(f"NeRF RGB shape in first batch item: {sample['nerf_rgb'][0, 0, 0].shape}")
        nerf_rgb_to_show = sample['nerf_rgb'][0, 0, 0].cpu().permute(1, 2, 0).numpy()
        plt.figure(figsize=(8, 5))
        plt.imshow(nerf_rgb_to_show)
        plt.title(f"Key: nerf_rgb | Shape: {nerf_rgb_to_show.shape}")
        plt.axis('off') # Hide the pixel coordinate axes
        # plt.waitforbuttonpress()
        plt.show()
        
        # frist nerf depth
        print(f"NeRF depth shape in first batch item: {sample['nerf_depth'][0, 0, 0].shape}")
        nerf_depth_to_show = sample['nerf_depth'][0, 0, 0].cpu().numpy()
        plt.figure(figsize=(8, 5))
        if nerf_depth_to_show.ndim == 3:
            nerf_depth_to_show = nerf_depth_to_show[0]
        plt.imshow(nerf_depth_to_show, cmap='gray')
        plt.title(f"Key: nerf_depth | Shape: {nerf_depth_to_show.shape}")
        plt.axis('off') # Hide the pixel coordinate axes
        # plt.waitforbuttonpress()
        plt.show()
        
        # nerf extrinsics
        print(f"NeRF extrinsics shape in first batch item: {sample['nerf_extrinsics'][0, 0, 0].shape}")
        
        # nerf intrinsics
        print(f"NeRF intrinsics shape in first batch item: {sample['nerf_intrinsics'][0, 0, 0].shape}")
        
        print("-" * 50)
        break  # Only test the first batch for quick verification
             
    print(f"\n--- Finished testing DataLoader for {split} split ---")

if __name__ == "__main__":
    # Replace with your actual Zarr path
    TRAIN_ZARR = "zarr_datasets/nerf_peract/train.zarr"
    split = "train"
    
    test_dataloader(TRAIN_ZARR, split=split)