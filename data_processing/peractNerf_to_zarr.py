import argparse
import json
import os
from pathlib import Path
import pickle

import numpy as np
import zarr
from numcodecs import Blosc
from tqdm import tqdm
from PIL import Image

from data_processing.rlbench_utils import (
    keypoint_discovery,
    image_to_float_array,
    quat_to_euler_np,
    interpolate_trajectory,
    euler_to_quat_np,
    store_instructions
)
import numcodecs
from utils.common_utils import str2bool

STORE_EVERY = 1
NCAM = 5
NHAND = 1
IM_SIZE = 128
DEPTH_SCALE = 2**24 - 1
CAMERAS = ["left_shoulder", "right_shoulder", "overhead", "wrist", "front"]
INTERP_LEN = 50
NCAM_NERF = 20


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, default='/home/olivier/Desktop/ManiGaussian/data/')
    parser.add_argument('--tgt', type=str, required=True, default='/home/olivier/Desktop/3d_flowmatch_actor/zarr_datasets/nerf_peract/')   
    parser.add_argument('--store_trajectory', type=str2bool, default=True)
    return parser.parse_args()


def load_image(path):
    img = Image.open(path).resize((IM_SIZE, IM_SIZE))
    return np.array(img).transpose(2, 0, 1)  # (3,H,W)


def to_numpy(x):
    return np.array(x, dtype=np.float32)


# ----------------------------
# 🔑 LOAD NERF RGB
# ----------------------------
def load_nerf_rgb(episode_path, key_frames):
    """
    Expected structure:
    nerf_data/
    
        cam0/
            0.png, 1.png ...
        cam1/
        ...
    """

    rgb_seq = []

    for k in key_frames[:-1]:
        cam_rgb = []
        # nerf_data layout: nerf_data/{view_idx}/images/{cam_idx}.png
        for nerf_cam in range(NCAM_NERF):
            img_path = episode_path / "nerf_data" / str(k) / "images" / f"{nerf_cam}.png"
            cam_rgb.append(load_image(img_path))
        rgb_seq.append(np.stack(cam_rgb))  # (NCAM_NERF, 3, H, W)

    return np.stack(rgb_seq)  # (T, NCAM, 3, H, W)


def load_low_dim(episode_path):
    """
    Load low-dimensional observations from a pickle file.
    
    :param episode_path: Path to the episode directory containing the low_dim_obs.pkl file.
    :return: Tuple of numpy arrays (proprioception, actions)
    """
    # read observation file
    with open(episode_path / "low_dim_obs.pkl", "rb") as f:
        data = pickle.load(f)
    T = len(data)
    proprioceptive = []
    actions = []

    # loop through time steps and extract proprioception and actions
    for t in range(T):
        obs = data[t]

        # ⚠️ ADJUST KEYS IF NEEDED
        p = np.concatenate([
            obs["gripper_pose"],        # (7)
            [obs["gripper_open"]]       # (1)
        ])

        action = obs["action"]

        proprioceptive.append(p)
        actions.append(action)

    proprioceptive = np.array(proprioceptive, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)

    return proprioceptive, actions


# ----------------------------
# 🔑 TEMPORAL STACKING (PerAct style)
# ----------------------------
def build_proprio(prop):
    prop_1 = np.concatenate([prop[:1], prop[:-1]])
    prop_2 = np.concatenate([prop_1[:1], prop_1[:-1]])

    prop = np.concatenate([prop_2, prop_1, prop], axis=1)
    prop = prop.reshape(len(prop), 3, NHAND, -1)

    return prop

def load_pose_file(p: str):
    """
    Load a single pose file which contains a 4x4 extrinsics matrix
    followed by a 3x3 intrinsics matrix (as in your example).
    Returns (extrinsics, intrinsics) as numpy arrays (float32).
    """
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(f"No such pose file: {p}")

    text = p.read_text().strip()
    # keep only non-empty lines (handles blank line between blocks)
    lines = [ln for ln in text.splitlines() if ln.strip() != '']
    if len(lines) < 7:
        raise ValueError(f"Unexpected pose file format ({p}): need >=7 non-empty lines")

    # first 4 lines -> 4x4 extrinsics
    extrinsics = np.array([[float(x) for x in lines[i].split()] for i in range(4)], dtype=np.float32)
    # next 3 lines -> 3x3 intrinsics
    intrinsics = np.array([[float(x) for x in lines[i].split()] for i in range(4, 7)], dtype=np.float32)

    return extrinsics, intrinsics

def build_action(actions):
    return actions.reshape(len(actions), 1, NHAND, -1)

def _interpolate(traj, num_steps):
    # Convert to Euler
    traj = np.concatenate((
        traj[:, :3],
        quat_to_euler_np(traj[:, 3:7]),
        traj[:, 7:]
    ), 1)
    # Interpolate
    traj = interpolate_trajectory(traj, num_steps)
    # Convert to quaternion
    traj = np.concatenate((
        traj[:, :3],
        euler_to_quat_np(traj[:, 3:6]),
        traj[:, 6:]
    ), 1)
    return traj

def main(split, tasks, ROOT, STORE_PATH, store_trajectory):
    """
    This functions goes through the episodes, extracts keyframes, compute necessary information about the 
    transition between two consecutive keyframes and stores them in a Zarr file with the following fields.
    
    Results after running the main function:
    **** Finished processing split train. Total episodes stored: 200 Total keyframes stored: 1502
    **** Finished processing split val. Total episodes stored: 250 Total keyframes stored: 1547
    
    :param split: train / val / test    
    :param tasks: List of task names
    :param ROOT: Root directory containing the data
    :param STORE_PATH: Path to store the Zarr files
    :param store_trajectory: Boolean indicating whether to store the trajectory
    """
    
    # creat zarr store file
    filename = f"{STORE_PATH}/{split}.zarr"
    if os.path.exists(filename):
        print(f"{filename} exists, skipping")
        return

    # initialize compressor and dict
    compressor = Blosc(cname='lz4', clevel=1, shuffle=Blosc.SHUFFLE)
    task2id = {task: i for i, task in enumerate(tasks)}

    with zarr.open_group(filename, mode="w") as zarr_file:

        def _create(field, shape, dtype):
            """
            Create an array. Arrays are known as "datasets" in HDF5 terminology.
            
            :param field: Name of the field to create
            :param shape: Shape of the array
            :param dtype: Data type of the array
            """
            zarr_file.create_dataset(
                field,
                shape=(0,) + shape,
                chunks=(STORE_EVERY,) + shape,
                compressor=compressor,
                dtype=dtype
            )

        # create fields in the zarr file
        create_zarr_arrays(split, store_trajectory, zarr_file, _create)

        nb_episodes = 0
        nb_keyframes = 0
        
        # loop through tasks and episodes
        for task in tasks:

            # get episodes
            print(f"Processing {task}")
            task_path = Path(ROOT) / f"{split}_data" / task / "all_variations" / "episodes"
            episodes = sorted(task_path.glob("episode*"))
            if episodes is None or len(episodes) == 0:
                print(f"No episodes found for task {task} in {task_path}, skipping")
                continue

            # loop through episodes and append to the zarr file in each field 
            for episode in tqdm(episodes):
                # load picklel files
                random_descriptions, demo, variation_number = load_pickle_files(episode)
                
                # Note: instruction are chosen randomly during training, so no need to same them here.
                # during training, use random.choice(self._instructions[task][str(variation)] to select an instruction given a variation number

                # detect Keypose discovery
                key_frames = keypoint_discovery(demo, bimanual=False)
                key_frames.insert(0, 0)
                episode_keyframes_length = len(key_frames) - 1
                nb_keyframes += episode_keyframes_length

                # Loop through keyposes and store:
                # RGB (keyframes, cameras, 3, 256, 256)
                keyframes_rgb = np.stack([
                    np.stack([
                        np.array(Image.open(f"{episode}/{cam}_rgb/{k}.png")) for cam in CAMERAS
                    ])
                    for k in key_frames[:-1]
                ])
                keyframes_rgb = keyframes_rgb.transpose(0, 1, 4, 2, 3)
                
                # Loop through keyposes of the episode and store:
                # Depth (keyframes, cameras, 256, 256)
                depth_list = []
                for k in key_frames[:-1]:
                    cam_d = []
                    for cam in CAMERAS:
                        keyframes_depth = image_to_float_array(Image.open(
                            f"{episode}/{cam}_depth/{k}.png"
                        ), DEPTH_SCALE)
                        near = demo[k].misc[f'{cam}_camera_near']
                        far = demo[k].misc[f'{cam}_camera_far']
                        keyframes_depth = near + keyframes_depth * (far - near)
                        cam_d.append(keyframes_depth)
                    depth_list.append(np.stack(cam_d).astype(np.float16))
                keyframes_depth = np.stack(depth_list)
                
                # Loop through keyposes of the episode and store:
                # Proprioception (keyframes, 3, 1, 8)
                keyframes_states = np.stack([np.concatenate([demo[k].gripper_pose, [demo[k].gripper_open]]) for k in key_frames]).astype(np.float32)
                
                # Store current eef pose as well as two previous ones
                prop = keyframes_states[:-1]
                prop_1 = np.concatenate([prop[:1], prop[:-1]])
                prop_2 = np.concatenate([prop_1[:1], prop_1[:-1]])
                prop = np.concatenate([prop_2, prop_1, prop], 1)
                prop = prop.reshape(len(prop), 3, NHAND, 8)
                
                # Action (keyframes, 1, 1, 8)
                if not store_trajectory:
                    keyframes_actions = keyframes_states[1:].reshape(len(keyframes_states[1:]), 1, NHAND, 8)
                else:
                    keyframes_states = np.stack([np.concatenate([
                        demo[k].gripper_pose, [demo[k].gripper_open]
                    ]) for k in np.arange(len(demo))]).astype(np.float32)
                    keyframes_actions = np.ascontiguousarray([
                        _interpolate(keyframes_states[prev:next_ + 1], INTERP_LEN)
                        for prev, next_ in zip(key_frames[:-1], key_frames[1:])
                    ])
                    keyframes_actions = keyframes_actions.reshape(-1, INTERP_LEN, NHAND, 8)

                # Extrinsics (keyframes, cameras, 4, 4)
                keyframes_extrinsics = np.stack([
                    np.stack([
                        demo[k].misc[f'{cam}_camera_extrinsics'].astype(np.float16)
                        for cam in CAMERAS
                    ])
                    for k in key_frames[:-1]
                ])

                # Intrinsics (keyframes, cameras, 3, 3)
                keyframes_intrinsics = np.stack([
                    np.stack([
                        demo[k].misc[f'{cam}_camera_intrinsics'].astype(np.float16)
                        for cam in CAMERAS
                    ])
                    for k in key_frames[:-1]
                ])
                
                task_id = task2id[task]

                # nerf data is currently only stored for train split
                if split == "train":
                    
                    # load NeRF rgb
                    # nerf_rgb (keyframes, Nnerf cameras, 3, 256, 256)
                    nerf_rgb = np.stack([
                        np.stack([
                            np.array(Image.open(
                                f"{episode}/nerf_data/{k}//images/{nerf_cam}.png"
                            ))
                            for nerf_cam in range(NCAM_NERF)
                        ])
                        for k in key_frames[:-1]
                    ])
                    nerf_rgb = nerf_rgb.transpose(0, 1, 4, 2, 3)
                                    
                                    
                    # nerf_depth (keyframes, Nnerf cameras, 3, 256, 256)
                    nerf_depth = np.stack([
                        np.stack([
                            np.array(Image.open(
                                f"{episode}/nerf_data/{k}/depths/{nerf_cam}.png"
                            ))
                            for nerf_cam in range(NCAM_NERF)
                        ])
                        for k in key_frames[:-1]
                    ])
                    nerf_depth = nerf_depth.transpose(0, 1, 4, 2, 3)
                    
                    # Loop through nerf keyposes and store:
                    nerf_extrinsics = np.zeros((len(key_frames[:-1]), NCAM_NERF, 4, 4), dtype=np.float32)
                    nerf_intrinsics = np.zeros((len(key_frames[:-1]), NCAM_NERF, 3, 3), dtype=np.float32)

                    for idx, k in enumerate(key_frames[:-1]):
                        # nerf_data layout: nerf_data/{view_idx}/poses/{cam_idx}.txt
                        poses_dir = episode / 'nerf_data' / str(k) / 'poses'
                        if not poses_dir.exists():
                            raise FileNotFoundError(f"Missing poses dir: {poses_dir}")
                        for cam in range(NCAM_NERF):
                            pose_file = poses_dir / f"{cam}.txt"
                            if not pose_file.exists():
                                raise FileNotFoundError(f"Missing pose file: {pose_file}")
                            E, K = load_pose_file(pose_file)
                            nerf_extrinsics[idx, cam] = E
                            nerf_intrinsics[idx, cam] = K

                # for this episode write to each field in the zarr file (automatically appends along the first dimension)
                zarr_file["rgb"].append(keyframes_rgb.astype(np.uint8))
                zarr_file['depth'].append(keyframes_depth)
                zarr_file["proprioception"].append(prop)
                zarr_file["action"].append(keyframes_actions)
                zarr_file['extrinsics'].append(keyframes_extrinsics)
                zarr_file['intrinsics'].append(keyframes_intrinsics)
                repeated_variation = np.array([variation_number] * episode_keyframes_length, dtype=np.uint8)
                zarr_file['variation'].append(repeated_variation)
                repeated_task_id = np.array([task_id] * episode_keyframes_length, dtype=np.uint8)
                zarr_file['task_id'].append(repeated_task_id)  
                              
                # write nerf data
                if split == "train":
                    zarr_file["nerf_rgb"].append(nerf_rgb.astype(np.uint8))
                    zarr_file["nerf_depth"].append(nerf_depth.astype(np.float16))
                    zarr_file["nerf_extrinsics"].append(nerf_extrinsics.astype(np.float16))
                    zarr_file["nerf_intrinsics"].append(nerf_intrinsics.astype(np.float16)) 
                
                nb_episodes += 1

    print(f"**** Finished processing split {split}. Total episodes stored: {nb_episodes} Total keyframes stored: {nb_keyframes} ")

def load_pickle_files(episode):
    """
    Load pickle files for a given episode.
    variation_number is an integer identifier that specifies which variant/configuration of a robotic manipulation task to execute. 
    variation_descriptions are task description strings that describe what the specific variation should accomplish
    
    :param episode: Description
    """
    with open(episode / "variation_number.pkl", "rb") as f:
        variation = pickle.load(f)
                    
    # Load descriptions from the same level
    with open(episode / 'variation_descriptions.pkl', 'rb') as f:
        descriptions = pickle.load(f)

                # Read obs file from RLBench
    ld_file = f"{episode}/low_dim_obs.pkl"
    with open(ld_file, 'rb') as f:
        demo = pickle.load(f)
    return descriptions, demo, variation
    
def create_zarr_arrays(split, store_trajectory, zarr_file, _create):
    _create("rgb", (NCAM, 3, IM_SIZE, IM_SIZE), "uint8")
    _create("depth", (NCAM, IM_SIZE, IM_SIZE), "float16")
    _create("proprioception", (3, NHAND, 8), "float32")
    _create(
                "action",
                (1 if not store_trajectory else INTERP_LEN, NHAND, 8),
                "float32"
            ) 
    _create("extrinsics", (NCAM, 4, 4), "float16")
    _create("intrinsics", (NCAM, 3, 3), "float16")
    _create("task_id", (), "uint8")
    _create("variation", (), "uint8")

    # create fields in the zarr file
    if split == "train":
        _create("nerf_rgb", (NCAM_NERF, 3, IM_SIZE, IM_SIZE), "uint8")
        _create("nerf_depth", (NCAM_NERF, 3, IM_SIZE, IM_SIZE), "float16")  # TODO remove the 3 in shape if nerf depth is single-channel
        _create("nerf_extrinsics", (NCAM_NERF, 4, 4), "float16")
        _create("nerf_intrinsics", (NCAM_NERF, 3, 3), "float16")


if __name__ == "__main__":
    # remove previous zarr files if they exist to avoid appending to old data
    import shutil, os; shutil.rmtree('zarr_datasets/nerf_peract/train.zarr') if os.path.exists('zarr_datasets/nerf_peract/train.zarr') else None
    import shutil, os; shutil.rmtree('zarr_datasets/nerf_peract/val.zarr') if os.path.exists('zarr_datasets/nerf_peract/val.zarr') else None
    
    args = parse_arguments()
    tasks = [
        "place_cups", "close_jar", "insert_onto_square_peg",
        "light_bulb_in", "meat_off_grill", "open_drawer",
        "place_shape_in_shape_sorter", "place_wine_at_rack_location",
        "push_buttons", "put_groceries_in_cupboard",
        "put_item_in_drawer", "put_money_in_safe", "reach_and_drag",
        "slide_block_to_color_target", "stack_blocks", "stack_cups",
        "sweep_to_dustpan_of_size", "turn_tap"
    ]

    for split in ["train", "val"]:
        main(split, tasks, args.root, args.tgt, args.store_trajectory)