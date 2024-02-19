import glob
import os
import csv
import numpy as np
import torch
import h5py
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pdb
from tqdm import tqdm

hypersim_path = "/cpfs01/shared/pjlab-lingjun-landmarks/pjlab-lingjun-landmarks_hdd/jianglihan/Hypersim/portable_hard_drive/downloads/"
csv_filename = os.path.join(hypersim_path, "metadata_images_split_scene_v1.csv")
assert(os.path.exists(csv_filename))

# read the csv file first
with open(csv_filename, encoding="UTF-8") as file:
    reader = csv.DictReader(file)
    metadata = {}
    for row in reader:
        for column, value in row.items():
            metadata.setdefault(column, []).append(value)

# not only train
included_in_public_release = np.array(metadata["included_in_public_release"])
public_index = included_in_public_release=='True'
split_partitions = np.array(metadata["split_partition_name"])[public_index]
scene_names = np.array(metadata["scene_name"])[public_index]
camera_names = np.array(metadata["camera_name"])[public_index]
frame_ids = np.array(metadata["frame_id"])[public_index]

csv_file = open("normal_stats.csv", 'w', newline='')
csv_writer = csv.writer(csv_file)

# check if all file exists
for scene_name, camera_name, frame_id, split in tqdm(zip(scene_names, camera_names, frame_ids, split_partitions), total=len(scene_names)):
    prefix = os.path.join(hypersim_path, scene_name, "images", f"scene_{camera_name}_geometry_hdf5", f"frame.{int(frame_id):04d}")
    if not os.path.isfile(prefix + ".normal_cam.hdf5"):
        print(f"ERROR::{scene_name} {camera_name} {frame_id} .normal_cam.hdf5 not exists")
    if not os.path.isfile(prefix + ".normal_bump_cam.hdf5"):
        print(f"ERROR::{scene_name} {camera_name} {frame_id} .normal_bump_cam.hdf5 not exists")

    prefix1 = os.path.join(hypersim_path, scene_name, "images", f"scene_{camera_name}_final_hdf5", f"frame.{int(frame_id):04d}")
    if not os.path.isfile(prefix1 + ".color.hdf5"):
        print(f"ERROR::{scene_name} {camera_name} {frame_id} .color.hdf5 not exists")

print("Finishing checking completeness")

csv_writer.writerow(["scene_name","camera_name","frame_id",'split', 'nan_ratio', "bump_nan_ratio", 'norm_min', 'norm_max', "bump_norm_min", "bump_norm_max"])
for scene_name, camera_name, frame_id, split in tqdm(zip(scene_names, camera_names, frame_ids, split_partitions), total=len(scene_names)):
    normal_file = os.path.join(hypersim_path, scene_name, "images", f"scene_{camera_name}_geometry_hdf5", f"frame.{int(frame_id):04d}.normal_cam.hdf5")
    normal_file = h5py.File(normal_file, "r")
    normal = np.array(normal_file["dataset"])
    norm = np.linalg.norm(normal, axis=-1)
    nan_ma = np.isnan(norm)
    invalid_num = np.count_nonzero(nan_ma)
    invalid_ratio = invalid_num / norm.size

    if invalid_num == norm.size:
        min_norm = max_norm = 0

    max_norm = np.max(norm[~nan_ma])
    min_norm = np.min(norm[~nan_ma])
    
    normal_file1 = os.path.join(hypersim_path, scene_name, "images", f"scene_{camera_name}_geometry_hdf5", f"frame.{int(frame_id):04d}.normal_bump_cam.hdf5")
    normal_file1 = h5py.File(normal_file1, "r")
    normal1 = np.array(normal_file1["dataset"])
    norm1 = np.linalg.norm(normal1, axis=-1)
    nan_ma1 = np.isnan(norm1)
    invalid_num1 = np.count_nonzero(nan_ma1)
    invalid_ratio1 = invalid_num1 / norm1.size

    if invalid_num1 == norm1.size:
        min_norm1 = max_norm1 = 0

    max_norm1 = np.max(norm1[~nan_ma1])
    min_norm1 = np.min(norm1[~nan_ma1])

    csv_writer.writerow([scene_name, camera_name, frame_id, split, invalid_ratio, invalid_ratio1, min_norm, max_norm, min_norm1, max_norm1])

csv_file.close()
