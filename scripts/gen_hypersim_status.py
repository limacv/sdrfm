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
# for scene_name, camera_name, frame_id, split in tqdm(zip(scene_names, camera_names, frame_ids, split_partitions), total=len(scene_names)):
#     prefix = os.path.join(hypersim_path, scene_name, "images", f"scene_{camera_name}_geometry_hdf5", f"frame.{int(frame_id):04d}")
#     if not os.path.isfile(prefix + ".normal_cam.hdf5"):
#         print(f"ERROR::{scene_name} {camera_name} {frame_id} .normal_cam.hdf5 not exists")
#     if not os.path.isfile(prefix + ".normal_bump_cam.hdf5"):
#         print(f"ERROR::{scene_name} {camera_name} {frame_id} .normal_bump_cam.hdf5 not exists")

#     prefix1 = os.path.join(hypersim_path, scene_name, "images", f"scene_{camera_name}_final_hdf5", f"frame.{int(frame_id):04d}")
#     if not os.path.isfile(prefix1 + ".color.hdf5"):
#         print(f"ERROR::{scene_name} {camera_name} {frame_id} .color.hdf5 not exists")

# print("Finishing checking completeness")

csv_writer.writerow(["scene_name","camera_name","frame_id",'split', 'nan_ratio', "err2_ratio", 'stdx', 'stdy', 'stdz', 'z_min', 'norm_min', 'norm_max'])
for scene_name, camera_name, frame_id, split in tqdm(zip(scene_names, camera_names, frame_ids, split_partitions), total=len(scene_names)):
    normal_file = os.path.join(hypersim_path, scene_name, "images", f"scene_{camera_name}_geometry_hdf5", f"frame.{int(frame_id):04d}.normal_cam.hdf5")
    normal_file = h5py.File(normal_file, "r")
    normal = np.array(normal_file["dataset"]).astype(np.float32)
    norm = np.linalg.norm(normal, axis=-1)
    nan_ma = np.isnan(norm)
    invalid_num = np.count_nonzero(nan_ma)
    invalid_ratio = invalid_num / norm.size

    if invalid_num == norm.size:
        min_norm = max_norm = 0
        z_min = 0
        stdx = stdy = stdz = 0
        err2_ratio = 0
    else:
        max_norm = np.max(norm[~nan_ma])
        min_norm = np.min(norm[~nan_ma])

        norm_ma = norm[~nan_ma]
        err2_ma = np.logical_or(norm_ma > 2, norm_ma < 1 / 2)
        err2_num = np.count_nonzero(err2_ma)
        err2_ratio = err2_num / norm.size
        
        stdx, stdy, stdz = np.std(normal[..., 0][~nan_ma]), np.std(normal[..., 1][~nan_ma]), np.std(normal[..., 2][~nan_ma])
        z_min = normal[..., 2][~nan_ma].min()

    csv_writer.writerow([scene_name, camera_name, frame_id, split, invalid_ratio, err2_ratio, stdx, stdy, stdz, z_min, min_norm, max_norm])

csv_file.close()
