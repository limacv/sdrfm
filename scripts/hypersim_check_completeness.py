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

csv_file = open("multi_view_status.csv", 'w', newline='')
csv_writer = csv.writer(csv_file)

# check if all file exists
file_postfix_list = [
    ["_final_preview", ".tonemap.jpg"],
    ["_geometry_hdf5", ".position.hdf5"],
    ["_geometry_hdf5", ".depth_meters.hdf5"],
]
for scene_name, camera_name, frame_id, split in tqdm(zip(scene_names, camera_names, frame_ids, split_partitions), total=len(scene_names)):
    for postfix in file_postfix_list:
        if not os.path.isfile(
            os.path.join(hypersim_path, scene_name, "images", f"scene_{camera_name}{postfix[0]}", f"frame.{int(frame_id):04d}{postfix[1]}")
            ):
            print(f"ERROR::{scene_name} {camera_name} {frame_id} {postfix[1]} not exists")

print("Finishing checking completeness")