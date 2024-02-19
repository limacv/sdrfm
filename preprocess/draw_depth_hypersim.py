import glob
import os
import csv
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pdb

csv_filename = os.path.join('data/Hypersim', "metadata_images_split_scene_v1.csv")
assert(os.path.exists(csv_filename))

# read the csv file first
with open(csv_filename, encoding="UTF-8") as file:
    reader = csv.DictReader(file)
    metadata = {}
    for row in reader:
        for column, value in row.items():
            metadata.setdefault(column, []).append(value)

split_partition = np.array(metadata["split_partition_name"])
split_index = split_partition == "train"
scene_names = np.unique(np.array(metadata["scene_name"])[split_index])
camera_names = np.array(metadata["camera_name"])[split_index]
frame_ids = np.array(metadata["frame_id"])[split_index]

depth_files = [glob.glob(os.path.join(
    'data/Hypersim', scene_name, 'images', 'scene_cam_*_geometry_preview', '*.depth_meters.png')) for scene_name in scene_names][::10]
depth_files = sorted([i for item in depth_files for i in item])
pdb.set_trace()

large_image_size = (2000, 1500)  # 10x10 
small_image_size = (200, 150)

output_folder = os.path.join('outputs','depth_vis')
os.makedirs(output_folder,exist_ok=True)

row_count = 0
col_count = 0

large_image_number = 1

for i, image_file in enumerate(depth_files):
    small_image = Image.open(image_file).resize(small_image_size)
    
    large_image = Image.new('RGB', large_image_size) if col_count == 0 and row_count == 0 else large_image
    # pdb.set_trace()
    large_image.paste(small_image, (col_count * small_image_size[0], row_count * small_image_size[1]))
    # pdb.set_trace()
    col_count += 1
    if col_count == 10:
        col_count = 0
        row_count += 1
    
    if row_count == 10:
        output_file = os.path.join(output_folder, f"large_image_{large_image_number}.png")
        large_image.save(output_file)
        large_image_number += 1
        row_count = 0

    small_image.close()
    

output_file = os.path.join(output_folder, f"large_image_{large_image_number}.png")
large_image.save(output_file)

large_image.close()

print(f"Combined images saved to {output_folder}")
