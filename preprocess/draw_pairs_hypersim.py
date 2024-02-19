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
    'data/Hypersim', scene_name, 'images', 'scene_cam_*_geometry_preview', '*.depth_meters.png')) for scene_name in scene_names]

rgb_files = [glob.glob(os.path.join(
            'data/Hypersim', scene_name, 'images', 'scene_cam_*_final_preview', '*.tonemap.jpg')) for scene_name in scene_names]
depth_files = sorted([i for item in depth_files for i in item])
rgb_files = sorted([i for item in rgb_files for i in item])
# pdb.set_trace()

large_image_size = (1600, 1200)  # 6*6
small_image_size = (400, 300)

output_folder = os.path.join('outputs','pairs_vis')
os.makedirs(output_folder,exist_ok=True)

row_count = 0
col_count = 0

large_image_number = 1

for i in range(len(depth_files)):
    depth_small_image = Image.open(depth_files[i]).resize(small_image_size)
    rgb_small_image = Image.open(rgb_files[i]).resize(small_image_size)
    
    depth_large_image = Image.new('RGB', large_image_size) if col_count == 0 and row_count == 0 else depth_large_image
    rgb_large_image = Image.new('RGB', large_image_size) if col_count == 0 and row_count == 0 else rgb_large_image

    depth_large_image.paste(depth_small_image, (col_count * small_image_size[0], row_count * small_image_size[1]))
    # pdb.set_trace()
    # print(i)
    rgb_large_image.paste(rgb_small_image, (col_count * small_image_size[0], row_count * small_image_size[1]))
    # pdb.set_trace()
    col_count += 1
    if col_count == 4:
        col_count = 0
        row_count += 1
    
    if row_count == 4:
        depth_output_file = os.path.join(output_folder, f"depth_image_{large_image_number}.png")
        rgb_output_file = os.path.join(output_folder, f"rgb_image_{large_image_number}.png")
        depth_large_image.save(depth_output_file)
        rgb_large_image.save(rgb_output_file)
        large_image_number += 1
        row_count = 0

    # depth_large_image.close()
    # rgb_large_image.close()
    if i > 500:
        break
    

depth_output_file = os.path.join(output_folder, f"depth_image_{large_image_number}.png")
rgb_output_file = os.path.join(output_folder, f"rgb_image_{large_image_number}.png")
depth_large_image.save(depth_output_file)
rgb_large_image.save(rgb_output_file)

depth_large_image.close()
rgb_large_image.close()

print(f"Combined images saved to {output_folder}")
