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

csv_filename = os.path.join('data/Hypersim', "metadata_images_split_scene_v1.csv")
assert(os.path.exists(csv_filename))

def hypersim_distance_to_depth(npyDistance):
    intWidth, intHeight, fltFocal = 1024, 768, 886.81

    npyImageplaneX = np.linspace((-0.5 * intWidth) + 0.5, (0.5 * intWidth) - 0.5, intWidth).reshape(
        1, intWidth).repeat(intHeight, 0).astype(np.float32)[:, :, None]
    npyImageplaneY = np.linspace((-0.5 * intHeight) + 0.5, (0.5 * intHeight) - 0.5,
                                 intHeight).reshape(intHeight, 1).repeat(intWidth, 1).astype(np.float32)[:, :, None]
    npyImageplaneZ = np.full([intHeight, intWidth, 1], fltFocal, np.float32)
    npyImageplane = np.concatenate(
        [npyImageplaneX, npyImageplaneY, npyImageplaneZ], 2)

    npyDepth = npyDistance / np.linalg.norm(npyImageplane, 2, 2) * fltFocal
    return npyDepth

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
scene_names = np.unique(np.array(metadata["scene_name"])[public_index])
camera_names = np.array(metadata["camera_name"])[public_index]
frame_ids = np.array(metadata["frame_id"])[public_index]

depth_files = [glob.glob(os.path.join(
    'data/Hypersim', scene_name, 'images', 'scene_cam_*_geometry_hdf5', '*.depth_meters.hdf5')) for scene_name in scene_names]
depth_files = sorted([i for item in depth_files for i in item])
# pdb.set_trace()

save_csv_filename = os.path.join('data/Hypersim', "depth_stats.csv")
csv_file = open('depth_stats.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["scene_name","camera_name","frame_id", 'file_name', 'full_name','split', 'nan_ratio', 'max', 'min', 'mean', 'std'])

for i, depth_path in tqdm(enumerate(depth_files),desc='process depth'):
    
    scene_name = depth_path.split('/')[2]
    camera_name = camera_names[i]
    frame_id = frame_ids[i]
    file_name = depth_path.split('/')[-1]
    split_partition = split_partitions[i]
    
    depth_fd = h5py.File(depth_path, "r")
    # in meters (Euclidean distance)
    distance_meters = np.array(depth_fd['dataset'])
    depth = hypersim_distance_to_depth(
        distance_meters)  # in meters (planar depth)
    # depth[depth > 8] = -1
    depth = depth[..., None]
    w = depth.shape[0]
    h = depth.shape[1]
    
    invalid_depth = np.isnan(depth)
    invalid_num = np.sum(invalid_depth)
    invalid_ratio = invalid_num / (w*h)
    
    if invalid_num == w*h:
        csv_writer.writerow([scene_name, camera_name, frame_id, file_name, depth_path, split_partition, invalid_ratio,0,0,0,0])
        continue
    
    # before summary the stastic, have to remove the invalid_num
    valid_depth = depth[~invalid_depth]
    
    
    max_val = np.max(valid_depth)
    min_val = np.min(valid_depth)
    mean_val = np.mean(valid_depth)
    std_val = np.std(valid_depth)
    
    
    
    csv_writer.writerow([scene_name, camera_name, frame_id, file_name, depth_path, split_partition, invalid_ratio, max_val, min_val, mean_val, std_val])
    # if i > 2:
    #     pdb.set_trace()
csv_file.close()



