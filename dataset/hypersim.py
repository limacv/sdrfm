import glob
import os

import h5py
import csv
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Resize


_NAN_THRESHOLD = 0.04


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


class HyperSim(Dataset):
    def __init__(self, data_dir_root, preprocess=None, split='test'):
        # image paths are of the form <data_dir_root>/<scene>/images/scene_cam_#_final_preview/*.tonemap.jpg
        # depth paths are of the form <data_dir_root>/<scene>/images/scene_cam_#_geometry_hdf5/*.depth_meters.hdf5
        csv_filename = os.path.join(data_dir_root, "depth_stats.csv")
        assert(os.path.exists(csv_filename))
        
        # read the csv file first
        with open(csv_filename, encoding="UTF-8") as file:
            reader = csv.DictReader(file)
            metadata = {}
            for row in reader:
                for column, value in row.items():
                    metadata.setdefault(column, []).append(value)
        
        split_partition = np.array(metadata["split"])
        split_index = split_partition == split
        scene_names = np.unique(np.array(metadata["scene_name"])[split_index])
        nan_ratio= [float(x) for x in metadata["nan_ratio"]]
        nan_ratio = np.array(nan_ratio)[split_index]
        filter_mask = nan_ratio < _NAN_THRESHOLD # predefined thre

        # self.image_files = sorted(glob.glob(os.path.join(
        #     data_dir_root, 'ai_001_001', 'images', 'scene_cam_*_final_preview', '*.tonemap.jpg')))
        # self.depth_files = [r.replace("_final_preview", "_geometry_hdf5").replace(
        #     ".tonemap.jpg", ".depth_meters.hdf5") for r in self.image_files]
        
        self.image_files = [glob.glob(os.path.join(
            data_dir_root, scene_name, 'images', 'scene_cam_*_final_preview', '*.tonemap.jpg')) for scene_name in scene_names]
        self.image_files = np.array(sorted([i for item in self.image_files for i in item]))
        self.depth_files = np.array([r.replace("_final_preview", "_geometry_hdf5").replace(
        ".tonemap.jpg", ".depth_meters.hdf5") for r in self.image_files])
        
        self.image_files = self.image_files[filter_mask]
        self.depth_files = self.depth_files[filter_mask]
        
        self.preprocess = preprocess if preprocess is not None else lambda x: x

        
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        depth_path = self.depth_files[idx]
        

        image = Image.open(image_path)
        image = ToTensor()(image)
        image = image * 2 - 1

        # depth from hdf5
        depth_fd = h5py.File(depth_path, "r")
        # in meters (Euclidean distance)
        distance_meters = np.array(depth_fd['dataset']).astype(np.float32)

        depth = hypersim_distance_to_depth(distance_meters)  # in meters (planar depth)

        depth_max = depth[np.logical_not(np.isnan(depth))].max()
        depth = np.nan_to_num(depth, nan=depth_max, neginf=depth_max, posinf=depth_max)
        
        depth = torch.tensor(depth)[None]
        sample = dict(image=image, depth=depth, dataset="hypersim")
        return self.preprocess(sample)

    def __len__(self):
        return len(self.image_files)


def get_hypersim_loader(data_dir_root, batch_size=1, **kwargs):
    dataset = HyperSim(data_dir_root)
    return DataLoader(dataset, batch_size, **kwargs)

if __name__ == "__main__":
    # depth nan question: https://github.com/apple/ml-hypersim/issues/13
    # not all pixels contain actual scene geometry, e.g., background pixels
    # In most learning applications, masking them out and ignoring them \
        # when computing a per-pixel loss seems like a sensible approach.
    loader = get_hypersim_loader(
        data_dir_root="/cpfs01/shared/pjlab-lingjun-landmarks/pjlab-lingjun-landmarks_hdd/jianglihan/Hypersim/portable_hard_drive/downloads")
    print("Total files", len(loader.dataset))
    for i, sample in enumerate(loader):
        print(sample["image"].shape)
        print(sample["depth"].shape)
        # print(sample["dataset"])
        print(sample['depth'].min(), sample['depth'].max())
        if i > 5:
            break