import glob
import os

import h5py
import csv
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Resize


_NAN_THRESHOLD = 0.02
_ERR2_THRESHOLD = 0.1
_STD_THRESHOLD = 0.1
_NORMAL_PREFIX = "normal_bump_cam"  # "normal_cam", "normal_bump_cam"


class HyperSim(Dataset):
    def __init__(self, data_dir_root, preprocess=None, split='test'):
        # image paths are of the form <data_dir_root>/<scene>/images/scene_cam_#_final_preview/*.tonemap.jpg
        # depth paths are of the form <data_dir_root>/<scene>/images/scene_cam_#_geometry_hdf5/*.depth_meters.hdf5
        csv_filename = os.path.join(data_dir_root, "normal_stats_v2.csv")
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
        nan_ratio = np.array(metadata["nan_ratio"]).astype(np.float32)
        filter_mask = nan_ratio < _NAN_THRESHOLD # predefined thre
        filter_mask = np.logical_and(filter_mask, split_index)

        err2_ratio = np.array(metadata["err2_ratio"]).astype(np.float32)
        err2_mask = err2_ratio < _ERR2_THRESHOLD
        filter_mask = np.logical_and(filter_mask, err2_mask)

        stdx = np.array(metadata["stdx"]).astype(np.float32)
        stdy = np.array(metadata["stdy"]).astype(np.float32)
        stdz = np.array(metadata["stdz"]).astype(np.float32)
        std = np.stack([stdx, stdy, stdz], axis=-1).max(axis=-1)
        std_mask = std > _STD_THRESHOLD
        filter_mask = np.logical_and(filter_mask, std_mask)

        scene_names = np.array(metadata["scene_name"])[filter_mask]
        camera_names = np.array(metadata["camera_name"])[filter_mask]
        frame_ids = np.array(metadata["frame_id"])[filter_mask]

        self.image_files = []
        self.normal_files = []
        for scene, cam, frm in zip(scene_names, camera_names, frame_ids):
            self.image_files.append(os.path.join(
                data_dir_root, scene, "images", f"scene_{cam}_final_preview", f"frame.{int(frm):04d}.tonemap.jpg"
            ))
            self.normal_files.append(os.path.join(
                data_dir_root, scene, "images", f"scene_{cam}_geometry_hdf5", f"frame.{int(frm):04d}.{_NORMAL_PREFIX}.hdf5"
            ))

        for p in self.image_files:
            assert os.path.isfile(p), f"HyperSim::{p} does not exist"
        for p in self.normal_files:
            assert os.path.isfile(p), f"HyperSim::{p} does not exist"

        self.preprocess = preprocess if preprocess is not None else lambda x: x
        
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        normal_path = self.normal_files[idx]
        
        image = Image.open(image_path)
        image = ToTensor()(image)
        image = image * 2 - 1

        normal = h5py.File(normal_path, "r")
        # in meters (Euclidean distance)
        normal = np.array(normal['dataset']).astype(np.float32)
        nan_ma = np.isnan(normal.sum(axis=-1))
        normal[nan_ma, :] = np.array([0, 0, 1])
        
        normal = torch.tensor(normal).permute(2, 0, 1)
        sample = dict(image=image, normal=normal, dataset="hypersim")
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
        print(sample["normal"].shape)
        if i > 5:
            break