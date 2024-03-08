from enum import Enum
import os

import h5py
import csv
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Resize

HyperSim_Asset_List = [
    "depth", "normal", "d2normal",
    "albedo", "shading", "specular"
]


_NAN_THRESHOLD = 0.02
_ERR2_THRESHOLD = 0.1
_STD_THRESHOLD = 0.1
_NORMAL_PREFIX = "normal_cam"  # "normal_cam", "normal_bump_cam"


class HyperSimMono(Dataset):
    def __init__(self, data_dir_root, assets, preprocess=None, split='test'):
        # image paths are of the form <data_dir_root>/<scene>/images/scene_cam_#_final_preview/*.tonemap.jpg
        # depth paths are of the form <data_dir_root>/<scene>/images/scene_cam_#_geometry_hdf5/*.depth_meters.hdf5
        csv_filename = os.path.join(data_dir_root, "normal_stats_v2.csv")
        self.assets = assets
        assert all(a in HyperSim_Asset_List for a in self.assets), f"HyperSimMono::Unrecognized asset name {assets}"
        assert (os.path.exists(csv_filename))
        
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
        self.depth_files = []
        self.normal_files = []
        self.albedo_files = []
        self.shading_files = []
        self.specular_files = []
        for scene, cam, frm in zip(scene_names, camera_names, frame_ids):
            self.image_files.append(os.path.join(
                data_dir_root, scene, "images", f"scene_{cam}_final_preview", f"frame.{int(frm):04d}.tonemap.jpg"
            ))
            self.depth_files.append(os.path.join(
                data_dir_root, scene, "images", f"scene_{cam}_geometry_hdf5", f"frame.{int(frm):04d}.depth_meters.hdf5"
            ))
            self.normal_files.append(os.path.join(
                data_dir_root, scene, "images", f"scene_{cam}_geometry_hdf5", f"frame.{int(frm):04d}.{_NORMAL_PREFIX}.hdf5"
            ))
            self.albedo_files.append(os.path.join(
                data_dir_root, scene, "images", f"scene_{cam}_final_preview", f"frame.{int(frm):04d}.diffuse_reflectance.jpg"
            ))
            self.shading_files.append(os.path.join(
                data_dir_root, scene, "images", f"scene_{cam}_final_preview", f"frame.{int(frm):04d}.diffuse_illumination.jpg"
            ))
            self.specular_files.append(os.path.join(
                data_dir_root, scene, "images", f"scene_{cam}_final_preview", f"frame.{int(frm):04d}.residual.jpg"
            ))

        for p in self.image_files:
            assert os.path.isfile(p), f"HyperSim::{p} does not exist"
        # if "normal" in self.assets:
        #     for p in self.normal_files:
        #         assert os.path.isfile(p), f"HyperSim::{p} does not exist"
        # if "depth" or "d2normal" in self.assets:
        #     for p in self.depth_files:
        #         assert os.path.isfile(p), f"HyperSim::{p} does not exist"
        # if "albedo" in self.assets:
        #     for p in self.albedo_files:
        #         assert os.path.isfile(p), f"HyperSim::{p} does not exist"
        # if "shading" in self.assets:
        #     for p in self.shading_files:
        #         assert os.path.isfile(p), f"HyperSim::{p} does not exist"
        # if "specular" in self.assets:
        #     for p in self.specular_files:
        #         assert os.path.isfile(p), f"HyperSim::{p} does not exist"
        
        self.preprocess = preprocess if preprocess is not None else lambda x: x
        
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path)
        image = ToTensor()(image)
        image = image * 2 - 1

        depth = None

        sample = dict(image=image)
        if "depth" in self.assets or "d2normal" in self.assets:
            depth_path = self.depth_files[idx]
            depth_fd = h5py.File(depth_path, "r")
            # in meters (Euclidean distance)
            distance_meters = np.array(depth_fd['dataset']).astype(np.float32)
            depth = _hypersim_distance_to_depth(distance_meters)  # in meters (planar depth)
            depth_max = depth[np.logical_not(np.isnan(depth))].max()
            depth = np.nan_to_num(depth, nan=depth_max, neginf=depth_max, posinf=depth_max)
            sample["depth"] = torch.tensor(depth)[None]

        if "d2normal" in self.assets:
            normal = _hypersim_depth_to_normal(depth)
            normal = torch.tensor(normal).permute(2, 0, 1)
            sample["normal"] = normal

        if "normal" in self.assets:
            normal_path = self.normal_files[idx]
            normal = h5py.File(normal_path, "r")
            # in meters (Euclidean distance)
            normal = np.array(normal['dataset']).astype(np.float32)
            nan_ma = np.isnan(normal.sum(axis=-1))
            normal[nan_ma, :] = np.array([0, 0, 1])
            normal = torch.tensor(normal).permute(2, 0, 1)
            sample["normal"] = normal

        if "albedo" in self.assets:
            albedo_path = self.albedo_files[idx]
            albedo = Image.open(albedo_path)
            albedo = ToTensor()(albedo)
            albedo = albedo * 2 - 1
            sample["albedo"] = albedo
        
        if "shading" in self.assets:
            shading_path = self.shading_files[idx]
            shading = Image.open(shading_path)
            shading = ToTensor()(shading)
            shading = shading * 2 - 1
            sample["shading"] = shading
        
        if "specular" in self.assets:
            specular_path = self.specular_files[idx]
            specular = Image.open(specular_path)
            specular = ToTensor()(specular)
            specular = specular * 2 - 1
            sample["specular"] = specular
        
        return self.preprocess(sample)

    def __len__(self):
        return len(self.image_files)


def _hypersim_distance_to_depth(npyDistance):
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


def _hypersim_depth_to_normal(depth):
    intWidth, intHeight, fltFocal = 1024, 768, 886.81
    npyImageplaneX = np.linspace((-0.5 * intWidth) + 0.5, (0.5 * intWidth) - 0.5, intWidth).reshape(
        1, intWidth).repeat(intHeight, 0).astype(np.float32)[:, :, None]
    npyImageplaneY = np.linspace((-0.5 * intHeight) + 0.5, (0.5 * intHeight) - 0.5,
                                 intHeight).reshape(intHeight, 1).repeat(intWidth, 1).astype(np.float32)[:, :, None]
    npyImageplaneZ = np.full([intHeight, intWidth, 1], fltFocal, np.float32)
    npyImageplane = np.concatenate(
        [npyImageplaneX, npyImageplaneY, npyImageplaneZ], 2)
    xyz = npyImageplane * depth[..., None]
    dy = - xyz[1:, :-1] + xyz[:-1, :-1]
    dy = dy / (np.linalg.norm(dy, axis=-1, keepdims=True) + 1e-9)
    dx = xyz[:-1, 1:] - xyz[:-1, :-1]
    dx = dx / (np.linalg.norm(dx, axis=-1, keepdims=True) + 1e-9)
    dz = np.cross(dx, dy, axis=-1)
    dz = np.concatenate([dz, dz[-1:]], axis=0)
    dz = np.concatenate([dz, dz[:, -1:]], axis=1)
    return dz / (np.linalg.norm(dz, axis=-1, keepdims=True) + 1e-9)


def get_hypersim_loader(data_dir_root, batch_size=1, **kwargs):
    dataset = HyperSimMono(data_dir_root)
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