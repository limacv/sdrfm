import glob
import os

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Resize


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
    def __init__(self, data_dir_root, preprocess=None):
        # image paths are of the form <data_dir_root>/<scene>/images/scene_cam_#_final_preview/*.tonemap.jpg
        # depth paths are of the form <data_dir_root>/<scene>/images/scene_cam_#_geometry_hdf5/*.depth_meters.hdf5
        # self.image_files = glob.glob(os.path.join(
        #     data_dir_root, '*', 'images', 'scene_cam_*_final_preview', '*.tonemap.jpg'))
        # self.depth_files = [r.replace("_final_preview", "_geometry_hdf5").replace(
        #     ".tonemap.jpg", ".depth_meters.hdf5") for r in self.image_files]
        self.image_files = sorted(glob.glob(os.path.join(
            data_dir_root, 'ai_*_00*', 'images', 'scene_cam_*_final_preview', '*.tonemap.jpg')))
        self.depth_files = [r.replace("_final_preview", "_geometry_hdf5").replace(
            ".tonemap.jpg", ".depth_meters.hdf5") for r in self.image_files]
        self.preprocess = preprocess if preprocess is not None else lambda x: x

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        depth_path = self.depth_files[idx]

        image = Image.open(image_path)
        image = ToTensor()(image)

        # depth from hdf5
        depth_fd = h5py.File(depth_path, "r")
        # in meters (Euclidean distance)
        distance_meters = np.array(depth_fd['dataset']).astype(np.float32)

        # TODO, currently inpaint invalid
        import cv2
        mask = np.isnan(distance_meters).astype(np.uint8) * 255
        distance_meters = cv2.inpaint(distance_meters[..., None], mask[..., None], 3, cv2.INPAINT_TELEA)
        distance_meters = np.nan_to_num(distance_meters, nan=1., neginf=1., posinf=1.)

        depth = hypersim_distance_to_depth(
            distance_meters)  # in meters (planar depth)

        # depth[depth > 8] = -1
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