import glob
import os

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


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


class ToTensor(object):
    def __init__(self):
        # self.normalize = transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.normalize = lambda x: x
        self.resize = transforms.Resize((480, 640))

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        image = self.to_tensor(image)
        image = self.normalize(image)
        depth = self.to_tensor(depth)

        image = self.resize(image)

        return {'image': image, 'depth': depth, 'dataset': "hypersim"}

    def to_tensor(self, pic):

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        #         # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img


class HyperSim(Dataset):
    def __init__(self, data_dir_root):
        # image paths are of the form <data_dir_root>/<scene>/images/scene_cam_#_final_preview/*.tonemap.jpg
        # depth paths are of the form <data_dir_root>/<scene>/images/scene_cam_#_geometry_hdf5/*.depth_meters.hdf5
        # self.image_files = glob.glob(os.path.join(
        #     data_dir_root, '*', 'images', 'scene_cam_*_final_preview', '*.tonemap.jpg'))
        # self.depth_files = [r.replace("_final_preview", "_geometry_hdf5").replace(
        #     ".tonemap.jpg", ".depth_meters.hdf5") for r in self.image_files]
        
        self.image_files = glob.glob(os.path.join(
            data_dir_root, 'ai_001_00*', 'images', 'scene_cam_*_final_preview', '*.tonemap.jpg'))
        self.depth_files = [r.replace("_final_preview", "_geometry_hdf5").replace(
            ".tonemap.jpg", ".depth_meters.hdf5") for r in self.image_files]
        self.transform = ToTensor()

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        depth_path = self.depth_files[idx]

        image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0

        # depth from hdf5
        depth_fd = h5py.File(depth_path, "r")
        # in meters (Euclidean distance)
        distance_meters = np.array(depth_fd['dataset'])
        depth = hypersim_distance_to_depth(
            distance_meters)  # in meters (planar depth)

        # depth[depth > 8] = -1
        depth = depth[..., None]

        sample = dict(image=image, depth=depth)
        sample = self.transform(sample)

        if idx == 0:
            print(sample["image"].shape)

        return sample

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
        data_dir_root="data/Hypersim")
    print("Total files", len(loader.dataset))
    for i, sample in enumerate(loader):
        print(sample["image"].shape)
        print(sample["depth"].shape)
        # print(sample["dataset"])
        print(sample['depth'].min(), sample['depth'].max())
        # import pdb;pdb.set_trace()
        if i > 5:
            break