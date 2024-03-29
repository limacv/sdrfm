import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize

class DIODE(Dataset):
    def __init__(self, data_dir_root, preprocess=None, split='test'):
        import glob

        # image paths are of the form <data_dir_root>/scene_#/scan_#/*.png
        self.image_files = glob.glob(
            os.path.join(data_dir_root, '*', '*', '*.png'))
        self.depth_files = [r.replace(".png", "_depth.npy")
                            for r in self.image_files]
        self.depth_mask_files = [
            r.replace(".png", "_depth_mask.npy") for r in self.image_files]
        self.preprocess = preprocess if preprocess is not None else lambda x: x

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        depth_path = self.depth_files[idx]
        depth_mask_path = self.depth_mask_files[idx]

        image = Image.open(image_path)
        image = ToTensor()(image)
        
        depth = np.load(depth_path) # in meters
        depth = np.transpose(depth, (2, 0, 1))  
        depth = torch.tensor(depth)
        
        valid = np.load(depth_mask_path)[None]# binary
        valid = torch.tensor(valid)

        # depth[depth > 8] = -1
        # depth = depth[..., None]

        sample = dict(image=image, depth=depth, valid=valid)

        return self.preprocess(sample)

    def __len__(self):
        return len(self.image_files)


def get_diode_loader(data_dir_root, batch_size=1, **kwargs):
    dataset = DIODE(data_dir_root)
    return DataLoader(dataset, batch_size, **kwargs)

# get_diode_loader(data_dir_root="datasets/diode/val/outdoor")
