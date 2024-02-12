import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize

class DIML_Indoor(Dataset):
    def __init__(self, data_dir_root, preprocess=None, split='test'):
        import glob

        # image paths are of the form <data_dir_root>/{outleft, depthmap}/*.png
        self.image_files = glob.glob(os.path.join(
            data_dir_root, "LR", '*', 'color', '*.png'))
        self.depth_files = [r.replace("color", "depth_filled").replace(
            "_c.png", "_depth_filled.png") for r in self.image_files]
        self.preprocess = preprocess if preprocess is not None else lambda x: x

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        depth_path = self.depth_files[idx]

        image = Image.open(image_path)
        image = ToTensor()(image)
        depth = np.asarray(Image.open(depth_path),
                           dtype='uint16') / 1000.0  # mm to meters
        
        depth = torch.tensor(depth)[None]

        # depth[depth > 8] = -1
        # depth = depth[..., None]

        sample = dict(image=image, depth=depth, dataset="diml_outdoor")

        # return sample
        return self.preprocess(sample)

    def __len__(self):
        return len(self.image_files)


def get_diml_indoor_loader(data_dir_root, batch_size=1, **kwargs):
    dataset = DIML_Indoor(data_dir_root)
    return DataLoader(dataset, batch_size, **kwargs)

# get_diode_loader(data_dir_root="datasets/diode/val/outdoor")
