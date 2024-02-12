import os

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .preprocesses import remove_leading_slash

class KITTI(Dataset):
    def __init__(self, config, preprocess=None, split='test'):
        self.config = config
        if split == 'train':
            with open(config.filenames_file_eval, 'r') as f:
                self.filenames = f.readlines()
            self.data_path = self.config.data_path
            self.gt_path = self.config.gt_path
        else:
            with open(config.filenames_file_eval, 'r') as f:
                self.filenames = f.readlines()
            self.data_path = self.config.data_path_eval
            self.gt_path = self.config.gt_path_eval
        
        self.data_path = self.config.data_path_eval
        self.gt_path = self.config.gt_path_eval
        self.preprocess = preprocess if preprocess is not None else lambda x: x
    def __getitem__(self, idx):
        sample_path = self.filenames[idx]
        focal = float(sample_path.split()[2])
        sample = {}
        
        image_path = os.path.join(
            self.data_path, remove_leading_slash(sample_path.split()[0]))
        # only for eval, so we just scale the image to [0,1] range
        # and keep the image array format
        image = np.asarray(Image.open(image_path),
                            dtype=np.float32) / 255.0
        
        depth_path = os.path.join(
            self.gt_path, remove_leading_slash(sample_path.split()[1]))
        has_valid_depth = False
        try:
            depth_gt = Image.open(depth_path)
            has_valid_depth = True
        except IOError:
            depth_gt = False
            print('Missing gt for {}'.format(image_path))

        if has_valid_depth:
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)
            depth_gt = depth_gt / 256.0 # depth preprocess

            mask = np.logical_and(
                depth_gt >= self.config.min_depth, depth_gt <= self.config.max_depth).squeeze()[None, ...]
        else:
            mask = False
            
        if self.config.do_kb_crop:
            height = image.shape[0]
            width = image.shape[1]
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            image = image[top_margin:top_margin + 352,
                            left_margin:left_margin + 1216, :]
            if has_valid_depth:
                depth_gt = depth_gt[top_margin:top_margin +
                                    352, left_margin:left_margin + 1216, :]
                    
        sample = {'image': image, 'depth': depth_gt, 'focal': focal, 'has_valid_depth': has_valid_depth,
                          'image_path': sample_path.split()[0], 'depth_path': sample_path.split()[1],
                          'mask': mask}
        sample['dataset'] = self.config.dataset
        sample = {**sample, 'image_path': sample_path.split()[0], 'depth_path': sample_path.split()[1]}
        return self.preprocess(sample)
    
    def __len__(self):
        return len(self.filenames)

def get_kitti_loader(config, batch_size=1, **kwargs):
    dataset = KITTI(config)
    return DataLoader(dataset, batch_size, **kwargs)