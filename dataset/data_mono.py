import itertools
import os
import random

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.utils.data.distributed

from utils.easydict import EasyDict as edict
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# from .ddad import get_ddad_loader
# from .diml_indoor_test import get_diml_indoor_loader
# from .diml_outdoor_test import get_diml_outdoor_loader
# from .diode import get_diode_loader
from .hypersim import get_hypersim_loader
from .nyu_v2 import get_nyu_loader
from .ibims import get_ibims_loader
# from .sun_rgbd_loader import get_sunrgbd_loader
# from .vkitti import get_vkitti_loader
from .vkitti2 import get_vkitti2_loader

# from .preprocess import CropParams, get_white_border, get_black_border

class DepthDataLoader(object):
    def __init__(self, config, mode, device='cpu', transform=None, **kwargs):
        """
        Data loader for depth datasets

        Args:
            config (dict): Config dictionary. Refer to utils/config.py
            mode (str): "train" or "online_eval"
            device (str, optional): Device to load the data on. Defaults to 'cpu'.
            transform (torchvision.transforms, optional): Transform to apply to the data. Defaults to None.
        """

        self.config = config
        if config.dataset == 'nyu':
            self.data = get_nyu_loader(config, batch_size=1, num_workers=1)
            return

        if config.dataset == 'ibims':
            self.data = get_ibims_loader(config, batch_size=1, num_workers=1)
            return

        if config.dataset == 'sunrgbd':
            self.data = get_sunrgbd_loader(
                data_dir_root=config.sunrgbd_root, batch_size=1, num_workers=1)
            return

        if config.dataset == 'diml_indoor':
            self.data = get_diml_indoor_loader(
                data_dir_root=config.diml_indoor_root, batch_size=1, num_workers=1)
            return

        if config.dataset == 'diml_outdoor':
            self.data = get_diml_outdoor_loader(
                data_dir_root=config.diml_outdoor_root, batch_size=1, num_workers=1)
            return

        if "diode" in config.dataset:
            self.data = get_diode_loader(
                config[config.dataset+"_root"], batch_size=1, num_workers=1)
            return

        if config.dataset == 'hypersim_test':
            self.data = get_hypersim_loader(
                config.hypersim_test_root, batch_size=1, num_workers=1)
            return

        if config.dataset == 'vkitti':
            self.data = get_vkitti_loader(
                config.vkitti_root, batch_size=1, num_workers=1)
            return

        if config.dataset == 'vkitti2':
            self.data = get_vkitti2_loader(
                config.vkitti2_root, batch_size=1, num_workers=1)
            return

        if config.dataset == 'ddad':
            self.data = get_ddad_loader(config.ddad_root, resize_shape=(
                352, 1216), batch_size=1, num_workers=1)
            return

        img_size = self.config.get("img_size", None)
        img_size = img_size if self.config.get(
            "do_input_resize", False) else None

        if transform is None:
            transform = preprocessing_transforms(mode, size=img_size)

        if mode == 'train':

            Dataset = DataLoadPreprocess
            self.training_samples = Dataset(
                config, mode, transform=transform, device=device)

            if config.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                    self.training_samples)
            else:
                self.train_sampler = None

            self.data = DataLoader(self.training_samples,
                                   batch_size=config.batch_size,
                                   shuffle=(self.train_sampler is None),
                                   num_workers=config.workers,
                                   pin_memory=True,
                                   persistent_workers=True,
                                #    prefetch_factor=2,
                                   sampler=self.train_sampler)

        elif mode == 'eval':
            
            self.testing_samples = DataLoadPreprocess(
                config, mode, transform=None)
 
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=kwargs.get("shuffle_test", False),
                                   num_workers=1,
                                   pin_memory=False,
                                   sampler=None)

        else:
            print(
                'mode should be one of \'train, eval\'. Got {}'.format(mode))