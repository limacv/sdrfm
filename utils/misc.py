"""Miscellaneous utility functions."""

from scipy import ndimage

import base64
import math
import re
from io import BytesIO

import matplotlib
import matplotlib.cm
import numpy as np
import requests
import torch
import torch.distributed as dist
import torch.nn
import torch.nn as nn
import torch.utils.data.distributed
from PIL import Image
from torchvision.transforms import ToTensor


class RunningAverage:
    def __init__(self):
        self.avg = 0
        self.count = 0

    def append(self, value):
        self.avg = (value + self.count * self.avg) / (self.count + 1)
        self.count += 1

    def get_value(self):
        return self.avg


def denormalize(x):
    """Reverses the imagenet normalization applied to the input.

    Args:
        x (torch.Tensor - shape(N,3,H,W)): input tensor

    Returns:
        torch.Tensor - shape(N,3,H,W): Denormalized input
    """
    mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
    std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
    return x * std + mean


class RunningAverageDict:
    """A dictionary of running averages."""
    def __init__(self):
        self._dict = None

    def update(self, new_dict):
        if new_dict is None:
            return

        if self._dict is None:
            self._dict = dict()
            for key, value in new_dict.items():
                self._dict[key] = RunningAverage()

        for key, value in new_dict.items():
            self._dict[key].append(value)

    def get_value(self):
        if self._dict is None:
            return None
        return {key: value.get_value() for key, value in self._dict.items()}

def count_parameters(model, include_all=False):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or include_all)


def compute_errors(gt, pred):
    """Compute metrics for 'pred' compared to 'gt'

    Args:
        gt (numpy.ndarray): Ground truth values
        pred (numpy.ndarray): Predicted values

        gt.shape should be equal to pred.shape

    Returns:
        dict: Dictionary containing the following metrics:
            'a1': Delta1 accuracy: Fraction of pixels that are within a scale factor of 1.25
            'a2': Delta2 accuracy: Fraction of pixels that are within a scale factor of 1.25^2
            'a3': Delta3 accuracy: Fraction of pixels that are within a scale factor of 1.25^3
            'abs_rel': Absolute relative error
            'rmse': Root mean squared error
            'log_10': Absolute log10 error
            'sq_rel': Squared relative error
            'rmse_log': Root mean squared error on the log scale
            'silog': Scale invariant log error
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    # import pdb;pdb.set_trace()
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, log_10=log_10, rmse_log=rmse_log,
                silog=silog, sq_rel=sq_rel)

def compute_scale_pred(gt, pred, valid_mask, **kwargs):
    if 'config' in kwargs:
        config = kwargs['config']
        min_depth_eval = config.min_depth_eval
        max_depth_eval = config.max_depth_eval
    
    # compute shift and scale for fair comparsion
    a, b = np.polyfit(pred[valid_mask], gt[valid_mask], deg=1)
    pred = a * pred + b
    
    # pred = pred.squeeze()
    # pred[pred < min_depth_eval] = min_depth_eval
    # pred[pred > max_depth_eval] = max_depth_eval
    # pred[np.isinf(pred)] = max_depth_eval
    # pred[np.isnan(pred)] = min_depth_eval
    return pred
    
def compute_masks(gt, pred, interpolate=True, garg_crop=False, eigen_crop=True, dataset='nyu', min_depth_eval=0.1, max_depth_eval=10, **kwargs):
    """Compute metrics of predicted depth maps. Applies cropping and masking as necessary or specified via arguments. Refer to compute_errors for more details on metrics.
    """
    if 'config' in kwargs:
        config = kwargs['config']
        garg_crop = config.garg_crop
        eigen_crop = config.eigen_crop
        min_depth_eval = config.min_depth_eval
        max_depth_eval = config.max_depth_eval

    # import pdb;pdb.set_trace()
    if gt.shape[-2:] != pred.shape[-2:] and interpolate:
        pred = nn.functional.interpolate(
            pred, gt.shape[-2:], mode='bilinear', align_corners=True)

    gt_depth = gt.squeeze()
    valid_mask = np.logical_and(
        gt_depth > min_depth_eval, gt_depth < max_depth_eval)

    if garg_crop or eigen_crop:
        gt_height, gt_width = gt_depth.shape
        eval_mask = np.zeros(valid_mask.shape)

        if garg_crop:
            eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                      int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

        elif eigen_crop:
            # print("-"*10, " EIGEN CROP ", "-"*10)
            if dataset == 'kitti':
                eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                          int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
            else:
                # assert gt_depth.shape == (480, 640), "Error: Eigen crop is currently only valid for (480, 640) images"
                eval_mask[45:471, 41:601] = 1
        else:
            eval_mask = np.ones(valid_mask.shape)
    valid_mask = np.logical_and(valid_mask, eval_mask)
    return valid_mask


class colors:
    '''Colors class:
    Reset all colors with colors.reset
    Two subclasses fg for foreground and bg for background.
    Use as colors.subclass.colorname.
    i.e. colors.fg.red or colors.bg.green
    Also, the generic bold, disable, underline, reverse, strikethrough,
    and invisible work with the main class
    i.e. colors.bold
    '''
    reset = '\033[0m'
    bold = '\033[01m'
    disable = '\033[02m'
    underline = '\033[04m'
    reverse = '\033[07m'
    strikethrough = '\033[09m'
    invisible = '\033[08m'

    class fg:
        black = '\033[30m'
        red = '\033[31m'
        green = '\033[32m'
        orange = '\033[33m'
        blue = '\033[34m'
        purple = '\033[35m'
        cyan = '\033[36m'
        lightgrey = '\033[37m'
        darkgrey = '\033[90m'
        lightred = '\033[91m'
        lightgreen = '\033[92m'
        yellow = '\033[93m'
        lightblue = '\033[94m'
        pink = '\033[95m'
        lightcyan = '\033[96m'

    class bg:
        black = '\033[40m'
        red = '\033[41m'
        green = '\033[42m'
        orange = '\033[43m'
        blue = '\033[44m'
        purple = '\033[45m'
        cyan = '\033[46m'
        lightgrey = '\033[47m'


def printc(text, color):
    print(f"{color}{text}{colors.reset}")


def save_raw_16bit(depth, fpath="raw.png"):
    if isinstance(depth, torch.Tensor):
        depth = depth.squeeze().cpu().numpy()
    
    assert isinstance(depth, np.ndarray), "Depth must be a torch tensor or numpy array"
    assert depth.ndim == 2, "Depth must be 2D"
    depth = depth * 256  # scale for 16-bit png
    depth = depth.astype(np.uint16)
    depth = Image.fromarray(depth)
    depth.save(fpath)
    print("Saved raw depth to", fpath)