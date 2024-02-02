import numpy as np
import torch
from PIL import Image

def chw2hwc(chw):
    assert 3 == len(chw.shape)
    if isinstance(chw, torch.Tensor):
        hwc = torch.permute(chw, (1, 2, 0))
    elif isinstance(chw, np.ndarray):
        hwc = np.moveaxis(chw, 0, -1)
    return hwc

def depth2color(depth, colorize_func):
    depth_color = colorize_func(depth, depth.min(), depth.max()).squeeze()
    depth_colored = (depth_color * 255).astype(np.uint8)
    depth_colored_hwc = chw2hwc(depth_colored)
    depth_colored_img = Image.fromarray(depth_colored_hwc)
    return depth_colored_img