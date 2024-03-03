import torch
from torchvision.transforms import Resize, CenterCrop, InterpolationMode
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# depth range is not [-1, 1]?
def _normalize_depth_marigold(depth: torch.Tensor):
    n_element = depth.nelement()
    firstk, lastk = int(n_element * 0.02), int(n_element * 0.98)
    d2 = torch.kthvalue(depth.reshape(-1), firstk)[0]
    d98 = torch.kthvalue(depth.reshape(-1), lastk)[0]
    depth_normalized = (depth - d2) / (d98 - d2)
    depth_normalized = (depth_normalized - 0.5) * 2
    return depth_normalized.clamp(-1,1)


def _normalize_depth_vae_range(depth: torch.Tensor):
    min_value = torch.min(depth)
    max_value = torch.max(depth)
    depth_normalized = ((depth - min_value)/(max_value - min_value+1e-8) - 0.5) * 2  
    return depth_normalized


def _normalize_depth_inv(depth: torch.Tensor):
    disp = 1 / depth
    disp = torch.nan_to_num(disp, 100, 100, 100)
    n_element = disp.nelement()
    lastk = int(n_element * 0.98)
    d98 = torch.kthvalue(disp.reshape(-1), lastk)[0]
    d0 = disp.min()
    disp = (disp - d0) / (d98 - d0)
    disp = - (disp - 0.5) * 2
    return disp.clamp(-1, 1)


def _normalize_depth_inv_fix(depth: torch.Tensor):
    disp = 1 / depth
    disp = torch.nan_to_num(disp, 100, 100, 100)
    d98 = 0.5  # a fixed number, so it's scale variant disparity
    disp = disp / d98
    disp = - (disp - 0.5) * 2
    return disp.clamp(-1, 1)


def _normalize_depth_absolute(depth: torch.Tensor):
    depth = torch.where(depth > 1, 2 - 1 / depth, depth)
    depth = depth - 1
    return depth.clamp(-1, 1)  # unnecessary


def set_depth_normalize_fn(mode):
    global normalize_depth_fn
    print(f"Dataset.utils::set depth normalization mode to {mode}")
    if mode == "marigold":
        normalize_depth_fn = _normalize_depth_marigold
    elif mode == "disparity":
        normalize_depth_fn = _normalize_depth_inv_fix
    elif mode == "disparity_normalized":
        normalize_depth_fn = _normalize_depth_inv
    elif mode == 'vae_range':
        normalize_depth_fn = _normalize_depth_vae_range
    elif mode == "absolute":
        normalize_depth_fn = _normalize_depth_absolute
    else:
        raise RuntimeError(f"Unrecognized mode {mode}")


def resize_max_res(input_tensor, recom_resolution=768):
    """
    Resize image to limit maximum edge length while keeping aspect ratio.
    """
    assert input_tensor.shape[1]==3
    original_H, original_W = input_tensor.shape[2:]

    downscale_factor = min(recom_resolution/original_H,
                           recom_resolution/original_W)
    
    resized_input_tensor = F.interpolate(input_tensor,
                                         scale_factor=downscale_factor,mode='bilinear',
                                         align_corners=False)
    
    return resized_input_tensor


def vkitti_train_preprocess(sample):
    image, depth = sample["image"], sample["depth"]  # tensor of shape 3 x H x W
    # random flip
    if np.random.randint(2):
        image = torch.flip(image, dims=[-1])
        depth = torch.flip(depth, dims=[-1])
    if normalize_depth_fn is _normalize_depth_marigold:
        depth = depth.clamp_max(80)
    depth = normalize_depth_fn(depth)
    sample["image"] = image.contiguous()
    sample["depth"] = depth.contiguous()
    return sample


def vkitti_test_preprocess(sample):
    if normalize_depth_fn is _normalize_depth_marigold:
        sample["depth"] = sample["depth"].clamp_max(80)
    return sample


def hypersim_train_preprocess(sample):
    image, depth = sample["image"], sample["depth"]  # tensor of shape 3 x H x W
    resize = Resize(480, antialias=True)
    image = resize(image)
    depth = resize(depth)
    # random flip
    if np.random.randint(2):
        image = torch.flip(image, dims=[-1])
        depth = torch.flip(depth, dims=[-1])
    depth = normalize_depth_fn(depth)
    sample["image"] = image.contiguous()
    sample["depth"] = depth.contiguous()
    return sample


def hypersim_test_preprocess(sample):
    resize = Resize(480, antialias=True)
    sample["image"] = resize(sample["image"])
    sample["depth"] = resize(sample["depth"])
    return sample


# Global variables that control the behavior of the data loader
normalize_depth_fn = None
set_depth_normalize_fn("marigold")
preprocess_functions = {
    "vkitti": {"train": vkitti_train_preprocess, "test": vkitti_test_preprocess},
    "hypersim": {"train": hypersim_train_preprocess, "test": hypersim_test_preprocess}
}
