import random
import torch
from torchvision.transforms import Resize
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
    d0 = disp.min()
    d98 = torch.kthvalue(disp.reshape(-1), lastk)[0]
    disp = (disp - d0) / (d98 - d0)
    return (disp - 0.5) * 2

def augment_image(image):
    # gamma augmentation
    gamma = random.uniform(0.9, 1.1)
    image_aug = image ** gamma

    # brightness augmentation
    brightness = random.uniform(0.9, 1.1)
    image_aug = image_aug * brightness

    # color augmentation
    colors = np.random.uniform(0.9, 1.1, size=3)
    white = np.ones((image.shape[0], image.shape[1]))
    color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
    image_aug *= color_image
    image_aug = np.clip(image_aug, 0, 1)

    return image_aug

def random_crop(img, depth, height, width):
    assert img.shape[1] >= height
    assert img.shape[2] >= width
    assert img.shape[1] == depth.shape[1]
    assert img.shape[2] == depth.shape[2]
    x = random.randint(0, img.shape[2] - width)
    y = random.randint(0, img.shape[1] - height)
    img = img[:,y:y + height, x:x + width]
    depth = depth[:,y:y + height, x:x + width]

    return img, depth
    
def set_depth_normalize_fn(mode):  # choice from marigold, my
    global normalize_depth_fn
    print(f"Dataset.utils::set depth normalization mode to {mode}")
    if mode == "marigold":
        normalize_depth_fn = _normalize_depth_marigold
    elif mode == "my":
        normalize_depth_fn = _normalize_depth_inv
    elif mode == 'vae_range':
        normalize_depth_fn = _normalize_depth_vae_range
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
    
    # random gamma, brightness, color augmentation
    # if np.random.randint(2):
    #     image = augment_image(image)
    
    image, depth = random_crop(
        image, depth, 256, 256)
    
    image = nn.functional.interpolate(
        image[None], 768, mode='bilinear', align_corners=True).squeeze(0)
    
    depth = nn.functional.interpolate(
        depth[None], 768, mode='bilinear', align_corners=True).squeeze(0)
        
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
    image = image * 2 - 1
    resize = Resize(480, antialias=True)
    image = resize(image)
    depth = resize(depth)
    # random flip
    if np.random.randint(2):
        image = torch.flip(image, dims=[-1])
        depth = torch.flip(depth, dims=[-1])
    
    image, depth = random_crop(
        image, depth, 256, 256)
    
    image = nn.functional.interpolate(
        image[None], 768, mode='bilinear', align_corners=True).squeeze(0)
    
    depth = nn.functional.interpolate(
        depth[None], 768, mode='bilinear', align_corners=True).squeeze(0)
    
    depth = normalize_depth_fn(depth)
    sample["image"] = image.contiguous()
    sample["depth"] = depth.contiguous()
    return sample


def hypersim_test_preprocess(sample):
    resize = Resize(480, antialias=True)
    sample["image"] = resize(sample["image"])
    sample["depth"] = resize(sample["depth"])
    return sample

def diode_test_preprocess(sample):
    resize = Resize(480, antialias=True)
    sample["image"] = resize(sample["image"])
    sample["depth"] = resize(sample["depth"])
    sample["valid"] = resize(sample["valid"])
    return sample


def remove_leading_slash(s):
    if s[0] == '/' or s[0] == '\\':
        return s[1:]
    return s

# Global variables that control the behavior of the data loader
normalize_depth_fn = None
set_depth_normalize_fn("marigold")
preprocess_functions = {
    "vkitti": {"train": vkitti_train_preprocess, "test": vkitti_test_preprocess},
    "hypersim": {"train": hypersim_train_preprocess, "test": hypersim_test_preprocess},
    "diode":{'test':diode_test_preprocess}
}
