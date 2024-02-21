import torch
from torchvision.transforms import Resize, CenterCrop, InterpolationMode
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



def set_normal_normalize_fn(mode):
    global normalize_normal_fn
    print(f"Dataset.utils::set normal normalization mode to {mode}")
    if mode == "none":
        normalize_normal_fn = lambda n: n
    elif mode == "clip":
        normalize_normal_fn = lambda n: n.clamp(-1, 1)
    elif mode == "norm":
        normalize_normal_fn = lambda n: F.normalize(n, eps=1e-9, dim=0)
    elif mode == 'clipnorm':
        normalize_normal_fn = lambda n: F.normalize(n.clamp(-1, 1), eps=1e-9, dim=0)
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


def hypersim_train_preprocess(sample):
    image, normal = sample["image"], sample["normal"]  # tensor of shape 3 x H x W
    resize = Resize(480, antialias=True)
    resize_n = Resize(480, interpolation=InterpolationMode.NEAREST)
    image = resize(image)
    normal = resize_n(normal)
    # random flip
    if np.random.randint(2):
        image = torch.flip(image, dims=[-1])
        normal = torch.flip(normal, dims=[-1])
    normal = normalize_normal_fn(normal)
    sample["image"] = image.contiguous()
    sample["normal"] = normal.contiguous()
    return sample


def hypersim_test_preprocess(sample):
    resize = Resize(480, antialias=True)
    sample["image"] = resize(sample["image"])
    sample["normal"] = resize(sample["normal"])
    return sample


# Global variables that control the behavior of the data loader
normalize_normal_fn = None
set_normal_normalize_fn("clipnorm")
preprocess_functions = {
    "hypersim": {"train": hypersim_train_preprocess, "test": hypersim_test_preprocess}
}
