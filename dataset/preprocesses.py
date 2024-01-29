import torch
from torchvision.transforms import Resize
import numpy as np


def _normalize_depth_marigold(depth: torch.Tensor):
    n_element = depth.nelement()
    firstk, lastk = int(n_element * 0.02), int(n_element * 0.98)
    d2 = torch.kthvalue(depth.reshape(-1), firstk)[0]
    d98 = torch.kthvalue(depth.reshape(-1), lastk)[0]
    depth_normalized = (depth - d2) / (d98 - d2)
    return (depth_normalized - 0.5) * 2


def _normalize_depth_inv(depth: torch.Tensor):
    disp = 1 / depth
    disp = torch.nan_to_num(disp, 100, 100, 100)
    n_element = disp.nelement()
    firstk = int(n_element * 0.02)
    d2 = torch.kthvalue(disp.reshape(-1), firstk)
    disp = disp / d2
    return (disp - 0.5) * 2


def set_depth_normalize_fn(mode):  # choice from marigold, my
    global normalize_depth_fn
    print(f"Dataset.utils::set depth normalization mode to {mode}")
    if mode == "marigold":
        normalize_depth_fn = _normalize_depth_marigold
    elif mode == "my":
        normalize_depth_fn = _normalize_depth_inv
    else:
        raise RuntimeError(f"Unrecognized mode {mode}")


def vkitti_train_preprocess(sample):
    image, depth = sample["image"], sample["depth"]  # tensor of shape 3 x H x W
    # random flip
    if np.random.randint(1):
        image = image[..., ::-1]
        depth = depth[..., ::-1]
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
    if np.random.randint(1):
        image = image[..., ::-1]
        depth = depth[..., ::-1]
    depth = normalize_depth_fn(depth)
    sample["image"] = image.contiguous()
    sample["depth"] = depth.contiguous()
    return sample


def hypersim_test_preprocess(sample):
    resize = Resize(480, antialias=True)
    sample["image"] = resize(sample["image"])
    sample["depth"] = resize(sample["depth"])
    return sample


normalize_depth_fn = None
set_depth_normalize_fn("marigold")
preprocess_functions = {
    "vkitti": {"train": vkitti_train_preprocess, "test": vkitti_test_preprocess},
    "hypersim": {"train": hypersim_train_preprocess, "test": hypersim_test_preprocess}
}
