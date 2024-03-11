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


def set_hypersim_resolution(reso):
    global hypersim_resolution
    hypersim_resolution = reso


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


def hypersim_train_preprocess(sample: dict):
    # random augment parameters
    resize = Resize(hypersim_resolution, antialias=True)
    resize_n = Resize(hypersim_resolution, interpolation=InterpolationMode.NEAREST)
    do_flip = np.random.randint(2)

    image = sample["image"]  # tensor of shape 3 x H x W
    image = resize(image)
    if do_flip:
        image = torch.flip(image, dims=[-1])
    sample["image"] = image.contiguous()
    
    depth = sample.get("depth", None)
    if depth is not None:
        depth = resize(depth)
        if do_flip:
            depth = torch.flip(depth, dims=[-1])
        depth = normalize_depth_fn(depth)
        sample["depth"] = depth.contiguous()

    normal = sample.get("normal", None)
    if normal is not None:
        normal = resize(normal)
        if do_flip:
            normal = torch.flip(normal, dims=[-1])
            normal[0] = - normal[0]  # invert x axis
        normal = normalize_normal_fn(normal)
        sample["normal"] = normal.contiguous()

    albedo = sample.get("albedo", None)
    if albedo is not None:
        albedo = resize(albedo)
        if do_flip:
            albedo = torch.flip(albedo, dims=[-1])
        sample["albedo"] = albedo.contiguous()

    shading = sample.get("shading", None)
    if shading is not None:
        shading = resize(shading)
        if do_flip:
            shading = torch.flip(shading, dims=[-1])
        sample["shading"] = shading.contiguous()

    specular = sample.get("specular", None)
    if specular is not None:
        specular = resize(specular)
        if do_flip:
            specular = torch.flip(specular, dims=[-1])
        sample["specular"] = specular.contiguous()

    diffuse = sample.get("diffuse", None)
    if diffuse is not None:
        diffuse = resize(diffuse)
        if do_flip:
            diffuse = torch.flip(diffuse, dims=[-1])
        sample["diffuse"] = diffuse.contiguous()
    return sample


def hypersim_test_preprocess(sample):
    # TODO implement trainning
    resize = Resize(480, antialias=True)
    sample["image"] = resize(sample["image"])
    sample["normal"] = resize(sample["normal"])
    return sample


# Global variables that control the behavior of the data loader
normalize_normal_fn = None
hypersim_resolution = None
normalize_depth_fn = None
set_depth_normalize_fn("marigold")
set_normal_normalize_fn("clipnorm")
set_hypersim_resolution(480)
preprocess_functions = {
    "hypersim": {"train": hypersim_train_preprocess, "test": hypersim_test_preprocess}
}
