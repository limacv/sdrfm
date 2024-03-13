import numpy as np
import torch
from PIL import Image
from monointrin_pipeline import MonoIntrinPipeline
from diffusers.utils import load_image
import cv2
import os
from glob import glob
from tqdm import tqdm


pipe = MonoIntrinPipeline.from_pretrained(
    # "Bingxin/Marigold",
    "/cpfs01/shared/pjlab-lingjun-landmarks/mali1/outputs/Intrin480DeNo",
    # torch_dtype=torch.float16,                # (optional) Run with half-precision (16-bit float).
)

pipe.to("cuda")

# data load
input_rgb_dir = 'examples'
output_dir = 'outputs/Intrin480DeNo'
os.makedirs(output_dir,exist_ok=True)
EXTENSION_LIST=[".jpg", ".jpeg", ".png"]
rgb_filename_list = glob(os.path.join(input_rgb_dir, "*"))
rgb_filename_list = [
    f for f in rgb_filename_list if os.path.splitext(f)[1].lower() in EXTENSION_LIST
]
rgb_filename_list = sorted(rgb_filename_list)
n_images = len(rgb_filename_list)

for rgb_path in tqdm(rgb_filename_list, desc="Estimating", leave=True):
    image: Image.Image = load_image(rgb_path)

    pipeline_output = pipe(
        image,                  # Input image.
        ["depth", "normal"],
        denoising_steps=20,     # (optional) Number of denoising steps of each inference pass. Default: 10.
        ensemble_size=10,       # (optional) Number of inference passes in the ensemble. Default: 10.
        processing_res=768,     # (optional) Maximum resolution of processing. If set to 0: will not resize at all. Defaults to 768.
        match_input_res=True,   # (optional) Resize depth prediction to match input resolution.
        batch_size=0,           # (optional) Inference batch size, no bigger than `num_ensemble`. If set to 0, the script will automatically decide the proper batch size. Defaults to 0.
        show_progress_bar=True, # (optional) If true, will show progress bars of the inference progress.
    )
    
    name_base = os.path.splitext(os.path.basename(rgb_path))[0]
    pred_name_base = name_base + "_pred"
    os.makedirs(output_dir, exist_ok=True)
    for asset_k, (asset, asset_pil, asset_uncert) in pipeline_output.items():
        asset_pil.save(os.path.join(output_dir, f"{pred_name_base}_{asset_k}.png"))

        if asset_uncert is not None:
            asset_uncert = (asset_uncert * 5 * 255).clamp(0, 255).detach().cpu().numpy().astype(np.uint8)
            asset_uncert = cv2.applyColorMap(asset_uncert, cv2.COLORMAP_VIRIDIS)
            cv2.imwrite(os.path.join(output_dir, f"{pred_name_base}_{asset_k}_uncert.png"), asset_uncert)
