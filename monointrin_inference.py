import numpy as np
import torch
from PIL import Image
from mononorm_pipeline import MonoNormPipeline
from diffusers.utils import load_image
import cv2
import os
from glob import glob
from tqdm import tqdm


pipe = MonoNormPipeline.from_pretrained(
    # "Bingxin/Marigold",
    "/cpfs01/shared/pjlab-lingjun-landmarks/mali1/outputs/Normv2_norm",
    # torch_dtype=torch.float16,                # (optional) Run with half-precision (16-bit float).
)

pipe.to("cuda")

# data load
input_rgb_dir = 'examples'
output_dir = 'outputs/marigold/test_log_val/normal2'
os.makedirs(output_dir,exist_ok=True)
EXTENSION_LIST=[".jpg", ".jpeg", ".png"]
rgb_filename_list = glob(os.path.join(input_rgb_dir, "*"))
rgb_filename_list = [
    f for f in rgb_filename_list if os.path.splitext(f)[1].lower() in EXTENSION_LIST
]
rgb_filename_list = sorted(rgb_filename_list)
n_images = len(rgb_filename_list)

for rgb_path in tqdm(rgb_filename_list, desc="Estimating depth", leave=True):
    image: Image.Image = load_image(rgb_path)

    pipeline_output = pipe(
        image,                  # Input image.
        denoising_steps=20,     # (optional) Number of denoising steps of each inference pass. Default: 10.
        ensemble_size=10,       # (optional) Number of inference passes in the ensemble. Default: 10.
        processing_res=768,     # (optional) Maximum resolution of processing. If set to 0: will not resize at all. Defaults to 768.
        match_input_res=True,   # (optional) Resize depth prediction to match input resolution.
        batch_size=0,           # (optional) Inference batch size, no bigger than `num_ensemble`. If set to 0, the script will automatically decide the proper batch size. Defaults to 0.
        color_map="Spectral",   # (optional) Colormap used to colorize the depth map. Defaults to "Spectral".
        show_progress_bar=True, # (optional) If true, will show progress bars of the inference progress.
    )
    uncertainty: torch.Tensor = pipeline_output.uncertainty                    # Predicted uncertainty map
    normal: np.ndarray = pipeline_output.normal_np                    # Predicted depth map
    normal_colored: Image.Image = pipeline_output.normal_pil      # Colorized prediction

    rgb_name_base = os.path.splitext(os.path.basename(rgb_path))[0]
    pred_name_base = rgb_name_base + "_pred"

    # Save colorized depth map
    normal_colored.save(os.path.join(output_dir, f"{pred_name_base}_normal.png"))

    if uncertainty is not None:
        uncertainty = (uncertainty * 255).clamp(0, 255).detach().cpu().numpy().astype(np.uint8)
        uncertainty = cv2.applyColorMap(uncertainty, cv2.COLORMAP_VIRIDIS)
        cv2.imwrite(os.path.join(output_dir, f"{pred_name_base}_uncerty.png"), uncertainty)

