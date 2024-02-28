import numpy as np
import torch
from PIL import Image
from svd_pipeline import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
)
pipe.enable_model_cpu_offload()
# pipe.to("cuda")

# Load the conditioning image
image = load_image("examples/video/rocket.png")
image = image.resize((1024, 576))

generator = torch.manual_seed(42)
frames = pipe(image, decode_chunk_size=14, generator=generator).frames[0]
export_to_video(frames, "outputs/svd/generated.mov", fps=7)
