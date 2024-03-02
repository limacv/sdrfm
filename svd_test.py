import numpy as np
import torch
from torchvision.transforms import ToTensor
from PIL import Image
from svd_pipeline import StableVideoDiffusionPipeline, tensor2vid
from diffusers.utils import load_image, export_to_video

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
)
pipe.enable_model_cpu_offload()
# pipe.to("cuda")

# Load the conditioning image
image = load_image("examples/video/rocket.png")
image = image.resize((1024, 576))

num_frames = 3
image = pipe.image_processor.preprocess(image, height=576, width=1024).to(dtype=torch.float16, device='cuda:0')
# image = ToTensor()(image) * 2 - 1
# image = image[None].cuda()
pipe.vae.to(dtype=torch.float16)
latent = pipe.vae.encode(image).latent_dist.mode() * pipe.vae.config.scaling_factor
latent = latent.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)
frames = pipe.decode_latents(latent, num_frames, decode_chunk_size = 3)
frames = tensor2vid(frames.detach(), pipe.image_processor, "pil")
for i, f in enumerate(frames[0]):
    f.save(f"outputs/svd/{i:02d}.jpg")