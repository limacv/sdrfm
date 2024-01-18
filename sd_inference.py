import numpy as np
import torch
from PIL import Image
from sd_pipeline import StableDiffusionPipeline
from diffusers.utils import load_image

pipe = StableDiffusionPipeline.from_pretrained(
    'CompVis/stable-diffusion-v1-4'
)
pipe.to("cuda")
width = 768
height = 568

image = load_image("res_elephant.png")
input_image = image.resize((width, height)).convert("RGB")
image = np.asarray(input_image)
rgb = np.transpose(image, (2, 0, 1))  # [H, W, rgb] -> [rgb, H, W]
rgb_norm = rgb / 255.0
rgb_norm = torch.from_numpy(rgb_norm).to(torch.float32)
rgb_norm = rgb_norm.cuda()[None]

# vae encode
h = pipe.vae.encoder(rgb_norm)
moments = pipe.vae.quant_conv(h)
mean, logvar = torch.chunk(moments, 2, dim=1)
rgb_latent = mean * pipe.vae.config.scaling_factor

# vae output
# vae_out = pipe.vae.decode(rgb_latent / pipe.vae.config.scaling_factor)
# vae_out = vae_out[0][0].detach().permute(1,2,0).cpu().numpy() * 255
# vae_out = np.clip(vae_out, 0, 255).astype(np.uint8)

# # save
# input_image.save("res_input.png")
# Image.fromarray(vae_out).save("res_vae.png")


prompt = "depth map of a room"
torch.random.manual_seed(10)
noise = torch.randn_like(rgb_latent)
# for i, w in enumerate(np.linspace(1, 0.4, 10)):
#     output = pipe(prompt, width=width, height=height, latents=noise * w + rgb_latent * (1 - w))
#     output.images[0].save(f"res_{i}.png")

output = pipe(prompt, width=width, height=height, latents=noise)
output.images[0].save("res.png")

# output = pipe(prompt, width=width, height=height, latents=rgb_latent, num_inference_steps=999)
# output.images[0].save("res.png")
