import torch
import numpy as np
from PIL import Image

def decode_latents(latents, vae):
    """Decodes latents into an image using VAE."""
    latents = 1 / 0.18215 * latents  # Scale latents
    with torch.no_grad():
        image = vae.decode(latents).sample  # Decode using VAE
    image = (image / 2 + 0.5).clamp(0, 1)  # Normalize
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()[0]
    return Image.fromarray((image * 255).astype("uint8"))

def save_image(image, filename, folder="outputs/"):
    """Saves image to the specified folder."""
    image.save(f"{folder}/{filename}")
