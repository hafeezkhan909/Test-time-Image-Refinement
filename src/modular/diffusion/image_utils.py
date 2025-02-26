import torch
import numpy as np
from PIL import Image
import os

def decode_latents(latents, vae):
    """Decodes latents into an image using VAE."""
    latents = 1 / 0.18215 * latents  # Scale latents
    with torch.no_grad():
        image = vae.decode(latents).sample  # Decode using VAE
    image = (image / 2 + 0.5).clamp(0, 1)  # Normalize
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()[0]
    return Image.fromarray((image * 255).astype("uint8"))

def save_image(image, filename, folder=None):
    """
    Save a PIL image to the specified filename and optional folder.
    If folder is provided, it will be joined with the filename.
    """
    if folder:
        # Ensure the folder exists
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, filename)
    else:
        # Assume filename is the full path
        filepath = filename
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Save the image
    image.save(filepath)
    print(f"Image saved to {filepath}")
