from PIL import Image
import os

def load_image(image_path):
    """Loads an image from the given path."""
    return Image.open(image_path).convert("RGB")

def save_image(fig, save_path):
    """Saves the plotted figure to the specified path."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"✅ Image saved at: {save_path}")
