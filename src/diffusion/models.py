import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import CLIPTextModelWithProjection, CLIPTokenizer as OpenCLIPTokenizer

def load_models(device, model_version="1.5"):
    """
    Load models for the specified Stable Diffusion version.
    
    Args:
        device (str): Device to load models on (e.g., "cuda" or "cpu").
        model_version (str): Stable Diffusion version (e.g., "1.4", "1.5", "2.1").
    
    Returns:
        vae, tokenizer, text_encoder, unet, scheduler
    """
    # Define model paths based on version
    model_paths = {
        "1.4": "CompVis/stable-diffusion-v1-4",
        "1.5": "sd-legacy/stable-diffusion-v1-5",
        "2.1": "stabilityai/stable-diffusion-2-1"
    }
    
    if model_version not in model_paths:
        raise ValueError(f"Unsupported model version: {model_version}. Choose from {list(model_paths.keys())}")
    
    model_path = model_paths[model_version]
    
    # Load VAE, UNet, and scheduler
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet").to(device)
    scheduler = LMSDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
    
    # Load text encoder and tokenizer based on model version
    if model_version in ["1.4", "1.5"]:
        # Use CLIP for Stable Diffusion 1.4/1.5
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    elif model_version == "2.1":
        # Use OpenCLIP for Stable Diffusion 2.1
        tokenizer = OpenCLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="tokenizer")
        text_encoder = CLIPTextModelWithProjection.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="text_encoder").to(device)
    
    return vae, tokenizer, text_encoder, unet, scheduler