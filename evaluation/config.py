import torch

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# OWL-ViT model checkpoint
MODEL_NAME = "google/owlvit-base-patch32"