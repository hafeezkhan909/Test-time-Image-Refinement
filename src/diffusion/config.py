import torch

# Device setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Default Image Generation Parameters
# (overridable via CLI args in main.py: --height, --width, --num_inference_steps, --guidance_scale)
HEIGHT = 512
WIDTH = 512
NUM_INFERENCE_STEPS = 100
GUIDANCE_SCALE = 7.5
BATCH_SIZE = 1