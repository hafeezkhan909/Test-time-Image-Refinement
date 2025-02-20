import torch
import random

# Device setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Image Generation Parameters
HEIGHT = 512
WIDTH = 512
NUM_INFERENCE_STEPS = 100
GUIDANCE_SCALE = 7.5
BATCH_SIZE = 1
# GENERATOR_SEED = 488226
# GENERATOR_SEED = random.randint(0, 1000000)
