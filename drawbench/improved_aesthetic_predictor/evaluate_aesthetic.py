import torch
import clip
from PIL import Image
import os
import json
import numpy as np
import argparse
import pytorch_lightning as pl
import torch.nn as nn

# Argument parser to accept the image folder dynamically
parser = argparse.ArgumentParser("Evaluate Aesthetic Scores for All Images")
parser.add_argument("--image_folder", required=True, type=str, help="Path to the folder containing images")
parser.add_argument("--model_path", required=True, type=str, help="Path to the aesthetic model checkpoint")
parser.add_argument("--output_json", default="drawbench/drawbench_eval/aesthetic_results.json", type=str, help="Path to save the output JSON")
args = parser.parse_args()


# Define MLP Model (same as simple_inference.py)
class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)


def normalized(a, axis=-1, order=2):
    """L2 normalize the embeddings"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


# Load Aesthetic Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MLP(768).to(device)  # CLIP embedding dim is 768 for CLIP ViT-L/14
model.load_state_dict(torch.load(args.model_path))
model.eval()

# Load CLIP Model
clip_model, preprocess = clip.load("ViT-L/14", device=device)

# Evaluate all images in the folder
aesthetic_scores = {}

for img_name in os.listdir(args.image_folder):
    if img_name.endswith((".png", ".jpg", ".jpeg", ".webp")):
        img_path = os.path.join(args.image_folder, img_name)

        pil_image = Image.open(img_path)
        image = preprocess(pil_image).unsqueeze(0).to(device)

        # Extract CLIP image features
        with torch.no_grad():
            image_features = clip_model.encode_image(image)

        im_emb_arr = normalized(image_features.cpu().detach().numpy())
        prediction = model(torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor))

        aesthetic_scores[img_name] = prediction.item()  # Save as float

# Save results in JSON
with open(args.output_json, "w") as f:
    json.dump(aesthetic_scores, f, indent=4)

print(f"\n✅ Aesthetic Evaluation Complete! Results saved in `{args.output_json}`.")
