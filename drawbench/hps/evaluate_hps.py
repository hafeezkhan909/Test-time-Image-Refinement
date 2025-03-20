import torch
import clip
from PIL import Image
from torch.utils.data import Dataset
import json
import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser("Evaluate HPS")
parser.add_argument("--hpc", required=True, type=str, help="path to hpc checkpoint")
parser.add_argument("--image_folder", required=True, type=str, help="path to image folder")
parser.add_argument("--meta_file", required=True, type=str, help="path to meta file")
parser.add_argument("--output_json", required=True, type=str, help="Path to save HPS results JSON file")
parser.add_argument("--batch_size", default=16, type=int, help="batch size")
parser.add_argument("--num_workers", default=4, type=int, help="number of workers")
parser.add_argument("--device", default="cuda", type=str, help="device")
args = parser.parse_args()

device = args.device
model, preprocess = clip.load("ViT-L/14", device=device)
checkpoint = torch.load(args.hpc, weights_only=False)  # Load full checkpoint
params = checkpoint["state_dict"]  # Extract only model weights

model.load_state_dict(params)

class ImageTextDataset(Dataset):
    def __init__(self, meta_file, image_folder, transforms, tokenizer):
        with open(meta_file, 'r') as f:
            self.datalist = json.load(f)
        self.folder = image_folder
        self.transforms = transforms
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(self.datalist[idx]['file_path'][0]))
        input_ids = self.tokenizer(self.datalist[idx]['prompt'], context_length=77, truncate=True)[0]
        return images, input_ids

dataset = ImageTextDataset(args.meta_file, args.image_folder, preprocess, clip.tokenize)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

scores = []
with torch.no_grad():
    for i, (images, text) in tqdm(enumerate(dataloader)):
        images = images.to(device)
        text = text.to(device)

        image_features = model.encode_image(images)
        text_features = model.encode_text(text)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        hps = image_features @ text_features.T
        hps = hps.diagonal()
        if isinstance(hps, torch.Tensor):
            scores.extend(hps.squeeze().tolist() if hps.numel() > 1 else [hps.item()])
        else:
            scores.append(hps)  # In case hps is already a float


# print(f"HPS: {sum(scores) / len(scores)}")

# Save HPS results in JSON format
# Extract only the filename instead of full file path
hps_results = {}
for idx, score in enumerate(scores):
    image_filename = os.path.basename(dataset.datalist[idx]["file_path"][0])
    hps_results[image_filename] = score

# Save JSON file
with open(args.output_json, "w") as f:
    json.dump(hps_results, f, indent=4)

# print(f"\n✅ HPS evaluation complete! Results saved in `{args.output_json}`.")

