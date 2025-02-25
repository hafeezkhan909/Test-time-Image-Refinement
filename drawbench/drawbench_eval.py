import os
import json
import argparse
import ImageReward as RM

# Parse arguments from user
parser = argparse.ArgumentParser(description="Evaluate generated images using multiple metrics")
parser.add_argument("--image_folder", required=True, type=str, help="Path to folder containing generated images")
parser.add_argument("--prompt_json", required=True, type=str, help="Path to JSON file containing prompts")
parser.add_argument("--output_dir", default="drawbench/drawbench_eval", type=str, help="Directory to save evaluation results")
args = parser.parse_args()

# Ensure output directory exists
os.makedirs(args.output_dir, exist_ok=True)

print("\n🔹 Running Evaluation Pipeline...")

# ================================
# 1️⃣ CLIPScore Evaluation
# ================================
clipscore_output = os.path.join(args.output_dir, "clipscore_results.json")
print("\n🔹 Running CLIPScore Evaluation...")
os.system(f"python drawbench/clipscore/clipscore.py {args.prompt_json} {args.image_folder} --save_per_instance {clipscore_output}")
print(f"✅ CLIPScore results saved at `{clipscore_output}`")

# ================================
# 2️⃣ Improved Aesthetic Predictor Evaluation
# ================================
aesthetic_model_path = "drawbench/improved_aesthetic_predictor/sac+logos+ava1-l14-linearMSE.pth"
aesthetic_output = os.path.join(args.output_dir, "aesthetic_results.json")
print("\n🔹 Running Aesthetic Evaluation...")
os.system(f"python drawbench/improved_aesthetic_predictor/evaluate_aesthetic.py --image_folder {args.image_folder} --model_path {aesthetic_model_path} --output_json {aesthetic_output}")
print(f"✅ Aesthetic results saved at `{aesthetic_output}`")

# ================================
# 3️⃣ ImageReward Evaluation
# ================================
image_reward_output = os.path.join(args.output_dir, "imagereward_results.json")
print("\n🔹 Running ImageReward Evaluation...")
model = RM.load("ImageReward-v1.0")

# Load prompts
with open(args.prompt_json, "r") as f:
    prompts = json.load(f)

# Compute scores for each image
image_reward_scores = {}
for image_name in os.listdir(args.image_folder):
    if image_name.endswith(('.png', '.jpg', '.jpeg', '.webp')):
        image_path = os.path.join(args.image_folder, image_name)
        prompt = prompts.get(image_name, "No prompt available")
        score = model.score(prompt, image_path)
        image_reward_scores[image_name] = score

# Save ImageReward results
with open(image_reward_output, "w") as f:
    json.dump(image_reward_scores, f, indent=4)

print(f"✅ ImageReward results saved at `{image_reward_output}`")

# ================================
# 4️⃣ Human Preference Score (HPS) Evaluation
# ================================
hpc_checkpoint = "drawbench/hps/hpc.pt" # Download the hpc.pt from https://mycuhk-my.sharepoint.com/:u:/g/personal/1155172150_link_cuhk_edu_hk/EWDmzdoqa1tEgFIGgR5E7gYBTaQktJcxoOYRoTHWzwzNcw?e=b7rgYW
hps_meta_path = os.path.join(args.output_dir, "hps_meta.json")
hps_output = os.path.join(args.output_dir, "hps_results.json")

print("\n🔹 Preparing HPS metadata...")
hps_meta = []
for image_name, prompt in prompts.items():
    hps_meta.append({
        "human_preference": 1,
        "prompt": prompt,
        "id": len(hps_meta) + 1,
        "file_path": [os.path.join(args.image_folder, image_name)],
        "user_hash": "example_hash",
        "contain_name": False
    })

# Save HPS meta file
with open(hps_meta_path, "w") as f:
    json.dump(hps_meta, f, indent=4)

print("\n🔹 Running HPS Evaluation...")
os.system(f"python drawbench/hps/evaluate_hps.py --image_folder {args.image_folder} --hpc {hpc_checkpoint} --meta_file {hps_meta_path}")
print(f"✅ HPS results saved at `{hps_output}`")

# ================================
# 5️⃣ X-IQE Evaluation
# ================================
xiqe_output = os.path.join(args.output_dir, "xiqe_results.json")
print("\n🔹 Running X-IQE Evaluation...")
os.system(f"python drawbench/xiqe_evaluation.py --image_folder {args.image_folder} --prompt_json {args.prompt_json}")
print(f"✅ X-IQE results saved at `{xiqe_output}`")

# ================================
# 🎯 Final Step: Completion Message
# ================================
print("\n✅ All evaluations completed! Check results in:", args.output_dir)
