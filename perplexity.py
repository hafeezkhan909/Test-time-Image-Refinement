# import torch
# import math
# import pandas as pd
# from tqdm import tqdm
# from transformers import AutoModelForCausalLM, AutoTokenizer

# # === Load GPT-2 Model for Perplexity Calculation ===
# gpt2_model_name = "openai-community/gpt2-large"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# gpt2_tokenizer = AutoTokenizer.from_pretrained(gpt2_model_name)
# gpt2_model = AutoModelForCausalLM.from_pretrained(gpt2_model_name).to(device)

# # === Function to Compute Perplexity Using Sliding Window Approach ===
# def compute_perplexity_gpt2(text, model, tokenizer, stride=256):
#     """
#     Computes the perplexity of a given text using GPT-2 with a sliding window.

#     Args:
#         text (str): The input text (refined prompt).
#         model: Pretrained GPT-2 model.
#         tokenizer: GPT-2 tokenizer.
#         stride (int): Overlap between input chunks to improve accuracy.

#     Returns:
#         float: The perplexity score (lower is better).
#     """
#     # Tokenize input text
#     encodings = tokenizer(text, return_tensors="pt")
#     input_ids = encodings.input_ids.to(device)

#     max_length = model.config.n_positions  # Max sequence length for GPT-2
#     seq_len = input_ids.size(1)

#     # Handle very short text cases
#     if seq_len < 10:
#         print("⚠️ Warning: Input text is very short. Perplexity may be inaccurate.")
    
#     nll_sum = 0.0
#     n_tokens = 0
#     prev_end_loc = 0

#     for begin_loc in tqdm(range(0, seq_len, stride), desc="Processing Text"):
#         end_loc = min(begin_loc + max_length, seq_len)
#         trg_len = end_loc - prev_end_loc  # May differ on last iteration
#         input_chunk = input_ids[:, begin_loc:end_loc]

#         target_ids = input_chunk.clone()
#         target_ids[:, :-trg_len] = -100  # Ignore tokens for loss calculation

#         with torch.no_grad():
#             outputs = model(input_chunk, labels=target_ids)
#             neg_log_likelihood = outputs.loss

#         # Count valid tokens
#         num_valid_tokens = (target_ids != -100).sum().item()
#         batch_size = target_ids.size(0)
#         num_loss_tokens = max(1, num_valid_tokens - batch_size)  # Avoid zero division

#         nll_sum += neg_log_likelihood * num_loss_tokens
#         n_tokens += num_loss_tokens
#         prev_end_loc = end_loc

#         if end_loc == seq_len:
#             break

#     avg_nll = nll_sum / n_tokens  # Average negative log-likelihood per token
#     perplexity = torch.exp(avg_nll).item()
    
#     return perplexity

# # === Example Refined Prompts from Qwen2.5-VL-7B-Instruct ===
# refined_prompts = {
#     "10": "A blurry figure with unclear features. No objects found and indistinct background.",
#     "25": "A blurry figure of an animal with unclear features. Background indistinct.",
#     "50": "A dog and a cat are present, but the colors are faded. Background unclear.",
#     "65": "A white dog with an orange cat on a grassy hill, but some details are missing.",
#     "75": "An angry white dog with bared teeth next to a wide-eyed orange cat, sitting on a grassy hill under a warm sunset."
# }

# # === Compute Perplexity for Each Refined Prompt ===
# results = []
# for quality, prompt in refined_prompts.items():
#     print(f"\n🔹 Calculating Perplexity for: {quality}")
#     perplexity = compute_perplexity_gpt2(prompt, gpt2_model, gpt2_tokenizer)
#     results.append({"Image Quality": quality, "Perplexity": perplexity})

# # === Convert to DataFrame and Display Results ===
# df = pd.DataFrame(results)
# print("\n=== Perplexity Scores for Different Image Qualities ===")
# print(df)

# # === Optional: Plot the Perplexity vs. Image Quality ===
# import matplotlib.pyplot as plt

# plt.figure(figsize=(8, 5))
# plt.plot(df["Image Quality"], df["Perplexity"], marker="o", linestyle="-", color="b")
# plt.xlabel("Image Quality")
# plt.ylabel("Perplexity")
# plt.title("Perplexity of Refined Prompts vs. Latent Image Quality")
# plt.xticks(rotation=30)
# plt.grid(True)
# plt.show()

import torch
import math
import json
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# === Load GPT-2 Model for Perplexity Calculation ===
gpt2_model_name = "openai-community/gpt2-large"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gpt2_tokenizer = GPT2TokenizerFast.from_pretrained(gpt2_model_name)
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name).to(device)

# === Function to Compute Perplexity (Hugging Face Method) ===
def compute_perplexity_gpt2(text, model, tokenizer, stride=512):
    """
    Computes perplexity of a given text using GPT-2 with Hugging Face's sliding window approach.
    """
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)

    max_length = model.config.n_positions  # GPT-2 max sequence length
    seq_len = input_ids.size(1)

    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0

    for begin_loc in tqdm(range(0, seq_len, stride), desc="Processing Text"):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # May differ on last iteration
        input_chunk = input_ids[:, begin_loc:end_loc]

        target_ids = input_chunk.clone()
        target_ids[:, :-trg_len] = -100  # Ignore context tokens

        with torch.no_grad():
            outputs = model(input_chunk, labels=target_ids)
            neg_log_likelihood = outputs.loss

        num_valid_tokens = (target_ids != -100).sum().item()
        batch_size = target_ids.size(0)
        num_loss_tokens = num_valid_tokens - batch_size  # Adjust for shifted labels

        nll_sum += neg_log_likelihood * num_loss_tokens
        n_tokens += num_loss_tokens
        prev_end_loc = end_loc

        if end_loc == seq_len:
            break
       
    avg_nll = nll_sum / n_tokens  # Average negative log-likelihood per token
    perplexity = torch.exp(avg_nll).item()

    return perplexity

# === Load Refined Prompts from JSON ===
json_filename = "refined_prompts.json"
with open(json_filename, "r") as f:
    refined_prompt_sets = json.load(f)

# === Compute Perplexity for Each Step Across All Prompts ===
step_names = ["Step 10", "Step 25", "Step 50", "Step 65", "Step 75"]
step_perplexities = {step: [] for step in step_names}

for idx, prompt_set in enumerate(refined_prompt_sets):
    print(f"\n🔹 Processing Prompt Set {idx+1}/{len(refined_prompt_sets)}")
    for step in step_names:
        perplexity = compute_perplexity_gpt2(prompt_set[step], gpt2_model, gpt2_tokenizer)
        print(f"perplexity score for {idx} and {step}: {perplexity}")
        step_perplexities[step].append(perplexity)

# === Compute Aggregate Perplexity (Mean for Each Step) ===
aggregate_results = {step: sum(perplexities) / len(perplexities) for step, perplexities in step_perplexities.items()}

# Convert to DataFrame and Display Results
df_aggregate = pd.DataFrame(aggregate_results.items(), columns=["Step", "Average Perplexity"])
df_aggregate_sorted = df_aggregate.sort_values(by="Step")

# === Save and Plot Perplexity Graph ===
plt.figure(figsize=(8, 5))
plt.plot(df_aggregate_sorted["Step"], df_aggregate_sorted["Average Perplexity"], marker="o", linestyle="-", color="b")
plt.xlabel("Latent Image Step")
plt.ylabel("Average Perplexity")
plt.title("Aggregate Perplexity of Refined Prompts Across Steps")
plt.xticks(rotation=30)
plt.grid(True)
plt.savefig("aggregate_perplexity_vs_steps.png")  # Saves the plot
plt.show()

# Print the aggregate perplexity scores
print("\n=== Aggregate Perplexity Scores ===")
print(df_aggregate_sorted)
