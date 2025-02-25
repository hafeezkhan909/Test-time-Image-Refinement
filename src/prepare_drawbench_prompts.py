import json
from datasets import load_dataset
from collections import defaultdict

# ======================== #
# 🔹 Load DrawBench and Save in Structured Format
# ======================== #
def save_drawbench_prompts(output_path="drawbench_prompts.json"):
    """Loads DrawBench dataset and saves prompts categorized by their category."""

    print("\n🔹 Loading DrawBench dataset...")
    dataset = load_dataset("shunk031/DrawBench")

    # Dictionary to store prompts categorized by 'category'
    categorized_prompts = defaultdict(list)

    print("\n✅ Dataset Loaded! Organizing prompts by category...")

    # Iterate through dataset and group prompts by category
    for idx, entry in enumerate(dataset["test"]):
        category = str(entry["category"])  # Convert category to string (JSON keys must be strings)
        prompt = entry["prompts"]

        categorized_prompts[category].append({
            "prompt": prompt,
            "line_number": idx + 1
        })

    # Save to JSON file
    with open(output_path, "w") as f:
        json.dump(categorized_prompts, f, indent=4)

    print(f"\n✅ DrawBench prompts saved successfully in `{output_path}`!")

# ======================== #
# 🔹 Run the script
# ======================== #
if __name__ == "__main__":
    save_drawbench_prompts()
