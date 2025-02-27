import os
import json
import argparse

def load_json(file_path):
    """Loads a JSON file and returns its content."""
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found.")
        return {}
    with open(file_path, "r") as f:
        return json.load(f)

def compute_average(scores):
    """Computes the average score from a list."""
    return sum(scores) / len(scores) if scores else 0

def extract_simple_average(file_path):
    """Extracts scores from a JSON file where values are directly mapped."""
    data = load_json(file_path)
    return compute_average(list(data.values()))

def extract_clipscore(file_path):
    """Extracts CLIPScore values and computes the average."""
    data = load_json(file_path)
    clip_scores = [entry["CLIPScore"] for entry in data.values()]
    return compute_average(clip_scores)

def extract_xiqe_scores(file_path):
    """Extracts Fidelity, Alignment, and Aesthetics scores from X-IQE results."""
    xiqe_data = load_json(file_path)

    fidelity_scores = []
    alignment_scores = []
    aesthetics_scores = []

    for image_name, metrics in xiqe_data.items():
        try:
            if "Fidelity" in metrics and "Fidelity" in metrics["Fidelity"]:
                fidelity_score = int(metrics["Fidelity"]["Fidelity"].split("/")[0])  # Extract numeric value
                fidelity_scores.append(fidelity_score)
        except (KeyError, ValueError):
            print(f"Warning: Fidelity score missing or incorrect for {image_name}")

        try:
            if "Alignment" in metrics and "Alignment score" in metrics["Alignment"]:
                alignment_score = int(metrics["Alignment"]["Alignment score"].split("/")[0])  # Extract numeric value
                alignment_scores.append(alignment_score)
        except (KeyError, ValueError):
            print(f"Warning: Alignment score missing or incorrect for {image_name}")

        try:
            if "Aesthetics" in metrics and "Overall aesthetic score" in metrics["Aesthetics"]:
                aesthetics_score = int(metrics["Aesthetics"]["Overall aesthetic score"].split("/")[0])  # Extract numeric value
                aesthetics_scores.append(aesthetics_score)
        except (KeyError, ValueError):
            print(f"Warning: Aesthetic score missing or incorrect for {image_name}")

    avg_fidelity = compute_average(fidelity_scores)
    avg_alignment = compute_average(alignment_scores)
    avg_aesthetics = compute_average(aesthetics_scores)
    overall_xiqe_score = avg_fidelity + avg_alignment + avg_aesthetics

    return avg_fidelity, avg_alignment, avg_aesthetics, overall_xiqe_score

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Compute evaluation metrics for DrawBench results")
    parser.add_argument("--eval_dir", type=str, required=True, help="Path to the directory containing evaluation results (e.g., drawbench/drawbench_eval)")
    args = parser.parse_args()

    eval_dir = args.eval_dir

    # File paths
    aesthetic_results_path = os.path.join(eval_dir, "aesthetic_results.json")
    clipscore_results_path = os.path.join(eval_dir, "clipscore_results.json")
    hps_results_path = os.path.join(eval_dir, "hps_results.json")
    imagereward_results_path = os.path.join(eval_dir, "imagereward_results.json")
    xiqe_results_path = os.path.join(eval_dir, "xiqe_results.json")

    # Compute average scores
    average_aesthetic_score = extract_simple_average(aesthetic_results_path)
    average_clipscore = extract_clipscore(clipscore_results_path)
    average_hps_score = extract_simple_average(hps_results_path)
    average_imagereward_score = extract_simple_average(imagereward_results_path)

    avg_fidelity, avg_alignment, avg_aesthetics, overall_xiqe_score = extract_xiqe_scores(xiqe_results_path)

    # Print individual scores
    print(f"Average Aesthetic Score: {average_aesthetic_score}")
    print(f"Average CLIPScore: {average_clipscore}")
    print(f"Average HPS Score: {average_hps_score}")
    print(f"Average ImageReward Score: {average_imagereward_score}")
    print(f"Average Fidelity Score: {avg_fidelity}")
    print(f"Average Alignment Score: {avg_alignment}")
    print(f"Average Aesthetic Score (X-IQE): {avg_aesthetics}")
    print(f"Overall X-IQE Score: {overall_xiqe_score}")

    # Compute weighted average score
    weighted_avg_score = (
        average_clipscore * 0.15 +
        average_aesthetic_score * 0.15 +
        average_imagereward_score * 0.20 +
        average_hps_score * 0.20 +
        overall_xiqe_score * 0.10
    )

    print(f"\n🎯 Weighted Average Score: {weighted_avg_score}")

if __name__ == "__main__":
    main()
