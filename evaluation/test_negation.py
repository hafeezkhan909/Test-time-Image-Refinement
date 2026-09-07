import json
import os
import argparse
from object_detection import ObjectDetector
from image_utils import load_image
from visualization import save_detection_visualization
from image_selection import get_final_image_path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate negation prompts against generated images.")
    parser.add_argument("--query_json", type=str, default="json_files/negation_specific.json",
                         help="Path to the negation query JSON (keyed by line_number).")
    parser.add_argument("--images_dir", type=str, required=True,
                         help="Path to the pipeline's output_dir (contains <tag>/prompt_XXX/ subfolders).")
    parser.add_argument("--output_json", type=str, default="negation_results.json",
                         help="Path to save the results JSON.")
    parser.add_argument("--visualization_dir", type=str, default="detection_visualization/negation",
                         help="Directory to save detection visualization images.")
    parser.add_argument("--refinement_iterations", type=int, default=3,
                         help="Number of refinement rounds the pipeline ran (default: 3).")
    parser.add_argument("--last_restart_step", type=int, default=0,
                         help="Restart step used for the final refinement round (default: 0).")
    parser.add_argument("--selection_mode", type=str, default="first_early_stop",
                         choices=["first_early_stop", "final_only"],
                         help="How to pick the evaluated image per prompt (default: first_early_stop).")
    return parser.parse_args()


def main():
    args = parse_args()
    tag = "negation"

    with open(args.query_json, "r") as f:
        test_queries = json.load(f)  # Format: {"1": {"object": "laptop"}, ...}

    detector = ObjectDetector()

    total_correct = 0
    valid_queries = 0
    results = {}

    for line_number, query_data in test_queries.items():
        forbidden_object = query_data["object"]

        prompt_id = f"prompt_{int(line_number):03d}"
        prompt_dir = os.path.join(args.images_dir, tag, prompt_id)
        image_path = get_final_image_path(
            prompt_dir,
            refinement_iterations=args.refinement_iterations,
            last_restart_step=args.last_restart_step,
            selection_mode=args.selection_mode,
        )

        if image_path is None:
            print(f"Image not found for {prompt_id}, skipping...")
            continue

        valid_queries += 1
        image = load_image(image_path)

        detected_objects, bounding_boxes, scores = detector.detect_objects(image, [forbidden_object])

        accuracy = 100 if forbidden_object not in detected_objects else 0
        total_correct += accuracy

        save_path = os.path.join(args.visualization_dir, f"{prompt_id}_negation.png")
        save_detection_visualization(image, detected_objects, bounding_boxes, scores, save_path)

        results[line_number] = {
            "image_path": image_path,
            "forbidden_object": forbidden_object,
            "detected_objects": detected_objects,
            "bounding_boxes": bounding_boxes,
            "confidence_scores": scores,
            "accuracy": accuracy,
            "visualization_path": save_path
        }

        print(f"\n=== {prompt_id} Results ===")
        print(f"Expected absence of: {forbidden_object}")
        print(f"Detected objects: {detected_objects}")
        print(f"Accuracy: {accuracy:.2f}%")

    overall_accuracy = total_correct / valid_queries if valid_queries > 0 else 0
    results["overall_accuracy"] = overall_accuracy

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nFinal accuracy across {valid_queries} images: {overall_accuracy:.2f}%")
    print(f"All results saved to {args.output_json}")


if __name__ == "__main__":
    main()