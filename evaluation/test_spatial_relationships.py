import json
import os
import argparse
from object_detection import ObjectDetector
from image_utils import load_image
from visualization import save_detection_visualization
from image_selection import get_final_image_path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate spatial relationship prompts against generated images.")
    parser.add_argument("--query_json", type=str, default="json_files/position_specific.json",
                         help="Path to the position query JSON (keyed by line_number).")
    parser.add_argument("--images_dir", type=str, required=True,
                         help="Path to the pipeline's output_dir (contains <tag>/prompt_XXX/ subfolders).")
    parser.add_argument("--output_json", type=str, default="spatial_results.json",
                         help="Path to save the results JSON.")
    parser.add_argument("--visualization_dir", type=str, default="detection_visualization/position",
                         help="Directory to save detection visualization images.")
    parser.add_argument("--refinement_iterations", type=int, default=3,
                         help="Number of refinement rounds the pipeline ran (default: 3).")
    parser.add_argument("--last_restart_step", type=int, default=0,
                         help="Restart step used for the final refinement round (default: 0).")
    parser.add_argument("--selection_mode", type=str, default="first_early_stop",
                         choices=["first_early_stop", "final_only"],
                         help="How to pick the evaluated image per prompt (default: first_early_stop).")
    return parser.parse_args()


def analyze_spatial_relationships(detected_objects, bounding_boxes):
    """Determines spatial relationships between detected objects with a more lenient approach."""
    relationships = []
    bbox_dict = {obj: box for obj, box in zip(detected_objects, bounding_boxes)}

    for obj1, box1 in bbox_dict.items():
        for obj2, box2 in bbox_dict.items():
            if obj1 != obj2:
                x1_min, y1_min, x1_max, y1_max = box1
                x2_min, y2_min, x2_max, y2_max = box2

                if x1_max > x2_min:
                    relationships.append(f"{obj1} is to the right of {obj2}")
                if x1_min < x2_max:
                    relationships.append(f"{obj1} is to the left of {obj2}")
                if y1_min < y2_max:
                    relationships.append(f"{obj1} is above {obj2}")
                if y1_max > y2_min:
                    relationships.append(f"{obj1} is below {obj2}")

    return relationships


def main():
    args = parse_args()
    tag = "position"

    with open(args.query_json, "r") as f:
        test_queries = json.load(f)  # Format: {"201": {"objects": [...], "relationships": [...]}, ...}

    detector = ObjectDetector()

    total_correct = 0
    valid_queries = 0
    results = {}

    for line_number, query_data in test_queries.items():
        objects = query_data["objects"]
        expected_relationships = query_data["relationships"]

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

        detected_objects, bounding_boxes, scores = detector.detect_objects(image, objects)
        detected_relationships = analyze_spatial_relationships(detected_objects, bounding_boxes)

        correct_count = len(set(expected_relationships) & set(detected_relationships))
        accuracy = (correct_count / len(expected_relationships)) * 100 if expected_relationships else 0
        total_correct += accuracy

        save_path = os.path.join(args.visualization_dir, f"{prompt_id}_spatial.png")
        save_detection_visualization(image, detected_objects, bounding_boxes, scores, save_path)

        results[line_number] = {
            "image_path": image_path,
            "expected_objects": objects,
            "expected_relationships": expected_relationships,
            "detected_objects": detected_objects,
            "detected_relationships": detected_relationships,
            "bounding_boxes": bounding_boxes,
            "confidence_scores": scores,
            "accuracy": accuracy,
            "visualization_path": save_path
        }

        print(f"\n=== {prompt_id} Results ===")
        print(f"Expected relationships: {expected_relationships}")
        print(f"Detected relationships: {detected_relationships}")
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