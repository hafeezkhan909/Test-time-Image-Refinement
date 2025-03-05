import json
import os
from object_detection import ObjectDetector
from image_utils import load_image
from visualization import save_detection_visualization

# Load test queries from JSON
with open("data/negation/negation_specific.json", "r") as f:
    test_queries = json.load(f)  # Format: {"0": {"object": "aeroplane"}}

# Define base path for images
base_image_path = "data/negation/b_lmd+/"

# Initialize Object Detector
detector = ObjectDetector()

# Accuracy Tracking Variables
total_correct = 0
valid_queries = 0  # Count only images that exist

# Results dictionary to save as JSON
results = {}

# Process each query
for query_id, query_data in test_queries.items():
    forbidden_object = query_data["object"]

    image_path = os.path.join(base_image_path, query_id, "img_0.png")

    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}, skipping...")
        continue  # Skip missing images completely from calculations

    valid_queries += 1  # Only count queries with existing images

    # Load image
    image = load_image(image_path)

    # Run detection
    detected_objects, bounding_boxes, scores = detector.detect_objects(image, [forbidden_object])

    # Compute accuracy
    accuracy = 100 if forbidden_object not in detected_objects else 0  # Object absent = perfect score

    total_correct += accuracy

    # Save visualization (Even though ideally, no bounding boxes should be present)
    save_path = f"data/negation/b_lmd+/detection_visualization/img_{query_id}_negation.png"
    save_detection_visualization(image, detected_objects, bounding_boxes, scores, save_path)

    # Store results
    results[query_id] = {
        "image_path": image_path,
        "forbidden_object": forbidden_object,
        "detected_objects": detected_objects,
        "bounding_boxes": bounding_boxes,
        "confidence_scores": scores,
        "accuracy": accuracy,
        "visualization_path": save_path
    }

    # Print Results
    print(f"\n=== Image {query_id} Results ===")
    print(f"🚫 Expected Absence of: {forbidden_object}")
    print(f"🔍 Detected Objects: {detected_objects}")
    print(f"🖼 Bounding Boxes: {bounding_boxes}")
    print(f"📊 Confidence Scores: {scores}")
    print(f"✅ Accuracy: {accuracy:.2f}%")
    print(f"✅ Image with detections saved at: {save_path}")

# Compute overall accuracy only over valid queries (images that exist)
overall_accuracy = total_correct / valid_queries if valid_queries > 0 else 0
results["overall_accuracy"] = overall_accuracy

# Save results to JSON
output_file = "data/negation/b_lmd+/b_lmd+_negation_results.json"
with open(output_file, "w") as json_file:
    json.dump(results, json_file, indent=4)

print(f"\n📊 **Final Accuracy Across {valid_queries} Images: {overall_accuracy:.2f}%**")
print(f"✅ All results saved to {output_file}")
