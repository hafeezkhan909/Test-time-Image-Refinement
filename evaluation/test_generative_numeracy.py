import json
import os
from object_detection import ObjectDetector
from image_utils import load_image
from visualization import save_detection_visualization

# Load test queries from JSON
with open("data/generative_numeracy/test_numeracy_queries.json", "r") as f:
    test_queries = json.load(f)  # Format: {"0": {"object": "car", "count": 1}}

# Define base path for images
base_image_path = "data/generative_numeracy/method2/"

# Initialize Object Detector
detector = ObjectDetector()

# Accuracy Tracking Variables
total_accuracy = 0
total_queries = len(test_queries)

# Results dictionary to save as JSON
results = {}

# Process each query
for query_id, query_data in test_queries.items():
    object_name = query_data["object"]
    expected_count = query_data["count"]

    image_path = os.path.join(base_image_path, query_id, f"img_{query_id}.png")

    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}, skipping...")
        continue

    # Load image
    image = load_image(image_path)

    # Run detection
    detected_objects, bounding_boxes, scores = detector.detect_objects(image, [object_name])

    # Count detected instances of the object
    detected_count = detected_objects.count(object_name)

    # Compute accuracy based on count difference
    if expected_count == 0 and detected_count == 0:
        accuracy = 100  # Perfect match for absence
    else:
        accuracy = max(0, 100 - abs(expected_count - detected_count) / expected_count * 100)

    total_accuracy += accuracy

    # Save visualization
    save_path = f"data/generative_numeracy/method2/detection_visualization/img_{query_id}_numeracy.png"
    save_detection_visualization(image, detected_objects, bounding_boxes, scores, save_path)

    # Store results
    results[query_id] = {
        "image_path": image_path,
        "object": object_name,
        "expected_count": expected_count,
        "detected_count": detected_count,
        "bounding_boxes": bounding_boxes,
        "confidence_scores": scores,
        "accuracy": accuracy,
        "visualization_path": save_path
    }

    # Print Results
    print(f"\n=== Image {query_id} Results ===")
    print(f"🔍 Expected: {expected_count} {object_name}(s)")
    print(f"🔍 Detected: {detected_count} {object_name}(s)")
    print(f"🖼 Bounding Boxes: {bounding_boxes}")
    print(f"📊 Confidence Scores: {scores}")
    print(f"✅ Accuracy: {accuracy:.2f}%")
    print(f"✅ Image with detections saved at: {save_path}")

# Compute overall accuracy
overall_accuracy = total_accuracy / total_queries if total_queries > 0 else 0
results["overall_accuracy"] = overall_accuracy

# Save results to JSON
with open("data/generative_numeracy/method2/method2_numeracy_results.json", "w") as json_file:
    json.dump(results, json_file, indent=4)

print(f"\n📊 **Final Accuracy Across All Images: {overall_accuracy:.2f}%**")
print(f"✅ All results saved to numeracy_results.json")
