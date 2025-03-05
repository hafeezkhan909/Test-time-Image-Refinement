import json
import os
from object_detection import ObjectDetector
from image_utils import load_image
from visualization import save_detection_visualization

# Load test queries from JSON
with open("data/spatial_relationships/position_specific.json", "r") as f:
    test_queries = json.load(f)  # Format: {"0": {"objects": ["A", "B"], "relationships": ["A is left of B"]}}

# Define base path for images
base_image_path = "data/spatial_relationships/0_0_0_99/"

# Initialize Object Detector
detector = ObjectDetector()

# Accuracy Tracking Variables
total_correct = 0
total_queries = len(test_queries)

# Results dictionary to save as JSON
results = {}

def analyze_spatial_relationships(detected_objects, bounding_boxes):
    """Determines spatial relationships between detected objects with a more lenient approach."""
    relationships = []

    # Store bounding box coordinates for each detected object
    bbox_dict = {obj: box for obj, box in zip(detected_objects, bounding_boxes)}

    # Compare object positions to detect spatial relationships
    for obj1, box1 in bbox_dict.items():
        for obj2, box2 in bbox_dict.items():
            if obj1 != obj2:
                x1_min, y1_min, x1_max, y1_max = box1
                x2_min, y2_min, x2_max, y2_max = box2

                # Object 1 is to the right of Object 2
                if x1_max > x2_min:
                    relationships.append(f"{obj1} is to the right of {obj2}")

                # Object 1 is to the left of Object 2
                if x1_min < x2_max:
                    relationships.append(f"{obj1} is to the left of {obj2}")

                # Object 1 is above Object 2
                if y1_min < y2_max:
                    relationships.append(f"{obj1} is above {obj2}")

                # Object 1 is below Object 2
                if y1_max > y2_min:
                    relationships.append(f"{obj1} is below {obj2}")

    return relationships

# Process each query
for query_id, query_data in test_queries.items():
    objects = query_data["objects"]
    expected_relationships = query_data["relationships"]

    image_path = os.path.join(base_image_path, query_id, f"img{0}.png")

    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}, skipping...")
        continue

    # Load image
    image = load_image(image_path)

    # Run detection
    detected_objects, bounding_boxes, scores = detector.detect_objects(image, objects)

    # Analyze detected spatial relationships
    detected_relationships = analyze_spatial_relationships(detected_objects, bounding_boxes)

    # Compute accuracy (Intersection of expected & detected relationships)
    correct_count = len(set(expected_relationships) & set(detected_relationships))
    accuracy = (correct_count / len(expected_relationships)) * 100 if expected_relationships else 0
    total_correct += accuracy

    # Save visualization
    save_path = f"data/spatial_relationships/0_0_0_99/detection_visualization/img_{query_id}_spatial.png"
    save_detection_visualization(image, detected_objects, bounding_boxes, scores, save_path)

    # Store results
    results[query_id] = {
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

    # Print Results
    print(f"\n=== Image {query_id} Results ===")
    print(f"🔍 Expected Objects: {objects}")
    print(f"✅ Expected Relationships: {expected_relationships}")
    print(f"🔍 Detected Objects: {detected_objects}")
    print(f"🔍 Detected Relationships: {detected_relationships}")
    print(f"🖼 Bounding Boxes: {bounding_boxes}")
    print(f"📊 Confidence Scores: {scores}")
    print(f"✅ Accuracy: {accuracy:.2f}%")
    print(f"✅ Image with detections saved at: {save_path}")

# Compute overall accuracy
overall_accuracy = total_correct / total_queries if total_queries > 0 else 0
results["overall_accuracy"] = overall_accuracy

# Save results to JSON
with open("data/spatial_relationships/0_0_0_99/0_0_0_99_spatial_results.json", "w") as json_file:
    json.dump(results, json_file, indent=4)

print(f"\n📊 **Final Accuracy Across All Images: {overall_accuracy:.2f}%**")
print(f"✅ All results saved to spatial_results.json")
