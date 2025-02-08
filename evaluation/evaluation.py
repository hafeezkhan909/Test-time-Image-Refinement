import json
import os
from image_utils import load_image
from object_detection import ObjectDetector
from spatial_analysis import SpatialAnalyzer
from visualization import save_detection_visualization

class Evaluator:
    def __init__(self):
        self.detector = ObjectDetector()
        self.spatial_analyzer = SpatialAnalyzer()

    def evaluate_single_image(self, image_path, prompt, expected_relationships, save_dir="output_visualizations/"):
        """Evaluates a single image for object detection and spatial accuracy."""
        image = load_image(image_path)
        text_queries = ["red apple", "banana", "table", "fruit"]  # Modify based on use case

        # Run detection
        detected_objects, bounding_boxes, scores = self.detector.detect_objects(image, text_queries)

        # Compute object detection accuracy
        correct_count = sum(1 for obj in text_queries if obj in detected_objects)
        object_accuracy = (correct_count / len(text_queries)) * 100 if text_queries else 0

        # Compute spatial alignment accuracy
        spatial_accuracy, detected_relationships = self.spatial_analyzer.analyze(
            detected_objects, bounding_boxes, expected_relationships
        )

        # Save visualization
        save_path = os.path.join(save_dir, os.path.basename(image_path).replace(".jpg", "_detected.png"))
        save_detection_visualization(image, detected_objects, bounding_boxes, scores, save_path)

        # Print results
        print(f"✅ Expected Objects: {text_queries}")
        print(f"🔍 Detected Objects: {detected_objects}")
        print(f"📊 Object Detection Accuracy: {object_accuracy:.2f}%")
        print(f"✅ Expected Relationships: {expected_relationships}")
        print(f"🔍 Detected Relationships: {detected_relationships}")
        print(f"📊 Spatial Relationship Accuracy: {spatial_accuracy:.2f}%\n")

        return object_accuracy, spatial_accuracy

    def evaluate_dataset(self, image_folder, prompt_file, spatial_relationships_file):
        """Evaluates multiple images using JSON files."""
        with open(prompt_file, "r") as f:
            prompts = json.load(f)

        with open(spatial_relationships_file, "r") as f:
            spatial_data = json.load(f)

        total_object_accuracy = 0
        total_spatial_accuracy = 0
        total_images = len(prompts)

        for image_name, prompt in prompts.items():
            image_path = os.path.join(image_folder, image_name)
            expected_relationships = spatial_data.get(image_name, [])

            object_acc, spatial_acc = self.evaluate_single_image(image_path, prompt, expected_relationships)
            total_object_accuracy += object_acc
            total_spatial_accuracy += spatial_acc

        avg_object_accuracy = total_object_accuracy / total_images if total_images > 0 else 0
        avg_spatial_accuracy = total_spatial_accuracy / total_images if total_images > 0 else 0

        print(f"📊 **Dataset Object Detection Accuracy: {avg_object_accuracy:.2f}%**")
        print(f"📊 **Dataset Spatial Relationship Accuracy: {avg_spatial_accuracy:.2f}%**")

        return avg_object_accuracy, avg_spatial_accuracy
