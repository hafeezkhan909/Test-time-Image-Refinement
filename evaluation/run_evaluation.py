from evaluation import Evaluator

if __name__ == "__main__":
    evaluator = Evaluator()

    # Evaluate a single image
    object_acc, spatial_acc = evaluator.evaluate_single_image(
        "test.jpg", 
        "A red apple on the right of a banana.",
        ["red apple is to the right of banana"]
    )
    print(f"📊 Object Accuracy: {object_acc:.2f}%")
    print(f"📊 Spatial Accuracy: {spatial_acc:.2f}%")

    # Evaluate dataset
    avg_object_accuracy, avg_spatial_accuracy = evaluator.evaluate_dataset(
        "generated_images/", "prompts.json", "spatial_relationships.json"
    )
    print(f"📊 Dataset Object Detection Accuracy: {avg_object_accuracy:.2f}%")
    print(f"📊 Dataset Spatial Relationship Accuracy: {avg_spatial_accuracy:.2f}%")
