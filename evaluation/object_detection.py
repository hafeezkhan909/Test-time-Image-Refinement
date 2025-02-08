import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from config import DEVICE, MODEL_NAME

class ObjectDetector:
    def __init__(self):
        """Initializes OWL-ViT processor and model."""
        self.processor = OwlViTProcessor.from_pretrained(MODEL_NAME)
        self.model = OwlViTForObjectDetection.from_pretrained(MODEL_NAME).to(DEVICE).eval()

    def detect_objects(self, image, text_queries):
        """Runs object detection and returns detected objects, bounding boxes, and raw scores."""
        inputs = self.processor(images=image, text=[text_queries], return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]]).to(DEVICE)
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.05)

        detected_objects = []
        bounding_boxes = []
        scores = []

        for score, label, box in zip(results[0]["scores"], results[0]["labels"], results[0]["boxes"]):
            if score > 0.1:
                detected_objects.append(text_queries[label])  # Map label index to query text
                bounding_boxes.append(box.tolist())  # Convert tensor to list
                scores.append(float(score))  # Convert tensor to float
        
        # Print raw scores
        print(f"🔍 Detected Objects (Raw Scores): {scores}")
        return detected_objects, bounding_boxes, scores
