class SpatialAnalyzer:
    def analyze(self, detected_objects, bounding_boxes, expected_relationships):
        """Determines spatial relationships between detected objects."""
        detected_relationships = []
        centers = {}

        for obj, box in zip(detected_objects, bounding_boxes):
            xmin, ymin, xmax, ymax = box
            centers[obj] = ((xmin + xmax) / 2, (ymin + ymax) / 2)

        for obj1, center1 in centers.items():
            for obj2, center2 in centers.items():
                if obj1 != obj2:
                    x1, y1 = center1
                    x2, y2 = center2

                    if x1 > x2:
                        detected_relationships.append(f"{obj1} is to the right of {obj2}")
                    elif x1 < x2:
                        detected_relationships.append(f"{obj1} is to the left of {obj2}")

                    if y1 > y2:
                        detected_relationships.append(f"{obj1} is below {obj2}")
                    elif y1 < y2:
                        detected_relationships.append(f"{obj1} is above {obj2}")

        correct_count = sum(1 for rel in expected_relationships if rel in detected_relationships)
        total_count = len(expected_relationships)
        spatial_accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0

        return spatial_accuracy, detected_relationships
