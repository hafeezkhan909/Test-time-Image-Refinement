import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def save_detection_visualization(image, detected_objects, bounding_boxes, scores, save_path):
    """Saves an image with bounding boxes and labels drawn using Matplotlib."""
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(image)

    for (box, label, score) in zip(bounding_boxes, detected_objects, scores):
        xmin, ymin, xmax, ymax = box
        width, height = xmax - xmin, ymax - ymin
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

        ax.text(xmin, ymin - 5, f"{label} ({score:.2f})", color="red", fontsize=12, bbox=dict(facecolor="white", alpha=0.75))

    plt.axis("off")

    # Ensure save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Image with detections saved at: {save_path}")