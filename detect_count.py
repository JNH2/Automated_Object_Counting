"""
Automated Object Counting using YOLOv5
Author: JNH
Description: Detects and counts a specified object class in images using a pretrained YOLOv5 model.
"""

import os
import torch
import cv2

def detect_and_count(image_path, class_id=18, model_name='yolov5l', threshold=0.5):
    """
    Detect and count objects of a specified class in a single image.
    """
    # Load YOLOv5 pretrained model
    model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: cannot read image {image_path}")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform detection
    results = model(image)

    # Filter by class and confidence threshold
    mask = [(int(x[-1]) == class_id) and (x[-2] > threshold) for x in results.xyxy[0]]
    count = sum(mask)

    if count > 0:
        print(f"{count} objects of class {class_id} detected in {image_path}.")
    else:
        print(f"No objects of class {class_id} detected in {image_path}.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv5 Automated Object Counting")
    parser.add_argument("--image", type=str, required=True, help="Path to the image file")
    parser.add_argument("--class_id", type=int, default=18, help="YOLOv5 class index to count")
    parser.add_argument("--model", type=str, default="yolov5l", help="YOLOv5 model")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold")
    args = parser.parse_args()

    detect_and_count(args.image, class_id=args.class_id, model_name=args.model, threshold=args.threshold)
