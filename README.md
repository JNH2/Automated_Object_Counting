Overview
This project implements an automated object counting pipeline based on YOLOv5, focusing on practical data pipelines, batch inference, and quantitative error analysis rather than model novelty.
The goal is to demonstrate how a pre-trained object detection model can be integrated into a robust counting system, addressing real-world issues such as confidence thresholding, false positives, and dataset bias.

Pipeline Architecture
1. Input
  Image datasets (custom or public)
  User-specified target class (e.g., car, sheep)
2. Preprocessing
  Image loading and RGB conversion
  Dataset-level filtering and normalization
3. Object Detection
  YOLOv5 (pretrained on COCO)
  Bounding box prediction with confidence scores
4. Post-processing
  Confidence thresholding
  Class-based filtering
  Non-Maximum Suppression (NMS)
5. Counting
   Aggregate valid detections per image
  Output per-image and dataset-level counts
