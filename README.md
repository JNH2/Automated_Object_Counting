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


Counting Error Analysis
To evaluate counting performance, we define:
Counting Error = | Predicted Count − Ground Truth Count |

Evaluation metrics:
MAE (Mean Absolute Error)
RMSE (Root Mean Squared Error)

Observed error sources:
False positives due to background clutter
Missed detections under occlusion
Domain shift between custom datasets and COCO
Sensitivity to confidence threshold selection

Dataset
Custom image dataset with class-level annotations
Optional evaluation on FSC147-style counting benchmarks
Dataset curation emphasizes data quality and bias awareness

Ethical Considerations
Object counting systems may be repurposed for surveillance
Person detection raises privacy concerns

Mitigations:
Avoid storing raw images
Apply anonymization if people appear
Be transparent about system limitations

Tech Stack
Python
PyTorch
YOLOv5
OpenCV
NumPy

Future Work
Integrate IoU-based loss functions for improved localization
Extend to video-based counting with temporal consistency
Compare detection-based counting with density-based methods
