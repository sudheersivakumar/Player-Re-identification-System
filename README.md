# Player Re-identification System 
## This README provides complete instructions for setting up, running, and understanding the outputs of the Player Re-identification system implemented in cldvid.py. The system detects, tracks, and matches players between two videos using computer vision and deep learning.
## Overview
### This project implements a video-based player re-identification (Re-ID) pipeline:
- Detection: Uses YOLO for player detection in each frame.
- Tracking: Tracks detected players within each video using appearance and spatial features.
- Feature Extraction: Extracts color histogram features for robust player appearance representation.
- Re-Identification: Matches players across two videos using cosine similarity and the Hungarian algorithm.
- Visualization & Output: Generates annotated videos, comparison frames, and mapping reports.
## Dependencies and Environment Requirements
- Python: 3.8 or above recommended
- CUDA: Required for GPU acceleration (if available)
- Required Python Packages:
  - opencv-python
  - numpy
  - ultralytics (for YOLO model)
  - torch and torchvision
  - scikit-learn
  - scipy
  - pickle
  - json
  - logging
## Install dependencies:
<pre> ```sh curl -fsSL https://christitus.com/linux | sh ``` </pre> 
