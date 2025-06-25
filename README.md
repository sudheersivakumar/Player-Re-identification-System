# Player Re-identification System 
## This README provides complete instructions for setting up, running, and understanding the outputs of the Player Re-identification system implemented in cldvid.py. The system detects, tracks, and matches players between two videos using computer vision and deep learning.
## OUTPUT FILE & CODE
[Drive Link](https://drive.google.com/drive/folders/1PYGJ2zGAnBgGWWZD-Imuigu0hHQX4lIE?usp=sharing)
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
<pre> pip install opencv-python numpy ultralytics torch torchvision scikit-learn scipy </pre>
## Setup Instructions
### Clone or Download the Repository
- Place the cldvid.py file in your working directory.
### Prepare Model and Video Files
- Download or train a YOLO model (e.g., best.pt).
  - Place it at:
    - C:/Users/sudhe/OneDrive/Documents/liat_work/best.pt
- Place your two video files at:
   - C:/Users/sudhe/OneDrive/Documents/liat_work/broadcast.mp4
   - C:/Users/sudhe/OneDrive/Documents/liat_work/tacticam.mp4
- Update paths in cldvid.py if your files are located elsewhere.
- (Optional) Check CUDA Availability
   - The code will automatically use GPU if available; otherwise, it defaults to CPU.
## How to Run the Code
### Run the script from the command line:
<pre> python cldvid.py </pre>
### What the script does:
 - Loads the YOLO model
 - Processes both videos frame by frame
 - Detects and tracks players in each frame
 - Extracts appearance features for each player
 - Matches players between the two videos
 - Saves annotated videos, sample frames, and mapping reports to a timestamped output directory
## Output Files and Directories
- After completion, outputs are saved in a directory named like reid_test_results_YYYYMMDD_HHMMSS/. This directory contains:
  | File/Folder | Description |
  |----------|----------|
  | video1_tracked.mp4   | Video 1 with tracked player boxes and IDs    |
  | video2_tracked.mp4    | Video 2 with tracked player boxes and IDs    |
  | video1_frames/    | Sample frames from Video 1 (every 30 frames)    |
  | video2_frames/    | Sample frames from Video 2 (every 30 frames)    |
  | comparison_frames/    | Side-by-side comparison frames    |
  | player_mappings.json    | Player ID mappings between videos with similarity scores    |
  | player_mappings.pkl    | Pickled version of the mappings    |
  | reid_report.txt    | Human-readable mapping and statistics report    |
  | statistics.json    | JSON summary of detection and mapping statistics    |
  
## Customization and Advanced Usage
- Change detection threshold:
Adjust the conf parameter in YOLO inference for stricter or looser detection.
- Change output directory:
Edit the OUTPUT_DIR variable in the main() function.
- Process more frames:
Remove or adjust the frame_count > 1000 condition to process all frames.
## Troubleshooting
- FileNotFoundError:
Make sure the model and video paths are correct and files exist.
- No player mappings found:
Check detection confidence, video quality, and ensure players are visible.
- Slow performance:
Use a machine with a CUDA-capable GPU; otherwise, processing will be slow on CPU.
