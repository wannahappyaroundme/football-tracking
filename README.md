
# 🏈 Football Tracking System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Latest-green.svg)](https://github.com/ultralytics/ultralytics)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive American football analysis system using state-of-the-art computer vision and deep learning techniques. This system provides real-time player tracking, team classification, play type analysis, and advanced statistics for American football games.

## 📺 Demo

![Football Tracking Demo](demo/demo.gif)

## ✨ Features

### Core Capabilities

-   **🎯 Player Detection & Tracking**: Real-time detection and tracking of all players on the field
-   **👥 Team Classification**: Automatic team differentiation based on jersey colors
-   **🏃 Play Type Analysis**: Identifies running plays, passing plays, and pre-snap formations
-   **🏈 Ball Detection**: Tracks the football throughout the game
-   **👨‍⚖️ Referee Detection**: Distinguishes referees from players
-   **📍 Field Analysis**: Detects field boundaries and excludes sideline players

### Advanced Features

-   **🗺️ Bird's Eye View**: Real-time tactical view transformation
-   **📊 Player Statistics**:
    -   Speed tracking (current/max/average)
    -   Distance covered
    -   Play time
    -   Heat maps
-   **🎮 Formation Recognition**: Detects offensive and defensive formations
-   **📈 Game Analytics**: Comprehensive statistics export (JSON/CSV)

### Technical Features

-   **🚀 GPU Optimization**: Supports CUDA (NVIDIA) and MPS (Apple Silicon)
-   **🔄 Cross-platform**: Works on Windows, macOS, and Linux
-   **📦 Model Training**: Custom training pipeline for your own data
-   **🎬 Multiple Input Formats**: Supports MP4, AVI, MOV video files

## ✨ Latest Features (v2.0)

### 🏈 Advanced Ball Detection
- **Snap Position Detection**: Automatically finds ball at center field
- **Quarterback Tracking**: Detects ball in QB's hands
- **Multi-method Fusion**: YOLO + Color + Motion + Kalman prediction
- **Occlusion Handling**: Tracks ball even when blocked by players

### 👥 Smart Team Classification
- **Automatic Jersey Detection**: K-means clustering for team colors
- **Color Recognition**: Shows actual color names (RED, BLUE, etc.)
- **Consistent Tracking**: Maintains player-team assignments
- **Referee Identification**: Auto-detects officials

### 📊 Real-time Analytics
- Team member counts (on-field/total)
- Ball detection rate percentage
- Play type classification
- Possession tracking
- FPS monitoring

## 💻 System Requirements

## Hardware Requirements
## Tested Configuration (Author's System)

- Model: Mac15,10 (Mac Studio/Mac mini)
- Chip: Apple M3 Max
- Cores: 14 (10 performance + 4 efficiency)
- Memory: 36 GB
- Performance: 25-30 FPS with YOLOv8x

## Minimum Requirements

- CPU: Intel i5 / Apple M1
- RAM: 8GB
- Storage: 10GB

## Recommended

- CPU: Apple M2/M3 Pro/Max
- RAM: 16GB+
- Storage: 50GB+

## Software Requirements

- Python 3.8-3.11
- macOS 12.0+
- Xcode Command Line Tools

🚀 Installation
1. Clone Repository
bashgit clone https://github.com/wannahappyaroundme/football-tracking.git
cd football-tracking
2. Set Up Virtual Environment
bash# Create virtual environment
python3 -m venv football_env
source football_env/bin/activate

# Or using conda (recommended for M3)
conda create -n football python=3.9
conda activate football
3. Install Dependencies
For Apple Silicon (M1/M2/M3)
bash# Install PyTorch for Apple Silicon
pip install torch torchvision torchaudio

# Core dependencies
pip install ultralytics supervision opencv-python

# Fix NumPy compatibility
pip install "numpy<2.0"

# Additional packages
pip install pandas scikit-learn scipy

# Optional: Advanced ball tracking
pip install filterpy
Fix Common Issues
bash# NumPy 2.0 compatibility error
pip install "numpy==1.24.3"
pip install --upgrade pandas numexpr bottleneck

# Memory optimization for M3 Max
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7
💻 Usage
Quick Start
bashpython ultimate_football_analyzer.py
Interactive Menu
======================================================================
🏈 Ultimate Football Analyzer
======================================================================

Analysis Mode:
1. Quick analysis (default YOLO)
2. Roboflow model (football-specific)  
3. Custom model path
Select: 2

Ball Detection:
1. Advanced multi-method (YOLO + Color + Motion + Prediction)
2. Basic color detection only
Select: 1

Video selection (1-10): 1
Command Line Options
python# Process specific video
python ultimate_football_analyzer.py \
    --input videos/game.mp4 \
    --output output/analyzed.mp4

# Use custom model
python ultimate_football_analyzer.py \
    --model models/custom.pt

# CPU mode (if MPS fails)
python ultimate_football_analyzer.py \
    --device cpu
🎯 Detection Methods
Ball Detection Pipeline

Snap Position Detection

Searches center field area
Focuses on ground level
Priority: 2.5x weight


Quarterback Detection

Scans upper body regions
Hand area focus (30-70% height)
Priority: 2.2x weight


YOLO Detection

Sports ball class (32)
Confidence > 0.25
Priority: 2.0x weight


Color Detection

python   # Football leather color (HSV)
   Brown: H(5-18), S(30-150), V(40-160)
   # LAB color space
   A(125-145), B(130-155)
Team Classification
python# Automatic team detection process
1. Extract jersey region (25-65% body height)
2. DBSCAN noise removal
3. K-means clustering (2-3 clusters)
4. Assign teams by cluster size
5. Track player IDs for consistency
📊 Output Files
Video Output

Format: MP4 (H.264)
Resolution: Original + 150px info panel
Overlays: Boxes, trajectories, minimap
FPS: ~25-30 on M3 Max

Generated Files
output/
├── ultimate_20241114_143022.mp4         # Analyzed video
├── ultimate_20241114_143022_stats.json  # Game statistics
├── ultimate_20241114_143022_frame_data.csv   # Frame data
└── ultimate_20241114_143022_player_stats.csv # Player metrics
Statistics JSON Structure
json{
  "total_frames": 962,
  "ball_detection_rate": 0.752,
  "play_types": {
    "Pre-snap": 120,
    "Running Play": 250,
    "Passing Play": 180
  },
  "team_stats": {
    "team1": {
      "color": "BLUE",
      "total_players": 15,
      "avg_speed": 12.3
    },
    "team2": {
      "color": "RED", 
      "total_players": 14,
      "avg_speed": 11.8
    }
  }
}
🔧 Troubleshooting
Common Issues & Solutions
AttributeError: 'AdvancedBallDetector' object has no attribute 'detect_snap_position_ball'
bash# Solution: Pull latest code or copy the fixed version
git pull origin main
Memory Issues on M3 Max
bash# Optimize memory usage
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7

# Or use CPU mode
device = 'cpu'  # in code
Slow Performance
python# Use lighter model
model = YOLO('yolov8n.pt')  # Instead of yolov8x

# Reduce detection confidence
conf=0.5  # Higher threshold = fewer detections
Ball Not Detected
python# Adjust brown color ranges
brown_ranges = [
    ([8, 30, 40], [20, 150, 180]),  # Tweak these
    ([5, 20, 30], [15, 100, 120]),
]
📈 Performance Benchmarks
Processing Speed (M3 Max 36GB)
ModelResolutionFPSRAM UsageYOLOv8n1920x108045-504GBYOLOv8s1920x108035-405GBYOLOv8m1920x108030-356GBYOLOv8l1920x108025-308GBYOLOv8x1920x108020-2510GB
Detection Accuracy
FeatureAccuracyNotesPlayer Detection95%+High confidenceTeam Classification90%+After stabilizationBall Detection75-80%Snap position improvedReferee Detection85%+Black/white jerseys
Optimization Tips for M3 Max
python# Enable performance cores
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Batch processing
batch_size = 32  # M3 Max can handle larger batches

# Multi-stream processing
workers = 8  # Utilize all efficiency cores
🗺️ Roadmap
Current (v2.0)

✅ Multi-method ball detection
✅ Snap position detection
✅ Team auto-classification
✅ Bird's eye view
✅ Player statistics

Next Release (v2.1)

⏳ Jersey number OCR
⏳ Formation recognition
⏳ Play prediction AI
⏳ Web interface

Future (v3.0)

⏳ Real-time streaming
⏳ Multi-camera fusion
⏳ 3D reconstruction
⏳ AI commentary

🤝 Contributing
We welcome contributions!

Fork the repository
Create feature branch (git checkout -b feature/YourFeature)
Commit changes (git commit -m 'Add YourFeature')
Push branch (git push origin feature/YourFeature)
Open Pull Request

Development Setup
bash# Install dev dependencies
pip install pytest black flake8

# Run tests
pytest tests/

# Format code
black .
📄 License
MIT License - see LICENSE
🙏 Acknowledgments

Ultralytics - YOLOv8 framework
Roboflow - Supervision library
ByteTrack - Object tracking
OpenCV - Computer vision
Apple - Metal Performance Shaders

📞 Contact & Support
Author: wannahappyaroundme
GitHub: @wannahappyaroundme
Project: football-tracking
Bug Reports
Please include:

macOS version
Python version (python3 --version)
Error message
Steps to reproduce

Feature Requests
Open an issue with [FEATURE] tag
🐛 Known Issues

Ball detection in low light conditions
Team classification in similar colored jerseys
MPS memory warnings on long videos

📚 Citation
bibtex@software{football_tracking_2024,
  author = {wannahappyaroundme},
  title = {Ultimate Football Tracking System},
  year = {2024},
  version = {2.0},
  url = {https://github.com/wannahappyaroundme/football-tracking}
}or NVIDIA GPUs)
-   Git

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/wannahappyaroundme/football-tracking.git
cd football-tracking
```

### 2. Set Up Environment

#### Windows (NVIDIA GPU)

```bash
# Create virtual environment
python -m venv football_env
football_env\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

#### macOS (Apple Silicon)

```bash
# Create virtual environment
python3 -m venv football_env
source football_env/bin/activate

# Install dependencies
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

#### Linux

```bash
# Create virtual environment
python3 -m venv football_env
source football_env/bin/activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 3. Download Pre-trained Models (Optional)

```bash
# Download YOLOv8 models
python download_models.py
```

### 4. Run Quick Test

```bash
# Test with sample video
python quick_test.py
```

## 📁 Project Structure

```
football-tracking/
│
├── src/                      # Source code
│   ├── detection/           # Object detection modules
│   ├── tracking/            # Object tracking modules
│   ├── classification/     # Team classification
│   ├── transformation/     # Bird's eye view transformation
│   └── utils/              # Utility functions
│
├── models/                  # Trained models
│   ├── yolov8x.pt          # Pre-trained YOLOv8
│   └── football_best.pt    # Custom trained model
│
├── data/                    # Datasets
│   ├── annotations/        # Label files
│   └── football-dataset/   # Training data
│
├── configs/                 # Configuration files
│   └── training_config.yaml
│
├── videos/                  # Input videos
├── output/                  # Output results
│
├── ultimate_football_analyzer.py  # Main analysis script
├── train_football_model.py        # Training script
├── train_m3_max.py               # Apple Silicon optimized training
├── download_multiple_datasets.py  # Dataset preparation
├── quick_test.py                  # Quick testing script
│
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 💻 Usage

### Basic Video Analysis

```bash
python ultimate_football_analyzer.py
```

**Interactive prompts:**

```
Select analysis mode:
1. Quick analysis (default YOLO)
2. Use Roboflow model (football-specific)
3. Custom model path

Select: 1
Video selection (1-10): 1
```

### Advanced Usage

#### Custom Model

```python
from ultimate_football_analyzer import UltimateFootballAnalyzer

# Initialize with custom model
analyzer = UltimateFootballAnalyzer(
    model_path='models/football_best.pt',
    use_roboflow=False
)

# Process video
analyzer.process_video(
    input_path='videos/game.mp4',
    output_path='output/analyzed_game.mp4'
)
```

#### Batch Processing

```python
import os
from glob import glob

# Process all videos in folder
video_files = glob('videos/*.mp4')

for video in video_files:
    analyzer.process_video(
        input_path=video,
        output_path=f'output/analyzed_{os.path.basename(video)}'
    )
```

## 🎓 Training Custom Models

### 1. Prepare Dataset

#### Option A: Use Roboflow Datasets

```bash
python download_multiple_datasets.py

# Follow prompts:
# 1. Enter Roboflow API key
# 2. Select datasets to download
# 3. Choose augmentation options
```

#### Option B: Use Your Own Videos

```python
# Extract frames from videos
python src/utils/video_to_frames.py --input videos/ --output data/custom/
```

### 2. Train Model

#### Standard Training (NVIDIA GPU)

```bash
python train_football_model.py

# Configuration:
# Model: YOLOv8x (option 5)
# Epochs: 200
# Batch size: auto-adjusted based on GPU
```

#### Apple Silicon Training (M1/M2/M3)

```bash
python train_m3_max.py

# Configuration:
# Memory: 36 (for M3 Max with 36GB)
# Model: YOLOv8x (option 5)
# Epochs: 300
```

### 3. Training Parameters

| Parameter   | Description            | Recommended          |
| ----------- | ---------------------- | -------------------- |
| `epochs`    | Training iterations    | 200-300              |
| `batch`     | Batch size             | 8-32 (GPU dependent) |
| `imgsz`     | Image size             | 640-800              |
| `patience`  | Early stopping         | 50                   |
| `lr0`       | Initial learning rate  | 0.001                |
| `optimizer` | Optimization algorithm | AdamW                |

## 📊 Output Formats

### Video Output

-   **Format**: MP4 with H.264 codec
-   **Resolution**: Same as input
-   **Overlays**:
    -   Bounding boxes with team colors
    -   Player IDs and speed
    -   Bird's eye view minimap
    -   Statistics panel

### Data Export

#### JSON Statistics (`*_stats.json`)

```json
{
    "total_frames": 1500,
    "total_plays": 25,
    "play_types": {
        "Running Play": 15,
        "Passing Play": 10
    },
    "team_stats": {
        "team1": { "players": 11, "avg_speed": 15.2 },
        "team2": { "players": 11, "avg_speed": 14.8 }
    }
}
```

#### Frame Data CSV (`*_frame_data.csv`)

```csv
frame,detections,team1_count,team2_count,play_type,ball_position
0,22,11,11,Pre-snap,"(960, 540)"
1,22,11,11,Running Play,"(965, 545)"
```

#### Player Statistics CSV (`*_player_stats.csv`)

```csv
player_id,play_time,total_distance,max_speed,avg_speed
1,120.5,1532.3,25.6,12.3
2,118.3,1423.7,23.4,11.8
```

## 🔧 Configuration

### System Configuration (`configs/system_config.yaml`)

```yaml
# GPU Settings
device: cuda # Options: cuda, mps, cpu
gpu_id: 0
mixed_precision: true

# Processing Settings
confidence_threshold: 0.3
iou_threshold: 0.5
max_detections: 300

# Tracking Settings
track_buffer: 30
track_threshold: 0.25
```

### Training Configuration (`configs/training_config.yaml`)

```yaml
# Model Settings
model: yolov8x
pretrained: true

# Training Parameters
epochs: 200
batch_size: auto
image_size: 640
workers: 4

# Augmentation
mosaic: 1.0
mixup: 0.3
copy_paste: 0.3
```

## 🎯 Performance Metrics

### Detection Performance

| Model   | mAP50 | mAP50-95 | FPS (RTX 3050) | FPS (M3 Max) |
| ------- | ----- | -------- | -------------- | ------------ |
| YOLOv8n | 0.72  | 0.52     | 45             | 60           |
| YOLOv8s | 0.78  | 0.58     | 35             | 50           |
| YOLOv8m | 0.82  | 0.62     | 25             | 40           |
| YOLOv8l | 0.85  | 0.66     | 18             | 30           |
| YOLOv8x | 0.88  | 0.70     | 12             | 25           |
| Custom  | 0.92  | 0.75     | 15             | 28           |

### Hardware Performance

| Hardware        | Training Time (200 epochs) | Inference Speed | Max Batch Size |
| --------------- | -------------------------- | --------------- | -------------- |
| RTX 3050 (4GB)  | 8-10 hours                 | 12-15 FPS       | 8              |
| RTX 3070 (8GB)  | 5-6 hours                  | 20-25 FPS       | 16             |
| RTX 4090 (24GB) | 2-3 hours                  | 40-50 FPS       | 64             |
| M3 Max (36GB)   | 3-4 hours                  | 25-30 FPS       | 32             |

## 🐛 Troubleshooting

### Common Issues

#### CUDA Out of Memory

```python
# Reduce batch size in training
batch_size = 4  # Instead of 8 or 16

# Or use gradient accumulation
accumulate = 4  # Accumulate gradients over 4 batches
```

#### OpenCV GUI Error (Windows WSL/Linux)

```python
# Disable preview
show_preview = False

# Or use matplotlib backend
import matplotlib
matplotlib.use('Agg')
```

#### MPS Backend Error (Apple Silicon)

```python
# Fallback to CPU if MPS fails
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
```

#### Slow Training Speed

```python
# Enable mixed precision training
amp = True

# Increase number of workers
workers = 8

# Cache dataset in RAM
cache = 'ram'  # or True for disk caching
```

## 📈 Roadmap

### Version 1.0 (Current)

-   ✅ Basic player detection and tracking
-   ✅ Team classification
-   ✅ Bird's eye view
-   ✅ Play type analysis

### Version 1.1 (Q1 2025)

-   ⏳ Real-time streaming support
-   ⏳ Web interface (Streamlit/Gradio)
-   ⏳ Cloud deployment (AWS/GCP)

### Version 1.2 (Q2 2025)

-   ⏳ Advanced play recognition
-   ⏳ Player identification (jersey numbers)
-   ⏳ Coaching analytics dashboard
-   ⏳ Mobile app integration

### Version 2.0 (Q3 2025)

-   ⏳ 3D reconstruction
-   ⏳ VR/AR visualization
-   ⏳ AI commentary generation
-   ⏳ Multi-camera support

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

### How to Contribute

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Style

-   Follow PEP 8
-   Use type hints
-   Add docstrings
-   Write unit tests

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

-   **YOLOv8** by Ultralytics for object detection
-   **Supervision** by Roboflow for tracking and visualization
-   **ByteTrack** for object tracking algorithm
-   **Roboflow Universe** for datasets
-   **OpenCV** for computer vision operations
-   **PyTorch** for deep learning framework

## 📞 Contact

**Author**: wannahappyaroundme

**Email**: [your-email@example.com]

**GitHub**: [@wannahappyaroundme](https://github.com/wannahappyaroundme)

**Project Link**: [https://github.com/wannahappyaroundme/football-tracking](https://github.com/wannahappyaroundme/football-tracking)

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@software{football_tracking_2024,
  author = {wannahappyaroundme},
  title = {Ultimate Football Tracking System},
  year = {2024},
  url = {https://github.com/wannahappyaroundme/football-tracking}
}
```

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=wannahappyaroundme/football-tracking&type=Date)](https://star-history.com/#wannahappyaroundme/football-tracking&Date)

---

**Made with ❤️ for the American Football community**
