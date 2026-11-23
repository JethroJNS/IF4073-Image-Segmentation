# IF4073 - Image Segmentation Application

An integrated image segmentation platform developed for **IF4073 Image Processing** (Semester 7, Institut Teknologi Bandung). The system provides a unified MATLAB programmatic graphical interface and command-line pipelines for three complementary segmentation paradigms:

1. **Edge Detection-based Segmentation** (Classical gradient and second-derivative operators with morphological boundary closure and connected component analysis).
2. **Split and Merge Segmentation** (Quadtree-based region splitting and adjacent region merging with color and homogeneity thresholds).
3. **Deep Learning Semantic Segmentation** (DeepLabV3 architecture with MobileNetV3-Large backbone, trained on the Cambridge-driving Labeled Video Database (CamVid) with 11 semantic classes).

---

## Table of Contents
- [Project Overview](#project-overview)
- [Segmentation Methods](#segmentation-methods)
  - [1. Edge Detection](#1-edge-detection)
  - [2. Split and Merge](#2-split-and-merge)
  - [3. Deep Learning (DeepLabV3 MobileNetV3-Large)](#3-deep-learning-deeplabv3-mobilenetv3-large)
- [Directory Structure](#directory-structure)
- [System Requirements & Dependencies](#system-requirements--dependencies)
  - [MATLAB Requirements](#matlab-requirements)
  - [Python Dependencies](#python-dependencies)
- [Deep Learning Model & Training Configuration](#deep-learning-model--training-configuration)
  - [Dataset & 11-Class Schema](#dataset--11-class-schema)
  - [Training Configuration](#training-configuration)
  - [Test Set Evaluation Results](#test-set-evaluation-results)
- [Usage Guide](#usage-guide)
  - [Launching the MATLAB GUI](#launching-the-matlab-gui)
  - [Configuring the Python Executable](#configuring-the-python-executable)
  - [Standalone Python Inference CLI](#standalone-python-inference-cli)
- [Verification & Tests](#verification--tests)
- [Important Limitations & Notes](#important-limitations--notes)

---

## Project Overview

Image segmentation partitions a digital image into multiple meaningful segments or regions. This project evaluates classical and modern deep learning methodologies under a consistent workflow, allowing interactive comparison of boundary extraction, region grouping, and deep semantic scene parsing on arbitrary input images as well as automotive driving scenes.

The application includes an end-to-end programmatic MATLAB GUI (`main_gui.m`) that coordinates MATLAB core algorithms and an external Python bridge for PyTorch deep learning inference.

---

## Segmentation Methods

### 1. Edge Detection
- **Algorithms**: Implements 6 distinct edge detection operators:
  - **Sobel** (First-order derivative gradient magnitude).
  - **Prewitt** (First-order directional gradient operator).
  - **Roberts** (Cross-gradient $2 \times 2$ differential operator).
  - **Canny** (Gaussian smoothing, gradient computation, non-maximum suppression, and hysteresis thresholding; supports builtin `edge` or custom pure-MATLAB fallback).
  - **Laplace** (Second-order derivative with zero-crossing thresholding and gradient gating).
  - **LoG** (Laplacian of Gaussian with scale parameter $\sigma$).
- **Object Extraction Pipeline**:
  - Image smoothing $\rightarrow$ Edge operator $\rightarrow$ Zero-crossing / thresholding $\rightarrow$ Morphological closing & opening $\rightarrow$ Hole filling $\rightarrow$ Connected component labeling $\rightarrow$ Bounding / centroid ranking $\rightarrow$ Mask overlay.

### 2. Split and Merge
- **Approach**: Quadtree recursive region decomposition followed by adjacency-graph region merging.
- **Key Parameters**:
  - `SplitThreshold`: Standard deviation threshold determining whether a quadtree block is homogeneous.
  - `MergeThreshold`: Maximum Euclidean mean-difference allowed to merge two neighboring regions.
  - `MinBlockSize`: Minimum block size ($2^k$) for recursive subdivision (e.g., $4 \times 4$).
  - `MinRegionSize`: Minimum pixel area to eliminate minor fragmented segments.
  - `ColorMode`: Supports `RGB` (3D Euclidean distance), `Grayscale`, or `L*a*b*`.

### 3. Deep Learning (DeepLabV3 MobileNetV3-Large)
- **Model**: PyTorch `torchvision.models.segmentation.deeplabv3_mobilenet_v3_large`.
- **Pretrained Weights**: Initialized with ImageNet / COCO weights and fine-tuned end-to-end.
- **Task**: 11-class semantic segmentation on urban driving scenes.
- **Integration**: Invoked seamlessly from MATLAB via `run_deep_learning_segmentation.m` through a high-performance CLI bridge (`predict_single.py`) with persistent validation caching.

---

## Directory Structure

```
IF4073-Image-Segmentation/
├── main_gui.m                            # Main entry point to launch MATLAB GUI
├── models/
│   └── camvid11_full_25epochs_best.pth   # Best trained DeepLabV3 model checkpoint (Val mIoU 65.43%)
├── data/
│   └── CamVid/
│       ├── class_dict.csv                # CamVid 32-class raw color dictionary
│       └── label_colors.txt              # Standard CamVid RGB color map
├── results/
│   ├── test_evaluation/                  # Full 168-image CamVid test set evaluation
│   │   ├── test_metrics.json             # Complete quantitative test metrics
│   │   ├── test_per_class_metrics.csv    # Per-class IoU, precision, recall, F1
│   │   ├── test_confusion_matrix.csv     # 11x11 normalized confusion matrix
│   │   ├── test_confusion_matrix.png     # Heatmap visualization of confusion matrix
│   │   ├── test_evaluation_report.md     # Detailed markdown test evaluation report
│   │   ├── test_visualization_contact_sheet.png # Contact sheet of 15 test predictions
│   │   └── test_sample_*.png             # Individual 4-panel test visualizations
│   └── training/                         # 25-epoch full training logs and curves
│       ├── camvid11_full_25epochs_history.csv
│       ├── camvid11_full_25epochs_summary.json
│       └── camvid11_full_25epochs_curves_combined.png
├── src/
│   ├── edge_detection/                   # Edge detection algorithms & morphological pipeline
│   │   ├── canny_edge.m
│   │   ├── default_edge_params.m
│   │   ├── detect_edges.m
│   │   ├── edge_method_list.m
│   │   ├── edge_segmentation.m
│   │   ├── laplace_edge.m
│   │   ├── log_edge.m
│   │   ├── prewitt_edge.m
│   │   ├── roberts_edge.m
│   │   ├── segment_from_edges.m
│   │   ├── sobel_edge.m
│   │   └── zero_crossing.m
│   ├── split_merge/                      # Quadtree split and merge implementation
│   │   ├── isHomogeneous.m
│   │   ├── mergeRegions.m
│   │   ├── splitMergeParams.m
│   │   ├── splitMergeSegmentation.m
│   │   └── splitRegions.m
│   ├── deep_learning/                    # PyTorch semantic segmentation module
│   │   ├── config.py                     # 11-class mapping, palette, class weights
│   │   ├── dataset.py                    # CamVid Dataset loader with augmentations
│   │   ├── evaluate_test.py              # 168-sample test evaluation script
│   │   ├── metrics.py                    # Streamed confusion matrix, Pixel Acc, mIoU
│   │   ├── model.py                      # DeepLabV3 MobileNetV3-Large definition
│   │   ├── predict_single.py             # CLI single-image inference bridge
│   │   ├── train.py                      # Multi-epoch PyTorch training pipeline
│   │   └── visualize_predictions.py      # Color palette and overlay generator
│   ├── gui/                              # Integrated MATLAB GUI and Python adapter
│   │   ├── find_python_executable.m      # Python discoverer with persistent cache
│   │   ├── launch_segmentation_gui.m     # Programmatic MATLAB GUI implementation
│   │   └── run_deep_learning_segmentation.m # MATLAB-to-Python execution adapter
│   └── utils/                            # Shared utilities (image loading, colorization)
│       ├── colorizeLabels.m
│       ├── fill_holes.m
│       ├── labelBoundaries.m
│       ├── load_image.m
│       └── overlayBoundaries.m
└── tests/                                # Automated unit and structure tests
    ├── test_dl_adapter.m                 # DL adapter bridge test suite
    ├── test_edge_segmentation.m          # Edge detection test suite
    ├── test_gui_structure.m              # GUI structure and dependency verification
    └── test_split_merge.m                # Split and merge test suite
```

---

## System Requirements & Dependencies

### MATLAB Requirements
- **MATLAB R2019b or newer** (Recommended: R2021a - R2024b).
- **Toolbox Requirements**:
  - Image Processing Toolbox (Optional; fallback pure-MATLAB implementations are provided for Sobel, Prewitt, Roberts, Laplacian, morphology, and connected-component operations if the toolbox is unavailable).

### Python Dependencies
Python is required for the Deep Learning module. Python **3.10 to 3.12** is recommended.
Install dependencies using `pip`:

```bash
pip install torch torchvision numpy pillow matplotlib
```

Verify your Python environment:
```bash
python -c "import torch, torchvision; print('PyTorch:', torch.__version__, '| Torchvision:', torchvision.__version__)"
```

---

## Deep Learning Model & Training Configuration

### Dataset & 11-Class Schema
Trained on the **Cambridge-driving Labeled Video Database (CamVid)**, mapping the original 32 annotations into **11 standardized operational classes** plus an ignore label:

| Class ID | Class Name | RGB Color | Description |
|:---:|:---|:---:|:---|
| 0 | Sky | `[128, 128, 128]` | Sky and horizon |
| 1 | Building | `[128, 0, 0]` | Buildings, walls, bridges |
| 2 | Pole | `[192, 192, 128]` | Utility poles, lamp posts, traffic sign poles |
| 3 | Road | `[128, 64, 128]` | Drivable road surface |
| 4 | Pavement | `[0, 0, 192]` | Sidewalks, pedestrian walkways |
| 5 | Tree | `[128, 128, 0]` | Trees, foliage, vegetation |
| 6 | SignSymbol | `[192, 128, 128]` | Traffic signs, signals, markings |
| 7 | Fence | `[64, 64, 128]` | Fences, guardrails, barriers |
| 8 | Car | `[64, 0, 128]` | Cars, buses, vans, trucks |
| 9 | Pedestrian | `[64, 64, 0]` | Pedestrians, walking persons |
| 10 | Bicyclist | `[0, 128, 192]` | Bicycles and riders |
| *255* | *Void / Ignore* | `[0, 0, 0]` | Unlabeled or ambiguous pixels (excluded from loss & metrics) |

### Training Configuration
- **Architecture**: `deeplabv3_mobilenet_v3_large` with ASPP (Atrous Spatial Pyramid Pooling).
- **Dataset Split**:
  - Training: 367 images (`data/CamVid/train`)
  - Validation: 101 images (`data/CamVid/val`)
  - Test: 168 images (`data/CamVid/test`)
- **Resolution**: $360 \times 480$ (standard CamVid benchmark dimensions).
- **Loss Function**: `CrossEntropyLoss` with inverse class-frequency weighting and `ignore_index=255`.
- **Optimizer**: AdamW ($\text{LR} = 5 \times 10^{-4}$, weight decay $= 10^{-4}$).
- **Scheduler**: Cosine Annealing LR ($T_{\max} = 25$, $\eta_{\min} = 10^{-6}$).
- **Epochs**: 25 epochs.
- **Checkpoint**: Best model saved at Epoch 22 with Validation mIoU **65.43%** (`models/camvid11_full_25epochs_best.pth`).

### Test Set Evaluation Results
Evaluation on all **168 unseen images** in `data/CamVid/test`:
- **Overall Test Pixel Accuracy**: **90.46%**
- **Overall Test Mean IoU (mIoU)**: **65.46%**

#### Per-Class IoU Breakdown:
| Class | Class Name | IoU (%) | Precision (%) | Recall (%) | F1-Score (%) |
|:---:|:---|:---:|:---:|:---:|:---:|
| 0 | Sky | 88.08% | 93.68% | 93.64% | 93.66% |
| 1 | Building | 80.37% | 88.07% | 90.11% | 89.08% |
| 2 | Pole | 36.37% | 61.64% | 45.42% | 52.29% |
| 3 | Road | 93.30% | 96.17% | 96.88% | 96.53% |
| 4 | Pavement | 75.64% | 85.90% | 86.29% | 86.10% |
| 5 | Tree | 74.07% | 87.72% | 82.52% | 85.04% |
| 6 | SignSymbol | 35.25% | 66.86% | 42.13% | 51.69% |
| 7 | Fence | 43.14% | 65.80% | 54.89% | 59.85% |
| 8 | Car | 78.43% | 89.96% | 85.74% | 87.80% |
| 9 | Pedestrian | 57.06% | 76.99% | 68.91% | 72.73% |
| 10 | Bicyclist | 68.39% | 83.18% | 79.17% | 81.12% |

Complete tables, confusion matrix heatmaps, and individual sample visual comparisons are archived in `results/test_evaluation/`.

---

## Usage Guide

### Launching the MATLAB GUI
Open MATLAB, navigate to the repository root, and run:

```matlab
>> main_gui
```

Alternatively:
```matlab
>> addpath(genpath('src'));
>> launch_segmentation_gui;
```

#### GUI Workflow:
1. Click **"Browse Image..."** to select an input image (supports JPG, PNG, BMP).
2. Choose a method: **Edge Detection**, **Split and Merge**, or **Deep Learning**.
3. Adjust the contextual parameters in the parameter panel:
   - *Edge Detection*: Operator (Sobel, Prewitt, Roberts, Canny, Laplace, LoG), Smooth Sigma, Zero-Crossing Ratio, Object Count.
   - *Split and Merge*: Split Threshold, Merge Threshold, Min Block Size, Min Region Size, Color Mode.
   - *Deep Learning*: Checkpoint path, Hardware Device (CUDA/CPU), Python Executable path.
4. Click **"Run Segmentation"**.
5. The 4-quadrant display renders:
   - *Top-Left*: Original Image
   - *Top-Right*: Intermediate / Binary Map (Edge Mask / Quadtree / Semantic Mask)
   - *Bottom-Left*: Segmented Result (Extracted Object / Mean Region Image / Colored Prediction)
   - *Bottom-Right*: Contour / Boundary Overlay or Semantic Class Distribution
6. Click **"Save Results..."** to export output images and metadata JSON.

### Configuring the Python Executable
The adapter automatically searches for a valid Python executable with PyTorch & Torchvision in:
1. Environment variables (`IF4073_PYTHON`, `PYTHON_EXECUTABLE`).
2. Local user Python installations (`%LOCALAPPDATA%\Programs\Python\Python312\python.exe`).
3. MATLAB `pyenv` executable path.
4. System `PATH`.

To explicitly configure a specific Python binary, set the environment variable prior to starting MATLAB:
```powershell
# Windows PowerShell
$env:IF4073_PYTHON="C:\Users\LENOVO\AppData\Local\Programs\Python\Python312\python.exe"
```
Or specify it directly in the GUI's **"Python Executable"** field. The validation result is persistently cached in MATLAB memory to eliminate repeated subprocess overhead.

### Standalone Python Inference CLI
To segment an image from the terminal without MATLAB:

```bash
python src/deep_learning/predict_single.py `
  --image "data/CamVid/test/0016E5_08131.png" `
  --checkpoint "models/camvid11_full_25epochs_best.pth" `
  --output-dir "results/single_prediction_test/with_gt_0016E5_08131" `
  --ground-truth "data/CamVid/test_labels/0016E5_08131_L.png" `
  --device "auto"
```

---

## Verification & Tests

The project includes test scripts located in `tests/`:

- **MATLAB Structural & Unit Tests**:
  - `tests/test_gui_structure.m`: Validates project paths, file existence, GUI specifications, and Python bridge availability.
  - `tests/test_dl_adapter.m`: Tests MATLAB-to-Python execution adapter with and without ground truth.
  - `tests/test_edge_segmentation.m`: Verifies all 6 edge detection operators and morphology pipelines.
  - `tests/test_split_merge.m`: Tests split and merge segmentation across synthetic and sample images.
- **Python Unit Tests**:
  - `python src/deep_learning/test_dataset.py`: Verifies dataset mapping and tensor shapes.
  - `python src/deep_learning/test_model.py`: Verifies model forward pass and parameter counts.
  - `python src/deep_learning/test_training.py`: Tests training loop iterations and metrics calculation.

To inspect the GUI configuration non-interactively in MATLAB:
```matlab
>> spec = launch_segmentation_gui('inspect')
```

---

## Important Limitations & Notes

> [!IMPORTANT]
> **MATLAB Runtime GUI Verification Required**:
> Automated validation in the current development environment was performed using static code analysis, block balance verification, and Python subprocess simulations because MATLAB Desktop is not installed on the local build host. Before final submission or deployment, verify GUI interactions, interactive file dialogs, and display rendering in an authentic MATLAB Desktop environment.

> [!NOTE]
> **Checkpoints and Large Files**:
> Checkpoint files (`models/*.pth`) and raw image folders in `data/CamVid/` exceed standard GitHub file size limits and are tracked in `.gitignore`. The best model checkpoint (`camvid11_full_25epochs_best.pth`, 44.8 MB) should be retained in `models/` for inference.
