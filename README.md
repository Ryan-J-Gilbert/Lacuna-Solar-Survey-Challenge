# Solar Panel and Boiler Counting Model

This repository contains a deep learning pipeline for counting solar panels and boilers in aerial images. The model integrates image features with metadata and employs advanced techniques such as cross-attention, test-time augmentation (TTA), and calibration for optimal performance.

## Pipeline Overview

1. **Data Preprocessing**:
   - Images are resized and normalized.
   - Metadata (e.g., image origin, placement) is encoded using one-hot encoding.

2. **Model Architecture**:
   - **Backbone**: EfficientNetV2 is used for feature extraction from images.
   - **Metadata Processor**: A fully connected network processes metadata.
   - **Cross-Attention**: Combines image and metadata features using multi-head attention.
   - **Counting Head**: A custom regressor predicts the number of solar panels and boilers.

3. **Training**:
   - Uses a combination of Huber Loss and L1 Loss for robust training.
   - Employs advanced augmentation techniques to improve generalization.
   - Implements early stopping and a OneCycleLR scheduler for efficient training.

4. **Validation**:
   - Performs cross-validation with K-Fold splitting.
   - Optimizes rounding thresholds for better integer predictions.

5. **Inference**:
   - Supports Test-Time Augmentation (TTA) to improve robustness.
   - Applies calibration thresholds to refine predictions.

6. **Submission**:
   - Generates three types of submissions:
     - Raw predictions.
     - Integer predictions (standard rounding).
     - Calibrated predictions (optimized thresholds).

## Features

- **Metadata Integration**: Combines image and metadata features for improved accuracy.
- **Cross-Attention**: Enhances feature interaction between image and metadata.
- **Advanced Augmentation**: Includes geometric and color transformations.
- **Test-Time Augmentation (TTA)**: Averages predictions over multiple augmented versions of test images.
- **Calibration**: Optimizes rounding thresholds for better integer predictions.

## Requirements

- Python 3.8+
- PyTorch 1.12+
- Albumentations
- timm
- scikit-learn
- tqdm

## Usage

### Training
Run the training pipeline with cross-validation:
```bash
python newbest.py
```

### Inference
Perform inference on the test dataset:
```bash
python newbest.py
```

### Outputs
- `submission_raw.csv`: Raw predictions.
- `submission_integer.csv`: Integer predictions (standard rounding).
- `submission_calibrated.csv`: Calibrated predictions.

## Acknowledgements

This project is inspired by contributions from the Zindi community, including valuable insights and code snippets from users like zulo40.

---
