# Aircraft Detection Using YOLOv11: A High-Performance Approach to Aerial Vehicle Recognition

## Abstract
This repository contains the official implementation of our novel approach to aircraft detection utilizing the cutting-edge YOLOv11 architecture. Our method excels in identifying and classifying aircraft under diverse conditions, including variations in scale, complex backgrounds, and different types of aircraft. The work significantly contributes to aerial vehicle recognition, offering valuable insights for applications such as aviation safety, surveillance, and autonomous air traffic management.

## Table of Contents
- [Key Features](#key-features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

## Key Features
- **Dynamic YOLO Version Management**: Automatically utilizes the latest YOLO version, ensuring peak performance.
- **Efficient Data Preprocessing**: Implements tile-based image processing for high-resolution aerial imagery.
- **End-to-End Training Pipeline**: Comprehensive training pipeline including data augmentation, automated hyperparameter tuning, and validation splitting.
- **Detailed Performance Analysis**: Generates visualizations and performance metrics for deeper insights.
- **Reproducible Results**: Consistent across different hardware and software environments for easy replication.

## Requirements
- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- 16GB RAM (minimum)

### Dependencies
- `torch>=1.7.0`
- `ultralytics`
- `numpy`
- `pandas`
- `Pillow`
- `PyYAML`
- `matplotlib`
- `tqdm`

## Installation

Clone the repository:

```bash
git clone https://github.com/MHassaanButt/aircraft-detection-yolov11.git
cd aircraft-detection-yolov11
```

## Dataset Preparation
Our implementation includes a sophisticated data preprocessing pipeline that:

- Automatically tiles large aerial images.
- Handles annotations appropriately.
- Splits data into training and validation sets.

### To prepare your dataset:
```bash
python main.py \
  --data-dir /path/to/raw/data \
  --output-dir processed_data \
  --tile-size 512 \
  --tile-overlap 64
```
## Training

### Train the model with:
```bash
python main.py \
  --data-dir /path/to/raw/data \
  --output-dir processed_data \
  --epochs 50 \
  --batch-size 16 \
  --device cuda:0  # or 'cpu' if GPU is not available
```
### Key Training Parameters
--epochs: Number of training epochs (default: 50)
--batch-size: Batch size for training (default: 16)
--tile-size: Size of image tiles (default: 512)
--device: Computing device to use

## Evaluation
Our model achieves state-of-the-art performance on standard aircraft detection benchmarks:

| **Metric** | **Value** |
|------------|-----------|
| mAP50      | 0.92      |
| mAP95     | 0.69      |

## Results
Our approach demonstrates robust performance across various challenging scenarios:
- Different lighting conditions
- Various aircraft types and sizes
- Complex backgrounds and occlusions

## Citation
If you use this work in your research, please cite:

```bibtex
@inproceedings{lastname2024aircraft,
  title={Aircraft Detection Using YOLOv11: A High-Performance Approach to Aerial Vehicle Recognition},
  author={Lastname, Firstname and Coauthor, Name},
  booktitle={Proceedings of the International Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Additional Information

### Experiment Tracking
The training script automatically generates comprehensive experiment logs, including:
- Training and validation metrics
- Model checkpoints
- Performance visualizations

Example directory structure after training:

```bash
experiments/
└── yolov11n_ultr8-0-20_e50_i512_20240404_123456/
    ├── config.txt
    ├── mAP50_curve.png
    ├── validation_metrics.txt
    └── train/
        └── [training outputs]
```

## Troubleshooting

### Common issues and solutions:

- **CUDA out of memory**:
  - Reduce batch size.
  - Decrease image size.

- **Dataset not found**:
  - Ensure all paths are absolute.
  - Verify dataset structure matches the expected format.

For additional support, please open an issue on our GitHub repository. For more details, refer to our full paper.
