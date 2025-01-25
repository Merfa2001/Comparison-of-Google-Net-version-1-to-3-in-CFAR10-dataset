# Comparison-of-Google-Net-version-1-to-3-in-CFAR10-dataset
# GoogleNet Versions Comparison (v1, v2, v3) on CIFAR-10

This repository contains implementations and comparisons of GoogleNet (Inception) architectures (v1, v2, and v3) trained on the CIFAR-10 dataset. The project demonstrates the evolution of the Inception architecture and its impact on image classification performance.

## Table of Contents
1. [Introduction](#introduction)
2. [Architecture Overview](#architecture-overview)
3. [Dataset](#dataset)
4. [Implementation Details](#implementation-details)
5. [Results](#results)
6. [Usage](#usage)
7. [Dependencies](#dependencies)
8. [Contributing](#contributing)
9. [License](#license)

---

## Introduction

GoogleNet (Inception) is a family of deep convolutional neural networks designed for image classification and recognition tasks. This project compares three versions of GoogleNet (v1, v2, and v3) on the CIFAR-10 dataset, highlighting the architectural improvements and their impact on performance.

---

## Architecture Overview

### **GoogleNet v1**
- **Key Features**:
  - Original Inception modules with parallel 1x1, 3x3, and 5x5 convolutions
  - Auxiliary classifiers for intermediate supervision
  - 22 layers deep
- **Strengths**:
  - Efficient use of parameters
  - Multi-scale feature extraction
- **Limitations**:
  - No batch normalization
  - Limited factorization of convolutions

### **GoogleNet v2**
- **Key Features**:
  - Introduced batch normalization
  - Factorized convolutions (e.g., 5x5 → two 3x3 layers)
  - Asymmetric convolutions (e.g., 3x3 → 1x3 + 3x1)
- **Strengths**:
  - Faster convergence
  - Better regularization
- **Limitations**:
  - Still uses large filters in some cases

### **GoogleNet v3**
- **Key Features**:
  - Advanced factorization (e.g., 7x7 → 1x7 + 7x1)
  - Label smoothing
  - RMSprop optimizer
  - More efficient grid size reduction
- **Strengths**:
  - Higher accuracy
  - Better computational efficiency
- **Limitations**:
  - More complex implementation

---

## Dataset

### **CIFAR-10**
- **Description**: A dataset of 60,000 32x32 color images in 10 classes (e.g., airplane, cat, dog).
- **Classes**: 10 classes with 6,000 images per class.
- **Split**:
  - Training: 50,000 images
  - Testing: 10,000 images
- **Preprocessing**:
  - Normalized pixel values to [0, 1]
  - One-hot encoded labels

---

## Implementation Details

### **Code Structure**
- `googlenet_v1.py`: Implementation of GoogleNet v1.
- `googlenet_v2.py`: Implementation of GoogleNet v2.
- `googlenet_v3.py`: Implementation of GoogleNet v3.
- `train.py`: Training script for all versions.
- `evaluate.py`: Evaluation and visualization script.
- `utils.py`: Utility functions for data loading and preprocessing.

### **Training**
- **Optimizer**: Adam (v1, v2), RMSprop (v3)
- **Learning Rate**: 0.001 (v1, v2), 0.0005 (v3)
- **Batch Size**: 64
- **Epochs**: 20
- **Loss Function**: Categorical Crossentropy with label smoothing (v3)
- **Callbacks**:
  - Early stopping
  - Learning rate reduction on plateau
  - Memory monitoring

### **Evaluation Metrics**
- Top-1 Accuracy
- Top-5 Accuracy
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)


## Usage

### **Training**
To train all versions of GoogleNet:
```bash
python train.py
```

### **Evaluation**
To evaluate and visualize results:
```bash
python evaluate.py
```

### **Saving and Loading Models**
- Models are saved in the `saved_models/` directory.
- To load a specific model:
  ```python
  from tensorflow.keras.models import load_model
  model = load_model("saved_models/googlenet_v3.h5")
  ```

---

## Dependencies

- Python 3.8+
- TensorFlow 2.10+
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

Install dependencies using:
```bash
pip install -r requirements.txt
```

---

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature/bugfix.
3. Submit a pull request with a detailed description of your changes.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments
- Google Research for the original Inception papers.
- CIFAR-10 dataset providers.
- TensorFlow team for the deep learning framework.

---

For questions or feedback, please open an issue or contact the maintainers.
