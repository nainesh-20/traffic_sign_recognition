# 🚦 Real-Time Traffic Sign Classification System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Accuracy](https://img.shields.io/badge/Accuracy-99.72%25-brightgreen.svg)](https://github.com/yourusername/traffic-sign-classification)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-ready deep learning system for real-time traffic sign classification that achieves **99.72% accuracy** through advanced data engineering and class-aware training strategies.

## 🎯 Project Highlights

- **🏆 State-of-the-art Performance**: 99.72% test accuracy on GTSRB dataset
- **⚖️ Balanced Classification**: 99.69% F1-score on minority classes vs 99.75% on majority classes
- **🔧 Production-Ready**: Handles class imbalance, data quality issues, and real-world challenges
- **📊 Data-Driven Approach**: Comprehensive analysis leading to targeted solutions
- **🚀 Advanced Techniques**: Class-aware augmentation + weighted focal loss

## 📋 Table of Contents

- [Problem Statement](#problem-statement)
- [Solution Architecture](#solution-architecture)
- [Dataset Analysis](#dataset-analysis)
- [Technical Implementation](#technical-implementation)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Future Work](#future-work)

## 🚨 Problem Statement

Road safety is a critical global concern, with traffic signs playing a vital role in ensuring safe driving. However, several challenges exist:

- **Human Error**: Drivers may miss or misinterpret signs due to distractions or poor visibility
- **Autonomous Vehicles**: Self-driving cars require accurate, real-time traffic sign detection
- **Varying Conditions**: Signs appear under different lighting, weather, and viewing angles
- **Critical Safety**: Misclassification can lead to serious accidents

**Goal**: Develop a robust CNN-based system that can detect and classify traffic signs in real-time with production-level accuracy.

## 🏗️ Solution Architecture

### Core Components

1. **Advanced Data Pipeline**
   - Comprehensive data quality assessment
   - Class imbalance analysis and mitigation
   - Class-aware augmentation strategies

2. **Intelligent Model Training**
   - Transfer learning with MobileNetV2
   - Combined weighted focal loss function
   - Stratified sampling and early stopping

3. **Production Features**
   - Systematic evaluation framework
   - Model versioning and experiment tracking
   - Real-world performance validation

## 📊 Dataset Analysis

### GTSRB Dataset Overview
- **Source**: German Traffic Sign Recognition Benchmark
- **Classes**: 43 traffic sign categories
- **Total Samples**: 39,209 images
- **Format**: RGB images of varying sizes (resized to 224×224)

### Key Challenges Identified

| Challenge | Impact | Solution |
|-----------|--------|----------|
| **Severe Class Imbalance** | 10.71:1 ratio between majority/minority classes | Class-aware augmentation + weighted loss |
| **Data Quality Issues** | 13.77% low-quality images | Quality-focused preprocessing |
| **Minority Class Performance** | 11 classes with <500 samples | Aggressive augmentation for minority classes |

### Data Quality Assessment

📈 Quality Issues by Severity:

- Low Quality Images: 13.77%
- Most Affected Classes: 8, 10, 6, 42 (30-39% quality issues)
- Perfect Quality Classes: 0, 29, 32, 37 (0% issues)


## 🔧 Technical Implementation

### Model Architecture

MobileNetV2 with custom classifier

```
model = mobilenet_v2(weights="IMAGENET1K_V1")
model.classifier = nn.Sequential(
nn.Dropout(0.2),
nn.Linear(model.last_channel, 43)
)
```

### Class-Aware Augmentation Strategy

| Class Type | Augmentation Intensity | Techniques Applied |
|------------|----------------------|-------------------|
| **Minority + Quality Issues** | AGGRESSIVE | Heavy rotation, weather simulation, noise injection |
| **Minority Classes** | MODERATE | Standard augmentation with increased probability |
| **Quality Issues** | QUALITY_FOCUSED | Brightness/contrast enhancement, CLAHE |
| **Majority Classes** | STANDARD | Basic augmentation for generalization |

### Advanced Loss Function

Combined Weighted CrossEntropy + Focal Loss
loss = 0.7 * weighted_ce_loss + 0.3 * focal_loss


### Training Configuration
- **Optimizer**: Adam (lr=0.001, weight_decay=1e-4)
- **Scheduler**: ReduceLROnPlateau
- **Batch Size**: 32
- **Early Stopping**: 7 epochs patience
- **Data Split**: 70% train, 15% validation, 15% test (stratified)

## 📈 Results

### Performance Metrics

| Metric | Score | Industry Benchmark |
|--------|-------|-------------------|
| **Test Accuracy** | **99.72%** | 95-98% |
| **Validation Accuracy** | **99.77%** | 95-98% |
| **Overall F1-Score** | **99.74%** | 95-97% |
| **Minority Classes F1** | **99.69%** | Often <90% |
| **Majority Classes F1** | **99.75%** | 95-98% |

### Class Performance Analysis

🎯 Perfect Performance (F1 = 1.0): 5 classes including minority class 0
📊 Lowest Performance: Class 24 (F1 = 0.981) - still excellent
🔄 Total Misclassifications: Only 13 across entire test set



### Training Dynamics
- **Convergence**: Smooth training with no overfitting
- **Generalization**: Validation accuracy > training accuracy
- **Efficiency**: Achieved optimal performance in 15 epochs

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Setup


Clone repository
```
git clone https://github.com/nainesh-20/traffic-sign-classification.git
cd traffic-sign-classification
```

Install dependencies
```
pip install torch torchvision albumentations timm mlflow seaborn plotly
```

Download GTSRB dataset
```
wget https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip
unzip GTSRB_Final_Training_Images.zip
```


## 💻 Usage

### Quick Start

Load trained model
```
from traffic_sign_classifier import TrafficSignClassifier

classifier = TrafficSignClassifier('best_model.pth')
prediction = classifier.predict('path/to/traffic_sign.jpg')
print(f"Predicted class: {prediction['class']}, Confidence: {prediction['confidence']:.2f}")
```


### Training from Scratch

Run complete training pipeline
```
python train.py --config config/production.yaml
```


### Evaluation

Comprehensive evaluation
```
python evaluate.py --model best_model.pth --dataset test
```


## 📁 Project Structure
```
traffic-sign-classification/
├── data/
│ ├── GTSRB/
│ └── processed/
├── src/
│ ├── data/
│ │ ├── dataset.py # Custom dataset classes
│ │ ├── augmentation.py # Class-aware augmentation
│ │ └── analysis.py # Data quality assessment
│ ├── models/
│ │ ├── architectures.py # Model definitions
│ │ ├── loss.py # Advanced loss functions
│ │ └── trainer.py # Training pipeline
│ ├── evaluation/
│ │ ├── metrics.py # Evaluation metrics
│ │ └── visualization.py # Result visualization
│ └── utils/
│ ├── config.py # Configuration management
│ └── helpers.py # Utility functions
├── notebooks/
│ ├── 01_data_analysis.ipynb
│ ├── 02_model_training.ipynb
│ └── 03_evaluation.ipynb
├── config/
│ └── production.yaml
├── requirements.txt
├── README.md
└── LICENSE
```

## 🔮 Future Work

### Immediate Enhancements
- [ ] **Real-time Video Processing**: Webcam integration for live classification
- [ ] **Model Deployment**: REST API with FastAPI/Flask
- [ ] **Mobile Optimization**: TensorFlow Lite conversion for edge devices
- [ ] **Gradio Interface**: Interactive web demo

### Advanced Features
- [ ] **Multi-country Support**: Extend to other traffic sign datasets
- [ ] **Uncertainty Quantification**: Confidence estimation for predictions
- [ ] **Active Learning**: Continuous improvement with new data
- [ ] **Model Monitoring**: Production performance tracking

### Research Directions
- [ ] **Vision Transformers**: Experiment with ViT architectures
- [ ] **Self-supervised Learning**: Reduce dependency on labeled data
- [ ] **Federated Learning**: Distributed training across devices

## 🏆 Key Achievements

✅ **Production-Ready Performance**: 99.72% accuracy exceeds industry standards  
✅ **Balanced Classification**: Solved severe class imbalance (10.71:1 ratio)  
✅ **Data Quality Management**: Systematically addressed 13.77% low-quality images  
✅ **Advanced ML Engineering**: Class-aware augmentation + weighted focal loss  
✅ **Comprehensive Evaluation**: Beyond accuracy to precision, recall, F1-score  
✅ **Reproducible Results**: Systematic approach with experiment tracking  

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **GTSRB Dataset**: Institut für Neuroinformatik, Ruhr-Universität Bochum
- **PyTorch Community**: For the deep learning framework
- **Albumentations**: For advanced image augmentation capabilities
- **MLflow**: For experiment tracking and model management

## 📞 Contact

- **Author**: Nainesh Rathod
- **Email**: naineshrathod3000@gmail.com
- **LinkedIn**: https://www.linkedin.com/in/nainesh-rathod/
- **GitHub**: https://github.com/nainesh-20/

---

<div align="center">

**⭐ If this project helped you, please give it a star! ⭐**

</div>