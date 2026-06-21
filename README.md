<div align="center">

# 🐱 Animal Classifier 🐶
### Classical ML → CNNs → Vision Transformers

A comprehensive benchmarking study across **six architectures** — from Logistic Regression to Vision Transformers — on binary image classification.

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Models Evaluated](#-models-evaluated)
- [Dataset](#-dataset)
- [Results & Benchmarks](#-results--benchmarks)
- [Key Techniques](#️-key-techniques)
- [Project Structure](#️-project-structure)
- [Getting Started](#-getting-started)
- [Training Pipeline](#-training-pipeline)
- [Evaluation Metrics](#-evaluation-metrics)
- [Sample Inference](#-sample-inference)
- [Future Enhancements](#-future-enhancements)

---

## 🔭 Overview

This project systematically evaluates **six image classification architectures** spanning three generations of computer vision, all benchmarked on the same Cat vs. Dog binary classification task.

The goal is to measure not just accuracy, but the *full picture*: training time, inference speed, parameter count, FLOPs, and generalization capability — providing a practical guide to choosing the right model for real-world constraints.

```
Traditional ML  ──────────────────────────►  Vision Transformers
   (seconds)       CNNs    Transfer Learning        (minutes)
Low accuracy  ──────────────────────────►  High accuracy
Few params    ──────────────────────────►  Millions of params
```

---

## 🚀 Models Evaluated

### 🧮 Traditional Machine Learning *(from scratch)*

| Model | Approach |
|-------|----------|
| **Logistic Regression** | Hand-crafted pixel/feature vectors, gradient descent |
| **Support Vector Machine (SVM)** | Kernel-based margin maximization on flattened inputs |

### 🧠 Deep Learning

| Model | Approach |
|-------|----------|
| **Custom CNN** | Lightweight convolutional network built from scratch |
| **ResNet-34** | Fine-tuned on pretrained ImageNet weights (skip connections) |
| **EfficientNet-B0** | Fine-tuned compound-scaled network (best efficiency/accuracy tradeoff) |

### 🔮 Vision Transformers

| Model | Approach |
|-------|----------|
| **ViT-Base/Patch16-224** | Fine-tuned attention-based transformer on 16×16 image patches |

---

## 📊 Dataset

Binary image classification dataset:

```
dataset/
├── cats/   🐱  (n images)
└── dogs/   🐶  (n images)
```

> Train/validation split is performed programmatically inside each notebook.

**Preprocessing pipeline:**

| Step | Detail |
|------|--------|
| Resize | 224 × 224 px (ViT & CNNs) |
| Normalize | ImageNet mean/std `([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])` |
| Augmentation | Random flip, crop, color jitter, rotation |
| Label encoding | Cat → `0`, Dog → `1` |

---

## 📈 Results & Benchmarks

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1 Score | Params | Training Time |
|-------|----------|-----------|--------|----------|--------|---------------|
| Logistic Regression | ~63% | — | — | — | Low | Seconds |
| SVM | ~67% | — | — | — | Low | Seconds |
| Custom CNN | ~80% | — | — | — | ~500K | Minutes |
| ResNet-34 (FT) | ~95%+ | — | — | — | 21.8M | ~30 min |
| EfficientNet-B0 (FT) | ~96%+ | — | — | — | 5.3M | ~25 min |
| ViT-Base/16 (FT) | ~97%+ | — | — | — | 86M | ~45 min |

> 📝 Fill in the exact values from your training runs. Placeholder estimates shown above.

---

### 📉 Summaries & Confusion Matrices

<details>
<summary><b>Logistic Regression</b></summary>

![Logistic Regression Results](images/logistic.png)

</details>

<details>
<summary><b>Support Vector Machine (SVM)</b></summary>

![SVM Results](images/svm.png)

</details>

<details>
<summary><b>Custom CNN</b></summary>

![Custom CNN Results](images/cnn.png)

</details>

<details>
<summary><b>ResNet-34 (Fine-Tuned)</b></summary>

![ResNet-34 Results](images/resnet.png)

</details>

<details>
<summary><b>EfficientNet-B0 (Fine-Tuned)</b></summary>

![Eff_Net_B0 Results](images/eff_net.png)

</details>

<details>
<summary><b>ViT-Base/Patch16-224 (Fine-Tuned)</b></summary>

![Vit-Base/Patch16 Results](images/vit.png)

</details>

---

## 🛠️ Key Techniques

- **Data Augmentation** — Random crops, flips, and color jitter for regularization
- **Feature Extraction** — Pixel flattening for classical ML; learned hierarchical features for CNNs
- **Transfer Learning** — Leveraging ImageNet pretraining for rapid convergence
- **Fine-Tuning** — Unfreezing top layers to adapt pretrained representations
- **Hyperparameter Tuning** — Learning rate scheduling, weight decay, batch size sweeps
- **Performance Benchmarking** — Accuracy, speed, complexity across all models
- **Custom Inference** — Single-image prediction pipeline
- **Confusion Matrix Analysis** — Per-class error identification
- **Precision / Recall / F1** — Balanced evaluation beyond raw accuracy

---

## 🏗️ Project Structure

```
Animal_Classifier/
│
├── 📓 dataset_analysis.ipynb       # Exploratory data analysis & class distribution
├── 📓 logistic_reg.ipynb           # Logistic Regression (from scratch)
├── 📓 svm.ipynb                    # SVM (from scratch)
├── 📓 small-cnn-final.ipynb        # Custom lightweight CNN
├── 📓 resnet-34-ft.ipynb           # ResNet-34 fine-tuning
├── 📓 efficient_net_B0_FT.ipynb    # EfficientNet-B0 fine-tuning
├── 📓 vit_base_patch_16_FT.ipynb   # ViT-Base/16 fine-tuning
│
├── 📄 requirements.txt             # Python dependencies
├── 🐱 cat.png                      # Sample inference image
└── 🐶 dog.jpeg                     # Sample inference image
```

---

## ⚡ Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Animal_Classifier.git
cd Animal_Classifier
```

### 2. Set up environment

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Prepare dataset

Place your dataset in the following structure (or update the paths inside the notebooks):

```
dataset/
├── cats/
└── dogs/
```

The train/validation split is handled automatically inside each notebook.

### 4. Run notebooks in order

```bash
jupyter notebook dataset_analysis.ipynb   # Start here
```

Then proceed through each model notebook in any order.

---

## 🔄 Training Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                   TRAINING PIPELINE                     │
│                                                         │
│  Dataset Analysis  ──►  Data Preprocessing             │
│                               │                         │
│                               ▼                         │
│                      Data Augmentation                  │
│                               │                         │
│                    ┌──────────┴──────────┐              │
│                    ▼                     ▼              │
│             Train from Scratch     Fine-Tune Pretrained │
│             (ML / Custom CNN)      (ResNet/EffNet/ViT)  │
│                    │                     │              │
│                    └──────────┬──────────┘              │
│                               ▼                         │
│                          Evaluation                     │
│                               │                         │
│                               ▼                         │
│                    Benchmark Comparison                 │
│                               │                         │
│                               ▼                         │
│                   Custom Image Inference                │
└─────────────────────────────────────────────────────────┘
```

---

## 📏 Evaluation Metrics

Every model is evaluated on a consistent set of metrics for fair comparison:

| Metric | Purpose |
|--------|---------|
| **Accuracy** | Overall correct predictions |
| **Precision** | True positives out of predicted positives |
| **Recall** | True positives out of actual positives |
| **F1 Score** | Harmonic mean of precision and recall |
| **Confusion Matrix** | Per-class breakdown of errors |
| **Training Time** | Wall-clock training duration |
| **Inference Time** | Per-image prediction latency |
| **Parameters (#)** | Model size in parameter count |
| **FLOPs** | Computational cost per forward pass |

---

## 🔍 Sample Inference

```python
from PIL import Image
from torchvision import transforms
import torch

# Load your trained model
model.eval()

# Preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Run inference
img = Image.open("dog.jpeg").convert("RGB")
tensor = transform(img).unsqueeze(0)          # Add batch dimension

with torch.no_grad():
    output = model(tensor)
    pred = torch.argmax(output, dim=1).item()

label_map = {0: "🐱 Cat", 1: "🐶 Dog"}
print(f"Prediction: {label_map[pred]}")
# Prediction: 🐶 Dog
```

---

## ⚙️ Technology Stack

<div align="center">

| Library | Purpose |
|---------|---------|
| ![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) | Core language |
| ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white) | Deep learning framework |
| ![Torchvision](https://img.shields.io/badge/Torchvision-EE4C2C?style=flat-square&logo=pytorch&logoColor=white) | Pretrained models & transforms |
| ![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black) | ViT model & tokenizer |
| ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white) | Classical ML & evaluation |
| ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white) | Numerical computation |
| ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square) | Plotting & visualization |
| ![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat-square) | Statistical visualization |

</div>

---

## 🎯 Future Enhancements

- [ ] **MobileNetV3** — Lightweight mobile-optimized benchmark
- [ ] **ConvNeXt** — Modern pure-CNN architecture comparison
- [ ] **Swin Transformer** — Hierarchical ViT fine-tuning
- [ ] **Automated Hyperparameter Optimization** — Optuna / Ray Tune integration
- [ ] **Streamlit Web App** — Interactive drag-and-drop inference demo
- [ ] **FastAPI Deployment** — REST endpoint for production inference
- [ ] **Model Quantization** — INT8 / FP16 optimization for edge deployment
- [ ] **Grad-CAM Visualizations** — Saliency maps for model explainability
- [ ] **ONNX Export** — Cross-framework model portability

---

<div align="center">

**Found this project useful?**
⭐ Star the repository to support the work!

*Built with curiosity — from pixels to patches.*

</div>