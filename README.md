# Multi-Task CNN Models in PyTorch

## 📌 Project Overview

This project implements multiple convolutional neural network (CNN) architectures using PyTorch.

It includes models for:

- Image Classification
- Semantic Segmentation
- Bounding Box Regression
- Image Generation (Autoencoder)

All models are built using reusable modular blocks.

---

## 🧱 Core Architecture Components

- **BasicBlock**: Conv2d → Normalization → ReLU  
- **DownBlock**: BasicBlock → MaxPool2d  
- **UpBlock**: ConvTranspose2d → BasicBlock  

These blocks are reused across tasks to build encoder-decoder structures.

---

## 🧠 Implemented Models

### 1️⃣ ImageClassifier
- CNN backbone with DownBlocks
- Adaptive Average Pooling
- Fully connected classification head

### 2️⃣ ImageSegmenter
- Simplified U-Net style encoder-decoder
- Skip connections
- 1x1 convolution for pixel-wise classification

### 3️⃣ BBoxRegressor
- Shared convolutional backbone
- Regression head predicting bounding box coordinates

### 4️⃣ ImageGenerator
- Convolutional autoencoder
- Encoder-decoder structure
- Sigmoid output for normalized pixel values

---

## 🛠 Requirements

- Python 3.9+
- PyTorch

Install dependencies:

```bash
pip install torch
