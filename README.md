<div align="center">

# 🚗 Self-Driving Car — Traffic Sign Classifier

### Deep Learning for Autonomous Vehicle Perception

![Python](https://img.shields.io/badge/Python-Computer_Vision-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep_Learning-orange?style=for-the-badge&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-CNN-red?style=for-the-badge&logo=keras)
![OpenCV](https://img.shields.io/badge/OpenCV-Image_Processing-green?style=for-the-badge&logo=opencv)
![Self Driving Cars](https://img.shields.io/badge/Domain-Autonomous_Vehicles-purple?style=for-the-badge)

Udacity Self-Driving Car Engineer Nanodegree Project

</div>

---

# Overview

Traffic sign recognition is a fundamental capability for autonomous vehicles.

This project implements a Convolutional Neural Network (CNN) capable of classifying German road signs from camera imagery. The system learns visual features directly from image data and predicts the correct traffic sign category.

The project demonstrates a complete computer vision workflow:

- Dataset exploration
- Data preprocessing
- CNN design
- Model training
- Performance evaluation
- Feature visualization

---

# Project Objectives

The goals of this project were to:

- Explore and analyze a real-world traffic sign dataset
- Preprocess images for neural network training
- Build a Convolutional Neural Network in TensorFlow/Keras
- Train and evaluate the classifier
- Investigate learned visual representations
- Demonstrate how deep learning can support autonomous driving perception systems

---

# Dataset

The project uses the German Traffic Sign Recognition Benchmark (GTSRB).

The dataset contains thousands of labeled traffic sign images across multiple classes including:

- Speed limits
- Warning signs
- Stop signs
- Yield signs
- Priority signs
- Directional signs

This dataset is widely used as a benchmark for autonomous vehicle perception research.

---

# Technical Skills Demonstrated

## Computer Vision

- Image preprocessing
- Normalization
- Dataset visualization
- Feature extraction

## Deep Learning

- Convolutional Neural Networks (CNNs)
- Activation functions
- Model training
- Validation workflows

## Autonomous Vehicle Perception

- Traffic sign recognition
- Scene understanding
- Visual classification pipelines

---

# Repository Structure

```text
Traffic_Sign_Classifier.ipynb     Main notebook
writeup_report.pdf               Project report
visualize_cnn.png                CNN filter visualization
signnames.csv                    Class labels
lenet.*                          Trained model files
examples/                        Example images
```

---

# CNN Feature Visualization

The project includes visualization of learned convolutional filters.

These visualizations help understand which visual patterns the network learns to detect when recognizing road signs.

<img src="visualize_cnn.png" width="900">

---

# Results

The trained model successfully classifies traffic signs from unseen validation data and demonstrates the core principles behind modern perception systems used in autonomous vehicles.

The project also explores how convolutional filters progressively learn geometric structures such as:

- Edges
- Shapes
- Borders
- Sign silhouettes
- Symbol patterns

---

# Key Takeaways

This project provided practical experience with:

- TensorFlow
- Keras
- CNN architectures
- Autonomous vehicle perception
- Traffic sign recognition
- Deep learning experimentation

It also serves as one of the foundational perception modules completed as part of the Self-Driving Car Engineer Nanodegree.

---

# Related Project

This repository is part of a larger autonomous driving portfolio including:

- Finding Lane Lines
- Advanced Lane Finding
- Traffic Sign Classification
- Behavioral Cloning
- Extended Kalman Filters
- Kidnapped Vehicle Localization
- Highway Path Planning
- PID Control Systems

---

# Disclaimer

This repository is provided for educational and portfolio purposes.

Students may read the code and reports for learning purposes, but submitting this work as coursework would constitute plagiarism and may violate academic integrity policies.

Copyright © Sabrina Palis
