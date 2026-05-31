<div align="center">

# 🚗 Self-Driving Car — Behavioral Cloning

### End-to-End Deep Learning for Autonomous Steering

![Python](https://img.shields.io/badge/Python-Deep_Learning-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Neural_Networks-orange?style=for-the-badge&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-CNN-red?style=for-the-badge&logo=keras)
![Computer Vision](https://img.shields.io/badge/Computer_Vision-Autonomous_Driving-green?style=for-the-badge)
![Udacity](https://img.shields.io/badge/Udacity-Self_Driving_Car_Engineer-purple?style=for-the-badge)

Udacity Self-Driving Car Engineer Nanodegree Project

</div>

---

# Overview

Can a neural network learn to drive simply by watching a human driver?

In this project, a Convolutional Neural Network (CNN) is trained to predict steering angles directly from front-facing camera images collected in a driving simulator.

Rather than manually programming lane-following rules, the model learns driving behavior from examples, an approach commonly known as **Behavioral Cloning**.

The final model successfully drives autonomously around the simulator track without leaving the road.

---

# Project Objectives

The goals of this project were to:

- Train an end-to-end steering model
- Learn vehicle control directly from image data
- Explore CNN architectures for autonomous driving
- Apply data augmentation techniques
- Validate autonomous driving performance in simulation

---

# Model Architecture

The model is based on NVIDIA's End-to-End Learning for Self-Driving Cars architecture.

Pipeline:

1. Image cropping
2. Image normalization
3. Convolutional feature extraction
4. Fully connected control layers
5. Steering angle prediction

Key components include:

- Cropping layer to remove irrelevant sky and vehicle hood information
- Image normalization layer
- Five convolutional layers
- ELU activations
- Dropout regularization
- Fully connected steering prediction head

The architecture was adapted from NVIDIA's research paper:

> End-to-End Deep Learning for Self-Driving Cars

---

# Training Strategy

Several techniques were used to improve generalization:

## Data Augmentation

- Horizontal image flipping
- Steering angle inversion for flipped images

## Multi-Camera Training

Images from:

- Center camera
- Left camera
- Right camera

were incorporated into training.

Steering corrections were applied to left and right camera images to teach recovery behavior.

## Validation

- 80% training data
- 20% validation data

Validation performance was monitored throughout training to reduce overfitting.

---

# Regularization

To improve robustness:

- Dropout layer (25%)
- Validation split
- Limited number of training epochs

These measures helped reduce overfitting while maintaining autonomous driving performance.

---

# Results

The trained model successfully completed laps around the simulator track in autonomous mode.

The project demonstrates that a relatively compact CNN can learn steering behavior directly from raw camera imagery without explicit lane detection rules.

---

# Repository Structure

```text
model.py                 Training pipeline
drive.py                 Autonomous driving script
model.h5                 Trained neural network
writeup_report.pdf       Technical project report
video.mp4                Autonomous driving demonstration
```

---

# Technical Skills Demonstrated

- Deep Learning
- Convolutional Neural Networks
- Keras
- TensorFlow
- Computer Vision
- Data Augmentation
- Autonomous Driving
- End-to-End Learning
- Model Validation

---

# References

- NVIDIA: End-to-End Learning for Self-Driving Cars
- Udacity Self-Driving Car Engineer Nanodegree

---

# Related Self-Driving Car Projects

This repository is part of a larger autonomous driving portfolio:

- Finding Lane Lines
- Advanced Lane Finding
- Traffic Sign Classifier
- Behavioral Cloning
- Extended Kalman Filters
- Kidnapped Vehicle
- Highway Driving
- PID Controller

---

# Disclaimer

This repository is provided for educational and portfolio purposes.

Copyright © Sabrina Palis
