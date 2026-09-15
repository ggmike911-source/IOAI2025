# Chicken Counting: Density Map Prediction

## Overview

The objective of this task is to generate density maps for chicken images. The input image shape is [3, 720, 1280], meaning there are 3 RGB channels and the image resolution is 720 (height) x 1280 (width). The output density map is constrained to [1, 180, 320].

> This task is a continuation of the take-home weather forecasting task, though it is not closely related to the other day 1 tasks.

---

## Problem Setup

- Input dimensions: [3, 720, 1280]
- Output density map dimensions: [1, 180, 320]
- Goal: estimate a spatial density map that reflects the number of chickens in each image.

---

## Baseline Information

| Optimizer | Initial Learning Rate | Gamma | Weight Decay | Total Epochs | Batch Size | Train Time | Accuracy |
|:---------:|:---------------------:|:-----:|:------------:|:------------:|:----------:|:----------:|:--------:|
| Adam | 1e-4 (0.0001) | 1e-5 (0.00001) | 1e-4 (0.0001) | 20 | 8 | 13 minutes | 0.69020 |

```python
def __init__(self, in_channels=3):
    super(FeatureExtraction, self).__init__()
    self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=2, dilation=2)
    self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2)
    self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=2, dilation=2)
    self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=2, dilation=2)
    self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    # The two MaxPool2d layers reduce the image size from 720 x 1280 to 180 x 320.
```

### Why this architecture

The two max-pooling layers reduce the spatial size as follows:

- 720 x 1280 -> 360 x 640
- 360 x 640 -> 180 x 320

This matches the required output dimensions while retaining the important spatial features needed for density estimation.

---

## Thought Process

At the competition, the initial approach was intentionally simple. The most direct way to improve model accuracy is typically to increase the depth of the network, increase the number of feature channels between layers, or tune hyperparameters. At the same time, the training runtime had to remain under 20 minutes, so the model could not be overly complex. There were only 100 training samples, which also made overfitting a significant concern.

### Initial Approach

| Optimizer | Initial Learning Rate | Gamma | Weight Decay | Total Epochs | Batch Size | Train Time | Accuracy |
|:---------:|:---------------------:|:-----:|:------------:|:------------:|:----------:|:----------:|:--------:|
| Adam | 1e-4 (0.0001) | 1e-4 (0.0001) | 5e-4 (0.0005) | 20 | 8 | 13 minutes | 0.69020 |

```python
def __init__(self, in_channels=3):
    super(FeatureExtraction, self).__init__()
    self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=2, dilation=2)
    self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=2, dilation=2)
    self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=2, dilation=2)
    self.conv4 = nn.Conv2d(256, 128, kernel_size=3, padding=2, dilation=2)
    self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
```

This version increased the number of channels and widened the feature extraction path, but it still remained constrained by the runtime and dataset size.

---

## Summary

The baseline model demonstrates a practical trade-off between model capacity and efficiency. It keeps training time under the competition limit while producing a density map at the expected reduced spatial resolution.
