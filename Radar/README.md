# Radar

## Overview

We are given 6 heatmaps by processing the radar data. There are static and dynamic heatmaps which emphasizes reflections from stationary objects and highlights changes caused by moving objects. The objective of this task is to train a model that can predict the semantic label map based on the 6 heatmaps. In another word, we have to assign a label to every pixel of an image (could be background).

---

## Problem Setup

- Input Dimension: [6, 50, 181]
- Output Dimension: [50, 181]
- Goal: Train a model to predict the semantic label map

---

## Baseline Information

| Optimizer | Initial Learning Rate | Total Epochs | Batch Size | Train Time | Accuracy |
|:---------:|:---------------------:|:------------:|:----------:|:----------:|:--------:|
| Adam | 0.001 | 40 | 4 | 11 minutes | 0.63858 |

### Score Details

| Normalized Score | Background Accuracy | Non-Background Accuracy |
|:----------------:|:-------------------:|:-----------------------:|
| 0.638582 | 99.81% | 33.82% |

```python
def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, padding=1)  
    self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)  
    self.conv3 = nn.Conv2d(in_channels=32, out_channels=5, kernel_size=3, padding=1)

    self.relu = nn.ReLU()
```

The baseline scored around 0.64. As we can see in the score details, the baseline did not perform well at predicting non-background pixels which contributes to the majority of the score because 1 background pixel is 1 score and 1 target pixel is 50 score. 

## Thought Process

This task is very similar to the take-home task 1 which is pretty much the same question except instead of just having 2 classes (background and target), this time we have 5 classes. My first instinct is to use UNet in this problem because it is specializied for pixel segmentation tasks due to its encoder and decoder architecture, and skip connections. In addition, I also used UNet for the take-home task 1 and have an accuracy with 0.90+ so UNet should also perform well here.

### UNet

| Optimizer | Initial Learning Rate | Total Epochs | Batch Size | Train Time | Accuracy |
|:---------:|:---------------------:|:------------:|:----------:|:----------:|:--------:|
| Adam | 0.001 | 40 | 4 | 7 minutes | 0.90998 |

### Score Details

| Normalized Score | Background Accuracy | Non-Background Accuracy |
|:----------------:|:-------------------:|:-----------------------:|
| 0.909975 | 99.73% | 83.70% |

```python
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
```

```python
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Encoder
        self.enc1 = DoubleConv(6, 64)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = DoubleConv(128, 256)

        # Decoder
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128) 

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)   

        self.final_conv = nn.Conv2d(64, 5, kernel_size=1)

    def forward(self, x):
        # Encoder path (save skip connections)
        s1 = self.enc1(x)
        p1 = self.pool1(s1)

        s2 = self.enc2(p1)
        p2 = self.pool2(s2)

        # Bottleneck
        b = self.bottleneck(p2)

        # Decoder path (concatenate skip connections along channel dim=1)
        d2 = self.up2(b)
        d2 = F.interpolate(d2, size=s2.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, s2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = F.interpolate(d1, size=s1.shape[2:], mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, s1], dim=1)
        d1 = self.dec1(d1)

        return self.final_conv(d1)
```

As we can see in the score details, UNet did a much better job at predicting non-background pixels and therefore got a higher score. It is worth noting that the resolution of the heatmaps are [50, 181] which does not fit the general requirement of UNet where the input width and height have to be divisible by 2 ^ n, where n is the number of max-pooling layers in the encoding (2 ^ 2 in our case). Therefore, we have to make some adjustments to the tensor during the decoding part. Hence, we have to we need the 2 ```F.interpolate() ``` to resize our tensor back to the dimension of our inputs.

