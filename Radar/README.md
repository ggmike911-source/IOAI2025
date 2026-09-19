# Radar

## Overview

This task uses six radar-derived heatmaps as input to generate a pixel-wise semantic label map. The channels can be divided into static and dynamic heatmaps: static channels emphasize reflections from stationary objects, while dynamic channels highlight changes caused by moving objects. The objective is to train a model that assigns a class label to every pixel in the image, including background regions.

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

The baseline model achieves a strong background accuracy, but it performs poorly on non-background pixels. This is a major issue because the scoring metric gives far more value to correctly predicting foreground objects than to predicting background. In other words, the model learns to classify empty space well while missing the actual targets that matter most.

## Thought Process

This task is very similar to the take-home task 1 which is pretty much the same question except instead of just having 2 classes (background and target), this time we have 5 classes. My first instinct is to use UNet in this problem because it is specializied for pixel segmentation tasks due to its encoder and decoder architecture, and skip connections. In addition, I also used UNet for the take-home task 1 and have an accuracy with 0.90+ so UNet should also perform well here.

### U-Net

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

The results show that the U-Net performs much better than the baseline at predicting non-background pixels, which leads to a substantially higher normalized score. This is exactly what we want for a segmentation task where foreground objects contribute much more to the final evaluation than background pixels.

One of the challenge I struggled with is that the input resolution is [50, 181], which is not divisible by $2^n$ for the two downsampling stages in the encoder. To address this, we resize the feature maps during decoding with bilinear interpolation so the output can be restored back to the original spatial dimensions. The two `F.interpolate()` calls are therefore essential to preserving alignment between the decoder output and the input image.

## Summary

This project is a five-class semantic segmentation task based on radar heatmaps. The baseline CNN is able to classify background pixels reasonably well, but it fails to capture most of the foreground structure, which is the key source of performance loss in this metric. By using a U-Net with skip connections, we are able to retain spatial detail while learning strong semantic features, resulting in a large improvement in non-background accuracy and overall score.



