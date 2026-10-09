---
title: Image Segmentation Lab
emoji: 🖼️
colorFrom: blue
colorTo: indigo
sdk: gradio
python_version: "3.10"
app_file: app.py
---

# Image Segmentation Lab

Upload an image and select Otsu thresholding, adaptive thresholding, watershed or pretrained DeepLabV3-ResNet50. The output is a segmentation overlay; watershed also reports the number of segmented foreground regions.

DeepLabV3 is loaded only when first selected. Its pretrained weights are downloaded on first use. This Space is designed for CPU Basic (free), so initial inference and model loading may take longer than classical image processing.
