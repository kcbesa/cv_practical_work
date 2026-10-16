---
title: Your Application
sdk: gradio
app_file: app.py
python_version: "3.12"
---
# Image Classification with Fine-Tuned MobileNetV3-Small

## Project Overview

This project implements image classification using a selected four-class image dataset:

- Bikes
- Cat
- Cars
- Dog

A CNN trained from scratch was compared with a pretrained MobileNetV3-Small model using transfer learning. The MobileNetV3-Small model was first trained with its feature extractor frozen and was subsequently fine-tuned by unfreezing the final feature block.

The fine-tuned MobileNetV3-Small model was exported as a PyTorch `state_dict` and deployed as an interactive Gradio web application on Hugging Face Spaces (https://huggingface.co/spaces/kbesa/mobilenet-image-classifier). The application accepts an uploaded image and returns the top-3 predicted classes together with their probabilities.

---

## Classes

The dataset was loaded using PyTorch's `ImageFolder`. The folder names and numerical labels were:

| Model Output | Dataset Folder | Class |
|---:|---|---|
| 0 | `class1` | Bikes |
| 1 | `class2` | Cat |
| 2 | `class3` | Cars |
| 3 | `class4` | Dog |

The explicit class mapping used by the application is:
## Inference time
The average inference time is 11.65 ms. The results indicate that the model performs inference quickly under the tested conditions, making it suitable for an interactive educational application. However, inference time measures the model's prediction process and does not necessarily represent the total time users experience, which may also include image uploading, preprocessing, server response and network latency.
 For real deployment changes, I would prioritise validating the model on realistic, independent data, protecting uploaded images, verifying the dataset's exact licence and measuring the real response time of the hosted application.

```python
classes = ["bikes", "cat", "cars", "dog"]
