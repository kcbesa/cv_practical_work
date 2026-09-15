# Image Classification with Fine-Tuned MobileNetV3-Small

## Project Overview

This project implements image classification using a selected four-class image dataset:

- Bikes
- Cat
- Cars
- Dog

A CNN trained from scratch was compared with a pretrained MobileNetV3-Small model using transfer learning. The MobileNetV3-Small model was first trained with its feature extractor frozen and was subsequently fine-tuned by unfreezing the final feature block.

The fine-tuned MobileNetV3-Small model was exported as a PyTorch `state_dict` and deployed as an interactive Gradio web application on Hugging Face Spaces. The application accepts an uploaded image and returns the top-3 predicted classes together with their probabilities.

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

```python
classes = ["bikes", "cat", "cars", "dog"]