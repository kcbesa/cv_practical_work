######################################################
# GRADIO INFERENCE CODE
######################################################
import torch
import torch.nn as nn
from torchvision import models, transforms
import gradio as gr
from PIL import Image


# Path to the checkpoint downloaded from Colab.
CHECKPOINT_PATH = "MLR_model.pth"

# Load checkpoint on CPU, which is suitable for a basic
# Gradio app running without a dedicated GPU.
checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu"
)

# Recreate exactly the same ResNet-50 architecture used in training.
# weights=None prevents downloading ImageNet weights because
# the learned model parameters are loaded from saved checkpoint.
model = models.resnet50(weights=None)

# Recreated the same two-class output head.
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 2)
)

# Restore the fine-tuned parameters.
model.load_state_dict(checkpoint["model_state_dict"])

# Put the model in evaluation mode to disable dropout.
model.eval()

# Kept this order consistent with the training class mapping.
class_names = ["Parasitized", "Uninfected"]

# Apply the same resizing and normalization used in training.
# The values are read from the saved checkpoint to help keep
# inference preprocessing consistent with training.
transform = transforms.Compose([
    transforms.Resize((checkpoint["image_size"], checkpoint["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize(
        checkpoint["mean"],
        checkpoint["std"]
    )
])


def predict(img):
    """
    Predict the model's class probabilities for one cell image.

    img: PIL image supplied by Gradio.

    Returns a dictionary mapping class names to probabilities.
    These scores describe model output, not clinical certainty.
    """

    # Ensure the input has three color channels.
    img = img.convert("RGB")

    # Apply preprocessing and add a batch dimension.
    # Shape becomes [1, 3, 224, 224].
    tensor = transform(img).unsqueeze(0)

    # Prediction does not require gradients.
    with torch.inference_mode():

        # Get the two raw class scores.
        outputs = model(tensor)

        # Convert scores to probabilities that sum to 1.
        probabilities = torch.softmax(outputs, dim=1)[0]

    # Gradio's Label output can display these class scores.
    return {
        class_names[i]: float(probabilities[i])
        for i in range(len(class_names))
    }

 

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    title="Malaria Cell Classification",
    description=(
        "Educational classification of blood-smear cell images. "
        "This model is not a medical diagnostic tool."
    )
)

interface.launch()

