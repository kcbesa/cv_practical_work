import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import time


###################################
# Configuration
###################################

# Explicit mapping: class1=0 → bikes, class2=1 → cat, class3=2 → cars, class4=3 → dog
classes = ["bikes", "cat", "cars", "dog"]

device = torch.device("cpu")


###################################
# Load MobileNetV3-Small architecture
###################################

model = models.mobilenet_v3_small(weights=None)

model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    len(classes)
)

model.load_state_dict(
    torch.load(
        "mobilenet_finetuned.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()


###################################
# Image preprocessing
###################################

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225)
    )
])


###################################
# Prediction function
###################################

def predict(image):
    image = image.convert("RGB")

    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    top_probs, top_indices = torch.topk(
        probabilities,
        k=3
    )

    results = {
        classes[idx.item()]: float(prob)
        for prob, idx in zip(top_probs, top_indices)
    }

    return results

###################################
# Benchmark function
###################################
def benchmark_model():
    model.eval()

    # Create a dummy 224x224 RGB image tensor
    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    # Warm-up runs
    for _ in range(10):
        with torch.no_grad():
            model(dummy_input)

    # Measure 50 inference runs
    times = []

    for _ in range(50):
        start = time.perf_counter()

        with torch.no_grad():
            model(dummy_input)

        end = time.perf_counter()

        times.append((end - start) * 1000)

    average_time = sum(times) / len(times)
    minimum_time = min(times)
    maximum_time = max(times)

    return (
        f"Average inference time: {average_time:.2f} ms\n"
        f"Minimum: {minimum_time:.2f} ms\n"
        f"Maximum: {maximum_time:.2f} ms"
    )

###################################
# Gradio interface
###################################

with gr.Blocks() as demo:

    gr.Markdown("# Image Classifier")

    image_input = gr.Image(
        type="pil",
        label="Upload an Image"
    )

    predict_button = gr.Button("Classify Image")

    prediction_output = gr.Label(
        num_top_classes=3,
        label="Top-3 Predictions"
    )

    predict_button.click(
        fn=predict,
        inputs=image_input,
        outputs=prediction_output
    )

    gr.Markdown("## CPU Inference Benchmark")

    benchmark_button = gr.Button(
        "Benchmark CPU Inference"
    )

    benchmark_output = gr.Textbox(
        label="CPU Inference Results"
    )

    benchmark_button.click(
        fn=benchmark_model,
        inputs=None,
        outputs=benchmark_output
    )


###################################
# Launch
###################################

if __name__ == "__main__":
    demo.launch()