"""CPU-friendly image segmentation demo for Hugging Face Spaces."""

import os
import threading
import time

import cv2
import gradio as gr
import numpy as np
from PIL import Image

# Limit CPU oversubscription on small Spaces instances.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

MAX_SIDE = 768
_model = None
_weights = None
_model_lock = threading.Lock()


def resize_image(image: Image.Image) -> np.ndarray:
    """Convert an image to RGB and cap its dimensions for CPU/RAM usage."""
    image = image.convert("RGB")
    image.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.uint8)


def blend_mask(rgb: np.ndarray, mask: np.ndarray, color=(255, 105, 50)) -> np.ndarray:
    """Highlight selected pixels using a translucent overlay."""
    out = rgb.copy()
    if np.any(mask):
        colored = np.asarray(color, dtype=np.float32)
        out[mask] = (0.58 * rgb[mask] + 0.42 * colored).astype(np.uint8)
    return out


def otsu_segment(rgb: np.ndarray):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    threshold, mask = cv2.threshold(
        cv2.GaussianBlur(gray, (5, 5), 0),
        0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return blend_mask(rgb, mask > 0), f"Otsu threshold: {threshold:.0f}. White-thresholded pixels are highlighted."


def adaptive_segment(rgb: np.ndarray):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = cv2.adaptiveThreshold(
        cv2.GaussianBlur(gray, (5, 5), 0),
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        4,
    )
    return blend_mask(rgb, mask > 0), "Adaptive Gaussian thresholding: locally thresholded pixels are highlighted."


def watershed_segment(rgb: np.ndarray):
    """Separate thresholded foreground regions and count non-background labels."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Foreground is chosen as the less-common Otsu partition, suitable for many objects on a plain background.
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.count_nonzero(binary) > binary.size // 2:
        binary = cv2.bitwise_not(binary)

    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opened, kernel, iterations=3)
    distances = cv2.distanceTransform(opened, cv2.DIST_L2, 5)

    if distances.max() == 0:
        return rgb.copy(), 0, "No foreground regions detected. Try a higher-contrast image."

    _, sure_fg = cv2.threshold(distances, 0.40 * distances.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Distinct connected foreground markers become distinct watershed seeds.
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers.astype(np.int32) + 1  # background is marker 1
    markers[unknown > 0] = 0
    markers = cv2.watershed(bgr, markers)

    unique_ids = [int(i) for i in np.unique(markers) if i > 1]
    result = rgb.copy()
    palette = np.random.default_rng(42)
    for label in unique_ids:
        color = palette.integers(35, 256, size=3)
        m = markers == label
        result[m] = (0.45 * rgb[m] + 0.55 * color).astype(np.uint8)
    result[markers == -1] = (255, 30, 30)  # watershed boundaries

    count = len(unique_ids)
    return result, count, f"Watershed found {count} foreground region(s). Red lines show region boundaries; this is an estimate of object count."


def get_deeplab():
    """Lazy-load pretrained weights only when DeepLabV3 is requested."""
    global _model, _weights
    with _model_lock:
        if _model is None:
            import torch
            from torchvision.models.segmentation import (
                DeepLabV3_ResNet50_Weights,
                deeplabv3_resnet50,
            )
            torch.set_num_threads(2)
            weights = DeepLabV3_ResNet50_Weights.DEFAULT
            model = deeplabv3_resnet50(weights=weights).eval().cpu()
            _model, _weights = model, weights
        return _model, _weights


def deeplab_segment(rgb: np.ndarray):
    import torch
    import torch.nn.functional as F

    model, weights = get_deeplab()
    # Weight transforms handle resize, normalization, and batching requirements.
    input_tensor = weights.transforms()(Image.fromarray(rgb)).unsqueeze(0)
    with _model_lock, torch.inference_mode():
        logits = model(input_tensor)["out"]
        logits = F.interpolate(logits, size=rgb.shape[:2], mode="bilinear", align_corners=False)
        classes = logits.argmax(dim=1)[0].cpu().numpy()

    labels = weights.meta["categories"]
    rng = np.random.default_rng(11)
    palette = rng.integers(45, 255, size=(len(labels), 3), dtype=np.uint8)
    palette[0] = 0
    visible = classes != 0
    result = rgb.copy()
    result[visible] = (0.50 * rgb[visible] + 0.50 * palette[classes[visible]]).astype(np.uint8)
    found = [(labels[i], int(np.count_nonzero(classes == i))) for i in np.unique(classes) if i != 0]
    found.sort(key=lambda x: -x[1])
    summary = ", ".join(name for name, _ in found[:8]) or "no recognized foreground class"
    return result, f"DeepLabV3 semantic classes: {summary}. The same class has the same overlay colour."


def segment(image, method):
    if image is None:
        raise gr.Error("Upload an image first.")

    rgb = resize_image(image)
    start = time.perf_counter()
    try:
        if method == "Otsu":
            overlay, description = otsu_segment(rgb)
            count_text = "Not applicable"
        elif method == "Adaptive":
            overlay, description = adaptive_segment(rgb)
            count_text = "Not applicable"
        elif method == "Watershed":
            overlay, count, description = watershed_segment(rgb)
            count_text = str(count)
        elif method == "DeepLabV3":
            overlay, description = deeplab_segment(rgb)
            count_text = "Not applicable"
        else:
            raise gr.Error(f"Unknown segmentation method: {method}")
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(f"Segmentation failed: {exc}") from exc

    duration = time.perf_counter() - start
    return Image.fromarray(overlay), count_text, f"{description}\n\nProcessing time: {duration:.2f} s (Space CPU)."


with gr.Blocks(title="Image Segmentation Lab") as demo:
    gr.Markdown(
        "# Image Segmentation Lab\n"
        "Upload an image and compare classical segmentation with pretrained DeepLabV3 semantic segmentation.\n\n"
        "**CPU note:** DeepLabV3 runs on the free CPU tier. Inference may take several seconds "
        "(especially on the first use, when pretrained weights are downloaded)."
    )
    with gr.Row():
        with gr.Column():
            source = gr.Image(type="pil", label="Upload image", sources=["upload"])
            method = gr.Dropdown(
                choices=["Otsu", "Adaptive", "Watershed", "DeepLabV3"],
                value="Otsu", label="Segmentation method",
            )
            run = gr.Button("Segment image", variant="primary")
        with gr.Column():
            output = gr.Image(type="pil", label="Colour-coded segmentation overlay")
            object_count = gr.Textbox(label="Watershed object count", interactive=False)
            details = gr.Textbox(label="Results", lines=4, interactive=False)

    run.click(
        fn=segment, inputs=[source, method], outputs=[output, object_count, details],
        concurrency_limit=1,
    )

demo.queue(default_concurrency_limit=1)

if __name__ == "__main__":
    demo.launch()
