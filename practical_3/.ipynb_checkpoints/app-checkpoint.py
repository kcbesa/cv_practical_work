import cv2
import gradio as gr
import numpy as np
from ultralytics import YOLO



# CONFIGURATION

YOLO_MODEL_PATH = "yolov8n.pt"
HAAR_CASCADE_PATH = cv2.data.haarcascades + \
    "haarcascade_frontalface_default.xml"


# LOAD MODELS

print("Loading models...")

# Load YOLO
yolo_model = YOLO(YOLO_MODEL_PATH)

# Load Haar cascade
face_cascade = cv2.CascadeClassifier(
    HAAR_CASCADE_PATH
)

if face_cascade.empty():
    raise RuntimeError(
        "Could not load Haar cascade."
    )

print("Models loaded successfully.")


# HAAR FACE DETECTION

def detect_faces_haar(
    image,
    confidence
):
    """
    Detect frontal faces using the Viola-Jones
    Haar cascade.

    The confidence slider is mapped to the
    Haar minNeighbors parameter.
    """

    # Convert RGB image from Gradio to grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    # Map confidence slider to minNeighbors.
    #
    # Higher confidence -> higher minNeighbors
    # -> fewer but more reliable detections.
    min_neighbors = int(
        3 + confidence * 7
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=min_neighbors,
        minSize=(30, 30)
    )

    annotated = image.copy()

    for i, (x, y, w, h) in enumerate(faces):

        # Draw bounding box
        cv2.rectangle(
            annotated,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Label
        label = f"Face {i + 1}"

        cv2.putText(
            annotated,
            label,
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    return annotated


# YOLO OBJECT DETECTION

def detect_objects_yolo(
    image,
    confidence
):
    """
    Detect objects using pretrained YOLOv8n.
    """

    results = yolo_model.predict(
        source=image,
        conf=confidence,
        verbose=False
    )

    result = results[0]

    # YOLO's plot() returns an annotated image
    annotated = result.plot()

    # Convert BGR -> RGB for Gradio
    annotated = cv2.cvtColor(
        annotated,
        cv2.COLOR_BGR2RGB
    )

    return annotated



# MAIN DETECTION FUNCTION

def run_detection(
    image,
    backend,
    confidence
):
    """
    Main function called by the Gradio interface.
    """

    if image is None:
        return None

    if backend == "Haar faces":

        return detect_faces_haar(
            image,
            confidence
        )

    elif backend == "YOLO objects":

        return detect_objects_yolo(
            image,
            confidence
        )

    return image



# GRADIO INTERFACE

description = """
Upload an image and choose a computer-vision detector.

• Haar faces: detects frontal faces using the Viola-Jones
  Haar cascade.

• YOLO objects: detects common objects using pretrained
  YOLOv8n.

Use the confidence slider to control how selective the
detector is.
"""


with gr.Blocks(
    title="Vision Detection App"
) as app:

    gr.Markdown(
        "# 🔍 Vision Detection App"
    )

    gr.Markdown(
        description
    )

    with gr.Row():

        # 
        # INPUT
        # 

        with gr.Column():

            image_input = gr.Image(
                type="numpy",
                label="Upload an image"
            )

            backend = gr.Radio(
                choices=[
                    "Haar faces",
                    "YOLO objects"
                ],
                value="YOLO objects",
                label="Choose detection backend"
            )

            confidence = gr.Slider(
                minimum=0.10,
                maximum=0.90,
                value=0.25,
                step=0.05,
                label="Confidence / Detection threshold"
            )

            detect_button = gr.Button(
                "Run Detection",
                variant="primary"
            )

        # 
        # OUTPUT
        #

        with gr.Column():

            output_image = gr.Image(
                type="numpy",
                label="Annotated result"
            )


    #
    # BUTTON ACTION
    #

    detect_button.click(
        fn=run_detection,
        inputs=[
            image_input,
            backend,
            confidence
        ],
        outputs=output_image
    )


#
# LAUNCH APPLICATION
# 

if __name__ == "__main__":

    app.launch()