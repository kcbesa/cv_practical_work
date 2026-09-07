# Computer Vision Practical 3 — Object Detection & Vision Application

A practical computer vision project implementing and comparing classical and modern object detection techniques. The project progresses from **template matching** and the **Viola–Jones Haar cascade** to **YOLOv8n**, followed by a simple interactive **Gradio vision application**.

The project demonstrates the strengths and limitations of different approaches to visual detection under changes in viewpoint, scale, lighting, occlusion, and object appearance.

---

## 📌 Project Overview

This practical explores three generations of computer vision techniques:

1. **Template Matching**
   - Detects a previously cropped object or face by comparing image regions with a template.
   - Demonstrates the limitations of pixel-level matching when viewpoint or appearance changes.

2. **Viola–Jones Face Detection**
   - Uses Haar-like features and a pretrained cascade classifier.
   - Evaluates how `scaleFactor` and `minNeighbors` affect detection performance.
   - Processes both still images and video.

3. **YOLOv8n Object Detection**
   - Uses a pretrained YOLOv8n model for one-shot object detection.
   - Evaluates different confidence thresholds.
   - Demonstrates bounding boxes, class labels, and non-maximum suppression.
   - Compares processing speed and robustness against the Haar detector.

4. **Gradio Vision Application**
   - Provides a user-friendly web interface.
   - Allows users to upload an image.
   - Allows selection between Haar face detection and YOLO object detection.
   - Provides a confidence/detection threshold.
   - Displays an annotated image containing detection boxes and labels.

---

# 🎯 Objectives

The main objectives of this practical are to:

- Understand how template matching performs object localization.
- Investigate the limitations of template matching under viewpoint changes.
- Implement face detection using the Viola–Jones Haar cascade.
- Study the effect of detection parameters on Haar detection.
- Implement modern one-shot object detection using YOLOv8n.
- Investigate how confidence thresholds affect the number of detections.
- Understand the purpose of Non-Maximum Suppression (NMS).
- Compare the processing speed of Haar and YOLO on video.
- Compare the robustness of classical and modern detectors.
- Develop a simple interactive computer vision application using Gradio.

---

# 🧠 Methods

## 1. Template Matching

Template matching searches for regions in an image that are visually similar to a supplied template.

The project uses OpenCV's:

```python
cv2.matchTemplate()
```

with:

```python
cv2.TM_CCOEFF_NORMED
```

The highest correlation score is selected as the predicted location.

### Experiment

A face was cropped from a photograph containing three people and saved as a separate template image.

The template was then matched against:

- The original photograph.
- Another photograph containing the same people but with the target person's face viewed from a different angle.

### Example results

| Scene | Matching Score | Result |
|---|---:|---|
| Original scene | 0.9995 | Correct match |
| Changed scene | 0.4575 | Incorrect match |

The original image produced a very high score because the template and target region had a highly similar pixel arrangement.

The second image produced a substantially lower score because the change in viewing angle altered the appearance and relative arrangement of facial features.

### Limitation

Template matching does not inherently understand that two different appearances belong to the same object or person. It primarily compares local pixel patterns.

Consequently, changes in:

- viewpoint,
- rotation,
- scale,
- lighting,
- deformation,
- occlusion,

can significantly reduce the matching score.

Even when the match is poor, `matchTemplate()` still returns the location with the highest correlation. Therefore, the highest-scoring location is not necessarily a correct detection.

---

# 👤 2. Viola–Jones Face Detection

The second stage uses OpenCV's pretrained Haar cascade:

```text
haarcascade_frontalface_default.xml
```

The Viola–Jones approach detects objects using Haar-like features and a cascade of classifiers.

The detector was evaluated on:

- A group photograph.
- A street/group video.

### Important parameters

#### `scaleFactor`

Controls how much the image is reduced between detection scales.

Example values:

```python
1.05
1.10
1.30
```

A smaller value searches more scales and can detect faces of different sizes more reliably, but increases computation.

A larger value is faster but may miss faces at intermediate scales.

#### `minNeighbors`

Controls how many neighboring detections are required before a detection is accepted.

Example values:

```python
3
5
8
```

Lower values are more permissive and may produce more false positives.

Higher values are more conservative and may reduce false positives but can cause genuine faces to be missed.

### Typical trade-off

| Parameter | Lower value | Higher value |
|---|---|---|
| `scaleFactor` | More scale searches, slower | Fewer scale searches, faster |
| `minNeighbors` | More detections, potentially more false positives | Fewer detections, potentially more missed faces |

### Known limitations

The Haar face detector is primarily designed for frontal face detection and can struggle with:

- Side-profile faces.
- Strong head rotation.
- Partial occlusion.
- Very small faces.
- Poor lighting.
- Strong shadows.
- Unusual image scales.

These limitations arise because Haar-like features represent local intensity patterns and the classifier is trained around particular visual arrangements associated with frontal faces.

---

# 🤖 3. YOLOv8n Object Detection

The third stage uses a pretrained **YOLOv8n** model.

YOLO performs object detection in a single inference pipeline, predicting object classes, confidence scores, and bounding boxes.

The model is loaded using the Ultralytics package:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
```

YOLOv8n was selected because it is a lightweight version of the YOLO family and is suitable for experimentation on a CPU.

---

## YOLO Detection Tasks

YOLOv8n was evaluated on:

- The group photograph.
- The street video.
- An everyday-objects image.

For the group photograph, detection was restricted to the COCO `person` class:

```python
PERSON_CLASS = 0
```

For the everyday-objects image, the detector was allowed to detect all supported classes.

---

# 📊 Confidence Threshold Experiment

The confidence threshold was varied using:

```text
0.10
0.25
0.50
0.75
```

The threshold determines the minimum confidence required for a prediction to be retained.

### General expected behaviour

| Confidence | Expected effect |
|---:|---|
| 0.10 | More detections, including uncertain predictions |
| 0.25 | Balanced detection sensitivity |
| 0.50 | More selective detections |
| 0.75 | Highly selective; more possible missed detections |

As the confidence threshold increases, low-confidence detections are removed.

The actual detection counts are recorded experimentally rather than assumed.

---

# 🧹 Non-Maximum Suppression

YOLO can produce multiple bounding boxes around the same object.

Non-Maximum Suppression (NMS) reduces these duplicate predictions.

The general process is:

1. Generate candidate bounding boxes.
2. Rank boxes according to confidence.
3. Keep the highest-confidence box.
4. Calculate overlap between that box and other boxes using Intersection over Union (IoU).
5. Suppress boxes with excessive overlap.
6. Continue until the redundant boxes have been removed.

The result is generally one strong bounding box per detected object.

### Why NMS is necessary

Without NMS, a single person or object could have several overlapping bounding boxes.

NMS therefore improves the final detection output by removing redundant predictions.

---

# 🎥 Video Processing

Both detectors were evaluated on the same video.

For each frame:

1. The frame is read.
2. The detector processes the frame.
3. The processing time is measured.
4. Detection results are generated.
5. Bounding boxes and labels are drawn.
6. The annotated frame is written to an output video.

The average processing time is calculated using:

```python
time.perf_counter()
```

The benchmark measures detector processing rather than relying on the video's nominal FPS.

---

# ⚡ Haar vs YOLO Performance

The project compares the average processing time per frame for:

- Viola–Jones Haar face detection.
- YOLOv8n object detection.

The benchmark uses the same input video and CPU environment.

The comparison is reported using:

| Detector | Frames | Average ms/frame |
|---|---:|---:|
| Viola–Jones Haar | *Measured value* | *Measured value* |
| YOLOv8n | *Measured value* | *Measured value* |

> Replace the values above with the actual measurements produced by the benchmark script.

The comparison should be interpreted in the context of what each detector is designed to do. Haar is a specialized frontal-face detector, whereas YOLO is a general object detector capable of identifying many object classes.

---

# 🔍 Robustness Comparison

| Condition | Haar Cascade | YOLOv8n |
|---|---|---|
| Frontal faces | Good | Good |
| Side-angle faces | Limited | Generally more robust |
| Occlusion | Can struggle | Generally more robust |
| Lighting changes | Can struggle | Generally more robust |
| Small objects/faces | Can struggle | Generally better, depending on resolution |
| General objects | Face-specific | Multiple object classes |
| Computational cost | Low | Higher |

YOLO generally provides greater robustness because it is trained for modern object detection across varied visual conditions.

However, YOLO is computationally more demanding than the lightweight Haar cascade.

---

# 🖥️ 4. Gradio Vision Application

The final stage packages the detection methods into an interactive Gradio application.

The application is contained in:

```text
app.py
```

The user interface provides:

- Image upload.
- Detection backend selection.
- Confidence/detection threshold slider.
- Run Detection button.
- Annotated image output.

### Available backends

```text
Haar faces
YOLO objects
```

### Haar mode

Detects frontal faces using:

```text
haarcascade_frontalface_default.xml
```

### YOLO mode

Detects objects using:

```text
yolov8n.pt
```

The YOLO model can identify common COCO objects such as people, vehicles, animals, and everyday objects.

---

# 📁 Project Structure

A recommended project structure is:

```text
Computer-Vision-Practical-3/
│
├── app.py
├── yolov8n.pt
├── README.md
├── requirements.txt
│
├── images/
│   ├── group_photo.jpg
│   ├── cropped_face.jpg
│   ├── changed_scene.jpg
│   └── everyday_objects.jpg
│
├── video/
│   ├── yolo_video.mp4
│   └── yolo_annotated.mp4
│
├── results/
│   ├── template_matching/
│   ├── haar/
│   ├── yolo/
│   └── benchmark/
│
└── notebooks/
    └── practical_3.ipynb
```

Large media files do not necessarily need to be committed to GitHub. If the files are too large, use Git LFS or provide instructions for obtaining the data.

---

# ⚙️ Installation

## Requirements

The project requires:

- Python 3.x
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Ultralytics
- Gradio

Install the dependencies using:

```bash
pip install -r requirements.txt
```

Alternatively:

```bash
pip install opencv-python numpy pandas matplotlib ultralytics gradio
```

---

# 🐍 Running in a Conda Environment

An existing Conda environment can be used.

For example:

```bash
conda activate ml_env
```

Verify the Python interpreter:

```bash
where python
```

The path should point to the selected Conda environment.

For example:

```text
C:\Users\<username>\.conda\envs\ml_env\python.exe
```

Install the dependencies into that environment:

```bash
python -m pip install -r requirements.txt
```

---

# 🚀 Running the Gradio Application

Navigate to the directory containing `app.py`.

If the project is located on another Windows drive, use:

```bash
cd /d "D:\path\to\Computer-Vision-Practical-3"
```

Then activate the Conda environment:

```bash
conda activate ml_env
```

Run:

```bash
python app.py
```

The application will provide a local URL similar to:

```text
http://127.0.0.1:7860
```

Open this address in a web browser.

---

# 🧪 Example Usage

### YOLO object detection

1. Launch the application.
2. Upload an image.
3. Select **YOLO objects**.
4. Set the confidence threshold.
5. Click **Run Detection**.
6. Examine the annotated output.

### Haar face detection

1. Upload a group photograph.
2. Select **Haar faces**.
3. Adjust the detection threshold.
4. Click **Run Detection**.
5. Examine the detected face bounding boxes.

---

# 📦 `requirements.txt`

The following can be saved as `requirements.txt`:

```text
opencv-python
numpy
pandas
matplotlib
ultralytics
gradio
```

Install with:

```bash
python -m pip install -r requirements.txt
```

---

# 📈 Results

The practical produces several types of results.

## Template Matching

- Template matching score.
- Predicted location.
- Annotated detection image.
- Comparison between original and changed scenes.

Example:

```text
Original scene score: 0.9995
Changed scene score: 0.4575
```

---

## Haar Detection

The parameter study records detection behaviour for combinations of:

```text
scaleFactor = 1.05, 1.10, 1.30
minNeighbors = 3, 5, 8
```

For each configuration, record:

- Number of detections.
- Missed faces.
- False positives.
- Processing time where applicable.

---

## YOLO Detection

The confidence study records:

- Confidence threshold.
- Number of detections.

Example format:

| Confidence | Detections |
|---:|---:|
| 0.10 | *Measured* |
| 0.25 | *Measured* |
| 0.50 | *Measured* |
| 0.75 | *Measured* |

---

## Detector Benchmark

The final benchmark compares:

```text
Haar average processing time
vs.
YOLO average processing time
```

on the same video and hardware.

---

# 🧩 Key Findings

The experiments demonstrate that different computer vision methods have different strengths.

### Template Matching

Template matching is simple and effective when the target has a very similar appearance to the supplied template. However, it is sensitive to viewpoint, scale, rotation, illumination, and other appearance changes.

### Viola–Jones

The Haar cascade is lightweight and computationally efficient for frontal-face detection. Its main limitation is reduced robustness when faces are rotated, occluded, poorly illuminated, or very small.

### YOLOv8n

YOLOv8n provides general-purpose object detection and is considerably more flexible than the Haar cascade. It can detect multiple object classes and generally handles variations in appearance and viewpoint better, although this comes with increased computational cost.

### Gradio

The Gradio application demonstrates how computer vision algorithms can be packaged into an accessible user interface. Users can interact with the detectors without needing to understand the underlying Python implementation.

---

# ⚠️ Limitations

Several limitations should be considered:

- YOLOv8n performance depends on image resolution and object size.
- The pretrained YOLO model is limited to the object categories represented by its training dataset.
- Haar detection is strongly affected by face orientation and image quality.
- Template matching is sensitive to changes between the template and target.
- CPU-based YOLO inference is slower than lightweight classical detection methods.
- Detection counts alone do not fully describe detector accuracy.
- False positives and missed detections should ideally be evaluated against manually labelled ground truth.

---

# 🔐 Responsible Computer Vision

Computer vision systems should be developed and evaluated responsibly.

When working with photographs or videos containing people:

- Obtain appropriate consent before using identifiable individuals.
- Avoid collecting unnecessary personal information.
- Use images for legitimate educational purposes.
- Consider potential bias in detection performance.
- Do not assume that a detection is always correct.
- Clearly communicate the limitations of the detector.

The results from a computer vision model should be interpreted as predictions rather than guaranteed facts.

---

# 📝 Academic Context

This repository was developed as part of a computer vision practical exercise covering:

- Image processing.
- Feature and object matching.
- Classical object detection.
- Modern deep-learning-based object detection.
- Performance evaluation.
- Computer vision application development.

The project demonstrates the transition from manually designed visual features and classifiers toward learned object detection systems.

---

# 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| OpenCV | Image processing and Haar detection |
| NumPy | Numerical operations |
| Pandas | Results and benchmark tables |
| Matplotlib | Visualization |
| Ultralytics | YOLOv8n object detection |
| Gradio | Interactive web application |
| Conda | Environment management |

---

# 📚 Main Concepts Demonstrated

- Template Matching
- Normalized Cross-Correlation
- Viola–Jones Object Detection
- Haar-like Features
- Cascade Classifiers
- Scale Factor
- Minimum Neighbors
- One-shot Object Detection
- YOLO
- Confidence Threshold
- Bounding Boxes
- Intersection over Union (IoU)
- Non-Maximum Suppression (NMS)
- Video Object Detection
- Processing-Time Benchmarking
- Interactive Computer Vision Applications

---

# 📄 License

This repository is intended primarily for educational and academic use.

Individual pretrained models, datasets, and third-party libraries may have their own licenses and terms of use. Users should check the applicable licenses before redistributing them.

---

# 👨‍💻 Author

**Computer Vision Practical Project**

Developed as part of an academic computer vision course.

---

# ⭐ Summary

This project demonstrates a progression from simple image-based matching to modern object detection:

```text
Template Matching
       │
       ▼
Viola–Jones / Haar
       │
       ▼
YOLOv8n
       │
       ▼
Interactive Gradio Application
```

The experiments show that there is no single detector that is optimal for every situation. Template matching can be effective in controlled conditions, Haar cascades provide lightweight frontal-face detection, and YOLO provides more general and robust object detection at a higher computational cost.