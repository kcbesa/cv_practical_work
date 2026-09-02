# 🖼️ Image Processing Command-Line Tool

A Python command-line utility for performing common image processing operations such as **denoising**, **sharpening**, **edge detection**, and **morphology-based object counting**.  
This tool wraps multiple practical tasks into a single script (`process.py`) so you can run them easily from the terminal.

---

## 🚀 Features
- **[Denoise](ca://s?q=Image_denoising_with_filters)**: Apply median filtering to reduce Gaussian or salt-and-pepper noise.
- **[Sharpen](ca://s?q=Unsharp_masking_for_image_sharpening)**: Use unsharp masking to enhance edges.
- **[Edges](ca://s?q=Canny_edge_detection_in_images)**: Detect edges with the Canny operator.
- **[Morphology](ca://s?q=Morphology_mask_cleaning_and_object_count)**: Threshold, clean, and count objects using morphological operations.

---

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/kcbesa/cv_practical_work/image-processing-tool.git
   cd image-processing-tool
2. Create virtual environment
   python -m venv venv
   venv\Scripts\activate   # Windows CMD
   source venv/bin/activate  # Linux/Mac
3. Install dependencies
   pip install -r requirements.txt
4. Usage
   python process.py --input <image_path> --op <operation> [--ksize N] [--min_area M] --output <output_path>



