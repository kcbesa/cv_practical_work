import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pandas as pd

# --- Utility functions ---
def threshold_otsu(image):
    _, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask

def psnr(original, restored):
    mse = np.mean((original.astype(np.float32) - restored.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def unsharp_mask(original, blurred, amount):
    sharpened = original + amount * (original - blurred)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

# --- Operations ---
def op_denoise(img, ksize):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    noisy = gray + np.random.normal(0, 15, gray.shape)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    restored = cv2.medianBlur(noisy, ksize)
    return restored

def op_sharpen(img, ksize):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    sharpened = unsharp_mask(gray, blurred, 1.0)
    return sharpened

def op_edges(img, ksize):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 100)
    return edges

def op_morphology(img, ksize):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = threshold_otsu(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel)
    return mask_closed

# --- Main CLI ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image processing tool")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--op", required=True, choices=["denoise", "sharpen", "edges", "morphology"], help="Operation")
    parser.add_argument("--ksize", type=int, default=3, help="Kernel size")
    parser.add_argument("--output", default="output.jpg", help="Output file path")
    args = parser.parse_args()

    img = cv2.imread(args.input)

    if args.op == "denoise":
        result = op_denoise(img, args.ksize)
    elif args.op == "sharpen":
        result = op_sharpen(img, args.ksize)
    elif args.op == "edges":
        result = op_edges(img, args.ksize)
    elif args.op == "morphology":
        result = op_morphology(img, args.ksize)

    cv2.imwrite(args.output, result)
    print(f"Operation {args.op} complete. Result saved to {args.output}")
