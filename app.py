import os
import json
import cv2
import numpy as np
import pandas as pd
# import pytesseract
from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
# from pytesseract import Output
import random

app = Flask(__name__)

# Configurations
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg','tif'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Image processing and hierarchical structure functions
def process_image(file_path):
    # Function to generate a random color
    def random_color():
        return [random.randint(0, 255) for _ in range(3)]

    # Function to determine if a contour is a block
    def is_block(contour):
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        area = w * h
        return aspect_ratio < 30 and h > 10

    # Read the image from the file path
    image = cv2.imread(file_path)
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 2)
    
    # Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    # Detect vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    
    # Dilate lines to ensure consistency
    horizontal_lines = cv2.dilate(horizontal_lines, np.ones((2, 2), np.uint8), iterations=2)
    vertical_lines = cv2.dilate(vertical_lines, np.ones((2, 2), np.uint8), iterations=2)
    
    # Combine horizontal and vertical lines
    lines = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(lines, cv2.MORPH_CLOSE, kernel, iterations=2)
    closed = cv2.dilate(closed, kernel, iterations=1)
    closed = cv2.erode(closed, kernel, iterations=1)
    
    # Find contours in the processed image
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Initialize list to hold segments
    segments = []
    image_height, image_width = image.shape[:2]
    for idx, contour in enumerate(contours):
        if is_block(contour):
            x, y, w, h = cv2.boundingRect(contour)
            x = max(0, x)
            y = max(0, y)
            w = min(image_width - x, w)
            h = min(image_height - y, h)
            color = random_color()
            area = w * h
            aspect_ratio = w / float(h)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            segment = image[y:y + h, x:x + w]
            segment_path = os.path.join(app.config['OUTPUT_FOLDER'], f'segment_{idx}.png')
            cv2.imwrite(segment_path, segment)
            segments.append({
                'Index': idx,
                'Color': color,
                'BoundingBox': [x, y, w, h],
                'Image': segment_path,
                'Width': w,
                'Height': h,
                'Area': area,
                'AspectRatio': aspect_ratio
            })
            
            
    # Function to exclude child segments from parent segments
    def exclude_child_segments(parent_img, children):
        mask = np.zeros(parent_img.shape[:2], dtype=np.uint8)
        for child in children:
            x, y, w, h = child['BoundingBox']
            mask[y:y + h, x:x + w] = 255
        inverted_mask = cv2.bitwise_not(mask)
        result = cv2.bitwise_and(parent_img, parent_img, mask=inverted_mask)
        return result        

    # Function to check if one bounding box is a child of another
    def is_child(parent_bbox, child_bbox):
        px, py, pw, ph = parent_bbox
        cx, cy, cw, ch = child_bbox
        return px < cx < px + pw and py < cy < py + ph and cx + cw < px + pw and cy + ch < py + ph

    # Function to build the hierarchical structure of segments
    def build_hierarchy(segments):
        hierarchy = []
        for parent in segments:
            children = [child for child in segments if is_child(parent['BoundingBox'], child['BoundingBox'])]
            if children:
                parent['Children'] = children
            hierarchy.append(parent)
        return hierarchy

    # Build the hierarchical segments
    hierarchical_segments = build_hierarchy(segments)
    output_json_path = os.path.join(app.config['OUTPUT_FOLDER'], 'hierarchical_segments.json')
    with open(output_json_path, 'w') as f:
        json.dump(hierarchical_segments, f, indent=4)
    return output_json_path, segments

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        json_path, segments = process_image(file_path)
        return jsonify({"message": "Processing complete", "json_path": json_path, "segments": segments})

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
