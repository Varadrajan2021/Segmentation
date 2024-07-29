# Segmentation

# Image Segmentation Application

This application performs image segmentation using OpenCV to detect and extract specific regions from an image. The extracted segments are saved as individual images, and a JSON file containing the hierarchical structure of the segments is generated.


### Code Explanation

The segmentation script performs the following steps:

1. **Read the Image**: The image is read from the file path.
    ```python
    image = cv2.imread(file_path)
    ```
    <img width="311" alt="image" src="https://github.com/user-attachments/assets/a8ac507a-eb76-4fd4-8ad5-172d395694f0">


2. **Convert to Grayscale**: The image is converted to a grayscale image.
    ```python
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ```

3. **Apply Gaussian Blur**: Gaussian blur is applied to the grayscale image to reduce noise.
    ```python
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    ```

4. **Apply Adaptive Thresholding**: Adaptive thresholding is applied to the blurred image to create a binary image.
    ```python
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 2)
    ```

5. **Detect Lines**: Horizontal and vertical lines are detected using morphological operations.
    ```python
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    ```

6. **Combine Lines and Apply Morphological Operations**: The horizontal and vertical lines are combined, and morphological operations are applied to close gaps.
    ```python
    lines = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(lines, cv2.MORPH_CLOSE, kernel, iterations=2)
    closed = cv2.dilate(closed, kernel, iterations=1)
    closed = cv2.erode(closed, kernel, iterations=1)
    ```

7. **Find Contours**: Contours are found in the processed image.
    ```python
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ```

8. **Filter and Extract Segments**: Contours are filtered based on aspect ratio and size, and the segments are extracted and saved as individual images.
    ```python
    for idx, contour in enumerate(contours):
        if is_block(contour):
            x, y, w, h = cv2.boundingRect(contour)
            # Ensure the bounding box is within the image dimensions
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
    ```

9. **Build Hierarchical Structure**: A hierarchical structure of the segments is built, excluding child segments from parent segments.
    ```python
    def exclude_child_segments(parent_img, children):
        mask = np.zeros(parent_img.shape[:2], dtype=np.uint8)
        for child in children:
            x, y, w, h = child['BoundingBox']
            mask[y:y + h, x:x + w] = 255
        inverted_mask = cv2.bitwise_not(mask)
        result = cv2.bitwise_and(parent_img, parent_img, mask=inverted_mask)
        return result        

    def is_child(parent_bbox, child_bbox):
        px, py, pw, ph = parent_bbox
        cx, cy, cw, ch = child_bbox
        return px < cx < px + pw and py < cy < py + ph and cx + cw < px + pw and cy + ch < py + ph

    def build_hierarchy(segments):
        hierarchy = []
        for parent in segments:
            children = [child for child in segments if is_child(parent['BoundingBox'], child['BoundingBox'])]
            if children:
                parent['Children'] = children
            hierarchy.append(parent)
        return hierarchy

    hierarchical_segments = build_hierarchy(segments)
    ```

10. **Save JSON Output**: The hierarchical segments are saved to a JSON file.
    ```python
    output_json_path = os.path.join(app.config['OUTPUT_FOLDER'], 'hierarchical_segments.json')
    with open(output_json_path, 'w') as f:
        json.dump(hierarchical_segments, f, indent=4)
    return output_json_path, segments
    ```

## Output

The application generates the following output:

1. **Segmented Images**: Individual images for each detected segment are saved in the output folder.
2. **JSON File**: A JSON file containing the hierarchical structure of the segments is generated.
