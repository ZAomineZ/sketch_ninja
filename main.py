import cv2
import numpy as np
from sklearn.cluster import KMeans
import sys


def get_dominant_color(image, k=4, image_processing_size=None):
    """
    Extract the dominant colors from an image using KMeans clustering.
    :param image: Input image in BGR format.
    :param k: Number of clusters to identify (default: 4).
    :param image_processing_size: Tuple (width, height) to resize the image for faster processing.
    :return: Array of dominant color centers.
    """
    if image_processing_size is not None:
        # Resize image for processing
        image = cv2.resize(image, image_processing_size, interpolation=cv2.INTER_AREA)

    # Flatten the image to a 2D array of pixels
    pixels = image.reshape(-1, 3)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixels)

    # Return the cluster centers (dominant colors)
    return kmeans.cluster_centers_


def create_sketch(image):
    """
    Convert an image to a pencil sketch.
    :param image: Input image in BGR format.
    :return: Sketch image in grayscale format.
    """
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Invert the grayscale image
    inverted_gray = cv2.bitwise_not(gray_image)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(inverted_gray, (21, 21), 0)

    # Invert the blurred image
    inverted_blur = cv2.bitwise_not(blurred)

    # Create the sketch by dividing the gray image by the inverted blur
    sketch = cv2.divide(gray_image, inverted_blur, scale=256.0)
    return sketch


def add_alpha_channel(image, alpha_mask):
    """
    Add an alpha channel to an image based on a mask.
    :param image: Input image in BGR format or grayscale.
    :param alpha_mask: Alpha mask for transparency (same dimensions as the input image).
    :return: Image with alpha channel (BGRA format).
    """
    # Ensure the input image is in BGR format (3 channels)
    if len(image.shape) == 2:  # Grayscale image
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    b, g, r = cv2.split(image)
    rgba = [b, g, r, alpha_mask]
    return cv2.merge(rgba, 4)


def main(image_file):
    # Extract file and directory information
    image_file_parts = image_file.split('/')
    file_name = image_file_parts.pop()
    directory = '/'.join(image_file_parts)

    file_base_name, file_extension = file_name.rsplit('.', 1)
    sketch_file = f"{file_base_name}-drawing.png"

    # Load the input image
    image = cv2.imread(image_file)
    if image is None:
        print(f"Error: Could not load image {image_file}.")
        sys.exit(1)

    # Extract dominant colors
    dominant_colors = get_dominant_color(image, k=4)
    dominant_colors = dominant_colors.flatten()
    dominant_colors = np.sort(dominant_colors)
    dominant_colors = list(filter(lambda color: color != 0.0, dominant_colors))

    if len(dominant_colors) > 0:
        # Create a pencil sketch
        sketch = create_sketch(image)

        # Save the sketch image
        success = cv2.imwrite(sketch_file, sketch)
        if not success:
            print(f"Error: Could not save sketch to {sketch_file}.")
            sys.exit(1)

        # Add transparency to the sketch
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, alpha_mask = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)

        sketch_with_alpha = add_alpha_channel(sketch, alpha_mask)

        # Save the sketch with transparency
        success = cv2.imwrite(sketch_file, sketch_with_alpha)
        if success:
            print(f"Sketch saved successfully: {sketch_file}")
        else:
            print(f"Error: Could not save transparent sketch to {sketch_file}.")
            sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <image_file>")
        sys.exit(1)

    input_image_file = sys.argv[1]
    main(input_image_file)
