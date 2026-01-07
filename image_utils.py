from PIL import Image
import numpy as np
from scipy.signal import convolve2d
import requests  # Added this for the image download

def load_image(file_path):
    """Loads a color image and converts it to a NumPy array."""
    img = Image.open(file_path)
    return np.array(img)

def edge_detection(image_array):
    """Performs Sobel edge detection on a 3-channel color image array."""
    # 1. Convert to grayscale (averaging R, G, and B)
    grayscale_image = np.mean(image_array, axis=2)

    # 2. Define Sobel Kernels
    kernelY = np.array([
        [ 1,  2,  1],
        [ 0,  0,  0],
        [-1, -2, -1]
    ])

    kernelX = np.array([
        [-1,  0,  1],
        [-2,  0,  2],
        [-1,  0,  1]
    ])

    # 3. Apply each filter using convolve2d (Fixed the name here!)
    # boundary='fill' with fillvalue=0 is the equivalent of zero-padding
    edgeY = convolve2d(grayscale_image, kernelY, mode='same', boundary='fill', fillvalue=0)
    edgeX = convolve2d(grayscale_image, kernelX, mode='same', boundary='fill', fillvalue=0)

    # 4. Compute edge magnitude: sqrt(Gx^2 + Gy^2)
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)

    return edgeMAG

# This block ensures the test code only runs if you run this file directly,
# not when main.py imports it.
if __name__ == "__main__":
    try:
        image_url = "https://picsum.photos/600/400"
        response = requests.get(image_url)
        with open("sample_image.jpg", "wb") as f:
            f.write(response.content)
        
        test_img = load_image("sample_image.jpg")
        edges = edge_detection(test_img)
        print(f"Success! Edge map shape: {edges.shape}")
    except Exception as e:
        print(f"An error occurred during testing: {e}")
