import numpy as np
from PIL import Image
from skimage.filters import median
from skimage.morphology import ball
import matplotlib.pyplot as plt
import requests
from io import BytesIO

# Import functions from the local image_utils.py file
from image_utils import load_image, edge_detection

def main():
    # --- Image Loading ---
    # If you have your own image, replace this URL or ensure 'sample_image.jpg' exists
    image_url = "https://picsum.photos/600/400"
    try:
        response = requests.get(image_url)
        response.raise_for_status() # Raise an exception for bad status codes
        with open("sample_image.jpg", "wb") as f:
            f.write(response.content)
        original_image_path = "sample_image.jpg"
    except requests.exceptions.RequestException as e:
        print(f"Could not download sample image: {e}")
        print("Please ensure 'sample_image.jpg' is available locally or update the image_url.")
        return

    print(f"Loading image from {original_image_path}...")
    original_image_array = load_image(original_image_path)
    print(f"Original image shape: {original_image_array.shape}")

    # --- Noise Suppression (Median Filter) ---
    print("Applying median filter for noise suppression...")
    # The median filter can handle multi-channel images directly
    clean_image = median(original_image_array, ball(3))
    print("Median filter applied.")

    # --- Edge Detection ---
    print("Detecting edges...")
    edge_magnitude_array = edge_detection(clean_image)
    print("Edge detection complete.")

    # --- Binarization using a Threshold ---
    # Calculate a threshold based on the histogram (e.g., top 90th percentile)
    # You might need to adjust this threshold based on your image content
    threshold = np.percentile(edge_magnitude_array, 90)
    print(f"Using threshold for binarization: {threshold:.2f}")
    edge_binary = edge_magnitude_array > threshold

    # Convert boolean array to uint8 (0 or 255) for saving
    edge_binary_display = (edge_binary * 255).astype(np.uint8)

    # --- Save Edge-Detected Image ---
    output_filename = 'my_edges.png'
    edge_image_pil = Image.fromarray(edge_binary_display)
    edge_image_pil.save(output_filename)
    print(f"Binary edge image saved as '{output_filename}'")

    # Optional: Display images (if running in an environment with a display)
    # plt.figure(figsize=(15, 5))
    # plt.subplot(1, 3, 1)
    # plt.imshow(original_image_array)
    # plt.title("Original Image")
    # plt.axis('off')

    # plt.subplot(1, 3, 2)
    # plt.imshow(clean_image.astype(np.uint8))
    # plt.title("Noise-Suppressed Image")
    # plt.axis('off')

    # plt.subplot(1, 3, 3)
    # plt.imshow(edge_binary_display, cmap='gray')
    # plt.title("Binary Edge Image")
    # plt.axis('off')
    # plt.show()

if __name__ == "__main__":
    main()
