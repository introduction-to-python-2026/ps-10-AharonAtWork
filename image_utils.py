from PIL import Image
import numpy as np
from scipy.signal import convolve2d

def load_image(file_path):
    """
    Loads a color image from the given file path and converts it to a NumPy array.

    Args:
        file_path (str): The path to the image file.

    Returns:
        np.array: The image as a NumPy array.
    """
    # Open the image file
    img = Image.open(file_path)
    # Convert the image to a NumPy array
    img_array = np.array(img)
    return img_array

# Test your function:
# First, let's download a sample image if you don't have one locally
image_url = "https://picsum.photos/600/400" # Changed to a different, accessible image URL
response = requests.get(image_url)
response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

# Directly write the content to a file
with open("sample_image.jpg", "wb") as f: # Using .jpg as the default for now, picsum.photos typically returns JPG
    f.write(response.content)

# Use the load_image function to load the sample image from the saved file
loaded_image_array = load_image("sample_image.jpg")

# Verify that the function works correctly and the output is as expected
print(f"Shape of the loaded image: {loaded_image_array.shape}")
print(f"Data type of the loaded image: {loaded_image_array.dtype}")


def edge_detection(image_array):
    """
    Performs edge detection on a 3-channel color image array.

    Args:
        image_array (np.array): A 3-channel color image as a NumPy array.

    Returns:
        np.array: The edge magnitude array.
    """
    # 1. Convert to grayscale
    # Average the three color channels (axis=2) to get a single channel grayscale image
    grayscale_image = np.mean(image_array, axis=2)

    # 2. Create filters
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

    # 3. Apply each filter using convolve
    # 'mode="constant"' with 'cval=0' ensures zero padding
    # 'origin=0' centers the kernel
    edgeY = convolve(grayscale_image, kernelY, mode='constant', cval=0)
    edgeX = convolve(grayscale_image, kernelX, mode='constant', cval=0)

    # 4. Compute edge magnitude
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)

    # 5. Return the edgeMAG array
    return edgeMAG

# Test your function:
# Ensure 'loaded_image_array' is available from the previous cell execution
if 'loaded_image_array' in locals():
    edge_magnitude_image = edge_detection(loaded_image_array)

    # Verify the result
    print(f"Shape of the edge magnitude image: {edge_magnitude_image.shape}")
    print(f"Data type of the edge magnitude image: {edge_magnitude_image.dtype}")
    # Display a small part of the array to visually inspect (optional)
    # print("Sample of edge magnitude values:")
    # print(edge_magnitude_image[100:105, 100:105])
else:
    print("Error: loaded_image_array not found. Please run the previous cell first.")
