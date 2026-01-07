from PIL import Image
import numpy as np
from scipy.signal import convolve2d

def load_image(file_path):
    img = Image.open(file_path)
    return np.array(img)

def edge_detection(image_array):
    # 1. Grayscale
    grayscale_image = np.mean(image_array, axis=2)

    # 2. Sobel Kernels
    kernelY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    kernelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    # 3. Convolution (Notice: convolve2d)
    edgeY = convolve2d(grayscale_image, kernelY, mode='same', boundary='fill', fillvalue=0)
    edgeX = convolve2d(grayscale_image, kernelX, mode='same', boundary='fill', fillvalue=0)

    # 4. Magnitude
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)
    return edgeMAG
