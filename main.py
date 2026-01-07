import image_utils
from PIL import Image
import numpy as np

# 1. Load your original image
# Make sure "my_image.jpg" is uploaded to your GitHub folder!
input_path = "my_image.jpg" 
original_array = image_utils.load_image(input_path)

# 2. Process the image using your utility function
edge_array = image_utils.edge_detection(original_array)

# 3. Save the result
# We convert the math array back into an actual image file
output_image = Image.fromarray(edge_array.astype(np.uint8))
output_image.save("edge_detected_result.jpg")

print("Image processed and saved successfully!")
