import cv2
import numpy as np
from PIL import Image
import matplotlib
import skimage

print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)
print("Pillow:", Image.__version__)
print("Matplotlib:", matplotlib.__version__)
print("scikit-image:", skimage.__version__)

# Create a simple test image and run a basic processing operation
img = np.zeros((100, 100, 3), dtype=np.uint8)
img[:, :] = (60, 160, 255)  # BGR fill
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)

pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
pil_img.save("sample_output.png")

print("\nAll image processing libraries working correctly.")
print("Sample image saved to sample_output.png")
