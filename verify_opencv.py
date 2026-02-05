import cv2
import numpy as np

print(f"OpenCV version: {cv2.__version__}")
try:
    # Try to create a window (in a non-blocking way if possible, or just check if the function exists and is callable)
    cv2.namedWindow("Test Window", cv2.WINDOW_NORMAL)
    print("SUCCESS: cv2.namedWindow is implemented.")
    cv2.destroyAllWindows()
except Exception as e:
    print(f"FAILURE: {e}")
