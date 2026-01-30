import mediapipe as mp
import numpy as np

try:
    print("Attempting to create mp.Image...")
    frame_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    print("Success!")
except Exception as e:
    print(f"Failed: {e}")
    import traceback
    traceback.print_exc()
