import cv2
import numpy as np

def preprocess_frame(frame, imgsz=640):
    # Resize to imgsz x imgsz using bilinear interpolation
    h_orig, w_orig = frame.shape[:2]
    resized = cv2.resize(frame, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    
    # BGR to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Convert to float32 and scale to [0, 1]
    blob = rgb.astype(np.float32) / 255.0
    
    # Transpose to BCHW: (1, 3, imgsz, imgsz)
    blob = np.transpose(blob, (2, 0, 1))
    blob = np.expand_dims(blob, axis=0)
    
    return blob, h_orig, w_orig
