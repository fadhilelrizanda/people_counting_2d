import os
os.system("pip install ultralytics")

import cv2
import time
import torch
from ultralytics import YOLO

def process_video(input_path, output_path, model_name="yolo26n-seg.pt", max_duration_sec=60):
    print(f"Loading model: {model_name}...")
    model = YOLO(model_name)
    
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        if "P100" in device_name:
            print("Warning: P100 GPU detected. PyTorch 2.x dropped support for P100. Falling back to CPU.")
            device_to_use = 'cpu'
            device_name = "P100 (Fallback: CPU)"
        else:
            device_to_use = 'cuda:0'
    else:
        device_name = "CPU"
        device_to_use = 'cpu'

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open input video {input_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = fps * max_duration_sec

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing video at {fps} FPS. Stopping after {max_duration_sec} seconds ({max_frames} frames).")

    frame_count = 0
    prev_time = time.time()
    
    while cap.isOpened() and frame_count < max_frames:
        success, frame = cap.read()
        if not success:
            break

        # predict only class 0 (person)
        results = model.predict(frame, conf=0.25, classes=[0], verbose=False, device=device_to_use)
        annotated_frame = results[0].plot()
        
        curr_time = time.time()
        inference_fps = 1 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time
        
        # Professional overlay
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (20, 20), (450, 200), (0, 0, 0), -1)
        alpha = 0.7
        annotated_frame = cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(annotated_frame, "INTELLIGENT PEOPLE TRACKING", (40, 50), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(annotated_frame, (40, 65), (430, 65), (0, 255, 255), 2)
        
        cv2.putText(annotated_frame, f"Model : {model_name}", (40, 95), font, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"GPU   : {device_name}", (40, 125), font, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"FPS   : {inference_fps:.1f}", (40, 155), font, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        
        people_count = len(results[0].boxes)
        cv2.putText(annotated_frame, f"Count : {people_count}", (40, 185), font, 0.7, (0, 165, 255), 2, cv2.LINE_AA)

        out.write(annotated_frame)
        frame_count += 1
        
        if frame_count % fps == 0:
            print(f"Processed {frame_count // fps} / {max_duration_sec} seconds...")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Finished! Output saved to: {output_path}")

if __name__ == "__main__":
    import glob
    video_files = glob.glob("/kaggle/input/**/*.mp4", recursive=True) + glob.glob("/kaggle/input/**/*.avi", recursive=True)
    if not video_files:
        print("Error: No video files found in /kaggle/input!")
    else:
        print(f"Found videos: {video_files}")
        INPUT_VIDEO = video_files[0]
        OUTPUT_VIDEO = "/kaggle/working/output_masked.mp4"
        process_video(INPUT_VIDEO, OUTPUT_VIDEO)
