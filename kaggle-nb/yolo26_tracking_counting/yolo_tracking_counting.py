import os
os.system("pip install ultralytics")

import cv2
import time
import torch
import numpy as np
from ultralytics import YOLO

def process_video(input_path, output_path, model_name="yolo26n-seg.pt", tracker_type="bytetrack", max_duration_sec=60):
    print(f"Loading model: {model_name}...")
    try:
        model = YOLO(model_name)
    except:
        # Fallback if yolo26m-seg doesn't exist
        print(f"Could not load {model_name}, falling back to yolov8n-seg.pt")
        model = YOLO("yolov8n-seg.pt")
        
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
    if fps == 0:
        fps = 25 # Fallback
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = fps * max_duration_sec

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing video at {fps} FPS with {tracker_type} tracker. Stopping after {max_duration_sec} seconds ({max_frames} frames).")

    frame_count = 0
    prev_time = time.time()
    
    # Define counting region (based on code/region/region_data.json)
    region_pts = np.array([
        [238, 291],
        [719, 103],
        [1891, 471],
        [1852, 918]
    ], np.int32)
    region_pts = region_pts.reshape((-1, 1, 2))
    
    track_states = {}
    in_count = 0
    out_count = 0
    track_history = {}
    max_history = 50 # 2 seconds at 25 fps

    while cap.isOpened() and frame_count < max_frames:
        success, frame = cap.read()
        if not success:
            break

        # predict only class 0 (person) and track
        tracker_yaml = "botsort.yaml" if tracker_type == "botsort" else "bytetrack.yaml"
        results = model.track(frame, persist=True, tracker=tracker_yaml, conf=0.25, classes=[0], verbose=False, device=device_to_use)
        
        annotated_frame = frame.copy()
        
        # Draw the counting region
        cv2.polylines(annotated_frame, [region_pts], isClosed=True, color=(0, 255, 255), thickness=2)
        overlay_region = annotated_frame.copy()
        cv2.fillPoly(overlay_region, [region_pts], (0, 255, 255))
        cv2.addWeighted(overlay_region, 0.2, annotated_frame, 0.8, 0, annotated_frame)
        
        prof_color = (255, 144, 30) # Dodger Blue in BGR
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                
                # Update and draw tracking history (trail)
                if track_id not in track_history:
                    track_history[track_id] = []
                track_history[track_id].append((cx, cy))
                if len(track_history[track_id]) > max_history:
                    track_history[track_id].pop(0)
                    
                pts = np.array(track_history[track_id], np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [pts], isClosed=False, color=(255, 0, 255), thickness=2)
                
                # IN/OUT counting logic
                inside_val = cv2.pointPolygonTest(region_pts, (cx, cy), False)
                is_inside = (inside_val >= 0)
                
                if track_id not in track_states:
                    track_states[track_id] = is_inside
                else:
                    prev_inside = track_states[track_id]
                    if not prev_inside and is_inside:
                        in_count += 1
                    elif prev_inside and not is_inside:
                        out_count += 1
                    track_states[track_id] = is_inside
                
                # Draw thin custom box
                color = (0, 255, 0) if is_inside else prof_color
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(annotated_frame, (cx, cy), 4, color, -1)
                
                # Draw small professional label
                font = cv2.FONT_HERSHEY_SIMPLEX
                text = f"ID: {track_id}"
                text_size = cv2.getTextSize(text, font, 0.4, 1)[0]
                cv2.rectangle(annotated_frame, (x1, y1 - text_size[1] - 4), (x1 + text_size[0] + 4, y1), color, -1)
                cv2.putText(annotated_frame, text, (x1 + 2, y1 - 2), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
        curr_time = time.time()
        inference_fps = 1 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time
        
        # Professional overlay
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (20, 20), (450, 270), (0, 0, 0), -1)
        alpha = 0.7
        annotated_frame = cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(annotated_frame, "INTELLIGENT PEOPLE COUNTING", (40, 50), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(annotated_frame, (40, 65), (430, 65), (0, 255, 255), 2)
        
        cv2.putText(annotated_frame, f"Model   : {model_name}", (40, 95), font, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"Tracker : {tracker_type.upper()}", (40, 125), font, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"FPS     : {inference_fps:.1f}", (40, 155), font, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        
        current_people = len(results[0].boxes) if results[0].boxes is not None else 0
        cv2.putText(annotated_frame, f"Current : {current_people}", (40, 185), font, 0.7, (0, 165, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"IN Count: {in_count}", (40, 215), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"OUT Count: {out_count}", (40, 245), font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        out.write(annotated_frame)
        frame_count += 1
        
        if frame_count % fps == 0:
            print(f"[{tracker_type}] Processed {frame_count // fps} / {max_duration_sec} seconds... IN: {in_count}, OUT: {out_count}")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Finished {tracker_type}! Output saved to: {output_path}")

if __name__ == "__main__":
    import glob
    video_files = glob.glob("/kaggle/input/**/*.mp4", recursive=True) + glob.glob("/kaggle/input/**/*.avi", recursive=True)
    
    # We also check local working directory in case we are testing locally or video is nearby
    if not video_files:
        video_files = glob.glob("*.mp4") + glob.glob("*.avi")

    if not video_files:
        print("Error: No video files found!")
    else:
        print(f"Found videos: {video_files}")
        INPUT_VIDEO = video_files[0]
        
        # Check if model exists locally, else rely on ultralytics download
        model_file = "yolo26n-seg.pt" if os.path.exists("yolo26n-seg.pt") else "yolo11n-seg.pt"
        
        # Run BotSORT
        OUTPUT_BOTSORT = "/kaggle/working/output_botsort.mp4" if os.path.exists("/kaggle/working") else "output_botsort.mp4"
        process_video(INPUT_VIDEO, OUTPUT_BOTSORT, model_name=model_file, tracker_type="botsort")
        
        # Run ByteTrack
        OUTPUT_BYTETRACK = "/kaggle/working/output_bytetrack.mp4" if os.path.exists("/kaggle/working") else "output_bytetrack.mp4"
        process_video(INPUT_VIDEO, OUTPUT_BYTETRACK, model_name=model_file, tracker_type="bytetrack")
