import glob
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from PIL import Image

def install_deps():
    deps = ["ultralytics", "transformers"]
    for dep in deps:
        if importlib.util.find_spec(dep) is None:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", dep])

install_deps()

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

REGION_FILE = Path(__file__).with_name("region_data.json")
DEFAULT_REGION = [
    [13, 1606],
    [779, 1773],
    [936, 1372],
    [177, 1278],
]
MODEL_CANDIDATES = [
    "yolo11x-pose.pt",
    "yolov8x-pose.pt",
]
DWELL_SECONDS_REQUIRED = 10.0


def load_region_points():
    for region_file in (REGION_FILE, Path.cwd() / "region_data.json"):
        if region_file.exists():
            with region_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"Loaded region file: {region_file}")
            region = data.get("region", DEFAULT_REGION)
            return np.array(region, np.int32).reshape((-1, 1, 2))
    print("Region file not found; using embedded selected region.")
    return np.array(DEFAULT_REGION, np.int32).reshape((-1, 1, 2))


def select_device():
    if not torch.cuda.is_available():
        return "cpu", "CPU"
    device_name = torch.cuda.get_device_name(0)
    if "P100" in device_name:
        return "cpu", "P100 fallback: CPU"
    return "cuda:0", device_name


def load_detection_model():
    last_error = None
    for model_name in MODEL_CANDIDATES:
        try:
            print(f"Loading pose detector: {model_name}")
            return YOLO(model_name), model_name
        except Exception as exc:
            last_error = exc
            print(f"Could not load {model_name}: {exc}")
    raise RuntimeError(f"Unable to load any detection model. Last error: {last_error}")


def load_depth_model(device_str):
    print("Loading Depth Anything model...")
    image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")
    depth_model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf")
    if "cuda" in device_str:
        depth_model = depth_model.to(device_str)
    return image_processor, depth_model


def draw_translucent_rect(frame, top_left, bottom_right, color, alpha):
    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_text(frame, text, origin, scale, color, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    cv2.putText(frame, text, origin, font, scale, color, thickness, cv2.LINE_AA)


def draw_metric(frame, label, value, x, y, accent):
    draw_text(frame, label.upper(), (x, y), 0.45, (164, 174, 193), 1)
    draw_text(frame, str(value), (x, y + 34), 0.95, accent, 2, cv2.FONT_HERSHEY_DUPLEX)


def draw_status_panel(
    frame,
    model_name,
    tracker_type,
    device_name,
    fps,
    elapsed_sec,
    interested_count,
    not_interested_count,
    active_tracks,
    active_in_region,
    dwell_seconds_required,
):
    panel_x, panel_y = 28, 28
    panel_w, panel_h = 520, 244
    border = (45, 212, 191)
    panel = (15, 23, 42)
    white = (248, 250, 252)
    muted = (203, 213, 225)
    green = (74, 222, 128)
    amber = (251, 191, 36)
    red = (248, 113, 113)

    draw_translucent_rect(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), panel, 0.84)
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (51, 65, 85), 1)
    cv2.line(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y), border, 4)

    draw_text(frame, "INTEREST ANALYTICS (POSE + DEPTH)", (panel_x + 22, panel_y + 38), 0.65, white, 1, cv2.FONT_HERSHEY_DUPLEX)
    draw_text(frame, "Dwell + Facing Booth + Depth Filter", (panel_x + 22, panel_y + 64), 0.45, muted, 1)

    draw_metric(frame, "Interested", interested_count, panel_x + 24, panel_y + 108, green)
    draw_metric(frame, "Not interested", not_interested_count, panel_x + 210, panel_y + 108, red)
    draw_metric(frame, "In region", active_in_region, panel_x + 410, panel_y + 108, amber)

    y0 = panel_y + 184
    draw_text(frame, f"Detector: {model_name} + DepthAnything", (panel_x + 24, y0), 0.48, muted, 1)
    draw_text(frame, f"Tracker: {tracker_type.upper()}  |  Active tracks: {active_tracks}", (panel_x + 24, y0 + 24), 0.48, muted, 1)
    draw_text(
        frame,
        f"Rule: > {dwell_seconds_required:.0f}s, facing front  |  Device: {device_name[:25]}  |  FPS: {fps:.1f}",
        (panel_x + 24, y0 + 48),
        0.48,
        muted,
        1,
    )


def process_video(input_path, output_path, tracker_type="botsort", max_duration_sec=120):
    device_to_use, device_name = select_device()
    model, model_name = load_detection_model()
    image_processor, depth_model = load_depth_model(device_to_use)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open input video {input_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = fps * max_duration_sec

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    region_pts = load_region_points()
    
    track_states = {}
    track_history = {}
    interested_count = 0
    not_interested_count = 0
    max_history = 50
    tracker_yaml = "botsort.yaml" if tracker_type == "botsort" else "bytetrack.yaml"
    prev_time = time.time()
    start_time = prev_time
    frame_count = 0

    print(f"Processing {input_path} with {model_name}, DepthAnything, {tracker_type}, {device_name}.")

    while cap.isOpened() and frame_count < max_frames:
        success, frame = cap.read()
        if not success:
            break

        # 1. Pose Tracking
        results = model.track(
            frame,
            persist=True,
            tracker=tracker_yaml,
            conf=0.25,
            iou=0.5,
            classes=[0],
            verbose=False,
            device=device_to_use,
        )

        annotated_frame = frame.copy()
        
        # Region overlay
        region_overlay = annotated_frame.copy()
        cv2.fillPoly(region_overlay, [region_pts], (45, 212, 191))
        cv2.addWeighted(region_overlay, 0.16, annotated_frame, 0.84, 0, annotated_frame)
        cv2.polylines(annotated_frame, [region_pts], isClosed=True, color=(45, 212, 191), thickness=7)

        # 2. Depth Estimation
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        inputs = image_processor(images=pil_img, return_tensors="pt")
        if "cuda" in device_to_use:
            inputs = {k: v.to(device_to_use) for k, v in inputs.items()}
            
        with torch.no_grad():
            depth_outputs = depth_model(**inputs)
            predicted_depth = depth_outputs.predicted_depth
            predicted_depth = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=(height, width),
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            depth_map = predicted_depth.cpu().numpy()

        # Calculate a median depth of the region to act as our "booth depth"
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [region_pts], 255)
        region_depths = depth_map[mask == 255]
        booth_depth = np.median(region_depths) if len(region_depths) > 0 else 0
        depth_tolerance = 5.0 # Relative depth tolerance

        active_tracks = 0
        active_in_region = 0
        
        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            keypoints = results[0].keypoints.data.cpu().numpy() if results[0].keypoints is not None else None
            
            active_tracks = len(track_ids)

            for i, (box, track_id) in enumerate(zip(boxes, track_ids)):
                x1, y1, x2, y2 = map(int, box)
                anchor = ((x1 + x2) // 2, y2)
                
                # Check 2D Region overlap
                in_2d_region = cv2.pointPolygonTest(region_pts, anchor, False) >= 0
                
                # Check 3D Depth
                person_depth = depth_map[min(y2, height-1), min((x1+x2)//2, width-1)]
                depth_valid = abs(person_depth - booth_depth) < depth_tolerance
                
                # Check Pose Orientation (Facing camera/booth)
                is_facing = False
                if keypoints is not None:
                    kps = keypoints[i] # Shape: (17, 3)
                    nose_conf = kps[0][2]
                    left_eye_conf = kps[1][2]
                    right_eye_conf = kps[2][2]
                    # Simple heuristic: if face features are highly visible, they are facing front
                    if nose_conf > 0.5 and (left_eye_conf > 0.5 or right_eye_conf > 0.5):
                        is_facing = True

                is_valid_interest = in_2d_region and depth_valid and is_facing

                if is_valid_interest:
                    active_in_region += 1

                state = track_states.setdefault(
                    int(track_id),
                    {
                        "inside": False,
                        "inside_since_frame": None,
                        "last_dwell_seconds": 0.0,
                        "classification": None,
                    },
                )

                if is_valid_interest:
                    if not state["inside"]:
                        state["inside_since_frame"] = frame_count
                    state["inside"] = True

                    inside_frames = frame_count - state["inside_since_frame"] + 1
                    dwell_seconds = inside_frames / fps
                    state["last_dwell_seconds"] = dwell_seconds
                    if state["classification"] is None and dwell_seconds > DWELL_SECONDS_REQUIRED:
                        interested_count += 1
                        state["classification"] = "interested"
                else:
                    dwell_seconds = state["last_dwell_seconds"]
                    if state["inside"] and state["classification"] is None:
                        not_interested_count += 1
                        state["classification"] = "not interested"
                    state["inside"] = False
                    state["inside_since_frame"] = None

                track_history.setdefault(track_id, []).append(anchor)
                if len(track_history[track_id]) > max_history:
                    track_history[track_id].pop(0)

                # Visuals
                color = (74, 222, 128) if state["classification"] == "interested" else (251, 191, 36) if is_valid_interest else (96, 165, 250)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(annotated_frame, anchor, 6, color, -1)
                
                # Draw skeleton if valid
                if is_facing and keypoints is not None:
                    kps = keypoints[i]
                    for kp in kps[:5]: # Face keypoints
                        if kp[2] > 0.5:
                            cv2.circle(annotated_frame, (int(kp[0]), int(kp[1])), 3, (0, 0, 255), -1)

                label = f"ID {track_id}"
                if state["classification"]: label += f" | {state['classification']}"
                elif is_valid_interest: label += f" | {dwell_seconds:.1f}s"
                elif not is_facing and in_2d_region: label += " | Not facing"
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                label_size, baseline = cv2.getTextSize(label, font, 0.48, 1)
                label_y = max(20, y1 - label_size[1] - baseline - 8)
                draw_translucent_rect(annotated_frame, (x1, label_y), (x1 + label_size[0] + 16, label_y + label_size[1] + baseline + 8), (15, 23, 42), 0.82)
                cv2.rectangle(annotated_frame, (x1, label_y), (x1 + label_size[0] + 16, label_y + label_size[1] + baseline + 8), color, 1)
                draw_text(annotated_frame, label, (x1 + 8, label_y + label_size[1] + 2), 0.48, (248, 250, 252), 1, font)

        curr_time = time.time()
        inference_fps = 1 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time
        elapsed_sec = curr_time - start_time
        draw_status_panel(
            annotated_frame, model_name, tracker_type, device_name,
            inference_fps, elapsed_sec, interested_count, not_interested_count,
            active_tracks, active_in_region, DWELL_SECONDS_REQUIRED
        )

        out.write(annotated_frame)
        frame_count += 1

        if frame_count % fps == 0:
            print(f"[{tracker_type}] {frame_count // fps}/{max_duration_sec}s | in_region={active_in_region} interested={interested_count} not_interested={not_interested_count}")

    for state in track_states.values():
        if state["inside"] and state["classification"] is None:
            if state["last_dwell_seconds"] > DWELL_SECONDS_REQUIRED:
                interested_count += 1
                state["classification"] = "interested"
            else:
                not_interested_count += 1
                state["classification"] = "not interested"

    cap.release()
    out.release()
    print(f"Final counts: interested={interested_count} not_interested={not_interested_count}")
    print(f"Finished. Output saved to: {output_path}")

if __name__ == "__main__":
    video_files = glob.glob("/kaggle/input/**/*.mp4", recursive=True) + glob.glob("/kaggle/input/**/*.avi", recursive=True)
    if not video_files: video_files = glob.glob("*.mp4") + glob.glob("*.avi")
    if not video_files:
        print("Error: No video files found.")
        sys.exit(1)

    print(f"Found videos: {video_files}")
    input_video = video_files[0]
    output_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
    
    process_video(
        input_video,
        os.path.join(output_dir, "output_yolo_pose_depth_counting.mp4"),
        tracker_type="botsort",
    )
