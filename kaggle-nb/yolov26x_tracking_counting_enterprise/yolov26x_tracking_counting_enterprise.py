import glob
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

if importlib.util.find_spec("ultralytics") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])

import cv2
import numpy as np
import torch
from ultralytics import YOLO


REGION_FILE = Path(__file__).with_name("region_data.json")
DEFAULT_REGION = [
    [13, 1606],
    [779, 1773],
    [936, 1372],
    [177, 1278],
]
MODEL_CANDIDATES = [
    "yolov26x.pt",
    "yolo26x.pt",
    "yolo11x.pt",
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
    else:
        print("Region file not found; using embedded selected region.")
        region = DEFAULT_REGION
    return np.array(region, np.int32).reshape((-1, 1, 2))


def select_device():
    if not torch.cuda.is_available():
        return "cpu", "CPU"

    device_count = torch.cuda.device_count()
    if device_count > 1:
        # Enable DDP / multi-GPU by passing multiple device IDs
        devices = ",".join(str(i) for i in range(device_count))
        device_name = f"DDP multi-GPU ({device_count}x {torch.cuda.get_device_name(0)})"
        return devices, device_name

    device_name = torch.cuda.get_device_name(0)
    if "P100" in device_name:
        print("Warning: P100 GPU detected. PyTorch 2.x dropped support for P100. Falling back to CPU.")
        return "cpu", "P100 fallback: CPU"
    return "cuda:0", device_name


def load_detection_model():
    last_error = None
    for model_name in MODEL_CANDIDATES:
        try:
            print(f"Loading detector: {model_name}")
            return YOLO(model_name), model_name
        except Exception as exc:
            last_error = exc
            print(f"Could not load {model_name}: {exc}")
    raise RuntimeError(f"Unable to load any detection model. Last error: {last_error}")


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

    draw_text(frame, "INTEREST ANALYTICS", (panel_x + 22, panel_y + 38), 0.72, white, 1, cv2.FONT_HERSHEY_DUPLEX)
    draw_text(frame, "Bottom-center dwell classification", (panel_x + 22, panel_y + 64), 0.45, muted, 1)

    draw_metric(frame, "Interested", interested_count, panel_x + 24, panel_y + 108, green)
    draw_metric(frame, "Not interested", not_interested_count, panel_x + 210, panel_y + 108, red)
    draw_metric(frame, "In region", active_in_region, panel_x + 410, panel_y + 108, amber)

    y0 = panel_y + 184
    draw_text(frame, f"Detector: {model_name}", (panel_x + 24, y0), 0.48, muted, 1)
    draw_text(frame, f"Tracker: {tracker_type.upper()}  |  Active tracks: {active_tracks}", (panel_x + 24, y0 + 24), 0.48, muted, 1)
    draw_text(
        frame,
        f"Interested rule: > {dwell_seconds_required:.0f}s in zone  |  Device: {device_name[:25]}  |  FPS: {fps:.1f}",
        (panel_x + 24, y0 + 48),
        0.48,
        muted,
        1,
    )


def draw_tracking_annotation(frame, box, track_id, is_inside, classification, dwell_seconds, trail):
    x1, y1, x2, y2 = map(int, box)
    anchor = ((x1 + x2) // 2, y2)
    accent_counted = (74, 222, 128)
    accent_inside = (251, 191, 36)
    accent_outside = (96, 165, 250)
    color = accent_counted if classification == "interested" else accent_inside if is_inside else accent_outside

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.circle(frame, anchor, 6, color, -1)

    if len(trail) >= 2:
        pts = np.array(trail, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=2)

    if classification:
        label = f"ID {track_id} | {classification}"
    elif is_inside:
        label = f"ID {track_id} | {dwell_seconds:.1f}s"
    else:
        label = f"ID {track_id}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    label_size, baseline = cv2.getTextSize(label, font, 0.48, 1)
    label_w, label_h = label_size
    label_y = max(20, y1 - label_h - baseline - 8)
    draw_translucent_rect(frame, (x1, label_y), (x1 + label_w + 16, label_y + label_h + baseline + 8), (15, 23, 42), 0.82)
    cv2.rectangle(frame, (x1, label_y), (x1 + label_w + 16, label_y + label_h + baseline + 8), color, 1)
    draw_text(frame, label, (x1 + 8, label_y + label_h + 2), 0.48, (248, 250, 252), 1, font)


def process_video(input_path, output_path, tracker_type="bytetrack", max_duration_sec=120):
    model, model_name = load_detection_model()
    device_to_use, device_name = select_device()

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
    flat_region = region_pts.reshape((-1, 2)).tolist()
    print(f"Decoded frame: width={width} height={height} fps={fps}")
    print(f"Loaded region points: {flat_region}")

    track_states = {}
    track_history = {}
    interested_count = 0
    not_interested_count = 0
    max_history = 50
    tracker_yaml = "botsort.yaml" if tracker_type == "botsort" else "bytetrack.yaml"
    prev_time = time.time()
    start_time = prev_time
    frame_count = 0

    print(
        f"Processing {input_path} with {model_name}, {tracker_type}, {device_name}. "
        f"Stopping after {max_duration_sec}s ({max_frames} frames)."
    )

    while cap.isOpened() and frame_count < max_frames:
        success, frame = cap.read()
        if not success:
            break

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

        region_overlay = annotated_frame.copy()
        cv2.fillPoly(region_overlay, [region_pts], (45, 212, 191))
        cv2.addWeighted(region_overlay, 0.16, annotated_frame, 0.84, 0, annotated_frame)
        cv2.polylines(annotated_frame, [region_pts], isClosed=True, color=(45, 212, 191), thickness=7)
        for idx, point in enumerate(region_pts.reshape((-1, 2)), start=1):
            px, py = map(int, point)
            cv2.circle(annotated_frame, (px, py), 8, (45, 212, 191), -1)
            draw_text(annotated_frame, str(idx), (px + 10, py - 10), 0.7, (248, 250, 252), 2)

        active_tracks = 0
        active_in_region = 0
        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            active_tracks = len(track_ids)

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                anchor = ((x1 + x2) // 2, y2)
                is_inside = cv2.pointPolygonTest(region_pts, anchor, False) >= 0
                if is_inside:
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

                if is_inside:
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

                draw_tracking_annotation(
                    annotated_frame,
                    box,
                    track_id,
                    is_inside,
                    state["classification"],
                    dwell_seconds,
                    track_history[track_id],
                )

        curr_time = time.time()
        inference_fps = 1 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time
        elapsed_sec = curr_time - start_time
        draw_status_panel(
            annotated_frame,
            model_name=model_name,
            tracker_type=tracker_type,
            device_name=device_name,
            fps=inference_fps,
            elapsed_sec=elapsed_sec,
            interested_count=interested_count,
            not_interested_count=not_interested_count,
            active_tracks=active_tracks,
            active_in_region=active_in_region,
            dwell_seconds_required=DWELL_SECONDS_REQUIRED,
        )

        out.write(annotated_frame)
        frame_count += 1

        if frame_count % fps == 0:
            print(
                f"[{tracker_type}] {frame_count // fps}/{max_duration_sec}s | "
                f"in_region={active_in_region} interested={interested_count} not_interested={not_interested_count}"
            )

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
    print(f"Finished {tracker_type}. Output saved to: {output_path}")


if __name__ == "__main__":
    video_files = glob.glob("/kaggle/input/**/*.mp4", recursive=True) + glob.glob("/kaggle/input/**/*.avi", recursive=True)

    if not video_files:
        video_files = glob.glob("*.mp4") + glob.glob("*.avi")

    if not video_files:
        print("Error: No video files found.")
        sys.exit(1)

    print(f"Found videos: {video_files}")
    input_video = video_files[0]
    output_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else "."

    process_video(
        input_video,
        os.path.join(output_dir, "output_yolov26x_bytetrack_2min.mp4"),
        tracker_type="bytetrack",
    )
