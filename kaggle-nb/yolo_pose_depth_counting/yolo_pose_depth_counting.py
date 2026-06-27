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
import torch.distributed as dist
import torch.multiprocessing as mp
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

REGION_FILE = Path(__file__).with_name("region_data.json")
DEFAULT_REGION = [
    [1316, 437],
    [1316, 623],
    [1722, 606],
    [1668, 415]
]
MODEL_CANDIDATES = [
    "yolov26x.pt",
    "yolo26x.pt",
]
DEPTH_MODEL_REPO = "depth-anything/Depth-Anything-V2-Large-hf"
MAX_DURATION_SEC = 120
DEPTH_EVERY_N_FRAMES = 5
DWELL_SECONDS_REQUIRED = 3.0


def load_region_points():
    for region_file in (REGION_FILE, Path.cwd() / "region_data.json"):
        if region_file.exists():
            with region_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            region = data.get("region", DEFAULT_REGION)
            return np.array(region, np.int32).reshape((-1, 1, 2))
    return np.array(DEFAULT_REGION, np.int32).reshape((-1, 1, 2))


def order_points(pts):
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # TL
    rect[2] = pts[np.argmax(s)] # BR
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # TR
    rect[3] = pts[np.argmax(diff)] # BL
    return rect


def get_homography(region_pts, bev_w, bev_h, margin=180):
    src_pts = order_points(region_pts)
    cx, cy = bev_w // 2, bev_h // 2
    zone_w = int(bev_w * 0.56)
    zone_h = int(bev_h * 0.46)
    dst_pts = np.array([
        [cx - zone_w // 2, cy - zone_h // 2],
        [cx + zone_w // 2, cy - zone_h // 2],
        [cx + zone_w // 2, cy + zone_h // 2],
        [cx - zone_w // 2, cy + zone_h // 2]
    ], dtype="float32")
    H, _ = cv2.findHomography(src_pts, dst_pts)
    return H, dst_pts


def get_depth_plane(region_pts, depth_norm):
    src_pts = order_points(region_pts)
    points = []
    h, w = depth_norm.shape
    for pt in src_pts:
        x, y = int(pt[0]), int(pt[1])
        x = max(0, min(w-1, x))
        y = max(0, min(h-1, y))
        d = depth_norm[y, x]
        points.append([x, y, d])
    
    points = np.array(points)
    A = np.c_[points[:, 0], points[:, 1], np.ones(4)]
    Z = points[:, 2]
    C, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)
    return C


def load_detection_model(device_str):
    for model_name in MODEL_CANDIDATES:
        try:
            model = YOLO(model_name)
            return model, model_name
        except Exception:
            pass
    raise RuntimeError("Unable to load any detection model.")


def load_depth_model(device_str):
    image_processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_REPO)
    depth_model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_REPO)
    if "cuda" in device_str:
        depth_model = depth_model.to(device_str)
    depth_model.eval()
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


def draw_enterprise_box(frame, x1, y1, x2, y2, color, thickness=2):
    length = min(30, int((x2 - x1) * 0.2))
    # Top-left
    cv2.line(frame, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + length), color, thickness)
    # Top-right
    cv2.line(frame, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + length), color, thickness)
    # Bottom-left
    cv2.line(frame, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - length), color, thickness)
    # Bottom-right
    cv2.line(frame, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - length), color, thickness)
    # Draw faint full rect
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)

def draw_status_panel(frame, model_name, tracker_type, device_name, fps, interested_count, not_interested_count, active_tracks, active_in_region, rank, frame_count, max_frames):
    x, y, w, h = 28, 28, 520, 236
    panel = (14, 20, 30)
    border = (55, 65, 80)
    white = (245, 247, 250)
    muted = (170, 180, 195)
    green = (90, 230, 155)
    blue = (105, 170, 255)
    amber = (255, 200, 80)

    draw_translucent_rect(frame, (x, y), (x + w, y + h), panel, 0.84)
    cv2.rectangle(frame, (x, y), (x + w, y + h), border, 1)
    cv2.line(frame, (x, y), (x + w, y), amber, 4)
    draw_text(frame, "DEPTH-AWARE BEV ANALYTICS", (x + 22, y + 36), 0.72, white, 1, cv2.FONT_HERSHEY_DUPLEX)
    draw_text(frame, "YOLO tracking + Depth-Anything V2 + homography map", (x + 22, y + 62), 0.46, muted, 1)

    def metric(mx, label, value, color):
        cv2.rectangle(frame, (mx, y + 88), (mx + 145, y + 154), (25, 32, 44), -1)
        cv2.rectangle(frame, (mx, y + 88), (mx + 145, y + 154), color, 1)
        draw_text(frame, label, (mx + 10, y + 112), 0.43, muted, 1)
        draw_text(frame, str(value), (mx + 10, y + 145), 0.86, color, 2, cv2.FONT_HERSHEY_DUPLEX)

    metric(x + 22, "INTERESTED", interested_count, green)
    metric(x + 188, "IGNORED", not_interested_count, blue)
    metric(x + 354, "IN ROI", active_in_region, amber)
    draw_text(frame, f"{model_name} | {tracker_type} | tracks {active_tracks} | rank {rank}", (x + 22, y + 188), 0.46, muted, 1)
    draw_text(frame, f"{device_name[:34]} | {fps:.1f} FPS | depth every {DEPTH_EVERY_N_FRAMES} frames", (x + 22, y + 214), 0.46, muted, 1)

    progress = min(1.0, frame_count / float(max(1, max_frames)))
    cv2.rectangle(frame, (x + 22, y + h - 12), (x + w - 22, y + h - 8), (45, 50, 60), -1)
    cv2.rectangle(frame, (x + 22, y + h - 12), (x + 22 + int((w - 44) * progress), y + h - 8), amber, -1)


def draw_bev_canvas(bev_w, bev_h, bev_dst_pts, frame_count):
    bev = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)
    bev[:] = (18, 20, 24)

    for y in range(0, bev_h, 72):
        shade = 28 if (y // 72) % 2 == 0 else 24
        cv2.rectangle(bev, (0, y), (bev_w, min(bev_h, y + 72)), (shade, shade + 3, shade + 7), -1)

    grid_major = (62, 70, 80)
    grid_minor = (38, 44, 52)
    for i in range(0, max(bev_w, bev_h), 40):
        color = grid_major if i % 160 == 0 else grid_minor
        cv2.line(bev, (i, 0), (i, bev_h), color, 1)
        cv2.line(bev, (0, i), (bev_w, i), color, 1)

    overlay = bev.copy()
    cv2.fillPoly(overlay, [np.int32(bev_dst_pts)], (46, 128, 190))
    cv2.addWeighted(overlay, 0.18, bev, 0.82, 0, bev)
    cv2.polylines(bev, [np.int32(bev_dst_pts)], isClosed=True, color=(96, 190, 255), thickness=3)

    for point in np.int32(bev_dst_pts):
        cv2.circle(bev, tuple(point), 6, (245, 210, 110), -1)

    cv2.rectangle(bev, (22, 22), (bev_w - 22, 92), (18, 20, 24), -1)
    cv2.rectangle(bev, (22, 22), (bev_w - 22, 92), (88, 98, 112), 1)
    draw_text(bev, "2D BEV MAP", (40, 50), 0.72, (245, 247, 250), 1, cv2.FONT_HERSHEY_DUPLEX)
    draw_text(bev, "zoomed-out homography projection, depth-plane gated", (40, 76), 0.43, (176, 188, 204), 1)
    return bev


def draw_bev_person(bev_canvas, point, track_id, color, classification):
    bx, by = point
    halo = bev_canvas.copy()
    cv2.circle(halo, (bx, by), 20, color, -1)
    cv2.addWeighted(halo, 0.22, bev_canvas, 0.78, 0, bev_canvas)
    cv2.circle(bev_canvas, (bx, by), 11, (245, 247, 250), -1)
    cv2.circle(bev_canvas, (bx, by), 8, color, -1)
    cv2.circle(bev_canvas, (bx, by), 14, color, 2)

    label = f"ID {track_id}"
    if classification:
        label += f" {classification}"
    label_w = max(64, cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)[0][0] + 18)
    x1 = min(max(8, bx + 16), bev_canvas.shape[1] - label_w - 8)
    y1 = min(max(102, by - 14), bev_canvas.shape[0] - 32)
    draw_translucent_rect(bev_canvas, (x1, y1), (x1 + label_w, y1 + 26), (20, 24, 31), 0.86)
    cv2.rectangle(bev_canvas, (x1, y1), (x1 + label_w, y1 + 26), color, 1)
    draw_text(bev_canvas, label, (x1 + 8, y1 + 18), 0.44, (245, 247, 250), 1)


def process_video(input_path, output_path, tracker_type, rank):
    device_to_use = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(rank) if torch.cuda.is_available() else "CPU"

    model, model_name = load_detection_model(device_to_use)
    image_processor, depth_model = load_depth_model(device_to_use)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[Rank {rank}] Error: Could not open {input_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    max_duration_sec = MAX_DURATION_SEC
    max_frames = fps * max_duration_sec

    target_h = 900
    scale = target_h / height
    target_w = int(width * scale)
    bev_w, bev_h = max(640, target_w), target_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_w + bev_w, target_h))
    
    region_pts = load_region_points()
    H, bev_dst_pts = get_homography(region_pts, bev_w, bev_h)
    plane_coeffs = None
    
    track_states = {}
    track_history = {}
    track_ema_depth = {}
    track_ema_anchor = {}
    EMA_ALPHA = 0.3
    depth_tolerance = 20.0
    
    interested_count = 0
    not_interested_count = 0
    max_history = 50
    tracker_yaml = "botsort.yaml" if tracker_type == "botsort" else "bytetrack.yaml"
    prev_time = time.time()
    frame_count = 0
    depth_norm = None

    print(f"[Rank {rank}] Processing {input_path} ({max_duration_sec} seconds) with optimized Depth-Anything V2 BEV...")

    while cap.isOpened() and frame_count < max_frames:
        success, frame = cap.read()
        if not success:
            break

        results = model.track(
            frame, persist=True, tracker=tracker_yaml, conf=0.25, iou=0.5,
            classes=[0], verbose=False, device=device_to_use,
        )

        annotated_frame = frame.copy()
        
        region_overlay = annotated_frame.copy()
        cv2.fillPoly(region_overlay, [region_pts], (45, 212, 191))
        cv2.addWeighted(region_overlay, 0.16, annotated_frame, 0.84, 0, annotated_frame)
        cv2.polylines(annotated_frame, [region_pts], isClosed=True, color=(45, 212, 191), thickness=7)

        if depth_norm is None or frame_count % DEPTH_EVERY_N_FRAMES == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            inputs = image_processor(images=pil_img, return_tensors="pt")
            if "cuda" in device_to_use:
                inputs = {k: v.to(device_to_use) for k, v in inputs.items()}
            with torch.inference_mode():
                depth_outputs = depth_model(**inputs)
                predicted_depth = torch.nn.functional.interpolate(
                    depth_outputs.predicted_depth.unsqueeze(1),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()
                depth_map = predicted_depth.cpu().numpy()
            depth_min, depth_max = depth_map.min(), depth_map.max()
            denom = max(float(depth_max - depth_min), 1e-6)
            depth_norm = ((depth_map - depth_min) / denom * 255).astype(np.uint8)

        if plane_coeffs is None:
            plane_coeffs = get_depth_plane(region_pts, depth_norm)

        bev_canvas = draw_bev_canvas(bev_w, bev_h, bev_dst_pts, frame_count)

        active_tracks = 0
        active_in_region = 0
        
        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            active_tracks = len(track_ids)

            for i, (box, track_id) in enumerate(zip(boxes, track_ids)):
                x1, y1, x2, y2 = map(int, box)
                raw_cx = (x1 + x2) // 2
                raw_y2 = y2
                
                # EMA Box Anchor Stabilization
                if track_id not in track_ema_anchor:
                    track_ema_anchor[track_id] = np.array([raw_cx, raw_y2], dtype=float)
                else:
                    track_ema_anchor[track_id] = EMA_ALPHA * np.array([raw_cx, raw_y2]) + (1 - EMA_ALPHA) * track_ema_anchor[track_id]
                
                cx, y2_smooth = int(track_ema_anchor[track_id][0]), int(track_ema_anchor[track_id][1])
                anchor = (cx, y2_smooth)
                
                # Patch-Based Depth Sampling
                patch_sz = 5
                sy1, sy2 = max(0, y2_smooth - patch_sz), min(height, y2_smooth + 1)
                sx1, sx2 = max(0, cx - patch_sz//2), min(width, cx + patch_sz//2 + 1)
                patch = depth_norm[sy1:sy2, sx1:sx2]
                raw_depth = np.median(patch) if patch.size > 0 else depth_norm[min(y2_smooth, height-1), min(cx, width-1)]
                
                # EMA Depth Stabilization
                if track_id not in track_ema_depth:
                    track_ema_depth[track_id] = raw_depth
                else:
                    track_ema_depth[track_id] = EMA_ALPHA * raw_depth + (1 - EMA_ALPHA) * track_ema_depth[track_id]
                person_depth = track_ema_depth[track_id]
                
                # Dynamic Ground-Plane Calibration Check
                expected_depth = plane_coeffs[0]*cx + plane_coeffs[1]*y2_smooth + plane_coeffs[2]
                depth_valid = abs(person_depth - expected_depth) < depth_tolerance
                
                in_2d_region = cv2.pointPolygonTest(region_pts, anchor, False) >= 0
                is_valid_interest = in_2d_region and depth_valid

                if is_valid_interest:
                    active_in_region += 1

                state = track_states.setdefault(
                    int(track_id),
                    {"total_dwell_seconds": 0.0, "classification": None, "frames_since_last_inside": 0, "inside": False}
                )

                if is_valid_interest:
                    state["inside"] = True
                    state["total_dwell_seconds"] += (1.0 / fps)
                    state["frames_since_last_inside"] = 0
                    
                    if state["classification"] != "interested" and state["total_dwell_seconds"] > DWELL_SECONDS_REQUIRED:
                        if state["classification"] == "not interested":
                            not_interested_count -= 1
                        interested_count += 1
                        state["classification"] = "interested"
                else:
                    state["inside"] = False
                    if state["total_dwell_seconds"] > 0:
                        state["frames_since_last_inside"] += 1
                        if state["classification"] is None and state["frames_since_last_inside"] > int(3.0 * fps):
                            not_interested_count += 1
                            state["classification"] = "not interested"
                            
                dwell_seconds = state["total_dwell_seconds"]

                track_history.setdefault(track_id, []).append(anchor)
                if len(track_history[track_id]) > max_history:
                    track_history[track_id].pop(0)

                color = (74, 222, 128) if state["classification"] == "interested" else (251, 191, 36) if is_valid_interest else (96, 165, 250)
                
                draw_enterprise_box(annotated_frame, x1, y1, x2, y2, color, 3)
                cv2.circle(annotated_frame, anchor, 6, color, -1)
                
                # True Homography BEV Projection
                pt = np.array([[[cx, y2_smooth]]], dtype="float32")
                bev_pt = cv2.perspectiveTransform(pt, H)[0][0]
                bx, by = int(bev_pt[0]), int(bev_pt[1])
                
                bx = max(0, min(bev_w-1, bx))
                by = max(0, min(bev_h-1, by))
                
                draw_bev_person(bev_canvas, (bx, by), track_id, color, state["classification"])

                label = f"ID {track_id}"
                if state["classification"]:
                    label += f" | {state['classification']}"
                elif is_valid_interest:
                    label += f" | {dwell_seconds:.1f}s"
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                label_size, baseline = cv2.getTextSize(label, font, 0.48, 1)
                label_y = max(20, y1 - label_size[1] - baseline - 8)
                draw_translucent_rect(annotated_frame, (x1, label_y), (x1 + label_size[0] + 16, label_y + label_size[1] + baseline + 8), (15, 23, 42), 0.82)
                cv2.rectangle(annotated_frame, (x1, label_y), (x1 + label_size[0] + 16, label_y + label_size[1] + baseline + 8), color, 1)
                draw_text(annotated_frame, label, (x1 + 8, label_y + label_size[1] + 2), 0.48, (248, 250, 252), 1, font)

        curr_time = time.time()
        inference_fps = 1 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time
        draw_status_panel(
            annotated_frame, model_name, tracker_type, device_name,
            inference_fps, interested_count, not_interested_count,
            active_tracks, active_in_region, rank, frame_count, max_frames
        )

        annotated_resized = cv2.resize(annotated_frame, (target_w, target_h))
        combined = np.zeros((target_h, target_w + bev_w, 3), dtype=np.uint8)
        combined[:, :target_w] = annotated_resized
        combined[:, target_w:] = bev_canvas
        out.write(combined)
        
        frame_count += 1

        if frame_count % fps == 0:
            print(f"[Rank {rank}] {frame_count // fps}/{max_duration_sec}s | in_region={active_in_region} int={interested_count} not_int={not_interested_count}")

    for state in track_states.values():
        if state["classification"] is None and state["total_dwell_seconds"] > 0:
            if state["total_dwell_seconds"] > DWELL_SECONDS_REQUIRED:
                interested_count += 1
                state["classification"] = "interested"
            else:
                not_interested_count += 1
                state["classification"] = "not interested"

    cap.release()
    out.release()
    print(f"[Rank {rank}] Final counts: interested={interested_count} not_interested={not_interested_count}")
    print(f"[Rank {rank}] Finished. Output saved to: {output_path}")


def main_worker(rank, world_size, video_files, output_dir):
    if world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

    my_videos = [v for i, v in enumerate(video_files) if i % world_size == rank]
    
    for input_video in my_videos:
        out_name = f"rank{rank}_" + os.path.basename(input_video).replace(".mp4", "_out.mp4").replace(".avi", "_out.mp4")
        out_path = os.path.join(output_dir, out_name)
        process_video(input_video, out_path, tracker_type="botsort", rank=rank)
        
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    video_files = glob.glob("/kaggle/input/**/*.mp4", recursive=True) + glob.glob("/kaggle/input/**/*.avi", recursive=True)
    if not video_files: video_files = glob.glob("*.mp4") + glob.glob("*.avi")
    if not video_files:
        print("Error: No video files found.")
        sys.exit(1)

    print(f"Found {len(video_files)} videos: {video_files}")
    output_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
    
    world_size = torch.cuda.device_count()
    if world_size > 1:
        print(f"Using DDP with {world_size} GPUs. Sharding {len(video_files)} videos.")
        mp.spawn(main_worker, args=(world_size, video_files, output_dir), nprocs=world_size, join=True)
    else:
        print("Using single GPU/CPU processing.")
        main_worker(0, 1, video_files, output_dir)
