# Agent Guide

This repository is a people-counting/video-analytics workspace. The current code has three active areas:

- `code/`: local CPU ONNX pipeline for detecting people, tracking centroids, and counting line crossings.
- `kaggle-nb/`: Kaggle script kernels for GPU video processing and experiment outputs.
- `kaggle-mcp/`: FastMCP wrapper around the Kaggle CLI for profile, dataset, kernel, and model operations.

Use `rtk` before shell commands in this repo, per `/home/fadhil/.codex/RTK.md`.

## Current Pipelines

### Local ONNX Counting

Entry point: `code/main.py`

The local pipeline:

1. Loads an ONNX model with `onnxruntime` on CPU.
2. Opens an input video with OpenCV.
3. Resizes each frame to the inference size via `script/utils.py`.
4. Reads detections shaped like `[x1, y1, x2, y2, score, class_id]`.
5. Keeps class `0` detections above `--conf`.
6. Tracks detection centroids with `code/tracker.py`.
7. Counts objects crossing a horizontal line at `--line-ratio`.
8. Writes an annotated video when `--output` is set.

Common command:

```bash
rtk python code/main.py --model path/to/best.onnx --video path/to/input.mp4 --output output/annotated_results.mp4
```

### Kaggle YOLO26 Tracking

Entry point: `kaggle-nb/yolo26_tracking_counting/yolo_tracking_counting.py`

This script runs Ultralytics YOLO tracking on the Oxford Town Centre dataset. It tries a YOLO26/YOLO11 segmentation model, runs BotSORT and ByteTrack, draws a fixed polygon region, tracks person IDs, and counts polygon entries/exits.

### Latest Kaggle Depth + BEV Analytics

Entry point: `kaggle-nb/yolo_pose_depth_counting/yolo_pose_depth_counting.py`

This is the latest Kaggle pipeline. The working tree may not currently contain this directory, but it is tracked in git and recent outputs are in `output_v13/`, `output_new_bev/`, and related `output_*` folders.

It:

1. Installs/imports `ultralytics` and `transformers` if missing.
2. Loads YOLO detection/tracking from `yolo11x.pt`, falling back to `yolov8x.pt`.
3. Loads Depth-Anything V2 Large from Hugging Face: `depth-anything/Depth-Anything-V2-Large-hf`.
4. Uses BotSORT tracking and person class filtering.
5. Loads a four-point region polygon from `region_data.json`, with an embedded fallback.
6. Computes a true homography with `cv2.findHomography` to project camera points into a 720x720 BEV map.
7. Runs Depth-Anything V2 per frame, normalizes the depth map, and estimates a floor depth plane from the region corners with least squares.
8. Uses the smoothed bottom-center of each person box as the ground anchor.
9. Samples a small depth patch under each anchor, smooths anchor/depth with EMA, and rejects tracks whose depth is too far from the floor plane.
10. Counts dwell time only when the anchor is inside the 2D region and passes the depth check.
11. Classifies tracks as `interested` after more than 10 seconds of valid dwell, otherwise `not interested` after leaving.
12. Writes a side-by-side output video: annotated camera view on the left, 2D BEV radar/map on the right.

The script supports multi-GPU Kaggle runs by spawning one worker per CUDA device and sharding videos by rank.

Kernel metadata: `kaggle-nb/yolo_pose_depth_counting/kernel-metadata.json`

Use T4/T4x2 for GPU kernels. If pushing with the Kaggle CLI, include:

```bash
rtk kaggle kernels push -p kaggle-nb/yolo_pose_depth_counting --accelerator NvidiaTeslaT4
```

### Kaggle Enterprise Interest Analytics

Entry point: `kaggle-nb/yolov26x_tracking_counting_enterprise/yolov26x_tracking_counting_enterprise.py`

Older polygon-only pipeline. It tracks people with ByteTrack, uses each box bottom-center point as the region anchor, classifies dwell over 10 seconds as `interested`, and writes `output_yolov26x_bytetrack_2min.mp4`.

### Dataset Visualizer

Entry point: `kaggle-nb/dataset_visualizer/download_and_visualize.py`

This finds Oxford Town Centre files under `/kaggle/input`, reads `TownCentre-groundtruth.top`, filters rows where `body_valid == 1`, skips NaN boxes, and writes a one-minute ground-truth overlay video.

## Region Selection

Use `code/region/region_selector.py` to click four polygon points on the first frame of a source video. It writes:

- `code/region/region_data.json`
- `code/region/region_preview.jpg`

Copy the JSON into the relevant Kaggle notebook directory when the remote script needs the same region.

## Kaggle MCP

`kaggle-mcp/server.py` exposes Kaggle CLI actions as MCP tools. It stores profile tokens in `~/.kaggle/profiles.json`, injects the active token as `KAGGLE_API_TOKEN`, and shells out to `kaggle`.

The root project uses Kaggle for remote GPU work, but the actual notebook scripts live in `kaggle-nb/`.

## Working Rules

- Keep changes small and update only the pipeline being touched.
- Do not modify generated outputs, downloaded models, or videos unless the task explicitly asks for it.
- For new Kaggle work, create a new subdirectory under `kaggle-nb/`.
- Prefer existing helpers and plain OpenCV/NumPy code over new dependencies.
- Remote GPU work should use T4/T4x2, not P100, unless the script explicitly handles CPU fallback.
- Always use the yolo26x detection model for person tracking and counting tasks.
- Always use Depth-Anything V2 (Large or highest available version) for depth estimation.
- Never overwrite the output video; each run should create and save to a new output directory.
