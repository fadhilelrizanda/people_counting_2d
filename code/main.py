import os
import sys
import argparse
import time
import numpy as np

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

try:
    import cv2
except ImportError:
    print("❌ OpenCV is not installed. Please install it using: pip install opencv-python")
    sys.exit(1)

try:
    import onnxruntime as ort
except ImportError:
    print("❌ ONNX Runtime is not installed. Please install it using: pip install onnxruntime")
    sys.exit(1)


from tracker import CentroidTracker
from script.utils import preprocess_frame


def run_pipeline(args):
    print("🎬 Initializing People Counting Pipeline...")
    print(f"📦 Model: {args.model}")
    print(f"📹 Video: {args.video}")
    
    if not os.path.exists(args.model):
        print(f"❌ Model file not found at: {args.model}")
        print("Please check the path or download the output weights from Kaggle first.")
        sys.exit(1)
        
    if not os.path.exists(args.video):
        print(f"❌ Video file not found at: {args.video}")
        sys.exit(1)

    # 1. Initialize ONNX session
    print("🦿 Loading ONNX model session...")
    try:
        session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    except Exception as e:
        print(f"❌ Failed to load ONNX model: {e}")
        sys.exit(1)
        
    input_name = session.get_inputs()[0].name
    outputs = session.get_outputs()
    output_names = [o.name for o in outputs]
    
    print(f"👉 Input node name: {input_name}")
    print(f"👈 Output node names: {output_names}")

    # 2. Open Video Stream
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"❌ Failed to open video: {args.video}")
        sys.exit(1)
        
    w_video = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_video = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"📺 Input resolution: {w_video}x{h_video} @ {fps_video} FPS ({total_frames} total frames)")

    # Define crossing line location (default: horizontal line at 50% height)
    line_y = int(h_video * args.line_ratio)
    line_start = (0, line_y)
    line_end = (w_video, line_y)
    print(f"➖ Virtual Counting Line set at Y={line_y} ({int(args.line_ratio*100)}% height)")

    # 3. Setup Output Writer if requested
    writer = None
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps_video, (w_video, h_video))
        print(f"💾 Saving output video to: {args.output}")

    # 4. Initialize Tracker and Counter States
    tracker = CentroidTracker(max_disappeared=args.max_disappeared)
    
    count_in = 0  # Top to bottom (or crosses down)
    count_out = 0 # Bottom to top (or crosses up)
    
    # Sets to track counted object IDs in each direction to avoid duplicate counts
    counted_in_ids = set()
    counted_out_ids = set()

    frame_count = 0
    start_time = time.time()

    # Create window if displaying
    if args.show:
        cv2.namedWindow("Top-View People Counting (YOLO26)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Top-View People Counting (YOLO26)", 1280, 720)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            t_frame_start = time.time()

            # Preprocess frame for YOLO
            blob, h_orig, w_orig = preprocess_frame(frame, imgsz=args.imgsz)

            # Run inference
            net_outputs = session.run(output_names, {input_name: blob})
            
            # The model has shape (1, 300, 6)
            # Detections: [x1, y1, x2, y2, score, class_id]
            detections = net_outputs[0][0]
            
            rects = []
            for det in detections:
                x1, y1, x2, y2, score, class_id = det
                
                # Check confidence threshold and make sure it is class 0 (person/head)
                # Since SCUT-HEAD contains only 1 class, class_id will typically be 0
                if score >= args.conf and int(class_id) == 0:
                    # Rescale coordinates to original frame size
                    x1_rescaled = int(x1 * w_orig / args.imgsz)
                    y1_rescaled = int(y1 * h_orig / args.imgsz)
                    x2_rescaled = int(x2 * w_orig / args.imgsz)
                    y2_rescaled = int(y2 * h_orig / args.imgsz)
                    
                    rects.append([x1_rescaled, y1_rescaled, x2_rescaled, y2_rescaled])
                    
                    # Draw detection bounding box
                    cv2.rectangle(frame, (x1_rescaled, y1_rescaled), (x2_rescaled, y2_rescaled), (0, 255, 0), 2)
                    
            # Update tracker
            objects = tracker.update(rects)

            # Check for line crossing and update trails
            for (obj_id, centroid) in objects.items():
                history = tracker.history[obj_id]
                
                # Check line crossing if we have at least 2 historic points
                if len(history) >= 2:
                    prev_centroid = history[-2]
                    curr_centroid = history[-1]
                    
                    # Line is horizontal: Y = line_y
                    # Top of frame is Y=0, bottom is Y=height.
                    # Crossing down (Top -> Bottom): prev_y < line_y and curr_y >= line_y
                    if prev_centroid[1] < line_y and curr_centroid[1] >= line_y:
                        if obj_id not in counted_in_ids:
                            count_in += 1
                            counted_in_ids.add(obj_id)
                            print(f"🚶‍♂️ Object ID {obj_id} crossed DOWN (IN) at frame {frame_count}")
                            
                    # Crossing up (Bottom -> Top): prev_y > line_y and curr_y <= line_y
                    elif prev_centroid[1] > line_y and curr_centroid[1] <= line_y:
                        if obj_id not in counted_out_ids:
                            count_out += 1
                            counted_out_ids.add(obj_id)
                            print(f"🚶‍♀️ Object ID {obj_id} crossed UP (OUT) at frame {frame_count}")

                # Draw track trail
                for j in range(1, len(history)):
                    pt1 = tuple(history[j - 1])
                    pt2 = tuple(history[j])
                    cv2.line(frame, pt1, pt2, (255, 100, 0), 2)

                # Draw centroid and ID label
                cv2.circle(frame, tuple(centroid), 4, (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    f"ID {obj_id}",
                    (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )

            # Draw virtual crossing line
            # Set color: red if someone is close/crossing, or green by default
            line_color = (0, 0, 255) if len(rects) > 0 else (0, 255, 255)
            cv2.line(frame, line_start, line_end, line_color, 3)
            cv2.putText(
                frame,
                "COUNTING LINE",
                (10, line_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                line_color,
                2
            )

            # Calculate FPS
            t_frame_end = time.time()
            fps_current = 1.0 / (t_frame_end - t_frame_start)

            # Draw HUD dashboard
            hud_overlay = frame.copy()
            cv2.rectangle(hud_overlay, (5, 5), (320, 130), (0, 0, 0), -1)
            cv2.addWeighted(hud_overlay, 0.6, frame, 0.4, 0, frame)

            cv2.putText(frame, f"FPS: {fps_current:.1f}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"IN (Cross Down): {count_in}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"OUT (Cross Up): {count_out}", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            cv2.putText(frame, f"Total Count: {count_in + count_out}", (15, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # Save frame to output video
            if writer is not None:
                writer.write(frame)

            # Show display
            if args.show:
                cv2.imshow("Top-View People Counting (YOLO26)", frame)
                # Esc key to break
                if cv2.waitKey(1) & 0xFF == 27:
                    print("🛑 Execution interrupted by user.")
                    break

            if frame_count % 100 == 0:
                print(f"📊 Processed {frame_count}/{total_frames} frames... | Current Count: IN={count_in}, OUT={count_out}")

    finally:
        # Cleanup
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
            
        duration = time.time() - start_time
        print("\n🏁 Pipeline Completed Summary:")
        print(f"⏱️ Total Time: {duration:.2f} seconds")
        print(f"📈 Average FPS: {frame_count / duration:.2f}")
        print(f"📥 Total Crossed DOWN (IN): {count_in}")
        print(f"📤 Total Crossed UP (OUT): {count_out}")
        print(f"👥 Combined Count: {count_in + count_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Top-View People Counting using Fine-Tuned YOLO26 ONNX Model")
    parser.add_argument(
        "--model",
        type=str,
        default="models/runs/train/yolo26_people_counting/weights/best.onnx",
        help="Path to the exported ONNX model"
    )
    parser.add_argument(
        "--video",
        type=str,
        default="data/sample_videos/people_demo.mp4",
        help="Path to the sample video file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/annotated_results.mp4",
        help="Path to save the annotated output video file"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image resolution"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Object confidence threshold"
    )
    parser.add_argument(
        "--line-ratio",
        type=float,
        default=0.5,
        help="Ratio of height where virtual line crosses (0.0 to 1.0)"
    )
    parser.add_argument(
        "--max-disappeared",
        type=int,
        default=25,
        help="Max frames an object can disappear before deregistration"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Whether to show CV2 display window"
    )
    
    args = parser.parse_args()
    run_pipeline(args)
