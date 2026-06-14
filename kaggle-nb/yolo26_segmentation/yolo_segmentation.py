import os
os.system("pip install ultralytics")

import cv2
from ultralytics import YOLO

def process_video(input_path, output_path, model_name="yolo26n-seg.pt", max_duration_sec=60):
    print(f"Loading model: {model_name}...")
    model = YOLO(model_name)

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
    while cap.isOpened() and frame_count < max_frames:
        success, frame = cap.read()
        if not success:
            break

        results = model.predict(frame, conf=0.25, verbose=False)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)
        frame_count += 1
        
        if frame_count % fps == 0:
            print(f"Processed {frame_count // fps} / {max_duration_sec} seconds...")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Finished! Output saved to: {output_path}")

if __name__ == "__main__":
    INPUT_VIDEO = "/kaggle/input/oxford-town-centre/TownCentreXVID.mp4"
    OUTPUT_VIDEO = "/kaggle/working/output_masked.mp4"
    process_video(INPUT_VIDEO, OUTPUT_VIDEO)
