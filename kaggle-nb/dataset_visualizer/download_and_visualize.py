import pandas as pd
import cv2
import os
import math

annotations_path = '/kaggle/input/oxford-town-centre/TownCentre-groundtruth.top'
video_path = '/kaggle/input/oxford-town-centre/TownCentreXVID.mp4'
output_path = 'output_1min.mp4'

df = pd.read_csv(annotations_path, header=None, names=[
    'person_id', 'frame_idx', 'head_valid', 'body_valid',
    'head_l', 'head_t', 'head_r', 'head_b',
    'body_l', 'body_t', 'body_r', 'body_b'
])

# Use only body valid == 1
df_body = df[df['body_valid'] == 1].copy()

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 25.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 1 minute = fps * 60 frames
max_frames = int(fps * 60)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_count = 0
while cap.isOpened() and frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Get annotations for this frame
    frame_ann = df_body[df_body['frame_idx'] == frame_count]
    
    for _, row in frame_ann.iterrows():
        try:
            if math.isnan(row['body_l']) or math.isnan(row['body_t']) or math.isnan(row['body_r']) or math.isnan(row['body_b']):
                continue
            x1, y1 = int(float(row['body_l'])), int(float(row['body_t']))
            x2, y2 = int(float(row['body_r'])), int(float(row['body_b']))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {int(row['person_id'])}", (x1, max(y1-5, 0)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        except Exception as e:
            pass
            
    out.write(frame)
    frame_count += 1

cap.release()
out.release()
print(f"Finished generating {output_path}")
