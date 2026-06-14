import os
import cv2
import pandas as pd
import subprocess

def download_dataset(dataset_dir="data"):
    """
    Downloads the Oxford Town Centre dataset using Kaggle API.
    """
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)
        
    print("Downloading Oxford Town Centre dataset via Kaggle API...")
    # NOTE: The dataset 'virenbr11/oxford-town-centre' is an example of the Oxford dataset on Kaggle.
    # Adjust the dataset identifier if using a different one.
    try:
        subprocess.run(["kaggle", "datasets", "download", "-d", "virenbr11/oxford-town-centre", "--unzip", "-p", dataset_dir], check=True)
        print("Download and extraction completed.")
    except Exception as e:
        print(f"Failed to download dataset: {e}")

def create_video_with_bboxes(video_path, csv_path, output_path):
    """
    Reads the original video and CSV ground truth, draws bounding boxes, 
    and writes to an output video.
    """
    print(f"Reading ground truth from {csv_path}...")
    # Oxford Town Centre ground truth format typically has no header
    # Format: personNumber, frameNumber, headValid, bodyValid, headLeft, headTop, headRight, headBottom, bodyLeft, bodyTop, bodyRight, bodyBottom
    col_names = ['personId', 'frameNumber', 'headValid', 'bodyValid', 
                 'headLeft', 'headTop', 'headRight', 'headBottom', 
                 'bodyLeft', 'bodyTop', 'bodyRight', 'bodyBottom']
    
    try:
        df = pd.read_csv(csv_path, names=col_names)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    print(f"Opening video {video_path}...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # mp4v codec for mp4 output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Filter bounding boxes for the current frame
        frame_data = df[df['frameNumber'] == frame_idx]
        
        for _, row in frame_data.iterrows():
            # Check if body is valid (typically 1 for valid in this dataset)
            if row.get('bodyValid', 1) == 1:
                x1, y1 = int(row['bodyLeft']), int(row['bodyTop'])
                x2, y2 = int(row['bodyRight']), int(row['bodyBottom'])
                
                # Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw the person ID
                cv2.putText(frame, f"ID: {int(row['personId'])}", (x1, max(y1 - 10, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        out.write(frame)
        frame_idx += 1
        
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx} frames...")
            
    cap.release()
    out.release()
    print(f"Video with ground truth bounding boxes saved to {output_path}")

if __name__ == "__main__":
    print("==========================================================================")
    print("REMINDER: The dataset and model are intended to be run in Kaggle via")
    print("kaggle-mcp, not on this local machine. Proceeding with the code...")
    print("==========================================================================\n")
    
    dataset_dir = "./data"
    
    # Typical filenames for the Oxford Town Centre dataset
    video_file = os.path.join(dataset_dir, "TownCentreXVID.avi")
    csv_file = os.path.join(dataset_dir, "TownCentre-groundtruth.top")
    output_file = "oxford_town_centre_gt.mp4"
    
    # 1. Download Dataset via Kaggle
    if not (os.path.exists(video_file) and os.path.exists(csv_file)):
        download_dataset(dataset_dir)
        
    # 2. Process Video
    if os.path.exists(video_file) and os.path.exists(csv_file):
        create_video_with_bboxes(video_file, csv_file, output_file)
    else:
        print("Dataset files not found. Ensure the kaggle download was successful,")
        print("and the filenames match the extracted contents.")
