#!/bin/bash
while true; do
  status=$(kaggle kernels status fadhildev/yolov26x-img0160-2min-bytetrack)
  echo "Current status: $status"
  if [[ "$status" == *"complete"* ]]; then
    echo "Kernel completed. Downloading output..."
    python helper/download_output.py fadhildev/yolov26x-img0160-2min-bytetrack --output-dir ./pull_dir/output_yolov26x_tracking_counting_enterprise
    break
  elif [[ "$status" == *"error"* || "$status" == *"cancel"* || "$status" == *"fatal"* ]]; then
    echo "Kernel failed or was cancelled."
    break
  fi
  sleep 15
done
