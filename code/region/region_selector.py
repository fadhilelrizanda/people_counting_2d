import cv2
import numpy as np
import argparse
import json
import os

points = []
img_copy = None
frame_orig = None

def mouse_callback(event, x, y, flags, param):
    global points, img_copy, frame_orig
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            cv2.circle(img_copy, (x, y), 5, (0, 0, 255), -1)
            
            # Draw lines between consecutive points
            if len(points) > 1:
                cv2.line(img_copy, points[-2], points[-1], (0, 255, 0), 2)
            
            if len(points) == 4:
                # Close the polygon
                cv2.line(img_copy, points[3], points[0], (0, 255, 0), 2)
                
                poly_points = np.array(points, np.int32)
                poly_points = poly_points.reshape((-1, 1, 2))
                
                # Add a semi-transparent overlay to visualize the region
                overlay = img_copy.copy()
                cv2.fillPoly(overlay, [poly_points], (255, 0, 0))
                cv2.addWeighted(overlay, 0.3, img_copy, 0.7, 0, img_copy)
                
                # Save the results
                save_dir = os.path.dirname(os.path.abspath(__file__))
                out_img_path = os.path.join(save_dir, "region_preview.jpg")
                cv2.imwrite(out_img_path, img_copy)
                
                data = {
                    "region": points
                }
                out_json_path = os.path.join(save_dir, "region_data.json")
                with open(out_json_path, 'w') as f:
                    json.dump(data, f, indent=4)
                
                print(f"\n[SUCCESS] Image saved to {out_img_path}")
                print(f"[SUCCESS] Coordinates saved to {out_json_path}")
                print("You can now close the window (press 'q').")
                
        cv2.imshow("Image", img_copy)

def main():
    global img_copy, frame_orig, points
    parser = argparse.ArgumentParser(description="Select region points")
    parser.add_argument("--source", type=str, required=True, help="Path to video")
    args = parser.parse_args()
    
    cap = cv2.VideoCapture(args.source)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error: Could not read video frame.")
        return
        
    frame_orig = frame.copy()
    img_copy = frame.copy()
    
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", 1280, 720) 
    cv2.setMouseCallback("Image", mouse_callback)
    
    print("=======================================")
    print("1. Click 4 points to define your counting region.")
    print("2. The region will be automatically saved.")
    print("3. Press 'c' to clear if you make a mistake.")
    print("4. Press 'q' or 'ESC' to exit.")
    print("=======================================")
    
    cv2.imshow("Image", img_copy)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('c'):
            img_copy = frame_orig.copy()
            points.clear()
            cv2.imshow("Image", img_copy)
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
