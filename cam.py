import cv2
import os
import time

def capture_images_loop(save_folder="Input", filename_prefix="captured_image", interval=10, count=10):
    # Ensure the save folder exists
    os.makedirs(save_folder, exist_ok=True)
    
    # Access the webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not access the webcam")
        return
    
    for i in range(count):
        # Capture a single frame
        ret, frame = cap.read()
        
        if ret:
            # Construct the file path with timestamp
            file_path = os.path.join(save_folder, f"{filename_prefix}_{i+1}.jpg")
            
            # Save the image
            cv2.imwrite(file_path, frame)
            print(f"Image saved at: {file_path}")
        else:
            print("Error: Could not capture image")
        
        # Wait for the specified interval before capturing the next image
        time.sleep(interval)
    
    # Release the webcam
    cap.release()
    cv2.destroyAllWindows()

# Capture and save images in a loop
capture_images_loop()
