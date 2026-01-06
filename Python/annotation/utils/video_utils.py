import cv2
import numpy as np
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def save_image_worker(args):
    """
    Independent worker function to save the image.
    Run on separate CPUs to avoid blocking the main video reader.
    """
    img_data, save_path = args
    cv2.imwrite(save_path, img_data)

def extract_frames_fast(video_path, output_folder, interval_sec, crop_size, margin_right, margin_top, file_prefix="frame"):
    """
    Extracts frames from a video with cropping and saves them using multiprocessing.
    
    Args:
        video_path (str): Path to the input video.
        output_folder (str): Folder to save extracted images.
        interval_sec (float): Time interval between frames in seconds.
        crop_size (int): Height and Width of the square crop.
        margin_right (int): Margin from the right edge of the video.
        margin_top (int): Margin from the top edge of the video.
        file_prefix (str): Prefix for the saved filenames.
    """
    # 1. Setup
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    os.makedirs(output_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    # 2. Metadata
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Unused but good to know
    
    # 3. Crop Configuration
    x_start = width - margin_right - crop_size
    y_start = margin_top
    
    # Ensure crop boundaries are valid
    if x_start < 0: x_start = 0
    if y_start < 0: y_start = 0

    frame_step = int(np.round(fps * interval_sec))
    if frame_step < 1: frame_step = 1

    print(f"--- Processing: {os.path.basename(video_path)} ---")
    print(f"FPS: {fps:.2f} | Step: {frame_step} frames")
    print(f"Crop: {crop_size}x{crop_size} at ({x_start}, {y_start})")
    
    # 4. Initialize Worker Pool
    # Reserve 1 CPU for the main reader loop
    worker_count = max(1, cpu_count() - 1) 
    print(f"Using {worker_count} background processes for saving.")
    
    pool = Pool(processes=worker_count)
    
    current_idx = 0
    saved_count = 0

    # 5. Fast Reader Loop
    print("Starting extraction...")
    with tqdm(total=total_frames, unit="frame") as pbar:
        while True:
            ret, frame = cap.read()

            if not ret:
                # Handle potential initial corruption or end of file
                if current_idx < 100 and current_idx < total_frames: 
                    current_idx += 1
                    pbar.update(1)
                    continue
                else:
                    break

            if current_idx % frame_step == 0:
                # Crop Logic
                # Ensure the crop doesn't exceed frame dimensions
                cropped = frame[y_start : y_start + crop_size, 
                                x_start : x_start + crop_size]
                
                # Construct filename
                filename = f"{file_prefix}_{current_idx:06d}.jpg"
                save_path = os.path.join(output_folder, filename)
                
                # Async Save
                pool.apply_async(save_image_worker, args=((cropped, save_path),))
                saved_count += 1

            current_idx += 1
            pbar.update(1)

    cap.release()
    
    print("\nReading finished. Waiting for remaining file writes to complete...")
    pool.close()
    pool.join()
    print(f"Done! Saved {saved_count} images to '{output_folder}'.")