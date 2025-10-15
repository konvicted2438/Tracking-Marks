import cv2
import os
import numpy as np
from pathlib import Path

def extract_frames(video_path, output_dir="extracted_frames", interval=1, quality=100):
    """
    Extract frames from a video at specified time intervals and save as high-quality images.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        interval: Time interval in seconds between frames
        quality: Image quality (1-100) for JPEG compression
    """
    video_path = Path(video_path)
    if video_path.suffix.lower() not in {".mp4", ".avi"}:
        raise ValueError(f"Unsupported video type: {video_path.suffix}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    
    # Open the video file
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"Video info: {total_frames} frames, {fps:.2f} FPS, Duration: {duration:.2f} seconds")
    
    # Calculate frame interval (frames to skip)
    frame_interval = int(fps * interval)
    
    # Extract frames
    count = 0
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Save frame at the specified interval
        if frame_count % frame_interval == 0:
            # Format the timestamp (seconds)
            timestamp = frame_count / fps
            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)
            
            # Create filename with timestamp
            filename = output_dir / f"exp_tensile_frame_{count:03d}_{minutes:02d}m_{seconds:02d}s.png"
            
            # Save as high quality PNG (lossless)
            cv2.imwrite(str(filename), frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])  # 0 = no compression (highest quality)
            
            print(f"Saved frame at {timestamp:.2f}s to {filename}")
            count += 1
            
        frame_count += 1
    
    # Release resources
    cap.release()
    print(f"Extraction complete. Saved {count} frames.")

if __name__ == "__main__":
    video_path = Path("test_video/wshape-sim.mp4")
    output_dir = Path("extracted_frames_wshape-sim")
    extract_frames(video_path, output_dir=output_dir)