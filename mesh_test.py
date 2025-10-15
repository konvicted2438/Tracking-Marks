import cv2
import os
import numpy as np

def preprocess_frame(frame):
    """Preprocess the frame using your existing method"""
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply morphological closing using a disk-shaped structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 40))
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    # Adjust the source image by dividing it by the result of the closing operation
    src_adjusted = cv2.divide(gray, closed, scale=255)

    # Convert the adjusted image to a binary image
    _, binary = cv2.threshold(src_adjusted, 200, 320, cv2.THRESH_BINARY)
    
    return binary

def detect_chess_pattern(binary):
    """Detect chessboard corners and edges"""
    # Find chessboard corners
    pattern_size = (4, 14)  # Adjust based on your pattern size
    ret, corners = cv2.findChessboardCorners(binary, pattern_size, None)
    
    # Apply Canny edge detection
    edges = cv2.Canny(binary, 40, 60)
    
    if ret:
        # Refine corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(binary, corners, (11,11), (-1,-1), criteria)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return True, corners, contours, edges
    
    return False, None, None, edges

def main():
    # Path to the video file
    video_path = os.path.join('test_video', 'blue-grid-sh30.mp4')

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Create windows for displaying results
    cv2.namedWindow('Original Frame', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Processed Frame', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Detected Pattern', cv2.WINDOW_NORMAL)

    while True:
        # Read a frame
        ret, frame = cap.read()
        if not ret:
            break

        # Get frame dimensions
        height, width = frame.shape[:2]

        # Preprocess the frame
        binary = preprocess_frame(frame)

        # Detect chess pattern
        success, corners, contours, edges = detect_chess_pattern(binary)
        print(success)
        # Create visualization
        vis_frame = frame.copy()
        if success:
            # Draw corners
            cv2.drawChessboardCorners(vis_frame, (4, 15), corners, True)
            # Draw contours
            cv2.drawContours(vis_frame, contours, -1, (0,255,0), 2)

        # Define the new width and calculate the new height
        new_width = 960
        aspect_ratio = height / width
        new_height = int(new_width * aspect_ratio)

        # Resize frames for display
        resized_original = cv2.resize(frame, (new_width, new_height))
        resized_edges = cv2.resize(edges, (new_width, new_height))
        resized_vis = cv2.resize(vis_frame, (new_width, new_height))

        # Display the frames
        cv2.imshow('Original Frame', resized_original)
        cv2.imshow('Processed Frame', resized_edges)
        cv2.imshow('Detected Pattern', resized_vis)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()