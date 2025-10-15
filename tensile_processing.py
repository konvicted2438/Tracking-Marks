import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.spatial import distance
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
import json
from datetime import datetime

def init_kalman_filter(x, y, radius):
    """Initialize a Kalman filter for tracking a circle."""
    kf = KalmanFilter(dim_x=6, dim_z=3)  # State: [x, y, r, dx, dy, dr], Measurement: [x, y, r]
    
    # State transition matrix
    kf.F = np.array([
        [1, 0, 0, 1, 0, 0],  # x = x + dx
        [0, 1, 0, 0, 1, 0],  # y = y + dy
        [0, 0, 1, 0, 0, 1],  # r = r + dr
        [0, 0, 0, 1, 0, 0],  # dx = dx
        [0, 0, 0, 0, 1, 0],  # dy = dy
        [0, 0, 0, 0, 0, 1],  # dr = dr
    ])
    
    # Measurement function
    kf.H = np.array([
        [1, 0, 0, 0, 0, 0],  # measure x
        [0, 1, 0, 0, 0, 0],  # measure y
        [0, 0, 1, 0, 0, 0],  # measure r
    ])
    
    # Measurement noise - DRASTICALLY DECREASED to trust measurements much more
    kf.R = np.eye(3) * 2.0  # Further reduced from 5.0 to 2.0
    
    # Process noise - SIGNIFICANTLY INCREASED for more dynamic tracking
    kf.Q = np.eye(6) * 0.3  # Increased from 0.1 to 0.3
    kf.Q[3:, 3:] *= 0.05   # Increased from 0.01 to 0.05 for velocity components
    
    # Initial state
    kf.x = np.array([[x], [y], [radius], [0], [0], [0]])
    
    # Initial covariance - GREATLY INCREASED for much faster adaptation
    kf.P = np.eye(6) * 20.0   # Increased from 10.0 to 20.0
    
    return kf

def save_trajectories_to_json(trajectories, video_name, frame_timestamps=None, output_dir="trajectory_data"):
    """
    Save circle trajectories to a JSON file with timestamps.
    
    Args:
        trajectories (dict): Dictionary of tracked circle trajectories
        video_name (str): Name of the processed video (for filename)
        frame_timestamps (list, optional): List of timestamps for each frame
        output_dir (str): Directory to save the JSON file
    
    Returns:
        str: Path to the saved JSON file
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(video_name))[0]
    json_filename = f"{base_name}_trajectories_{timestamp}.json"
    json_path = os.path.join(output_dir, json_filename)
    
    # Reformat the trajectories for JSON
    # Format: {frame_index: {point_id: {"x": x, "y": y, "r": r}, ...}, ...}
    formatted_data = {
        "metadata": {
            "video_name": video_name,
            "export_timestamp": timestamp,
            "num_points": len(trajectories),
            "total_frames": max(len(traj['x']) for traj in trajectories.values())
        },
        "frames": {}
    }
    
    # Get the maximum number of frames across all trajectories
    max_frames = max(len(traj['x']) for traj in trajectories.values())
    
    # Populate frame-by-frame data
    for frame_idx in range(max_frames):
        frame_data = {}
        
        # Add frame timestamp if available
        if frame_timestamps and frame_idx < len(frame_timestamps):
            frame_data["timestamp"] = frame_timestamps[frame_idx]
        else:
            # If no timestamps provided, use frame index as relative time (assuming constant FPS)
            frame_data["timestamp"] = frame_idx
        
        # Add point data for this frame
        frame_data["points"] = {}
        for point_id, traj in trajectories.items():
            # Check if this trajectory has data for this frame
            if frame_idx < len(traj['x']):
                frame_data["points"][str(point_id)] = {
                    "x": float(traj['x'][frame_idx]),
                    "y": float(traj['y'][frame_idx]),
                    "r": float(traj['r'][frame_idx])
                }
        
        formatted_data["frames"][str(frame_idx)] = frame_data
    
    # Add summary statistics for each point
    formatted_data["statistics"] = {}
    for point_id, traj in trajectories.items():
        if len(traj['x']) > 1:
            formatted_data["statistics"][str(point_id)] = {
                "frames_tracked": len(traj['x']),
                "max_x_displacement": float(max(traj['x']) - min(traj['x'])),
                "max_y_displacement": float(max(traj['y']) - min(traj['y'])),
                "average_radius": float(sum(traj['r']) / len(traj['r']))
            }
    
    # Write to JSON file
    with open(json_path, 'w') as json_file:
        json.dump(formatted_data, json_file, indent=2)
    
    print(f"Trajectory data saved to {json_path}")
    return json_path

def preprocess_frame(frame):
    """Preprocess the frame for circle detection."""
    # Apply medium blur to reduce noise
    blurred = cv2.medianBlur(frame, 3)
    
    # Convert to LAB color space
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Create red mask using LAB
    _, red_mask = cv2.threshold(a_channel, 130, 255, cv2.THRESH_BINARY)
    
    # Create red mask using HSV
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    # First red range (further expanded)
    lower_red1 = np.array([0, 40, 40])    # Further reduced saturation and value thresholds
    upper_red1 = np.array([45, 255, 255]) # Increased upper hue to include more orange-reddish tones
        
    # Second red range (further expanded)
    lower_red2 = np.array([140, 40, 40])  # Reduced lower bound to include more purple-red tones
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    hsv_mask = cv2.bitwise_or(mask1, mask2)
    
    # Combine LAB and HSV masks
    combined_mask = cv2.bitwise_and(red_mask, hsv_mask)
    
    # Clean the mask
    kernel = np.ones((3, 3), np.uint8)
    mask_cleaned = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Apply Gaussian blur for circle detection
    gaussian_blurred = cv2.GaussianBlur(mask_cleaned, (9, 9), 2)
    
    return blurred, mask_cleaned, gaussian_blurred

def detect_colored_centroids(mask):
    """
    Detect colored regions in the mask and calculate their centroids.
    Returns list of (x, y, radius) tuples, where radius is approximated from area.
    """
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    centroids = []
    min_area = 0.1  # Decreased from 20 to detect smaller regions
    
    for contour in contours:
        area = cv2.contourArea(contour)

        # Filter out small contours (noise)
        if area < min_area:
            continue

        # Calculate moments of the contour
        M = cv2.moments(contour)

        # Calculate centroid coordinates
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Approximate radius from the area for compatibility with existing code
            radius = int(np.sqrt(area / np.pi))

            centroids.append((cx, cy, radius))
    #print(f"Found {centroids} contours")

    return centroids

def detect_circles(gaussian_blurred):
    """Detect circles in the preprocessed image."""
    circles = cv2.HoughCircles(
        gaussian_blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1,
        minDist=40,
        param1=70,
        param2=20,
        minRadius=3,
        maxRadius=60
    )
    
    if circles is None:
        return []
    
    circles = np.uint16(np.around(circles[0]))
    return [(circle[0], circle[1], circle[2]) for circle in circles]  # (x, y, radius)

def match_circles_to_trackers(detections, trackers, max_distance=50):
    """Match detected circles to existing trackers."""
    if not trackers:
        return [], list(range(len(detections)))
    
    if not detections:
        return list(range(len(trackers))), []
    
    # Calculate distance matrix
    cost_matrix = np.zeros((len(trackers), len(detections)))
    for i, tracker in enumerate(trackers):
        for j, detection in enumerate(detections):
            # Compute distance between tracker prediction and detection
            cost_matrix[i, j] = distance.euclidean(
                (tracker['kf'].x[0, 0], tracker['kf'].x[1, 0]), 
                (detection[0], detection[1])
            )
    
    # Apply Hungarian algorithm for optimal assignment
    tracker_indices, detection_indices = linear_sum_assignment(cost_matrix)
    
    # Filter assignments with distance above threshold
    matches = []
    unmatched_trackers = list(range(len(trackers)))
    unmatched_detections = list(range(len(detections)))
    
    for t_idx, d_idx in zip(tracker_indices, detection_indices):
        if cost_matrix[t_idx, d_idx] <= max_distance:
            matches.append((t_idx, d_idx))
            unmatched_trackers.remove(t_idx)
            unmatched_detections.remove(d_idx)
    
    return unmatched_trackers, unmatched_detections, matches

def draw_circles(frame, circles, color=(146, 221, 242), cross_color=(65, 65, 65)):
    """Draw circles and crosses on the frame."""
    for circle_data in circles:
        center = (int(circle_data[0]), int(circle_data[1]))
        radius = int(circle_data[2])
        
        # Draw circle with specified color (#F2DD92 in BGR format)
        cv2.circle(frame, center, radius, color, 2)
        
        # Draw cross with specified color (#414141 in BGR format)
        cross_size = 5
        x, y = center
        cv2.line(frame, (x-cross_size, y), (x+cross_size, y), cross_color, 2)
        cv2.line(frame, (x, y-cross_size), (x, y+cross_size), cross_color, 2)
    
    return frame

def arrange_points_in_grid(points, grid_rows=3, grid_cols=3):
    """Arrange detected points in a grid pattern."""
    if len(points) < grid_rows * grid_cols:
        return points  # Not enough points to arrange
    
    # Extract x, y coordinates
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    
    # Sort points by x and y coordinates
    x_sorted_indices = np.argsort(x_coords)
    
    # Group points by rows
    grid_points = []
    for i in range(grid_rows):
        row_start = i * grid_cols
        row_end = (i + 1) * grid_cols
        row_indices = x_sorted_indices[row_start:row_end]
        
        # Sort points in this row by y-coordinate
        row_indices = sorted(row_indices, key=lambda idx: y_coords[idx])
        
        # Add points to the grid
        for idx in row_indices:
            grid_points.append(points[idx])
    
    return grid_points

def detect_and_track_circles(video_path, output_path=None, show_video=True):
    """
    Detect and track red regions in a video file using Kalman filtering.
    """
    # Check if file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} not found.")
        return []
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return []
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup video writer if output path is provided
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize trackers list
    trackers = []  # List to store active trackers
    all_circles = []  # List to store detected circles for each frame
    frame_timestamps = []  # List to store timestamps for each frame
    frame_count = 0
    expected_points = 7  # We expect 9 points (3x3 grid)
    
    print("Processing video...")
    
    # For trajectory plotting
    trajectories = {}  # Dictionary to store trajectories for each tracker
    
    # Get video FPS for timestamp calculation
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Calculate timestamp for this frame in milliseconds
        # For constant FPS videos, timestamp = frame_index / fps * 1000
        frame_timestamp = (frame_count - 1) / fps * 1000.0  # ms
        frame_timestamps.append(frame_timestamp)
        
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames, current time: {frame_timestamp/1000:.2f}s")
        
        # Preprocess the frame
        blurred, mask_cleaned, gaussian_blurred = preprocess_frame(frame)
        
        # Use the new centroid detection instead of circle detection
        detected_centroids = detect_colored_centroids(mask_cleaned)
        #print(f"Detected centroids: {detected_centroids}")
        # Prediction step for all existing trackers

        print(trackers)
        for tracker in trackers:
            tracker['kf'].predict()
        #print(trackers)
        # If this is the first frame with detections and we have enough centroids
        if not trackers and len(detected_centroids) >= expected_points:
            # Take the first expected_points centroids and arrange them in a grid
            points_to_track = detected_centroids[:expected_points]
            arranged_points = arrange_points_in_grid(points_to_track)
            
            # Initialize trackers
            for i, (x, y, r) in enumerate(arranged_points):
                kf = init_kalman_filter(x, y, r)
                trackers.append({
                    'id': i,
                    'kf': kf,
                    'age': 0,
                    'total_visible': 1,
                    'consecutive_invisible': 0
                })
                trajectories[i] = {'x': [float(x)], 'y': [float(y)], 'r': [float(r)]}
        
        else:
            # Match detections to existing trackers
            #print(f"Trackers: {trackers}", f"Detected centroids: {detected_centroids}")
            if trackers and detected_centroids:
                unmatched_trackers, unmatched_detections, matches = match_circles_to_trackers(
                    detected_centroids, trackers
                )
                
                # Update matched trackers with new detections
                for t_idx, d_idx in matches:
                    x, y, r = detected_centroids[d_idx]
                    z = np.array([[x], [y], [r]])
                    trackers[t_idx]['kf'].update(z)
                    trackers[t_idx]['age'] += 1
                    trackers[t_idx]['total_visible'] += 1
                    trackers[t_idx]['consecutive_invisible'] = 0
                    
                    # Update trajectory
                    tracker_id = trackers[t_idx]['id']
                    if tracker_id not in trajectories:
                        trajectories[tracker_id] = {'x': [], 'y': [], 'r': []}
                    trajectories[tracker_id]['x'].append(float(x))
                    trajectories[tracker_id]['y'].append(float(y))
                    trajectories[tracker_id]['r'].append(float(r))
                
                # Handle unmatched trackers (missing detections)
                for t_idx in unmatched_trackers:
                    trackers[t_idx]['age'] += 1
                    trackers[t_idx]['consecutive_invisible'] += 1
                    
                    # Add prediction to trajectory
                    tracker_id = trackers[t_idx]['id']
                    x = trackers[t_idx]['kf'].x[0, 0]
                    y = trackers[t_idx]['kf'].x[1, 0]
                    r = trackers[t_idx]['kf'].x[2, 0]
                    trajectories[tracker_id]['x'].append(x)
                    trajectories[tracker_id]['y'].append(y)
                    trajectories[tracker_id]['r'].append(r)
                
                # Initialize new trackers for unmatched detections if needed
                if len(trackers) < expected_points:
                    for d_idx in unmatched_detections:
                        if len(trackers) >= expected_points:
                            break
                        
                        x, y, r = detected_centroids[d_idx]
                        new_id = max([t['id'] for t in trackers]) + 1 if trackers else 0
                        kf = init_kalman_filter(x, y, r)
                        trackers.append({
                            'id': new_id,
                            'kf': kf,
                            'age': 0,
                            'total_visible': 1,
                            'consecutive_invisible': 0
                        })
                        trajectories[new_id] = {'x': [x], 'y': [y], 'r': [r]}
        #print(f"Active trackers: {len(trackers)}")

        # Remove trackers that have been invisible for too long
        trackers = [t for t in trackers if t['consecutive_invisible'] < 30]
        #print(f"Active trackers: {len(trackers)}")
        # Extract current circle positions from trackers
        current_circles = []
        for tracker in trackers:
            x = tracker['kf'].x[0, 0]
            y = tracker['kf'].x[1, 0]
            r = tracker['kf'].x[2, 0]
            current_circles.append((x, y, r))
            
            # Draw tracker ID
            cv2.putText(frame, f"#{tracker['id']}", 
                       (int(x) + 15, int(y)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        #print(f"Current circles: {current_circles}")
        # Draw circles on the frame
        draw_circles(frame, current_circles)
        
        all_circles.append(current_circles)
        
        # Write the frame to output video if specified
        if writer:
            writer.write(frame)
        
        # Display the processing steps (for debugging)
        if show_video:
            cv2.imshow('Tracking Circles', frame)
            cv2.imshow('Red Mask', mask_cleaned)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Clean up
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    # Plot the detected circle trajectories
    plt.figure(figsize=(12, 10))
    for tracker_id, traj in trajectories.items():
        plt.plot(traj['x'], traj['y'], '-', lw=1.5, label=f"Point #{tracker_id}")
    
    plt.grid(True)
    plt.title('Circle Center Trajectories')
    plt.xlabel('X position (pixels)')
    plt.ylabel('Y position (pixels)')
    plt.axis('equal')
    plt.legend()
    plt.savefig('trajectories.png')
    plt.show()
    
    print(f"Processed {frame_count} frames with {len(trackers)} tracked points")
    return trajectories, all_circles, frame_timestamps

def main():
    video_path = os.path.join('test_video', 'tensile_gen2.mp4')
    output_path = 'tensile_tracked_centroids_gen2.avi'
    
    # Detect and track centroids (still using the same function name)
    trajectories, all_circles, frame_timestamps = detect_and_track_circles(video_path, output_path, show_video=True)
    
    # Analyze the tracked trajectories
    if trajectories:
        # Save trajectories to JSON file with timestamps
        json_path = save_trajectories_to_json(trajectories, video_path, frame_timestamps)
        
        print(f"\nTrajectory data saved to JSON: {json_path}")
        print("You can use this JSON file for further analysis or visualization.")

if __name__ == "__main__":
    main()