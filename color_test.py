import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

def resize_image(image, new_width):
    """
    Resize the image to the specified width while maintaining the aspect ratio.
    
    Parameters:
    image (numpy.ndarray): The input image.
    new_width (int): The desired width of the resized image.
    
    Returns:
    numpy.ndarray: The resized image.
    """
    height, width = image.shape[:2]
    aspect_ratio = height / width
    new_height = int(new_width * aspect_ratio)
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image

def detect_features(image):
    # Convert to grayscale first
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. Harris Corner Detection
    harris = cv2.cornerHarris(gray, blockSize=5, ksize=5, k=0.1)
    harris = cv2.dilate(harris, None)
    
    # Create a copy of the original image for visualization
    corner_img = image.copy()
    corner_img[harris > 0.01 * harris.max()] = [0, 0, 255]  # Mark corners in red
    
    # 2. Shi-Tomasi Corner Detection
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.1, minDistance=50)
    if corners is not None:
        corners = np.int32(corners)
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(corner_img, (x, y), 3, [255, 0, 0], -1)  # Draw blue circles
            
    # 3. Canny Edge Detection
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    
    return corner_img, edges

def increase_sharpness(image, iterations=1):
    """
    Increase the sharpness of the image using a sharpening kernel.
    
    Parameters:
    image (numpy.ndarray): The input image.
    iterations (int): Number of times to apply the sharpening filter.
    
    Returns:
    numpy.ndarray: The sharpened image.
    """
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    sharpened = image.copy()
    for _ in range(iterations):
        sharpened = cv2.filter2D(sharpened, -1, kernel)
    return sharpened

# Path to the video file
video_path = os.path.join('test_video', 'test_video/blue-grid-sh30.mp4') 

# Open the video file
cap = cv2.VideoCapture(video_path)

# Read the first frame
ret, frame = cap.read()
if ret:
    # Print the size of the frame
    height, width, channels = frame.shape
    print(f"Frame size: {width}x{height}, Channels: {channels}")

    frame = cv2.medianBlur(frame, 5)
    frame = cv2.GaussianBlur(frame, (5, 5), 1)
    
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range for yellow color in HSV
    lower_yellow = np.array([15, 110, 110])
    upper_yellow = np.array([40, 255, 255])

    # Define the range for blue color in HSV
    lower_blue = np.array([75, 50, 50])
    upper_blue = np.array([145, 255, 255])

    # Create masks for yellow and blue colors
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Combine the masks
    mask = cv2.bitwise_or(mask_yellow, mask_blue)

    # Apply the mask to the original frame
    result = cv2.bitwise_and(frame, frame, mask=mask)
    
    # Increase the sharpness of the result
    sharpened_result = increase_sharpness(result, iterations=1)  # Apply sharpening filter multiple times

    corner_result, edge_result = detect_features(sharpened_result)

    # Resize the frame proportionally
    resized_frame = resize_image(sharpened_result, 960)
    # corner_result = resize_image(corner_result, 960)
    # edge_result = resize_image(edge_result, 960)
    
    # Display the frame with only yellow and blue colors
    cv2.imshow('Frame with Yellow and Blue Colors', resized_frame)
    cv2.imshow('Corner Detection', corner_result)
    cv2.imshow('Edge Detection', edge_result)
    # Wait for a key press and close the window
    cv2.waitKey(0)

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()