import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

def filter_blue_color(image):
    """
    Filter the image to show blue color and detect grid lines.
    
    Parameters:
    image (numpy.ndarray): The input image.
    
    Returns:
    numpy.ndarray: Image with detected lines drawn on it.
    """
    # Converting to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)

    # Merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl, a, b))

    # Converting image from LAB Color model to BGR color space
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    hsv = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2HSV)
        # Define the range for blue color in HSV
    lower_blue = np.array([85, 70, 70])
    upper_blue = np.array([135, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Apply Gaussian blur to reduce noise before edge detection
    blur_mask = cv2.GaussianBlur(mask_blue, (5, 5), 0)
    # Create a mask for blue color

    blur = cv2.GaussianBlur( blur_mask , (5, 5), 0)
    edges = cv2.Canny(blur, 30, 130, apertureSize=3)
    

    lines = detect_lines(edges)
    if lines is not None:
        image_with_lines = np.copy(image)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)
        display_image(image_with_lines, 'Detected Lines')
    # Return both the binary mask and the image with detected lines
    return edges
def display_image(image, title):
    plt.figure(figsize=(8, 8))
    plt.title(title)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
def detect_lines(edges, threshold=80, min_line_length=30, max_line_gap=300):
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
    return lines
# Path to the video file
video_path = os.path.join('test_video', 'blue-grid-sh30.mp4')

# Open the video file
cap = cv2.VideoCapture(video_path)

# Read the first frame
ret, frame = cap.read()
if ret:
    cv2.imshow('Original Frame', frame)
    
    # Get the processed image with lines
    edges = filter_blue_color(frame)
    
    # Display results
    cv2.imshow('Blue Mask with Lines', edges)
    # cv2.imshow('Original with Lines', original_with_lines)
    
    # # Print the number of lines detected
    # if lines is not None:
    #     print(f"Number of lines detected: {len(lines)}")
    # else:
    #     print("No lines detected")
    
    # Wait for a key press and close the window
    cv2.waitKey(0)

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()