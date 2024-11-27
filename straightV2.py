import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import time
import os
import cv2
import requests

def convert_hsl(image):
    """Convert an image from RGB to HSL."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

def HSL_color_selection(image):
    """Apply color selection to the HSL images to blackout everything except for white lane lines."""
    converted_image = convert_hsl(image)
    
    # White color mask
    lower_threshold = np.uint8([0, 200, 0])  # H, L, S
    upper_threshold = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(converted_image, lower_threshold, upper_threshold)
    
    # Combine masks (only white in this case)
    masked_image = cv2.bitwise_and(image, image, mask=white_mask)
    
    return masked_image

def gray_scale(image):
    """Convert an image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def gaussian_smoothing(image, kernel_size=5):
    """Apply Gaussian smoothing to the image."""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def canny_detector(image, low_threshold=50, high_threshold=150):
    """Apply Canny edge detection to the image."""
    return cv2.Canny(image, low_threshold, high_threshold)

def region_selection(image):
    """Determine and cut the region of interest in the input image."""
    mask = np.zeros_like(image)   
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    
    rows, cols = image.shape[:2]
    
    top_left     = [cols * 0.3, rows * 0.60]
    top_right    = [cols * 0.7, rows * 0.60]
    bottom_left  = [cols * 0.01, rows * 0.90]  
    bottom_right = [cols * 0.99, rows * 0.90]
    
    vertices = np.array([[bottom_left, top_left, bottom_right, top_right]], dtype=np.int32)

    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    
    return masked_image

def hough_transform(image):
    """Determine lines in the image using the Hough Transform."""
    rho = 1              
    theta = np.pi / 180  
    threshold = 50       
    minLineLength = 25   
    maxLineGap = 100     
    lines = cv2.HoughLinesP(image, rho=rho, theta=theta, threshold=threshold,
                             minLineLength=minLineLength, maxLineGap=maxLineGap)
    return lines if lines is not None else []

def average_slope_intercept(lines, slope_thresh=0.3):
    """Find the slope and intercept of the left and right lanes of each image."""
    if len(lines) == 0:
        return None, None

    left_lines = []  
    left_weights = []  
    right_lines = []  
    right_weights = []  
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:  # Skip vertical lines
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - (slope * x1)
            length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))

            if slope < 0:
                left_lines.append((slope, intercept))
                left_weights.append(length)
            else:
                right_lines.append((slope, intercept))
                right_weights.append(length)

    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
    
    return left_lane, right_lane

def pixel_points(y1, y2, line):
    """Converts the slope and intercept of each line into pixel points."""
    if line is None:
        return None
    slope, intercept = line
    
    if slope == 0:  # Prevent division by zero
        return None
    
    # Calculate x1 and x2 for the given y1 and y2
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    
    return ((x1, int(y1)), (x2, int(y2)))

def calculate_lane_offset(line, image_width, desired_offset_percent=0.25, is_left_lane=True):
    if line is None:
        return None, "STOP"
    
    # Calculate the x-coordinates of the line
    line_x1, line_x2 = line[0][0], line[1][0]
    line_center_x = (line_x1 + line_x2) / 2
    
    # Calculate desired offset
    desired_offset = image_width * desired_offset_percent
    
    # Determine steering based on lane side
    if is_left_lane:
        # Left lane: offset to the right
        target_x = line_center_x + desired_offset
        if target_x > image_width / 2:
            steering_action = "FORWARD"
        else:
            steering_action = "RIGHT"
    else:
        # Right lane: offset to the left
        target_x = line_center_x - desired_offset
        if target_x < image_width / 2:
            steering_action = "FORWARD"
        else:
            steering_action = "LEFT"
    
    # Calculate new line coordinates to maintain offset
    x_diff = int(target_x - line_center_x)
    new_line = (
        (line[0][0] + x_diff, line[0][1]),
        (line[1][0] + x_diff, line[1][1])
    )
    
    return new_line, steering_action

def lane_lines(image, lines):
    """Create full length lines from pixel points with offset calculation."""
    left_lane, right_lane = average_slope_intercept(lines)
    y1 = image.shape[0]
    y2 = y1 * 0.6
    
    left_line = pixel_points(y1, y2, left_lane)
    right_line = pixel_points(y1, y2, right_lane)
    
    # Process single lane scenarios with offset
    image_width = image.shape[1]
    
    if left_line is None and right_line is not None:
        # Only right lane detected
        right_line, steering_action = calculate_lane_offset(right_line, image_width, 0.25, is_left_lane=False)
        return right_line, None
    
    elif right_line is None and left_line is not None:
        # Only left lane detected
        left_line, steering_action = calculate_lane_offset(left_line, image_width, 0.25, is_left_lane=True)
        return None, left_line
    
    return left_line, right_line

def draw_lane_lines(image, lines, color=[0, 0, 255], thickness=15):
    """Draw lines onto the input image."""
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line, color, thickness)
    return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)

def frame_processor(image):
    """Process the input frame to detect lane lines."""
    color_select = HSL_color_selection(image)
    gray = gray_scale(color_select)
    smooth = gaussian_smoothing(gray)
    edges = canny_detector(smooth)
    region = region_selection(edges)

    cv2.imshow('Region of interest with 1st triangle', region)
    hough = hough_transform(region)
    left_line, right_line = lane_lines(image, hough)
    
    # Determine steering action and send command
    steering_action = "STOP"
    if left_line and right_line:
        steering_action = "FORWARD"
    elif left_line:
        _, steering_action = calculate_lane_offset(left_line, image.shape[1], 0.25, is_left_lane=True)
    elif right_line:
        _, steering_action = calculate_lane_offset(right_line, image.shape[1], 0.25, is_left_lane=False)
    
    print(f"Steering Action: {steering_action}")
    
    # Draw lane lines for visualization
    result = draw_lane_lines(image, [left_line, right_line])
    
    return result

def webcam_video_processing():
    """Capture video from the webcam and process it for lane detection."""
    cap = cv2.VideoCapture(0)  # Use 0 for the default webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        # Process the frame
        processed_frame = frame_processor(frame)
        
        processing_time = time.time() - start_time
        fps = 1 / processing_time if processing_time > 0 else 0

        # Display FPS on the frame
        cv2.putText(processed_frame, f"FPS: {fps:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('Lane Detection - White Lines Only', processed_frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(200) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the webcam processing
webcam_video_processing()