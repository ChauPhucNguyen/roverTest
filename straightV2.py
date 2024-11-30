import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import time
import os
import cv2
import math
from moviepy.editor import VideoFileClip
import requests
import serial_communication
import serial
import threading

# [All previous function definitions remain the same]

# New trigonometric steering functions
def calculate_steering_angle(prev_midpoint, current_midpoint):
    """
    Calculate steering angle and displacement based on midpoint changes.
    
    Args:
    prev_midpoint (tuple): Previous midpoint coordinates
    current_midpoint (tuple): Current midpoint coordinates
    
    Returns:
    dict: Steering information including angle, displacement, and direction
    """
    # Handle cases where midpoints are None
    if prev_midpoint is None or current_midpoint is None:
        return {
            'angle': 0,
            'displacement': 0,
            'direction': 'forward'
        }
    
    # Calculate the change in x and y coordinates
    dx = current_midpoint[0] - prev_midpoint[0]
    dy = current_midpoint[1] - prev_midpoint[1]
    
    # Calculate the angle using arctan2
    # arctan2 provides angle in radians, [-π, π]
    angle_radians = math.atan2(dy, dx)
    
    # Convert to degrees
    angle_degrees = math.degrees(angle_radians)
    
    # Calculate the magnitude of displacement
    displacement = math.sqrt(dx**2 + dy**2)
    
    return {
        'angle': angle_degrees,  # Steering angle
        'displacement': displacement,  # Distance moved
        'direction': 'left' if angle_degrees < 0 else 'right'
    }

def derive_steering_values(steering_info, max_speed=0.3):
    """
    Derive wheel speeds based on steering information.
    
    Args:
    steering_info (dict): Steering information from calculate_steering_angle
    max_speed (float): Maximum speed for the robot
    
    Returns:
    dict: Left and right wheel speeds
    """
    # Use the absolute value of angle and normalize
    normalized_angle = min(abs(steering_info['angle']) / 90.0, 1.0)
    
    # Base speed reduction factor
    speed_reduction = 1 - (normalized_angle * 0.5)  # Reduce speed by up to 50%
    
    if steering_info['direction'] == 'left':
        left_speed = max_speed * speed_reduction
        right_speed = max_speed
    else:
        left_speed = max_speed
        right_speed = max_speed * speed_reduction
    
    return {
        'left_speed': max(0, min(left_speed, 1)),
        'right_speed': max(0, min(right_speed, 1))
    }

def frame_processor(image):
    """
    Process the input frame to detect lane lines and control steering.
    
    Parameters:
    image: Single video frame.
    
    Returns:
    Processed frame with lane lines and annotations.
    """
    color_select = HSL_color_selection(image)
    gray = gray_scale(color_select)
    smooth = gaussian_smoothing(gray)
    edges = canny_detector(smooth)
    region = region_selection(edges)

    global prev_midpoint_left, prev_midpoint_right
    # Reset midpoints
    midpoint_left = None
    midpoint_right = None

    hough = hough_transform(region)
    left_line, right_line = lane_lines(image, hough)

    # Steering logic with trigonometric calculation
    if left_line is not None and right_line is not None:   
        # Calculate midpoints for left and right lines
        start_point_left, end_point_left = left_line
        start_point_right, end_point_right = right_line
        midpoint_left = (
            (start_point_left[0] + end_point_left[0]) // 2,  # x-coordinate
            (start_point_left[1] + end_point_left[1]) // 2   # y-coordinate
        )

        midpoint_right = (
            (start_point_right[0] + end_point_right[0]) // 2,  # x-coordinate
            (start_point_right[1] + end_point_right[1]) // 2   # y-coordinate
        )
        
        # Trigonometric steering for both lanes
        if prev_midpoint_left is not None and prev_midpoint_right is not None:
            # Calculate steering for left and right midpoints
            left_steering = calculate_steering_angle(prev_midpoint_left, midpoint_left)
            right_steering = calculate_steering_angle(prev_midpoint_right, midpoint_right)
            
            # Derive steering values (average of left and right)
            left_speeds = derive_steering_values(left_steering)
            right_speeds = derive_steering_values(right_steering)
            
            # Average the speeds for smoother control
            avg_left_speed = (left_speeds['left_speed'] + right_speeds['left_speed']) / 2
            avg_right_speed = (left_speeds['right_speed'] + right_speeds['right_speed']) / 2
            
            # Send steering command
            sendCommand(f'{{"T":1,"L":{avg_left_speed:.2f},"R":{avg_right_speed:.2f}}}')
            
            print(f"Steering - Left: {avg_left_speed:.2f}, Right: {avg_right_speed:.2f}")
    
    elif left_line is not None and right_line is None:
        # Calculate midpoint for left line
        start_point_left, end_point_left = left_line
        midpoint_left = (
            (start_point_left[0] + end_point_left[0]) // 2,  # x-coordinate
            (start_point_left[1] + end_point_left[1]) // 2   # y-coordinate
        )
        
        # Steering calculation for single lane
        if prev_midpoint_left is not None:
            left_steering = calculate_steering_angle(prev_midpoint_left, midpoint_left)
            steering_values = derive_steering_values(left_steering)
            
            sendCommand(f'{{"T":1,"L":{steering_values["left_speed"]:.2f},"R":{steering_values["right_speed"]:.2f}}}')
            print(f"Single Lane Steering - Left: {steering_values['left_speed']:.2f}, Right: {steering_values['right_speed']:.2f}")

    # Draw lane lines for visualization
    result = draw_lane_lines(image, [left_line, right_line])

    # Save current midpoints for the next iteration
    prev_midpoint_left = midpoint_left
    prev_midpoint_right = midpoint_right

    return result

# Rest of the code remains the same (webcam_video_processing function and main execution)