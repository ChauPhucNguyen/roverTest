def visualize_lane_offsets(image, left_line, right_line):
    """Create a detailed visualization of lane offsets."""
    # Create a copy of the image to draw on
    viz_image = image.copy()
    
    # Image dimensions
    height, width = image.shape[:2]
    center_line_x = width // 2
    
    # Draw center line (more visible)
    cv2.line(viz_image, (center_line_x, 0), (center_line_x, height), (0, 255, 0), 3)
    
    # Visualization for left lane
    if left_line:
        # Original line in blue
        cv2.line(viz_image, left_line[0], left_line[1], (255, 0, 0), 3)
        
        # Line center point
        line_center_x = (left_line[0][0] + left_line[1][0]) // 2
        cv2.circle(viz_image, (line_center_x, left_line[0][1]), 10, (255, 0, 0), -1)
        
        # Draw line from center to lane center
        cv2.line(viz_image, (center_line_x, left_line[0][1]), (line_center_x, left_line[0][1]), (0, 255, 255), 2)
        
        # Offset distance annotation
        offset_dist = abs(line_center_x - center_line_x)
        cv2.putText(viz_image, f"Left Lane Offset: {offset_dist} px", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Visualization for right lane
    if right_line:
        # Original line in blue
        cv2.line(viz_image, right_line[0], right_line[1], (255, 0, 0), 3)
        
        # Line center point
        line_center_x = (right_line[0][0] + right_line[1][0]) // 2
        cv2.circle(viz_image, (line_center_x, right_line[0][1]), 10, (255, 0, 0), -1)
        
        # Draw line from center to lane center
        cv2.line(viz_image, (center_line_x, right_line[0][1]), (line_center_x, right_line[0][1]), (0, 255, 255), 2)
        
        # Offset distance annotation
        offset_dist = abs(line_center_x - center_line_x)
        cv2.putText(viz_image, f"Right Lane Offset: {offset_dist} px", 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    return viz_image