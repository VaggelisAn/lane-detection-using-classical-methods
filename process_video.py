# Video Lane Detection
# Processes a video file and applies lane detection to each frame

import cv2
import numpy as np
from lane_detection import (
    calibrate_camera,
    crop_image_height,
    convert_bgr_image_to_gray,
    apply_gaussian_blur,
    sobel_x_threshold,
    hls_lighting_threshold,
    birds_eye_view,
    sliding_window_lane_detection,
    illustrate_lane_space,
    reverse_birds_eye_view,
    uncrop_image_height
)

# Camera parameters
K = np.loadtxt('resources/mtx.csv', delimiter=',')
D = np.loadtxt('resources/dist.csv', delimiter=',')

# Processing parameters
top_ratio = 0.55
bottom_ratio = 0.7

# Bird's-eye-view parameters
src_perc = np.float32([
    (0.15, 0.9),   # bottom-left
    (0.9, 0.9),    # bottom-right
    (0.72, 0.10),  # top-right
    (0.6, 0.10)])  # top-left
dst_perc = np.float32([
    (0.20, 0.80),   # bottom-left
    (0.80, 0.80),   # bottom-right
    (0.80, 0.20),   # top-right
    (0.20, 0.20)    # top-left
])

def process_frame(frame):
    """
    Apply lane detection to a single frame.
    
    Parameters:
    - frame (np.ndarray): Input video frame in BGR format
    
    Returns:
    - np.ndarray: Frame with detected lanes overlaid
    """
    try:
        # Calibrate and undistort
        undistorted_img = calibrate_camera(frame, K, D)
        
        # Crop image
        cropped_img = crop_image_height(undistorted_img, top_ratio=top_ratio, bottom_ratio=bottom_ratio)
        
        # Convert to grayscale and blur
        gray_img = convert_bgr_image_to_gray(cropped_img)
        blurred_img_gray = apply_gaussian_blur(gray_img)
        
        # Apply thresholding
        sobel_img = sobel_x_threshold(blurred_img_gray, thresh_min=25, thresh_max=75)
        hls_img = hls_lighting_threshold(cropped_img, thresh_min=170, thresh_max=255)
        
        # Combine thresholds
        combined_img = np.zeros_like(sobel_img)
        combined_img[(hls_img == 1) | (sobel_img == 1)] = 1
        
        # Bird's eye view
        birds_eye_image = birds_eye_view(combined_img, src_perc, dst_perc)
        
        # Sliding window detection
        left_fit, right_fit, debug_img = sliding_window_lane_detection(
            birds_eye_image,
            nwindows=13,
            margin=70,
            minpix=70
        )
        
        # Check if lane was detected
        if left_fit is None or right_fit is None:
            # Return original frame if no lane detected
            return frame
        
        # Illustrate lane space
        lane_space = illustrate_lane_space(birds_eye_image, left_fit, right_fit)
        
        # Reverse bird's eye view
        unwarped_lane_space = reverse_birds_eye_view(lane_space, src_perc, dst_perc)
        
        # Convert to BGR for overlay
        unwarped_lane_space_bgr = cv2.cvtColor(unwarped_lane_space, cv2.COLOR_GRAY2BGR)
        
        # Uncrop to original height
        uncropped_lane_space = uncrop_image_height(
            unwarped_lane_space_bgr,
            frame.shape,
            top_ratio=top_ratio,
            bottom_ratio=bottom_ratio
        )
        
        # Make detected lane red
        uncropped_lane_space[:, :, 2] = uncropped_lane_space[:, :, 2] * 255
        
        # Overlay on original frame
        result = cv2.addWeighted(frame.astype(np.uint8), 1.0, uncropped_lane_space.astype(np.uint8), 1.0, 0)
        
        return result
    
    except Exception as e:
        print(f"Error processing frame: {e}")
        return frame

def process_video(input_path, output_path=None, display=True):
    """
    Process a video and apply lane detection to each frame.
    
    Parameters:
    - input_path (str): Path to input video
    - output_path (str): Path to save output video (optional)
    - display (bool): Whether to display frames during processing
    """
    # Open video
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video {input_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {input_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")
    
    # Initialize video writer if output path specified
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Output will be saved to: {output_path}")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Process frame
        processed_frame = process_frame(frame)
        
        # Write to output video if specified
        if writer:
            writer.write(processed_frame)
        
        # Display frame if requested
        if display:
            cv2.imshow('Lane Detection', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Print progress
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")
    
    # Release resources
    cap.release()
    if writer:
        writer.release()
    if display:
        cv2.destroyAllWindows()
    
    print(f"Processing complete! Total frames processed: {frame_count}")

if __name__ == "__main__":
    # Process video
    input_video = "test_videos/busy.mp4"
    output_video = "output_lane_detection2.mp4"
    
    process_video(input_video, output_path=output_video, display=False)
