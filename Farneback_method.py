import cv2
import numpy as np

input_path = "data/"
output_path = "output_data/"


capture = cv2.VideoCapture(input_path)

# Reading the first frame
_, frame1 = capture.read()
# Convert to gray scale
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
# Create mask
hsv_mask = np.zeros_like(frame1)
# Make image saturation to a maximum value
hsv_mask[..., 1] = 255

# Get the width, height and FPS of the video
frame_width = int(capture.get(3))
frame_height = int(capture.get(4))
fps = capture.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
output_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))


while capture.isOpened():
    
    # Capture another frame and convert to gray scale
    _, frame2 = capture.read()
    if frame2 is None:
        break
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Optical flow is calculated using Farneback method
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # Compute magnitude and angle of 2D vector
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # Set image hue value according to the angle of optical flow
    hsv_mask[..., 0] = ang * 180 / np.pi / 2
    hsv_mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # Convert to rgb
    rgb_representation = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)

    cv2.imshow('frame2', rgb_representation)
    output_video.write(rgb_representation)

    kk = cv2.waitKey(20) & 0xff
    # Press 'e' to exit the video
    if kk == ord('e'):
        break
    
    prvs = next

capture.release()
output_video.release()
cv2.destroyAllWindows()
