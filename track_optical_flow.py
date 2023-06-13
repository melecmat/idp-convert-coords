import cv2
import numpy as np

# Define the points to track (initial keypoints)
points_to_track = np.array([(2054.4821184304824, 951.5860976951102), (3811.5836308398125, 1260.5615193107983), (2568.2025837398946, 930.7936289401422)], dtype=np.float32)#np.array([[50, 50], [100, 50], [50, 100], [100, 100]], dtype=np.float32)
# Create a list to store the tracked points
tracked_points = []

# Initialize video capture
cap = cv2.VideoCapture('/home/matous/school_work/idp/idp-convert-coords/2022-10-06T16-34-42/DJI_0777_cut.mp4')

# Read the first frame
ret, prev_frame = cap.read()

# Convert the frame to grayscale
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Create a mask for drawing the tracked points
mask = np.zeros_like(prev_frame)

# Define the parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

while True:
    # Read the next frame
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow for the points
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, points_to_track, None, **lk_params)
    status = status.flatten()
    # Select the tracked points that have valid status
    good_new = next_points[status == 1]
    good_old = points_to_track[status == 1, :]

    # Update the tracked points
    tracked_points.extend(good_new)

    # Draw the tracks
    for point in good_new:
        x, y = point.ravel()
        frame = cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

    # Combine the frame with the mask
    # Display the image
    cv2.imshow('Sparse Optical Flow', cv2.resize(frame, (960, 540)) )

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Update the previous frame and points
    prev_gray = gray.copy()
    points_to_track = good_new.reshape(-1, 1, 2)
    print(points_to_track)

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()
